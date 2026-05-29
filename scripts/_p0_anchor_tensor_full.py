"""Stage-1 tensor anchor: full 48-layer extraction-convention verification.

Confirms per-expert contiguous-block math (B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :])
matches Unsloth's documented `_extract_lora_from_wrapper` reshape+permute path
(moe_utils.py:421-426) for EVERY layer + sampled experts. Cheap, CPU, no model load.

This certifies the EXTRACTION step (does the merge read the right per-expert deltas).
The forward-pass anchor (Stage 2) certifies the APPLICATION step separately.
"""

from __future__ import annotations

import json
import math
import sys

import torch
from safetensors import safe_open

ADAPTER = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
CONFIG  = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_config.json"
REPORT  = "models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate/merge_report.json"
NUM_LAYERS = 48
NUM_EXPERTS = 128
SAMPLE_EXPERTS = [0, 1, 63, 127]  # include adjacent 0,1 to confirm per-expert distinctness


def unsloth_per_expert(A: torch.Tensor, B: torch.Tensor, e: int, r: int, num_experts: int, scale: float) -> torch.Tensor:
    """Unsloth's exact reshape+permute path → per-expert weight delta (out, in)."""
    in_features = A.shape[1]
    out_features = B.shape[0]
    first = A.view(num_experts, r, in_features).permute(0, 2, 1).contiguous()   # (E, in, R)
    second = B.view(out_features, num_experts, r).permute(1, 2, 0).contiguous()  # (E, R, out)
    return (first[e] @ second[e]).T * scale   # (out, in)


def raw_block(A: torch.Tensor, B: torch.Tensor, e: int, r: int, scale: float) -> torch.Tensor:
    """Merge-script contiguous-block path."""
    A_e = A[e * r:(e + 1) * r, :]
    B_e = B[:, e * r:(e + 1) * r]
    return (B_e @ A_e) * scale


def main() -> int:
    cfg = json.load(open(CONFIG))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = (alpha / math.sqrt(r)) if cfg.get("use_rslora", False) else (alpha / r)
    print(f"r={r} alpha={alpha} scale={scale}  layers={NUM_LAYERS} sample_experts={SAMPLE_EXPERTS}")

    max_diff_global = 0.0
    n_checks = 0
    fail = 0
    adjacent_distinct_min = float("inf")

    with safe_open(ADAPTER, framework="pt", device="cpu") as f:
        for L in range(NUM_LAYERS):
            for suffix, label in [("base_layer.", "gate_up"), ("", "down")]:
                A = f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.{suffix}lora_A.weight").float()
                B = f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.{suffix}lora_B.weight").float()
                per_expert_deltas = {}
                for e in SAMPLE_EXPERTS:
                    d_unsloth = unsloth_per_expert(A, B, e, r, NUM_EXPERTS, scale)
                    d_raw = raw_block(A, B, e, r, scale)
                    diff = (d_unsloth - d_raw).abs().max().item()
                    max_diff_global = max(max_diff_global, diff)
                    n_checks += 1
                    if diff > 1e-5:
                        fail += 1
                        print(f"  FAIL L{L} {label} e{e}: raw-vs-unsloth max_diff={diff:.2e}")
                    per_expert_deltas[e] = d_raw
                # adjacent-expert distinctness (0 vs 1)
                if 0 in per_expert_deltas and 1 in per_expert_deltas:
                    d01 = (per_expert_deltas[0] - per_expert_deltas[1]).abs().max().item()
                    adjacent_distinct_min = min(adjacent_distinct_min, d01)

    print(f"\n{'=' * 60}")
    print(f"TENSOR ANCHOR: {n_checks} checks ({NUM_LAYERS} layers × 2 proj × {len(SAMPLE_EXPERTS)} experts)")
    print(f"  raw-vs-unsloth max_diff (global): {max_diff_global:.2e}  (threshold 1e-5)")
    print(f"  adjacent-expert (0 vs 1) min distinctness: {adjacent_distinct_min:.6f}  (must be >1e-5)")
    tensor_pass = (fail == 0) and (max_diff_global < 1e-5) and (adjacent_distinct_min > 1e-5)
    print(f"  VERDICT: {'PASS' if tensor_pass else 'FAIL'} ({fail} failures)")
    print(f"{'=' * 60}")

    report = json.load(open(REPORT))
    report["tensor_anchor"] = "pass" if tensor_pass else "fail"
    report["tensor_anchor_detail"] = {
        "n_checks": n_checks,
        "raw_vs_unsloth_max_diff": max_diff_global,
        "adjacent_expert_min_distinct": adjacent_distinct_min,
        "failures": fail,
    }
    json.dump(report, open(REPORT, "w"), indent=2)
    print(f"  Report updated: {REPORT}")
    return 0 if tensor_pass else 1


if __name__ == "__main__":
    sys.exit(main())
