"""Pre-flight extraction-convention probe.

Compares two candidate per-expert delta interpretations against Unsloth's
documented `_extract_lora_from_wrapper` reshape (moe_utils.py:421-426).

Hypothesis A (council artifact): delta = (B @ A) * scale → broadcast to all 128 experts
Hypothesis C (Unsloth source):   delta[e] = B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] * scale per expert

The correct interpretation matches Unsloth's reshape:
    first  = A.view(E, R, in).permute(0, 2, 1)      → (E, in, R)
    second = B.view(out, E, R).permute(1, 2, 0)     → (E, R, out)
    delta[e] = (first[e] @ second[e]).T              → (out, in)

Sampled: layer 0 base_layer (gate_up_proj fused), 3 experts {0, 63, 127}.
Cheap — runs in seconds; bf16 CPU only.
"""

from __future__ import annotations

import json
import math
import sys

import torch
from safetensors import safe_open

ADAPTER = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
CONFIG  = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_config.json"
LAYER   = 0
NUM_EXPERTS = 128
SAMPLE_EXPERTS = [0, 63, 127]


def main() -> int:
    cfg = json.load(open(CONFIG))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    use_rslora = cfg.get("use_rslora", False)
    scale = (alpha / math.sqrt(r)) if use_rslora else (alpha / r)
    print(f"r={r}, alpha={alpha}, use_rslora={use_rslora}, scale={scale:.6f}")

    with safe_open(ADAPTER, framework="pt", device="cpu") as f:
        # ckpt-72 keys use `.weight` suffix (NOT `.default.weight`)
        A = f.get_tensor(f"base_model.model.model.layers.{LAYER}.mlp.experts.base_layer.lora_A.weight").float()
        B = f.get_tensor(f"base_model.model.model.layers.{LAYER}.mlp.experts.base_layer.lora_B.weight").float()

    print(f"A.shape={tuple(A.shape)} (expect (E*R, in)=(4096, 2048))")
    print(f"B.shape={tuple(B.shape)} (expect (out, E*R)=(1536, 4096))")
    in_features = A.shape[1]   # 2048 hidden
    out_features = B.shape[0]  # 1536 = 2*intermediate

    # ── Hypothesis A: council broadcast ─────────────────────────────────────
    delta_broadcast = (B @ A) * scale   # (1536, 2048) — same for all experts
    print(f"\n[H-A broadcast] delta.shape={tuple(delta_broadcast.shape)}")
    print(f"  L2 norm = {delta_broadcast.norm().item():.4f}")
    print(f"  max abs = {delta_broadcast.abs().max().item():.6f}")

    # ── Hypothesis C: per-expert contiguous-block (Unsloth reshape) ────────
    # Use Unsloth's exact reshape+permute path
    first  = A.view(NUM_EXPERTS, r, in_features).permute(0, 2, 1).contiguous()   # (E, in, R)
    second = B.view(out_features, NUM_EXPERTS, r).permute(1, 2, 0).contiguous()  # (E, R, out)
    print(f"\n[Unsloth reshape] first.shape={tuple(first.shape)}  second.shape={tuple(second.shape)}")

    # Per-expert delta via Unsloth's interpretation: (first[e] @ second[e]).T = (out, in)
    deltas_unsloth = []
    for e in SAMPLE_EXPERTS:
        # Unsloth runtime: out[N, in_dim] @ first[e](in_dim, R) → (N, R) ; then @ second[e](R, out) → (N, out)
        # Weight-space delta: first[e] @ second[e] = (in, out). Transpose for nn.Linear weight (out, in).
        delta_e = (first[e] @ second[e]).T * scale   # (out, in)
        deltas_unsloth.append((e, delta_e))
        print(f"  expert {e}: shape={tuple(delta_e.shape)} L2={delta_e.norm().item():.4f} max_abs={delta_e.abs().max().item():.6f}")

    # ── Hypothesis C-equiv: raw-matrix contiguous block (no permute) ────────
    # Sanity check: B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] should equal Unsloth's per-expert delta
    print(f"\n[Sanity: raw contiguous-block per-expert]")
    for e in SAMPLE_EXPERTS:
        A_e_raw = A[e * r:(e + 1) * r, :]       # (R, in)
        B_e_raw = B[:, e * r:(e + 1) * r]       # (out, R)
        delta_e_raw = (B_e_raw @ A_e_raw) * scale   # (out, in)
        # Compare with Unsloth-permute path
        e_idx = SAMPLE_EXPERTS.index(e)
        delta_e_unsloth = deltas_unsloth[e_idx][1]
        diff = (delta_e_raw - delta_e_unsloth).abs().max().item()
        match = "✓ MATCH" if diff < 1e-6 else f"✗ DIFFER (max_abs_diff={diff:.6f})"
        print(f"  expert {e}: raw-vs-unsloth-permute  {match}  L2_raw={delta_e_raw.norm().item():.4f}")

    # ── Cross-compare: Hypothesis A vs Hypothesis C per-expert ──────────────
    print(f"\n[Cross-compare: broadcast vs per-expert]")
    for e in SAMPLE_EXPERTS:
        e_idx = SAMPLE_EXPERTS.index(e)
        delta_e_uns = deltas_unsloth[e_idx][1]      # Unsloth per-expert
        diff = (delta_broadcast - delta_e_uns).abs()
        cos = torch.nn.functional.cosine_similarity(
            delta_broadcast.flatten().unsqueeze(0),
            delta_e_uns.flatten().unsqueeze(0),
        ).item()
        print(f"  expert {e}: max|broadcast - per_expert|={diff.max().item():.4f}  "
              f"L2_diff={diff.norm().item():.4f}  cos_sim={cos:.6f}")

    # ── Decision summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DECISION HINT:")
    print("  IF broadcast ≈ per_expert (cos > 0.999):")
    print("    → council interpretation acceptable (shared correction)")
    print("  IF broadcast ≠ per_expert (cos < 0.5):")
    print("    → per-expert is the trained signal; council script wrong → REVISE")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
