"""Council-mandated differential check for P0 v3 merge.

Compares v1 partial baseline vs v3 fresh merge on MoE expert tensors to
distinguish interpretation A (ParamWrapper.merge() fused target_parameters)
from interpretation B (silent re-copy of v1 partial baseline).

Sampled layers: 0, 12, 24, 36, 47 (5 depths across the 48-layer stack).
Sampled experts per layer: 0, 31, 63, 95, 127 (5 of 128 experts).
Per expert: gate_proj, up_proj, down_proj — 3 tensors each.
Total: 5 layers x 5 experts x 3 projs = 75 tensors per check.

Outputs go/no-go verdict per council rubric:
- identical (max_diff == 0):                 B — fusion failed
- non-trivial diff (max_diff > 1e-4, nonzero > 1000): A — fusion confirmed
- between:                                   partial — flag, treat as B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

V1_DIR = Path("models/qwen3-30b-wp-30_70-merged")
V3_DIR = Path("models/qwen3-30b-wp-30_70-merged-v2")

LAYERS = [0, 12, 24, 36, 47]
EXPERTS = [0, 31, 63, 95, 127]
PROJS = ["gate_proj", "up_proj", "down_proj"]


def load_index(model_dir: Path) -> dict[str, str]:
    with (model_dir / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def get_tensor(model_dir: Path, idx: dict[str, str], key: str) -> torch.Tensor:
    shard = idx[key]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main() -> int:
    v1_idx = load_index(V1_DIR)
    v3_idx = load_index(V3_DIR)

    print(f"v1 baseline: {V1_DIR}")
    print(f"v3 candidate: {V3_DIR}")
    print(f"Sampling: layers={LAYERS}, experts={EXPERTS}, projs={PROJS}")
    print(f"Tensor count: {len(LAYERS) * len(EXPERTS) * len(PROJS)}")
    print()

    rows = []
    identical_count = 0
    nontriv_count = 0
    partial_count = 0
    missing_count = 0

    for layer in LAYERS:
        for expert in EXPERTS:
            for proj in PROJS:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                if key not in v1_idx:
                    print(f"MISSING in v1: {key}")
                    missing_count += 1
                    continue
                if key not in v3_idx:
                    print(f"MISSING in v3: {key}")
                    missing_count += 1
                    continue
                t1 = get_tensor(V1_DIR, v1_idx, key).float()
                t3 = get_tensor(V3_DIR, v3_idx, key).float()
                if t1.shape != t3.shape:
                    print(f"SHAPE MISMATCH {key}: v1={t1.shape} v3={t3.shape}")
                    missing_count += 1
                    continue
                diff = (t3 - t1).abs()
                max_d = diff.max().item()
                mean_d = diff.mean().item()
                nz = (diff > 0).sum().item()
                numel = diff.numel()
                t1_norm = t1.norm().item()
                t3_norm = t3.norm().item()
                rel_change = abs(t3_norm - t1_norm) / max(t1_norm, 1e-9)

                if max_d == 0.0:
                    verdict = "identical"
                    identical_count += 1
                elif max_d > 1e-4 and nz > 1000:
                    verdict = "non-trivial"
                    nontriv_count += 1
                else:
                    verdict = "partial"
                    partial_count += 1
                rows.append(
                    (layer, expert, proj, verdict, max_d, mean_d, nz, numel, rel_change)
                )

    print(
        f"{'layer':>5} {'expert':>6} {'proj':>10} {'verdict':>12} "
        f"{'max_diff':>12} {'mean_diff':>12} {'nonzero':>14} {'rel_chg':>10}"
    )
    for layer, expert, proj, verdict, max_d, mean_d, nz, numel, rel in rows:
        print(
            f"{layer:>5} {expert:>6} {proj:>10} {verdict:>12} "
            f"{max_d:>12.6f} {mean_d:>12.6f} {nz:>7d}/{numel:<7d} {rel:>10.6f}"
        )

    total = len(rows)
    print()
    print(f"=== SUMMARY ({total} tensors, {missing_count} missing/mismatch) ===")
    print(f"identical:    {identical_count} / {total}")
    print(f"non-trivial:  {nontriv_count} / {total}")
    print(f"partial:      {partial_count} / {total}")

    if total == 0:
        print("\nVERDICT: ERROR — no tensors compared")
        return 2
    if identical_count == total:
        print("\nVERDICT: B — FUSION FAILED (v3 == v1 partial baseline)")
        print("Action: do not use v3; investigate adapter config + PEFT load path")
        return 1
    if nontriv_count == total:
        print("\nVERDICT: A — FUSION CONFIRMED (target_parameters fused into v3)")
        print("Action: proceed to extended 48-layer validation, then W0-03 smoke")
        return 0
    print("\nVERDICT: PARTIAL — some tensors fused, others not. Treat as B.")
    print("Action: inspect per-layer/per-expert pattern; do not advance to W0-03")
    return 1


if __name__ == "__main__":
    sys.exit(main())
