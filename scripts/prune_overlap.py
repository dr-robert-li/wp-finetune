"""AIMER-vs-REAP domain-specificity overlap analysis (PRUNE-04).

Given two [n_layers, n_experts] boolean keep-masks at a MATCHED compression
ratio (e.g. AIMER@25% vs REAP@25%), computes per-layer Jaccard overlap of the
kept expert sets, mirroring scripts/sieve_cross_seed_overlap.py's pairwise
Jaccard pattern (per-layer sets, mean/min/max roll-up).

Interpretation (13-CONTEXT's live question, filled in by the human at
selection time): high overlap => the two independent scoring signals (weight
norm vs calibration saliency) agree on what to keep => WordPress calibration
data isn't specialized enough to move the needle over weight-only AIMER.
Low overlap => REAP's calibration signal is capturing domain-specific routing
patterns AIMER's weight-only view misses.

Also rolls up the pre-committed layer_stability_notes band (low-Jaccard
{9,13,14,31,35,36} + late-layer {45,46,47}) into its own separate mean, since
13-CONTEXT flags those layers as needing more conservative pruning.

Usage:
    python -m scripts.prune_overlap \
        --mask-a output/prune/gated/masks/aimer_gen_r25_mask.npy \
        --mask-b output/prune/gated/masks/reap_gen_r25_mask.npy \
        --ratio 25 --out output/prune/aimer_reap_overlap_25.json

    python -m scripts.prune_overlap --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LOW_JACCARD_BAND = (9, 13, 14, 31, 35, 36)
LATE_LAYER_BAND = (45, 46, 47)


def per_layer_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Per-layer Jaccard(keep_a[l], keep_b[l]) for two [n_layers, n_experts] bool masks.

    Identical masks -> 1.0 every layer. Disjoint keep-sets -> 0.0 every layer.
    Both-empty layer (no experts kept in either) -> 1.0 (trivially identical,
    matches sieve_cross_seed_overlap.jaccard's convention).
    """
    assert mask_a.shape == mask_b.shape, f"{mask_a.shape} != {mask_b.shape}"
    n_layers = mask_a.shape[0]
    out = np.zeros(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        a, b = mask_a[layer], mask_b[layer]
        union = np.logical_or(a, b).sum()
        if union == 0:
            out[layer] = 1.0
        else:
            out[layer] = np.logical_and(a, b).sum() / union
    return out


def _band_summary(per_layer: np.ndarray, band: tuple[int, ...]) -> dict:
    valid = [layer for layer in band if layer < len(per_layer)]
    values = per_layer[valid]
    return {
        "layers": valid,
        "mean": float(values.mean()) if len(values) else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def build_overlap_report(mask_a: np.ndarray, mask_b: np.ndarray, ratio: int) -> dict:
    per_layer = per_layer_jaccard(mask_a, mask_b)
    return {
        "analysis": "aimer_reap_overlap",
        "ratio": ratio,
        "n_layers": int(per_layer.shape[0]),
        "per_layer_jaccard": per_layer.tolist(),
        "mean": float(per_layer.mean()),
        "min": float(per_layer.min()),
        "max": float(per_layer.max()),
        "layer_stability_notes": {
            "low_jaccard_band": _band_summary(per_layer, LOW_JACCARD_BAND),
            "late_layer_band": _band_summary(per_layer, LATE_LAYER_BAND),
        },
        "interpretation_stub": (
            "FILL AT SELECTION TIME: high overlap (mean close to 1.0) => WordPress "
            "not specialized enough for a REAP calibration advantage over AIMER "
            "weight-norms; low overlap => REAP captures domain routing patterns "
            "AIMER's weight-only signal misses."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask-a", required=True, help="AIMER keep-mask .npy")
    ap.add_argument("--mask-b", required=True, help="REAP keep-mask .npy")
    ap.add_argument("--ratio", type=int, required=True, help="compression ratio (25/50/75)")
    ap.add_argument("--out", required=True, help="output overlap report .json path")
    args = ap.parse_args()

    mask_a = np.load(args.mask_a)
    mask_b = np.load(args.mask_b)
    report = build_overlap_report(mask_a, mask_b, args.ratio)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"ratio={args.ratio}: mean_jaccard={report['mean']:.4f} "
          f"min={report['min']:.4f} max={report['max']:.4f}")
    print(f"Wrote {out_path}")
    return 0


def _self_check() -> None:
    """Assert-based self-check on tiny synthetic 48-layer fixtures (no GPU/model)."""
    n_layers, n_experts = 48, 16

    identical_a = np.zeros((n_layers, n_experts), dtype=bool)
    identical_a[:, :8] = True
    identical_b = identical_a.copy()
    identical_jaccard = per_layer_jaccard(identical_a, identical_b)
    assert identical_jaccard.shape == (48,)
    assert np.allclose(identical_jaccard, 1.0)

    disjoint_a = np.zeros((n_layers, n_experts), dtype=bool)
    disjoint_a[:, :8] = True
    disjoint_b = np.zeros((n_layers, n_experts), dtype=bool)
    disjoint_b[:, 8:] = True
    disjoint_jaccard = per_layer_jaccard(disjoint_a, disjoint_b)
    assert disjoint_jaccard.shape == (48,)
    assert np.allclose(disjoint_jaccard, 0.0)

    report = build_overlap_report(identical_a, identical_b, ratio=25)
    assert report["n_layers"] == 48
    assert report["mean"] == 1.0
    assert report["layer_stability_notes"]["low_jaccard_band"]["mean"] == 1.0

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
