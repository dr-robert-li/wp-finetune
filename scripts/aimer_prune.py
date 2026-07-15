"""AIMER weight-norm expert importance scorer (PRUNE-01).

Calibration-free, weight-only per-expert importance score (arxiv 2603.18492,
formula pinned in deprecated/wp-moe.md): for each (layer, expert), concatenate that
expert's gate/up/down projection weights and score them as

    score = P / sqrt(N * Q)

where P = L1 norm (sum |w|), N = element count, Q = squared L2 norm (sum
w**2). Higher score = more important (keep). By Cauchy-Schwarz the score is
scale-invariant (multiplying an expert's weights by c>0 leaves it unchanged)
and bounded in [1/sqrt(N), 1].

On-disk checkpoints are SHARDED (13 files for the models this phase reads)
with per-expert UNSTACKED tensors at keys
model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight (verified against
model.safetensors.index.json this session — differs from the stacked
gate_up_proj/down_proj naming assumed in 13-RESEARCH's skeleton). Each key is
resolved through the index's weight_map and streamed one tensor at a time via
safetensors.safe_open; only running (P, N, Q) scalars are accumulated, so no
full model (~60GB) is ever loaded.

Usage (CLI):
    python -m scripts.aimer_prune \
        --checkpoint models/qwen3-30b-wp-30_70-reasoning-merged-v4 \
        --out output/prune/aimer_scores_gen.npy

    # judge axis, shared prune signal -> elementwise MEAN across seeds:
    python -m scripts.aimer_prune \
        --checkpoint models/_staging/qwen3-30b-wp-v1.3-s0-merged \
                     models/_staging/qwen3-30b-wp-v1.3-merged \
                     models/_staging/qwen3-30b-wp-v1.3-s2-merged \
        --out output/prune/aimer_scores_judge.npy

    python -m scripts.aimer_prune --self-check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

N_LAYERS = 48
N_EXPERTS = 128
PROJS = ("gate_proj", "up_proj", "down_proj")


def _expert_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def compute_aimer_scores(
    checkpoint_dir: str | Path, n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS
) -> np.ndarray:
    """Return [n_layers, n_experts] float32 AIMER importance scores.

    Streams each expert's gate/up/down tensors from its owning shard (per
    model.safetensors.index.json), one tensor at a time -- never holds more
    than one tensor + scalar accumulators in memory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    weight_map = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    key_to_layer_expert: dict[str, tuple[int, int]] = {}
    for layer in range(n_layers):
        for expert in range(n_experts):
            for proj in PROJS:
                key_to_layer_expert[_expert_key(layer, expert, proj)] = (layer, expert)

    missing = [k for k in key_to_layer_expert if k not in weight_map]
    if missing:
        # T-13-01: a missing/renamed expert key must raise, not silently produce a zero score.
        raise KeyError(
            f"{len(missing)} expert tensor keys missing from weight_map, e.g. {sorted(missing)[:3]}"
        )

    keys_by_shard: dict[str, list[str]] = {}
    for key in key_to_layer_expert:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    P = np.zeros((n_layers, n_experts), dtype=np.float64)
    Q = np.zeros((n_layers, n_experts), dtype=np.float64)
    N = np.zeros((n_layers, n_experts), dtype=np.int64)

    for shard_name, keys in keys_by_shard.items():
        with safe_open(checkpoint_dir / shard_name, framework="pt") as f:
            for key in keys:
                layer, expert = key_to_layer_expert[key]
                w = f.get_tensor(key).float()
                P[layer, expert] += w.abs().sum().item()
                Q[layer, expert] += (w**2).sum().item()
                N[layer, expert] += w.numel()

    scores = (P / np.sqrt(N * Q)).astype(np.float32)
    assert np.isfinite(scores).all(), "AIMER scores must be finite for every expert"
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="1+ checkpoint dirs; multiple = elementwise MEAN across models (shared judge profile)",
    )
    ap.add_argument("--out", required=True, help="output score array .npy path")
    args = ap.parse_args()

    all_scores = [compute_aimer_scores(c) for c in args.checkpoint]
    scores = np.mean(all_scores, axis=0).astype(np.float32) if len(all_scores) > 1 else all_scores[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, scores)

    print(f"checkpoints: {args.checkpoint}")
    print(
        f"scores shape={scores.shape} min={scores.min():.4f} max={scores.max():.4f} "
        f"mean={scores.mean():.4f}"
    )
    print(f"Wrote {out_path}")
    return 0


def _self_check() -> None:
    """Assert-based self-check on a tiny synthetic on-disk fixture (no GPU, no real checkpoint)."""
    import tempfile

    import torch
    from safetensors.torch import save_file

    n_layers, n_experts = 2, 3
    gen = torch.Generator().manual_seed(0)
    base_tensors = {
        (layer, expert, proj): torch.randn(4, 5, generator=gen)
        for layer in range(n_layers)
        for expert in range(n_experts)
        for proj in PROJS
    }

    def _write(ckpt_dir: Path, scale: float) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        weight_map: dict[str, str] = {}
        shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for (layer, expert, proj), w in base_tensors.items():
            key = _expert_key(layer, expert, proj)
            shard = f"model-{layer:05d}.safetensors"
            weight_map[key] = shard
            shard_tensors.setdefault(shard, {})[key] = (w * scale).float()
        for shard, tensors in shard_tensors.items():
            save_file(tensors, ckpt_dir / shard)
        (ckpt_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_a, ckpt_b = tmp_path / "a", tmp_path / "b"
        _write(ckpt_a, scale=1.0)
        _write(ckpt_b, scale=5.0)

        scores_a = compute_aimer_scores(ckpt_a, n_layers=n_layers, n_experts=n_experts)
        scores_b = compute_aimer_scores(ckpt_b, n_layers=n_layers, n_experts=n_experts)
        scores_a2 = compute_aimer_scores(ckpt_a, n_layers=n_layers, n_experts=n_experts)

        assert scores_a.shape == (n_layers, n_experts)
        assert np.isfinite(scores_a).all()
        assert np.allclose(scores_a, scores_b, rtol=1e-4), "AIMER score must be scale-invariant"
        assert np.array_equal(scores_a, scores_a2), "AIMER score must be deterministic"
        n = 4 * 5 * len(PROJS)
        lower = 1.0 / np.sqrt(n)
        assert (scores_a >= lower - 1e-6).all() and (scores_a <= 1.0 + 1e-6).all(), (
            "AIMER score must be bounded in [1/sqrt(N), 1]"
        )

    print("self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
