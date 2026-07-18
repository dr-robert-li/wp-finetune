"""Physical expert-removal surgery (PRUNE-06).

Real on-disk format verified in 13-01/13-02 (NOT 13-RESEARCH's stacked-tensor
skeleton): experts are per-expert UNSTACKED tensors
model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight sharded across 13
files; the router is model.layers.{L}.mlp.gate.weight (shape
[num_local_experts, hidden_size]); the config key is num_local_experts.

Two-step surgery:
    1. build_uniform_keep_mask(scores, protected, K): a [n_layers, n_experts]
       mask with EXACTLY K True per layer -- every protected expert kept,
       remaining K-minus-protected_count budget filled by the highest-score
       non-protected experts. Raises if any layer's protected count exceeds K
       (the physical-feasibility floor prune_selection.py also enforces --
       real data: layer 1 alone carries 40 protected experts).
    2. apply_physical(checkpoint_dir, keep_mask, out_dir): for each layer,
       drops the removed experts' three tensors, RENUMBERS kept experts to
       contiguous 0..K-1 (sorted by original index -- HF/vLLM loaders expect
       0..num_local_experts-1), slices the router's kept rows in the same new
       order (softmax renormalizes automatically over the fewer rows -- the
       physical analogue of sieve_expert_mask_inference.apply_mask's -inf
       trick), and rewrites num_local_experts=K in config.json.

ponytail: re-shards by writing each output shard under its ORIGINAL shard
filename (just fewer/renamed tensors per shard) rather than repacking for
optimal file-size balance -- correct and simple; repack for balanced shard
sizes only if the real 13-07 run needs it (out of scope for a Wave-0
synthetic-fixture module).

Usage:
    python -m scripts.prune_apply_physical \
        --checkpoint models/qwen3-30b-wp-30_70-reasoning-merged-v4 \
        --score output/prune/aimer_scores_gen.npy \
        --protected output/profiling/reasoning-merged-v4/protected_expert_mask.npy \
        --ratio 25 --out models/_staging/qwen3-30b-wp-pruned-25

    python -m scripts.prune_apply_physical --self-check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

N_LAYERS = 48
N_EXPERTS = 128
RATIO_TO_K = {25: 96, 50: 64, 75: 32}

EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
ROUTER_KEY_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")


def build_uniform_keep_mask(scores: np.ndarray, protected: np.ndarray, k: int) -> np.ndarray:
    """[n_layers, n_experts] bool mask, EXACTLY k True per layer.

    Every protected expert is kept; the remaining k-minus-protected_count
    budget is filled by the highest-score non-protected experts. Raises
    ValueError if any layer's protected count exceeds k (infeasible ratio --
    would force dropping a protected expert).
    """
    assert scores.shape == protected.shape, f"{scores.shape} != {protected.shape}"
    n_layers, n_experts = scores.shape
    protected_count = protected.sum(axis=1)

    bad_layers = np.where(protected_count > k)[0]
    if len(bad_layers):
        layer = int(bad_layers[0])
        raise ValueError(
            f"infeasible k={k}: layer {layer} has {int(protected_count[layer])} protected "
            f"experts > k (uniform per-layer keep-count cannot drop a protected expert)"
        )

    kept = protected.copy()
    for layer in range(n_layers):
        need = k - int(protected_count[layer])
        if need <= 0:
            continue
        candidates = np.where(~protected[layer])[0]
        ranked = candidates[np.argsort(-scores[layer, candidates])]
        kept[layer, ranked[:need]] = True

    assert (kept.sum(axis=1) == k).all(), "build_uniform_keep_mask must keep exactly k/layer"
    return kept


def apply_physical(checkpoint_dir: str | Path, keep_mask: np.ndarray, out_dir: str | Path) -> dict:
    """Physically remove non-kept experts, renumber survivors contiguously,
    slice the router to the kept rows, and rewrite num_local_experts.

    Requires the safetensors and torch packages (real weight I/O); imported
    lazily so CPU-only unit-test collection of this module never needs a GPU.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers, n_experts = keep_mask.shape
    per_layer_k = keep_mask.sum(axis=1)
    assert (per_layer_k == per_layer_k[0]).all(), "keep_mask must have a UNIFORM per-layer count"
    k = int(per_layer_k[0])

    # kept original expert indices per layer, sorted ascending -> becomes 0..k-1.
    kept_indices = {
        layer: sorted(np.where(keep_mask[layer])[0].tolist()) for layer in range(n_layers)
    }

    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    config = json.loads((checkpoint_dir / "config.json").read_text())

    new_weight_map: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors_out = {}
        with safe_open(checkpoint_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                m = EXPERT_KEY_RE.match(key)
                if m:
                    layer, expert = int(m.group("layer")), int(m.group("expert"))
                    proj = m.group("proj")
                    if layer >= n_layers or expert not in kept_indices[layer]:
                        continue  # dropped expert (or a layer outside pruning scope)
                    new_expert = kept_indices[layer].index(expert)
                    new_key = f"model.layers.{layer}.mlp.experts.{new_expert}.{proj}.weight"
                    tensors_out[new_key] = f.get_tensor(key)
                    new_weight_map[new_key] = shard_name
                    continue

                rm = ROUTER_KEY_RE.match(key)
                if rm:
                    layer = int(rm.group("layer"))
                    tensor = f.get_tensor(key)
                    if layer < n_layers:
                        tensor = tensor[kept_indices[layer], :]
                    tensors_out[key] = tensor
                    new_weight_map[key] = shard_name
                    continue

                # non-expert, non-router tensor: copy unchanged.
                tensors_out[key] = f.get_tensor(key)
                new_weight_map[key] = shard_name

        if tensors_out:
            save_file(tensors_out, out_dir / shard_name)

    total_size = sum((out_dir / s).stat().st_size for s in set(new_weight_map.values()))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {**index.get("metadata", {}), "total_size": total_size},
             "weight_map": new_weight_map},
            indent=2,
        )
    )

    config["num_local_experts"] = k
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    return {"k": k, "n_layers": n_layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--score", required=True, help="[n_layers, n_experts] score .npy")
    ap.add_argument("--protected", required=True, help="protected_expert_mask.npy")
    ap.add_argument("--ratio", type=int, choices=[25, 50, 75], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = np.load(args.score)
    protected = np.load(args.protected)
    k = RATIO_TO_K[args.ratio]
    keep_mask = build_uniform_keep_mask(scores, protected, k)
    result = apply_physical(args.checkpoint, keep_mask, args.out)

    print(f"ratio={args.ratio} k={result['k']} n_layers={result['n_layers']}")
    print(f"Wrote {args.out}")
    return 0


def _write_fixture_checkpoint(ckpt_dir: Path, n_layers: int, n_experts: int,
                               hidden: int, moe_dim: int, seed: int = 0) -> None:
    """Tiny synthetic checkpoint: experts + router + one unrelated tensor, single shard."""
    import torch
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, "torch.Tensor"] = {}
    weight_map: dict[str, str] = {}
    shard = "model.safetensors"

    for layer in range(n_layers):
        for expert in range(n_experts):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"] = (
                torch.randn(moe_dim, hidden, generator=gen)
            )
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = (
                torch.randn(moe_dim, hidden, generator=gen)
            )
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = (
                torch.randn(hidden, moe_dim, generator=gen)
            )
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(
            n_experts, hidden, generator=gen
        )
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(
            hidden, generator=gen
        )

    for key in tensors:
        weight_map[key] = shard
    save_file(tensors, ckpt_dir / shard)
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps({"num_local_experts": n_experts, "num_hidden_layers": n_layers})
    )


def _self_check() -> None:
    """Assert-based self-check on a tiny synthetic checkpoint (no GPU, no real model)."""
    import tempfile

    import torch
    from safetensors import safe_open

    n_layers, n_experts, hidden, moe_dim, k = 2, 8, 6, 4, 5

    scores = np.random.RandomState(0).rand(n_layers, n_experts)
    protected = np.zeros((n_layers, n_experts), dtype=bool)
    protected[1, 7] = True  # coldest-scored expert in layer 1, protected -> must survive

    # infeasible-k raises
    try:
        build_uniform_keep_mask(scores, protected, k=0)
        raise AssertionError("expected ValueError for infeasible k")
    except ValueError:
        pass

    keep_mask = build_uniform_keep_mask(scores, protected, k)
    assert (keep_mask.sum(axis=1) == k).all()
    assert keep_mask[1, 7]  # protected expert survives

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_dir = tmp_path / "ckpt"
        out_dir = tmp_path / "out"
        _write_fixture_checkpoint(ckpt_dir, n_layers, n_experts, hidden, moe_dim)

        result = apply_physical(ckpt_dir, keep_mask, out_dir)
        assert result["k"] == k

        config = json.loads((out_dir / "config.json").read_text())
        assert config["num_local_experts"] == k

        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]

        for layer in range(n_layers):
            expert_keys = [
                key for key in weight_map
                if EXPERT_KEY_RE.match(key) and int(EXPERT_KEY_RE.match(key).group("layer")) == layer
            ]
            assert len(expert_keys) == k * 3, f"layer {layer} expects {k*3} expert tensors"
            new_experts = sorted({int(EXPERT_KEY_RE.match(key).group("expert")) for key in expert_keys})
            assert new_experts == list(range(k)), "experts must be renumbered contiguously 0..k-1"

            with safe_open(out_dir / weight_map[f"model.layers.{layer}.mlp.gate.weight"], framework="pt") as f:
                router = f.get_tensor(f"model.layers.{layer}.mlp.gate.weight")
            assert router.shape == (k, hidden), "router weight must have k rows"

        # Protected expert's tensor must be byte-identical after renumbering (layer 1, orig idx 7).
        with safe_open(ckpt_dir / "model.safetensors", framework="pt") as f:
            orig = f.get_tensor("model.layers.1.mlp.experts.7.gate_proj.weight")
        kept_sorted = sorted(np.where(keep_mask[1])[0].tolist())
        new_idx = kept_sorted.index(7)
        with safe_open(out_dir / weight_map[f"model.layers.1.mlp.experts.{new_idx}.gate_proj.weight"], framework="pt") as f:
            surv = f.get_tensor(f"model.layers.1.mlp.experts.{new_idx}.gate_proj.weight")
        assert torch.equal(orig, surv), "protected expert's weight must survive unmodified"

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
