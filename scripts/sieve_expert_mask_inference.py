"""Inference-time expert-masking module for the training-free Sieve k-sweep (SIEVE-04).

No training, no gradients: masks the coldest routed experts per layer at a budget
k in {13, 32, 64}, ALWAYS keeping every protected expert (union with the top-k hot
set) plus the CLI/driver wiring to actually apply that mask at vLLM inference time
(see scripts/_sieve_vllm_patch/sitecustomize.py, which imports build_ksweep_mask-
equivalent logic is NOT needed there — the patch reads the mask.npy this module
writes and applies it via a forward hook on the MoE router).

Core contract (tests/test_sieve_ksweep_mask.py, Wave-0):
    build_ksweep_mask(counts, protected, k) -> [n_layers, n_experts] bool
        per layer: top-k experts by routing count UNION protected experts for
        that layer. Never drops a protected expert; never keeps fewer than k.

apply_mask operates on router logits (pre-softmax): setting a masked expert's
logit to -inf is mathematically identical to "zero its softmax weight and
renormalize over the kept set" (softmax renormalizes automatically over
whatever is left) -- this is the same renormalization Phase 13 PRUNE-06 will
need, applied without ever touching a weight tensor or running a gradient step.

Usage (CLI, build a mask file for one k):
    python -m scripts.sieve_expert_mask_inference \
        --routing-report output/profiling/reasoning-merged-v4/routing_report.jsonl \
        --protected-mask output/profiling/reasoning-merged-v4/protected_expert_mask.npy \
        --k 13 --out output/sieve/masks/gen_k13.npy

    # judge axis, sieve_profile_mode=shared -> sum multiple seed reports into one profile:
    python -m scripts.sieve_expert_mask_inference \
        --routing-report output/sieve/judge-s0/routing_report.jsonl \
                         output/sieve/judge-s1/routing_report.jsonl \
                         output/sieve/judge-s2/routing_report.jsonl \
        --protected-mask output/profiling/reasoning-merged-v4/protected_expert_mask.npy \
        --k 32 --out output/sieve/masks/judge_shared_k32.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Bootstrap for direct `python scripts/sieve_expert_mask_inference.py` execution
# (established repo convention, e.g. scripts/sieve_ksweep_run.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sieve_cross_seed_overlap import load_seed_counts  # noqa: E402

# Import-back-compat only (NOT load-bearing): build_ksweep_mask/apply_mask are
# already shape-driven off the arrays passed in (128 experts for v3, 256 for
# v4 -- GATE4-02 SC1); neither constant is read by any function in this module.
N_LAYERS = 48
N_EXPERTS = 128
NEG_INF = -1.0e9


def build_ksweep_mask(counts: np.ndarray, protected: np.ndarray, k: int) -> np.ndarray:
    """Per-layer keep-mask: top-k hot experts UNION protected experts.

    Args:
        counts: [n_layers, n_experts] routing counts (higher = hotter).
        protected: [n_layers, n_experts] bool, the immutable protected-expert mask.
        k: hot-expert budget per layer.

    Returns:
        [n_layers, n_experts] bool. kept[l].sum() == max(k, protected[l].sum())
        when protected experts overlap the top-k hot set only partially; never
        drops a protected expert; masked-out (False) experts are always the
        coldest non-protected experts.
    """
    assert counts.shape == protected.shape, f"{counts.shape} != {protected.shape}"
    n_layers, n_experts = counts.shape
    kept = np.zeros((n_layers, n_experts), dtype=bool)
    for layer in range(n_layers):
        hot_idx = np.argsort(counts[layer])[-k:]
        kept[layer, hot_idx] = True
        kept[layer] |= protected[layer]
    return kept


def apply_mask(router_logits, keep_mask_row) -> "np.ndarray | object":
    """Force non-kept experts' router logit to -inf (softmax weight -> 0,
    renormalized over the kept set). Works on numpy arrays or torch tensors
    (duck-typed: only uses +, dtype/device-free ops on numpy; torch tensors
    get a torch-native path so autograd/device placement is preserved).

    Args:
        router_logits: [..., n_experts] pre-softmax logits (last dim = experts).
        keep_mask_row: [n_experts] bool, True = keep, False = mask out.
    """
    keep_mask_row = np.asarray(keep_mask_row, dtype=bool)
    add = np.where(keep_mask_row, 0.0, NEG_INF).astype(np.float32)
    try:
        import torch
        if isinstance(router_logits, torch.Tensor):
            t = torch.from_numpy(add).to(dtype=router_logits.dtype, device=router_logits.device)
            return router_logits + t
    except ImportError:
        pass
    return np.asarray(router_logits) + add


def load_protected_mask(npy_path: str | Path) -> np.ndarray:
    mask = np.load(npy_path)
    assert mask.dtype == bool, f"protected mask dtype {mask.dtype} != bool"
    return mask


def build_profile_counts(routing_report_paths: list[str | Path]) -> np.ndarray:
    """Sum routing counts across 1+ routing_report.jsonl files.

    sieve_profile_mode="shared" (11-03 decision, mean cross-seed Jaccard 0.9332):
    ONE profile covers all judge seeds -> pass all 3 seed reports here and sum.
    A single path (gen axis) is the no-op sum-of-one case.
    """
    total = None
    for p in routing_report_paths:
        c = load_seed_counts(p)
        total = c if total is None else total + c
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing-report", nargs="+", required=True,
                     help="1+ routing_report.jsonl paths; multiple = summed (shared profile)")
    ap.add_argument("--protected-mask", required=True, help="protected_expert_mask.npy")
    ap.add_argument("--k", type=int, required=True, help="hot-expert budget per layer")
    ap.add_argument("--out", required=True, help="output keep-mask .npy path")
    args = ap.parse_args()

    counts = build_profile_counts(args.routing_report)
    protected = load_protected_mask(args.protected_mask)
    kept = build_ksweep_mask(counts, protected, args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, kept)

    per_layer_kept = kept.sum(axis=1)
    print(f"k={args.k}: kept {int(kept.sum())} total experts across {kept.shape[0]} layers "
          f"(min/layer={int(per_layer_kept.min())}, max/layer={int(per_layer_kept.max())})")
    print(f"protected_retained: {bool(np.all((~protected) | kept))}")
    print(f"Wrote {out_path}")
    return 0


def _self_check():
    """Assert-based self-check on a tiny synthetic 2-layer fixture (no GPU)."""
    n_experts = 8
    counts = np.stack([np.arange(n_experts, 0, -1, dtype=float),
                        np.arange(n_experts, 0, -1, dtype=float)])
    protected = np.zeros((2, n_experts), dtype=bool)
    protected[1, 7] = True  # coldest expert in layer 1, protected -> must survive

    kept = build_ksweep_mask(counts, protected, k=3)
    assert kept[0].sum() == 3
    assert set(np.where(kept[0])[0]) == {0, 1, 2}
    assert kept[1, 7] == True  # noqa: E712
    assert kept[1].sum() == 4  # top-3 (0,1,2) union protected (7)
    assert np.all((~protected) | kept)  # protected always retained

    # apply_mask: renormalization property (post-softmax weights sum to 1 over kept)
    logits = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 0.1, 0.2, 5.0])
    keep_row = kept[1]
    masked = apply_mask(logits, keep_row)
    exp = np.exp(masked - masked.max())
    weights = exp / exp.sum()
    assert np.isclose(weights[~keep_row].sum(), 0.0, atol=1e-6), "masked experts must get ~0 softmax weight"
    assert np.isclose(weights[keep_row].sum(), 1.0, atol=1e-6), "kept experts must renormalize to 1"

    # torch path parity (if torch available)
    try:
        import torch
        t_logits = torch.tensor(logits, dtype=torch.float32)
        t_masked = apply_mask(t_logits, keep_row)
        assert torch.allclose(t_masked, torch.tensor(masked, dtype=torch.float32), atol=1e-4)
    except ImportError:
        pass

    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
