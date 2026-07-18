"""Cross-seed routing overlap for the 3 judge-seed merged checkpoints (SIEVE-01).

Resolves 11-CONTEXT.md Open Question 2: do the 3 judge seeds route similarly
enough that ONE Sieve masking profile covers all 3 ("shared"), or does the
k-sweep need the union of per-seed hot experts ("union")?

Reads output/sieve/judge-s{0,1,2}/routing_report.jsonl (Phase-7 schema, one
record per MoE layer), builds per-layer top-k active-expert sets per seed
(same argsort top-k convention as profile_merged_model.compute_jaccard_stability),
and computes pairwise per-layer Jaccard overlap across seeds.

Decision rule (recorded in the output JSON so the choice is auditable):
    sieve_profile_mode = "shared" if global mean Jaccard >= 0.90 else "union"

Output: output/sieve/cross_seed_overlap.json

Usage:
    python -m scripts.sieve_cross_seed_overlap [--top-k 32]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

# Bootstrap for direct `python scripts/sieve_cross_seed_overlap.py` execution
# (established repo convention, e.g. scripts/sieve_ksweep_run.py) -- `scripts`
# is only importable as a package once the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sieve_arch import infer_dims_from_records  # noqa: E402

# Import-back-compat only (NOT load-bearing): load_seed_counts infers dims
# from the file itself (GATE4-02 SC1); these are the empty-file fallback only.
N_LAYERS = 48
N_EXPERTS = 128
DEFAULT_TOP_K = 32
SHARED_THRESHOLD = 0.90

SEED_DIRS = {
    "s0": "output/sieve/judge-s0",
    "s1": "output/sieve/judge-s1",
    "s2": "output/sieve/judge-s2",
}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets. Both empty -> 1.0 (trivially identical)."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def load_seed_counts(jsonl_path: str | Path) -> np.ndarray:
    """Load a routing_report.jsonl into a [n_layers, n_experts] total-count array.

    Dims are inferred from the file's own layer_idx/expert-id range (GATE4-02
    SC1) -- (40, 256) for a v4 report, (48, 128) for a v3 report. Falls back to
    the module N_LAYERS/N_EXPERTS defaults only for an empty file.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if records:
        n_layers, n_experts = infer_dims_from_records(records)
    else:
        n_layers, n_experts = N_LAYERS, N_EXPERTS

    counts = np.zeros((n_layers, n_experts), dtype=float)
    for rec in records:
        layer_idx = int(rec["layer_idx"])
        if not 0 <= layer_idx < n_layers:
            continue
        for k, v in rec.get("expert_counts_total", {}).items():
            e = int(k)
            if 0 <= e < n_experts:
                counts[layer_idx, e] = float(v)
    return counts


def topk_sets(counts: np.ndarray, top_k: int) -> list[set]:
    """Per-layer top-k expert-id sets, same argsort convention as
    profile_merged_model.compute_jaccard_stability."""
    return [
        set(np.argsort(counts[layer])[-top_k:].tolist())
        for layer in range(counts.shape[0])
    ]


def pairwise_layer_jaccard(seed_topk: dict[str, list[set]]) -> dict[tuple[str, str], list[float]]:
    """Pairwise per-layer Jaccard across seeds.

    Args:
        seed_topk: {seed_name: [top-k set per layer]} — all seeds must have the
            same number of layers.

    Returns:
        {(seed_a, seed_b): [jaccard per layer]} for each unordered pair,
        keyed in sorted seed-name order.
    """
    result: dict[tuple[str, str], list[float]] = {}
    for a, b in itertools.combinations(sorted(seed_topk), 2):
        sets_a, sets_b = seed_topk[a], seed_topk[b]
        assert len(sets_a) == len(sets_b), f"layer count mismatch: {a} vs {b}"
        result[(a, b)] = [jaccard(sa, sb) for sa, sb in zip(sets_a, sets_b)]
    return result


def compute_overlap_report(seed_topk: dict[str, list[set]], top_k: int) -> dict:
    """Full overlap report dict: per-layer pairwise Jaccard, means, and the
    shared-vs-union decision against SHARED_THRESHOLD."""
    pairwise = pairwise_layer_jaccard(seed_topk)
    n_layers = len(next(iter(seed_topk.values())))

    per_layer = []
    for layer in range(n_layers):
        pairs = {f"{a}-{b}": pairwise[(a, b)][layer] for (a, b) in pairwise}
        pairs["layer_idx"] = layer
        pairs["mean"] = float(np.mean([pairwise[p][layer] for p in pairwise]))
        per_layer.append(pairs)

    mean_overlap = float(np.mean([j for vals in pairwise.values() for j in vals]))
    mode = "shared" if mean_overlap >= SHARED_THRESHOLD else "union"

    return {
        "analysis": "cross_seed_overlap",
        "seeds": sorted(seed_topk),
        "top_k": top_k,
        "n_layers": n_layers,
        "per_layer_jaccard": per_layer,
        "pairwise_mean": {
            f"{a}-{b}": float(np.mean(vals)) for (a, b), vals in pairwise.items()
        },
        "mean_overlap": mean_overlap,
        "shared_threshold": SHARED_THRESHOLD,
        "sieve_profile_mode": mode,
        "decision_rule": f"shared if mean_overlap >= {SHARED_THRESHOLD} else union",
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-seed routing overlap (Open Question 2)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Top-k experts per layer for the overlap sets (default {DEFAULT_TOP_K})")
    parser.add_argument("--output", default="output/sieve/cross_seed_overlap.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    seed_topk = {}
    for seed, d in SEED_DIRS.items():
        jsonl = project_root / d / "routing_report.jsonl"
        counts = load_seed_counts(jsonl)
        assert counts.sum() > 0, f"{jsonl} has no routing counts"
        seed_topk[seed] = topk_sets(counts, args.top_k)

    report = compute_overlap_report(seed_topk, args.top_k)

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"mean_overlap={report['mean_overlap']:.4f} "
          f"(threshold {SHARED_THRESHOLD}) -> sieve_profile_mode={report['sieve_profile_mode']}")
    for pair, m in report["pairwise_mean"].items():
        print(f"  {pair}: {m:.4f}")
    print(f"Wrote {out_path}")


def _self_check():
    """Assert-based self-check on a tiny synthetic 2-seed fixture."""
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert jaccard(set(), set()) == 1.0
    seed_topk = {"s0": [{1, 2, 3}, {10, 11, 12}], "s1": [{2, 3, 4}, {10, 11, 12}]}
    pw = pairwise_layer_jaccard(seed_topk)
    assert pw[("s0", "s1")] == [0.5, 1.0]
    report = compute_overlap_report(seed_topk, top_k=3)
    assert report["mean_overlap"] == 0.75
    assert report["sieve_profile_mode"] == "union"  # 0.75 < 0.90
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
