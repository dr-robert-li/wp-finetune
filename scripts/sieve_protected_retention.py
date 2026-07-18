"""Protected-mask subset verification for the training-free Sieve (SIEVE-01).

Verifies that the IMMUTABLE Phase-7 protected-expert mask
(output/profiling/reasoning-merged-v4/protected_expert_mask.npy, [48,128] bool,
1,480 experts) is a subset of the hot/cold retained set implied by the
cross-seed overlap decision (sieve_profile_mode from cross_seed_overlap.json)
at each candidate k the k-sweep will use ({13, 32, 64}).

READ-ONLY with respect to the mask: this module never writes to
output/profiling/reasoning-merged-v4/ (T-11-06). It emits
output/sieve/protected_retention_check.json, carrying layer_stability_notes
forward verbatim from the Phase-7 mask JSON for plan 11-05 / Phase 13.

Usage:
    python -m scripts.sieve_protected_retention
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Bootstrap for direct `python scripts/sieve_protected_retention.py` execution
# (established repo convention, e.g. scripts/sieve_ksweep_run.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sieve_cross_seed_overlap import SEED_DIRS, load_seed_counts, topk_sets  # noqa: E402

MASK_DIR = PROJECT_ROOT / "output/profiling/reasoning-merged-v4"
CANDIDATE_KS = (13, 32, 64)


def retention_check(retained: dict[int, set], mask: np.ndarray) -> bool:
    """True iff every protected (layer, expert) pair is in the retained set.

    Args:
        retained: {layer_idx: set of retained expert ids}. Missing layers count
            as retaining nothing.
        mask: [n_layers, n_experts] bool protected mask.
    """
    for layer_idx in range(mask.shape[0]):
        protected = set(np.where(mask[layer_idx])[0].tolist())
        if not protected <= retained.get(layer_idx, set()):
            return False
    return True


def at_risk_count(retained: dict[int, set], mask: np.ndarray) -> int:
    """Number of protected (layer, expert) pairs NOT in the retained set."""
    n = 0
    for layer_idx in range(mask.shape[0]):
        protected = set(np.where(mask[layer_idx])[0].tolist())
        n += len(protected - retained.get(layer_idx, set()))
    return n


def build_retained_sets(seed_counts: dict[str, np.ndarray], mode: str, k: int) -> dict[int, set]:
    """Retained (hot) expert sets per layer at budget k, per the profile mode.

    "shared": one profile = top-k of the summed counts across seeds.
    "union":  union of each seed's own top-k.
    """
    n_layers = next(iter(seed_counts.values())).shape[0]
    if mode == "shared":
        summed = sum(seed_counts.values())
        sets = topk_sets(summed, k)
        return {layer: sets[layer] for layer in range(n_layers)}
    if mode == "union":
        retained: dict[int, set] = {layer: set() for layer in range(n_layers)}
        for counts in seed_counts.values():
            sets = topk_sets(counts, k)
            for layer in range(n_layers):
                retained[layer] |= sets[layer]
        return retained
    raise ValueError(f"unknown sieve_profile_mode: {mode!r}")


def main():
    # --- Load + assert the immutable protected-expert mask (read-only) ---
    # Shape/count are derived from the loaded mask itself (GATE4-02): the v3
    # mask is [48,128]/1480, the v4 mask is a fresh Phase-25 profile of unknown
    # count/shape -- only dtype and non-emptiness are universal invariants.
    mask = np.load(MASK_DIR / "protected_expert_mask.npy")
    assert mask.dtype == bool, f"mask dtype {mask.dtype} != bool"
    assert mask.sum() > 0, "protected mask is empty"

    mask_json = json.loads((MASK_DIR / "protected_expert_mask.json").read_text())
    layer_stability_notes = mask_json.get("layer_stability_notes")
    assert layer_stability_notes is not None, "layer_stability_notes missing from Phase-7 mask JSON"

    # --- Cross-seed decision (Task 2 output) ---
    overlap = json.loads((PROJECT_ROOT / "output/sieve/cross_seed_overlap.json").read_text())
    mode = overlap["sieve_profile_mode"]

    # --- Per-seed routing counts ---
    seed_counts = {
        seed: load_seed_counts(PROJECT_ROOT / d / "routing_report.jsonl")
        for seed, d in SEED_DIRS.items()
    }

    # --- Per-k retention check ---
    per_k = {}
    for k in CANDIDATE_KS:
        retained = build_retained_sets(seed_counts, mode, k)
        per_k[str(k)] = {
            "protected_retained": retention_check(retained, mask),
            "protected_at_risk": at_risk_count(retained, mask),
        }

    report = {
        "analysis": "protected_retention_check",
        "mask_source": "output/profiling/reasoning-merged-v4/protected_expert_mask.npy",
        "mask_total_protected": int(mask.sum()),
        "sieve_profile_mode": mode,
        "mean_overlap": overlap["mean_overlap"],
        "candidate_ks": list(CANDIDATE_KS),
        "per_k": per_k,
        "note": (
            "protected_at_risk counts protected (layer, expert) pairs OUTSIDE the "
            "pure top-k hot set at that budget; the k-sweep mask MUST additionally "
            "retain these (protected mask is inviolable per 11-CONTEXT HARD CONSTRAINT 1)."
        ),
        "layer_stability_notes": layer_stability_notes,
    }

    out_path = PROJECT_ROOT / "output/sieve/protected_retention_check.json"
    out_path.write_text(json.dumps(report, indent=2))
    for k, r in per_k.items():
        print(f"k={k}: protected_retained={r['protected_retained']} at_risk={r['protected_at_risk']}")
    print(f"Wrote {out_path}")


def _self_check():
    """Assert-based self-check on the tiny synthetic fixture from the test contract."""
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, 1] = True
    mask[1, 3] = True
    assert retention_check({0: {0, 2}, 1: {3}}, mask) is False
    assert retention_check({0: {0, 1, 2}, 1: {1, 3}}, mask) is True
    assert at_risk_count({0: {0, 2}, 1: {3}}, mask) == 1
    # union mode: two seeds with disjoint top-1 -> union of both
    counts = {"a": np.array([[5.0, 1.0]]), "b": np.array([[1.0, 5.0]])}
    assert build_retained_sets(counts, "union", 1) == {0: {0, 1}}
    assert build_retained_sets(counts, "shared", 1) == {0: {0}} or build_retained_sets(counts, "shared", 1) == {0: {1}}
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
