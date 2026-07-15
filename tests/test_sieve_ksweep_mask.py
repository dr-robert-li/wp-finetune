"""Unit tests for scripts.sieve_expert_mask_inference (SIEVE-04: k-sweep expert mask).

Tests are GPU-free: tiny synthetic 2-layer routing-count fixture (no model load).
Module-level importorskip so this file SKIPs cleanly while
scripts/sieve_expert_mask_inference.py is absent (lands in a later wave).
"""
from __future__ import annotations

import numpy as np
import pytest

sieve_mask_inf = pytest.importorskip("scripts.sieve_expert_mask_inference")


N_EXPERTS = 128
K = 13


def _routing_counts() -> np.ndarray:
    """2 layers x 128 experts; descending unique counts so hot/cold rank is unambiguous.

    Expert index 0 is hottest (count=128) ... expert index 127 is coldest (count=1),
    identical ranking in both layers.
    """
    row = np.arange(N_EXPERTS, 0, -1, dtype=float)  # 128, 127, ..., 1
    return np.stack([row, row])


def _protected_mask_no_overlap_outside_topk() -> np.ndarray:
    """Layer 1: protected experts are already inside the top-13 hot set (indices 0-12)."""
    mask = np.zeros((2, N_EXPERTS), dtype=bool)
    mask[1, 0] = True
    mask[1, 5] = True
    return mask


def _protected_mask_with_cold_outlier() -> np.ndarray:
    """Layer 0: one protected expert (index 127, the globally coldest) sits OUTSIDE top-13."""
    mask = np.zeros((2, N_EXPERTS), dtype=bool)
    mask[0, 3] = True     # inside top-13 hot set
    mask[0, 127] = True   # coldest expert, but protected -> must be retained anyway
    return mask


class TestKsweepMaskUnion:
    def test_protected_subset_of_hot_keeps_exactly_k(self):
        """When protected experts are already in the top-k hot set, union size == k."""
        counts = _routing_counts()
        protected = _protected_mask_no_overlap_outside_topk()
        kept = sieve_mask_inf.build_ksweep_mask(counts, protected, k=K)

        layer1_kept = np.where(kept[1])[0]
        assert kept[1].sum() == K, f"Expected exactly {K} kept experts, got {kept[1].sum()}"
        assert set(layer1_kept) == set(range(K)), "Kept set should be exactly top-13 hottest indices"

    def test_protected_outside_hot_expands_union_beyond_k(self):
        """A protected-but-cold expert must be retained even though it is NOT in the top-k hot set."""
        counts = _routing_counts()
        protected = _protected_mask_with_cold_outlier()
        kept = sieve_mask_inf.build_ksweep_mask(counts, protected, k=K)

        assert kept[0, 127] == True, "Protected coldest expert must be retained (never masked out)"
        assert kept[0].sum() == K + 1, (
            f"Union of top-{K} hot + 1 protected outlier should keep {K + 1} experts, "
            f"got {kept[0].sum()}"
        )
        # never fewer than the protected count
        assert kept[0].sum() >= int(protected[0].sum())

    def test_masked_out_experts_are_the_coldest(self):
        """Non-protected masked-out experts must all have LOWER routing counts than kept hot experts."""
        counts = _routing_counts()
        protected = _protected_mask_no_overlap_outside_topk()
        kept = sieve_mask_inf.build_ksweep_mask(counts, protected, k=K)

        kept_counts = counts[1][kept[1]]
        masked_out_counts = counts[1][~kept[1]]
        assert masked_out_counts.max() < kept_counts.min(), (
            "Every masked-out expert should have a strictly lower routing count than every kept expert"
        )


class TestKsweepMaskV4Scale:
    def test_v4_scale_256_experts_shape_generic(self):
        """build_ksweep_mask is shape-driven off the arrays passed in -- no
        hardcoded 128 -- so it works unchanged at the v4 256-expert scale."""
        n_experts = 256
        row = np.arange(n_experts, 0, -1, dtype=float)
        counts = np.stack([row, row])
        protected = np.zeros((2, n_experts), dtype=bool)
        protected[1, 255] = True  # coldest v4 expert, protected -> must survive

        kept = sieve_mask_inf.build_ksweep_mask(counts, protected, k=32)
        assert kept[1, 255] == True  # noqa: E712
        assert kept[1].sum() == 33  # top-32 hot + 1 protected outlier
