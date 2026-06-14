"""Unit tests for compute_jaccard_stability (PROF-03, D-06 literal subsample-vs-FULL).

Tests are GPU-free: uses synthetic numpy arrays.
The CI-aware jaccard_ci_lower disposition is tested separately in test_bootstrap_ci.py
where bootstrap_ci / jaccard_disposition live (Task 2).
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.profile_merged_model import compute_jaccard_stability


# ---------------------------------------------------------------------------
# Tests: compute_jaccard_stability
# ---------------------------------------------------------------------------


class TestComputeJaccardStability:
    """D-06 literal: subsample_counts vs full_counts per layer."""

    def _uniform_counts(self, n_layers: int = 4, n_experts: int = 8) -> np.ndarray:
        """All experts equal counts — top-K is arbitrary but deterministic."""
        return np.ones((n_layers, n_experts), dtype=float)

    def test_identical_counts_all_ones(self):
        """When subsample == full, Jaccard == 1.0 for every layer."""
        counts = np.random.default_rng(42).integers(1, 100, size=(4, 8)).astype(float)
        result = compute_jaccard_stability(counts, counts, top_k=2)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, 1.0, atol=1e-9)

    def test_disjoint_top_k_returns_zero(self):
        """When full and subsample top-K sets are disjoint, Jaccard == 0.0."""
        n_experts = 8
        full_counts = np.zeros((1, n_experts))
        sub_counts = np.zeros((1, n_experts))
        # full top-2: experts 0,1 (high counts)
        full_counts[0, 0] = 100
        full_counts[0, 1] = 90
        # subsample top-2: experts 6,7 (high counts in different positions)
        sub_counts[0, 6] = 100
        sub_counts[0, 7] = 90

        result = compute_jaccard_stability(full_counts, sub_counts, top_k=2)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_partial_overlap(self):
        """Partial overlap: 1 shared expert out of top-2 -> Jaccard = 1/(2+2-1) = 1/3."""
        full_counts = np.zeros((1, 8))
        sub_counts = np.zeros((1, 8))
        # full top-2: experts 0, 1
        full_counts[0, 0] = 100
        full_counts[0, 1] = 90
        # subsample top-2: experts 0, 2 (share expert 0)
        sub_counts[0, 0] = 100
        sub_counts[0, 2] = 80

        result = compute_jaccard_stability(full_counts, sub_counts, top_k=2)
        # intersection=1, union=3 -> Jaccard = 1/3
        assert result[0] == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_output_shape_matches_n_layers(self):
        """Output array length equals number of layers."""
        n_layers = 48
        counts = np.random.default_rng(0).integers(1, 50, size=(n_layers, 128)).astype(float)
        result = compute_jaccard_stability(counts, counts, top_k=8)
        assert result.shape == (n_layers,)

    def test_values_between_zero_and_one(self):
        """All Jaccard values must be in [0, 1]."""
        rng = np.random.default_rng(7)
        full = rng.integers(1, 200, size=(10, 32)).astype(float)
        sub = rng.integers(1, 200, size=(10, 32)).astype(float)
        result = compute_jaccard_stability(full, sub, top_k=4)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_gate_passes_when_all_above_threshold(self):
        """np.all(jaccards >= 0.94) is True when all layers are at/above threshold."""
        # All layers identical -> all Jaccard == 1.0 -> gate passes
        counts = np.random.default_rng(99).integers(1, 100, size=(48, 128)).astype(float)
        result = compute_jaccard_stability(counts, counts, top_k=8)
        assert np.all(result >= 0.94)

    def test_gate_fails_when_any_below_threshold(self):
        """np.all(jaccards >= 0.94) is False when any layer is below threshold."""
        n_experts = 8
        full_counts = np.ones((3, n_experts))
        sub_counts = np.ones((3, n_experts))
        # Make layer 1 fully disjoint
        full_counts[1, :] = 0
        full_counts[1, 0] = 10
        full_counts[1, 1] = 9
        sub_counts[1, :] = 0
        sub_counts[1, 6] = 10
        sub_counts[1, 7] = 9

        result = compute_jaccard_stability(full_counts, sub_counts, top_k=2)
        assert not np.all(result >= 0.94)

    def test_empty_layer_returns_one(self):
        """Layer with all-zero counts treated gracefully (union == 0 -> Jaccard = 1.0)."""
        full_counts = np.zeros((2, 8))
        sub_counts = np.zeros((2, 8))
        result = compute_jaccard_stability(full_counts, sub_counts, top_k=2)
        # All-zero counts: both top-K sets are the same (argsort of zeros is arbitrary
        # but deterministic) -> Jaccard == 1.0
        assert result[0] == pytest.approx(1.0, abs=1e-9)
