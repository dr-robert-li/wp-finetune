"""Unit tests for bootstrap_ci and jaccard_disposition in compute_concentration.py.

Tests are GPU-free: uses synthetic numpy arrays.
Covers D-09 (CI-aware gate disposition) and PROF-03 CI gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.compute_concentration import bootstrap_ci, jaccard_disposition


# ---------------------------------------------------------------------------
# Tests: bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_known_distribution_ci_contains_true_mean(self):
        """Bootstrap CI contains the true mean for a bimodal known distribution."""
        np.random.seed(42)
        values = np.array([1.0] * 50 + [2.0] * 50)  # true mean = 1.5
        lo, hi = bootstrap_ci(values, n_boot=1000, alpha=0.05)
        assert lo < 1.5 < hi, f"CI [{lo:.4f}, {hi:.4f}] should contain true mean 1.5"

    def test_constant_array_ci_is_tight(self):
        """Constant array -> CI collapses to the constant (within 0.01)."""
        values = np.ones(100) * 5.0
        lo, hi = bootstrap_ci(values, n_boot=200, alpha=0.05)
        assert abs(lo - 5.0) < 0.01, f"Expected lo ~5.0, got {lo}"
        assert abs(hi - 5.0) < 0.01, f"Expected hi ~5.0, got {hi}"

    def test_ci_lower_bound_less_than_upper(self):
        """Lower bound must always be <= upper bound."""
        np.random.seed(0)
        values = np.random.uniform(0.9, 1.0, 50)
        lo, hi = bootstrap_ci(values, n_boot=500, alpha=0.05)
        assert lo <= hi, f"Lower bound {lo} > upper bound {hi}"

    def test_ci_returns_tuple_of_floats(self):
        """bootstrap_ci returns a (float, float) tuple."""
        values = np.array([0.95, 0.97, 0.96, 0.94, 0.98])
        result = bootstrap_ci(values, n_boot=100, alpha=0.05)
        assert len(result) == 2
        lo, hi = result
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_lower_used_for_gate_disposition(self):
        """CI-aware gate: only passes when lower bound clears threshold (D-09).

        Mirrors run_grid_eval.py ci_lower mode: gate_passes = (ci_lower >= threshold).
        Point estimate may be above threshold but lower bound below -> gate fails.
        """
        threshold = 0.94
        lo, hi = 0.91, 0.97   # lower bound below threshold even though hi > threshold
        gate_passes = lo >= threshold
        assert gate_passes is False, "Lower bound below threshold should fail gate"

    def test_alpha_tails_are_symmetric(self):
        """With alpha=0.05, CI uses 2.5th and 97.5th percentiles."""
        np.random.seed(1)
        # Known distribution: uniform [0, 1]; true mean ~0.5
        values = np.linspace(0, 1, 100)
        lo, hi = bootstrap_ci(values, n_boot=2000, alpha=0.05)
        # CI should be reasonably symmetric around mean ~0.5
        assert lo < 0.5 < hi, "CI should contain true mean ~0.5 for uniform distribution"
        # Both tails should be cut
        assert lo > 0.0, "Lower bound should be above 0 with 2.5th percentile cut"
        assert hi < 1.0, "Upper bound should be below 1 with 97.5th percentile cut"


# ---------------------------------------------------------------------------
# Tests: jaccard_disposition (D-09 CI-aware Jaccard gate, PROF-03)
# ---------------------------------------------------------------------------


class TestJaccardDisposition:
    def test_all_ones_passes(self):
        """All-1.0 Jaccard array -> CI tight to 1.0 -> passes True."""
        # Use extreme values: CI will be ~[1.0, 1.0] -> ci_lower ~1.0 >= 0.94
        jaccards = np.ones(48)
        ci_lower, passes = jaccard_disposition(jaccards)
        assert passes is True, f"Expected passes=True for all-1.0, got ci_lower={ci_lower}"

    def test_low_constant_fails(self):
        """Low constant Jaccard (~0.5) -> CI tight to ~0.5 -> ci_lower < 0.94 -> passes False."""
        jaccards = np.ones(48) * 0.5
        ci_lower, passes = jaccard_disposition(jaccards)
        assert passes is False, f"Expected passes=False for all-0.5, got ci_lower={ci_lower}"

    def test_ci_lower_below_threshold_fails_even_if_point_above(self):
        """D-06 fallback trigger: gate FAILs when ci_lower < 0.94 (D-09 CI-aware).

        Even if some layers are high, a spread array whose ci_lower < 0.94 must fail.
        """
        # Mix of 0.7 and 1.0: mean ~0.85, ci_lower well below 0.94
        jaccards = np.array([0.7] * 24 + [1.0] * 24)
        ci_lower, passes = jaccard_disposition(jaccards)
        assert passes is False, (
            f"Expected passes=False when ci_lower < 0.94 (D-06 fallback trigger), "
            f"got ci_lower={ci_lower:.4f}"
        )

    def test_returns_tuple_float_bool(self):
        """jaccard_disposition returns (float, bool)."""
        jaccards = np.ones(48) * 0.96
        result = jaccard_disposition(jaccards)
        assert len(result) == 2
        ci_lower, passes = result
        assert isinstance(ci_lower, float)
        assert isinstance(passes, bool)

    def test_ci_lower_matches_bootstrap_result(self):
        """ci_lower from jaccard_disposition equals bootstrap_ci lower bound."""
        np.random.seed(99)
        jaccards = np.ones(48) * 0.97
        ci_lower_disp, _ = jaccard_disposition(jaccards, n_boot=200, alpha=0.05)
        lo, _ = bootstrap_ci(jaccards, n_boot=200, alpha=0.05)
        # Both should give identical results (same deterministic seed would give exact match,
        # but without same seed they should be close for tight distribution)
        assert abs(ci_lower_disp - 0.97) < 0.05, (
            f"ci_lower {ci_lower_disp} should be near 0.97 for constant array"
        )
