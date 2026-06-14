"""Unit tests for concentration metrics in compute_concentration.py (PROF-04).

Tests are GPU-free: uses synthetic numpy arrays and known-value assertions.
Covers: CV, cumulative coverage, E_eff uniform, eeff_delta direction.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.compute_concentration import (
    compute_cv,
    cumulative_coverage,
    layer_depth_skew,
    compute_eeff_delta,
)
from scripts.profile_base_model import compute_eeff


# ---------------------------------------------------------------------------
# Tests: CV (coefficient of variation)
# ---------------------------------------------------------------------------


class TestComputeCV:
    def test_cv_uniform_distribution(self):
        """Uniform counts -> CV ~= 0 (std == 0)."""
        counts = np.ones(128) * 100.0
        cv = compute_cv(counts)
        assert cv < 0.01, f"Expected CV ~0 for uniform distribution, got {cv}"

    def test_cv_concentrated_distribution(self):
        """Concentrated counts (one expert dominates) -> CV > 1."""
        counts = np.zeros(128)
        counts[0] = 10000.0
        cv = compute_cv(counts)
        assert cv > 1.0, f"Expected CV > 1 for concentrated distribution, got {cv}"

    def test_cv_zero_mean_returns_zero(self):
        """All-zero counts -> CV == 0 (guard against division by zero)."""
        counts = np.zeros(128)
        cv = compute_cv(counts)
        assert cv == 0.0, f"Expected 0.0 for all-zero counts, got {cv}"


# ---------------------------------------------------------------------------
# Tests: cumulative_coverage
# ---------------------------------------------------------------------------


class TestCumulativeCoverage:
    def test_coverage_sums_to_one(self):
        """Cumulative coverage at all experts == 1.0."""
        counts = np.random.default_rng(42).integers(1, 100, 128).astype(float)
        curve = cumulative_coverage(counts)
        assert curve.shape == (128,)
        assert abs(curve[-1] - 1.0) < 1e-9, f"Expected 1.0 at end, got {curve[-1]}"

    def test_coverage_monotone_increasing(self):
        """Cumulative coverage is monotone non-decreasing."""
        counts = np.random.default_rng(7).integers(1, 50, 64).astype(float)
        curve = cumulative_coverage(counts)
        diffs = np.diff(curve)
        assert np.all(diffs >= -1e-12), "Cumulative coverage must be non-decreasing"

    def test_coverage_single_expert_steps_to_one(self):
        """Single-expert counts: coverage jumps to 1.0 at the first sorted position."""
        counts = np.zeros(8)
        counts[3] = 100.0
        curve = cumulative_coverage(counts)
        assert abs(curve[0] - 1.0) < 1e-9, "Single expert should dominate entirely"

    def test_coverage_uniform_linear(self):
        """Uniform counts: coverage is linear, 1/n per expert."""
        n = 8
        counts = np.ones(n) * 50.0
        curve = cumulative_coverage(counts)
        for i in range(n):
            expected = (i + 1) / n
            assert abs(curve[i] - expected) < 1e-9, f"Position {i}: expected {expected}, got {curve[i]}"


# ---------------------------------------------------------------------------
# Tests: compute_eeff (via profile_base_model import)
# ---------------------------------------------------------------------------


class TestEeffUniform:
    def test_eeff_uniform_128_is_near_128(self):
        """Uniform expert counts -> E_eff ~= 128 (within 0.5)."""
        counts = {i: 100 for i in range(128)}
        result = compute_eeff(counts)
        assert abs(result - 128.0) < 0.5, f"Expected ~128, got {result}"


# ---------------------------------------------------------------------------
# Tests: eeff_delta direction
# ---------------------------------------------------------------------------


class TestEeffDelta:
    def test_eeff_delta_direction(self):
        """E_eff delta = merged - base; more-concentrated merged -> negative delta."""
        base_eeff = 45.0
        merged_eeff = 42.0
        delta = compute_eeff_delta(merged_eeff, base_eeff)
        assert delta == pytest.approx(-3.0), f"Expected -3.0, got {delta}"

    def test_eeff_delta_more_diffuse_is_positive(self):
        """More-diffuse merged -> positive delta."""
        base_eeff = 40.0
        merged_eeff = 50.0
        delta = compute_eeff_delta(merged_eeff, base_eeff)
        assert delta == pytest.approx(10.0), f"Expected 10.0, got {delta}"


# ---------------------------------------------------------------------------
# Tests: layer_depth_skew
# ---------------------------------------------------------------------------


class TestLayerDepthSkew:
    def test_skew_uniform_is_near_one(self):
        """Uniform CV across layers -> skew ~= 1.0 (early/late ratio)."""
        cv_per_layer = np.ones(48) * 0.5
        skew = layer_depth_skew(cv_per_layer)
        assert abs(skew - 1.0) < 0.01, f"Expected ~1.0 for uniform, got {skew}"

    def test_skew_early_concentrated_greater_than_one(self):
        """Early layers more concentrated (lower E_eff / higher CV) -> skew > 1."""
        cv_per_layer = np.zeros(48)
        cv_per_layer[:16] = 2.0   # early layers high CV
        cv_per_layer[32:] = 0.5   # late layers low CV
        skew = layer_depth_skew(cv_per_layer)
        assert skew > 1.0, f"Expected skew > 1 when early layers more concentrated, got {skew}"
