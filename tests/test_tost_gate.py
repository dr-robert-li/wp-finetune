"""Unit tests for scripts.tost_gate (SIEVE-05: TOST equivalence gate, epsilon=2pp).

Tests are GPU-free: synthetic numpy score arrays. Module-level importorskip so this
file SKIPs cleanly while scripts/tost_gate.py is absent (lands in a later wave),
mirroring tests/test_bootstrap_ci.py's assertion style and fixture shape.
"""
from __future__ import annotations

import numpy as np
import pytest

tost_gate = pytest.importorskip("scripts.tost_gate")


EPSILON = 0.02  # 2pp equivalence margin, per 11-CONTEXT.md TOST gate spec


class TestTostEquivalence:
    def test_mean_difference_inside_epsilon_is_equivalent(self):
        """Two score arrays with mean difference well inside +/-epsilon -> equivalent=True."""
        np.random.seed(0)
        a = np.random.normal(loc=0.85, scale=0.01, size=100)
        b = np.random.normal(loc=0.855, scale=0.01, size=100)  # diff ~0.005 << 0.02
        assert tost_gate.tost_equivalence(a, b, epsilon=EPSILON) is True

    def test_mean_difference_outside_epsilon_is_not_equivalent(self):
        """Two score arrays with mean difference well outside +/-epsilon -> equivalent=False."""
        np.random.seed(1)
        a = np.random.normal(loc=0.85, scale=0.01, size=100)
        b = np.random.normal(loc=0.75, scale=0.01, size=100)  # diff ~0.10 >> 0.02
        assert tost_gate.tost_equivalence(a, b, epsilon=EPSILON) is False

    def test_identical_arrays_are_equivalent(self):
        """Zero mean difference is trivially within epsilon -> equivalent=True."""
        values = np.array([0.9, 0.91, 0.89, 0.92, 0.88])
        assert tost_gate.tost_equivalence(values, values, epsilon=EPSILON) is True

    def test_returns_bool(self):
        """tost_equivalence returns a plain bool (not numpy bool_ or tuple)."""
        a = np.array([0.9, 0.91, 0.89])
        b = np.array([0.9, 0.91, 0.89])
        result = tost_gate.tost_equivalence(a, b, epsilon=EPSILON)
        assert isinstance(result, bool)
