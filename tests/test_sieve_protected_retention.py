"""Unit tests for the Phase-7 protected-expert mask + Wave-0 retention-check contract.

Tests are GPU-free: the mask-artifact test loads the real Phase-7 .npy (small,
committed artifact); the retention-check tests use synthetic in-memory masks.
Covers SIEVE-01 (cross-seed routing profile / protected-expert subset check).

The mask-artifact test runs TODAY (no importorskip -- scripts/extract_protected_mask.py
and the mask .npy both already exist since Phase 7 sign-off). The retention-check
tests importorskip scripts.sieve_protected_retention per-test (not module-level) so
this file still SKIPs just those two tests -- not the whole module -- until a later
wave adds that module, matching the Wave-0 stub convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.extract_protected_mask import extract_protected_mask  # noqa: F401 (already exists)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = PROJECT_ROOT / "output/profiling/reasoning-merged-v4/protected_expert_mask.npy"


# ---------------------------------------------------------------------------
# Tests: Phase-7 mask artifact (runs today, no skip)
# ---------------------------------------------------------------------------


class TestProtectedMaskArtifact:
    def test_mask_shape_dtype_and_count(self):
        """Phase-7 mask loads as [48,128] bool with exactly 1,480 protected experts."""
        mask = np.load(MASK_PATH)
        assert mask.shape == (48, 128), f"Expected (48, 128), got {mask.shape}"
        assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"
        assert mask.sum() == 1480, f"Expected 1480 protected experts, got {mask.sum()}"


# ---------------------------------------------------------------------------
# Tests: retention check (SIEVE-01) -- future module, per-test importorskip
# ---------------------------------------------------------------------------


class TestRetentionCheck:
    def _tiny_mask(self) -> np.ndarray:
        """2 layers x 4 experts; layer0 protects expert1, layer1 protects expert3."""
        mask = np.zeros((2, 4), dtype=bool)
        mask[0, 1] = True
        mask[1, 3] = True
        return mask

    def test_retained_set_omitting_protected_expert_fails(self):
        """A retained-set that OMITS any protected expert -> retention_check returns False."""
        sieve_retention = pytest.importorskip("scripts.sieve_protected_retention")
        mask = self._tiny_mask()
        retained = {0: {0, 2}, 1: {3}}  # layer0 omits protected expert 1
        assert sieve_retention.retention_check(retained, mask) is False

    def test_superset_retained_set_passes(self):
        """A retained-set that is a superset of protected experts -> retention_check returns True."""
        sieve_retention = pytest.importorskip("scripts.sieve_protected_retention")
        mask = self._tiny_mask()
        retained = {0: {0, 1, 2}, 1: {1, 3}}  # superset: includes all protected experts
        assert sieve_retention.retention_check(retained, mask) is True
