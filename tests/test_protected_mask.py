"""Unit tests for extract_protected_mask.py (D-03 co-activation rule).

Tests are GPU-free: uses synthetic numpy arrays.
Covers: co-activation mask logic, sensitivity table, export functions.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.extract_protected_mask import (
    extract_protected_mask,
    sensitivity_table,
    export_mask,
)


# ---------------------------------------------------------------------------
# Tests: extract_protected_mask — D-03 co-activation rule
# ---------------------------------------------------------------------------


class TestExtractProtectedMask:
    """D-03: Expert protected iff above per-layer mean in BOTH wp_gen AND wp_judge."""

    def test_co_activation_flags_dual_purpose(self):
        """Expert above mean in BOTH splits is flagged True."""
        gen = np.zeros((1, 4))
        judge = np.zeros((1, 4))
        gen[0] = [200, 50, 50, 50]    # expert 0 above mean in gen
        judge[0] = [200, 50, 50, 50]  # expert 0 above mean in judge
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == True, "Expert 0 high in both splits should be masked"

    def test_single_split_above_mean_not_flagged(self):
        """Expert above mean in only ONE split is NOT flagged."""
        gen = np.zeros((1, 4))
        judge = np.zeros((1, 4))
        gen[0] = [200, 50, 50, 50]    # expert 0 above mean in gen, not judge
        judge[0] = [50, 200, 50, 50]  # expert 1 above mean in judge, not gen
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == False, "High gen only should NOT be masked"
        assert mask[0, 1] == False, "High judge only should NOT be masked"

    def test_mask_shape_and_dtype(self):
        """Output mask shape is [n_layers, n_experts], dtype bool."""
        gen = np.ones((48, 128))
        judge = np.ones((48, 128))
        mask = extract_protected_mask(gen, judge)
        assert mask.shape == (48, 128), f"Expected (48, 128), got {mask.shape}"
        assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"

    def test_all_equal_counts_no_mask(self):
        """Uniform counts: no expert strictly above mean -> mask all False."""
        gen = np.ones((2, 8)) * 50.0
        judge = np.ones((2, 8)) * 50.0
        mask = extract_protected_mask(gen, judge)
        assert not mask.any(), "Uniform counts: no expert should be masked (none strictly above mean)"

    def test_co_activation_uses_and_not_or(self):
        """Mask is AND (both above mean), not OR (either above mean)."""
        gen = np.zeros((1, 4))
        judge = np.zeros((1, 4))
        # Expert 0: high gen, low judge
        # Expert 1: low gen, high judge
        # Expert 2: high gen, high judge -> should be masked
        gen[0] = [200, 10, 200, 10]
        judge[0] = [10, 200, 200, 10]
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == False, "High gen only (expert 0) should NOT be masked"
        assert mask[0, 1] == False, "High judge only (expert 1) should NOT be masked"
        assert mask[0, 2] == True, "High both (expert 2) SHOULD be masked"

    def test_per_layer_mean_is_computed_per_layer(self):
        """Mean threshold is computed per-layer independently."""
        # Layer 0: expert 0 has count 100 (above per-layer mean), all others 10
        # Layer 1: expert 0 has count 10 (equal to per-layer mean), should NOT be masked
        gen = np.ones((2, 4)) * 10.0
        judge = np.ones((2, 4)) * 10.0
        gen[0, 0] = 100.0
        judge[0, 0] = 100.0
        # Layer 1: all equal at 10 -> no expert strictly above mean
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == True, "Layer 0: expert 0 should be masked"
        assert not mask[1].any(), "Layer 1: no expert should be masked (all equal)"


# ---------------------------------------------------------------------------
# Tests: sensitivity_table
# ---------------------------------------------------------------------------


class TestSensitivityTable:
    def test_has_three_threshold_variants(self):
        """sensitivity_table returns entries for mean, median, and top-k thresholds."""
        gen = np.random.default_rng(0).integers(1, 100, (4, 16)).astype(float)
        judge = np.random.default_rng(1).integers(1, 100, (4, 16)).astype(float)
        table = sensitivity_table(gen, judge, top_k=8)
        assert "mean_threshold" in table, "Missing mean_threshold"
        assert "median_threshold" in table, "Missing median_threshold"
        # top_k intersection is the third variant
        topk_keys = [k for k in table.keys() if "topk" in k or "top_k" in k or "top" in k.lower()]
        assert len(topk_keys) >= 1, f"Missing top-k threshold variant, got keys: {list(table.keys())}"

    def test_each_variant_has_mask_sizes_and_total(self):
        """Each threshold variant reports per_layer mask sizes and total_protected."""
        gen = np.random.default_rng(2).integers(10, 200, (4, 16)).astype(float)
        judge = np.random.default_rng(3).integers(10, 200, (4, 16)).astype(float)
        table = sensitivity_table(gen, judge, top_k=4)
        for variant_name, variant in table.items():
            assert "mask_size_per_layer" in variant, f"{variant_name} missing mask_size_per_layer"
            assert "total_protected" in variant, f"{variant_name} missing total_protected"

    def test_mean_threshold_consistent_with_extract(self):
        """mean_threshold total_protected matches extract_protected_mask total."""
        gen = np.random.default_rng(4).integers(1, 100, (4, 16)).astype(float)
        judge = np.random.default_rng(5).integers(1, 100, (4, 16)).astype(float)
        mask = extract_protected_mask(gen, judge)
        table = sensitivity_table(gen, judge, top_k=8)
        assert table["mean_threshold"]["total_protected"] == int(mask.sum())


# ---------------------------------------------------------------------------
# Tests: export_mask
# ---------------------------------------------------------------------------


class TestExportMask:
    def test_exports_npy_and_json_sidecar(self):
        """export_mask writes both .npy and .json sidecar files."""
        gen = np.random.default_rng(6).integers(1, 100, (4, 8)).astype(float)
        judge = np.random.default_rng(7).integers(1, 100, (4, 8)).astype(float)
        mask = extract_protected_mask(gen, judge)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            export_mask(mask, out_dir)
            assert (out_dir / "protected_expert_mask.npy").exists(), "Missing .npy file"
            assert (out_dir / "protected_expert_mask.json").exists(), "Missing .json sidecar"

    def test_npy_has_correct_shape_and_dtype(self):
        """Loaded .npy has original shape and bool dtype."""
        mask = np.zeros((4, 8), dtype=bool)
        mask[1, 3] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            export_mask(mask, out_dir)
            loaded = np.load(out_dir / "protected_expert_mask.npy")
            assert loaded.shape == mask.shape
            assert loaded.dtype == bool
            assert loaded[1, 3] == True

    def test_json_sidecar_has_layer_expert_mapping(self):
        """JSON sidecar maps str(layer_idx) -> list of protected expert IDs."""
        mask = np.zeros((2, 4), dtype=bool)
        mask[0, 2] = True
        mask[1, 0] = True
        mask[1, 3] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            export_mask(mask, out_dir)
            with open(out_dir / "protected_expert_mask.json") as f:
                sidecar = json.load(f)
            assert "0" in sidecar
            assert "1" in sidecar
            assert 2 in sidecar["0"]
            assert 0 in sidecar["1"]
            assert 3 in sidecar["1"]
