"""Unit tests for scripts.prune_overlap (PRUNE-04: AIMER-vs-REAP domain-specificity
overlap analysis).

Tests are GPU-free: pure numpy boolean-mask fixtures, no checkpoint I/O.
Module-level importorskip so this file SKIPs cleanly while
scripts/prune_overlap.py is absent, mirroring tests/test_aimer_prune.py's
convention.
"""
from __future__ import annotations

import numpy as np
import pytest

prune_overlap = pytest.importorskip("scripts.prune_overlap")

N_LAYERS = 48
N_EXPERTS = 16


def test_identical_masks_jaccard_one():
    mask_a = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    mask_a[:, :8] = True
    mask_b = mask_a.copy()
    per_layer = prune_overlap.per_layer_jaccard(mask_a, mask_b)
    assert per_layer.shape == (N_LAYERS,)
    assert np.allclose(per_layer, 1.0)


def test_disjoint_masks_jaccard_zero():
    mask_a = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    mask_a[:, :8] = True
    mask_b = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    mask_b[:, 8:] = True
    per_layer = prune_overlap.per_layer_jaccard(mask_a, mask_b)
    assert per_layer.shape == (N_LAYERS,)
    assert np.allclose(per_layer, 0.0)


def test_hand_computed_partial_overlap():
    # Layer 0: keep {0,1,2,3} vs {2,3,4,5} -> intersection 2, union 6 -> 1/3.
    mask_a = np.zeros((1, N_EXPERTS), dtype=bool)
    mask_a[0, :4] = True
    mask_b = np.zeros((1, N_EXPERTS), dtype=bool)
    mask_b[0, 2:6] = True
    per_layer = prune_overlap.per_layer_jaccard(mask_a, mask_b)
    assert np.isclose(per_layer[0], 2 / 6)


def test_build_overlap_report_length_and_band_rollup():
    mask_a = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    mask_a[:, :8] = True
    mask_b = mask_a.copy()
    report = prune_overlap.build_overlap_report(mask_a, mask_b, ratio=25)
    assert report["n_layers"] == 48
    assert len(report["per_layer_jaccard"]) == 48
    assert report["mean"] == 1.0
    assert report["layer_stability_notes"]["low_jaccard_band"]["layers"] == [9, 13, 14, 31, 35, 36]
    assert report["layer_stability_notes"]["late_layer_band"]["layers"] == [45, 46, 47]
    assert "interpretation_stub" in report
