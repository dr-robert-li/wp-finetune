"""Unit tests for scripts.reap_prune (PRUNE-02: REAP calibration-saliency scorer).

Tests are GPU-free: drive REAPCollector with a synthetic fixture of
(layer_idx, expert_idx, gate_weight, expert_output_norm) hook events -- no
model load, no forward pass. Module-level importorskip so this file SKIPs
cleanly while scripts/reap_prune.py is absent, mirroring
tests/test_aimer_prune.py's convention.
"""
from __future__ import annotations

import numpy as np
import pytest

reap_prune = pytest.importorskip("scripts.reap_prune")

N_LAYERS = 48
N_EXPERTS = 128


def test_scores_shape():
    c = reap_prune.REAPCollector(n_layers=N_LAYERS, n_experts=N_EXPERTS)
    c.record(layer_idx=0, expert_idx=0, gate_weight=0.5, expert_output_norm=2.0)
    scores = c.scores()
    assert scores.shape == (N_LAYERS, N_EXPERTS)


def test_hand_computed_saliency_mean():
    """S_j = mean over active tokens of g_j(x) * ||f_j(x)||_2 (deprecated/wp-moe.md REAP formula)."""
    c = reap_prune.REAPCollector(n_layers=2, n_experts=4)
    # Expert (layer=0, expert=1) activated by 3 tokens: gate*norm products 0.4, 1.2, 0.8
    events = [
        (0, 1, 0.2, 2.0),   # product 0.4
        (0, 1, 0.3, 4.0),   # product 1.2
        (0, 1, 0.4, 2.0),   # product 0.8
    ]
    for layer, expert, gate, norm in events:
        c.record(layer_idx=layer, expert_idx=expert, gate_weight=gate, expert_output_norm=norm)
    scores = c.scores()
    expected = np.mean([0.4, 1.2, 0.8])
    assert np.isclose(scores[0, 1], expected)

    # A second active expert, single activation.
    c.record(layer_idx=1, expert_idx=3, gate_weight=1.0, expert_output_norm=0.5)
    scores = c.scores()
    assert np.isclose(scores[1, 3], 0.5)


def test_inactive_expert_scores_zero_no_divide_by_zero():
    c = reap_prune.REAPCollector(n_layers=2, n_experts=4)
    c.record(layer_idx=0, expert_idx=0, gate_weight=1.0, expert_output_norm=1.0)
    scores = c.scores()
    # every (layer, expert) never recorded must be a defined 0.0, not nan/inf.
    assert np.isfinite(scores).all()
    assert scores[0, 1] == 0.0
    assert scores[1, 2] == 0.0


def test_reset_clears_accumulators():
    c = reap_prune.REAPCollector(n_layers=2, n_experts=4)
    c.record(layer_idx=0, expert_idx=0, gate_weight=1.0, expert_output_norm=1.0)
    c.reset()
    scores = c.scores()
    assert np.all(scores == 0.0)
