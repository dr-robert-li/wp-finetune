"""Unit tests for scripts.prune_selection (PRUNE-05: winner-selection rule).

Tests are GPU-free: a synthetic gate-result table (dicts), no checkpoint or
vLLM I/O. Module-level importorskip so this file SKIPs cleanly while
scripts/prune_selection.py is absent, mirroring tests/test_aimer_prune.py's
convention.
"""
from __future__ import annotations

import numpy as np
import pytest

prune_selection = pytest.importorskip("scripts.prune_selection")

N_LAYERS = 4
N_EXPERTS = 128


def _protected_mask():
    """Layer 1 carries 40 protected experts (mirrors the real prune_set_for_phase13
    fixture: layer 1 alone forces K >= 40, ruling out ratio=75/K=32)."""
    protected = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    protected[1, :40] = True
    return protected


def _base_record(method="aimer", ratio=25, **overrides):
    record = {
        "method": method, "ratio": ratio,
        "gen_wp_bench": 0.45, "judge_ensemble_rho": 0.78, "judge_parse_rate": 0.97,
        "d2_security_retention": 0.90, "d2_security_baseline": 0.90,
        "protected_retained": True,
    }
    record.update(overrides)
    return record


def test_clean_25_percent_winner():
    protected = _protected_mask()
    verdict = prune_selection.select_winner([_base_record(ratio=25)], protected)
    assert verdict["verdict"] == "winner"
    assert verdict["winner"] == {"method": "aimer", "ratio": 25, "k": 96}


def test_75_percent_passes_bars_but_physically_infeasible():
    protected = _protected_mask()
    clean_25 = _base_record(ratio=25)
    passes_but_infeasible_75 = _base_record(ratio=75)
    verdict = prune_selection.select_winner([clean_25, passes_but_infeasible_75], protected)
    by_ratio = {v["ratio"]: v for v in verdict["per_variant"]}
    assert by_ratio[75]["eligible"] is False
    assert any("infeasible" in r for r in by_ratio[75]["reasons"])
    # 25% still wins even though 75% was in the candidate set.
    assert verdict["winner"] == {"method": "aimer", "ratio": 25, "k": 96}


def test_d2_security_regression_disqualifies():
    protected = _protected_mask()
    regressed = _base_record(
        ratio=50, d2_security_retention=0.60, d2_security_baseline=0.90
    )
    verdict = prune_selection.select_winner([regressed], protected)
    assert verdict["verdict"] == "no_winner"
    reasons = verdict["per_variant"][0]["reasons"]
    assert any("d2_security regressed" in r for r in reasons)


def test_all_fail_returns_no_winner():
    protected = _protected_mask()
    infeasible_75 = _base_record(ratio=75)
    regressed_50 = _base_record(ratio=50, d2_security_retention=0.5, d2_security_baseline=0.9)
    failing_bar_25 = _base_record(ratio=25, gen_wp_bench=0.1)
    verdict = prune_selection.select_winner(
        [infeasible_75, regressed_50, failing_bar_25], protected
    )
    assert verdict["verdict"] == "no_winner"
    assert verdict["winner"] is None
    assert len(verdict["per_variant"]) == 3
    assert all(not v["eligible"] for v in verdict["per_variant"])


def test_prefers_smaller_k_higher_compression():
    protected = _protected_mask()
    variant_50 = _base_record(ratio=50)  # k=64
    variant_25 = _base_record(ratio=25)  # k=96
    verdict = prune_selection.select_winner([variant_50, variant_25], protected)
    assert verdict["winner"]["ratio"] == 50  # k=64 < k=96, higher compression wins


def test_ties_broken_by_higher_d2_security():
    protected = _protected_mask()
    aimer_25 = _base_record(method="aimer", ratio=25, d2_security_retention=0.85)
    reap_25 = _base_record(method="reap", ratio=25, d2_security_retention=0.89)
    verdict = prune_selection.select_winner([aimer_25, reap_25], protected)
    assert verdict["winner"]["method"] == "reap"  # same k=96, higher d2_security wins
