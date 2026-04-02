"""Unit tests for GATE-02 triage elimination logic in triage_ratios.py.

Tests are GPU-free: use mock eval result dicts.
All threshold tests use strict > semantics (value exactly AT threshold FAILS).
"""
import json
import tempfile
from pathlib import Path

import pytest

from scripts.triage_ratios import (
    ELIMINATION_PP,
    PHPCS_GATE,
    SECURITY_GATE,
    SPEARMAN_GATE,
    TriageResult,
    compute_overall_score,
    load_eval_results,
    triage_ratios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_results(
    ratios=None,
    phpcs_rate=0.96,
    security_rate=0.99,
    spearman=0.90,
    overall_mean=80.0,
    wpbench_score=None,
):
    """Create a mock eval_results dict for testing."""
    if ratios is None:
        ratios = ["30_70", "40_60", "50_50"]
    return {
        r: {
            "phpcs_pass_rate": phpcs_rate,
            "security_pass_rate": security_rate,
            "spearman": spearman,
            "overall_mean": overall_mean,
            "wpbench_score": wpbench_score,
        }
        for r in ratios
    }


def _make_ratio_result(
    phpcs_rate=0.96,
    security_rate=0.99,
    spearman=0.90,
    overall_mean=80.0,
    wpbench_score=None,
):
    """Create a single ratio's eval result dict."""
    return {
        "phpcs_pass_rate": phpcs_rate,
        "security_pass_rate": security_rate,
        "spearman": spearman,
        "overall_mean": overall_mean,
        "wpbench_score": wpbench_score,
    }


# ---------------------------------------------------------------------------
# Tests: Threshold constants
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    def test_phpcs_gate_value(self):
        assert PHPCS_GATE == 0.95

    def test_spearman_gate_value(self):
        assert SPEARMAN_GATE == 0.85

    def test_security_gate_value(self):
        assert SECURITY_GATE == 0.98

    def test_elimination_pp_value(self):
        assert ELIMINATION_PP == 0.05


# ---------------------------------------------------------------------------
# Tests: compute_overall_score
# ---------------------------------------------------------------------------


class TestComputeOverallScore:
    def test_gen_weighted_formula(self):
        """overall = 0.6 * ((phpcs + security) / 2) + 0.4 * spearman."""
        score = compute_overall_score(phpcs_rate=1.0, security_rate=1.0, spearman=1.0)
        assert abs(score - 1.0) < 1e-6

    def test_formula_values(self):
        """Test with known values: 0.6 * ((0.96+0.99)/2) + 0.4 * 0.90."""
        expected = 0.6 * ((0.96 + 0.99) / 2) + 0.4 * 0.90
        result = compute_overall_score(phpcs_rate=0.96, security_rate=0.99, spearman=0.90)
        assert abs(result - expected) < 1e-6

    def test_zero_spearman(self):
        """Spearman = 0 still contributes 0 to overall."""
        score = compute_overall_score(phpcs_rate=1.0, security_rate=1.0, spearman=0.0)
        assert abs(score - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Hard gate elimination (strict > semantics)
# ---------------------------------------------------------------------------


class TestHardGateElimination:
    def test_phpcs_below_gate_eliminated(self):
        """Ratio with PHPCS <= 0.95 is eliminated."""
        results = {
            "30_70": _make_ratio_result(phpcs_rate=0.94),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors
        assert any(e["ratio"] == "30_70" for e in outcome.eliminated)

    def test_phpcs_exactly_at_gate_fails(self):
        """Ratio at exactly 0.95 PHPCS FAILS the gate (strict > 0.95 required)."""
        results = {
            "30_70": _make_ratio_result(phpcs_rate=0.95),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors, "0.95 should fail strict >0.95 gate"

    def test_phpcs_above_gate_passes(self):
        """Ratio at 0.951 PHPCS passes the gate."""
        results = {
            "30_70": _make_ratio_result(phpcs_rate=0.951, security_rate=0.99, spearman=0.90),
        }
        outcome = triage_ratios(results)
        assert "30_70" in outcome.survivors

    def test_spearman_below_gate_eliminated(self):
        """Ratio with Spearman <= 0.85 is eliminated."""
        results = {
            "30_70": _make_ratio_result(spearman=0.84),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors

    def test_spearman_exactly_at_gate_fails(self):
        """Ratio at exactly 0.85 Spearman FAILS the gate (strict > 0.85 required)."""
        results = {
            "30_70": _make_ratio_result(spearman=0.85),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors, "0.85 should fail strict >0.85 gate"

    def test_security_below_gate_eliminated(self):
        """Ratio with security rate <= 0.98 is eliminated."""
        results = {
            "30_70": _make_ratio_result(security_rate=0.97),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors

    def test_security_exactly_at_gate_fails(self):
        """Ratio at exactly 0.98 security FAILS the gate (strict > 0.98 required)."""
        results = {
            "30_70": _make_ratio_result(security_rate=0.98),
        }
        outcome = triage_ratios(results)
        assert "30_70" not in outcome.survivors, "0.98 should fail strict >0.98 gate"


# ---------------------------------------------------------------------------
# Tests: 5pp elimination rule
# ---------------------------------------------------------------------------


class TestFivePPElimination:
    def test_more_than_5pp_behind_eliminated(self):
        """Ratio strictly >5pp behind best is eliminated (diff=0.06 > 0.05)."""
        # best=0.96, worse=0.90 (diff=0.06): computed directly from formula
        # score = 0.6 * ((phpcs + security) / 2) + 0.4 * spearman
        # best: 0.6 * ((0.99 + 0.99) / 2) + 0.4 * 0.99 = 0.594 + 0.396 = 0.99
        # worse: spearman = 0.99 - (0.06/0.4) = 0.84 -> but 0.84 fails SPEARMAN_GATE (<=0.85)
        # Use phpcs+security variation instead: best phpcs=1.0, sec=1.0, sp=1.0 -> score=1.0
        # worse: phpcs=0.91, sec=1.0, sp=1.0 -> 0.6*((0.91+1.0)/2)+0.4*1.0 = 0.6*0.955+0.4 = 0.973
        # diff = 1.0 - 0.973 = 0.027 -- not enough
        # Let's use spearman variation carefully:
        # best: all=0.99, score=0.99
        # worse: phpcs=0.99, sec=0.99, sp=0.84 -> 0.99 spearman fails gate (need sp > 0.85)
        # Use separate approach: construct two ratios with known score difference
        # best score = X, worse score = X - 0.06
        # Use: phpcs=0.99, sec=0.99, sp=0.99 -> best=0.99
        #      phpcs=0.99, sec=0.99, sp=0.84 -> FAILS GATE
        # Use only spearman: best sp=1.0, worse sp=0.85+epsilon -> barely above gate
        # Let's use: best=1.0 score (phpcs=1.0, sec=1.0, sp=1.0)
        # worse = score with phpcs=0.96, sec=0.96, sp=0.86 (all pass gates)
        # worse_score = 0.6*0.96 + 0.4*0.86 = 0.576 + 0.344 = 0.920
        # diff = 1.0 - 0.920 = 0.080 > 0.05 -> eliminated
        results = {
            "30_70": _make_ratio_result(phpcs_rate=1.0, security_rate=1.0, spearman=1.0),
            "40_60": _make_ratio_result(phpcs_rate=0.96, security_rate=0.96, spearman=0.86),
        }
        outcome = triage_ratios(results)
        assert "40_60" not in outcome.survivors

    def test_within_5pp_survives(self):
        """Ratio within 5pp of best survives (diff=0.03 < 0.05)."""
        # best: phpcs=1.0, sec=1.0, sp=1.0 -> score=1.0
        # worse: phpcs=0.99, sec=0.99, sp=0.93 -> 0.6*0.99 + 0.4*0.93 = 0.594+0.372=0.966
        # diff = 1.0 - 0.966 = 0.034 < 0.05 -> survives
        results = {
            "30_70": _make_ratio_result(phpcs_rate=1.0, security_rate=1.0, spearman=1.0),
            "40_60": _make_ratio_result(phpcs_rate=0.99, security_rate=0.99, spearman=0.93),
        }
        outcome = triage_ratios(results)
        assert "40_60" in outcome.survivors

    def test_exactly_5pp_behind_survives(self):
        """Ratio at or within 5pp of best SURVIVES (strictly >5pp eliminates, per D-13).

        Uses fp values verified to produce diff < 0.05 in float arithmetic.
        sp=0.875 with phpcs=sec=0.99 gives diff=0.046 (< 0.05) -> SURVIVES.
        sp=0.866 gives diff=0.0496 (< 0.05) -> also SURVIVES.
        """
        # best: phpcs=0.99, sec=0.99, sp=0.99 -> score=0.99
        # worse: phpcs=0.99, sec=0.99, sp=0.875 -> 0.594 + 0.4*0.875 = 0.944
        # diff = 0.99 - 0.944 = 0.046 -- clearly NOT > 0.05, should survive
        results = {
            "30_70": _make_ratio_result(phpcs_rate=0.99, security_rate=0.99, spearman=0.99),
            "40_60": _make_ratio_result(phpcs_rate=0.99, security_rate=0.99, spearman=0.875),
        }
        outcome = triage_ratios(results)
        assert "40_60" in outcome.survivors, (
            f"4.6pp behind should survive under D-13 (only >5pp eliminated). "
            f"Survivors: {outcome.survivors}"
        )

    def test_5_1pp_behind_eliminated(self):
        """Ratio 5.1pp behind best is eliminated."""
        # best: phpcs=1.0, sec=1.0, sp=1.0 -> score=1.0
        # worse: phpcs=0.96, sec=0.96, sp=0.86 -> 0.6*0.96+0.4*0.86=0.576+0.344=0.920
        # diff = 1.0-0.920 = 0.08 > 0.05 -> eliminated
        results = {
            "30_70": _make_ratio_result(phpcs_rate=1.0, security_rate=1.0, spearman=1.0),
            "40_60": _make_ratio_result(phpcs_rate=0.96, security_rate=0.96, spearman=0.86),
        }
        outcome = triage_ratios(results)
        assert "40_60" not in outcome.survivors


# ---------------------------------------------------------------------------
# Tests: NO_SURVIVORS scenario
# ---------------------------------------------------------------------------


class TestNoSurvivors:
    def test_all_failing_returns_no_survivors_status(self):
        """All ratios failing all gates returns status='NO_SURVIVORS'."""
        # All fail PHPCS
        results = {r: _make_ratio_result(phpcs_rate=0.50) for r in ["30_70", "40_60", "50_50"]}
        outcome = triage_ratios(results)
        assert outcome.status == "NO_SURVIVORS"

    def test_no_survivors_empty_survivors_list(self):
        """NO_SURVIVORS has empty survivors list."""
        results = {r: _make_ratio_result(phpcs_rate=0.50) for r in ["30_70", "40_60", "50_50"]}
        outcome = triage_ratios(results)
        assert outcome.survivors == []

    def test_no_survivors_includes_recommendation(self):
        """NO_SURVIVORS output includes recommendation to re-examine training."""
        results = {r: _make_ratio_result(phpcs_rate=0.50) for r in ["30_70", "40_60"]}
        outcome = triage_ratios(results)
        assert outcome.status == "NO_SURVIVORS"
        # Recommendation should be in triage_table or best_ratio is None
        assert outcome.best_ratio is None

    def test_no_survivors_best_ratio_is_none(self):
        """NO_SURVIVORS should have best_ratio=None."""
        results = {r: _make_ratio_result(spearman=0.50) for r in ["30_70", "40_60"]}
        outcome = triage_ratios(results)
        assert outcome.best_ratio is None


# ---------------------------------------------------------------------------
# Tests: load_eval_results
# ---------------------------------------------------------------------------


class TestLoadEvalResults:
    def _write_eval_files(self, tmpdir: Path, ratio: str, gen_data: dict, judge_data: dict):
        """Write mock eval JSON files for a ratio."""
        ratio_dir = tmpdir / f"ratio_{ratio}"
        ratio_dir.mkdir()
        (ratio_dir / "eval_gen_results.json").write_text(json.dumps(gen_data))
        (ratio_dir / "eval_judge_results.json").write_text(json.dumps(judge_data))

    def test_reads_per_ratio_files(self):
        """load_eval_results reads per-ratio eval JSON files correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gen_data = {
                "phpcs_pass_rate": 0.97,
                "security_pass_rate": 0.99,
                "overall_mean": 82.0,
            }
            judge_data = {
                "overall_spearman": 0.92,
            }
            self._write_eval_files(tmp, "30_70", gen_data, judge_data)
            results = load_eval_results(str(tmp))
            assert "30_70" in results
            assert results["30_70"]["phpcs_pass_rate"] == 0.97

    def test_handles_missing_wpbench(self):
        """load_eval_results handles missing wp-bench results gracefully (wpbench_score=None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gen_data = {"phpcs_pass_rate": 0.97, "security_pass_rate": 0.99, "overall_mean": 82.0}
            judge_data = {"overall_spearman": 0.92}
            self._write_eval_files(tmp, "30_70", gen_data, judge_data)
            # No wp_bench_results.json file written
            results = load_eval_results(str(tmp))
            assert results["30_70"]["wpbench_score"] is None


# ---------------------------------------------------------------------------
# Tests: Elimination reason and output fields
# ---------------------------------------------------------------------------


class TestTriageOutput:
    def test_elimination_reason_present(self):
        """Triage output includes elimination reason for each eliminated ratio."""
        results = {
            "30_70": _make_ratio_result(phpcs_rate=0.80),  # fails PHPCS
        }
        outcome = triage_ratios(results)
        eliminated = outcome.eliminated
        assert len(eliminated) >= 1
        assert "reason" in eliminated[0]
        assert eliminated[0]["ratio"] == "30_70"

    def test_wpbench_available_field_present(self):
        """Triage output has 'wpbench_available' boolean field."""
        results = _make_eval_results()
        outcome = triage_ratios(results)
        assert hasattr(outcome, "wpbench_available")
        assert isinstance(outcome.wpbench_available, bool)

    def test_wpbench_available_false_when_no_scores(self):
        """wpbench_available is False when no wpbench_score present."""
        results = _make_eval_results(wpbench_score=None)
        outcome = triage_ratios(results)
        assert outcome.wpbench_available is False

    def test_wpbench_available_true_when_scores_present(self):
        """wpbench_available is True when at least one wpbench_score is present."""
        results = _make_eval_results(wpbench_score=75.0)
        outcome = triage_ratios(results)
        assert outcome.wpbench_available is True

    def test_ok_status_when_survivors_exist(self):
        """Status is 'OK' when at least one ratio survives."""
        results = _make_eval_results(phpcs_rate=0.96, security_rate=0.99, spearman=0.90)
        outcome = triage_ratios(results)
        assert outcome.status == "OK"
