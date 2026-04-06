"""Tests for eval/eval_gen.py — updated to match current API surface.

All tests are pure unit tests — no GPU, no vLLM, no external services.
Tests exercise _extract_php_code and _compute_summary against the
rubric-scorer-based implementation (rubric refactor, April 2026).
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_gen import _extract_php_code, _compute_summary
from eval.rubric_scorer import RubricScore


# ---------------------------------------------------------------------------
# Helpers: build mock RubricScore objects without calling score_code()
# (avoids phpcs/phpstan subprocess calls in unit tests)
# ---------------------------------------------------------------------------

def _make_rubric_score(
    overall: float,
    grade: str = "Good",
    d2_security: float = 8.0,
) -> RubricScore:
    """Return a minimal RubricScore with controllable overall and D2_security."""
    dimension_scores = {
        "D1_wpcs": 8.0,
        "D2_security": d2_security,
        "D3_sql": 8.0,
        "D4_perf": 8.0,
        "D5_wp_api": 8.0,
        "D6_i18n": 8.0,
        "D7_a11y": 8.0,
        "D8_errors": 8.0,
        "D9_structure": 8.0,
    }
    return RubricScore(
        file_path="<test>",
        dimension_scores=dimension_scores,
        dimension_na=[],
        overall=overall,
        triggered_checks={},
        check_evidence={},
        grade=grade,
        floor_rules_applied=[],
        llm_checks_skipped=0,
    )


# ---------------------------------------------------------------------------
# test_extract_php_code
# ---------------------------------------------------------------------------

def test_extract_php_code():
    """_extract_php_code handles fenced blocks and raw PHP text."""
    # Fenced ```php block — most common model output format
    php_fenced = "```php\n<?php echo esc_html($val); ?>\n```"
    result = _extract_php_code(php_fenced)
    assert result == "<?php echo esc_html($val); ?>"

    # Generic ``` fenced block (no language hint)
    generic_fenced = "```\n<?php wp_enqueue_script('foo'); ?>\n```"
    result_generic = _extract_php_code(generic_fenced)
    assert result_generic == "<?php wp_enqueue_script('foo'); ?>"

    # Raw PHP text with no fences — returned as-is (stripped)
    raw_php = "<?php function hello() { return 'world'; } "
    result_raw = _extract_php_code(raw_php)
    assert result_raw == "<?php function hello() { return 'world'; }"

    # Prefer ```php over generic ``` when both present (first match wins)
    combined = "```php\n<?php // php block\n```\n```\n<?php // generic block\n```"
    result_combined = _extract_php_code(combined)
    assert "php block" in result_combined


# ---------------------------------------------------------------------------
# test_compute_summary_basic
# ---------------------------------------------------------------------------

def test_compute_summary_basic():
    """_compute_summary aggregates RubricScores into a summary dict."""
    # 5 scores: three above 80 (phpcs_pass proxy), two below
    scores = [
        _make_rubric_score(overall=85.0, grade="Good", d2_security=9.0),
        _make_rubric_score(overall=90.0, grade="Excellent", d2_security=9.5),
        _make_rubric_score(overall=82.0, grade="Good", d2_security=8.5),
        _make_rubric_score(overall=60.0, grade="Acceptable", d2_security=6.0),
        _make_rubric_score(overall=50.0, grade="Poor", d2_security=5.0),
    ]

    summary = _compute_summary(scores)

    # Required keys
    assert summary["total"] == 5
    assert "overall_mean" in summary
    assert "overall_median" in summary
    assert "grade_distribution" in summary
    assert "per_dimension" in summary
    assert "floor_rules" in summary
    assert "phpcs_pass_rate" in summary
    assert "security_pass_rate" in summary
    assert "n_applicable_dims_mean" in summary

    # overall_mean should be average of the 5 overall values
    expected_mean = (85.0 + 90.0 + 82.0 + 60.0 + 50.0) / 5
    assert abs(summary["overall_mean"] - round(expected_mean, 2)) < 1e-6

    # overall_median of [85, 90, 82, 60, 50] sorted = [50, 60, 82, 85, 90] → median = 82
    assert summary["overall_median"] == 82.0

    # phpcs_pass_rate: fraction with overall >= 80 → 3/5 = 0.6
    assert abs(summary["phpcs_pass_rate"] - 0.6) < 1e-6

    # security_pass_rate: fraction with D2_security >= 8 → scores [9.0, 9.5, 8.5] pass,
    # [6.0, 5.0] fail → 3/5 = 0.6
    assert abs(summary["security_pass_rate"] - 0.6) < 1e-6

    # per_dimension should have all 9 dimension keys
    assert "D1_wpcs" in summary["per_dimension"]
    assert "D2_security" in summary["per_dimension"]
    assert "D9_structure" in summary["per_dimension"]

    # per_dimension should have new transparency fields
    dim_info = summary["per_dimension"]["D1_wpcs"]
    assert "pass_rate_8_inclusive" in dim_info, "pass_rate_8_inclusive must be present"
    assert "na_rate" in dim_info, "na_rate must be present"
    assert "na_count" in dim_info

    # n_applicable_dims_mean: all 9 dimensions applicable (no N/A in _make_rubric_score)
    assert summary["n_applicable_dims_mean"] == 9.0


def test_compute_summary_na_transparency():
    """N/A dimensions are correctly reflected in pass_rate_8_inclusive and na_rate."""

    def _make_score_with_na(overall: float, na_dims: list) -> RubricScore:
        """Build a RubricScore where specified dimensions are None (N/A)."""
        dimension_scores = {
            "D1_wpcs": None if "D1_wpcs" in na_dims else 8.0,
            "D2_security": None if "D2_security" in na_dims else 8.0,
            "D3_sql": None if "D3_sql" in na_dims else 8.0,
            "D4_perf": None if "D4_perf" in na_dims else 8.0,
            "D5_wp_api": None if "D5_wp_api" in na_dims else 8.0,
            "D6_i18n": None if "D6_i18n" in na_dims else 8.0,
            "D7_a11y": None if "D7_a11y" in na_dims else 8.0,
            "D8_errors": None if "D8_errors" in na_dims else 8.0,
            "D9_structure": None if "D9_structure" in na_dims else 8.0,
        }
        return RubricScore(
            file_path="<test>",
            dimension_scores=dimension_scores,
            dimension_na=na_dims,
            overall=overall,
            triggered_checks={},
            check_evidence={},
            grade="Good",
            floor_rules_applied=[],
            llm_checks_skipped=0,
        )

    # 4 examples: D2_security is N/A for 3, applicable for 1 (scores >= 8)
    scores = [
        _make_score_with_na(overall=85.0, na_dims=["D2_security"]),
        _make_score_with_na(overall=82.0, na_dims=["D2_security"]),
        _make_score_with_na(overall=78.0, na_dims=["D2_security"]),
        _make_score_with_na(overall=90.0, na_dims=[]),  # D2 applicable, score=8.0
    ]

    summary = _compute_summary(scores)
    d2 = summary["per_dimension"]["D2_security"]

    # na_count = 3, na_rate = 3/4 = 0.75
    assert d2["na_count"] == 3
    assert abs(d2["na_rate"] - 0.75) < 1e-6

    # pass_rate_8: among applicable only (1 example with score=8.0) → 1/1 = 1.0
    assert abs(d2["pass_rate_8"] - 1.0) < 1e-6

    # pass_rate_8_inclusive: treats N/A as failing → 1 pass out of 4 total = 0.25
    assert abs(d2["pass_rate_8_inclusive"] - 0.25) < 1e-6

    # n_applicable_dims_mean: 3 examples have 8 dims, 1 example has 9 dims
    # mean = (8+8+8+9)/4 = 33/4 = 8.25
    assert abs(summary["n_applicable_dims_mean"] - 8.25) < 1e-6


def test_compute_summary_security_null_when_all_na():
    """security_pass_rate is None when all examples have D2_security=N/A."""

    def _make_score_no_security(overall: float) -> RubricScore:
        dimension_scores = {
            "D1_wpcs": 8.0,
            "D2_security": None,
            "D3_sql": 8.0,
            "D4_perf": 8.0,
            "D5_wp_api": 8.0,
            "D6_i18n": 8.0,
            "D7_a11y": 8.0,
            "D8_errors": 8.0,
            "D9_structure": 8.0,
        }
        return RubricScore(
            file_path="<test>",
            dimension_scores=dimension_scores,
            dimension_na=["D2_security"],
            overall=overall,
            triggered_checks={},
            check_evidence={},
            grade="Good",
            floor_rules_applied=[],
            llm_checks_skipped=0,
        )

    scores = [_make_score_no_security(80.0), _make_score_no_security(85.0)]
    summary = _compute_summary(scores)

    # security_pass_rate must be None (not 1.0) when no applicable examples
    assert summary["security_pass_rate"] is None, (
        f"Expected None when no security-applicable examples, got {summary['security_pass_rate']}"
    )


# ---------------------------------------------------------------------------
# test_compute_summary_empty
# ---------------------------------------------------------------------------

def test_compute_summary_empty():
    """_compute_summary returns minimal dict for empty input."""
    summary = _compute_summary([])
    assert summary == {"total": 0}
