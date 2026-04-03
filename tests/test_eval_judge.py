"""Tests for eval/eval_judge.py — updated to match current API surface.

All tests are pure unit tests — no GPU, no vLLM, no external services.
Tests exercise parse_judge_response and _safe_spearman against the
current implementation (rubric refactor, April 2026).

Removed: test_score_inversion (tested nonexistent invert_phpcs_errors).
"""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_judge import parse_judge_response, _safe_spearman


# ---------------------------------------------------------------------------
# test_spearman_computation
# ---------------------------------------------------------------------------

def test_spearman_computation():
    """Assert scipy.stats.spearmanr returns expected correlation.

    Uses simple hand-computed cases:
      - Perfect positive correlation: model_scores == phpcs_scores → 1.0
      - Perfect inverse correlation: model_scores == reversed phpcs_scores → -1.0
    """
    from scipy.stats import spearmanr

    # Perfect positive correlation
    model_scores = [1, 2, 3, 4, 5]
    phpcs_scores = [10, 20, 30, 40, 50]
    result = spearmanr(model_scores, phpcs_scores)
    assert abs(result.statistic - 1.0) < 1e-9, f"Expected 1.0 but got {result.statistic}"

    # Perfect inverse (negative) correlation
    model_scores_inv = [5, 4, 3, 2, 1]
    phpcs_scores_inv = [10, 20, 30, 40, 50]
    result_inv = spearmanr(model_scores_inv, phpcs_scores_inv)
    assert abs(result_inv.statistic - (-1.0)) < 1e-9, f"Expected -1.0 but got {result_inv.statistic}"

    # Moderate correlation: no strict value but within range
    a = [1, 2, 3, 4, 5]
    b = [1, 3, 2, 5, 4]
    result_mod = spearmanr(a, b)
    assert -1.0 <= result_mod.statistic <= 1.0


# ---------------------------------------------------------------------------
# test_judge_output_parsing
# ---------------------------------------------------------------------------

def test_judge_output_parsing():
    """Assert parser extracts 'overall_score' from model response JSON.

    Tests:
      - Well-formed JSON with overall_score → correct value
      - JSON in fenced code block → correct value
      - Malformed response → returns None
      - Response missing overall_score → parse succeeds but key absent
    """
    # Valid response with overall_score
    valid_response = json.dumps({
        "overall_score": 87,
        "wpcs_compliance": 90,
        "security_score": 85,
        "performance_score": 80,
        "verdict": "PASS",
        "must_fix_issues": [],
        "notes": "Good code",
    })
    parsed = parse_judge_response(valid_response)
    assert parsed is not None
    assert parsed["overall_score"] == 87

    # JSON embedded in markdown code fence
    fenced_response = f"```json\n{valid_response}\n```"
    parsed_fenced = parse_judge_response(fenced_response)
    assert parsed_fenced is not None
    assert parsed_fenced["overall_score"] == 87

    # Malformed / non-JSON response → should return None
    malformed = "I cannot evaluate this code."
    result = parse_judge_response(malformed)
    assert result is None

    # Response with missing overall_score key → parse succeeds, key absent
    no_score_response = json.dumps({"verdict": "PASS", "notes": "looks good"})
    parsed_no_score = parse_judge_response(no_score_response)
    assert parsed_no_score is not None
    assert "overall_score" not in parsed_no_score


# ---------------------------------------------------------------------------
# test_safe_spearman_edge_cases
# ---------------------------------------------------------------------------

def test_safe_spearman_edge_cases():
    """_safe_spearman handles degenerate inputs gracefully."""
    # Fewer than 2 items → corr=0.0, p_value=1.0
    result_empty = _safe_spearman([], [])
    assert result_empty["corr"] == 0.0
    assert result_empty["p_value"] == 1.0
    assert result_empty["n_pairs"] == 0

    result_single = _safe_spearman([5.0], [3.0])
    assert result_single["corr"] == 0.0
    assert result_single["p_value"] == 1.0
    assert result_single["n_pairs"] == 1

    # All-identical values in xs → corr=0.0
    result_identical_x = _safe_spearman([7.0, 7.0, 7.0], [1.0, 2.0, 3.0])
    assert result_identical_x["corr"] == 0.0
    assert result_identical_x["p_value"] == 1.0

    # All-identical values in ys → corr=0.0
    result_identical_y = _safe_spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert result_identical_y["corr"] == 0.0

    # Valid pairs → dict with required keys
    result_valid = _safe_spearman([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
    assert "corr" in result_valid
    assert "p_value" in result_valid
    assert "n_pairs" in result_valid
    assert result_valid["n_pairs"] == 5
    # Perfect positive correlation
    assert abs(result_valid["corr"] - 1.0) < 1e-6
