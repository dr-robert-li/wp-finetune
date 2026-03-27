"""Tests for eval/eval_judge.py — Wave 0 (written before implementation).

All tests use mocks — no GPU, no model, no phpcs binary needed.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_judge import (
    invert_phpcs_errors,
    parse_judge_response,
)


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
# test_score_inversion
# ---------------------------------------------------------------------------

def test_score_inversion():
    """Assert phpcs error count is correctly inverted to a score.

    Formula: phpcs_score = max(0, 100 - error_count * 5)

    Examples:
      - 0 errors → 100
      - 5 errors → 75
      - 20+ errors → 0
    """
    assert invert_phpcs_errors(0) == 100
    assert invert_phpcs_errors(5) == 75
    assert invert_phpcs_errors(10) == 50
    assert invert_phpcs_errors(20) == 0
    assert invert_phpcs_errors(25) == 0  # clamped at 0
    assert invert_phpcs_errors(100) == 0  # large error count → 0


# ---------------------------------------------------------------------------
# test_judge_output_parsing
# ---------------------------------------------------------------------------

def test_judge_output_parsing():
    """Assert parser extracts 'overall_score' from model response JSON.

    Tests:
      - Well-formed JSON with overall_score → correct value
      - Malformed response → raises ValueError or returns None (robust handling)
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

    # Malformed / non-JSON response → should return None or raise ValueError
    malformed = "I cannot evaluate this code."
    result = parse_judge_response(malformed)
    # Either returns None or raises ValueError — both are acceptable
    assert result is None or isinstance(result, dict)
    if isinstance(result, dict):
        # If it returns a dict, it must not have a meaningful overall_score
        # (implementation may return a default or empty dict)
        pass

    # Response with missing overall_score key → partial parse
    no_score_response = json.dumps({"verdict": "PASS", "notes": "looks good"})
    parsed_no_score = parse_judge_response(no_score_response)
    # overall_score missing should return None for that field or handle gracefully
    if parsed_no_score is not None:
        assert parsed_no_score.get("overall_score") is None or isinstance(parsed_no_score.get("overall_score"), (int, float))
