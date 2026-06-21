"""Tests for eval/eval_judge.py — updated to match current API surface.

All tests are pure unit tests — no GPU, no vLLM, no external services.
Tests exercise parse_judge_response, _safe_spearman, and
_extract_gt_from_assistant against the current implementation
(GT source fix, April 2026).

Removed: test_score_inversion (tested nonexistent invert_phpcs_errors).
"""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_judge import _extract_gt_from_assistant, _safe_spearman, parse_judge_response


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

def test_extract_gt_from_assistant():
    """_extract_gt_from_assistant extracts GT scores from the assistant response.

    Tests:
      - Well-formed assistant response with all fields -> correct overall and dim scores
      - Assistant response missing overall_score -> returns None
      - Non-JSON assistant response -> returns None
      - No assistant message -> returns None
      - dimension_scores only contains fields in _GT_FIELD_TO_DIM (not documentation_score)
    """
    gt_response = {
        "overall_score": 45,
        "wpcs_compliance": 55,
        "security_score": 10,
        "performance_score": 80,
        "i18n_score": 55,
        "accessibility_score": 65,
        "documentation_score": 50,  # should be ignored (no dim_key)
        "must_fix_issues": [],
    }

    messages_with_gt = [
        {"role": "user", "content": "<wp_judge> Evaluate this WordPress code:\n<?php echo 'hi'; ?>"},
        {"role": "assistant", "content": json.dumps(gt_response)},
    ]

    result = _extract_gt_from_assistant(messages_with_gt)
    assert result is not None
    assert result["overall"] == 45.0
    assert result["dimension_scores"]["D1_wpcs"] == 55.0
    assert result["dimension_scores"]["D2_security"] == 10.0
    assert result["dimension_scores"]["D4_perf"] == 80.0
    assert result["dimension_scores"]["D6_i18n"] == 55.0
    assert result["dimension_scores"]["D7_a11y"] == 65.0
    # documentation_score has no dim_key — must not appear
    assert "documentation_score" not in result["dimension_scores"]
    # Dimensions not in GT fields must not appear
    for absent_dim in ("D3_sql", "D5_wp_api", "D8_errors", "D9_structure"):
        assert absent_dim not in result["dimension_scores"]

    # Missing overall_score -> None
    no_overall = {"wpcs_compliance": 80}
    messages_no_overall = [
        {"role": "user", "content": "code"},
        {"role": "assistant", "content": json.dumps(no_overall)},
    ]
    assert _extract_gt_from_assistant(messages_no_overall) is None

    # Non-JSON assistant response -> None
    messages_non_json = [
        {"role": "user", "content": "code"},
        {"role": "assistant", "content": "I cannot evaluate this."},
    ]
    assert _extract_gt_from_assistant(messages_non_json) is None

    # No assistant message -> None
    messages_no_assistant = [
        {"role": "user", "content": "code"},
    ]
    assert _extract_gt_from_assistant(messages_no_assistant) is None

    # JSON in markdown code fence -> should be parsed (uses parse_judge_response)
    fenced_content = "```json\n" + json.dumps(gt_response) + "\n```"
    messages_fenced = [
        {"role": "user", "content": "code"},
        {"role": "assistant", "content": fenced_content},
    ]
    result_fenced = _extract_gt_from_assistant(messages_fenced)
    assert result_fenced is not None
    assert result_fenced["overall"] == 45.0


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


# ---------------------------------------------------------------------------
# RL reward-path judge parse fixes (live Phase 9 GSPO run, 2026-06-22):
#  Bug A — <judge_output> tags + [REASONING] prose with literal braces poisoned
#          the greedy {.*} scan -> None -> group-mean imputation.
#  Bug B — bimodal judge omits overall_score -> .get(...) None -> imputation.
# These are the EXACT shapes captured live from the served wp_judge endpoint.
# ---------------------------------------------------------------------------

# Real shape: [REASONING] prose quoting code with `{$wpdb->prefix}` braces, then a
# <judge_output> block WITHOUT overall_score, verdict FAIL. Was 66.7% of failures.
_REAL_JUDGE_NO_OVERALL = (
    "myplugin_handle_form concatenates $_GET into SQL via "
    "\"UPDATE {$wpdb->prefix}users SET x=\".$id and echoes $_POST unescaped.\n\n"
    "[/REASONING]\n\n"
    "<judge_output>\n"
    "{\n"
    '  "wpcs_compliance": 6,\n  "sql_safety": 2,\n  "security": 2,\n'
    '  "performance": 6,\n  "wp_api_usage": 6,\n  "code_quality": 6,\n'
    '  "dependency_integrity": 8,\n  "i18n": 8,\n  "accessibility": 8,\n'
    '  "verdict": "FAIL"\n'
    "}\n"
    "</judge_output>"
)

# Real shape: judge emitted its OWN overall_score — must be used verbatim.
_REAL_JUDGE_WITH_OVERALL = (
    "[REASONING] uses {$wpdb} unsafely [/REASONING]\n"
    "<judge_output>\n"
    '{ "wpcs_compliance": 4, "sql_safety": 1, "security": 1, "overall_score": 19, '
    '"verdict": "FAIL" }\n'
    "</judge_output>"
)


def _fake_judge_client(content: str):
    """Minimal stand-in for openai.OpenAI whose chat.completions.create returns
    `content`. judge_score_single routes through _judge_create -> this."""
    from types import SimpleNamespace

    def _create(*_a, **_k):
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


def test_parse_judge_response_judge_output_tag_brace_poison():
    """Bug A: <judge_output> block is recovered despite brace-laden [REASONING]
    prose that poisons the greedy {.*} fallback. Pre-fix this returned None."""
    parsed = parse_judge_response(_REAL_JUDGE_NO_OVERALL)
    assert parsed is not None, "tag-bounded JSON must parse despite prose braces"
    assert parsed["verdict"] == "FAIL"
    assert parsed["security"] == 2


def test_parse_judge_response_stays_pure_no_overall_injection():
    """GT-purity: the PARSER must never synthesize overall_score (teacher-GT
    extraction relies on its absence -> canonical rubric_scorer, not a proxy)."""
    parsed = parse_judge_response(_REAL_JUDGE_NO_OVERALL)
    assert "overall_score" not in parsed
    assert "_overall_derived" not in parsed


def test_judge_score_single_derives_missing_overall_fail_capped():
    """Bug B: missing overall_score -> derived (not None), weighted over canonical
    DIMENSION_WEIGHTS, FAIL-capped below the PASS threshold (70)."""
    from eval.eval_judge import judge_score_single

    score = judge_score_single(
        "<?php bad();", _fake_judge_client(_REAL_JUDGE_NO_OVERALL), "wp_judge"
    )
    assert score is not None, "must derive, not impute"
    assert 0.0 < score < 70.0, f"FAIL verdict must derive sub-PASS score, got {score}"


def test_judge_score_single_prefers_judge_own_overall():
    """When the judge emits overall_score, use it verbatim — never derive over it."""
    from eval.eval_judge import judge_score_single

    score = judge_score_single(
        "<?php x();", _fake_judge_client(_REAL_JUDGE_WITH_OVERALL), "wp_judge"
    )
    assert score == 19.0


def test_derive_overall_weighting_and_fallback():
    """_derive_overall_from_dims: weighted mean x10 over mapped dims; FAIL cap;
    plain-mean fallback when nothing maps; None when no numeric dims."""
    from eval.eval_judge import _derive_overall_from_dims

    # All dims = 8/10, no verdict -> 80.0 (weights renormalize to 1.0).
    alleight = {k: 8 for k in (
        "wpcs_compliance", "sql_safety", "security", "performance",
        "wp_api_usage", "i18n", "accessibility")}
    assert abs(_derive_overall_from_dims(alleight) - 80.0) < 1e-6
    # Same scores but FAIL -> capped below 70.
    assert _derive_overall_from_dims({**alleight, "verdict": "FAIL"}) < 70.0
    # No mapped dims, but a numeric field present -> plain mean x10 fallback.
    assert _derive_overall_from_dims({"made_up_dim": 5}) == 50.0
    # No numeric dims at all -> None.
    assert _derive_overall_from_dims({"verdict": "FAIL"}) is None
