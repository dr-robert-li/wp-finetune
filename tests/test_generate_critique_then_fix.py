"""Wave 0 test stubs for scripts/generate_critique_then_fix.py.

Tests FAIL with ImportError until the production script exists (RED phase).
Once scripts/generate_critique_then_fix.py is created, these tests will
pass for all non-API functions.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_critique_then_fix import (
    load_critique_seeds,
    passes_quality_gate,
    validate_pilot_batch,
    format_training_example,
    php_lint_check,
    check_critique_fix_alignment,
    REQUIRED_DIMENSIONS,
    SEVERITY_LEVELS,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_good_ctf_result():
    """Return a complete valid critique-then-fix result with all required fields."""
    return {
        "verdict": "FAIL",
        "overall_score": 35,
        "key_observation": "Multiple security vulnerabilities found.",
        "dimension_analysis": {
            d: {
                "score": 4,
                "severity": "high",
                "analysis": f"Issue found in {d}: missing sanitization.",
            }
            for d in REQUIRED_DIMENSIONS
        },
        "corrected_code": "<?php function fixed_foo() { return esc_html(sanitize_text_field($_POST['x'])); } ?>",
    }


# ---------------------------------------------------------------------------
# Task 1: load_critique_seeds
# ---------------------------------------------------------------------------


def test_load_critique_seeds_returns_ctf_only():
    """load_critique_seeds() must return only seed_type == 'critique_then_fix', count >= 59."""
    seeds = load_critique_seeds()
    assert all(s["seed_type"] == "critique_then_fix" for s in seeds), (
        "load_critique_seeds() returned non-critique_then_fix seeds"
    )
    assert len(seeds) >= 59, f"Expected >= 59 critique_then_fix seeds, got {len(seeds)}"


# ---------------------------------------------------------------------------
# Task 2–4: passes_quality_gate
# ---------------------------------------------------------------------------


def test_passes_quality_gate_valid():
    """A complete result with all 9 dimensions, valid severity, and corrected_code passes."""
    result = _make_good_ctf_result()
    assert passes_quality_gate(result) is True


def test_passes_quality_gate_missing_corrected_code():
    """A result with empty corrected_code fails the quality gate."""
    result = _make_good_ctf_result()
    result["corrected_code"] = ""
    assert passes_quality_gate(result) is False


def test_passes_quality_gate_bad_severity():
    """A result with severity='extreme' (not in SEVERITY_LEVELS) fails the gate."""
    result = _make_good_ctf_result()
    result["dimension_analysis"]["security"]["severity"] = "extreme"
    assert passes_quality_gate(result) is False


# ---------------------------------------------------------------------------
# Task 5–6: validate_pilot_batch
# ---------------------------------------------------------------------------


def test_pilot_validation_severity_coverage():
    """validate_pilot_batch() confirms all 4 severity levels appear in pilot examples."""
    # Build 20 examples cycling through all severity levels
    severities = list(SEVERITY_LEVELS)
    pilot_examples = []
    for i in range(20):
        severity = severities[i % len(severities)]
        example = {
            "critique": f"Issue found: severity is {severity}",
            "corrected_code": "<?php function ok() { return esc_html($x); } ?>",
            "result": {
                "verdict": "FAIL",
                "overall_score": 40,
                "key_observation": "ok",
                "corrected_code": "<?php function ok() { return esc_html($x); } ?>",
                "dimension_analysis": {
                    d: {"score": 4, "severity": severity, "analysis": "Issue found."}
                    for d in REQUIRED_DIMENSIONS
                },
            },
        }
        pilot_examples.append(example)
    report = validate_pilot_batch(pilot_examples)
    assert report["missing_severities"] == [], (
        f"Expected all severities covered, missing: {report['missing_severities']}"
    )


def test_pilot_validation_dimension_coverage():
    """validate_pilot_batch() reports 0 missing dimensions for complete pilot batch."""
    pilot_examples = [
        {
            "critique": "Thorough critique of all dimensions.",
            "corrected_code": "<?php function ok() { return esc_html($x); } ?>",
            "result": {
                "verdict": "FAIL",
                "overall_score": 40,
                "key_observation": "Issues found.",
                "corrected_code": "<?php function ok() { return esc_html($x); } ?>",
                "dimension_analysis": {
                    d: {"score": 4, "severity": "high", "analysis": f"Issue in {d}."}
                    for d in REQUIRED_DIMENSIONS
                },
            },
        }
        for _ in range(20)
    ]
    report = validate_pilot_batch(pilot_examples)
    assert report["missing_dimensions"] == [], (
        f"Expected 0 missing dimensions, got {report['missing_dimensions']}"
    )


# ---------------------------------------------------------------------------
# Task 7: format_training_example schema
# ---------------------------------------------------------------------------


def test_format_training_example_schema():
    """format_training_example() output has all required keys."""
    source_info = {
        "source_file": "wp-content/plugins/test/bad.php",
        "function_name": "bad_function",
        "code": "<?php function bad_function() { echo $_POST['x']; } ?>",
    }
    result = _make_good_ctf_result()
    critique = "Missing sanitization and escaping."
    example = format_training_example(source_info, result, critique)
    required_keys = [
        "source_file", "function_name", "defective_code",
        "critique", "corrected_code", "dimensions_addressed", "generation_method",
    ]
    for key in required_keys:
        assert key in example, f"Missing required key: {key}"


# ---------------------------------------------------------------------------
# Task 8: corrected_code XML tag fallback extraction (stub)
# ---------------------------------------------------------------------------


def test_corrected_code_fallback_extraction():
    """Stub: verify <corrected_code> XML tag regex fallback exists in production code.

    This test verifies that the production script exports an extraction helper
    that can find corrected code from XML-tagged LLM responses as a fallback
    when JSON parsing fails.
    """
    # Import the extraction helper to confirm it is exported
    from scripts.generate_critique_then_fix import extract_corrected_code_from_xml
    # Test the extraction with a valid XML-tagged response
    llm_response = (
        "Here is the corrected version:\n"
        "<corrected_code>\n"
        "<?php function foo() { return esc_html($x); } ?>\n"
        "</corrected_code>\n"
        "This is much more secure."
    )
    extracted = extract_corrected_code_from_xml(llm_response)
    assert extracted is not None, "Expected to extract code from XML tag"
    assert "esc_html" in extracted


# ---------------------------------------------------------------------------
# Task 9–10: PHP lint checks (addresses review concern #2)
# ---------------------------------------------------------------------------


def test_php_lint_rejects_syntax_errors():
    """php_lint_check() returns False for PHP code with syntax errors.

    Test: missing semicolon causes a parse error.
    """
    # Missing semicolon after 'return true'
    invalid_php = "<?php function foo() { return true }"
    result = php_lint_check(invalid_php)
    assert result is False, (
        f"Expected php_lint_check to return False for invalid PHP, got {result}"
    )


def test_php_lint_accepts_valid_php():
    """php_lint_check() returns True for syntactically valid PHP code."""
    valid_php = "<?php function foo() { return true; }"
    result = php_lint_check(valid_php)
    assert result is True, (
        f"Expected php_lint_check to return True for valid PHP, got {result}"
    )


# ---------------------------------------------------------------------------
# Task 11: Critique-fix alignment check (addresses review concern #2)
# ---------------------------------------------------------------------------


def test_critique_fix_alignment_check():
    """check_critique_fix_alignment() flags misalignment between critique and corrected code.

    Scenario: critique identifies 'missing $wpdb->prepare' as a critical issue,
    but the corrected_code does NOT contain '$wpdb->prepare'.
    The function must flag this as a misalignment.
    """
    critique = (
        "Critical issue: The query uses raw user input without $wpdb->prepare, "
        "leaving the code vulnerable to SQL injection. "
        "The fix MUST add $wpdb->prepare() around all database queries."
    )
    # corrected_code does not use $wpdb->prepare
    corrected_code = (
        "<?php function safe_query($user_id) {\n"
        "    global $wpdb;\n"
        "    $results = $wpdb->get_results('SELECT * FROM wp_posts WHERE ID=' . intval($user_id));\n"
        "    return $results;\n"
        "} ?>"
    )
    alignment_result = check_critique_fix_alignment(critique, corrected_code)
    assert alignment_result["is_aligned"] is False, (
        f"Expected misalignment detected, got: {alignment_result}"
    )
    assert len(alignment_result["mismatches"]) > 0, (
        f"Expected at least one mismatch, got: {alignment_result['mismatches']}"
    )
