"""Tests for scripts/generate_critique_then_fix.py.

Tests non-API functions: quality gates, validation, formatting, PHP lint,
critique-fix alignment, and XML tag extraction.
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
    extract_corrected_code_from_xml,
    REQUIRED_DIMENSIONS,
    SEVERITY_LEVELS,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_good_ctf_result():
    """Return a complete valid critique-then-fix result with all required fields."""
    return {
        "summary": "Multiple issues found across dimensions.",
        "key_observation": "Multiple security vulnerabilities found.",
        "dimensions": {
            d: {
                "severity": "high",
                "issue": f"Issue found in {d}: missing sanitization.",
                "fix": f"Add proper sanitization for {d}.",
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
# Task 2-4: passes_quality_gate
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
    result["dimensions"]["security"]["severity"] = "extreme"
    assert passes_quality_gate(result) is False


# ---------------------------------------------------------------------------
# Task 5-6: validate_pilot_batch
# ---------------------------------------------------------------------------


def test_pilot_validation_dimension_coverage():
    """validate_pilot_batch() reports 0 missing dimensions for complete pilot batch."""
    pilot_examples = [
        {
            "critique": {
                "summary": "Issues found.",
                "dimensions": {
                    d: {"severity": "high", "issue": f"Issue in {d}.", "fix": "Fix it."}
                    for d in REQUIRED_DIMENSIONS
                },
                "key_observation": "Issues found.",
            },
            "corrected_code": "<?php function ok() { return esc_html($x); } ?>",
            "php_lint": {"valid": True, "errors": ""},
            "critique_fix_alignment": {"alignment_ratio": 1.0, "critical_high_issues": 0,
                                        "addressed_issues": 0, "unaddressed_issues": []},
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
    defective_code = source_info["code"]
    example = format_training_example(source_info, result, defective_code)
    required_keys = [
        "source_file", "function_name", "defective_code",
        "critique", "corrected_code", "dimensions_addressed", "generation_method",
    ]
    for key in required_keys:
        assert key in example, f"Missing required key: {key}"


# ---------------------------------------------------------------------------
# Task 8: corrected_code XML tag fallback extraction
# ---------------------------------------------------------------------------


def test_corrected_code_fallback_extraction():
    """extract_corrected_code_from_xml finds corrected code from XML-tagged responses."""
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


def test_corrected_code_fallback_none_when_no_tag():
    """extract_corrected_code_from_xml returns None when no XML tag present."""
    assert extract_corrected_code_from_xml("no tags here") is None


# ---------------------------------------------------------------------------
# Task 9-10: PHP lint checks
# ---------------------------------------------------------------------------


def test_php_lint_rejects_syntax_errors():
    """php_lint_check() returns valid=False for PHP code with syntax errors."""
    invalid_php = "<?php function foo() { return true }"
    result = php_lint_check(invalid_php)
    assert result["valid"] is False, (
        f"Expected php_lint_check to return valid=False for invalid PHP, got {result}"
    )


def test_php_lint_accepts_valid_php():
    """php_lint_check() returns valid=True for syntactically valid PHP code."""
    valid_php = "<?php function foo() { return true; }"
    result = php_lint_check(valid_php)
    assert result["valid"] is True, (
        f"Expected php_lint_check to return valid=True for valid PHP, got {result}"
    )


# ---------------------------------------------------------------------------
# Task 11: Critique-fix alignment check
# ---------------------------------------------------------------------------


def test_critique_fix_alignment_detects_missing_api():
    """check_critique_fix_alignment() flags when critique says add $wpdb->prepare but fix lacks it."""
    critique_dict = {
        "dimensions": {
            "sql_safety": {
                "severity": "critical",
                "issue": "Raw SQL without $wpdb->prepare",
                "fix": "Add $wpdb->prepare around all database queries",
            },
        },
    }
    defective_code = "<?php $wpdb->get_results('SELECT * FROM wp_posts WHERE ID=' . $id); ?>"
    corrected_code = "<?php $wpdb->get_results('SELECT * FROM wp_posts WHERE ID=' . intval($id)); ?>"
    result = check_critique_fix_alignment(critique_dict, defective_code, corrected_code)
    # $wpdb->prepare was cited in fix but not added to corrected code
    assert result["critical_high_issues"] >= 1
    # The unaddressed list should have the sql_safety dimension
    assert len(result["unaddressed_issues"]) >= 1


def test_critique_fix_alignment_accepts_grounded_fix():
    """check_critique_fix_alignment() accepts when fix APIs appear in corrected code."""
    critique_dict = {
        "dimensions": {
            "security": {
                "severity": "critical",
                "issue": "Missing nonce verification",
                "fix": "Add wp_verify_nonce check",
            },
        },
    }
    defective_code = "<?php function save() { update_option('key', $_POST['val']); } ?>"
    corrected_code = "<?php function save() { if (!wp_verify_nonce($_POST['_wpnonce'], 'save')) wp_die(); update_option('key', sanitize_text_field($_POST['val'])); } ?>"
    result = check_critique_fix_alignment(critique_dict, defective_code, corrected_code)
    assert result["alignment_ratio"] >= 0.5


# ---------------------------------------------------------------------------
# No Anthropic API references
# ---------------------------------------------------------------------------


def test_no_anthropic_import():
    """generate_critique_then_fix.py must not import anthropic."""
    import inspect
    import scripts.generate_critique_then_fix as ctf
    source = inspect.getsource(ctf)
    assert "import anthropic" not in source
