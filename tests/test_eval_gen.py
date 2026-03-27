"""Tests for eval/eval_gen.py — Wave 0 (written before implementation).

All tests use mocks — no GPU, no model, no phpcs binary needed.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_gen import (
    run_phpcs,
    compute_pass_rate,
    classify_security,
)


# ---------------------------------------------------------------------------
# test_phpcs_eval_runs
# ---------------------------------------------------------------------------

def test_phpcs_eval_runs():
    """Mock subprocess.run for phpcs; assert eval counts pass/fail correctly.

    Provides 3 sample PHP code strings:
      - 1 clean (0 errors)  → pass
      - 1 with errors (3 errors) → fail
      - 1 with security issue (1 security error) → fail
    """
    clean_result = json.dumps({
        "totals": {"errors": 0, "warnings": 0, "fixable": 0},
        "files": {},
    })
    errors_result = json.dumps({
        "totals": {"errors": 3, "warnings": 1, "fixable": 0},
        "files": {
            "/tmp/code.php": {
                "errors": 3,
                "warnings": 1,
                "messages": [
                    {"message": "Space after opening brace", "source": "WordPress.WhiteSpace.ControlStructureSpacing", "type": "ERROR"},
                    {"message": "Expected capital letter", "source": "WordPress.NamingConventions.ValidFunctionName", "type": "ERROR"},
                    {"message": "Missing nonce check", "source": "WordPress.Security.NonceVerification", "type": "ERROR"},
                ],
            }
        },
    })
    security_result = json.dumps({
        "totals": {"errors": 1, "warnings": 0, "fixable": 0},
        "files": {
            "/tmp/code.php": {
                "errors": 1,
                "warnings": 0,
                "messages": [
                    {"message": "Missing nonce verification", "source": "WordPress.Security.NonceVerification", "type": "ERROR"},
                ],
            }
        },
    })

    phpcs_outputs = [clean_result, errors_result, security_result]

    def mock_phpcs_side_effect(*args, **kwargs):
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = phpcs_outputs.pop(0)
        mock.stderr = ""
        return mock

    sample_codes = [
        "<?php function clean_function() { return esc_html($val); }",
        "<?php function bad_function() { echo $val; }",
        "<?php function handler() { update_option('key', $_POST['val']); }",
    ]

    with patch("subprocess.run", side_effect=mock_phpcs_side_effect):
        results = [run_phpcs(code) for code in sample_codes]

    # clean code: 0 errors → pass
    assert results[0]["errors"] == 0
    assert results[0]["passed"] is True

    # code with errors: 3 errors → fail
    assert results[1]["errors"] == 3
    assert results[1]["passed"] is False

    # security issue: 1 error → fail
    assert results[2]["errors"] == 1
    assert results[2]["passed"] is False


# ---------------------------------------------------------------------------
# test_security_rate_detection
# ---------------------------------------------------------------------------

def test_security_rate_detection():
    """Assert security filter correctly classifies security sniff names.

    Tests:
      - PHP with missing nonce check → security fail
      - PHP with missing escaping → security fail
      - Clean code → security pass
    """
    # Missing nonce: WordPress.Security.NonceVerification sniff
    phpcs_nonce = {
        "totals": {"errors": 1},
        "files": {
            "/tmp/code.php": {
                "errors": 1,
                "messages": [{"source": "WordPress.Security.NonceVerification", "type": "ERROR"}],
            }
        },
    }

    # Missing escaping: WordPress.Security.EscapeOutput sniff
    phpcs_escape = {
        "totals": {"errors": 1},
        "files": {
            "/tmp/code.php": {
                "errors": 1,
                "messages": [{"source": "WordPress.Security.EscapeOutput", "type": "ERROR"}],
            }
        },
    }

    # Clean code: no security sniffs
    phpcs_clean = {
        "totals": {"errors": 0},
        "files": {},
    }

    # Missing nonce → security violation detected
    assert classify_security(phpcs_nonce) is True  # has security issue

    # Missing escaping → security violation detected
    assert classify_security(phpcs_escape) is True  # has security issue

    # Clean → no security violation
    assert classify_security(phpcs_clean) is False  # no security issue


# ---------------------------------------------------------------------------
# test_phpcs_pass_rate_calculation
# ---------------------------------------------------------------------------

def test_phpcs_pass_rate_calculation():
    """Assert pass_rate computation is correct given (code, phpcs_result) pairs.

    Example: 8/10 = 0.80
    """
    # Build 10 results: 8 passes, 2 failures
    results = []
    for i in range(8):
        results.append({
            "errors": 0,
            "passed": True,
            "phpcs_output": {"totals": {"errors": 0}},
        })
    for i in range(2):
        results.append({
            "errors": 3,
            "passed": False,
            "phpcs_output": {"totals": {"errors": 3}},
        })

    rate = compute_pass_rate(results)
    assert abs(rate - 0.80) < 1e-9, f"Expected 0.80 but got {rate}"

    # Edge case: all pass
    all_pass = [{"passed": True} for _ in range(5)]
    assert compute_pass_rate(all_pass) == 1.0

    # Edge case: none pass
    none_pass = [{"passed": False} for _ in range(5)]
    assert compute_pass_rate(none_pass) == 0.0

    # Edge case: empty list
    assert compute_pass_rate([]) == 0.0
