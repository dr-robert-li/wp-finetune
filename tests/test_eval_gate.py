"""Tests for eval/eval_gate.py — Wave 0 (written before implementation).

All tests use mocks/fixtures — no GPU, no model, no external deps needed.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_gate import check_gates, load_thresholds


# ---------------------------------------------------------------------------
# Helper: default thresholds matching config/train_config.yaml targets
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "phpcs_pass_target": 0.95,
    "spearman_target": 0.85,
    "security_pass_target": 0.98,
}


# ---------------------------------------------------------------------------
# test_gate_pass
# ---------------------------------------------------------------------------

def test_gate_pass():
    """All metrics above thresholds → gate returns pass (would exit 0)."""
    results = {
        "phpcs_pass_rate": 0.96,
        "spearman_corr": 0.87,
        "security_pass_rate": 0.99,
    }
    passed, failures = check_gates(results, DEFAULT_THRESHOLDS)
    assert passed is True
    assert failures == []


# ---------------------------------------------------------------------------
# test_gate_fail_phpcs
# ---------------------------------------------------------------------------

def test_gate_fail_phpcs():
    """phpcs_pass_rate below 0.95 → gate returns fail."""
    results = {
        "phpcs_pass_rate": 0.90,  # below 0.95 target
        "spearman_corr": 0.87,
        "security_pass_rate": 0.99,
    }
    passed, failures = check_gates(results, DEFAULT_THRESHOLDS)
    assert passed is False
    assert len(failures) >= 1
    # At least one failure mentions phpcs
    assert any("phpcs" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# test_gate_fail_spearman
# ---------------------------------------------------------------------------

def test_gate_fail_spearman():
    """spearman_corr below 0.85 → gate returns fail."""
    results = {
        "phpcs_pass_rate": 0.96,
        "spearman_corr": 0.80,  # below 0.85 target
        "security_pass_rate": 0.99,
    }
    passed, failures = check_gates(results, DEFAULT_THRESHOLDS)
    assert passed is False
    assert len(failures) >= 1
    assert any("spearman" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# test_gate_fail_security
# ---------------------------------------------------------------------------

def test_gate_fail_security():
    """security_pass_rate below 0.98 → gate returns fail."""
    results = {
        "phpcs_pass_rate": 0.96,
        "spearman_corr": 0.87,
        "security_pass_rate": 0.96,  # below 0.98 target
    }
    passed, failures = check_gates(results, DEFAULT_THRESHOLDS)
    assert passed is False
    assert len(failures) >= 1
    assert any("security" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# test_gate_reads_thresholds_from_config
# ---------------------------------------------------------------------------

def test_gate_reads_thresholds_from_config():
    """Gate reads targets from config/train_config.yaml eval section, not hardcoded."""
    # Write a temporary train_config.yaml with custom eval thresholds
    config_content = """
training:
  epochs: 3
  lr: 2e-4

eval:
  phpcs_pass_target: 0.91
  spearman_target: 0.80
  security_pass_target: 0.92
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name

    try:
        thresholds = load_thresholds(config_path=tmp_path)
        assert thresholds["phpcs_pass_target"] == 0.91
        assert thresholds["spearman_target"] == 0.80
        assert thresholds["security_pass_target"] == 0.92
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Verify gate uses loaded thresholds, not defaults
    results_that_fail_default_but_pass_custom = {
        "phpcs_pass_rate": 0.92,   # fails default 0.95, passes custom 0.91
        "spearman_corr": 0.82,     # fails default 0.85, passes custom 0.80
        "security_pass_rate": 0.93,  # fails default 0.98, passes custom 0.92
    }
    passed, failures = check_gates(results_that_fail_default_but_pass_custom, thresholds)
    assert passed is True, f"Expected gate to pass with custom thresholds but got failures: {failures}"
