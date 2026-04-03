"""Tests for eval/eval_gate.py — updated to match current API surface.

All tests use fixtures/helpers — no GPU, no model, no external deps needed.

Breaking changes from Wave-0 tests:
  - check_gates() returns (bool, list[dict]) not (bool, list[str])
  - Thresholds dict must include ALL _FALLBACK_THRESHOLDS keys
  - Results dict must include all keys that check_gates reads
"""
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_gate import check_gates, load_thresholds, _FALLBACK_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers: build full passing results/thresholds dicts
# ---------------------------------------------------------------------------

def _full_thresholds(overrides: dict = None) -> dict:
    """Return a complete thresholds dict, optionally overriding specific values."""
    t = dict(_FALLBACK_THRESHOLDS)
    if overrides:
        t.update(overrides)
    return t


def _full_results(overrides: dict = None) -> dict:
    """Return a full passing results dict, optionally overriding specific values."""
    r = {
        # Overall gen mean (gate: overall_mean_score)
        "overall_mean": 80.0,
        # Per-dimension gen pass rates (empty means no dim gates to check)
        "gen_dimension_pass_rates": {},
        # Overall Spearman (gate: overall_spearman)
        "overall_spearman": 0.90,
        # Per-dimension judge correlations (empty means no dim gates to check)
        "judge_dimension_correlations": {},
        # Legacy gates
        "phpcs_pass_rate": 0.97,
        "spearman_corr": 0.88,
        "security_pass_rate": 0.99,
    }
    if overrides:
        r.update(overrides)
    return r


# ---------------------------------------------------------------------------
# test_gate_pass
# ---------------------------------------------------------------------------

def test_gate_pass():
    """All metrics above all thresholds — gate returns passed=True."""
    results = _full_results()
    thresholds = _full_thresholds()

    passed, gate_rows = check_gates(results, thresholds)

    assert passed is True
    assert isinstance(gate_rows, list)
    assert len(gate_rows) > 0
    assert all(row["passed"] for row in gate_rows)
    # Each row has required keys
    for row in gate_rows:
        assert "gate" in row
        assert "target" in row
        assert "actual" in row
        assert "passed" in row


# ---------------------------------------------------------------------------
# test_gate_fail_phpcs
# ---------------------------------------------------------------------------

def test_gate_fail_phpcs():
    """phpcs_pass_rate below target — gate returns passed=False."""
    results = _full_results(overrides={"phpcs_pass_rate": 0.80})  # below default 0.95
    thresholds = _full_thresholds()

    passed, gate_rows = check_gates(results, thresholds)

    assert passed is False
    # At least one gate_row with gate name containing "phpcs" must be failed
    phpcs_rows = [r for r in gate_rows if "phpcs" in r["gate"].lower()]
    assert len(phpcs_rows) >= 1
    assert any(not r["passed"] for r in phpcs_rows)


# ---------------------------------------------------------------------------
# test_gate_fail_spearman
# ---------------------------------------------------------------------------

def test_gate_fail_spearman():
    """spearman_corr below target — gate returns passed=False."""
    results = _full_results(overrides={"spearman_corr": 0.70})  # below default 0.85
    thresholds = _full_thresholds()

    passed, gate_rows = check_gates(results, thresholds)

    assert passed is False
    # At least one gate_row with gate name containing "spearman" must be failed
    spearman_rows = [r for r in gate_rows if "spearman" in r["gate"].lower()]
    assert len(spearman_rows) >= 1
    assert any(not r["passed"] for r in spearman_rows)


# ---------------------------------------------------------------------------
# test_gate_fail_security
# ---------------------------------------------------------------------------

def test_gate_fail_security():
    """security_pass_rate below target — gate returns passed=False."""
    results = _full_results(overrides={"security_pass_rate": 0.90})  # below default 0.98
    thresholds = _full_thresholds()

    passed, gate_rows = check_gates(results, thresholds)

    assert passed is False
    # At least one gate_row with gate name containing "security" must be failed
    security_rows = [r for r in gate_rows if "security" in r["gate"].lower()]
    assert len(security_rows) >= 1
    assert any(not r["passed"] for r in security_rows)


# ---------------------------------------------------------------------------
# test_gate_reads_thresholds_from_config
# ---------------------------------------------------------------------------

def test_gate_reads_thresholds_from_config():
    """Gate reads targets from config YAML eval section, not hardcoded defaults."""
    # Write a temporary train_config.yaml with custom (lower) eval thresholds
    config_content = """
training:
  epochs: 3
  lr: 2e-4

eval:
  phpcs_pass_target: 0.91
  spearman_target: 0.80
  security_pass_target: 0.92
  overall_mean_target: 70.0
  overall_spearman_target: 0.75
  gen_dimension_targets: {}
  judge_dimension_targets: {}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        tmp_path = f.name

    try:
        thresholds = load_thresholds(config_path=tmp_path)
        assert thresholds["phpcs_pass_target"] == 0.91
        assert thresholds["spearman_target"] == 0.80
        assert thresholds["security_pass_target"] == 0.92
        assert thresholds["overall_mean_target"] == 70.0
        assert thresholds["overall_spearman_target"] == 0.75
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Verify gate uses loaded thresholds:
    # These values would FAIL under default thresholds (0.95, 0.85, 0.98, 75.0, 0.80)
    # but should PASS under the custom thresholds (0.91, 0.80, 0.92, 70.0, 0.75)
    results_that_fail_defaults_but_pass_custom = _full_results(overrides={
        "phpcs_pass_rate": 0.92,     # fails default 0.95, passes custom 0.91
        "spearman_corr": 0.82,       # fails default 0.85, passes custom 0.80
        "security_pass_rate": 0.93,  # fails default 0.98, passes custom 0.92
        "overall_mean": 72.0,        # fails default 75.0, passes custom 70.0
        "overall_spearman": 0.77,    # fails default 0.80, passes custom 0.75
    })
    passed, gate_rows = check_gates(results_that_fail_defaults_but_pass_custom, thresholds)
    assert passed is True, (
        f"Expected gate to pass with custom thresholds but got failures: "
        f"{[r for r in gate_rows if not r['passed']]}"
    )
