"""Tests for eval/eval_gate.py — updated to match current API surface.

All tests use fixtures/helpers — no GPU, no model, no external deps needed.

Breaking changes from Wave-0 tests:
  - check_gates() returns (bool, list[dict]) not (bool, list[str])
  - Thresholds dict must include ALL _FALLBACK_THRESHOLDS keys
  - Results dict must include all keys that check_gates reads
"""
import json
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


# ---------------------------------------------------------------------------
# test_per_dimension_gen_gates
# ---------------------------------------------------------------------------

def test_per_dimension_gen_gates():
    """Per-dimension gen pass rate gates correctly fail when below target."""
    thresholds = _full_thresholds(overrides={
        "gen_dimension_targets": {"D1_standards": 0.80, "D2_security": 0.90},
    })
    # D1 passes (0.85 >= 0.80), D2 fails (0.70 < 0.90)
    results = _full_results(overrides={
        "gen_dimension_pass_rates": {"D1_standards": 0.85, "D2_security": 0.70},
    })
    passed, gate_rows = check_gates(results, thresholds)
    assert passed is False
    d2_rows = [r for r in gate_rows if "D2_security" in r["gate"]]
    assert len(d2_rows) == 1
    assert d2_rows[0]["passed"] is False
    assert d2_rows[0]["actual"] == 0.70
    d1_rows = [r for r in gate_rows if "D1_standards" in r["gate"]]
    assert len(d1_rows) == 1
    assert d1_rows[0]["passed"] is True


# ---------------------------------------------------------------------------
# test_per_dimension_judge_gates
# ---------------------------------------------------------------------------

def test_per_dimension_judge_gates():
    """Per-dimension judge correlation gates correctly fail when below target."""
    thresholds = _full_thresholds(overrides={
        "judge_dimension_targets": {"D1_standards": 0.70, "D3_performance": 0.75},
    })
    # D1 passes (0.80 >= 0.70), D3 fails (0.60 < 0.75)
    results = _full_results(overrides={
        "judge_dimension_correlations": {"D1_standards": 0.80, "D3_performance": 0.60},
    })
    passed, gate_rows = check_gates(results, thresholds)
    assert passed is False
    d3_rows = [r for r in gate_rows if "D3_performance" in r["gate"]]
    assert len(d3_rows) == 1
    assert d3_rows[0]["passed"] is False


# ---------------------------------------------------------------------------
# test_run_gate_extracts_per_dimension_from_eval_output
# ---------------------------------------------------------------------------

def test_run_gate_extracts_per_dimension_from_eval_output(tmp_path):
    """run_gate() correctly extracts nested per_dimension fields from eval JSON.

    This tests the field name mapping fix: eval scripts write 'per_dimension'
    with nested dicts, but check_gates expects flat pass_rate/corr values.
    run_gate() must bridge the gap.
    """
    # Simulate eval_gen output format (per_dimension[dim] = {mean, pass_rate_8, na_count})
    gen_results = {
        "overall_mean": 80.0,
        "per_dimension": {
            "D1_standards": {"mean": 8.5, "pass_rate_8": 0.85, "na_count": 0},
            "D2_security": {"mean": 9.0, "pass_rate_8": 0.95, "na_count": 0},
        },
        "phpcs_pass_rate": 0.97,
        "security_pass_rate": 0.99,
    }
    # Simulate eval_judge output format (per_dimension[dim] = {corr, p_value, n_pairs})
    judge_results = {
        "overall_spearman": {"corr": 0.90, "p_value": 0.001, "n_pairs": 100},
        "per_dimension": {
            "D1_standards": {"corr": 0.85, "p_value": 0.01, "n_pairs": 100},
            "D2_security": {"corr": 0.75, "p_value": 0.02, "n_pairs": 100},
        },
        "spearman_corr": 0.90,
    }

    (tmp_path / "eval_gen_results.json").write_text(json.dumps(gen_results))
    (tmp_path / "eval_judge_results.json").write_text(json.dumps(judge_results))

    # Config with per-dimension targets
    config_content = """
eval:
  overall_mean_target: 75.0
  overall_spearman_target: 0.80
  gen_dimension_targets:
    D1_standards: 0.80
    D2_security: 0.90
  judge_dimension_targets:
    D1_standards: 0.70
    D2_security: 0.80
  phpcs_pass_target: 0.95
  spearman_target: 0.85
  security_pass_target: 0.98
"""
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(config_content)

    from eval.eval_gate import run_gate
    passed, gate_rows = run_gate(
        results_dir=str(tmp_path),
        config_path=str(config_path),
    )

    # Find the per-dim gen gate rows
    d1_gen = [r for r in gate_rows if r["gate"] == "gen_pass_rate/D1_standards"]
    assert len(d1_gen) == 1
    assert d1_gen[0]["actual"] == 0.85  # extracted from per_dimension nested dict

    d2_gen = [r for r in gate_rows if r["gate"] == "gen_pass_rate/D2_security"]
    assert len(d2_gen) == 1
    assert d2_gen[0]["actual"] == 0.95

    # Find the per-dim judge gate rows
    d1_judge = [r for r in gate_rows if r["gate"] == "judge_corr/D1_standards"]
    assert len(d1_judge) == 1
    assert d1_judge[0]["actual"] == 0.85  # extracted from per_dimension nested dict

    d2_judge = [r for r in gate_rows if r["gate"] == "judge_corr/D2_security"]
    assert len(d2_judge) == 1
    assert d2_judge[0]["actual"] == 0.75
    assert d2_judge[0]["passed"] is False  # 0.75 < 0.80 target

    # Overall should fail because D2 judge corr is below target
    assert passed is False
