"""Tests for scripts/launch_validated_smoke.sh (RVAL-05).

All tests are CPU/offline — NO GPU, NO Tinker, NO training ever launched.

Test cases:
  test_default_is_dry_print          — default invocation exits 0, prints two
                                        rl_train.py commands, no live-launch marker
  test_command_carries_validated_config — assembled commands carry the validated
                                        flags: --codegen-probe-every, --calib-form,
                                        --calib-weight, --total-steps 250,
                                        --checkpoint-every 50
  test_refuses_without_confirm       — the confirm-flag guard is present; dry path
                                        taken; no training invocation without
                                        --i-understand-this-spends-gpu
  test_null_selected_falls_back_to_hybrid08 — when sweep_results.json has
                                        selected=null (the real file), the assembled
                                        command carries --calib-form hybrid
                                        --calib-weight 0.8
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_validated_smoke.sh"
REAL_SWEEP = REPO_ROOT / "output" / "reward_validity" / "sweep_results.json"


def _run_launcher(*extra_args: str, sweep_path: str | None = None) -> subprocess.CompletedProcess:
    """Run the launcher with optional extra args; return CompletedProcess."""
    env = os.environ.copy()
    if sweep_path is not None:
        env["SWEEP_RESULTS_PATH"] = sweep_path
    return subprocess.run(
        ["bash", str(LAUNCHER)] + list(extra_args),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def _minimal_sweep_fixture(selected: dict | None, tmpdir: Path) -> str:
    """Write a minimal sweep_results.json fixture to tmpdir; return path."""
    ranked = [
        {
            "form": "hybrid",
            "calib_weight": 0.8,
            "spearman_vs_target": 0.84,
            "ci_lo": 0.3725,
            "ci_hi": 0.9872,
            "valid": True,
            "n_ckpts": 10,
            "frac_mid": 0.7419,
            "group_reward_std_mean": 0.2535,
            "frac_groups_all_zero": 0.0,
            "frac_groups_nonuniform": 1.0,
            "n_samples": 372,
            "n_groups": 93,
            "echo_reward": 0.99,
        },
    ]
    data = {"ranked": ranked, "selected": selected}
    path = tmpdir / "sweep_results.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# test_default_is_dry_print
# ---------------------------------------------------------------------------


def test_default_is_dry_print(tmp_path):
    """Default invocation (no confirm flag):
    - exits 0
    - prints exactly two rl_train.py invocations (one per seed)
    - does NOT contain a live-launch marker
    """
    sweep = _minimal_sweep_fixture(selected=None, tmpdir=tmp_path)
    result = _run_launcher(sweep_path=sweep)

    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}:\n{result.stderr}"

    output = result.stdout + result.stderr

    # Two distinct seed commands must appear
    seed_a_present = "--lora-seed 12345" in output
    seed_b_present = "--lora-seed 99999" in output
    assert seed_a_present, "Seed A (--lora-seed 12345) not found in output"
    assert seed_b_present, "Seed B (--lora-seed 99999) not found in output"

    # Both lines must reference rl_train.py
    rl_train_count = output.count("rl_train.py")
    assert rl_train_count >= 2, (
        f"Expected at least 2 references to rl_train.py, found {rl_train_count}"
    )

    # No live-launch marker
    assert "LAUNCH CONFIRMED" not in output, (
        "Live-launch marker 'LAUNCH CONFIRMED' appeared in dry-print output"
    )


# ---------------------------------------------------------------------------
# test_command_carries_validated_config
# ---------------------------------------------------------------------------


def test_command_carries_validated_config(tmp_path):
    """The assembled commands include the validated config + trip-wire flags."""
    sweep = _minimal_sweep_fixture(selected=None, tmpdir=tmp_path)
    result = _run_launcher(sweep_path=sweep)

    assert result.returncode == 0
    output = result.stdout + result.stderr

    # Codegen trip-wire wired at checkpoint cadence
    assert "--codegen-probe-every" in output, "Missing --codegen-probe-every"

    # Plan-04 candidate reward config
    assert "--calib-form" in output, "Missing --calib-form"
    assert "--calib-weight" in output, "Missing --calib-weight"

    # Hard budget ceiling (never 500)
    assert "--total-steps 250" in output, "Missing --total-steps 250"

    # Kill-gate cadence
    assert "--checkpoint-every 50" in output, "Missing --checkpoint-every 50"


# ---------------------------------------------------------------------------
# test_refuses_without_confirm
# ---------------------------------------------------------------------------


def test_refuses_without_confirm(tmp_path):
    """Without --i-understand-this-spends-gpu, the launcher takes the dry path.

    Specifically:
    - exits 0 (not an error; guard is a policy gate, not a failure)
    - output indicates dry-print mode
    - does NOT invoke training (no 'LAUNCH CONFIRMED' marker)
    - the guard message references the confirm flag by name
    """
    sweep = _minimal_sweep_fixture(selected=None, tmpdir=tmp_path)
    result = _run_launcher(sweep_path=sweep)

    assert result.returncode == 0
    output = result.stdout + result.stderr

    # Guard communicates the confirm flag to the operator
    assert "--i-understand-this-spends-gpu" in output, (
        "Guard did not mention --i-understand-this-spends-gpu in dry-print output"
    )

    # Dry-print, not a live launch
    assert "LAUNCH CONFIRMED" not in output, (
        "Training was launched without the confirm flag"
    )

    # DRY-PRINT marker present
    assert "DRY-PRINT" in output.upper() or "dry-print" in output.lower(), (
        "Dry-print mode indicator not found in output"
    )


# ---------------------------------------------------------------------------
# test_null_selected_falls_back_to_hybrid08
# ---------------------------------------------------------------------------


def test_null_selected_falls_back_to_hybrid08(tmp_path):
    """When selected=null in sweep_results.json, the command carries hybrid@0.8.

    This tests the REAL code path: Plan 04 left selected=null (documented
    escalation). The launcher must not fail or emit empty flags — it falls back
    to the top oracle-valid ranked entry (hybrid, calib_weight=0.8).
    """
    sweep = _minimal_sweep_fixture(selected=None, tmpdir=tmp_path)
    result = _run_launcher(sweep_path=sweep)

    assert result.returncode == 0
    output = result.stdout + result.stderr

    assert "--calib-form hybrid" in output, (
        f"Expected '--calib-form hybrid' in output (selected=null fallback). Got:\n{output}"
    )
    assert "--calib-weight 0.8" in output, (
        f"Expected '--calib-weight 0.8' in output (selected=null fallback). Got:\n{output}"
    )


# ---------------------------------------------------------------------------
# test_null_selected_with_real_sweep_file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_SWEEP.exists(),
    reason="Real sweep_results.json not present in test env",
)
def test_null_selected_with_real_sweep_file():
    """Against the REAL sweep_results.json (selected=null), the launcher
    assembles commands with hybrid@0.8 — the top oracle-valid ranked entry.
    """
    result = _run_launcher(sweep_path=str(REAL_SWEEP))

    assert result.returncode == 0
    output = result.stdout + result.stderr

    assert "--calib-form hybrid" in output, (
        "Real sweep_results.json: expected --calib-form hybrid in output"
    )
    assert "--calib-weight 0.8" in output, (
        "Real sweep_results.json: expected --calib-weight 0.8 in output"
    )
    assert "rl_train.py" in output
    assert "LAUNCH CONFIRMED" not in output
