"""RVAL-04 — offline reward (form, weight) sweep tests (CPU/$0, no GPU/vLLM/API).

Locks: grid coverage + required row keys; weight-0 reproduces the fix_correctness INVALID
verdict; select_config honors BOTH offline gates (never returns oracle-invalid or echo-hackable);
determinism. Set REWARD_SKIP_PHPCS_ASSERT=1 in the environment (offline reward path).
"""
import os
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

import scripts.reward_form_sweep as s  # noqa: E402

_ROWS = s.run_sweep()  # compute once; run_sweep is deterministic + pure-offline

_REQUIRED_KEYS = {"form", "calib_weight", "ci_lo", "valid", "frac_mid", "echo_reward"}


def test_sweep_covers_grid():
    """run_sweep returns exactly one row per (form, weight); each carries the lens keys."""
    expected = {(f, w) for f in s.CALIB_FORMS for w in s.WEIGHT_GRID}
    got = {(r["form"], r["calib_weight"]) for r in _ROWS}
    assert got == expected, f"grid mismatch: missing={expected - got} extra={got - expected}"
    assert len(_ROWS) == len(expected) >= 7
    for r in _ROWS:
        assert _REQUIRED_KEYS <= set(r), f"row missing keys: {_REQUIRED_KEYS - set(r)}"


def test_weight_zero_matches_fix_correctness():
    """calib_weight==0 reproduces the fix_correctness-INVALID oracle verdict (CI-lower<=0)."""
    zero = [r for r in _ROWS if r["calib_weight"] == 0.0]
    assert zero, "no weight-0 configs in grid"
    for r in zero:
        assert r["valid"] is False, f"weight-0 {r['form']} should be oracle-INVALID (fix_correctness)"
        assert r["ci_lo"] is not None and r["ci_lo"] <= 0.0


def test_selection_respects_both_gates():
    """select_config never returns an oracle-invalid OR echo-hackable config."""
    sel = s.select_config(_ROWS)
    if sel is not None:
        assert sel["valid"] is True
        assert sel["ci_lo"] is not None and sel["ci_lo"] > 0.0
        assert sel["echo_reward"] is not None and sel["echo_reward"] <= 0.30
    # If None, it must be because no row clears BOTH gates (documented escalation).
    else:
        qualifying = [r for r in _ROWS
                      if r["valid"] is True and (r["ci_lo"] or -9) > 0
                      and (r["echo_reward"] if r["echo_reward"] is not None else 1) <= 0.30]
        assert not qualifying, "select_config returned None but a qualifying config exists"


def test_selection_deterministic():
    """run_sweep + select_config twice -> identical selected config (deterministic)."""
    sel1 = s.select_config(s.run_sweep())
    sel2 = s.select_config(s.run_sweep())
    assert sel1 == sel2
