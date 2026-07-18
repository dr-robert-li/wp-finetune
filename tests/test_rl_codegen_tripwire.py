"""Tests for the RVAL-03 codegen trip-wire (scripts/rl_codegen_tripwire.py).

All tests are CPU/offline only — NO GPU, NO vLLM, NO Anthropic API.
run_codegen_probe (the live GPU path) is NEVER called here.

Test cases:
  test_decision_only              — check_codegen_tripwire pure decision logic
  test_args_wired                 — _parse_args exposes the new trip-wire flags
  test_tripwire_fires_below_bar   — run_training_step halts on below-bar inject
  test_tripwire_silent_above_bar  — run_training_step continues on above-bar inject
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Reuse shared test helpers from the integration suite to avoid drift.
# ---------------------------------------------------------------------------
from tests.test_rl_train_integration import (
    _gen_pool,
    _judge_pool,
    _make_args,
    _wire,
)


# ---------------------------------------------------------------------------
# test_decision_only
# ---------------------------------------------------------------------------


def test_decision_only():
    """check_codegen_tripwire: pure decision, no GPU required."""
    from scripts.rl_codegen_tripwire import (
        CODEGEN_BAR_V12,
        EXEC_FLOOR,
        KNOWLEDGE_FLOOR,
        check_codegen_tripwire,
    )

    # Sanity: constant is exactly the v1.2 SFT bar (T-082-05 guard).
    assert abs(CODEGEN_BAR_V12 - 0.4616) < 1e-9, (
        f"CODEGEN_BAR_V12 should be 0.4616, got {CODEGEN_BAR_V12}"
    )

    # Below bar → halt reason returned (not None).
    reason = check_codegen_tripwire(0.40)
    assert reason is not None, "score below bar must return a halt reason"
    assert "0.40" in reason or "CODEGEN" in reason  # reason is descriptive

    # Above bar → None (no halt).
    assert check_codegen_tripwire(0.50) is None, "score above bar must return None"

    # Exactly at bar → None (>= check, not strictly >).
    assert check_codegen_tripwire(0.4616) is None, "score at bar must return None (>= passes)"

    # None score → None (probe not run / unavailable; silently skip).
    assert check_codegen_tripwire(None) is None, "None score must return None (skip)"

    # Sub-floor breach: exec below EXEC_FLOOR even if overall is at bar.
    reason_exec = check_codegen_tripwire(
        0.4616,
        sub_scores={"exec": EXEC_FLOOR - 0.01},
    )
    assert reason_exec is not None, (
        "exec sub-floor breach must return a halt reason even if overall passes"
    )
    assert "exec" in reason_exec.lower()

    # Sub-floor breach: knowledge below KNOWLEDGE_FLOOR.
    reason_know = check_codegen_tripwire(
        0.4616,
        sub_scores={"knowledge": KNOWLEDGE_FLOOR - 0.01},
    )
    assert reason_know is not None, "knowledge sub-floor breach must return a halt reason"
    assert "knowledge" in reason_know.lower()

    # Sub-scores above floors → None (no halt from sub-floors).
    assert check_codegen_tripwire(
        0.4616,
        sub_scores={"knowledge": KNOWLEDGE_FLOOR, "exec": EXEC_FLOOR},
    ) is None, "sub-scores at or above floors with overall at bar must return None"

    # Custom bar override.
    assert check_codegen_tripwire(0.50, bar=0.60) is not None, (
        "score below custom bar must return reason"
    )
    assert check_codegen_tripwire(0.65, bar=0.60) is None, (
        "score above custom bar must return None"
    )


# ---------------------------------------------------------------------------
# test_args_wired
# ---------------------------------------------------------------------------


def test_args_wired():
    """_parse_args exposes the trip-wire flags with documented defaults."""
    rl_train = pytest.importorskip("scripts.rl_train")
    from scripts.rl_codegen_tripwire import CODEGEN_BAR_V12

    # Default values.
    args = rl_train._parse_args([])
    assert hasattr(args, "codegen_probe_every"), "_parse_args must expose codegen_probe_every"
    assert args.codegen_probe_every == 0, "default codegen_probe_every must be 0 (disabled)"
    assert hasattr(args, "codegen_bar"), "_parse_args must expose codegen_bar"
    assert abs(args.codegen_bar - CODEGEN_BAR_V12) < 1e-9, (
        f"default codegen_bar must equal CODEGEN_BAR_V12 ({CODEGEN_BAR_V12})"
    )
    assert hasattr(args, "codegen_score_override"), "_parse_args must expose codegen_score_override"
    assert args.codegen_score_override is None, "default codegen_score_override must be None"

    # Non-default values round-trip correctly.
    args2 = rl_train._parse_args([
        "--codegen-probe-every", "50",
        "--codegen-bar", "0.50",
        "--codegen-score-override", "0.42",
    ])
    assert args2.codegen_probe_every == 50
    assert abs(args2.codegen_bar - 0.50) < 1e-9
    assert abs(args2.codegen_score_override - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# test_tripwire_fires_below_bar
# ---------------------------------------------------------------------------


def test_tripwire_fires_below_bar(monkeypatch, tmp_path):
    """run_training_step halts (returns True) when a below-bar score is injected.

    Wiring: codegen_probe_every=1 (probe every step) + codegen_score_override=0.40
    (below CODEGEN_BAR_V12=0.4616) → halt_reason set → return True + emergency ckpt.
    run_codegen_probe (GPU path) is NOT called.
    """
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.0)

    # Capture _save_checkpoint calls to confirm emergency checkpoint is requested.
    saved_names: list[str] = []
    monkeypatch.setattr(
        rl_train,
        "_save_checkpoint",
        lambda tc_, name, manifest: saved_names.append(name) or "/fake/emergency",
    )

    # Guard: run_codegen_probe must NEVER be called in this test.
    monkeypatch.setattr(
        rl_train,
        "run_codegen_probe",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_codegen_probe must not be called in offline tests")
        ),
    )

    args = _make_args(
        codegen_probe_every=1,           # probe every step
        codegen_score_override=0.40,     # below CODEGEN_BAR_V12 = 0.4616
        codegen_bar=0.4616,
    )
    manifest = {"checkpoints": []}
    sampling_client = tc.save_weights_and_get_sampling_client()

    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest=manifest,
    )

    # Must halt (return True).
    assert halted is True, (
        "run_training_step must return True (halt) when injected score is below bar"
    )

    # Emergency checkpoint must have been saved.
    assert any("emergency" in n for n in saved_names), (
        f"emergency checkpoint must be saved on codegen halt; got saved_names={saved_names}"
    )

    # optim_step must NOT have been called (gradient not committed on halt).
    assert not tc.optim_step.called, (
        "optim_step must NOT be called when trip-wire halts (CR-04 ordering)"
    )


# ---------------------------------------------------------------------------
# test_tripwire_silent_above_bar
# ---------------------------------------------------------------------------


def test_tripwire_silent_above_bar(monkeypatch, tmp_path):
    """run_training_step does NOT halt when injected score is at/above bar.

    Wiring: codegen_probe_every=1 + codegen_score_override=0.50 (above bar)
    → check_codegen_tripwire returns None → training continues normally.
    run_codegen_probe (GPU path) is NOT called.
    """
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.0)

    # Guard: run_codegen_probe must NEVER be called in this test.
    monkeypatch.setattr(
        rl_train,
        "run_codegen_probe",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_codegen_probe must not be called in offline tests")
        ),
    )

    args = _make_args(
        codegen_probe_every=1,           # probe every step
        codegen_score_override=0.50,     # above CODEGEN_BAR_V12 = 0.4616
        codegen_bar=0.4616,
    )
    manifest = {"checkpoints": []}
    sampling_client = tc.save_weights_and_get_sampling_client()

    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest=manifest,
    )

    # Must NOT halt on the codegen axis (returns per the other guards: KL=0 → no halt).
    assert halted is False, (
        "run_training_step must return False when injected score is at/above bar"
    )

    # optim_step IS called (training continued — safe path committed the gradient).
    assert tc.optim_step.called, (
        "optim_step must be called on the safe path (no codegen halt)"
    )


# ---------------------------------------------------------------------------
# test_tripwire_at_bar_passes
# ---------------------------------------------------------------------------


def test_tripwire_at_bar_passes(monkeypatch, tmp_path):
    """Injected score exactly at bar (0.4616) does NOT halt — >= check."""
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.0)

    monkeypatch.setattr(
        rl_train,
        "run_codegen_probe",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("run_codegen_probe must not be called in offline tests")
        ),
    )

    args = _make_args(
        codegen_probe_every=1,
        codegen_score_override=0.4616,   # exactly at bar
        codegen_bar=0.4616,
    )
    manifest = {"checkpoints": []}
    sampling_client = tc.save_weights_and_get_sampling_client()

    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest=manifest,
    )

    assert halted is False, (
        "score exactly at bar (>=) must not halt — codegen trip-wire uses >= not >"
    )


# ---------------------------------------------------------------------------
# test_tripwire_disabled_by_default
# ---------------------------------------------------------------------------


def test_tripwire_disabled_by_default(monkeypatch, tmp_path):
    """When codegen_probe_every=0 (default), the probe is never triggered."""
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.0)

    # run_codegen_probe must not be called; check_codegen_tripwire neither
    # (probe cadence gated by codegen_probe_every > 0).
    call_log: list[str] = []
    monkeypatch.setattr(
        rl_train,
        "run_codegen_probe",
        lambda *a, **kw: call_log.append("run_codegen_probe") or {},
    )
    monkeypatch.setattr(
        rl_train,
        "check_codegen_tripwire",
        lambda *a, **kw: call_log.append("check_codegen_tripwire") or None,
    )

    # _make_args has no codegen_probe_every → getattr defaults to 0 in run_training_step.
    args = _make_args()
    manifest = {"checkpoints": []}
    sampling_client = tc.save_weights_and_get_sampling_client()

    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest=manifest,
    )

    assert halted is False
    assert not call_log, (
        f"probe/decision must not be called when codegen_probe_every=0; called={call_log}"
    )
