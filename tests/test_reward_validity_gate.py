"""Regression tests for the Phase 08.2 reward-validity gate (RVAL-01).

Locks the known oracle verdicts and the TRAIN-GT anti-leakage invariant.
All tests are CPU/offline — no vLLM, no GPU, no Anthropic API.
Reuses on-disk oracle inputs (output/rl_eval/*/judge_responses.jsonl + openai_val.jsonl)
and the pre-built sidecar (data/rl_probe/judge_gt_sidecar.jsonl).

Run:
    REWARD_SKIP_PHPCS_ASSERT=1 python3 -m pytest tests/test_reward_validity_gate.py -x -q
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import pytest

# Set before any reward-related import (mirrors _gen_judge_probe_corpus.py convention)
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

REPO = Path(__file__).resolve().parent.parent
SIDECAR_PATH = REPO / "data/rl_probe/judge_gt_sidecar.jsonl"


# ---------------------------------------------------------------------------
# Test 1 — pairwise_rank_agreement is VALID (CI-lower > 0)
# ---------------------------------------------------------------------------

class TestPairwiseRankAgreementValid:
    """Regression: pairwise_rank_agreement must remain VALID (ci_lo > 0).

    This is the validated replacement reward for seedA's invalid fix-correctness.
    A future refactor that breaks this would indicate the oracle pipeline changed
    in a way that invalidates the Phase 08.2 replacement decision.
    """

    def test_pairwise_rank_agreement_valid(self):
        """run_validity_gate('pairwise_rank_agreement') -> valid=True AND ci_lo>0."""
        from scripts.reward_validity_gate import run_validity_gate

        result = run_validity_gate("pairwise_rank_agreement")
        assert result.valid is True, (
            f"pairwise_rank_agreement must be VALID (ci_lo>0); got valid={result.valid}, "
            f"ci_lo={result.ci_lo:.3f}. Oracle corpus or pipeline may have changed."
        )
        assert result.ci_lo > 0, (
            f"pairwise_rank_agreement CI-lower must be > 0 (standing gate rule D-08.2); "
            f"got ci_lo={result.ci_lo:.3f}, ci_hi={result.ci_hi:.3f}."
        )


# ---------------------------------------------------------------------------
# Test 2 — fix_correctness_BASELINE is INVALID (Goodhart verdict locked)
# ---------------------------------------------------------------------------

class TestFixCorrectnessInvalid:
    """Regression: fix_correctness_BASELINE must remain INVALID.

    The Phase 10 RLEV proved empirically that seedA's optimized proxy does NOT
    track teacher-Spearman (corr ≈ −0.24, CI includes 0). Locking this verdict
    prevents a future refactor from silently flipping it to 'valid', which would
    undermine the phase rationale (Goodhart must remain confirmed).
    """

    def test_fix_correctness_invalid(self):
        """run_validity_gate('fix_correctness_BASELINE') -> valid=False (ci_lo<=0)."""
        from scripts.reward_validity_gate import run_validity_gate

        result = run_validity_gate("fix_correctness_BASELINE")
        assert result.valid is False, (
            f"fix_correctness_BASELINE must be INVALID (ci_lo<=0, Goodhart confirmed); "
            f"got valid={result.valid}, ci_lo={result.ci_lo:.3f}. "
            f"If this changed, re-verify the FIX_CORR series and oracle pipeline."
        )
        # Explicit ci_lo check: NaN also fails (valid would be False via NaN-safe logic,
        # but we want a clear signal if something produced a bogus positive ci_lo)
        assert result.ci_lo <= 0 or result.ci_lo != result.ci_lo, (  # <=0 OR is NaN
            f"fix_correctness_BASELINE ci_lo should be <=0 or NaN; "
            f"got ci_lo={result.ci_lo:.3f} (spearman={result.spearman_vs_target:.3f})."
        )


# ---------------------------------------------------------------------------
# Test 3 — GT sidecar coverage + TRAIN-only anti-leakage invariant
# ---------------------------------------------------------------------------

class TestGTSidecarCoverageAndTrainOnly:
    """Validates judge_gt_sidecar.jsonl: coverage >= 60 rows + source=='train' everywhere.

    Anti-leakage invariant (T-09-LEAK / T-082-01): val GT must NEVER enter the reward
    path. Using val GT would make SC2 (pairwise_rank_agreement calibration) circular —
    the oracle uses val as its held-out target; if val GT also flowed into the reward,
    the gate would be testing against its own training signal.

    source=='train' on EVERY row is the machine-checkable proxy for this invariant.
    """

    def test_gt_sidecar_coverage_and_train_only(self):
        """sidecar has >=60 rows; all rows have teacher_overall+code_hash; source=='train'."""
        assert SIDECAR_PATH.exists(), (
            f"GT sidecar not found: {SIDECAR_PATH}. "
            f"Run: REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/build_reward_gt_sidecar.py"
        )

        rows = [json.loads(l) for l in SIDECAR_PATH.open() if l.strip()]

        assert len(rows) >= 60, (
            f"GT sidecar must have >=60 rows (covering distinct prompt_ids with TRAIN GT); "
            f"got {len(rows)}. Re-run build_reward_gt_sidecar.py."
        )

        for i, row in enumerate(rows):
            assert "teacher_overall" in row, (
                f"Row {i} missing 'teacher_overall' field: {list(row.keys())}"
            )
            assert "code_hash" in row, (
                f"Row {i} missing 'code_hash' field (join key): {list(row.keys())}"
            )
            assert row.get("source") == "train", (
                f"Row {i} source=={row.get('source')!r} != 'train'. "
                f"Val GT must NOT enter the sidecar (T-09-LEAK anti-leakage invariant)."
            )

        # Spot-check type correctness
        for i, row in enumerate(rows[:5]):
            assert isinstance(row["teacher_overall"], (int, float)), (
                f"Row {i} teacher_overall must be numeric; got {type(row['teacher_overall'])}"
            )
            assert isinstance(row["code_hash"], str) and len(row["code_hash"]) > 0, (
                f"Row {i} code_hash must be a non-empty string; got {row['code_hash']!r}"
            )


# ---------------------------------------------------------------------------
# Test 4 — GateResult dataclass shape contract
# ---------------------------------------------------------------------------

class TestGateResultShape:
    """GateResult must carry the exact fields Plans 03/04 depend on."""

    def test_gate_result_shape(self):
        """GateResult has: form, spearman_vs_target, ci_lo, ci_hi, valid, n_ckpts."""
        from scripts.reward_validity_gate import GateResult

        required_fields = {"form", "spearman_vs_target", "ci_lo", "ci_hi", "valid", "n_ckpts"}
        actual_fields = {f.name for f in fields(GateResult)}

        missing = required_fields - actual_fields
        assert not missing, (
            f"GateResult is missing required fields: {missing}. "
            f"Plans 03/04 depend on all 6 fields for sweep and calibration wiring."
        )

        # Verify field types via a live gate call
        from scripts.reward_validity_gate import run_validity_gate

        result = run_validity_gate("pairwise_rank_agreement")

        assert isinstance(result.form, str), f"form must be str; got {type(result.form)}"
        assert isinstance(result.spearman_vs_target, float), (
            f"spearman_vs_target must be float; got {type(result.spearman_vs_target)}"
        )
        assert isinstance(result.ci_lo, float), (
            f"ci_lo must be float; got {type(result.ci_lo)}"
        )
        assert isinstance(result.ci_hi, float), (
            f"ci_hi must be float; got {type(result.ci_hi)}"
        )
        assert isinstance(result.valid, bool), (
            f"valid must be bool; got {type(result.valid)}"
        )
        assert isinstance(result.n_ckpts, int) and result.n_ckpts > 0, (
            f"n_ckpts must be a positive int; got {result.n_ckpts!r}"
        )
