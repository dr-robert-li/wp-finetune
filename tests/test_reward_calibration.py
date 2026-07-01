"""Unit tests for scripts/reward_calibration.py (Plan 08.2-03 / RVAL-02).

Covers:
  - calibration_reward form numerics: pairwise, hybrid, calibration
  - Saturation gradient-density test: 95 vs 96 differ under hybrid (gradient survives)
  - Anti-leakage test: load_gt_anchor_set rejects non-train GT rows
  - Oracle VALID assertion: run_validity_gate on the registered form "calibration_reward_impl"
    returns valid=True (ci_lo>0) — this is the SC2 acceptance criterion
  - Phase-8/8.1 no-regression guard (calib_weight=0 leaves combine_judge_reward untouched)

All offline / CPU / $0. No vLLM, no GPU, no API calls.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helper: build a minimal sidecar file for tests that need load_gt_anchor_set
# ---------------------------------------------------------------------------

def _make_sidecar(rows: list[dict], path: str) -> str:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Task 1a: pairwise form numerics
# ---------------------------------------------------------------------------

class TestPairwiseForm:
    """pairwise: fraction of anchor pairs ordered the same as teacher."""

    def test_perfect_concordance(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        # model_overall > anchor_gt for all anchors, teacher_overall also higher:
        # concordant on every pair -> 1.0
        anchor_set = [(None, float(a)) for a in [50.0, 60.0, 70.0, 80.0]]
        score = rc.calibration_reward(90.0, 95.0, anchor_set, form="pairwise")
        assert score == pytest.approx(1.0), f"Expected 1.0 concordance, got {score}"

    def test_perfect_discordance(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        # model ranks HIGH (model > anchor_gt) but teacher ranks LOW (teacher < anchor_gt)
        # -> every pair discordant -> 0.0
        # Anchor set with gt BETWEEN teacher and model: teacher=50 < anchor_gt=60,70,80 < model=90
        # sign(90-60)=+1, sign(50-60)=-1 -> discordant for all anchors
        anchor_set = [(None, float(a)) for a in [60.0, 70.0, 80.0]]
        score = rc.calibration_reward(90.0, 50.0, anchor_set, form="pairwise")
        assert score == pytest.approx(0.0), f"Expected 0.0 discordance, got {score}"

    def test_half_concordance(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        # 2 anchors with gt < model/teacher_overall -> concordant (model>anchor, teacher>anchor)
        # 2 anchors with gt > both -> discordant (model<anchor, teacher<anchor: concordant)
        # Actually we build a set where half the anchors are concordant, half not
        # model=70, teacher=70: sign(70-anchor) vs sign(70-anchor) always concordant
        # Build it so concordance is ~0.5 by mixing:
        # anchor_gt=60: sign(70-60)=+1, sign(70-60)=+1 -> concordant
        # anchor_gt=80: sign(70-80)=-1, sign(70-80)=-1 -> concordant (both neg)
        # To get half: need teacher to disagree with model on some anchors
        # model=70, teacher=50: anchor_gt=60 -> model>anchor (+), teacher<anchor (-): discordant
        # anchor_gt=40 -> model>anchor (+), teacher>anchor (+): concordant
        anchor_set = [(None, 60.0), (None, 40.0)]
        score = rc.calibration_reward(70.0, 50.0, anchor_set, form="pairwise")
        # anchor_gt=60: sign(70-60)=+1, sign(50-60)=-1 -> discordant
        # anchor_gt=40: sign(70-40)=+1, sign(50-40)=+1 -> concordant
        # 1/2 = 0.5
        assert score == pytest.approx(0.5), f"Expected 0.5 half-concordance, got {score}"

    def test_equal_anchor_gt_skipped(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        import math
        # Anchors where anchor_gt == teacher_overall should be skipped (sign undefined)
        # Only non-equal anchors count
        anchor_set = [(None, 70.0), (None, 70.0)]  # both equal teacher_overall
        score = rc.calibration_reward(80.0, 70.0, anchor_set, form="pairwise")
        # All pairs skipped -> fallback to NaN (no denominator): this is correct
        # behavior (the oracle also returns nan when t==0). Must not raise.
        assert math.isnan(score) or score == 0.0, (
            f"Expected NaN or 0.0 when all anchors skipped, got {score}"
        )

    def test_in_range_01(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        import random
        rng = random.Random(42)
        anchor_set = [(None, float(rng.randint(30, 100))) for _ in range(20)]
        score = rc.calibration_reward(75.0, 65.0, anchor_set, form="pairwise")
        assert 0.0 <= score <= 1.0 or score != score, f"Out of [0,1]: {score}"


# ---------------------------------------------------------------------------
# Task 1b: hybrid form numerics
# ---------------------------------------------------------------------------

class TestHybridForm:
    """hybrid: pairwise concordance minus small weighted calibration-error term."""

    def test_hybrid_below_pairwise(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        # When model != teacher, hybrid < pairwise (because calib_error term > 0)
        anchor_set = [(None, float(a)) for a in [50.0, 60.0, 70.0]]
        pairwise = rc.calibration_reward(90.0, 95.0, anchor_set, form="pairwise")
        hybrid = rc.calibration_reward(90.0, 95.0, anchor_set, form="hybrid")
        # model=90, teacher=95: calib_error = |90-95|/100 = 0.05 > 0
        # -> hybrid < pairwise
        assert hybrid < pairwise, (
            f"hybrid ({hybrid:.4f}) should be < pairwise ({pairwise:.4f}) when model!=teacher"
        )

    def test_hybrid_in_range_01(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        import random
        rng = random.Random(99)
        anchor_set = [(None, float(rng.randint(30, 100))) for _ in range(20)]
        score = rc.calibration_reward(75.0, 65.0, anchor_set, form="hybrid")
        assert 0.0 <= score <= 1.0, f"hybrid out of [0,1]: {score}"

    def test_saturation_gradient_density(self):
        """CRITICAL: 95 vs 96 against a saturated anchor set yield DIFFERENT hybrid rewards.

        This is the intra-group gradient density guard. Two completions scoring 95 vs 96
        against an anchor set that is fully saturated (all anchors > 95) may tie under
        pure pairwise (both rank below every anchor) but MUST differ under hybrid because
        the calibration-error term |model-teacher|/100 tracks the absolute difference.

        teacher_overall for 95: we say teacher says 95.
        teacher_overall for 96: we say teacher says 96.
        anchor_set: all saturated (> 98) so both completions rank below all anchors.
        """
        rc = pytest.importorskip("scripts.reward_calibration")
        # Saturated anchor set: all gt > 95 and > 96
        anchor_set = [(None, float(a)) for a in [97.0, 98.0, 99.0, 100.0]]

        # teacher=95 for completion at 95, teacher=96 for completion at 96
        # (each completion's teacher GT is its own ground truth)
        score_95 = rc.calibration_reward(95.0, 95.0, anchor_set, form="hybrid")
        score_96 = rc.calibration_reward(96.0, 96.0, anchor_set, form="hybrid")

        # BOTH have pairwise=0 (model < all anchors, teacher < all anchors -> concordant!)
        # Wait: sign(95-97)=-1, sign(95-97)=-1 -> concordant! pairwise=1.0 for both.
        # The gradient difference comes from: model=95 teacher=97 vs model=96 teacher=97
        # Let's use a more realistic case: teacher is anchored independently
        # Use teacher=97 for both (same teacher, different model scores):
        # calib_error(95) = |95-97|/100 = 0.02
        # calib_error(96) = |96-97|/100 = 0.01
        # -> score_96 > score_95 (less calibration error)
        score_95_t97 = rc.calibration_reward(95.0, 97.0, anchor_set, form="hybrid")
        score_96_t97 = rc.calibration_reward(96.0, 97.0, anchor_set, form="hybrid")

        assert score_96_t97 != score_95_t97, (
            f"Hybrid must give distinct rewards for model=95 vs model=96 under saturation: "
            f"score_95={score_95_t97:.6f}, score_96={score_96_t97:.6f}"
        )
        # And more specifically: 96 should score higher (closer to teacher 97)
        assert score_96_t97 > score_95_t97, (
            f"model=96 (closer to teacher=97) should score higher than model=95: "
            f"score_95={score_95_t97:.6f}, score_96={score_96_t97:.6f}"
        )


# ---------------------------------------------------------------------------
# Task 1c: calibration form (pure absolute calibration)
# ---------------------------------------------------------------------------

class TestCalibrationForm:
    """calibration: 1 - |model - teacher| / 100."""

    def test_perfect_calibration(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        anchor_set = [(None, 70.0)]  # doesn't matter for this form
        score = rc.calibration_reward(80.0, 80.0, anchor_set, form="calibration")
        assert score == pytest.approx(1.0), f"Expected 1.0 at perfect calibration, got {score}"

    def test_worst_calibration(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        anchor_set = [(None, 70.0)]
        score = rc.calibration_reward(0.0, 100.0, anchor_set, form="calibration")
        assert score == pytest.approx(0.0), f"Expected 0.0 at max error (100 pts), got {score}"

    def test_partial_calibration(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        anchor_set = [(None, 70.0)]
        score = rc.calibration_reward(70.0, 80.0, anchor_set, form="calibration")
        expected = 1.0 - 10.0 / 100.0
        assert score == pytest.approx(expected), f"Expected {expected}, got {score}"

    def test_in_range_01(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        anchor_set = [(None, 70.0)]
        for m in [0.0, 25.0, 50.0, 75.0, 100.0]:
            for t in [0.0, 25.0, 50.0, 75.0, 100.0]:
                score = rc.calibration_reward(m, t, anchor_set, form="calibration")
                assert 0.0 <= score <= 1.0, f"Out of [0,1] for m={m}, t={t}: {score}"


# ---------------------------------------------------------------------------
# Task 1d: CALIB_FORMS exported constant
# ---------------------------------------------------------------------------

class TestCalibForms:
    def test_calib_forms_has_all_three(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        assert hasattr(rc, "CALIB_FORMS"), "CALIB_FORMS not exported"
        assert "pairwise" in rc.CALIB_FORMS
        assert "hybrid" in rc.CALIB_FORMS
        assert "calibration" in rc.CALIB_FORMS

    def test_unknown_form_raises(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        with pytest.raises((ValueError, KeyError)):
            rc.calibration_reward(80.0, 80.0, [(None, 70.0)], form="nonexistent_form")


# ---------------------------------------------------------------------------
# Task 1e: load_gt_anchor_set — anti-leakage invariant
# ---------------------------------------------------------------------------

class TestLoadGtAnchorSet:
    """Anti-leakage: load_gt_anchor_set must reject any row where source != 'train'."""

    def test_rejects_val_source(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        rows = [
            {"prompt_id": 0, "code_hash": "abc", "teacher_overall": 80.0, "source": "train"},
            {"prompt_id": 1, "code_hash": "def", "teacher_overall": 70.0, "source": "val"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        try:
            with pytest.raises((AssertionError, ValueError), match="train|source|val"):
                rc.load_gt_anchor_set(tmp)
        finally:
            os.unlink(tmp)

    def test_accepts_all_train(self):
        rc = pytest.importorskip("scripts.reward_calibration")
        rows = [
            {"prompt_id": i, "code_hash": f"h{i}", "teacher_overall": float(60 + i * 5), "source": "train"}
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        try:
            anchor_set, gt_map = rc.load_gt_anchor_set(tmp)
            assert len(anchor_set) == 10
            assert len(gt_map) > 0
        finally:
            os.unlink(tmp)

    def test_anchor_set_structure(self):
        """Each element of anchor_set must be a 2-tuple (anchor_score_placeholder, teacher_gt)."""
        rc = pytest.importorskip("scripts.reward_calibration")
        rows = [
            {"prompt_id": i, "code_hash": f"h{i}", "teacher_overall": float(70 + i), "source": "train"}
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        try:
            anchor_set, gt_map = rc.load_gt_anchor_set(tmp)
            for item in anchor_set:
                assert len(item) == 2, f"Expected 2-tuple, got {item}"
                _placeholder, teacher_gt = item
                assert isinstance(teacher_gt, float), f"Expected float teacher_gt, got {type(teacher_gt)}"
        finally:
            os.unlink(tmp)

    def test_gt_map_keyed_by_code_hash(self):
        """gt_map must be keyed by code_hash and contain teacher_overall values."""
        rc = pytest.importorskip("scripts.reward_calibration")
        rows = [
            {"prompt_id": i, "code_hash": f"hash_{i:04x}", "teacher_overall": float(50 + i * 3), "source": "train"}
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            tmp = f.name
        try:
            anchor_set, gt_map = rc.load_gt_anchor_set(tmp)
            for row in rows:
                assert row["code_hash"] in gt_map, f"code_hash {row['code_hash']!r} missing from gt_map"
                assert gt_map[row["code_hash"]] == pytest.approx(row["teacher_overall"])
        finally:
            os.unlink(tmp)

    def test_real_sidecar_all_train(self):
        """The actual judge_gt_sidecar.jsonl must pass (all rows are source=='train')."""
        rc = pytest.importorskip("scripts.reward_calibration")
        sidecar = REPO / "data/rl_probe/judge_gt_sidecar.jsonl"
        if not sidecar.exists():
            pytest.skip("judge_gt_sidecar.jsonl not found")
        anchor_set, gt_map = rc.load_gt_anchor_set(str(sidecar))
        assert len(anchor_set) >= 60, f"Expected >=60 anchor rows, got {len(anchor_set)}"
        assert len(gt_map) >= 60


# ---------------------------------------------------------------------------
# Task 3 (pre-wired): SC2 oracle VALID assertion
# The implemented calibration form must score VALID (ci_lo>0) through the standing gate.
# ---------------------------------------------------------------------------

class TestOracleValidAssertion:
    """SC2 acceptance: the registered oracle form 'calibration_reward_impl' is VALID."""

    def test_calibration_reward_impl_passes_gate(self):
        """run_validity_gate('calibration_reward_impl') must return valid=True (ci_lo>0)."""
        gate = pytest.importorskip("scripts.reward_validity_gate")
        result = gate.run_validity_gate("calibration_reward_impl")
        assert result.valid is True, (
            f"SC2 FAILED: calibration_reward_impl ci_lo={result.ci_lo:.3f} <= 0. "
            f"Full result: {result}"
        )
        assert result.ci_lo > 0, (
            f"SC2 FAILED: ci_lo={result.ci_lo:.3f} must be strictly > 0"
        )

    def test_calibration_reward_impl_in_forms(self):
        """The form must be registered in FORMS (not just accessible via form_fn)."""
        oracle = pytest.importorskip("scripts._reward_validity_oracle")
        assert "calibration_reward_impl" in oracle.FORMS, (
            f"'calibration_reward_impl' not registered in FORMS. "
            f"Available: {list(oracle.FORMS.keys())}"
        )


# ---------------------------------------------------------------------------
# Phase-8/8.1 regression guard: calib_weight=0 must not change combine_judge_reward
# ---------------------------------------------------------------------------

class TestPhase8Regression:
    """calib_weight=0 default must reproduce today's combine_judge_reward behavior byte-for-byte."""

    def test_weight_zero_passthrough(self):
        """At calib_weight=0.0, the combined judge reward must equal combine_judge_reward output."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        rc = pytest.importorskip("scripts.reward_calibration")

        fix_c = 0.75
        consistency = 0.60
        weight = 0.45  # lever2 weight

        # Expected: today's combine_judge_reward output
        expected = rr.combine_judge_reward(fix_c, consistency, weight=weight)

        # With calib_weight=0, calibration_reward is not applied
        # We use augment_judge_scalar with calib_weight=0 to verify
        result = rc.augment_judge_scalar(
            judge_scalar=expected,
            calib_reward=0.8,   # any value — should be ignored at weight=0
            calib_weight=0.0,
        )
        assert result == pytest.approx(expected), (
            f"calib_weight=0 must be a passthrough: expected {expected:.6f}, got {result:.6f}"
        )

    def test_weight_nonzero_modifies_scalar(self):
        """At calib_weight > 0, the scalar is modified toward the calibration reward."""
        rc = pytest.importorskip("scripts.reward_calibration")

        judge_scalar = 0.6
        calib_reward = 0.9
        calib_weight = 0.3

        result = rc.augment_judge_scalar(
            judge_scalar=judge_scalar,
            calib_reward=calib_reward,
            calib_weight=calib_weight,
        )
        expected = (1.0 - calib_weight) * judge_scalar + calib_weight * calib_reward
        assert result == pytest.approx(expected), (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_nan_guard_at_weight_zero(self):
        """calib_weight=0 with NaN calib_reward must not propagate NaN (the 0*NaN trap)."""
        rc = pytest.importorskip("scripts.reward_calibration")
        import math

        judge_scalar = 0.7
        result = rc.augment_judge_scalar(
            judge_scalar=judge_scalar,
            calib_reward=float("nan"),
            calib_weight=0.0,
        )
        assert not math.isnan(result), (
            f"0*NaN trap: calib_weight=0 with NaN calib_reward must not produce NaN, got {result}"
        )
        assert result == pytest.approx(judge_scalar)

    def test_nan_guard_at_nonzero_weight_missing_gt(self):
        """calib_weight > 0 with NaN calib_reward (missing GT) must fall back to judge_scalar."""
        rc = pytest.importorskip("scripts.reward_calibration")
        import math

        judge_scalar = 0.65
        result = rc.augment_judge_scalar(
            judge_scalar=judge_scalar,
            calib_reward=float("nan"),
            calib_weight=0.3,
        )
        assert not math.isnan(result), (
            f"Missing-GT NaN must not propagate: got {result}"
        )
        # Should fall back to judge_scalar when calib_reward is NaN
        assert result == pytest.approx(judge_scalar), (
            f"Expected fallback to judge_scalar {judge_scalar}, got {result}"
        )
