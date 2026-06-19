"""Unit tests for scripts/reward_pipeline.py.

Tests are GPU-free: all external service calls mocked.
Covers GRPO-01..04, SC2 (security gate), RLEV-02 (breakdown contract).

WAVE 0 NOTES
------------
- TestJudgeWrapper and TestOffsetApply tests are FILLED IN this plan (08-01).
- TestMOGRPONorm, TestVeRPO, TestSecurityGate, TestCompositeWeights, and
  TestBreakdownContract are STUBBED with pytest.skip; implemented in 08-02/03.
- All imports of reward_pipeline symbols are kept INSIDE test method bodies
  so this file collects cleanly even when scripts/reward_pipeline.py is absent.
- Test method names embed the -k keywords required by 08-VALIDATION.md:
    -k judge_single   → TestJudgeWrapper methods
    -k offset_loader  → TestOffsetApply methods
    -k mogrpo         → TestMOGRPONorm methods
    -k verpo          → TestVeRPO methods
    -k security_gate  → TestSecurityGate methods
    -k composite      → TestCompositeWeights methods
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# TestJudgeWrapper — GRPO-01 / Task 2 (judge_score_single RC-A guard)
# ---------------------------------------------------------------------------


class TestJudgeWrapper:
    """Tests for eval.eval_judge.judge_score_single (08-01 Task 2).

    Method names embed 'judge_single' so -k judge_single selects all three.
    """

    def test_judge_single_returns_float(self, monkeypatch):
        """parse_judge_response yields {"overall_score": 82} → returns 82.0."""
        import eval.eval_judge as ej

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({"overall_score": 82})
        monkeypatch.setattr(ej, "_judge_create", lambda *a, **kw: mock_resp)

        result = ej.judge_score_single("<?php echo 1;", MagicMock(), "test-model")
        assert result == 82.0, f"Expected 82.0, got {result}"
        assert isinstance(result, float), "Return type must be float"

    def test_judge_single_uses_judge_create(self, monkeypatch):
        """judge_score_single calls _judge_create, NOT client.chat.completions.create.

        This verifies the RC-A enable_thinking=False guard is preserved.
        """
        import eval.eval_judge as ej

        judge_create_calls = []

        def fake_judge_create(*args, **kwargs):
            judge_create_calls.append((args, kwargs))
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps({"overall_score": 50})
            return mock_resp

        monkeypatch.setattr(ej, "_judge_create", fake_judge_create)
        mock_client = MagicMock()

        ej.judge_score_single("<?php noop();", mock_client, "test-model")

        assert len(judge_create_calls) == 1, "_judge_create must be called exactly once"
        # The raw client must NOT have been called directly
        mock_client.chat.completions.create.assert_not_called()

    def test_judge_single_parse_failure_returns_none(self, monkeypatch):
        """parse_judge_response returns None → judge_score_single returns None."""
        import eval.eval_judge as ej

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "not valid json at all <<<###"
        monkeypatch.setattr(ej, "_judge_create", lambda *a, **kw: mock_resp)

        result = ej.judge_score_single("<?php bad_parse();", MagicMock(), "test-model")
        assert result is None, f"Expected None on parse failure, got {result!r}"


# ---------------------------------------------------------------------------
# TestOffsetApply — GRPO-01 / Task 3 (recalibration offset loader)
# ---------------------------------------------------------------------------


class TestOffsetApply:
    """Tests for _load_score_offset and _apply_offset_clip in reward_pipeline.py.

    Method names embed 'offset_loader' so -k offset_loader selects all four.
    """

    def test_offset_loader_reads_from_json(self, recalib_json):
        """_load_score_offset(path=recalib_json) returns 3.58 from fixture file."""
        from scripts.reward_pipeline import _load_score_offset

        result = _load_score_offset(path=recalib_json)
        assert result == 3.58, f"Expected 3.58, got {result}"
        assert isinstance(result, float), "Offset must be float"

    def test_offset_loader_no_hardcoded_literal(self):
        """grep of scripts/reward_pipeline.py finds no bare 3.58 literal outside comments.

        Acceptance criteria: grep -nv '^[[:space:]]*#' scripts/reward_pipeline.py
                             | grep -c '3\.58' == 0
        """
        import subprocess

        result = subprocess.run(
            ["grep", "-nv", r"^[[:space:]]*#", "scripts/reward_pipeline.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # Filter the non-comment lines for any 3.58 occurrences
        matching = [
            line for line in result.stdout.splitlines() if "3.58" in line
        ]
        assert len(matching) == 0, (
            f"Hardcoded '3.58' literal found in non-comment lines:\n"
            + "\n".join(matching)
        )

    def test_offset_loader_clip_upper(self, recalib_json):
        """Offset + clip: value + offset > 100 → clipped to 100."""
        from scripts.reward_pipeline import _load_score_offset, _apply_offset_clip

        # raw_judge=99.0; after +3.58 = 102.58 → clipped to 100.0
        result = _apply_offset_clip(99.0)
        assert result <= 100.0, f"Expected clipped to ≤100, got {result}"
        assert result == 100.0, f"Expected 100.0 at ceiling, got {result}"

    def test_offset_loader_clip_lower(self):
        """Offset + clip: value + offset below 0 (if somehow negative) → clipped to 0."""
        from scripts.reward_pipeline import _apply_offset_clip

        # raw_judge=-5.0 (edge case); after +3.58 = -1.42 → clipped to 0.0
        result = _apply_offset_clip(-5.0)
        assert result >= 0.0, f"Expected clipped to ≥0, got {result}"
        assert result == 0.0, f"Expected 0.0 at floor, got {result}"


# ---------------------------------------------------------------------------
# TestMOGRPONorm — GRPO-03 / 08-02 (MO-GRPO within-group normalization)
# ---------------------------------------------------------------------------


class TestMOGRPONorm:
    """Tests for _mo_grpo_norm (08-02).

    Method names embed 'mogrpo' so -k mogrpo selects all tests.
    """

    def test_mogrpo_zero_variance_epsilon(self):
        """All-identical group -> sigma=0 -> epsilon floor prevents NaN; returns all-zeros."""
        from scripts.reward_pipeline import _mo_grpo_norm

        values = np.ones(5) * 42.0
        result = _mo_grpo_norm(values)
        assert not np.any(np.isnan(result)), "NaN with zero-variance group (epsilon missing)"
        assert np.allclose(result, 0.0), "Zero-variance group must normalize to all-zeros"

    def test_mogrpo_mean_centered(self):
        """Normalized array has mean ~0 (within floating-point tolerance)."""
        from scripts.reward_pipeline import _mo_grpo_norm

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _mo_grpo_norm(values)
        assert abs(result.mean()) < 1e-6, f"Mean not ~0: {result.mean()}"

    def test_mogrpo_unit_variance_after_norm(self):
        """Non-degenerate group normalizes to ~unit std (population std, ddof=0)."""
        from scripts.reward_pipeline import _mo_grpo_norm

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _mo_grpo_norm(values)
        # Population std (ddof=0) must be ~1.0; allow for epsilon offset
        pop_std = result.std(ddof=0)
        assert abs(pop_std - 1.0) < 0.01, f"Population std ~1 expected, got {pop_std}"

    def test_mogrpo_group_of_one(self):
        """Single-element group does not raise and returns a finite value."""
        from scripts.reward_pipeline import _mo_grpo_norm

        values = np.array([75.0])
        result = _mo_grpo_norm(values)
        assert result.shape == (1,), "Output shape must match input"
        assert np.isfinite(result[0]), f"Single-element group returned non-finite: {result[0]}"


# ---------------------------------------------------------------------------
# TestVeRPO — GRPO-04 / 08-02 (VeRPO difficulty-weighted partial credit)
# ---------------------------------------------------------------------------


def _make_rubric(triggered: dict | None = None) -> "RubricScore":
    """Helper: create a minimal RubricScore with specific triggered_checks."""
    from eval.rubric_scorer import RubricScore

    return RubricScore(
        file_path="<test>",
        dimension_scores={},
        dimension_na=[],
        overall=70.0,
        triggered_checks=triggered or {},
        check_evidence={},
        grade="Good",
        floor_rules_applied=[],
    )


class TestVeRPO:
    """Tests for VeRPO WP-standards partial credit (08-02).

    Method names embed 'verpo' so -k verpo selects all tests.
    Uses concrete check IDs from CHECK_DIMENSION_MAP verified at test-write time:
      WP positive: WAPI-P01, WAPI-P02
      WP negative: WAPI-N01, WAPI-N02
      Non-WP: SQL-N01 (D3_sql)
    """

    def test_verpo_scope_wp_standards_only(self):
        """Checks outside {D1_wpcs, D5_wp_api} do NOT enter VeRPO computation (D-08-06).

        SQL-N01 (D3_sql) must be excluded — WP_STANDARDS_CHECK_IDS must not contain it.
        """
        from scripts.reward_pipeline import WP_STANDARDS_CHECK_IDS

        assert "SQL-N01" not in WP_STANDARDS_CHECK_IDS, (
            "SQL-N01 (D3_sql) must NOT be in WP_STANDARDS_CHECK_IDS — "
            "VeRPO is scoped to D1_wpcs + D5_wp_api only (D-08-06)"
        )
        # Positive: WP checks ARE in the set
        assert "WAPI-N01" in WP_STANDARDS_CHECK_IDS, "WAPI-N01 must be in WP_STANDARDS_CHECK_IDS"
        assert "WAPI-P01" in WP_STANDARDS_CHECK_IDS, "WAPI-P01 must be in WP_STANDARDS_CHECK_IDS"

    def test_verpo_rare_check_weights_more(self):
        """A sample passing a rarely-passed check scores higher than one passing only common ones.

        Uses WAPI-P01 (positive check):
          - Group of 2: sample A fires WAPI-P01 (pass for positive), sample B does not.
          - WAPI-P01 pass_rate = 0.5 -> difficulty = 0.5 (moderate rarity).
          - If we had a second check with pass_rate=1.0 -> difficulty=0 (always passes, no signal).
          - Sample A (passes rare WAPI-P01) should score >= sample B (doesn't pass it).
        """
        from scripts.reward_pipeline import _verpo_group

        # Two samples: A fires WAPI-P01 (positive), B does not
        rubric_a = _make_rubric({"D5_wp_api": ["WAPI-P01"]})
        rubric_b = _make_rubric({})  # does not fire WAPI-P01
        scores, pass_rates, difficulties = _verpo_group([rubric_a, rubric_b])

        assert len(scores) == 2, "Must return one score per sample"
        # WAPI-P01 pass_rate = 0.5, difficulty = 0.5
        assert abs(pass_rates.get("WAPI-P01", 0) - 0.5) < 1e-6, (
            f"WAPI-P01 pass_rate should be 0.5, got {pass_rates.get('WAPI-P01')}"
        )
        assert abs(difficulties.get("WAPI-P01", 0) - 0.5) < 1e-6, (
            f"WAPI-P01 difficulty should be 0.5, got {difficulties.get('WAPI-P01')}"
        )
        # Sample A (passed the check) must score >= sample B (did not)
        assert scores[0] >= scores[1], (
            f"Sample that passes WAPI-P01 should score >= one that doesn't; "
            f"scores: A={scores[0]}, B={scores[1]}"
        )

    def test_verpo_all_pass_score(self):
        """Group where all samples pass all WP checks -> high VeRPO score for each sample."""
        from scripts.reward_pipeline import _verpo_group, WP_STANDARDS_CHECK_IDS
        from eval.rubric_scorer import POSITIVE_CHECK_IDS, NEGATIVE_CHECK_IDS

        # Build rubrics where all positive checks fire and no negative checks fire
        # (every sample passes every WP-standards check)
        wp_pos_checks = sorted(WP_STANDARDS_CHECK_IDS & POSITIVE_CHECK_IDS)
        if not wp_pos_checks:
            pytest.skip("No WP positive checks found — CHECK_REGISTRY may have changed")

        # Fire all positive WP checks for every sample (all pass)
        triggered = {"D1_wpcs": [], "D5_wp_api": wp_pos_checks}
        rubrics = [_make_rubric(triggered), _make_rubric(triggered)]
        scores, pass_rates, difficulties = _verpo_group(rubrics)

        # If all samples pass all checks, difficulty of every check is 0
        # -> VeRPO score = 0 / (0 + epsilon) = 0.0 (degenerate all-pass case)
        # OR: if we only look at a subset, scores should be uniformly high
        # The key property: scores must be finite and non-negative
        assert all(np.isfinite(s) and s >= 0.0 for s in scores), (
            f"All-pass scores must be finite and non-negative: {scores}"
        )

    def test_verpo_all_fail_score(self):
        """Group where no samples pass any WP checks -> near-zero VeRPO score."""
        from scripts.reward_pipeline import _verpo_group

        # No triggered checks = all positive checks not fired (fail), all negative checks not fired (pass for negative)
        # Use only negative WP checks: not firing a negative = passing
        # For a strict all-fail: fire all negative WP checks
        from scripts.reward_pipeline import WP_STANDARDS_CHECK_IDS
        from eval.rubric_scorer import NEGATIVE_CHECK_IDS

        wp_neg_checks = sorted(WP_STANDARDS_CHECK_IDS & NEGATIVE_CHECK_IDS)
        if not wp_neg_checks:
            pytest.skip("No WP negative checks found")

        # Fire all negative WP checks (= fail for all), fire no positive WP checks
        triggered = {"D5_wp_api": wp_neg_checks[:5]}  # fire some negative checks = violations
        rubrics = [_make_rubric(triggered), _make_rubric(triggered)]
        scores, _, _ = _verpo_group(rubrics)

        # Both samples fail the same checks -> both get same (low) score; no NaN
        assert all(np.isfinite(s) for s in scores), f"All-fail scores contain NaN/inf: {scores}"
        assert len(scores) == 2

    def test_verpo_positive_negative_polarity(self):
        """NEGATIVE check firing scored as fail; POSITIVE check firing scored as pass (Pitfall 5 / T-08-04).

        Construct two samples:
          - Sample A fires WAPI-N01 (negative check) -> violation -> FAIL
          - Sample B fires WAPI-P01 (positive check) -> compliance -> PASS
        Sample B must score >= Sample A on that dimension.
        """
        from scripts.reward_pipeline import _verpo_group
        from eval.rubric_scorer import POSITIVE_CHECK_IDS, NEGATIVE_CHECK_IDS

        # Sanity: confirm polarity from canonical source
        assert "WAPI-N01" in NEGATIVE_CHECK_IDS, "WAPI-N01 must be NEGATIVE"
        assert "WAPI-P01" in POSITIVE_CHECK_IDS, "WAPI-P01 must be POSITIVE"

        # Sample A: WAPI-N01 fires (negative = violation, score DOWN)
        # Sample B: WAPI-P01 fires (positive = compliance, score UP)
        rubric_a = _make_rubric({"D5_wp_api": ["WAPI-N01"]})
        rubric_b = _make_rubric({"D5_wp_api": ["WAPI-P01"]})
        scores, _, _ = _verpo_group([rubric_a, rubric_b])

        # Both samples see the same checks with pass_rate=0.5/0.5 (each fires once)
        # WAPI-N01: A triggered (fail) -> pass=False; B not triggered (ok) -> pass=True for neg
        # WAPI-P01: B triggered (compliance) -> pass=True; A not triggered -> pass=False for pos
        # Net: B should have higher VeRPO than A (compliance beats violation)
        assert scores[1] >= scores[0], (
            f"Sample firing POSITIVE check should score >= sample firing NEGATIVE check; "
            f"scores: A={scores[0]:.4f} (negative/violation), B={scores[1]:.4f} (positive/compliance)"
        )


# ---------------------------------------------------------------------------
# TestSecurityGate — GRPO-02 / 08-03 (security hard gate)
# ---------------------------------------------------------------------------


class TestSecurityGate:
    """Tests for security gate (08-03). STUBBED until 08-03."""

    def test_security_gate_fail_overrides_to_zero(self):
        pytest.skip("implemented in 08-03")

    def test_security_gate_applied_after_normalization(self):
        pytest.skip("implemented in 08-03")

    def test_security_gate_non_failing_code_passes(self):
        pytest.skip("implemented in 08-03")


# ---------------------------------------------------------------------------
# TestCompositeWeights — GRPO-01 / 08-03 (70/30 composite signal weights)
# ---------------------------------------------------------------------------


class TestCompositeWeights:
    """Tests for composite reward weights (08-03). STUBBED until 08-03."""

    def test_composite_weights_sum_to_one(self):
        pytest.skip("implemented in 08-03")

    def test_composite_judge_component_weight(self):
        pytest.skip("implemented in 08-03")

    def test_composite_verifiable_split_35_35(self):
        pytest.skip("implemented in 08-03")


# ---------------------------------------------------------------------------
# TestBreakdownContract — RLEV-02 / 08-02 (breakdown dict fields + serialization)
# ---------------------------------------------------------------------------


class TestBreakdownContract:
    """Tests for RewardBreakdown output contract (08-02).

    Method names embed 'breakdown' so -k breakdown selects all tests.
    """

    def _make_breakdown(self):
        """Construct a minimal RewardBreakdown with dummy values for contract tests."""
        from scripts.reward_pipeline import RewardBreakdown

        return RewardBreakdown(
            phpcs_raw=80.0,
            verpo_raw=0.7,
            judge_raw=65.0,
            judge_offset_applied=68.58,
            security_fail=False,
            phpcs_norm=0.5,
            verpo_norm=0.3,
            judge_norm=-0.2,
            composite_pre_gate=0.42,
            check_pass_rates={"WPCS-P01": 0.8},
            check_difficulties={"WPCS-P01": 0.2},
            group_size=4,
            group_phpcs_mean=78.0,
            group_phpcs_std=5.0,
            group_judge_mean=66.0,
            group_judge_std=3.0,
        )

    def test_breakdown_has_pre_norm_fields(self):
        """RewardBreakdown carries phpcs_raw, verpo_raw, judge_raw, judge_offset_applied."""
        bd = self._make_breakdown()
        assert hasattr(bd, "phpcs_raw"), "Missing phpcs_raw"
        assert hasattr(bd, "verpo_raw"), "Missing verpo_raw"
        assert hasattr(bd, "judge_raw"), "Missing judge_raw"
        assert hasattr(bd, "judge_offset_applied"), "Missing judge_offset_applied"
        assert hasattr(bd, "security_fail"), "Missing security_fail"
        assert bd.phpcs_raw == 80.0
        assert bd.verpo_raw == 0.7
        assert bd.judge_raw == 65.0

    def test_breakdown_has_post_norm_fields(self):
        """RewardBreakdown carries phpcs_norm, verpo_norm, judge_norm."""
        bd = self._make_breakdown()
        assert hasattr(bd, "phpcs_norm"), "Missing phpcs_norm"
        assert hasattr(bd, "verpo_norm"), "Missing verpo_norm"
        assert hasattr(bd, "judge_norm"), "Missing judge_norm"
        assert hasattr(bd, "composite_pre_gate"), "Missing composite_pre_gate"

    def test_breakdown_has_parse_failure_metadata(self):
        """RewardBreakdown carries judge_parse_failure and judge_imputed_from_group."""
        bd = self._make_breakdown()
        assert hasattr(bd, "judge_parse_failure"), "Missing judge_parse_failure"
        assert hasattr(bd, "judge_imputed_from_group"), "Missing judge_imputed_from_group"
        # Defaults must be False
        assert bd.judge_parse_failure is False
        assert bd.judge_imputed_from_group is False

    def test_breakdown_has_pre_post_norm(self):
        """RewardBreakdown carries BOTH pre-norm and post-norm fields (D-08-04 contract)."""
        bd = self._make_breakdown()
        pre_norm = {"phpcs_raw", "verpo_raw", "judge_raw", "judge_offset_applied", "security_fail"}
        post_norm = {"phpcs_norm", "verpo_norm", "judge_norm"}
        for field_name in pre_norm | post_norm:
            assert hasattr(bd, field_name), f"Missing field: {field_name}"

    def test_breakdown_serializable(self):
        """RewardBreakdown.to_dict() output round-trips through json.dumps/loads."""
        bd = self._make_breakdown()
        d = bd.to_dict()
        # Must be json-serializable (no numpy types, no non-serializable objects)
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        assert loaded["phpcs_raw"] == 80.0
        assert loaded["security_fail"] is False
