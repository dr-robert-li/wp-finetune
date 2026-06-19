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
    """Tests for _mo_grpo_norm (08-02). STUBBED until 08-02."""

    def test_mogrpo_zero_variance_epsilon(self):
        pytest.skip("implemented in 08-02")

    def test_mogrpo_mean_centered(self):
        pytest.skip("implemented in 08-02")

    def test_mogrpo_unit_variance_after_norm(self):
        pytest.skip("implemented in 08-02")


# ---------------------------------------------------------------------------
# TestVeRPO — GRPO-04 / 08-02 (VeRPO difficulty-weighted partial credit)
# ---------------------------------------------------------------------------


class TestVeRPO:
    """Tests for VeRPO WP-standards partial credit (08-02). STUBBED until 08-02."""

    def test_verpo_wpcs_subset_only(self):
        pytest.skip("implemented in 08-02")

    def test_verpo_difficulty_weight_inverse_pass_rate(self):
        pytest.skip("implemented in 08-02")

    def test_verpo_all_pass_gives_zero_difficulty(self):
        pytest.skip("implemented in 08-02")


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
# TestBreakdownContract — RLEV-02 / 08-03 (breakdown dict fields)
# ---------------------------------------------------------------------------


class TestBreakdownContract:
    """Tests for RewardBreakdown output contract (08-03). STUBBED until 08-03."""

    def test_breakdown_has_pre_norm_fields(self):
        pytest.skip("implemented in 08-03")

    def test_breakdown_has_post_norm_fields(self):
        pytest.skip("implemented in 08-03")

    def test_breakdown_has_parse_failure_metadata(self):
        pytest.skip("implemented in 08-03")
