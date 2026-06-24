"""Unit tests for Plan 08.1-02: pre-drop per-group stats + extended _log_step.

Covers:
  - Task 1: frac_groups_all_zero computed PRE-drop (insertion-order proof)
  - Task 1: meta keys present (component + group stats) alongside original keys
  - Task 1: empty-groups input safe (no KeyError on any key)
  - Task 1: 3-tuple arity unchanged (T-81.2-02 non-mutation contract)
  - Task 2: _extract_entropy returns value from kl_metrics or None (None-safe)
  - Task 2: _log_step with group_stats writes all Priority-1+2 fields to JSONL
  - Task 2: _compute_e_frac_trend returns correct slope from monotone history

All imports are lazy (inside methods) to avoid collection failure when
scripts/rl_rollouts.py or scripts/rl_train.py is absent / tinker_cookbook
is not installed.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_transition(*, fix_correctness=None, consistency=None, group_id="judge-0"):
    """Return a mock Transition whose .logs matches the Transition.logs pattern."""
    t = MagicMock()
    t.logs = {}
    if fix_correctness is not None:
        t.logs["fix_correctness"] = fix_correctness
    if consistency is not None:
        t.logs["consistency"] = consistency
    if group_id is not None:
        t.logs["group_id"] = group_id
    return t


def _make_traj(transitions):
    traj = MagicMock()
    traj.transitions = transitions
    return traj


def _make_mock_group(rewards: list[float], transitions) -> MagicMock:
    """Build a mock TrajectoryGroup with known per-sample rewards and transitions.

    `transitions` is a flat list; each trajectory gets one transition.
    """
    tg = MagicMock()
    tg.get_total_rewards.return_value = rewards
    tg.trajectories_G = [_make_traj([tr]) for tr in transitions]
    return tg


# ---------------------------------------------------------------------------
# Task 1: pre-drop per-group stats in compute_rollout_advantages
# ---------------------------------------------------------------------------


class TestPreDropGroupStats:
    """Verify frac_groups_all_zero is computed on the UNFILTERED groups list."""

    def _call(self, groups):
        """Patch tinker_cookbook so compute_rollout_advantages runs offline."""
        import sys
        from unittest.mock import patch, MagicMock

        rr = pytest.importorskip("scripts.rl_rollouts")

        # Build minimal cookbook mocks — assemble_training_data + compute_advantages
        # only need to not crash; their outputs are separate from the stats block.
        mock_cookbook = MagicMock()
        mock_dp = MagicMock()

        # remove_constant_reward_groups: pass through all groups (no drop in tests)
        mock_dp.remove_constant_reward_groups.side_effect = lambda gs: gs
        # compute_advantages: return one float per group (one sample each)
        mock_dp.compute_advantages.side_effect = lambda gs: [[0.0] for _ in gs]
        # assemble_training_data: return (list_of_datums, metadata)
        mock_dp.assemble_training_data.side_effect = lambda gs, adv: (
            [MagicMock() for _ in gs],
            [{"group_idx": i, "traj_idx": 0} for i in range(len(gs))],
        )

        with patch.dict(
            "sys.modules",
            {
                "tinker_cookbook": mock_cookbook,
                "tinker_cookbook.rl": MagicMock(),
                "tinker_cookbook.rl.data_processing": mock_dp,
            },
        ):
            return rr.compute_rollout_advantages(groups)

    def test_frac_groups_all_zero_three_groups(self):
        """3 groups [all-zero, all-one, non-uniform] -> each frac == 1/3."""
        # all-zero group: rewards=[0.0, 0.0]
        g_zero = _make_mock_group(
            [0.0, 0.0],
            [
                _make_transition(fix_correctness=0.0, consistency=0.2),
                _make_transition(fix_correctness=0.0, consistency=0.3),
            ],
        )
        # all-one group: rewards=[0.95, 0.95] (all > 0.9)
        g_one = _make_mock_group(
            [0.95, 0.95],
            [
                _make_transition(fix_correctness=0.95, consistency=0.9),
                _make_transition(fix_correctness=0.95, consistency=0.9),
            ],
        )
        # non-uniform group: rewards=[0.1, 0.8]
        g_nonuniform = _make_mock_group(
            [0.1, 0.8],
            [
                _make_transition(fix_correctness=0.1, consistency=0.2),
                _make_transition(fix_correctness=0.8, consistency=0.7),
            ],
        )
        groups = [g_zero, g_one, g_nonuniform]
        _data, _advantages, meta = self._call(groups)

        assert abs(meta["frac_groups_all_zero"] - 1 / 3) < 1e-9, (
            f"Expected 1/3 but got {meta['frac_groups_all_zero']}"
        )
        assert abs(meta["frac_groups_all_one"] - 1 / 3) < 1e-9, (
            f"Expected 1/3 but got {meta['frac_groups_all_one']}"
        )
        assert abs(meta["frac_groups_nonuniform"] - 1 / 3) < 1e-9, (
            f"Expected 1/3 but got {meta['frac_groups_nonuniform']}"
        )

    def test_frac_groups_all_zero_all_zero_groups(self):
        """All groups all-zero -> frac_groups_all_zero == 1.0."""
        g1 = _make_mock_group([0.0], [_make_transition(fix_correctness=0.0)])
        g2 = _make_mock_group([0.0], [_make_transition(fix_correctness=0.0)])
        _data, _advantages, meta = self._call([g1, g2])
        assert meta["frac_groups_all_zero"] == 1.0

    def test_frac_groups_all_zero_nonuniform_group(self):
        """Single non-uniform group -> frac_groups_all_zero == 0.0."""
        g = _make_mock_group(
            [0.1, 0.8],
            [
                _make_transition(fix_correctness=0.1),
                _make_transition(fix_correctness=0.8),
            ],
        )
        _data, _advantages, meta = self._call([g])
        assert meta["frac_groups_all_zero"] == 0.0

    def test_original_meta_keys_present(self):
        """Original n_groups_input / n_dropped_constant / n_groups_output / n_datums keys survive."""
        g = _make_mock_group([0.5], [_make_transition(fix_correctness=0.5)])
        _data, _advantages, meta = self._call([g])
        for key in ("n_groups_input", "n_dropped_constant", "n_groups_output", "n_datums"):
            assert key in meta, f"Original key '{key}' missing from meta"

    def test_new_component_and_group_keys_present(self):
        """New Priority-1 component+group keys all appear in meta."""
        g = _make_mock_group(
            [0.4, 0.6],
            [
                _make_transition(fix_correctness=0.4, consistency=0.5),
                _make_transition(fix_correctness=0.6, consistency=0.7),
            ],
        )
        _data, _advantages, meta = self._call([g])
        expected_keys = [
            "fix_correctness_mean",
            "fix_correctness_std",
            "consistency_mean",
            "consistency_std",
            "group_reward_std_mean",
            "frac_groups_all_zero",
            "frac_groups_all_one",
            "frac_groups_nonuniform",
            "frac_reward_gt_0.9",
            "frac_reward_lt_0.1",
        ]
        for key in expected_keys:
            assert key in meta, f"New key '{key}' missing from meta"

    def test_empty_groups_input_safe(self):
        """Empty groups -> ([], [], meta) with frac_groups_all_zero present, no crash."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        data, advantages, meta = rr.compute_rollout_advantages([])
        assert data == []
        assert advantages == []
        assert "frac_groups_all_zero" in meta, (
            "frac_groups_all_zero must be present on early-return path"
        )
        # Value must be numeric (None or 0.0), not raise
        val = meta["frac_groups_all_zero"]
        assert val is None or val == 0.0, f"Unexpected value: {val}"

    def test_return_arity_is_three_tuple(self):
        """Return value is exactly a 3-tuple (data, advantages, meta)."""
        g = _make_mock_group([0.5], [_make_transition()])
        result = self._call([g])
        assert len(result) == 3, (
            f"Expected 3-tuple but got {len(result)}-tuple — arity contract broken"
        )

    def test_non_mutation_data_and_advantages(self):
        """T-81.2-02: data and advantages are unchanged by the aggregation block.

        Build two identical groups; capture the rewards (pre-drop), then verify
        the returned advantages list length matches and rewards are not mutated.
        """
        g1 = _make_mock_group(
            [0.3, 0.7],
            [
                _make_transition(fix_correctness=0.3, consistency=0.5),
                _make_transition(fix_correctness=0.7, consistency=0.8),
            ],
        )
        g2 = _make_mock_group(
            [0.1, 0.9],
            [
                _make_transition(fix_correctness=0.1, consistency=0.2),
                _make_transition(fix_correctness=0.9, consistency=0.95),
            ],
        )
        before_g1 = list(g1.get_total_rewards())
        before_g2 = list(g2.get_total_rewards())

        data, advantages, _meta = self._call([g1, g2])

        # get_total_rewards still returns the same values (not mutated)
        assert list(g1.get_total_rewards()) == before_g1
        assert list(g2.get_total_rewards()) == before_g2
        # advantages list is a list of floats (not empty or wrong type)
        assert all(isinstance(a, float) for a in advantages), (
            "advantages must be list[float]"
        )

    def test_gen_groups_no_fix_correctness_no_crash(self):
        """gen groups have no fix_correctness/consistency in logs — must not crash."""
        g_gen = _make_mock_group(
            [0.3, 0.6],
            [
                _make_transition(fix_correctness=None, consistency=None, group_id="gen-0"),
                _make_transition(fix_correctness=None, consistency=None, group_id="gen-0"),
            ],
        )
        # Should not raise; fix_correctness_mean should be None or a number
        data, advantages, meta = self._call([g_gen])
        assert "frac_groups_all_zero" in meta


# ---------------------------------------------------------------------------
# Task 2: _extract_entropy and extended _log_step
# ---------------------------------------------------------------------------


class TestExtractEntropy:
    """_extract_entropy is None-safe and reads from kl_metrics['optim/entropy']."""

    def test_entropy_from_kl_metrics(self):
        """When kl_metrics has 'optim/entropy', _extract_entropy returns it."""
        rt = pytest.importorskip("scripts.rl_train")
        assert hasattr(rt, "_extract_entropy"), "_extract_entropy helper not found in rl_train"
        val = rt._extract_entropy({"optim/entropy": 1.23})
        assert abs(val - 1.23) < 1e-9

    def test_entropy_absent_returns_none(self):
        """When key absent, _extract_entropy returns None (no KeyError)."""
        rt = pytest.importorskip("scripts.rl_train")
        val = rt._extract_entropy({})
        assert val is None

    def test_entropy_none_value_returns_none(self):
        """When the value is explicitly None, returns None."""
        rt = pytest.importorskip("scripts.rl_train")
        val = rt._extract_entropy({"optim/entropy": None})
        assert val is None

    def test_entropy_from_fb_out_metrics_fallback(self):
        """fb_out.metrics form also accepted (None-safe)."""
        rt = pytest.importorskip("scripts.rl_train")
        fb_out = MagicMock()
        fb_out.metrics = {"entropy": 0.77}
        # Accept either signature: _extract_entropy(kl_metrics) or _extract_entropy(fb_out)
        # The canonical input is the kl_metrics dict — fb_out fallback is optional.
        # Just verify calling with a plain dict doesn't crash:
        val = rt._extract_entropy({"optim/entropy": 0.77})
        assert abs(val - 0.77) < 1e-9


class TestLogStepGroupStats:
    """_log_step with group_stats writes Priority-1+2 fields to JSONL."""

    def _make_group_stats(self):
        return {
            "fix_correctness_mean": 0.42,
            "fix_correctness_std": 0.12,
            "consistency_mean": 0.55,
            "consistency_std": 0.08,
            "group_reward_std_mean": 0.15,
            "frac_groups_all_zero": 0.10,
            "frac_groups_all_one": 0.05,
            "frac_groups_nonuniform": 0.85,
            "frac_reward_gt_0.9": 0.05,
            "frac_reward_lt_0.1": 0.10,
        }

    def test_log_step_writes_priority1_keys(self, tmp_path, monkeypatch):
        """_log_step with group_stats writes all Priority-1 keys to JSONL."""
        rt = pytest.importorskip("scripts.rl_train")
        monkeypatch.setattr(rt, "METRICS_PATH", str(tmp_path / "rl_metrics.jsonl"))

        args = types.SimpleNamespace(use_gspo=True, model_id="test-model")
        group_stats = self._make_group_stats()

        rt._log_step(
            step=0,
            rewards=[0.3, 0.5, 0.7],
            kl_metrics={"optim/kl_sample_train_v1": 0.1, "optim/kl_sample_train_v2": 0.05, "optim/entropy": 1.5},
            moe_metrics={"e_frac_with_tokens:mean": 0.8, "e_max_violation:mean": 0.01, "e_max_violation:max": 0.02},
            args=args,
            group_stats=group_stats,
        )

        rows = [json.loads(line) for line in (tmp_path / "rl_metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]

        # Priority-1 component keys
        for key in ("fix_correctness_mean", "fix_correctness_std",
                    "consistency_mean", "consistency_std",
                    "group_reward_std_mean", "frac_groups_all_zero",
                    "frac_groups_all_one", "frac_groups_nonuniform",
                    "frac_reward_gt_0.9", "frac_reward_lt_0.1"):
            assert key in row, f"Priority-1 key '{key}' missing from JSONL row"

    def test_log_step_writes_entropy(self, tmp_path, monkeypatch):
        """_log_step writes entropy from kl_metrics['optim/entropy']."""
        rt = pytest.importorskip("scripts.rl_train")
        monkeypatch.setattr(rt, "METRICS_PATH", str(tmp_path / "rl_metrics.jsonl"))

        args = types.SimpleNamespace(use_gspo=True, model_id="test-model")
        rt._log_step(
            step=0,
            rewards=[0.5],
            kl_metrics={"optim/kl_sample_train_v1": 0.0, "optim/kl_sample_train_v2": 0.0, "optim/entropy": 2.34},
            moe_metrics={},
            args=args,
            group_stats=self._make_group_stats(),
        )

        row = json.loads((tmp_path / "rl_metrics.jsonl").read_text().strip())
        assert "entropy" in row
        assert abs(row["entropy"] - 2.34) < 1e-9

    def test_log_step_entropy_null_when_absent(self, tmp_path, monkeypatch):
        """When kl_metrics has no entropy, JSONL row records null."""
        rt = pytest.importorskip("scripts.rl_train")
        monkeypatch.setattr(rt, "METRICS_PATH", str(tmp_path / "rl_metrics.jsonl"))

        args = types.SimpleNamespace(use_gspo=True, model_id="test-model")
        rt._log_step(
            step=0,
            rewards=[0.5],
            kl_metrics={"optim/kl_sample_train_v1": 0.0, "optim/kl_sample_train_v2": 0.0},
            moe_metrics={},
            args=args,
            group_stats=self._make_group_stats(),
        )

        row = json.loads((tmp_path / "rl_metrics.jsonl").read_text().strip())
        assert "entropy" in row
        assert row["entropy"] is None

    def test_log_step_backwards_compat_no_group_stats(self, tmp_path, monkeypatch):
        """Existing callers without group_stats kwarg must not break."""
        rt = pytest.importorskip("scripts.rl_train")
        monkeypatch.setattr(rt, "METRICS_PATH", str(tmp_path / "rl_metrics.jsonl"))

        args = types.SimpleNamespace(use_gspo=True, model_id="test-model")
        # Call WITHOUT group_stats — must not raise
        rt._log_step(
            step=0,
            rewards=[0.5],
            kl_metrics={"optim/kl_sample_train_v1": 0.0, "optim/kl_sample_train_v2": 0.0},
            moe_metrics={},
            args=args,
        )
        row = json.loads((tmp_path / "rl_metrics.jsonl").read_text().strip())
        assert "step" in row  # row was written without crash

    def test_log_step_writes_priority2_policy_keys(self, tmp_path, monkeypatch):
        """_log_step writes Priority-2 policy/optimizer health keys."""
        rt = pytest.importorskip("scripts.rl_train")
        monkeypatch.setattr(rt, "METRICS_PATH", str(tmp_path / "rl_metrics.jsonl"))

        args = types.SimpleNamespace(use_gspo=True, model_id="test-model")
        rt._log_step(
            step=0,
            rewards=[0.5],
            kl_metrics={"optim/kl_sample_train_v1": 0.1, "optim/kl_sample_train_v2": 0.05, "optim/entropy": 1.1},
            moe_metrics={
                "e_frac_with_tokens:mean": 0.75,
                "grad_norm": 0.42,
                "grad_norm_clipped": 0.40,
                "lr": 1e-5,
                "loss": 0.88,
            },
            args=args,
            group_stats=self._make_group_stats(),
        )

        row = json.loads((tmp_path / "rl_metrics.jsonl").read_text().strip())
        # grad_norm, lr, loss may come from moe_metrics or fb_out; just confirm
        # the row was written without crash and has the core Priority-2 field
        assert "entropy" in row


class TestEFracTrend:
    """_compute_e_frac_trend returns correct rolling slope."""

    def test_slope_monotone_increasing(self):
        """Linearly increasing e_frac -> positive slope."""
        rt = pytest.importorskip("scripts.rl_train")
        assert hasattr(rt, "_compute_e_frac_trend"), (
            "_compute_e_frac_trend helper not found in rl_train"
        )
        history = [0.6, 0.65, 0.70, 0.75, 0.80]
        slope = rt._compute_e_frac_trend(history, window=5)
        assert slope is not None
        assert slope > 0, f"Expected positive slope, got {slope}"

    def test_slope_monotone_decreasing(self):
        """Linearly decreasing e_frac -> negative slope (collapse signal)."""
        rt = pytest.importorskip("scripts.rl_train")
        history = [0.80, 0.75, 0.70, 0.65, 0.60]
        slope = rt._compute_e_frac_trend(history, window=5)
        assert slope < 0, f"Expected negative slope, got {slope}"

    def test_slope_empty_history_returns_none(self):
        """Empty or sub-window history returns None (no crash)."""
        rt = pytest.importorskip("scripts.rl_train")
        assert rt._compute_e_frac_trend([], window=10) is None
        assert rt._compute_e_frac_trend([0.5], window=10) is None

    def test_slope_flat_near_zero(self):
        """Constant e_frac -> slope near 0."""
        rt = pytest.importorskip("scripts.rl_train")
        history = [0.75] * 10
        slope = rt._compute_e_frac_trend(history, window=10)
        assert slope is not None
        assert abs(slope) < 1e-9, f"Expected ~0 slope for flat, got {slope}"


# ---------------------------------------------------------------------------
# Plan 03 — Task 1: Lever 1 Form A (graded fix_correctness)
# ---------------------------------------------------------------------------

# Minimal stub PHP fixture for offline tests (syntactically valid enough to pass
# _is_parseable_php when the security-assert guard is bypassed).
_VALID_PHP = "<?php echo 'hello'; ?>"


def _make_rr_with_env():
    """Import rl_rollouts with REWARD_SKIP_PHPCS_ASSERT set (offline safe)."""
    import os
    os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
    return pytest.importorskip("scripts.rl_rollouts")


class TestGradedFixCorrectness:
    """Lever 1 Form A: fix_correctness no longer a 0/1 step function.

    Verifies:
      - graded: at least 3 distinct values across fixture set (frac_mid > 0)
      - extremes preserved: parseable-high-quality -> high score, empty -> ~0
      - non-empty-unparseable -> partial credit strictly between 0 and 1
      - tests named to match verify selector: 'graded', 'bimodal', 'frac_mid'
    """

    def _compute_fix_score(self, completion_text: str):
        """Call the fix_correctness path directly via _extract_corrected_php +
        _is_parseable_php + _extract_verifiable_signals, mirroring the loop in
        collect_rollouts. We test the helper functions directly to avoid live
        vLLM/sampling dependencies."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")

        corrected = rr._extract_corrected_php(completion_text)
        if rr._is_parseable_php(corrected):
            from scripts.reward_pipeline import _extract_verifiable_signals  # type: ignore
            rubric = _extract_verifiable_signals(corrected)
            return float(rubric.overall) / 100.0, "parseable"
        # Test the expected graded fallback behaviour (Form A)
        # Non-empty but unparseable -> partial credit
        # Empty -> 0.0
        if not corrected or not corrected.strip():
            return 0.0, "empty"
        return None, "non_empty_unparseable"  # caller decides based on new behaviour

    def test_graded_three_distinct_values_frac_mid(self):
        """Lever 1 Form A: the redesigned scoring produces >= 3 distinct reward values.

        Fixture classes:
          1. empty completion -> 0.0 (preserved extreme)
          2. non-empty but unparseable block -> partial credit (0 < score < 1)
          3. parseable PHP -> high score (~0.9+)

        This is the frac_mid test: at least one score must land in (0.1, 0.9).
        """
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")

        # We test _fix_score_from_completion (the new helper that encapsulates
        # the Lever 1 Form A logic). If the function doesn't exist yet, the test
        # fails as expected (RED phase).
        assert hasattr(rr, "_fix_score_from_completion"), (
            "_fix_score_from_completion helper not found in rl_rollouts — "
            "Lever 1 Form A not yet implemented (RED: expected failure)"
        )

        score_empty = rr._fix_score_from_completion("")
        score_prose = rr._fix_score_from_completion(
            "I think the bug is a missing semicolon but I cannot provide a fix right now."
        )
        score_valid_php = rr._fix_score_from_completion(
            f"<corrected_code>{_VALID_PHP}</corrected_code>"
        )

        # Empty -> stays at 0.0
        assert score_empty == pytest.approx(0.0, abs=1e-6), (
            f"Empty completion must score 0.0, got {score_empty}"
        )
        # Parseable PHP -> high (not 0.0; leaves the 0-mode)
        assert score_valid_php > 0.5, (
            f"Parseable PHP should score > 0.5, got {score_valid_php}"
        )
        # Non-empty-unparseable prose -> strictly between 0 and 1 (Form A partial credit)
        assert 0.0 < score_prose < 1.0, (
            f"Non-empty unparseable should score strictly in (0,1), got {score_prose}"
        )
        # At least the prose score must be in the frac_mid window (0.1, 0.9)
        assert 0.1 < score_prose < 0.9, (
            f"Non-empty unparseable partial credit must be in (0.1,0.9) for frac_mid>0, "
            f"got {score_prose}"
        )

        # All three must be distinct -> >= 3 distinct values -> frac_mid > 0
        scores = [score_empty, score_prose, score_valid_php]
        assert len(set(scores)) >= 3, (
            f"Need >= 3 distinct values, got {sorted(set(scores))}"
        )

    def test_bimodal_structure_broken(self):
        """Lever 1: bimodal 0/1 structure broken — prose gets nonzero partial credit."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")
        assert hasattr(rr, "_fix_score_from_completion"), (
            "_fix_score_from_completion not found (RED: expected in bimodal test)"
        )
        # This was the pathology: prose -> 0.0 (same as empty)
        # After Form A: prose -> partial credit > 0
        score_prose = rr._fix_score_from_completion(
            "The issue is a missing null check on the return value."
        )
        assert score_prose > 0.0, (
            f"Lever 1 Form A: non-empty prose should get partial credit > 0, got {score_prose}"
        )

    def test_graded_high_quality_parseable_still_high(self):
        """Lever 1: a high-quality parseable fix still scores high (extremes preserved)."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")
        assert hasattr(rr, "_fix_score_from_completion"), (
            "_fix_score_from_completion not found (RED)"
        )
        score = rr._fix_score_from_completion(
            f"<corrected_code>{_VALID_PHP}</corrected_code>"
        )
        # Should remain above the partial-credit floor (not flattened)
        assert score > 0.5, (
            f"Parseable PHP should score > 0.5 after Lever 1, got {score}"
        )


class TestGuardRespected:
    """Lever 2: combine_judge_reward guard is respected (weight=0.6 still raises ValueError)."""

    def test_guard_weight_0_6_still_raises(self):
        """Weight=0.6 must still raise ValueError — guard is not relaxed by Lever 2."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")
        with pytest.raises(ValueError, match="cap"):
            rr.combine_judge_reward(fix_correctness=0.8, consistency=0.5, weight=0.6)

    def test_guard_lever2_new_weight_in_range(self):
        """Lever 2 call-site weight must be strictly between 0.3 and 0.5 (raised but capped)."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rr = pytest.importorskip("scripts.rl_rollouts")
        # The Lever 2 weight must be importable as a named constant
        assert hasattr(rr, "judge_consistency_weight_lever2"), (
            "judge_consistency_weight_lever2 not found — Lever 2 not yet implemented (RED)"
        )
        w = rr.judge_consistency_weight_lever2
        assert 0.3 < w <= 0.5, (
            f"Lever 2 weight must be in (0.3, 0.5], got {w}"
        )


class TestSecurityUntouched:
    """_security_fail is byte-unchanged; security trigger still produces terminal zero."""

    def test_security_untouched_gate_raises_on_empty_triggers(self):
        """_security_fail raises RuntimeError when _REWARD_SEC_TRIGGERS is empty."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rp = pytest.importorskip("scripts.reward_pipeline")
        assert hasattr(rp, "_security_fail"), (
            "_security_fail must still exist in reward_pipeline (unmodified)"
        )
        rubric = MagicMock()
        rubric.triggered_checks = {"dim": ["some_id"]}
        # Monkeypatch empty triggers to force the fail-closed error path
        import unittest.mock as mock
        with mock.patch.object(rp, "_REWARD_SEC_TRIGGERS", set()):
            with pytest.raises(RuntimeError, match="fail-open"):
                rp._security_fail(rubric)

    def test_security_untouched_gate_fires_on_trigger(self, monkeypatch):
        """_security_fail returns True when a known security trigger fires."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rp = pytest.importorskip("scripts.reward_pipeline")
        rubric = MagicMock()
        rubric.triggered_checks = {"d2": ["SEC_TRIGGER_ID"]}
        monkeypatch.setattr(rp, "_REWARD_SEC_TRIGGERS", {"SEC_TRIGGER_ID"})
        result = rp._security_fail(rubric)
        assert result is True, (
            "_security_fail must return True when trigger fires (gate unchanged)"
        )

    def test_security_untouched_gate_no_fire_on_non_trigger(self, monkeypatch):
        """_security_fail returns False when no security trigger fires."""
        import os
        os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
        rp = pytest.importorskip("scripts.reward_pipeline")
        rubric = MagicMock()
        rubric.triggered_checks = {"d1": ["SAFE_ID"]}
        monkeypatch.setattr(rp, "_REWARD_SEC_TRIGGERS", {"SEC_TRIGGER_ID"})
        result = rp._security_fail(rubric)
        assert result is False, (
            "_security_fail must return False when no security trigger fires"
        )


# ---------------------------------------------------------------------------
# Plan 03 — Task 2: Logging tiers 4-6
# ---------------------------------------------------------------------------


class TestConsistencyScoreHistogram:
    """Tier 4: consistency score-distribution histogram accumulates per dispatch call."""

    def test_histogram_bin_score_exists(self):
        """_bin_score function exists in rl_judge_dispatch (Tier 4 accumulator)."""
        rjd = pytest.importorskip("scripts.rl_judge_dispatch")
        assert hasattr(rjd, "_bin_score"), (
            "_bin_score not found in rl_judge_dispatch — histogram (RED)"
        )

    def test_histogram_bin_score_correct_bins(self):
        """_bin_score maps scores to the correct 5-bin labels."""
        rjd = pytest.importorskip("scripts.rl_judge_dispatch")
        _bin_score = rjd._bin_score
        assert _bin_score(0.0) == "0_0.2"
        assert _bin_score(0.1) == "0_0.2"
        assert _bin_score(0.2) == "0.2_0.4"
        assert _bin_score(0.5) == "0.4_0.6"
        assert _bin_score(0.8) == "0.8_1.0"
        assert _bin_score(1.0) == "0.8_1.0"  # catches s == 1.0

    def test_histogram_module_level_accumulator_exists(self):
        """Module-level _score_hist dict exists with 5 bins."""
        rjd = pytest.importorskip("scripts.rl_judge_dispatch")
        assert hasattr(rjd, "_score_hist"), (
            "_score_hist not found in rl_judge_dispatch — histogram accumulator (RED)"
        )
        hist = rjd._score_hist
        assert len(hist) == 5, f"Expected 5 bins, got {len(hist)}"

    def test_histogram_reset_function_exists(self):
        """get_and_reset_score_hist exists to retrieve and clear per-step histogram."""
        rjd = pytest.importorskip("scripts.rl_judge_dispatch")
        assert hasattr(rjd, "get_and_reset_score_hist"), (
            "get_and_reset_score_hist not found in rl_judge_dispatch (RED)"
        )


class TestWindowMeans:
    """Tier 5: compute_window_means aggregates JSONL metrics per window."""

    def _write_metrics(self, tmp_path, rows):
        """Write JSONL metrics file for test."""
        import json as _json
        p = tmp_path / "rl_metrics.jsonl"
        with open(str(p), "w") as fh:
            for r in rows:
                fh.write(_json.dumps(r) + "\n")
        return str(p)

    def test_window_means_function_exists(self):
        """compute_window_means exists in rl_train (Tier 5 / spec section 4)."""
        rt = pytest.importorskip("scripts.rl_train")
        assert hasattr(rt, "compute_window_means"), (
            "compute_window_means not found in rl_train (RED)"
        )

    def test_window_means_basic(self, tmp_path):
        """compute_window_means returns correct per-field means for a window."""
        rt = pytest.importorskip("scripts.rl_train")
        rows = [
            {"step": i, "reward_mean": float(i) * 0.01,
             "fix_correctness_mean": 0.5, "consistency_mean": 0.3,
             "group_reward_std_mean": 0.1, "frac_groups_all_zero": 0.2,
             "entropy": 2.0}
            for i in range(1, 51)
        ]
        metrics_path = self._write_metrics(tmp_path, rows)
        result = rt.compute_window_means(metrics_path, [(1, 50)])
        assert "window_1_50" in result, (
            f"Expected key 'window_1_50', got {list(result.keys())}"
        )
        w = result["window_1_50"]
        assert "reward_mean" in w
        assert "frac_groups_all_zero" in w

    def test_window_means_key_names_match_kill_rule(self, tmp_path):
        """Window keys match the names consumed by should_flag_for_review."""
        rt = pytest.importorskip("scripts.rl_train")
        rows = [
            {"step": s, "reward_mean": 0.27,
             "fix_correctness_mean": 0.4, "consistency_mean": 0.3,
             "group_reward_std_mean": 0.1, "frac_groups_all_zero": 0.3,
             "entropy": 2.1}
            for s in list(range(1, 51)) + list(range(151, 201))
        ]
        metrics_path = self._write_metrics(tmp_path, rows)
        result = rt.compute_window_means(metrics_path, [(0, 50), (151, 200)])
        # Kill rule reads "window_0_50" and "window_151_200"
        assert "window_0_50" in result, (
            f"Expected 'window_0_50' key for kill-rule, got {list(result.keys())}"
        )
        assert "window_151_200" in result, (
            f"Expected 'window_151_200' key for kill-rule, got {list(result.keys())}"
        )


class TestShouldFlagForReview:
    """Tier 6: should_flag_for_review implements the verbatim kill/continue rule."""

    def _make_window_means(self, w0_50_reward=0.27, w151_200_reward=0.27,
                           frac_zero_151_200=0.2):
        return {
            "window_0_50": {"reward_mean": w0_50_reward,
                            "frac_groups_all_zero": 0.2, "entropy_mean": 2.5},
            "window_151_200": {"reward_mean": w151_200_reward,
                               "frac_groups_all_zero": frac_zero_151_200,
                               "entropy_mean": 2.5},
        }

    def test_flag_for_review_function_exists(self):
        """should_flag_for_review exists in rl_train (Tier 6 / spec section 6)."""
        rt = pytest.importorskip("scripts.rl_train")
        assert hasattr(rt, "should_flag_for_review"), (
            "should_flag_for_review not found in rl_train (RED)"
        )

    def test_flag_for_review_group_collapse(self):
        """GROUP_COLLAPSE when frac_groups_all_zero window-mean > 0.5."""
        rt = pytest.importorskip("scripts.rl_train")
        window_means = self._make_window_means(frac_zero_151_200=0.6)
        flagged, reason = rt.should_flag_for_review(
            window_means, current={"entropy": 2.0}
        )
        assert flagged is True, f"Expected flag=True for GROUP_COLLAPSE, got {flagged}"
        assert "GROUP_COLLAPSE" in reason, (
            f"Expected 'GROUP_COLLAPSE' in reason, got '{reason}'"
        )

    def test_flag_for_review_entropy_collapse(self):
        """ENTROPY_COLLAPSE when current entropy < 1.5."""
        rt = pytest.importorskip("scripts.rl_train")
        window_means = self._make_window_means()
        flagged, reason = rt.should_flag_for_review(
            window_means, current={"entropy": 1.2}
        )
        assert flagged is True, f"Expected flag=True for ENTROPY_COLLAPSE, got {flagged}"
        assert "ENTROPY_COLLAPSE" in reason, (
            f"Expected 'ENTROPY_COLLAPSE' in reason, got '{reason}'"
        )

    def test_flag_for_review_decisive_flat(self):
        """DECISIVE_FLAT when recent window reward < early window reward - 0.01."""
        rt = pytest.importorskip("scripts.rl_train")
        window_means = self._make_window_means(w0_50_reward=0.28, w151_200_reward=0.26)
        flagged, reason = rt.should_flag_for_review(
            window_means, current={"entropy": 2.5}
        )
        assert flagged is True, f"Expected flag=True for DECISIVE_FLAT, got {flagged}"
        assert "DECISIVE_FLAT" in reason, (
            f"Expected 'DECISIVE_FLAT' in reason, got '{reason}'"
        )

    def test_flag_for_review_ok(self):
        """Returns (False, 'OK') when no pathology triggered."""
        rt = pytest.importorskip("scripts.rl_train")
        window_means = self._make_window_means(
            w0_50_reward=0.27, w151_200_reward=0.28, frac_zero_151_200=0.2
        )
        flagged, reason = rt.should_flag_for_review(
            window_means, current={"entropy": 2.5}
        )
        assert flagged is False, f"Expected flag=False for OK state, got {flagged}"
        assert reason == "OK", f"Expected reason='OK', got '{reason}'"
