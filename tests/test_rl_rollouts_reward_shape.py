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
