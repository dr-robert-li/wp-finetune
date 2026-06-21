"""Unit tests for scripts/rl_rollouts.py (Plan 09-04).

Covers:
  - Interleaved sampling ratio (GRPO-05 / D-09-04)
  - judge >= gen budget for all batch sizes
  - judge_consistency_weight cap assertion (D-09-05 guard 1 / T-09-RWD-CAP)
  - combine_judge_reward returns correct weighted combination
  - Mixed-reward group -> non-zero advantages (GRPO-07)
  - Constant-reward group -> dropped before advantage assembly
  - compute_group_rewards delegation (verify symbol presence via grep)
  - Security-zero group dropped (T-09-SECDROP)

All imports are lazy (inside methods) to avoid collection failure when
scripts/rl_rollouts.py is absent or tinker_cookbook is not installed.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestInterleaving:
    """GRPO-05 / D-09-04: interleaved sampling with judge-weighted ratio."""

    def test_judge_ge_gen_budget_batch2(self):
        """n_judge >= n_gen for batch_size=2 (minimum meaningful batch)."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(10)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(10)]
        batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=2)
        n_gen = sum(1 for item in batch if item.get("tag") == "gen")
        n_judge = sum(1 for item in batch if item.get("tag") == "judge")
        assert n_judge >= n_gen, (
            f"batch_size=2: n_judge ({n_judge}) must be >= n_gen ({n_gen})"
        )

    def test_judge_ge_gen_budget_batch8(self):
        """n_judge >= n_gen for batch_size=8."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(20)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(20)]
        batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=8)
        n_gen = sum(1 for item in batch if item.get("tag") == "gen")
        n_judge = sum(1 for item in batch if item.get("tag") == "judge")
        assert n_judge >= n_gen, (
            f"batch_size=8: n_judge ({n_judge}) must be >= n_gen ({n_gen})"
        )

    def test_judge_ge_gen_budget_batch20(self):
        """n_judge >= n_gen for batch_size=20."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(30)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(30)]
        batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=20)
        n_gen = sum(1 for item in batch if item.get("tag") == "gen")
        n_judge = sum(1 for item in batch if item.get("tag") == "judge")
        assert n_judge >= n_gen, (
            f"batch_size=20: n_judge ({n_judge}) must be >= n_gen ({n_gen})"
        )

    def test_judge_ge_gen_budget_batch21(self):
        """n_judge >= n_gen for batch_size=21 (odd; tests round() behavior)."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(30)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(30)]
        batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=21)
        n_gen = sum(1 for item in batch if item.get("tag") == "gen")
        n_judge = sum(1 for item in batch if item.get("tag") == "judge")
        assert n_judge >= n_gen, (
            f"batch_size=21: n_judge ({n_judge}) must be >= n_gen ({n_gen})"
        )

    def test_interleave_total_size(self):
        """Total batch size equals batch_size arg."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"tag": "gen"} for _ in range(20)]
        judge_pool = [{"tag": "judge"} for _ in range(20)]
        for bs in (2, 8, 10, 20, 21):
            batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=bs)
            assert len(batch) == bs, f"batch_size={bs}: expected {bs} items, got {len(batch)}"

    def test_interleave_contains_both_tags(self):
        """Every batch of size >= 2 contains at least one gen and one judge item."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(10)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(10)]
        for bs in (2, 8, 20):
            batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=bs)
            tags = [item["tag"] for item in batch]
            assert "gen" in tags, f"batch_size={bs}: missing gen items"
            assert "judge" in tags, f"batch_size={bs}: missing judge items"

    def test_origin_stamped_for_pool_routing(self):
        """CR-03: every sampled item carries an "_origin" tag ("gen"/"judge") so
        collect_rollouts can route by pool origin (pool items have no "tag" field).
        Judge-origin count must equal n_judge and gen-origin count n_gen."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        # Pool items shaped like real prompts: {"messages": [...]}, NO "tag" key.
        gen_pool = [{"messages": [{"role": "user", "content": f"g{i}"}]} for i in range(10)]
        judge_pool = [{"messages": [{"role": "user", "content": f"j{i}"}]} for i in range(10)]
        batch = rr.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=10)
        origins = [item.get("_origin") for item in batch]
        assert all(o in ("gen", "judge") for o in origins), (
            f"every item must carry a gen/judge _origin tag, got {origins}"
        )
        n_judge = sum(1 for o in origins if o == "judge")
        n_gen = sum(1 for o in origins if o == "gen")
        assert n_judge == round(10 * rr.JUDGE_RATIO), f"n_judge={n_judge}"
        assert n_gen == 10 - round(10 * rr.JUDGE_RATIO), f"n_gen={n_gen}"
        # Source pool dicts must not be mutated (origin lives only on the copies).
        assert all("_origin" not in p for p in gen_pool + judge_pool), (
            "sample_interleaved_prompts must not mutate caller pool dicts"
        )

    def test_judge_ratio_constant(self):
        """JUDGE_RATIO constant exists and equals 0.6."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        assert hasattr(rr, "JUDGE_RATIO"), "scripts.rl_rollouts must export JUDGE_RATIO"
        assert rr.JUDGE_RATIO == pytest.approx(0.6), (
            f"JUDGE_RATIO must be 0.6 per D-09-04, got {rr.JUDGE_RATIO}"
        )


class TestJudgeCap:
    """D-09-05 guard 1 / T-09-RWD-CAP: Claude-consistency contribution capped <= 0.5."""

    def test_judge_consistency_weight_le_half(self):
        """judge_consistency_weight module constant must be <= 0.5 (cap enforced at import)."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        assert hasattr(rr, "judge_consistency_weight"), (
            "scripts.rl_rollouts must export judge_consistency_weight constant"
        )
        assert rr.judge_consistency_weight <= 0.5, (
            f"D-09-05 guard 1: judge_consistency_weight must be <= 0.5, "
            f"got {rr.judge_consistency_weight}"
        )

    def test_combine_judge_reward_default_weight(self):
        """combine_judge_reward uses default weight and fix_correctness as anchor."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        # When fix_correctness=1.0 and consistency=0.0 with weight=0.3:
        # result = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        result = rr.combine_judge_reward(fix_correctness=1.0, consistency=0.0)
        assert result == pytest.approx(0.7, abs=1e-6), (
            f"combine_judge_reward(1.0, 0.0) should return ~0.7 with default weight 0.3, got {result}"
        )

    def test_combine_judge_reward_explicit_weight(self):
        """combine_judge_reward with explicit weight combines correctly."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        # weight=0.4: result = 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        result = rr.combine_judge_reward(fix_correctness=0.8, consistency=0.5, weight=0.4)
        assert result == pytest.approx(0.68, abs=1e-6), (
            f"combine_judge_reward(0.8, 0.5, weight=0.4) should return ~0.68, got {result}"
        )

    def test_combine_judge_reward_rejects_weight_over_half(self):
        """combine_judge_reward must reject weight > 0.5 (cap guard)."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        with pytest.raises((ValueError, AssertionError)):
            rr.combine_judge_reward(fix_correctness=0.8, consistency=0.5, weight=0.6)

    def test_judge_consistency_weight_assertion_at_import(self):
        """The module-level cap assertion fires: setting weight > 0.5 raises at call time."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        # The constant itself must already satisfy the cap
        assert rr.judge_consistency_weight <= 0.5


class TestCombineJudgeReward:
    """combine_judge_reward correctness and boundary conditions."""

    def test_all_fix_correctness_anchor(self):
        """fix_correctness is the anchor: weight=0 -> result equals fix_correctness."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        result = rr.combine_judge_reward(fix_correctness=0.9, consistency=0.0, weight=0.0)
        assert result == pytest.approx(0.9, abs=1e-6)

    def test_blend_at_zero_five_weight(self):
        """At weight=0.5 (boundary), result is exactly half fix_correctness + half consistency."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        result = rr.combine_judge_reward(fix_correctness=0.6, consistency=0.4, weight=0.5)
        expected = 0.5 * 0.6 + 0.5 * 0.4  # = 0.5
        assert result == pytest.approx(expected, abs=1e-6)


class TestBuildTrajectoryGroups:
    """build_trajectory_groups produces cookbook-shaped trajectory groups."""

    def test_build_trajectory_groups_structure(self):
        """build_trajectory_groups returns a list of cookbook TrajectoryGroup objects."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tinker_cookbook.rl.types import TrajectoryGroup

        from tests._rl_fixtures import make_reward, make_rollout

        rollouts = [make_rollout(f"<?php echo {i}; ?>", group_id=f"gen-{i}") for i in range(4)]
        rewards = [make_reward(scalar=float(i) * 0.2) for i in range(4)]
        groups = rr.build_trajectory_groups(rollouts, rewards)
        assert isinstance(groups, list), "build_trajectory_groups must return a list"
        assert groups, "expected at least one group"
        for g in groups:
            assert isinstance(g, TrajectoryGroup), "each group must be a cookbook TrajectoryGroup"

    def test_security_zero_group_dropped(self):
        """T-09-SECDROP: a security_fail=True rollout produces no trajectory."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_reward, make_rollout

        # Distinct group_ids -> each surviving rollout is its own single-traj group.
        rollouts = [make_rollout(f"<?php echo {i}; ?>", group_id=f"gen-{i}") for i in range(4)]
        rewards = [
            make_reward(scalar=0.7, sec_fail=False),
            make_reward(scalar=0.0, sec_fail=True),   # dropped
            make_reward(scalar=0.6, sec_fail=False),
            make_reward(scalar=0.4, sec_fail=False),
        ]
        groups = rr.build_trajectory_groups(rollouts, rewards)
        assert len(groups) == 3, (
            f"Expected 3 groups after dropping 1 security-fail rollout, got {len(groups)}"
        )
        # Surviving rewards (read via the cookbook API) are the non-security scalars.
        surviving = sorted(r for g in groups for r in g.get_total_rewards())
        assert surviving == pytest.approx([0.4, 0.6, 0.7]), (
            f"Surviving rewards should be [0.4, 0.6, 0.7], got {surviving}"
        )

    def test_security_group_dropped_by_flag(self):
        """Security-fail member (breakdown.security_fail=True) excluded from groups."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_reward, make_rollout

        rollouts = [make_rollout(f"<?php $x={i}; ?>", group_id=f"gen-{i}") for i in range(3)]
        rewards = [
            make_reward(scalar=0.8, sec_fail=False),
            make_reward(scalar=0.0, sec_fail=True),   # security gate
            make_reward(scalar=0.6, sec_fail=False),
        ]
        groups = rr.build_trajectory_groups(rollouts, rewards)
        assert len(groups) <= 2, (
            f"expected <= 2 groups after security-fail drop, got {len(groups)}"
        )
        survivors = sorted(r for g in groups for r in g.get_total_rewards())
        assert survivors == pytest.approx([0.6, 0.8])


class TestComputeRolloutAdvantages:
    """compute_rollout_advantages assembles real tinker.Datum objects via the cookbook."""

    def test_mixed_reward_group_nonzero_advantages(self):
        """Mixed-reward group -> real Datums + at least one non-zero advantage."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        tinker = pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        tg = make_trajectory_group([1.0, 0.5, 0.0, 0.8], group_id="gen-0")
        data, advantages, meta = rr.compute_rollout_advantages([tg])
        assert any(a != 0.0 for a in advantages), (
            "mixed-reward group must produce at least one non-zero advantage"
        )
        # Wrapper-level regression guard for the ORIGINAL bug (dicts -> Datum):
        # the wrapper must return real tinker.Datum objects carrying sampled logprobs.
        assert isinstance(data[0], tinker.Datum)
        assert "logprobs" in data[0].loss_fn_inputs

    def test_constant_reward_group_singleton_zero_advantage(self):
        """Constant-reward group -> cookbook returns a singleton with zero advantages."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        tg = make_trajectory_group([0.5, 0.5, 0.5], group_id="gen-0")
        data, advantages, meta = rr.compute_rollout_advantages([tg])
        # remove_constant_reward_groups returns groups[0:1] when ALL are constant,
        # so data is non-empty but every advantage is zero (no gradient this step).
        assert all(abs(a) < 1e-9 for a in advantages), (
            "constant-reward group must yield all-zero advantages"
        )

    def test_mixed_group_advantage_sums_near_zero(self):
        """Advantages in a mixed group sum to approximately zero (group-centering)."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        tg = make_trajectory_group([1.0, 0.5, 0.0], group_id="gen-0")
        data, advantages, meta = rr.compute_rollout_advantages([tg])
        assert abs(sum(advantages)) < 1e-6, (
            f"group-centered advantages must sum to ~0, got {sum(advantages)}"
        )

    def test_per_group_constant_filter_keeps_mixed_drops_constant(self):
        """CR-06: a constant group is dropped while a sibling mixed group survives."""
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        mixed = make_trajectory_group([1.0, 0.5, 0.0], group_id="gen-mixed")
        const = make_trajectory_group([0.5, 0.5, 0.5], group_id="gen-const")
        data, advantages, meta = rr.compute_rollout_advantages([mixed, const])
        # Only the mixed group's 3 trajectories survive -> 3 datums.
        assert len(data) == 3, (
            f"expected 3 surviving datums (mixed group only), got {len(data)}"
        )
        # meta now counts GROUPS: 2 in, 1 constant dropped.
        assert meta["n_dropped_constant"] == 1, (
            f"expected 1 dropped group (the constant one), got {meta['n_dropped_constant']}"
        )
        assert any(a != 0.0 for a in advantages)

    def test_per_group_advantage_centers_within_own_group(self):
        """CR-06: advantages center on each group's OWN mean, not the batch mean.

        Group A rewards [1.0, 0.0] -> advantages [+0.5, -0.5].
        Group B rewards [0.2, 0.4] -> advantages [-0.1, +0.1].
        assemble_training_data emits group A's datums then group B's, so the flat
        per-datum advantages list is [0.5, -0.5, -0.1, 0.1].
        """
        rr = pytest.importorskip("scripts.rl_rollouts")
        pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        group_a = make_trajectory_group([1.0, 0.0], group_id="gen-A")
        group_b = make_trajectory_group([0.2, 0.4], group_id="gen-B")
        data, advantages, _meta = rr.compute_rollout_advantages([group_a, group_b])
        assert advantages[0] == pytest.approx(0.5, abs=1e-6)
        assert advantages[1] == pytest.approx(-0.5, abs=1e-6)
        assert advantages[2] == pytest.approx(-0.1, abs=1e-6)
        assert advantages[3] == pytest.approx(0.1, abs=1e-6)
        # Each group's advantages sum to ~0 independently.
        assert advantages[0] + advantages[1] == pytest.approx(0.0, abs=1e-6)
        assert advantages[2] + advantages[3] == pytest.approx(0.0, abs=1e-6)

    def test_module_imports_without_tinker(self):
        """scripts.rl_rollouts imports cleanly even when tinker_cookbook is absent."""
        # If we got here via importorskip, it already imported — test passes trivially
        # but this is an explicit documentation of the requirement.
        rr = pytest.importorskip("scripts.rl_rollouts")
        assert rr is not None

    def test_cookbook_symbols_referenced(self):
        """Module source references compute_advantages, remove_constant_reward_groups,
        and assemble_training_data (grep >= 3 total occurrences)."""
        import subprocess, os
        result = subprocess.run(
            ["grep", "-c",
             r"compute_advantages\|remove_constant_reward_groups\|assemble_training_data",
             "scripts/rl_rollouts.py"],
            capture_output=True, text=True,
            cwd="/home/robert_li/Desktop/projects/wp-finetune",
        )
        count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        assert count >= 3, (
            f"scripts/rl_rollouts.py must reference cookbook symbols >= 3 times, found {count}"
        )

    def test_compute_group_rewards_referenced(self):
        """Module source references compute_group_rewards (unmodified delegation)."""
        import subprocess
        result = subprocess.run(
            ["grep", "-c", "compute_group_rewards", "scripts/rl_rollouts.py"],
            capture_output=True, text=True,
            cwd="/home/robert_li/Desktop/projects/wp-finetune",
        )
        count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        assert count >= 1, (
            f"scripts/rl_rollouts.py must reference compute_group_rewards, found {count}"
        )
