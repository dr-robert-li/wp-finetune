"""Shared cookbook-RL test fixtures (09-07).

Builds REAL tinker_cookbook Trajectory/TrajectoryGroup objects so the rl_rollouts /
rl_train tests assert against the actual Datum contract — no synthetic-dict shim.
Import these only inside test bodies guarded by pytest.importorskip("tinker"); the
helpers themselves import tinker lazily so this module is importable without it.
"""
from __future__ import annotations

from typing import Any


def make_trajectory_group(
    rewards: list[float],
    group_id: str = "grp",
    logs_extra: list[dict] | None = None,
) -> Any:
    """Build a TrajectoryGroup of single-turn trajectories, one per reward.

    Each trajectory is one Transition (ob = a 2-token ModelInput, ac = 2 sampled
    tokens with distinct non-zero logprobs, reward, episode_done=True). group_id and
    origin (group_id before the first '-') are stamped into Transition.logs, mirroring
    build_trajectory_groups, so panickssery / origin-survival checks can read them back.
    """
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Trajectory, Transition, TrajectoryGroup

    trajectories = []
    for i, r in enumerate(rewards):
        logs: dict = {"group_id": str(group_id), "origin": str(group_id).split("-")[0]}
        if logs_extra and i < len(logs_extra):
            logs.update(logs_extra[i])
        transition = Transition(
            ob=tinker.ModelInput.from_ints([10, 11]),
            ac=TokensWithLogprobs(tokens=[20 + i, 21 + i], maybe_logprobs=[-0.3, -0.6]),
            reward=float(r),
            episode_done=True,
            logs=logs,
        )
        trajectories.append(
            Trajectory(transitions=[transition], final_ob=tinker.ModelInput.from_ints([]))
        )
    n = len(trajectories)
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0] * n,
        metrics_G=[{} for _ in range(n)],
    )


def make_rollout(
    completion: str,
    group_id: str,
    n_tokens: int = 2,
) -> Any:
    """Build a real rl_rollouts._Completion carrying tokens/logprobs/model_input."""
    import tinker

    import scripts.rl_rollouts as rl_rollouts

    base = abs(hash(completion)) % 1000
    tokens = [base + j for j in range(n_tokens)]
    logprobs = [-0.3 - 0.1 * j for j in range(n_tokens)]
    return rl_rollouts._Completion(
        completion=completion,
        group_id=group_id,
        model_input=tinker.ModelInput.from_ints([10, 11]),
        tokens=tokens,
        logprobs=logprobs,
    )


def make_reward(
    scalar: float,
    sec_fail: bool = False,
    fix_correctness: float | None = None,
    consistency: float | None = None,
) -> Any:
    """Build a RewardResult-like object (.scalar + .breakdown.security_fail)."""
    import types

    breakdown = types.SimpleNamespace(
        security_fail=sec_fail,
        fix_correctness=fix_correctness,
        consistency=consistency,
    )
    return types.SimpleNamespace(scalar=scalar, breakdown=breakdown)
