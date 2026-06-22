"""DATUM-01/02 offline schema guard for the GSPO datum/logprob assembly (09-07).

Proves the cookbook trajectory -> Datum pipeline bakes the four loss_fn_inputs keys
(target_tokens, logprobs, advantages, mask) with REAL sampled logprobs and a mask that
selects action positions — the contract the rl_rollouts/rl_train refactor depends on.

No Tinker training client, no weights, no reward pipeline — runs in <1s in .venv-tinker.
The original blocker (forward_backward_custom handed plain dicts -> AttributeError, with the
loss falling into its seq_ratio=1.0 fallback) is dead only if real Datums carry real logprobs;
these tests are the unit-speed proof of the cookbook half of that contract.
"""
from __future__ import annotations

import pytest

# The cookbook + tinker live in .venv-tinker; skip cleanly if absent (base conda has no tinker).
pytest.importorskip("tinker")

import tinker  # noqa: E402
from tinker_cookbook.completers import TokensWithLogprobs  # noqa: E402
from tinker_cookbook.rl.data_processing import (  # noqa: E402
    assemble_training_data,
    compute_advantages,
    trajectory_to_data,
)
from tinker_cookbook.rl.types import Trajectory, Transition, TrajectoryGroup  # noqa: E402


def _make_traj(
    obs_tokens: list[int],
    action_tokens: list[int],
    logprobs: list[float],
    reward: float,
) -> Trajectory:
    """Build a single-transition Trajectory (one WP completion = 1 Transition)."""
    return Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints(obs_tokens),
                ac=TokensWithLogprobs(tokens=action_tokens, maybe_logprobs=logprobs),
                reward=reward,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


def test_trajectory_to_datum_schema():
    """DATUM-01: a Datum carries all four loss_fn_inputs keys; logprobs are real (not all-zero)."""
    traj = _make_traj(
        obs_tokens=[10, 11],
        action_tokens=[20, 21, 22],
        logprobs=[-0.5, -1.25, -0.75],  # distinct non-zero sampled logprobs
        reward=1.0,
    )
    data = trajectory_to_data(traj, traj_advantage=0.4)
    assert len(data) == 1
    lfi = data[0].loss_fn_inputs
    for key in ("target_tokens", "logprobs", "advantages", "mask"):
        assert key in lfi, f"missing loss_fn_inputs key: {key}"

    logprobs_tensor = lfi["logprobs"].to_torch()
    # Real sampled values must flow — an all-zero logprobs tensor means the fallback path,
    # which is exactly the bug this plan kills.
    assert logprobs_tensor.abs().sum().item() > 0.0, "logprobs tensor is all-zero (fallback)"

    mask_tensor = lfi["mask"].to_torch()
    assert (mask_tensor == 1.0).any().item(), "mask has no action (1.0) position"


def test_datum_assembly_len_matches_advantages():
    """DATUM-02: assemble_training_data yields one Datum per trajectory, meta parallel to data."""
    tg = TrajectoryGroup(
        trajectories_G=[
            _make_traj([10, 11], [20, 21], [-0.3, -0.6], reward=0.3),
            _make_traj([10, 11], [30, 31], [-0.4, -0.9], reward=0.7),
        ],
        final_rewards_G=[0.0, 0.0],
        metrics_G=[{}, {}],
    )
    advantages_P = compute_advantages([tg])
    data_D, metadata_D = assemble_training_data([tg], advantages_P)
    assert len(data_D) == 2
    assert len(data_D) == len(metadata_D)


def test_zero_advantage_member_is_safe():
    """A legitimately-centered 0.0 advantage member still has a 1.0 mask position.

    Rewards [0.2,0.4,0.6] center to advantages [-0.2, 0.0, +0.2]. The middle Datum's
    advantage is exactly 0.0, but the loss fn must recover it via mask==1 (NOT via
    advantages!=0, which would select nothing and NaN). This guards the Task 3 footgun.
    """
    tg = TrajectoryGroup(
        trajectories_G=[
            _make_traj([10, 11], [20, 21], [-0.3, -0.6], reward=0.2),
            _make_traj([10, 11], [30, 31], [-0.4, -0.9], reward=0.4),
            _make_traj([10, 11], [40, 41], [-0.5, -0.8], reward=0.6),
        ],
        final_rewards_G=[0.0, 0.0, 0.0],
        metrics_G=[{}, {}, {}],
    )
    advantages_P = compute_advantages([tg])
    # Per-group centering -> middle advantage is exactly 0.0.
    assert abs(float(advantages_P[0][1])) < 1e-9
    data_D, _meta = assemble_training_data([tg], advantages_P)
    middle_mask = data_D[1].loss_fn_inputs["mask"].to_torch()
    assert (middle_mask == 1.0).any().item(), "zero-advantage member lost its action mask"


def test_empty_completion_mask_is_safe():
    """An immediate-EOS completion (tokens=[]) yields a Datum with NO 1.0 mask positions.

    The Task 3 loss fn must zero (not crash on) such a datum: adv_weights[mask == 1] is an
    empty tensor, so a [0] index would IndexError without the numel() guard.
    """
    traj = _make_traj(
        obs_tokens=[10, 11, 12],
        action_tokens=[],
        logprobs=[],
        reward=0.5,
    )
    data = trajectory_to_data(traj, traj_advantage=0.4)
    assert len(data) == 1
    mask_tensor = data[0].loss_fn_inputs["mask"].to_torch()
    assert not (mask_tensor == 1.0).any().item(), "empty completion unexpectedly has a 1.0 mask"
