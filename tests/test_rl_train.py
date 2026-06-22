"""Wave-0 test contract for Phase 9 GSPO/GRPO training scripts.

Encodes behavioral contracts GRPO-05, GRPO-06, GRPO-07, GRPO-08 as named
importorskip-guarded stubs. All stubs SKIP while scripts.rl_train /
scripts.rl_rollouts are absent (plans 09-03/04/05 write those modules).
Once the modules land, each test becomes RED on wrong/missing symbol and
GREEN only when the seam is correctly implemented.

File-ownership rule: plans 09-03/04/05 write scripts only and must NOT
edit this file (see plan frontmatter: owned-by: 09-02).

No top-level import of scripts.rl_* to avoid import-collection failure
during Wave 0 scaffolding.
"""
from __future__ import annotations

import pytest


class TestRLTrainUnit:
    """Unit contract tests for Phase 9 RL training seams.

    Maps to: GRPO-05 (interleaved dual-mode), GRPO-06 (LoRA config + router
    freeze), GRPO-07 (GSPO primary + RSPO floor), GRPO-08 (autohalt guards).
    """

    # -------------------------------------------------------------------------
    # GRPO-05: Interleaved dual-mode batch composition
    # -------------------------------------------------------------------------

    def test_dual_mode_batch(self, mock_tinker_client):
        """GRPO-05: sample_interleaved_prompts returns items from BOTH gen and judge pools.

        Asserts that the interleaved sampler includes at least one item tagged
        for generation and at least one tagged for judge-scoring in every batch.
        The ~60% judge / ~40% gen split is enforced by test_judge_ge_gen_budget.
        """
        rl_rollouts = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(10)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(10)]
        batch = rl_rollouts.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=8)
        tags = [item["tag"] for item in batch]
        assert "gen" in tags, "batch must contain at least one gen item"
        assert "judge" in tags, "batch must contain at least one judge item"

    def test_judge_ge_gen_budget(self, mock_tinker_client):
        """GRPO-05: judge items >= gen items per batch (D-09-04: ~60% judge / ~40% gen).

        The interleaved sampler must allocate more budget to judge prompts than
        generation prompts per the D-09-04 locked ratio.
        """
        rl_rollouts = pytest.importorskip("scripts.rl_rollouts")
        gen_pool = [{"prompt": f"gen_{i}", "tag": "gen"} for i in range(20)]
        judge_pool = [{"prompt": f"judge_{i}", "tag": "judge"} for i in range(20)]
        batch = rl_rollouts.sample_interleaved_prompts(gen_pool, judge_pool, batch_size=10)
        n_gen = sum(1 for item in batch if item.get("tag") == "gen")
        n_judge = sum(1 for item in batch if item.get("tag") == "judge")
        assert n_judge >= n_gen, (
            f"n_judge ({n_judge}) must be >= n_gen ({n_gen}) per D-09-04 ratio"
        )

    # -------------------------------------------------------------------------
    # GRPO-06: LoRA config — frozen router, trained MLP/attn/unembed
    # -------------------------------------------------------------------------

    def test_lora_config(self, mock_tinker_client):
        """GRPO-06: build_training_client passes train_mlp/attn/unembed=True, NO train_router.

        Tinker SDK v0.22.3 LoraConfig has no train_router field (D-09-02: router
        gates FROZEN). Asserting absence of train_router kwarg catches any
        regression that re-introduces the frozen router as a trained parameter.
        """
        rl_train = pytest.importorskip("scripts.rl_train")
        from unittest.mock import MagicMock, patch

        args = MagicMock()
        args.model_id = "fake-model"
        args.lora_rank = 16
        args.lora_seed = 42

        captured_kwargs = {}

        def fake_create_lora_training_client(*a, **kw):
            captured_kwargs.update(kw)
            return mock_tinker_client

        with patch.object(rl_train, "create_lora_training_client", fake_create_lora_training_client):
            rl_train.build_training_client(args)

        assert captured_kwargs.get("train_mlp") is True, "train_mlp must be True"
        assert captured_kwargs.get("train_attn") is True, "train_attn must be True"
        assert captured_kwargs.get("train_unembed") is True, "train_unembed must be True"
        assert "train_router" not in captured_kwargs, (
            "train_router must NOT appear in LoraConfig kwargs (D-09-02: router frozen)"
        )

    def test_protected_mask_check(self, mock_tinker_client):
        """GRPO-06: protected_mask_jaccard returns float in [0, 1].

        Monitor-only check (D-09-02): protected-expert overlap is logged per
        step via ForwardBackwardOutput.metrics; no active regularizer injection.
        Jaccard score quantifies overlap between active routing and protected mask.
        """
        rl_train = pytest.importorskip("scripts.rl_train")
        import numpy as np

        # Build a minimal fake active-experts array (48 layers, 128 experts each)
        active_experts = np.zeros((48, 128), dtype=bool)
        active_experts[0, :10] = True  # activate 10 experts in layer 0

        score = rl_train.protected_mask_jaccard(
            active_experts,
            mask_path="output/profiling/reasoning-merged-v4/protected_expert_mask.npy",
        )
        assert isinstance(score, float), f"Jaccard score must be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Jaccard score {score} not in [0, 1]"

    # -------------------------------------------------------------------------
    # GRPO-07: GSPO primary (D-09-03) + RSPO floor
    # -------------------------------------------------------------------------

    def test_gspo_rspo_floor(self, mock_tinker_client):
        """GRPO-07: rspo_floored_ratio clamps below-1 IS ratios to 1.0 (RSPO stop-grad floor).

        Also asserts that build_loss_step() with no flag selects forward_backward_custom
        (GSPO sequence-level IS objective, D-09-03 locked default) NOT forward_backward
        (GRPO token-level fallback, selected only via --grpo-fallback/--no-gspo).
        """
        rl_train = pytest.importorskip("scripts.rl_train")

        # Part A: RSPO floor — seq_ratio.clamp(min=1.0)
        low_ratio = rl_train.rspo_floored_ratio(train_lp=-1.5, sampling_lp=-1.0)
        assert low_ratio == pytest.approx(1.0), (
            f"ratio < 1.0 must be clamped to 1.0 (RSPO floor), got {low_ratio}"
        )
        high_ratio = rl_train.rspo_floored_ratio(train_lp=-0.5, sampling_lp=-1.0)
        expected = pytest.approx(high_ratio)  # ratio > 1.0: exp(-0.5 - -1.0) = exp(0.5) ≈ 1.649
        assert high_ratio > 1.0, (
            f"ratio > 1.0 must pass through unchanged, got {high_ratio}"
        )

        # Part B: GSPO default path — build_loss_step with no flag uses forward_backward_custom
        data = MagicMock() if False else __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mock_tinker_client.forward_backward.reset_mock()
        mock_tinker_client.forward_backward_custom.reset_mock()
        rl_train.build_loss_step(mock_tinker_client, data)  # no --grpo-fallback flag
        assert mock_tinker_client.forward_backward_custom.called, (
            "build_loss_step() with no flag must use forward_backward_custom (GSPO default, D-09-03)"
        )
        assert not mock_tinker_client.forward_backward.called, (
            "build_loss_step() with no flag must NOT call forward_backward (GRPO fallback path)"
        )

    def test_grpo_advantages(self, mock_tinker_client):
        """GRPO-07: compute_rollout_advantages returns non-zero advantages for mixed-reward groups.

        Mixed groups (varying rewards within a group) must produce non-zero group-relative
        advantages. Constant-reward groups must be dropped (zero advantage / filtered out).
        """
        rl_rollouts = pytest.importorskip("scripts.rl_rollouts")
        tinker = pytest.importorskip("tinker")
        from tests._rl_fixtures import make_trajectory_group

        # Mixed-reward group: diverse rewards → non-zero advantages, real Datums.
        mixed = make_trajectory_group([1.0, 0.5, 0.0, 0.8], group_id="gen-0")
        data, advantages, meta = rl_rollouts.compute_rollout_advantages([mixed])
        assert any(a != 0.0 for a in advantages), (
            "mixed-reward group must produce at least one non-zero advantage"
        )
        # The wrapper must emit real tinker.Datum objects carrying sampled logprobs
        # (regression guard for the original dict→forward_backward_custom AttributeError).
        assert isinstance(data[0], tinker.Datum)
        assert "logprobs" in data[0].loss_fn_inputs

        # Constant-reward group: cookbook returns a singleton with zero advantages.
        const = make_trajectory_group([0.5, 0.5, 0.5], group_id="gen-1")
        _data_c, advantages_c, _meta_c = rl_rollouts.compute_rollout_advantages([const])
        assert all(abs(a) < 1e-9 for a in advantages_c), (
            "constant-reward group must yield all-zero advantages"
        )

    # -------------------------------------------------------------------------
    # GRPO-08: KL and routing autohalt guards
    # -------------------------------------------------------------------------

    def test_kl_autohalt(self, mock_tinker_client):
        """GRPO-08: check_halt raises or returns non-None when KL > hard threshold (0.3).

        KL thresholds: soft alert 0.1, HARD halt 0.3 on kl_sample_train_v1.
        kl_v1=0.4 exceeds 0.3 → must trigger halt (RuntimeError or non-None return).
        """
        rl_train = pytest.importorskip("scripts.rl_train")

        try:
            result = rl_train.check_halt(
                kl_v1=0.4,     # > hard threshold 0.3
                e_frac=0.8,    # OK (above soft threshold 0.7)
                kl_soft=0.1,
                kl_hard=0.3,
                efrac_soft=0.7,
                efrac_hard=0.5,
            )
            assert result is not None, (
                "check_halt must return non-None halt_reason when kl_v1 (0.4) > kl_hard (0.3)"
            )
        except RuntimeError:
            pass  # RuntimeError halt signal is also acceptable

    def test_routing_autohalt(self, mock_tinker_client):
        """GRPO-08: check_halt raises or returns non-None when e_frac < hard threshold (0.5).

        MoE routing thresholds: soft alert e_frac < 0.7, HARD halt < 0.5.
        e_frac=0.4 is below 0.5 → must trigger halt (RuntimeError or non-None return).
        """
        rl_train = pytest.importorskip("scripts.rl_train")

        try:
            result = rl_train.check_halt(
                kl_v1=0.05,    # OK (below soft threshold 0.1)
                e_frac=0.4,    # < hard threshold 0.5
                kl_soft=0.1,
                kl_hard=0.3,
                efrac_soft=0.7,
                efrac_hard=0.5,
            )
            assert result is not None, (
                "check_halt must return non-None halt_reason when e_frac (0.4) < efrac_hard (0.5)"
            )
        except RuntimeError:
            pass  # RuntimeError halt signal is also acceptable


# ---------------------------------------------------------------------------
# Direct GSPO loss-fn execution (09-07): the one path mocks bypass — exercises
# the loss BODY on a real Datum and pins the action-token masking of the IS ratio.
# ---------------------------------------------------------------------------


def test_gspo_loss_fn_masks_training_logprobs_to_action_tokens():
    """gspo_loss_fn must mask train_lps to action tokens before the IS ratio.

    train_lps (the SDK's forward logprobs) is FULL-length (obs + action). The loss
    must sum only action-token logprobs (mask > 0), matching sampling_lps (whose obs
    positions are 0). A large distinctive value planted on the obs position must NOT
    leak into the ratio — if it does, exp(train_sum - sampling_sum) underflows to the
    RSPO floor (1.0) and the loss collapses to -(1.0 * adv), failing this assertion.
    """
    import math

    rl_train = pytest.importorskip("scripts.rl_train")
    pytest.importorskip("tinker")
    import torch
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    from tests._rl_fixtures import make_trajectory_group

    # One trajectory: obs 2 tokens, action 2 tokens (group of 1 -> advantage 0 would
    # be uninformative), so build a 2-member group and take the high-reward datum.
    tg = make_trajectory_group([1.0, 0.0], group_id="gen-0")
    # Datum for the first trajectory (reward 1.0 in a [1.0,0.0] group -> advantage +0.5).
    from tinker_cookbook.rl.data_processing import compute_advantages

    adv_P = compute_advantages([tg])
    datum = trajectory_to_data(tg.trajectories_G[0], float(adv_P[0][0]))[0]

    mask = datum.loss_fn_inputs["mask"].to_torch()
    sampling = datum.loss_fn_inputs["logprobs"].to_torch()
    adv_weights = datum.loss_fn_inputs["advantages"].to_torch()
    action_mask = mask > 0
    expected_adv = float(adv_weights[action_mask][0])  # +0.5

    # Synthesize FULL-length training logprobs: a huge distinctive value on the FIRST
    # (obs) position that must be masked out. On action positions set train ABOVE
    # sampling (+0.3 each) so the IS ratio exp(train_sum - sampling_sum) is > 1 (NOT
    # floored to 1.0) — only then do the masked and unmasked sums give different
    # losses, making this test actually discriminate the obs-leak bug.
    train_lps = torch.zeros_like(sampling)
    train_lps[0] = -99.0  # obs position — MUST be excluded by the mask
    train_action_vals = []
    for j in range(len(train_lps)):
        if action_mask[j]:
            train_lps[j] = float(sampling[j]) + 0.3
            train_action_vals.append(float(train_lps[j]))

    loss_fn = rl_train._make_gspo_loss_fn([datum])  # closes over the FULL datum
    total_loss, metrics = loss_fn([datum], [train_lps])

    # Expected: ratio over ACTION tokens only.
    train_sum = sum(train_action_vals)
    sampling_sum = float(sampling[action_mask].sum())
    expected_ratio = max(1.0, math.exp(train_sum - sampling_sum))
    expected_loss = -(expected_ratio * expected_adv)

    assert torch.isfinite(total_loss), "loss must be finite"
    assert float(total_loss) == pytest.approx(expected_loss, abs=1e-5), (
        f"loss {float(total_loss)} != masked-IS expected {expected_loss} — "
        f"obs-token logprob (-99.0) likely leaked into train_sum"
    )
    assert metrics["gspo/n_sequences"] == 1.0


def test_gspo_loss_fn_empty_mask_datum_is_safe():
    """A Datum whose mask has no action position contributes 0, not a crash."""
    rl_train = pytest.importorskip("scripts.rl_train")
    pytest.importorskip("tinker")
    import torch
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition
    import tinker

    # Immediate-EOS completion: action tokens = [] -> mask has no 1.0 position.
    traj = Trajectory(
        transitions=[
            Transition(
                ob=tinker.ModelInput.from_ints([10, 11, 12]),
                ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                reward=0.5,
                episode_done=True,
            )
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )
    datum = trajectory_to_data(traj, 0.4)[0]
    train_lps = torch.zeros_like(datum.loss_fn_inputs["mask"].to_torch())

    loss_fn = rl_train._make_gspo_loss_fn([datum])
    total_loss, _metrics = loss_fn([datum], [train_lps])
    assert torch.isfinite(total_loss)
    assert float(total_loss) == pytest.approx(0.0, abs=1e-6)


def test_strip_to_target_tokens_satisfies_fbc_validator():
    """_strip_to_target_tokens output obeys forward_backward_custom's key constraint.

    The SDK forward validator rejects any loss_fn_inputs key outside
    {target_tokens, weights}. This pins the exact ValueError the live smoke hit
    (unexpected keys ['advantages','logprobs','mask']) at unit speed.
    """
    rl_train = pytest.importorskip("scripts.rl_train")
    pytest.importorskip("tinker")
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    from tests._rl_fixtures import make_trajectory_group

    tg = make_trajectory_group([1.0, 0.0], group_id="gen-0")
    full = trajectory_to_data(tg.trajectories_G[0], 0.5)[0]
    # Precondition: the cookbook datum carries the rejected keys.
    assert {"advantages", "logprobs", "mask"} & set(full.loss_fn_inputs)
    stripped = rl_train._strip_to_target_tokens(full)
    assert set(stripped.loss_fn_inputs) <= {"target_tokens", "weights"}
    assert "target_tokens" in stripped.loss_fn_inputs


def test_gspo_loss_fn_gradient_flows_to_action_positions():
    """loss.backward() puts gradient on action-token logprobs only (obs = 0).

    forward_backward_custom backprops the client loss w.r.t. the forward logprobs,
    so the loss must be differentiable in train_lps with grad confined to action
    positions (mask > 0). This is the offline proof the live gradient step will flow.
    """
    rl_train = pytest.importorskip("scripts.rl_train")
    pytest.importorskip("tinker")
    import torch
    from tinker_cookbook.rl.data_processing import compute_advantages, trajectory_to_data

    from tests._rl_fixtures import make_trajectory_group

    tg = make_trajectory_group([1.0, 0.0], group_id="gen-0")
    adv_P = compute_advantages([tg])
    datum = trajectory_to_data(tg.trajectories_G[0], float(adv_P[0][0]))[0]  # advantage +0.5
    mask = datum.loss_fn_inputs["mask"].to_torch()
    action_mask = mask > 0

    train_lps = torch.zeros_like(mask).requires_grad_(True)
    loss_fn = rl_train._make_gspo_loss_fn([datum])
    total_loss, _ = loss_fn([datum], [train_lps])
    assert total_loss.requires_grad, "loss must be differentiable w.r.t. train_lps"
    total_loss.backward()

    grad = train_lps.grad
    assert grad is not None, "no gradient computed for train_lps"
    assert grad[action_mask].abs().sum().item() > 0.0, "no gradient on action positions"
    assert grad[~action_mask].abs().sum().item() == pytest.approx(0.0, abs=1e-9), (
        "gradient leaked onto obs (non-action) positions"
    )
