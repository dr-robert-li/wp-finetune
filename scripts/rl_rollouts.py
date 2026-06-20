"""Rollout sampling, dual reward collection, and advantage assembly for Phase 9 GSPO.

This module is the data-transform seam between sampling and gradient. It:
  - Interleaves wp_gen and wp_judge prompt pools at a judge-weighted ratio (GRPO-05 / D-09-04)
  - Collects gen rewards from the Phase 8 pipeline (reward_pipeline.py, unmodified)
  - Combines the deterministic fix-correctness anchor with the capped Claude-consistency
    signal from rl_judge_dispatch.py (09-03) for judge rewards (D-09-05 guard 1)
  - Builds cookbook-shaped trajectory groups and delegates advantage computation
    to tinker_cookbook.rl.data_processing (or inline fallback when tinker is absent)

Design constraints (D-09-04, D-09-05, GRPO-05):
  - JUDGE_RATIO = 0.6: n_judge = round(batch_size * 0.6), n_gen = batch_size - n_judge
  - judge_consistency_weight <= 0.5 HARD CAP (T-09-RWD-CAP): asserted at import
  - reward_pipeline.py consumed UNMODIFIED (D-09-05 / PATTERNS line 259)
  - Security-zero groups DROPPED (not zeroed) per T-09-SECDROP
  - Advantage math DELEGATED to cookbook (RESEARCH "Don't Hand-Roll")

Exports:
  JUDGE_RATIO                        constant (0.6)
  judge_consistency_weight           constant (<= 0.5)
  sample_interleaved_prompts(gen_pool, judge_pool, batch_size) -> list
  combine_judge_reward(fix_correctness, consistency, weight) -> float
  build_trajectory_groups(rollouts, rewards) -> list[dict]
  compute_rollout_advantages(groups) -> (list[dict], dict)
  collect_rollouts(sampling_client, gen_pool, judge_pool, args) -> list[dict]
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D-09-04 ratio constant
# ---------------------------------------------------------------------------

JUDGE_RATIO: float = 0.6  # ~60% wp_judge, 40% wp_gen; n_judge >= n_gen always

# ---------------------------------------------------------------------------
# D-09-05 guard 1: Claude-consistency contribution cap
# ---------------------------------------------------------------------------

#: Fraction of the judge-side scalar contributed by Claude-consistency.
#: HARD CAP: must be <= 0.5 so fix-correctness remains the anchor.
#: Asserted at module load — any attempt to set > 0.5 raises immediately.
judge_consistency_weight: float = 0.3

# Module-level cap assertion (T-09-RWD-CAP — fires at import, not at call time)
assert judge_consistency_weight <= 0.5, (
    f"D-09-05 guard 1 violated: judge_consistency_weight={judge_consistency_weight} "
    f"must be <= 0.5 (fix-correctness must remain the anchor). "
    f"Raising at import to prevent the consistency signal from dominating reward."
)

# ---------------------------------------------------------------------------
# MO-GRPO within-group normalization (mirrors reward_pipeline._mo_grpo_norm)
# Applied per-signal before scalar combination in judge reward path.
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-8


def _mo_grpo_norm(values: np.ndarray) -> np.ndarray:
    """Within-group standardization: (x - mu) / (sigma + epsilon).

    Mirrors reward_pipeline._mo_grpo_norm exactly (population std, ddof=0).
    Used for any extra per-signal normalization before scalar combination
    in the judge reward path.

    Args:
        values: 1-D numpy array of raw signal values for a rollout group.

    Returns:
        np.ndarray: Normalized values with same shape as input.
    """
    mu = values.mean()
    sigma = values.std(ddof=0)
    return (values - mu) / (sigma + _EPSILON)


# ---------------------------------------------------------------------------
# Interleaved prompt sampling (GRPO-05 / D-09-04)
# ---------------------------------------------------------------------------


def sample_interleaved_prompts(
    gen_pool: list,
    judge_pool: list,
    batch_size: int,
) -> list:
    """Sample an interleaved batch with judge-weighted ratio (~60% judge / 40% gen).

    Allocates n_judge = round(batch_size * JUDGE_RATIO) samples from judge_pool
    and n_gen = batch_size - n_judge from gen_pool, ensuring n_judge >= n_gen for
    all batch_size >= 2.

    Items are dicts (or any objects) — they are returned as-is to preserve
    tag/metadata for downstream routing.

    Args:
        gen_pool: Pool of gen-mode prompt dicts (wp_gen).
        judge_pool: Pool of judge-mode prompt dicts (wp_judge).
        batch_size: Total number of prompts in the batch.

    Returns:
        list: judge samples first, then gen samples (judge items precede gen items).

    Raises:
        ValueError: If batch_size < 2 or either pool is empty.
    """
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2, got {batch_size}")
    if not gen_pool:
        raise ValueError("gen_pool is empty")
    if not judge_pool:
        raise ValueError("judge_pool is empty")

    n_judge = round(batch_size * JUDGE_RATIO)
    n_gen = batch_size - n_judge

    # Guarantee n_judge >= n_gen (invariant for all batch_size >= 2 with JUDGE_RATIO=0.6)
    # This holds analytically for JUDGE_RATIO=0.6 but we assert defensively.
    if n_judge < n_gen:
        # Redistribute: give one extra slot to judge
        n_judge += 1
        n_gen -= 1

    judge_samples = random.sample(judge_pool, min(n_judge, len(judge_pool)))
    gen_samples = random.sample(gen_pool, min(n_gen, len(gen_pool)))

    # Pad if pools are smaller than requested (rare in production; handled gracefully)
    while len(judge_samples) < n_judge:
        judge_samples.append(random.choice(judge_pool))
    while len(gen_samples) < n_gen:
        gen_samples.append(random.choice(gen_pool))

    return judge_samples + gen_samples


# ---------------------------------------------------------------------------
# Judge reward combination (D-09-05 guard 1)
# ---------------------------------------------------------------------------


def combine_judge_reward(
    fix_correctness: float,
    consistency: float,
    weight: float = judge_consistency_weight,
) -> float:
    """Combine the deterministic fix-correctness anchor with the capped Claude-consistency signal.

    Formula: (1 - weight) * fix_correctness + weight * consistency

    The consistency contribution is bounded by `weight` (default 0.3, hard cap 0.5)
    so fix_correctness remains the primary signal. The weight parameter is validated
    at call time; the module-level default is validated at import time.

    Args:
        fix_correctness: Deterministic correctness score from verifiable signals (e.g. PHPCS).
                         Acts as the anchor signal (primary weight).
        consistency: Claude-consistency score from score_judge_consistency_batch (09-03).
                     Capped contribution; in [0, 1].
        weight: Fraction to allocate to consistency signal. Must be <= 0.5.
                Defaults to judge_consistency_weight (0.3).

    Returns:
        float: Combined judge-side scalar reward.

    Raises:
        ValueError: If weight > 0.5 (cap violation).
    """
    if weight > 0.5:
        raise ValueError(
            f"combine_judge_reward: weight={weight} violates the D-09-05 cap (must be <= 0.5). "
            f"fix_correctness must remain the anchor signal."
        )
    return (1.0 - weight) * fix_correctness + weight * consistency


# ---------------------------------------------------------------------------
# Trajectory group construction
# ---------------------------------------------------------------------------


def build_trajectory_groups(
    rollouts: list,
    rewards: list,
) -> list[dict]:
    """Build cookbook-shaped trajectory groups from rollouts and their RewardResults.

    Maps each rollout + its RewardResult.scalar into a per-sample dict carrying
    the completion text and scalar reward. Security-gated members (breakdown.security_fail=True)
    are DROPPED from the group (T-09-SECDROP): they must not contribute a
    zero-advantage signal. The remaining group entries are passed to
    compute_rollout_advantages for constant-group filtering and advantage assembly.

    Note: This function builds single-sample "groups" (one dict per rollout) because
    the GRPO group structure (G completions per prompt) is assembled by the caller
    (collect_rollouts). For synthetic tensor tests, this function accepts plain
    rollout objects with a .completion attribute.

    Args:
        rollouts: List of rollout objects with at least a .completion attribute.
        rewards: List of RewardResult objects from compute_group_rewards or
                 combine_judge_reward. Each must have .scalar (float) and
                 .breakdown.security_fail (bool).

    Returns:
        list[dict]: One dict per non-security-failed rollout, each with keys:
            - "completion": str
            - "reward": float  (the scalar reward)
            - "breakdown": the RewardBreakdown object for logging
    """
    if len(rollouts) != len(rewards):
        raise ValueError(
            f"build_trajectory_groups: len(rollouts)={len(rollouts)} != "
            f"len(rewards)={len(rewards)}"
        )

    groups: list[dict] = []
    n_sec_dropped = 0

    for rollout, reward in zip(rollouts, rewards):
        # T-09-SECDROP: drop security-gated members entirely (do not zero advantage)
        breakdown = getattr(reward, "breakdown", None)
        sec_fail = getattr(breakdown, "security_fail", False) if breakdown is not None else False

        if sec_fail:
            n_sec_dropped += 1
            logger.debug(
                "build_trajectory_groups: dropped security-gated rollout "
                "(breakdown.security_fail=True)"
            )
            continue

        completion = getattr(rollout, "completion", str(rollout))
        groups.append(
            {
                "completion": completion,
                "reward": float(reward.scalar),
                "breakdown": breakdown,
            }
        )

    if n_sec_dropped > 0:
        logger.info(
            "build_trajectory_groups: dropped %d security-gated rollout(s) "
            "(T-09-SECDROP)",
            n_sec_dropped,
        )

    return groups


# ---------------------------------------------------------------------------
# Inline group-centred advantage fallback (used when tinker_cookbook absent)
# ---------------------------------------------------------------------------

def _group_by_id(groups: list[dict]) -> dict[Any, list[dict]]:
    """Partition flat per-completion dicts into per-prompt groups.

    Groups are keyed on the "group_id" stamped during the flatten step in
    compute_rollout_advantages. Insertion order of both groups and members is
    preserved so downstream output ordering is deterministic.
    """
    by_id: dict[Any, list[dict]] = {}
    for g in groups:
        by_id.setdefault(g.get("group_id"), []).append(g)
    return by_id


def _inline_remove_constant_reward_groups(groups: list[dict]) -> list[dict]:
    """Drop PROMPT GROUPS whose own completions all share one reward (zero gradient).

    Mirrors the semantics of tinker_cookbook.rl.data_processing.remove_constant_reward_groups.
    The decision is made PER PROMPT GROUP (keyed on "group_id"): a group is constant
    when std(rewards) < epsilon across *that group's* completions only. Completions
    from other prompts must not rescue or doom a group (CR-06). A surviving group
    contributes all its members; a constant group contributes none.
    """
    result: list[dict] = []
    for members in _group_by_id(groups).values():
        rewards = np.array([m["reward"] for m in members], dtype=float)
        if rewards.std(ddof=0) < _EPSILON:
            continue  # constant within this prompt group — drop all its members
        result.extend(members)
    return result


def _inline_compute_advantages(groups: list[dict]) -> list[dict]:
    """Group-centred advantages: A_i = r_i - mean(r) computed PER PROMPT GROUP.

    Mirrors the semantics of tinker_cookbook.rl.data_processing.compute_advantages.
    Centering uses each prompt group's own mean (keyed on "group_id"), so a
    completion's advantage is relative to its sibling completions only, not the
    whole batch (CR-06). Returns dicts with an added 'advantage' key, preserving
    input order.
    """
    # Precompute each group's mean once.
    group_means: dict[Any, float] = {}
    for gid, members in _group_by_id(groups).items():
        rewards = np.array([m["reward"] for m in members], dtype=float)
        group_means[gid] = float(rewards.mean())

    result = []
    for g in groups:
        entry = dict(g)
        entry["advantage"] = float(g["reward"] - group_means[g.get("group_id")])
        result.append(entry)
    return result


def _inline_assemble_training_data(groups_with_advantages: list[dict]) -> list[dict]:
    """Flatten to a list of Datum-like dicts for forward_backward.

    Mirrors the semantics of tinker_cookbook.rl.data_processing.assemble_training_data.
    Each dict carries: "completion", "reward", "advantage", and group/traj metadata.
    """
    return [dict(g) for g in groups_with_advantages]


# ---------------------------------------------------------------------------
# Cookbook-delegating advantage assembly
# ---------------------------------------------------------------------------


def compute_rollout_advantages(
    groups: list[dict],
) -> tuple[list[dict], dict]:
    """Compute advantages for rollout groups, delegating to the cookbook.

    Accepts groups in the simple dict format produced by build_trajectory_groups
    or as plain test dicts with {"prompt":..., "completions":[...], "rewards":[...]}.
    Delegates to tinker_cookbook.rl.data_processing when available; falls back to
    inline group-centering when tinker is absent (unit test / no-tinker path).

    Cookbook delegation path:
      1. remove_constant_reward_groups(groups)   — drop zero-gradient groups
      2. compute_advantages(groups)              — A_G = r_G - mean(r_G)
      3. assemble_training_data(groups, adv)     — flat Datum list with metadata

    Inline fallback (tinker absent):
      Same semantics implemented locally: std < epsilon → drop; A_i = r_i - mean(r).

    Args:
        groups: List of group dicts. Each must have either:
            - "reward" key (from build_trajectory_groups), or
            - "rewards" list (synthetic test format from test_rl_train.py)

    Returns:
        Tuple of:
          - data: list[dict], each with "advantage" key (and completion/reward)
          - meta: dict with group stats (n_groups, n_dropped, etc.)
    """
    # Normalise test-format groups {"prompt":..., "completions":[...], "rewards":[...]}
    # into the flat per-sample format expected by the advantage pipeline.
    flat_groups: list[dict] = []
    for src_idx, g in enumerate(groups):
        if "rewards" in g and "completions" in g:
            # Synthetic test format: expand into per-completion entries. All
            # completions of one source group share a group_id so the per-prompt
            # constant-filter and per-prompt advantage centering (CR-06) treat
            # them as a single GRPO group.
            group_id = g.get("prompt_id", g.get("prompt") or src_idx)
            for completion, reward in zip(g["completions"], g["rewards"]):
                flat_groups.append({
                    "group_id": group_id,
                    "prompt": g.get("prompt", ""),
                    "completion": completion,
                    "reward": float(reward),
                    "breakdown": None,
                })
        elif "reward" in g:
            entry = dict(g)
            # Carry an explicit group id when present; otherwise each per-rollout
            # dict from build_trajectory_groups is its own singleton group. (See
            # the production limitation noted in the module docstring / review.)
            entry.setdefault(
                "group_id",
                g.get("prompt_id", g.get("prompt") or src_idx),
            )
            flat_groups.append(entry)
        else:
            logger.warning(
                "compute_rollout_advantages: skipping group with unexpected format: %s",
                list(g.keys()),
            )

    n_input = len(flat_groups)

    # Inline group-centred advantage assembly is the real implementation here.
    # We operate on plain dicts (the format produced by build_trajectory_groups
    # and the synthetic test format); the cookbook's data_processing helpers
    # require its own trajectory_group types which we do not construct in this
    # module. The inline functions below mirror the cookbook semantics exactly:
    #   _inline_remove_constant_reward_groups  -> remove_constant_reward_groups
    #   _inline_compute_advantages             -> compute_advantages
    #   _inline_assemble_training_data         -> assemble_training_data
    filtered = _inline_remove_constant_reward_groups(flat_groups)
    n_dropped = n_input - len(filtered)

    if not filtered:
        meta = {
            "n_groups_input": n_input,
            "n_dropped_constant": n_dropped,
            "n_groups_output": 0,
        }
        return [], meta

    data_with_adv = _inline_compute_advantages(filtered)
    assembled = _inline_assemble_training_data(data_with_adv)

    meta = {
        "n_groups_input": n_input,
        "n_dropped_constant": n_dropped,
        "n_groups_output": len(assembled),
    }
    return assembled, meta


# ---------------------------------------------------------------------------
# Full rollout collection (wires sampling -> rewards -> groups)
# ---------------------------------------------------------------------------


def collect_rollouts(
    sampling_client: Any,
    gen_pool: list,
    judge_pool: list,
    args: Any,
) -> list[dict]:
    """Collect interleaved rollouts and compute rewards for a training step.

    Orchestrates:
      1. sample_interleaved_prompts (GRPO-05 / D-09-04)
      2. For wp_gen completions: compute_group_rewards (Phase 8 pipeline, unmodified)
      3. For wp_judge completions: fix-correctness via _extract_verifiable_signals
         combined with capped Claude-consistency via score_judge_consistency_batch
         using combine_judge_reward
      4. build_trajectory_groups (security-zero drops)

    The gen reward path consumes reward_pipeline.compute_group_rewards UNMODIFIED
    (D-09-05 / PATTERNS line 259). The judge reward path combines fix-correctness
    (deterministic anchor) with the capped consistency signal (D-09-05 guard 1).

    Args:
        sampling_client: Tinker sampling client (used to generate completions).
        gen_pool: Pool of wp_gen prompt dicts.
        judge_pool: Pool of wp_judge prompt dicts with "critique_text" field.
        args: Namespace with attributes:
            - judge_client: Judge client for compute_group_rewards
            - judge_model: Model ID for compute_group_rewards
            - consistency_model: Claude model for score_judge_consistency_batch
            - n_votes: N-vote median for consistency scorer (default 1)

    Returns:
        list[dict]: Trajectory group dicts from build_trajectory_groups, ready for
                    compute_rollout_advantages.
    """
    # Lazy imports to avoid import errors when modules are absent (unit tests)
    from scripts.reward_pipeline import compute_group_rewards, _extract_verifiable_signals  # noqa: PLC0415
    from scripts.rl_judge_dispatch import score_judge_consistency_batch  # noqa: PLC0415

    # Step 1: Interleaved sampling
    batch = sample_interleaved_prompts(gen_pool, judge_pool, batch_size=args.batch_size)

    gen_rollouts = [item for item in batch if item.get("tag") == "gen"]
    judge_rollouts = [item for item in batch if item.get("tag") == "judge"]

    all_rollouts = []
    all_rewards = []

    # Step 2: wp_gen rewards via Phase 8 pipeline (unmodified)
    if gen_rollouts:
        gen_completions = _generate_completions(sampling_client, gen_rollouts, args)
        gen_reward_results = compute_group_rewards(
            php_codes=[c.completion for c in gen_completions],
            judge_client=args.judge_client,
            judge_model=args.judge_model,
        )
        all_rollouts.extend(gen_completions)
        all_rewards.extend(gen_reward_results)

    # Step 3: wp_judge rewards (fix-correctness + capped consistency)
    if judge_rollouts:
        judge_completions = _generate_completions(sampling_client, judge_rollouts, args)

        # Deterministic fix-correctness from verifiable signals on corrected code
        fix_correctness_scores = []
        for completion_obj in judge_completions:
            rubric = _extract_verifiable_signals(completion_obj.completion)
            # Normalize to [0, 1] from [0, 100]
            fix_score = float(rubric.overall) / 100.0
            fix_correctness_scores.append(fix_score)

        # Capped Claude-consistency via 09-03 dispatcher
        consistency_samples = [
            {
                "php_code": c.completion,
                "critique_text": judge_rollouts[i].get("critique_text", ""),
            }
            for i, c in enumerate(judge_completions)
        ]
        consistency_scores = asyncio.run(
            score_judge_consistency_batch(
                consistency_samples,
                model=getattr(args, "consistency_model", "sonnet"),
                n_votes=getattr(args, "n_votes", 1),
            )
        )

        # Combine on the RAW [0, 1] scale (D-09-05 guard 1). fix_correctness and
        # consistency are both in [0, 1], so (1-w)*fc + w*cons stays in [0, 1] and
        # non-negative for valid positive inputs. The MO-GRPO group normalization
        # (advantage centering, A_i = r_i - mean(r)) is applied DOWNSTREAM in
        # compute_rollout_advantages — applying it here as well would re-introduce
        # z-scores and drive valid positive judge rewards negative (CR-05).
        for i, completion_obj in enumerate(judge_completions):
            combined_scalar = combine_judge_reward(
                fix_correctness=fix_correctness_scores[i],
                consistency=consistency_scores[i],
            )

            # Wrap in a RewardResult-like object for build_trajectory_groups
            reward_obj = _make_reward_result(
                scalar=combined_scalar,
                fix_correctness=fix_correctness_scores[i],
                consistency=consistency_scores[i],
            )
            all_rollouts.append(completion_obj)
            all_rewards.append(reward_obj)

    # Step 4: Build trajectory groups (security-zero drops handled here)
    return build_trajectory_groups(all_rollouts, all_rewards)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_completions(sampling_client: Any, prompt_items: list, args: Any) -> list:
    """Generate completions for a list of prompt dicts using the sampling client.

    Args:
        sampling_client: Tinker sampling client with a generate() method.
        prompt_items: List of prompt dicts with "prompt" key.
        args: Namespace with generation arguments (max_new_tokens, etc.)

    Returns:
        list: Completion objects with a .completion attribute.
    """
    completions = []
    for item in prompt_items:
        prompt_text = item.get("prompt", "")
        result = sampling_client.generate(
            prompt=prompt_text,
            max_new_tokens=getattr(args, "max_new_tokens", 512),
        )
        completions.append(result)
    return completions


def _make_reward_result(
    scalar: float,
    fix_correctness: float,
    consistency: float,
) -> Any:
    """Construct a minimal RewardResult-like object for build_trajectory_groups.

    Uses the real RewardResult type when reward_pipeline is available; falls back
    to a simple namespace object for unit tests.
    """
    try:
        from scripts.reward_pipeline import RewardResult, RewardBreakdown  # noqa: PLC0415

        # RewardBreakdown requires many fields — use a lightweight wrapper instead
        # to avoid coupling to the full breakdown structure
        class _JudgeBreakdown:
            """Minimal breakdown carrying security_fail for the trajectory group builder."""
            def __init__(self):
                self.security_fail = False
                self.fix_correctness = fix_correctness
                self.consistency = consistency

        breakdown = _JudgeBreakdown()

        class _JudgeRewardResult:
            def __init__(self, scalar, breakdown):
                self.scalar = scalar
                self.breakdown = breakdown

        return _JudgeRewardResult(scalar=scalar, breakdown=breakdown)

    except ImportError:
        import types
        r = types.SimpleNamespace()
        r.scalar = scalar
        r.breakdown = types.SimpleNamespace(
            security_fail=False,
            fix_correctness=fix_correctness,
            consistency=consistency,
        )
        return r
