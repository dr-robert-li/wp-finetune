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

    Each returned item is a SHALLOW COPY of the source dict with an added
    "_origin" key ("judge" or "gen") recording which pool it came from. This is
    the authoritative gen-vs-judge routing signal for collect_rollouts — prompt
    pool items (shape {"messages": [...]}) carry no intrinsic "tag" field, so
    routing must be by pool origin, not by a field that does not exist (CR-03).
    The copy avoids mutating the caller's pool dicts.

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

    # Stamp pool origin on shallow copies so collect_rollouts can route gen vs
    # judge without depending on a "tag" field the pool items do not carry (CR-03).
    judge_items = [_stamp_origin(item, "judge") for item in judge_samples]
    gen_items = [_stamp_origin(item, "gen") for item in gen_samples]

    return judge_items + gen_items


def _stamp_origin(item: Any, origin: str) -> dict:
    """Return a shallow copy of `item` with an added "_origin" routing tag.

    Falls back to wrapping non-dict items so origin is always recoverable.
    """
    if isinstance(item, dict):
        return {**item, "_origin": origin}
    return {"item": item, "_origin": origin}


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
            - "group_id": the prompt-group id threaded from collect_rollouts
              (read from rollout.group_id when present). This is what makes the
              per-prompt constant filter and per-prompt advantage centering in
              compute_rollout_advantages operate on TRUE GRPO groups (G
              completions per prompt) rather than singletons.
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
        entry = {
            "completion": completion,
            "reward": float(reward.scalar),
            "breakdown": breakdown,
        }
        # Thread the true prompt-group id when the rollout carries one so that
        # G completions of the same prompt land in one GRPO group downstream.
        # Falls back to leaving group_id unset (compute_rollout_advantages then
        # treats the entry as its own singleton group).
        group_id = getattr(rollout, "group_id", None)
        if group_id is not None:
            entry["group_id"] = group_id
        groups.append(entry)

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

    # Route by pool origin stamped at sample time (CR-03). Pool items are
    # {"messages": [...]} and carry no intrinsic "tag" — sample_interleaved_prompts
    # adds "_origin" so both lists are correctly populated.
    gen_rollouts = [item for item in batch if item.get("_origin") == "gen"]
    judge_rollouts = [item for item in batch if item.get("_origin") == "judge"]

    # Stamp a unique per-prompt group id on every batch item BEFORE sampling so
    # the G completions of one prompt share a group_id (true GRPO group). The id
    # is globally unique across gen and judge so the two pathways never collide
    # in compute_rollout_advantages' per-group constant filter / centering.
    for gid, item in enumerate(gen_rollouts):
        item["_group_id"] = f"gen-{gid}"
    for gid, item in enumerate(judge_rollouts):
        item["_group_id"] = f"judge-{gid}"

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

        # Capped Claude-consistency via 09-03 dispatcher. Each judge completion
        # carries .group_id ("judge-<k>") identifying its source prompt; map
        # that back to the originating judge_rollouts item for its critique_text
        # (the flat completion index no longer aligns with the prompt list now
        # that we draw group_size completions per prompt).
        judge_by_gid = {item["_group_id"]: item for item in judge_rollouts}
        consistency_samples = [
            {
                "php_code": c.completion,
                "critique_text": judge_by_gid.get(
                    getattr(c, "group_id", None), {}
                ).get("critique_text", ""),
            }
            for c in judge_completions
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


class _Completion:
    """Lightweight completion carrier with the attributes downstream expects.

    `.completion` is the decoded text consumed by compute_group_rewards and
    _extract_verifiable_signals; `.group_id` is the prompt-group id threaded
    through build_trajectory_groups so G samples of one prompt form one GRPO
    group (true per-prompt constant filter + advantage centering, CR-06).
    """

    __slots__ = ("completion", "group_id")

    def __init__(self, completion: str, group_id: Any):
        self.completion = completion
        self.group_id = group_id


def build_rl_renderer() -> tuple[Any, Any]:
    """Return (renderer, tokenizer) for turning {"messages": [...]} into ModelInput.

    Module-level seam (mirrors rl_train.create_lora_training_client) so the
    integration test can patch it and avoid downloading a real tokenizer/model.
    Uses the SHARED constants from tinker_rl_data (BASE_MODEL / RENDERER_NAME),
    not a sibling training script, so production has a single source of truth.
    """
    from tinker_cookbook import renderers  # noqa: PLC0415
    from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: PLC0415

    from scripts.tinker_rl_data import BASE_MODEL, RENDERER_NAME  # noqa: PLC0415

    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)
    return renderer, tok


def _prompt_user_messages(item: dict) -> list:
    """Extract the user-turn messages from a prompt-pool item.

    Pool items have shape {"messages": [{"role": "user", "content": "..."}]}
    (load_rl_prompts strips the assistant turn). _stamp_origin adds "_origin"
    and the synthetic unit tests add "prompt"/"tag" — none of which are message
    turns, so we filter on role. Falls back to a single user turn built from a
    bare "prompt" field for the synthetic test format.
    """
    messages = item.get("messages")
    if messages:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            return user_msgs
    # Synthetic/test fallback: {"prompt": "..."} with no messages list.
    prompt_text = item.get("prompt", "")
    return [{"role": "user", "content": str(prompt_text)}]


def _decode_samples(resp: Any, tok: Any) -> list[str]:
    """Decode every sampled sequence from one .sample() response to text.

    Mirrors tinker_reasoning_sft._all_sample_texts: resolve the future, read
    .sequences/.samples, decode tokens with the tokenizer.
    """
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    out = []
    for seq in seqs:
        toks = (
            getattr(seq, "tokens", None)
            or getattr(seq, "token_ids", None)
            or getattr(seq, "output_tokens", None)
        )
        out.append(tok.decode(toks))
    return out


def _generate_completions(
    sampling_client: Any,
    prompt_items: list,
    args: Any,
    renderer: Any = None,
    tok: Any = None,
) -> list:
    """Generate G completions per prompt via the real Tinker sampling API.

    Uses SamplingClient.sample(prompt: ModelInput, num_samples, sampling_params)
    — there is NO .generate() on the real client (CR-01 root cause #2). Each
    prompt is rendered to a generation ModelInput via the renderer, sampled
    num_samples=group_size times, and each decoded completion is tagged with a
    per-prompt group_id so downstream advantage centering forms true GRPO groups.

    renderer/tok are injectable (built via build_rl_renderer when omitted) so the
    integration test can drive the seam without a real tokenizer download.

    Args:
        sampling_client: Tinker SamplingClient (.sample(...)).
        prompt_items: List of prompt dicts ({"messages": [...]} or {"prompt": ...}).
        args: Namespace with max_new_tokens, group_size, temperature.
        renderer, tok: Optional injected renderer/tokenizer.

    Returns:
        list[_Completion]: group_size completions per prompt, each carrying
        .completion (decoded text) and .group_id (prompt index).
    """
    import tinker  # noqa: PLC0415

    if renderer is None or tok is None:
        renderer, tok = build_rl_renderer()

    group_size = int(getattr(args, "group_size", 4))
    sp = tinker.SamplingParams(
        max_tokens=getattr(args, "max_new_tokens", 512),
        temperature=getattr(args, "temperature", 1.0),
        stop=renderer.get_stop_sequences(),
    )

    completions: list = []
    for prompt_idx, item in enumerate(prompt_items):
        user_msgs = _prompt_user_messages(item)
        prompt = renderer.build_generation_prompt(user_msgs)
        resp = sampling_client.sample(
            prompt=prompt,
            num_samples=group_size,
            sampling_params=sp,
        )
        group_id = item.get("_group_id", prompt_idx)
        for text in _decode_samples(resp, tok):
            completions.append(_Completion(completion=text, group_id=group_id))
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
