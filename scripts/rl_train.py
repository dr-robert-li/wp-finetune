"""Phase 9 GSPO/GRPO training loop — Tinker-native RL trainer.

Implements:
  - GSPO sequence-level IS via forward_backward_custom (PRIMARY / D-09-03 locked)
  - RSPO stop-gradient floor: seq_ratio.clamp(min=1.0) before multiply by advantage
  - GRPO token-level IS via forward_backward(loss_fn="importance_sampling") (FALLBACK only)
  - Per-step KL soft/hard autohalt (GRPO-08): kl_soft=0.1, kl_hard=0.3
  - MoE native autohalt: efrac_soft=0.7, efrac_hard=0.5
  - Every-N protected-expert Jaccard monitor (logging only, D-09-02)
  - Persistent checkpoint via save_weights_for_sampler(ttl_seconds=None)
  - Metrics sink: output/rl_checkpoints/metrics/rl_metrics.jsonl (RLEV-01/02)

Security: no hardcoded credentials. ServiceClient() reads ~/.tinker or env.
Router gates FROZEN (no router arg), D-09-02.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Project root on sys.path (so tests/conftest.py project-root insertion works)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — thresholds from GRPO-08
# ---------------------------------------------------------------------------

KL_SOFT_DEFAULT: float = 0.1
KL_HARD_DEFAULT: float = 0.3
EFRAC_SOFT_DEFAULT: float = 0.7
EFRAC_HARD_DEFAULT: float = 0.5

MASK_PATH_DEFAULT: str = (
    "output/profiling/reasoning-merged-v4/protected_expert_mask.npy"
)
MANIFEST_PATH: str = "output/rl_checkpoints/checkpoint_manifest.json"
METRICS_PATH: str = "output/rl_checkpoints/metrics/rl_metrics.jsonl"


# ---------------------------------------------------------------------------
# Utility helpers (mirrors tinker_reasoning_sft.py)
# ---------------------------------------------------------------------------


def _res(f: Any) -> Any:
    """Resolve APIFuture or already-resolved value.

    Only calls .result() on genuine Tinker APIFuture objects. Uses isinstance
    check when tinker is available; falls back gracefully when tinker is absent
    (test/offline path) so MagicMock objects are never called with .result().

    Raises:
        Any exception raised by f.result() (e.g. tinker.TinkerError) so callers
        receive the real error instead of silently getting back an unresolved Future.
    """
    try:
        import tinker as _tinker  # noqa: PLC0415
    except ImportError:
        # Tinker not available (test/offline path): return as-is, never call .result()
        return f
    if isinstance(f, _tinker.APIFuture):
        return f.result()  # let TinkerError propagate — do NOT swallow
    # Not an APIFuture (includes MagicMock, already-resolved values, etc.)
    return f


def _write_manifest(path: str, payload: dict) -> None:
    """Write JSON manifest, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Module-level seam — patchable by test_lora_config
# (must be a plain callable at module scope, not a method)
# ---------------------------------------------------------------------------


def create_lora_training_client(
    base_model: str,
    *,
    rank: int = 32,
    seed: int | None = None,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
) -> Any:
    """Thin wrapper around ServiceClient.create_lora_training_client.

    Separated at module scope so tests can patch rl_train.create_lora_training_client.
    Router gates FROZEN (no router arg), D-09-02.
    """
    import tinker  # noqa: PLC0415 — lazy import; tinker may be absent in tests

    sc = tinker.ServiceClient()
    return sc.create_lora_training_client(
        base_model=base_model,
        rank=rank,
        seed=seed,
        train_mlp=train_mlp,
        train_attn=train_attn,
        train_unembed=train_unembed,
    )


def build_training_client(args: Any) -> Any:
    """Create LoRA training client from parsed args.

    Uses literal True for all three train_* flags (D-09-02: MLP, attn, unembed
    all trained; router gates FROZEN, no router arg passed).
    """
    return create_lora_training_client(
        base_model=args.model_id,
        rank=args.lora_rank,
        seed=args.lora_seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )


# ---------------------------------------------------------------------------
# RSPO floor — sequence-level IS ratio with stop-gradient floor
# ---------------------------------------------------------------------------


def rspo_floored_ratio(train_lp: Any, sampling_lp: Any) -> Any:
    """Compute IS ratio exp(train_lp - sampling_lp) clamped to min=1.0 (RSPO floor).

    Accepts plain floats (for unit tests) or torch tensors (for live training).
    The clamp(min=1.0) implements the RSPO stop-gradient floor: ratios below 1
    (i.e., policy moved AWAY from the sampled completion) are treated as 1 so
    they do not contribute a negative gradient signal.

    Args:
        train_lp: Log-probability under training policy (float or tensor).
        sampling_lp: Log-probability under sampling policy (float or tensor).

    Returns:
        Floored IS ratio, same type as inputs.
    """
    # Handle plain float path (used by test_gspo_rspo_floor)
    if isinstance(train_lp, (int, float)) and isinstance(sampling_lp, (int, float)):
        ratio = math.exp(float(train_lp) - float(sampling_lp))
        return max(1.0, ratio)

    # Tensor path for live training
    import torch  # noqa: PLC0415

    ratio = torch.exp(train_lp - sampling_lp)
    return ratio.clamp(min=1.0)


# ---------------------------------------------------------------------------
# GSPO loss function (closure-based, real SDK signature)
# ---------------------------------------------------------------------------


def _make_gspo_loss_fn(full_data):
    """Return a CustomLossFnV1-compatible loss function for GSPO.

    The SDK calls: loss, metrics = loss_fn(data, logprobs_list)
    where logprobs_list[i] is the per-token TRAINING logprobs tensor for datum i.

    09-07 (corrective): `forward_backward_custom` only accepts datums whose
    loss_fn_inputs keys are a subset of {target_tokens, weights}, so the RL fields
    (mask/advantages/logprobs) CANNOT ride on the datum passed to it. They are read
    from `full_data` (the un-stripped cookbook datums) closed over here, indexed
    position-wise with logprobs_list (build_loss_step strips the datums for the SDK
    but preserves order). `data` (the stripped arg) is intentionally unused.

    Args:
        full_data: list[tinker.Datum] — the FULL cookbook datums (with mask /
            advantages / logprobs), aligned 1:1 with the stripped datums + logprobs_list.

    Returns:
        Callable: gspo_loss_fn(data, logprobs_list) -> (loss_tensor, metrics_dict)
    """

    def gspo_loss_fn(data, logprobs_list):
        import torch  # noqa: PLC0415

        losses = []
        for full_datum, train_lps in zip(full_data, logprobs_list):
            try:
                lfi = full_datum.loss_fn_inputs
                # mask > 0 selects ACTION token positions (obs positions are 0).
                # This is the cookbook convention (rl/metrics.compute_kl_sample_train
                # masks both sampling and training logprobs by mask>0 before use).
                action_mask = lfi["mask"].to_torch() > 0
                adv_weights = lfi["advantages"].to_torch()
                # Select the action-token advantage via the mask, NOT via
                # adv_weights != 0: a legitimately-centered 0.0 advantage on a kept
                # non-constant group would make the nonzero filter empty -> .mean()
                # NaN. An EMPTY mask (immediate-EOS completion, no action tokens)
                # makes the selection empty too — the numel() guard zeroes that
                # datum's contribution instead of IndexError-ing the whole batch.
                sel = adv_weights[action_mask]
                adv = sel[0] if sel.numel() else torch.tensor(0.0)

                # Mask BOTH logprob sums to action tokens BEFORE the IS ratio.
                # train_lps is the SDK's FULL-length per-target-token logprob (obs +
                # action); summing it unmasked would leak obs-token logprobs into the
                # ratio (they are absent from sampling_lps, whose obs positions are 0)
                # and corrupt exp(train_sum - sampling_sum). sampling_lps obs are
                # already 0, but masking it too keeps the two sums symmetric.
                sampling_lps = lfi["logprobs"].to_torch()
                train_sum = train_lps[action_mask].sum()
                sampling_sum = sampling_lps[action_mask].sum()
                seq_ratio = rspo_floored_ratio(train_sum, sampling_sum)
            except (AttributeError, KeyError):
                # Dry-run / mock path: no real Datum structure. NOT reachable on
                # the real datum path (where loss_fn_inputs carries real logprobs).
                seq_ratio = torch.tensor(1.0)
                adv = torch.tensor(0.0)

            # GSPO objective: -ratio * advantage (minimise negative reward)
            seq_loss = -(seq_ratio * adv)
            losses.append(seq_loss)

        if not losses:
            import torch  # noqa: PLC0415

            total_loss = torch.tensor(0.0, requires_grad=False)
        else:
            total_loss = torch.stack(losses).mean()

        metrics = {"gspo/n_sequences": float(len(losses))}
        return total_loss, metrics

    return gspo_loss_fn


# ---------------------------------------------------------------------------
# Loss step — GSPO primary (forward_backward_custom) / GRPO fallback
# ---------------------------------------------------------------------------


def _strip_to_target_tokens(datum: Any) -> Any:
    """Return a Datum carrying ONLY target_tokens (for forward_backward_custom).

    forward_backward_custom's forward validator rejects any loss_fn_inputs key
    outside {target_tokens, weights}; the cookbook datum carries mask/advantages/
    logprobs which the GSPO loss reads via closure instead. Non-Datum inputs
    (dry-run dicts / mocks) are passed through unchanged.
    """
    lfi = getattr(datum, "loss_fn_inputs", None)
    if lfi is None or "target_tokens" not in lfi:
        return datum  # dry-run/mock path — leave as-is
    import tinker  # noqa: PLC0415

    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={"target_tokens": lfi["target_tokens"]},
    )


def _strip_mask(datum: Any) -> Any:
    """Return a Datum without the 'mask' key (for backend forward_backward).

    The backend importance_sampling loss rejects 'mask' (the cookbook applies the
    same strip via _remove_mask). Non-Datum inputs pass through unchanged.
    """
    lfi = getattr(datum, "loss_fn_inputs", None)
    if lfi is None:
        return datum
    import tinker  # noqa: PLC0415

    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in lfi.items() if k != "mask"},
    )


def build_loss_step(
    tc: Any,
    data: Any,
    use_gspo: bool = True,
) -> Any:
    """Execute one forward-backward pass using GSPO (default) or GRPO fallback.

    GSPO (use_gspo=True, D-09-03 locked default):
        Strips each datum to {target_tokens} (forward_backward_custom's accepted
        input) and calls tc.forward_backward_custom(stripped, loss_fn). The GSPO
        loss closes over the FULL datums to read mask/advantages/sampling-logprobs
        and computes the RSPO-floored sequence-level IS ratio client-side.

    GRPO (use_gspo=False, --grpo-fallback):
        Strips 'mask' and calls tc.forward_backward(stripped, loss_fn="importance_sampling")
        (backend token-level loss; rejects the mask key).

    Args:
        tc: Tinker training client (or mock).
        data: List of full cookbook tinker.Datum objects (or mock).
        use_gspo: True = GSPO primary path; False = GRPO token-level fallback.

    Returns:
        ForwardBackwardOutput (or mock equivalent).
    """
    if not use_gspo:
        # GRPO token-level IS fallback (--grpo-fallback / --no-gspo)
        return _res(
            tc.forward_backward(
                [_strip_mask(d) for d in data], loss_fn="importance_sampling"
            )
        )

    # GSPO sequence-level IS (primary path, D-09-03): client-side loss over the FULL
    # datums; forward pass gets target-tokens-only datums (validator constraint).
    loss_fn = _make_gspo_loss_fn(data)
    fwd_data = [_strip_to_target_tokens(d) for d in data]
    return _res(tc.forward_backward_custom(fwd_data, loss_fn))


# ---------------------------------------------------------------------------
# KL computation seam (patchable; CR-04)
# ---------------------------------------------------------------------------

#: Sentinel KL value used when KL computation fails. It is set ABOVE any sane
#: kl_hard threshold so a compute failure trips a HARD halt rather than reading
#: as "perfect KL, never halt" (the CR-04 silent-0.0 swallow). check_halt then
#: stops training before optim_step commits a potentially divergent update.
_KL_COMPUTE_FAILED_SENTINEL: float = 1e9


def _compute_kl_metrics(fb_out: Any, data: list) -> dict[str, float]:
    """Compute kl_sample_train metrics from a ForwardBackwardOutput + data.

    Module-level seam (patchable by the integration test, mirroring
    create_lora_training_client) so a synthetic divergent KL can be injected
    without real Datum/logprob tensors.

    CR-04 contract: a KL COMPUTE FAILURE must NOT read as kl_v1=0.0 (which would
    disable the autohalt guard). On failure this returns the
    _KL_COMPUTE_FAILED_SENTINEL for kl_sample_train_v1, which is above any sane
    kl_hard and therefore forces a HARD halt in check_halt.

    The genuinely-empty case (no training logprobs available yet, e.g. a mock
    ForwardBackwardOutput) returns 0.0 — that is a structural absence of data,
    not a computation that errored.
    """
    training_lps = getattr(fb_out, "training_logprobs", []) or []
    if not data or not training_lps:
        return {
            "optim/kl_sample_train_v1": 0.0,
            "optim/kl_sample_train_v2": 0.0,
            "optim/entropy": 0.0,
        }
    try:
        from tinker_cookbook.rl.metrics import compute_kl_sample_train  # noqa: PLC0415

        return compute_kl_sample_train(data, training_lps)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "KL metric computation FAILED (%s) — treating as HARD halt "
            "(CR-04: a compute failure must not read as perfect KL)",
            exc,
        )
        return {
            "optim/kl_sample_train_v1": _KL_COMPUTE_FAILED_SENTINEL,
            "optim/kl_sample_train_v2": _KL_COMPUTE_FAILED_SENTINEL,
            "optim/entropy": 0.0,
        }


# ---------------------------------------------------------------------------
# GRPO-08: Per-step autohalt guards
# ---------------------------------------------------------------------------


def check_halt(
    kl_v1: float,
    e_frac: float,
    kl_soft: float = KL_SOFT_DEFAULT,
    kl_hard: float = KL_HARD_DEFAULT,
    efrac_soft: float = EFRAC_SOFT_DEFAULT,
    efrac_hard: float = EFRAC_HARD_DEFAULT,
) -> str | None:
    """Check KL and MoE routing health; return halt reason string or None.

    Soft thresholds → log WARNING only (no halt).
    Hard thresholds → return non-None halt reason string (caller halts).

    KL thresholds (kl_sample_train_v1):
      kl_v1 > kl_hard  → HARD halt
      kl_v1 > kl_soft  → soft alert

    MoE routing (e_frac_with_tokens:mean — fraction of tokens with any expert):
      e_frac < efrac_hard → HARD halt (routing collapsed)
      e_frac < efrac_soft → soft alert

    Args:
        kl_v1: KL divergence v1 metric from compute_kl_sample_train.
        e_frac: e_frac_with_tokens:mean from ForwardBackwardOutput.metrics.
        kl_soft, kl_hard: KL alert / halt thresholds.
        efrac_soft, efrac_hard: e_frac alert / halt thresholds (lower = worse).

    Returns:
        Non-None halt reason string if any HARD threshold is breached,
        None otherwise.
    """
    # KL checks
    if kl_v1 > kl_hard:
        return f"KL HARD halt: kl_sample_train_v1={kl_v1:.4f} > {kl_hard}"
    if kl_v1 > kl_soft:
        logger.warning(
            "KL soft alert: kl_sample_train_v1=%.4f > %.4f", kl_v1, kl_soft
        )

    # MoE routing checks (lower e_frac = fewer experts used = routing collapse)
    if e_frac < efrac_hard:
        return f"MoE HARD halt: e_frac_with_tokens:mean={e_frac:.4f} < {efrac_hard}"
    if e_frac < efrac_soft:
        logger.warning(
            "MoE soft alert: e_frac_with_tokens:mean=%.4f < %.4f", e_frac, efrac_soft
        )

    return None


# ---------------------------------------------------------------------------
# Protected-expert Jaccard monitor (D-09-02, logging only, no enforcement)
# ---------------------------------------------------------------------------


def protected_mask_jaccard(
    active_experts: np.ndarray,
    mask_path: str = MASK_PATH_DEFAULT,
) -> float:
    """Compute Jaccard similarity between active routing and protected-expert mask.

    Monitor-only (D-09-02): result is logged but never used to gate training.
    Called every-N steps from the main loop.

    Args:
        active_experts: Boolean array [n_layers, n_experts] of currently-active
            experts (from routing stats or ForwardBackwardOutput).
        mask_path: Path to the protected_expert_mask.npy file, shape [48, 128].

    Returns:
        float: Jaccard score in [0.0, 1.0]. Returns 0.0 if mask file missing.
    """
    mask_file = Path(mask_path) if not Path(mask_path).is_absolute() else Path(mask_path)
    # Resolve relative to project root if relative
    if not mask_file.is_absolute():
        mask_file = _PROJECT_ROOT / mask_path

    if not mask_file.exists():
        logger.warning(
            "protected_mask_jaccard: mask file not found at %s — returning 0.0",
            mask_file,
        )
        return 0.0

    protected = np.load(str(mask_file)).astype(bool)

    # Align shapes: truncate to min dimensions if mismatch
    min_layers = min(active_experts.shape[0], protected.shape[0])
    min_experts = min(active_experts.shape[1], protected.shape[1])
    active = active_experts[:min_layers, :min_experts].astype(bool)
    prot = protected[:min_layers, :min_experts]

    intersection = int(np.logical_and(active, prot).sum())
    union = int(np.logical_or(active, prot).sum())

    if union == 0:
        return 0.0

    return float(intersection / union)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


def _save_checkpoint(tc: Any, name: str, manifest: dict) -> str:
    """Save persistent weights and update manifest. Returns sampler path."""
    resp = _res(tc.save_weights_for_sampler(name=name, ttl_seconds=None))
    sampler_path = resp.path if hasattr(resp, "path") else str(resp)
    manifest["checkpoints"].append(
        {
            "name": name,
            "sampler_path": sampler_path,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _write_manifest(MANIFEST_PATH, manifest)
    logger.info("Checkpoint saved: %s -> %s", name, sampler_path)
    return sampler_path


# ---------------------------------------------------------------------------
# Metrics sink — RLEV-01/02
# ---------------------------------------------------------------------------


def _log_step(
    step: int,
    rewards: list[float],
    kl_metrics: dict[str, float],
    moe_metrics: dict[str, float],
    args: Any,
    jaccard: float | None = None,
    halt_reason: str | None = None,
) -> None:
    """Append one JSONL row to rl_metrics.jsonl (RLEV-01 / RLEV-02 fields).

    Written even in dry-run mode so Phase 10 consumer can be tested.
    """
    record = {
        "step": step,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_breakdown": {
            "n_samples": len(rewards),
            "reward_min": float(min(rewards)) if rewards else 0.0,
            "reward_max": float(max(rewards)) if rewards else 0.0,
        },
        # RLEV-01: KL metrics
        "kl_sample_train_v1": kl_metrics.get("optim/kl_sample_train_v1", 0.0),
        "kl_sample_train_v2": kl_metrics.get("optim/kl_sample_train_v2", 0.0),
        # RLEV-02: MoE routing metrics
        "e_frac_with_tokens_mean": moe_metrics.get("e_frac_with_tokens:mean", 0.0),
        "e_max_violation_mean": moe_metrics.get("e_max_violation:mean", 0.0),
        "e_max_violation_max": moe_metrics.get("e_max_violation:max", 0.0),
        # Optional extras
        "jaccard_protected": jaccard,
        "halt_reason": halt_reason,
        "use_gspo": getattr(args, "use_gspo", True),
        "model_id": getattr(args, "model_id", None),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path = Path(METRICS_PATH)
    os.makedirs(str(metrics_path.parent), exist_ok=True)
    with open(str(metrics_path), "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Panickssery divergence spot-check (D-09-05 R1, logging only)
# ---------------------------------------------------------------------------


def _panickssery_spot_check(trajectory_groups: list, step: int) -> None:
    """Log rollouts where judge_consistency diverges from fix_correctness by >0.3.

    Called every ~50 steps. Monitor-only per D-09-05 R1.

    09-07: the reward breakdown no longer rides on a Datum dict — it is stashed in
    each Transition.logs (group_id, origin, fix_correctness, consistency) by
    build_trajectory_groups, so this reads the pre-Datum trajectory groups.
    """
    divergent = []
    for tg in trajectory_groups:
        for traj in getattr(tg, "trajectories_G", []):
            for transition in getattr(traj, "transitions", []):
                logs = getattr(transition, "logs", {}) or {}
                fix_corr = logs.get("fix_correctness")
                consistency = logs.get("consistency")
                if fix_corr is None or consistency is None:
                    continue
                if abs(float(fix_corr) - float(consistency)) > 0.3:
                    divergent.append(
                        {
                            "group_id": logs.get("group_id", ""),
                            "fix_correctness": fix_corr,
                            "judge_consistency": consistency,
                        }
                    )
    if divergent:
        logger.warning(
            "Panickssery step %d: %d rollouts with |fix_corr - consistency| > 0.3: %s",
            step,
            len(divergent),
            divergent[:3],
        )


# ---------------------------------------------------------------------------
# Single training step seam (CR-04 ordering; testable without main())
# ---------------------------------------------------------------------------


def _adam_params(args: Any) -> Any:
    """Build tinker.AdamParams from args (TrainingClient.optim_step requires it)."""
    import tinker  # noqa: PLC0415

    lr = float(getattr(args, "learning_rate", 1e-5))
    return tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)


def run_training_step(
    step: int,
    tc: Any,
    sampling_client: Any,
    gen_pool: list,
    judge_pool: list,
    args: Any,
    manifest: dict,
) -> bool:
    """Run ONE RL step: rollouts -> advantages -> loss -> KL/halt -> optim/ckpt.

    Extracted as a seam so the real loop body is exercisable by an integration
    test driving the mock client over NON-EMPTY pools (the unit tests bypass
    this wiring entirely, which is how CR-01/CR-02/CR-04 stayed green).

    CR-04 ordering is the load-bearing invariant: forward_backward -> compute KL
    -> check_halt; on a HARD breach we save an emergency checkpoint and return
    True (halt) WITHOUT calling tc.optim_step(), so the divergent update is never
    committed. tc.optim_step() runs ONLY on the safe path.

    Returns:
        bool: True if a HARD halt fired this step (caller stops), else False.
    """
    from scripts.rl_rollouts import (  # noqa: PLC0415
        collect_rollouts,
        compute_rollout_advantages,
    )

    rollouts = collect_rollouts(
        sampling_client=sampling_client,
        gen_pool=gen_pool,
        judge_pool=judge_pool,
        args=args,
    )

    data, _advantages, _meta = compute_rollout_advantages(rollouts)

    if not data:
        logger.warning("Step %d: no training data after advantage filter", step)
        return False

    # Rewards for the metrics sink come from the trajectory groups' get_total_rewards
    # (real Datums carry baked advantages, not raw rewards). Pre-filter rollouts are
    # fine for a monitoring mean/min/max.
    rewards = [r for tg in rollouts for r in tg.get_total_rewards()]

    # Panickssery divergence spot-check every ~50 steps (D-09-05 R1) — reads the
    # pre-Datum trajectory groups (fix_correctness/consistency live in Transition.logs).
    if step % 50 == 0:
        _panickssery_spot_check(rollouts, step)

    # Forward-backward (GSPO primary or GRPO fallback). NOTE: NO optim_step yet —
    # the gradient is computed but NOT applied until after the halt check (CR-04).
    fb_out = build_loss_step(tc, data, use_gspo=args.use_gspo)

    # KL metrics BEFORE committing the update. A compute failure returns a
    # halt-worthy sentinel (never a silent 0.0) so the guard cannot be disabled.
    kl_metrics = _compute_kl_metrics(fb_out, data)
    moe_metrics = getattr(fb_out, "metrics", {}) or {}

    # Autohalt guard (GRPO-08) — evaluated BEFORE optim_step.
    halt_reason = check_halt(
        kl_v1=kl_metrics.get("optim/kl_sample_train_v1", 0.0),
        e_frac=moe_metrics.get("e_frac_with_tokens:mean", 1.0),
        kl_soft=args.kl_soft,
        kl_hard=args.kl_hard,
        efrac_soft=args.efrac_soft,
        efrac_hard=args.efrac_hard,
    )

    # Protected-expert Jaccard monitor (every N steps, logging only)
    jaccard: float | None = None
    if step % args.jaccard_every == 0:
        active = np.zeros((48, 128), dtype=bool)
        jaccard = protected_mask_jaccard(active, mask_path=args.mask_path)
        logger.info("Step %d: protected-expert Jaccard=%.4f", step, jaccard)

    # Metrics sink (written for every step, including the halting step)
    _log_step(
        step=step,
        rewards=rewards,
        kl_metrics=kl_metrics,
        moe_metrics=moe_metrics,
        args=args,
        jaccard=jaccard,
        halt_reason=halt_reason,
    )

    # CR-04: HARD halt -> save emergency checkpoint of the PRE-update weights and
    # stop WITHOUT committing the divergent gradient. optim_step is NOT called.
    if halt_reason is not None:
        logger.error("HALT at step %d: %s", step, halt_reason)
        _save_checkpoint(tc, name=f"emergency-halt-step-{step}", manifest=manifest)
        return True

    # Safe path only: commit the gradient update. The real TrainingClient.optim_step
    # requires AdamParams (no-arg crashes); construct from args.learning_rate.
    tc.optim_step(_adam_params(args))

    # Scheduled checkpoint
    if (step + 1) % args.checkpoint_every == 0:
        _save_checkpoint(tc, name=f"step-{step + 1}", manifest=manifest)

    logger.info(
        "Step %d/%d: reward_mean=%.4f, kl_v1=%.4f",
        step + 1,
        args.total_steps,
        float(np.mean(rewards)) if rewards else 0.0,
        kl_metrics.get("optim/kl_sample_train_v1", 0.0),
    )
    return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9 GSPO/GRPO RL training loop (Tinker-native)"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-30B-A3B",
        help=(
            "Base model ID for Tinker LoRA client. MUST be a Tinker-supported model "
            "name — the MoE 'Qwen/Qwen3-30B-A3B' (NOT 'Qwen/Qwen3-30B', which Tinker "
            "rejects with a 400 'not supported')."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-seed", type=int, default=42)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Adam learning rate for optim_step (TrainingClient.optim_step AdamParams).",
    )
    parser.add_argument(
        "--total-steps", type=int, default=500, help="Total RL gradient steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Rollout batch size"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Completions sampled per prompt (GRPO group size G)",
    )
    parser.add_argument(
        "--max-pool",
        type=int,
        default=None,
        help="Cap each prompt pool to this many items (dry-run / smoke test)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--jaccard-every",
        type=int,
        default=20,
        help="Compute protected-expert Jaccard every N steps",
    )
    parser.add_argument(
        "--protected-expert-mask",
        dest="mask_path",
        default=MASK_PATH_DEFAULT,
        help="Path to protected_expert_mask.npy",
    )
    parser.add_argument(
        "--kl-soft", type=float, default=KL_SOFT_DEFAULT
    )
    parser.add_argument(
        "--kl-hard", type=float, default=KL_HARD_DEFAULT
    )
    parser.add_argument(
        "--efrac-soft", type=float, default=EFRAC_SOFT_DEFAULT
    )
    parser.add_argument(
        "--efrac-hard", type=float, default=EFRAC_HARD_DEFAULT
    )
    # GSPO primary (default) / GRPO fallback flag
    parser.add_argument(
        "--grpo-fallback",
        "--no-gspo",
        dest="use_gspo",
        action="store_false",
        help="Use GRPO token-level IS fallback instead of GSPO (not recommended)",
    )
    parser.set_defaults(use_gspo=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one synthetic step with mock client; write metrics; exit 0",
    )
    # Judge args: required by rl_rollouts.collect_rollouts on the live path.
    # --judge-model: served model name passed to judge_score_single / compute_group_rewards.
    # Default "wp_judge" matches the vLLM endpoint convention used in build_antihack_set.py
    # and eval_judge.judge_score_single (which accepts a served-model name, not an HF id).
    parser.add_argument(
        "--judge-model",
        default="wp_judge",
        help=(
            "Served model name for the vLLM judge endpoint "
            "(passed to compute_group_rewards / judge_score_single). "
            "Default: 'wp_judge'."
        ),
    )
    # --consistency-model: Claude model for score_judge_consistency_batch (09-03).
    # getattr(args, 'consistency_model', 'sonnet') guards already exist in rl_rollouts,
    # but adding the arg here makes the default explicit and discoverable via --help.
    parser.add_argument(
        "--consistency-model",
        default="sonnet",
        help=(
            "Claude model slug for score_judge_consistency_batch (09-03 judge path). "
            "Default: 'sonnet'."
        ),
    )
    # --n-votes: N-vote median for the consistency scorer (default 1 = single call).
    parser.add_argument(
        "--n-votes",
        type=int,
        default=1,
        help=(
            "Number of votes for score_judge_consistency_batch median (09-03). "
            "Default: 1."
        ),
    )
    # Judge endpoint: on the live path the reward uses an openai.OpenAI client
    # pointed at a vLLM judge endpoint (eval_judge.judge_score_single). When
    # --judge-base-url is given, main() constructs the client itself so the
    # documented `python scripts/rl_train.py ...` command runs end-to-end
    # without a hand-built wrapper. A caller may still pre-attach
    # args.judge_client (decoupled/test path); that takes precedence.
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help=(
            "Base URL of the vLLM judge endpoint (e.g. http://localhost:8000/v1). "
            "When set, main() builds args.judge_client from it on the live path. "
            "If omitted, a caller must pre-attach args.judge_client before main()."
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default="EMPTY",
        help="API key for the vLLM judge endpoint (vLLM ignores it; default 'EMPTY').",
    )
    # Output-path overrides — let a smoke/test run isolate its outputs from the
    # canonical (git-tracked) manifest/metrics that Phase 10 D-10-02 reads.
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=f"Override checkpoint manifest path (default: {MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help=f"Override rl_metrics.jsonl path (default: {METRICS_PATH}).",
    )
    return parser.parse_args(argv)


def _dry_run(args: argparse.Namespace) -> None:
    """Execute one synthetic GSPO step, write real metrics row, exit cleanly.

    Uses a minimal mock client so no Tinker credentials are required.
    Still exercises: rspo_floored_ratio, build_loss_step (GSPO path),
    check_halt, protected_mask_jaccard, _log_step.
    """
    from unittest.mock import MagicMock  # noqa: PLC0415

    logger.info("DRY RUN: one synthetic GSPO step")

    # Build mock Tinker client
    tc = MagicMock()
    fb_out = MagicMock()
    fb_out.metrics = {
        "e_frac_with_tokens:mean": 0.75,
        "e_max_violation:mean": 0.001,
        "e_max_violation:max": 0.005,
    }
    fb_out.training_logprobs = []
    tc.forward_backward_custom.return_value = fb_out
    tc.forward_backward.return_value = fb_out
    tc.optim_step.return_value = None
    tc.save_weights_for_sampler.return_value.path = "/dry-run/sampler"

    # Synthetic data (2 items with uniform advantages)
    data = [{"prompt": "dry", "completion": "run", "reward": 1.0, "advantage": 0.5},
            {"prompt": "dry", "completion": "ok", "reward": 0.8, "advantage": -0.5}]

    rewards = [float(d.get("reward", 0.0)) for d in data]

    # GSPO forward-backward (primary path). dry-run data is plain dicts -> the loss
    # fn's AttributeError fallback path is taken (forward_backward_custom is mocked
    # here anyway, so the loss fn is never actually invoked).
    fb_result = build_loss_step(tc, data, use_gspo=True)
    tc.optim_step()

    # KL metrics (synthetic in dry-run — no real Datum objects)
    kl_metrics = {
        "optim/kl_sample_train_v1": 0.02,
        "optim/kl_sample_train_v2": 0.0002,
        "optim/entropy": 2.5,
    }

    # In dry-run the fb_result is a MagicMock; use the explicit dict we configured above
    moe_metrics = fb_out.metrics

    # Autohalt check
    halt_reason = check_halt(
        kl_v1=kl_metrics["optim/kl_sample_train_v1"],
        e_frac=moe_metrics.get("e_frac_with_tokens:mean", 1.0),
        kl_soft=args.kl_soft,
        kl_hard=args.kl_hard,
        efrac_soft=args.efrac_soft,
        efrac_hard=args.efrac_hard,
    )

    # Protected-expert Jaccard (synthetic mask)
    synthetic_active = np.zeros((48, 128), dtype=bool)
    synthetic_active[0, :5] = True
    jaccard = protected_mask_jaccard(synthetic_active, mask_path=args.mask_path)
    logger.info("DRY RUN jaccard=%.4f", jaccard)

    # Write metrics row (real file, not skipped)
    _log_step(
        step=0,
        rewards=rewards,
        kl_metrics=kl_metrics,
        moe_metrics=moe_metrics,
        args=args,
        jaccard=jaccard,
        halt_reason=halt_reason,
    )

    # Checkpoint
    manifest: dict = {"checkpoints": [], "run_args": vars(args)}
    _save_checkpoint(tc, name="dry-run-step-0", manifest=manifest)

    logger.info(
        "DRY RUN complete — halt_reason=%s, metrics written to %s",
        halt_reason,
        METRICS_PATH,
    )


def _apply_path_overrides(args: argparse.Namespace) -> None:
    """Reassign the module-global manifest/metrics paths from CLI overrides.

    _save_checkpoint and _log_step read MANIFEST_PATH / METRICS_PATH at call time,
    so reassigning the globals here (before the loop) is sufficient to redirect a
    smoke/test run's outputs away from the canonical, git-tracked artifacts.
    """
    global MANIFEST_PATH, METRICS_PATH  # noqa: PLW0603
    if getattr(args, "manifest_path", None):
        MANIFEST_PATH = args.manifest_path
    if getattr(args, "metrics_path", None):
        METRICS_PATH = args.metrics_path


def _build_judge_client(base_url: str, api_key: str = "EMPTY") -> Any:
    """Construct an openai.OpenAI judge client pointed at a vLLM endpoint.

    This is the client compute_group_rewards / judge_score_single expect. Lazy
    import so the module stays importable (and dry-run/unit tests run) without
    the openai package installed.
    """
    import openai  # noqa: PLC0415

    return openai.OpenAI(base_url=base_url, api_key=api_key)


def run_training(args: argparse.Namespace) -> None:
    """Run the live GSPO RL loop. Assumes args.judge_client is already attached.

    Extracted from main() so the live orchestration (manifest init, sampling
    client, per-step loop, final checkpoint) is a single code path shared by the
    CLI entrypoint and any programmatic caller — a smoke run exercises exactly
    what a full run does, with no drift.
    """
    logger.info(
        "Starting GSPO RL training: model=%s, steps=%d, use_gspo=%s",
        args.model_id,
        args.total_steps,
        args.use_gspo,
    )

    tc = build_training_client(args)

    # Initialise checkpoint manifest. Exclude judge_client — it is a live
    # openai.OpenAI object and is not JSON-serializable.
    manifest: dict = {
        "checkpoints": [],
        "run_args": {k: v for k, v in vars(args).items() if k != "judge_client"},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    _write_manifest(MANIFEST_PATH, manifest)

    # CR-02: load the real prompt pools (load_rl_prompts returns {"messages":[...]}
    # items). They were previously hardcoded empty, so every step produced no
    # rollouts. BASE_MODEL pulled from the shared data module for logging parity.
    from scripts.tinker_rl_data import load_rl_prompts  # noqa: PLC0415

    gen_pool = load_rl_prompts("gen")
    judge_pool = load_rl_prompts("judge")
    if getattr(args, "max_pool", None):
        gen_pool = gen_pool[: args.max_pool]
        judge_pool = judge_pool[: args.max_pool]
    logger.info(
        "Loaded prompt pools: gen=%d, judge=%d (group_size=%d)",
        len(gen_pool),
        len(judge_pool),
        getattr(args, "group_size", 4),
    )

    # CR-01: obtain a SAMPLING CLIENT (supports .sample(...)), not a checkpoint
    # ref. save_weights_and_get_sampling_client() returns a SamplingClient with
    # the current weights. The persistent-checkpoint save_weights_for_sampler is
    # a separate concern, handled inside _save_checkpoint.
    sampling_client = _res(tc.save_weights_and_get_sampling_client())

    for step in range(args.total_steps):
        halted = run_training_step(
            step=step,
            tc=tc,
            sampling_client=sampling_client,
            gen_pool=gen_pool,
            judge_pool=judge_pool,
            args=args,
            manifest=manifest,
        )
        if halted:
            raise RuntimeError(f"Training halted at step {step}")

    # Final checkpoint
    _save_checkpoint(tc, name=f"final-step-{args.total_steps}", manifest=manifest)
    logger.info("Training complete. %d steps. Manifest: %s", args.total_steps, MANIFEST_PATH)


def main(argv: list[str] | None = None) -> None:
    """Main RL training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    _apply_path_overrides(args)

    if args.dry_run:
        _dry_run(args)
        return

    # --- Live training path ---

    # Ensure a judge client is attached before entering the live loop.
    # rl_rollouts.collect_rollouts reads args.judge_client directly (hard attribute
    # access). Precedence: a caller-attached args.judge_client wins (decoupled/test
    # path); otherwise build one from --judge-base-url; otherwise fail fast with a
    # clear message rather than an AttributeError deep inside the rollout loop.
    if getattr(args, "judge_client", None) is None:
        if getattr(args, "judge_base_url", None):
            args.judge_client = _build_judge_client(
                args.judge_base_url, getattr(args, "judge_api_key", "EMPTY")
            )
            logger.info(
                "Built judge client -> %s (judge_model=%s)",
                args.judge_base_url,
                args.judge_model,
            )
        else:
            args.judge_client = None
    if args.judge_client is None:
        raise SystemExit(
            "rl_train: live run requires a judge client. Either pass "
            "--judge-base-url http://<host>:<port>/v1 (main() builds the client), "
            "or pre-attach an openai.OpenAI instance to args.judge_client before "
            "calling run_training(args). "
            "See rl_rollouts.collect_rollouts -> compute_group_rewards for the "
            "judge_client interface."
        )

    run_training(args)


if __name__ == "__main__":
    main()
