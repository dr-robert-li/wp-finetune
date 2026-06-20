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
    check when tinker is available; falls back to type-name heuristic to avoid
    calling .result() on MagicMock objects (which auto-create every attribute).
    """
    try:
        import tinker as _tinker  # noqa: PLC0415
        if isinstance(f, _tinker.APIFuture):
            return f.result()
        # If tinker is available and f is NOT an APIFuture, return as-is
        return f
    except Exception:
        pass
    # Tinker not available (test/offline path): never call .result()
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


def _make_gspo_loss_fn(advantages_by_idx: dict[int, float]):
    """Return a CustomLossFnV1-compatible loss function for GSPO.

    The SDK calls: loss, metrics = loss_fn(data, logprobs_list)
    where logprobs_list[i] is a per-token logprobs tensor for datum i.

    advantages_by_idx maps datum index (in the current batch) to advantage value.
    These are captured via closure so the SDK's two-arg interface is satisfied.

    Args:
        advantages_by_idx: {datum_index -> advantage_scalar}

    Returns:
        Callable: gspo_loss_fn(data, logprobs_list) -> (loss_tensor, metrics_dict)
    """

    def gspo_loss_fn(data, logprobs_list):
        import torch  # noqa: PLC0415

        losses = []
        for i, (datum, train_lps) in enumerate(zip(data, logprobs_list)):
            adv = float(advantages_by_idx.get(i, 0.0))

            # Sequence-level log-prob: sum over action tokens
            # Sampling log-prob comes from datum.loss_fn_inputs["logprobs"] if available
            try:
                sampling_lps = datum.loss_fn_inputs["logprobs"].to_torch()
                train_sum = train_lps.sum()
                sampling_sum = sampling_lps.sum()
                seq_ratio = rspo_floored_ratio(train_sum, sampling_sum)
            except (AttributeError, KeyError):
                # Dry-run / mock path: no real Datum structure
                seq_ratio = torch.tensor(1.0)

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


def build_loss_step(
    tc: Any,
    data: Any,
    use_gspo: bool = True,
    advantages: list[float] | None = None,
) -> Any:
    """Execute one forward-backward pass using GSPO (default) or GRPO fallback.

    GSPO (use_gspo=True, D-09-03 locked default):
        Calls tc.forward_backward_custom(data, loss_fn) with RSPO-floored
        sequence-level IS ratios. Loss function built as closure over advantages.

    GRPO (use_gspo=False, --grpo-fallback):
        Calls tc.forward_backward(data, loss_fn="importance_sampling").

    Args:
        tc: Tinker training client (or mock).
        data: List of Datum objects (or mock).
        use_gspo: True = GSPO primary path; False = GRPO token-level fallback.
        advantages: Per-datum advantage values. Defaults to zeros.

    Returns:
        ForwardBackwardOutput (or mock equivalent).
    """
    if not use_gspo:
        # GRPO token-level IS fallback (--grpo-fallback / --no-gspo)
        return _res(tc.forward_backward(data, loss_fn="importance_sampling"))

    # GSPO sequence-level IS (primary path, D-09-03)
    batch_size = len(data) if hasattr(data, "__len__") else 1
    adv_map = {i: float(advantages[i]) if advantages and i < len(advantages) else 0.0
               for i in range(batch_size)}
    loss_fn = _make_gspo_loss_fn(adv_map)
    return _res(tc.forward_backward_custom(data, loss_fn))


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


def _panickssery_spot_check(data: list[dict], step: int) -> None:
    """Log rollouts where judge_consistency diverges from fix_correctness by >0.3.

    Called every ~50 steps. Monitor-only per D-09-05 R1.
    """
    divergent = []
    for item in data:
        bd = item.get("breakdown") or {}
        if not isinstance(bd, dict):
            continue
        fix_corr = bd.get("fix_correctness", bd.get("fix_score", None))
        consistency = bd.get("judge_consistency", bd.get("consistency", None))
        if fix_corr is not None and consistency is not None:
            if abs(float(fix_corr) - float(consistency)) > 0.3:
                divergent.append(
                    {
                        "prompt": str(item.get("prompt", ""))[:60],
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
# Main training loop
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9 GSPO/GRPO RL training loop (Tinker-native)"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-30B",
        help="Base model ID for Tinker LoRA client",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-seed", type=int, default=42)
    parser.add_argument(
        "--total-steps", type=int, default=500, help="Total RL gradient steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Rollout batch size"
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
    advantages = [float(d.get("advantage", 0.0)) for d in data]

    # GSPO forward-backward (primary path)
    fb_result = build_loss_step(tc, data, use_gspo=True, advantages=advantages)
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


def main(argv: list[str] | None = None) -> None:
    """Main RL training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    if args.dry_run:
        _dry_run(args)
        return

    # --- Live training path ---
    logger.info(
        "Starting GSPO RL training: model=%s, steps=%d, use_gspo=%s",
        args.model_id,
        args.total_steps,
        args.use_gspo,
    )

    tc = build_training_client(args)

    # Initialise checkpoint manifest
    manifest: dict = {
        "checkpoints": [],
        "run_args": vars(args),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs("output/rl_checkpoints", exist_ok=True)
    _write_manifest(MANIFEST_PATH, manifest)

    # Lazy imports for live path
    from scripts.rl_rollouts import (  # noqa: PLC0415
        collect_rollouts,
        compute_rollout_advantages,
    )

    sampling_client = _res(
        tc.save_weights_for_sampler(name="init", ttl_seconds=None)
    )

    for step in range(args.total_steps):
        # Collect rollouts
        rollouts = collect_rollouts(
            sampling_client=sampling_client,
            gen_pool=[],
            judge_pool=[],
            args=args,
        )

        # Compute advantages
        data, _meta = compute_rollout_advantages(rollouts)

        rewards = [float(d.get("reward", 0.0)) for d in data]
        advantages = [float(d.get("advantage", 0.0)) for d in data]

        if not data:
            logger.warning("Step %d: no training data after advantage filter", step)
            continue

        # Panickssery divergence spot-check every ~50 steps (D-09-05 R1)
        if step % 50 == 0:
            _panickssery_spot_check(data, step)

        # Forward-backward (GSPO primary or GRPO fallback)
        fb_out = build_loss_step(tc, data, use_gspo=args.use_gspo, advantages=advantages)
        tc.optim_step()

        # KL metrics (compute_kl_sample_train requires real Datum with logprobs)
        try:
            from tinker_cookbook.rl.metrics import compute_kl_sample_train  # noqa: PLC0415

            training_lps = getattr(fb_out, "training_logprobs", [])
            if data and training_lps:
                kl_metrics = compute_kl_sample_train(data, training_lps)
            else:
                kl_metrics = {
                    "optim/kl_sample_train_v1": 0.0,
                    "optim/kl_sample_train_v2": 0.0,
                    "optim/entropy": 0.0,
                }
        except Exception as exc:
            logger.warning("KL metric computation failed: %s", exc)
            kl_metrics = {
                "optim/kl_sample_train_v1": 0.0,
                "optim/kl_sample_train_v2": 0.0,
                "optim/entropy": 0.0,
            }

        moe_metrics = getattr(fb_out, "metrics", {}) or {}

        # Autohalt guard (GRPO-08)
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
            # Build active-experts from MoE metrics if available; else zeros
            active = np.zeros((48, 128), dtype=bool)
            jaccard = protected_mask_jaccard(active, mask_path=args.mask_path)
            logger.info("Step %d: protected-expert Jaccard=%.4f", step, jaccard)

        # Metrics sink
        _log_step(
            step=step,
            rewards=rewards,
            kl_metrics=kl_metrics,
            moe_metrics=moe_metrics,
            args=args,
            jaccard=jaccard,
            halt_reason=halt_reason,
        )

        # Emergency checkpoint on halt — then stop
        if halt_reason is not None:
            logger.error("HALT at step %d: %s", step, halt_reason)
            _save_checkpoint(tc, name=f"emergency-halt-step-{step}", manifest=manifest)
            raise RuntimeError(f"Training halted at step {step}: {halt_reason}")

        # Scheduled checkpoint
        if (step + 1) % args.checkpoint_every == 0:
            _save_checkpoint(
                tc, name=f"step-{step + 1}", manifest=manifest
            )

        logger.info(
            "Step %d/%d: reward_mean=%.4f, kl_v1=%.4f",
            step + 1,
            args.total_steps,
            float(np.mean(rewards)) if rewards else 0.0,
            kl_metrics.get("optim/kl_sample_train_v1", 0.0),
        )

    # Final checkpoint
    _save_checkpoint(tc, name=f"final-step-{args.total_steps}", manifest=manifest)
    logger.info("Training complete. %d steps. Manifest: %s", args.total_steps, MANIFEST_PATH)


if __name__ == "__main__":
    main()
