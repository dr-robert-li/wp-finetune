"""Reward pipeline for Phase 8 GRPO training.

This module provides:
  - _load_score_offset(path): injectable recalibration loader (D-08-02)
  - _SCORE_OFFSET: module-level singleton loaded at import time
  - _apply_offset_clip(raw_judge): applies offset then clips to [0, 100]
  - _mo_grpo_norm(values): MO-GRPO within-group normalization (GRPO-03)
  - RewardBreakdown / RewardResult: (scalar, breakdown_dict) contract (D-08-04)
  - _verpo_group(rubrics): VeRPO difficulty-weighted scores on WP-standards (GRPO-04)
  - _extract_verifiable_signals(php_code): wraps score_code() for rubric signals
  - WP_STANDARDS_CHECK_IDS: D1_wpcs + D5_wp_api check ids for VeRPO scope (D-08-06)

Wave 3 (08-03) adds: _security_fail, full security gate, composite weights,
compute_reward, compute_group_rewards.

CRITICAL: LLM-checks are suppressed at module load (deterministic reward
signals only in training — Pitfall 6 from 08-PATTERNS.md).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure deterministic reward compute — do NOT run LLM checks in training.
# This must come BEFORE any eval.rubric_scorer import.
os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)

# ---------------------------------------------------------------------------
# Recalibration offset — loaded from artifact, NEVER hardcoded
# ---------------------------------------------------------------------------

# Source of truth: output/eval_reasoning_v4_winner/judge_recalibration.json
# Contains {"score_offset": <float>, ...} where the float is the +3.58 paired
# mean correction for the bf16-merge inference path (D-08-02 / D-V4-09).
# NEVER write the numeric literal directly in this file — it must always be
# read at runtime so a single JSON artifact is the authoritative source.
_RECALIB_PATH = Path("output/eval_reasoning_v4_winner/judge_recalibration.json")


def _load_score_offset(path: Path = _RECALIB_PATH) -> float:
    """Load the judge recalibration score_offset from a JSON artifact.

    The path parameter is injectable for tests — pass a tmp fixture path to
    avoid reading the real artifact. The default is the canonical artifact
    produced in Phase 4.4 (D-V4-09).

    Args:
        path: Path to judge_recalibration.json (default: production artifact).

    Returns:
        float: score_offset value from the JSON file.

    Raises:
        FileNotFoundError: if the artifact is missing.
        KeyError: if "score_offset" is absent from the JSON.
    """
    data = json.loads(Path(path).read_text())
    return float(data["score_offset"])


# Module-level singleton — loaded once at import using the real artifact.
# Tests that call _load_score_offset(path=<fixture>) bypass this singleton
# and go directly to the injectable function.
_SCORE_OFFSET: float = _load_score_offset()


# ---------------------------------------------------------------------------
# Offset application + clipping
# ---------------------------------------------------------------------------

_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 100.0


def _apply_offset_clip(raw_judge: float) -> float:
    """Apply the recalibration offset to a raw judge score, then clip to [0, 100].

    Order: offset → clip (per D-08-02).  MO-GRPO normalization happens later
    in compute_group_rewards (08-03); this function only handles the
    recalibration step.

    Args:
        raw_judge: Raw overall_score from judge_score_single (0-100 range,
                   but may be outside that range before clipping).

    Returns:
        float: raw_judge + _SCORE_OFFSET, clipped to [0.0, 100.0].
    """
    offset_score = raw_judge + _SCORE_OFFSET
    return float(max(_SCORE_MIN, min(_SCORE_MAX, offset_score)))


# ---------------------------------------------------------------------------
# Lazy imports after env-var suppression (must come AFTER os.environ.pop above)
# ---------------------------------------------------------------------------

from eval.rubric_scorer import (  # noqa: E402
    score_code,
    RubricScore,
    CHECK_DIMENSION_MAP,
    POSITIVE_CHECK_IDS,
    NEGATIVE_CHECK_IDS,
)

# ---------------------------------------------------------------------------
# VeRPO WP-standards subset (D-08-06)
# Apply VeRPO ONLY to D1_wpcs + D5_wp_api checks — not all 9 dimensions.
# The other dimensions are covered by the 30% judge component.
# ---------------------------------------------------------------------------

WP_STANDARDS_CHECK_IDS: frozenset = frozenset(
    cid for cid, dim in CHECK_DIMENSION_MAP.items() if dim in ("D1_wpcs", "D5_wp_api")
)

# ---------------------------------------------------------------------------
# MO-GRPO within-group normalization (GRPO-03)
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-8  # Epsilon floor to prevent NaN on zero-variance groups (T-08-03)


def _mo_grpo_norm(values: np.ndarray) -> np.ndarray:
    """Within-group standardization: (x - mu) / (sigma + epsilon).

    Uses population std (ddof=0) consistent with group-level normalization.
    The epsilon floor prevents NaN / divide-by-zero on zero-variance groups
    (T-08-03 / Pitfall 4 from 08-PATTERNS.md).

    Args:
        values: 1-D numpy array of raw reward signals for a rollout group.

    Returns:
        np.ndarray: Normalized values with same shape as input.
    """
    mu = values.mean()
    sigma = values.std(ddof=0)  # population std — consistent with group-level norm
    return (values - mu) / (sigma + _EPSILON)


# ---------------------------------------------------------------------------
# Output contract dataclasses (D-08-04 / RLEV-02)
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Per-sample reward breakdown carrying pre- and post-normalization signals.

    Mirrors RubricScore.to_dict() pattern for RLEV-02 logging.
    Fields designed for the full (08-03) compute_reward/compute_group_rewards API;
    partial population is valid in intermediate wave tests.
    """

    # --- Pre-normalization (raw signal values) ---
    phpcs_raw: float            # rubric_scorer overall (0-100)
    verpo_raw: float            # VeRPO weighted pass fraction (0-1)
    judge_raw: Optional[float]  # raw wp_judge overall_score or None on parse failure
    judge_offset_applied: float # judge_raw + _SCORE_OFFSET, clipped [0, 100]
    security_fail: bool         # whether CRITICAL_FLOOR_RULE for D2_security triggered

    # --- Post-normalization ---
    phpcs_norm: float
    verpo_norm: float
    judge_norm: float

    # --- Composite pre-gate ---
    composite_pre_gate: float

    # --- VeRPO per-check data ---
    check_pass_rates: dict   # {check_id: pass_rate_across_group}
    check_difficulties: dict # {check_id: difficulty_weight = 1 - pass_rate}

    # --- Group stats ---
    group_size: int
    group_phpcs_mean: float
    group_phpcs_std: float
    group_judge_mean: float
    group_judge_std: float

    # --- Parse-failure metadata (D-08-07) ---
    judge_parse_failure: bool = False
    judge_imputed_from_group: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for json.dumps (RLEV-02 logging).

        Mirrors RubricScore.to_dict() pattern — all values are JSON-native types.
        numpy floats are cast to Python float to avoid serialization failures.
        """
        return {
            # Pre-norm
            "phpcs_raw": float(self.phpcs_raw),
            "verpo_raw": float(self.verpo_raw),
            "judge_raw": float(self.judge_raw) if self.judge_raw is not None else None,
            "judge_offset_applied": float(self.judge_offset_applied),
            "security_fail": bool(self.security_fail),
            # Post-norm
            "phpcs_norm": float(self.phpcs_norm),
            "verpo_norm": float(self.verpo_norm),
            "judge_norm": float(self.judge_norm),
            # Composite
            "composite_pre_gate": float(self.composite_pre_gate),
            # VeRPO per-check data
            "check_pass_rates": {k: float(v) for k, v in self.check_pass_rates.items()},
            "check_difficulties": {k: float(v) for k, v in self.check_difficulties.items()},
            # Group stats
            "group_size": int(self.group_size),
            "group_phpcs_mean": float(self.group_phpcs_mean),
            "group_phpcs_std": float(self.group_phpcs_std),
            "group_judge_mean": float(self.group_judge_mean),
            "group_judge_std": float(self.group_judge_std),
            # Parse-failure metadata
            "judge_parse_failure": bool(self.judge_parse_failure),
            "judge_imputed_from_group": bool(self.judge_imputed_from_group),
        }


@dataclass
class RewardResult:
    """Final reward result for a single generation in a rollout group."""

    scalar: float           # final reward (0.0 if security gate, else composite)
    breakdown: RewardBreakdown


# ---------------------------------------------------------------------------
# Verifiable signal extraction (wraps score_code for rubric signals)
# ---------------------------------------------------------------------------


def _extract_verifiable_signals(php_code: str) -> RubricScore:
    """Run the deterministic rubric scorer on a PHP code string.

    Wraps score_code() from eval.rubric_scorer. The RUBRIC_USE_LLM_CHECKS env
    var is already suppressed at module load (Pitfall 6), ensuring only
    deterministic PHPCS / PHPStan / regex signals are produced.

    Args:
        php_code: PHP source code as a string.

    Returns:
        RubricScore: Full rubric scoring result including triggered_checks,
                     dimension_scores, and overall (0-100).
    """
    return score_code(php_code)


# ---------------------------------------------------------------------------
# VeRPO group scoring (GRPO-04 / D-08-06)
# ---------------------------------------------------------------------------


def _verpo_group(
    rubrics: list[RubricScore],
) -> tuple[list[float], dict[str, float], dict[str, float]]:
    """Compute VeRPO difficulty-weighted partial credit for a rollout group.

    Applies VeRPO ONLY to the WP-standards subset (WP_STANDARDS_CHECK_IDS):
    D1_wpcs (WPCS-*) + D5_wp_api (WAPI-*) check ids. Other dimensions are
    covered by the 30% judge component (D-08-06 — locked decision).

    For each WP-standards check across the group of G rubric results:
      - POSITIVE check fired → pass=True (compliance / Pitfall 5 polarity guard)
      - NEGATIVE check fired → pass=False (violation)
      - Check absent from triggered_checks → not-passed (pass=False)
    Computes:
      pass_rate_c = group_passes_c / G
      difficulty_c = 1 - pass_rate_c   (rare checks contribute more signal)
      verpo_i = sum(difficulty_c * pass_i_c) / (sum(difficulty_c) + _EPSILON)

    Args:
        rubrics: List of RubricScore results for the rollout group (G samples).

    Returns:
        Tuple of:
          - per_sample_verpo: list of G floats in [0, 1]
          - check_pass_rates: {check_id: pass_rate across group}
          - check_difficulties: {check_id: difficulty_weight = 1 - pass_rate}
    """
    G = len(rubrics)
    if G == 0:
        return [], {}, {}

    # Collect all triggered check ids per sample (flat set across all dimensions)
    all_triggered_per_sample: list[set[str]] = [
        {cid for ids in r.triggered_checks.values() for cid in ids}
        for r in rubrics
    ]

    # --- Compute per-check pass counts across the group ---
    check_pass_counts: dict[str, int] = {}
    for check_id in WP_STANDARDS_CHECK_IDS:
        passes = 0
        for triggered in all_triggered_per_sample:
            if check_id in POSITIVE_CHECK_IDS:
                # POSITIVE check: fired = pass (compliance detected)
                passes += 1 if check_id in triggered else 0
            else:
                # NEGATIVE check: fired = fail (violation detected); NOT fired = pass
                passes += 1 if check_id not in triggered else 0
        check_pass_counts[check_id] = passes

    # --- pass_rate and difficulty per check ---
    check_pass_rates: dict[str, float] = {
        cid: cnt / G for cid, cnt in check_pass_counts.items()
    }
    check_difficulties: dict[str, float] = {
        cid: 1.0 - rate for cid, rate in check_pass_rates.items()
    }

    total_difficulty = sum(check_difficulties.values())

    # --- Per-sample VeRPO score ---
    per_sample_verpo: list[float] = []
    for sample_idx, triggered in enumerate(all_triggered_per_sample):
        weighted_passes = 0.0
        for check_id in WP_STANDARDS_CHECK_IDS:
            if check_id in POSITIVE_CHECK_IDS:
                passed = check_id in triggered
            else:
                passed = check_id not in triggered
            weighted_passes += check_difficulties[check_id] * float(passed)
        verpo_i = weighted_passes / (total_difficulty + _EPSILON)
        per_sample_verpo.append(float(verpo_i))

    return per_sample_verpo, check_pass_rates, check_difficulties
