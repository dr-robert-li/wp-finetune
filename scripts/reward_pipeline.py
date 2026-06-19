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
from eval.rubric_definitions import (  # noqa: E402
    CRITICAL_FLOOR_RULES,
    CHECK_REGISTRY,
)
from eval.eval_judge import judge_score_single  # noqa: E402  (module-level for patch binding)

# ---------------------------------------------------------------------------
# VeRPO WP-standards subset (D-08-06)
# Apply VeRPO ONLY to D1_wpcs + D5_wp_api checks — not all 9 dimensions.
# The other dimensions are covered by the 30% judge component.
# ---------------------------------------------------------------------------

WP_STANDARDS_CHECK_IDS: frozenset = frozenset(
    cid for cid, dim in CHECK_DIMENSION_MAP.items() if dim in ("D1_wpcs", "D5_wp_api")
)

# ---------------------------------------------------------------------------
# Security hard gate — GRPO-02 / D-08-05 / T-08-SEC
# ---------------------------------------------------------------------------
# _REWARD_SEC_TRIGGERS: deterministic subset of D2_security CRITICAL_FLOOR_RULE trigger ids.
# Derived programmatically: D2_security trigger ids where CHECK_REGISTRY[cid].method != "llm".
#
# SEC-N04 (the only llm-method D2_security trigger) is excluded BY DESIGN:
#   reward_pipeline.py pops RUBRIC_USE_LLM_CHECKS at module load (Pitfall 6), so
#   score_code() never fires llm-method checks during reward compute. SEC-N04 therefore
#   cannot appear in triggered_checks at reward time. Excluding it here keeps the gate
#   strictly consistent with what can actually fire (D-08 locked constraint).
#   DO NOT re-enable RUBRIC_USE_LLM_CHECKS to "rescue" SEC-N04 — that breaks determinism.
#
# Result: {SEC-N01, SEC-N03, SEC-N06, SEC-N08, SEC-N19, SEC-N20} (6 ids, all phpcs/regex)
_REWARD_SEC_TRIGGERS: frozenset = frozenset(
    cid
    for rule in CRITICAL_FLOOR_RULES
    if rule[0] == "D2_security"
    for cid in rule[2]
    if CHECK_REGISTRY[cid].method != "llm"
)

# Fail-CLOSED guard at module load: if the trigger set is somehow empty (structural breakage),
# raise immediately so the misconfiguration is surfaced at import time, not at inference time.
if not _REWARD_SEC_TRIGGERS:
    raise RuntimeError(
        "reward_pipeline: _REWARD_SEC_TRIGGERS is empty — cannot derive the D2_security "
        "deterministic trigger set from CRITICAL_FLOOR_RULES + CHECK_REGISTRY. "
        "This is a structural misconfiguration; refusing to start with an empty gate "
        "(empty gate = fail-open = HIGH severity T-08-SEC)."
    )


def _security_fail(rubric: RubricScore) -> bool:
    """Return True iff the rubric contains a D2_security CRITICAL_FLOOR_RULE trigger.

    GATE SEMANTICS (D-08-05 Option C / GRPO-02):
      - Reads triggered_checks ONLY — the dict of dim -> [check_ids that fired].
        The apply_floor_rules() side-effect list is intentionally NOT consulted:
        it is only appended to when the current score exceeds the floor cap,
        so an already-below-cap D2 dimension yields an empty list even when a
        trigger has fired. Gate uses triggered_checks for reliable membership.
      - Intersects ALL fired check ids (across all dimensions) against
        _REWARD_SEC_TRIGGERS (the DETERMINISTIC D2_security trigger subset).
      - Returns True iff the intersection is non-empty.

    FAIL-CLOSED contract:
      - Raises RuntimeError if _REWARD_SEC_TRIGGERS is empty (config breakage).
        Returning False on an empty gate would let insecure code earn reward (T-08-SEC HIGH).

    Args:
        rubric: RubricScore from score_code(); must have .triggered_checks dict.

    Returns:
        bool: True if a deterministic D2_security trigger fired; False otherwise.
    """
    if not _REWARD_SEC_TRIGGERS:
        raise RuntimeError(
            "_security_fail: _REWARD_SEC_TRIGGERS is empty — gate cannot operate. "
            "This indicates structural breakage in CRITICAL_FLOOR_RULES / CHECK_REGISTRY. "
            "Raising (not returning False) to prevent fail-open (T-08-SEC HIGH severity)."
        )
    # Flatten all triggered check ids across all dimensions into a single set
    all_fired: set[str] = {
        cid
        for ids in rubric.triggered_checks.values()
        for cid in ids
    }
    return bool(all_fired & _REWARD_SEC_TRIGGERS)


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


# ---------------------------------------------------------------------------
# Composite reward assembly — public API (GRPO-01 / D-08-04 / D-08-07)
# ---------------------------------------------------------------------------

# Weight constants (35 / 35 / 30 split — locked default per D-08)
_W_PHPCS: float = 0.35
_W_VERPO: float = 0.35
_W_JUDGE: float = 0.30

# Warning threshold: if more than 10% of batch has None judge score, emit a warning (D-08-07)
_JUDGE_IMPUTE_WARN_RATE: float = 0.10


def compute_group_rewards(
    php_codes: list[str],
    judge_client: object,
    judge_model: str,
) -> list[RewardResult]:
    """Compute composite GRPO rewards for a rollout group (two-pass, group-normalized).

    Two-pass algorithm:
      Pass 1 — collect signals:
        For each code in the group:
          a. Run _extract_verifiable_signals(code) -> RubricScore
          b. Call judge_score_single(code, judge_client, judge_model) -> Optional[float]
          c. Track judge parse failures (None returns)
      Impute None judge scores from the group mean of valid scores (D-08-07).
      If >10% of the group has None judge scores, emit a warning.

      Pass 2 — normalize + compose + gate:
        a. Apply _apply_offset_clip to each judge score (offset then clip [0, 100])
        b. Compute VeRPO group scores via _verpo_group
        c. MO-GRPO normalize phpcs_raw, verpo, and offset-clipped judge independently
        d. Per-sample composite_pre_gate = 0.35*phpcs_norm + 0.35*verpo_norm + 0.30*judge_norm
        e. Terminal override: scalar = 0.0 if _security_fail(rubric) else composite_pre_gate

    Security override is TERMINAL (applied AFTER normalize+combine). The failing member's
    composite_pre_gate retains its real composite value; only scalar is zeroed.
    This preserves the group statistics (the fail member's pre-override signals remain in
    the normalization denominator) per Pitfall 1 / GRPO-02.

    Args:
        php_codes: List of PHP source code strings for this rollout group.
        judge_client: Judge client object passed to judge_score_single.
        judge_model: Model identifier string passed to judge_score_single.

    Returns:
        list[RewardResult]: One result per input, each with (scalar, breakdown).
    """
    import logging
    import warnings
    G = len(php_codes)
    if G == 0:
        return []

    # -----------------------------------------------------------------------
    # Pass 1: collect rubric scores and raw judge scores
    # -----------------------------------------------------------------------
    rubrics: list[RubricScore] = []
    raw_judge_scores: list[Optional[float]] = []

    for code in php_codes:
        rubric = _extract_verifiable_signals(code)
        rubrics.append(rubric)
        judge_raw = judge_score_single(code, judge_client, judge_model)
        raw_judge_scores.append(judge_raw)

    # -----------------------------------------------------------------------
    # Judge parse-failure imputation (D-08-07)
    # -----------------------------------------------------------------------
    judge_parse_failures: list[bool] = [score is None for score in raw_judge_scores]
    judge_imputed_flags: list[bool] = [False] * G

    fail_count = sum(judge_parse_failures)
    if fail_count > 0:
        # Compute mean of valid scores for imputation
        valid_scores = [s for s in raw_judge_scores if s is not None]
        if valid_scores:
            group_judge_mean_raw = float(np.mean(valid_scores))
        else:
            # All scores are None — impute with 0.0 (conservative fallback)
            group_judge_mean_raw = 0.0

        for i, is_failure in enumerate(judge_parse_failures):
            if is_failure:
                raw_judge_scores[i] = group_judge_mean_raw
                judge_imputed_flags[i] = True

        fail_rate = fail_count / G
        if fail_rate > _JUDGE_IMPUTE_WARN_RATE:
            warnings.warn(
                f"compute_group_rewards: judge parse failure rate {fail_rate:.1%} "
                f"({fail_count}/{G}) exceeds {_JUDGE_IMPUTE_WARN_RATE:.0%} threshold. "
                "Imputing from group mean — check judge endpoint health (D-08-07).",
                RuntimeWarning,
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # Pass 2: normalize + compose + gate
    # -----------------------------------------------------------------------

    # Apply offset+clip to judge scores
    judge_offset_scores: list[float] = [
        _apply_offset_clip(float(s)) for s in raw_judge_scores  # type: ignore[arg-type]
    ]

    # phpcs raw scores (rubric.overall is the 0-100 rubric overall)
    phpcs_raws: list[float] = [float(r.overall) for r in rubrics]

    # VeRPO scores for the group
    verpo_scores, check_pass_rates, check_difficulties = _verpo_group(rubrics)

    # MO-GRPO normalize each signal independently
    phpcs_arr = np.array(phpcs_raws, dtype=float)
    verpo_arr = np.array(verpo_scores, dtype=float)
    judge_arr = np.array(judge_offset_scores, dtype=float)

    phpcs_norm_arr = _mo_grpo_norm(phpcs_arr)
    verpo_norm_arr = _mo_grpo_norm(verpo_arr)
    judge_norm_arr = _mo_grpo_norm(judge_arr)

    # Group stats (using population std, ddof=0 — consistent with _mo_grpo_norm)
    group_phpcs_mean = float(phpcs_arr.mean())
    group_phpcs_std = float(phpcs_arr.std(ddof=0))
    group_judge_mean = float(judge_arr.mean())
    group_judge_std = float(judge_arr.std(ddof=0))

    # -----------------------------------------------------------------------
    # Per-sample composite + terminal security override
    # -----------------------------------------------------------------------
    results: list[RewardResult] = []
    for i, rubric in enumerate(rubrics):
        phpcs_norm = float(phpcs_norm_arr[i])
        verpo_norm = float(verpo_norm_arr[i])
        judge_norm = float(judge_norm_arr[i])

        composite_pre_gate = (
            _W_PHPCS * phpcs_norm
            + _W_VERPO * verpo_norm
            + _W_JUDGE * judge_norm
        )

        # Evaluate security gate (reads triggered_checks, NOT a dimension score cut)
        sec_fail = _security_fail(rubric)

        # Terminal override: applied AFTER normalize+combine (Pitfall 1 / GRPO-02)
        final_scalar = 0.0 if sec_fail else composite_pre_gate

        breakdown = RewardBreakdown(
            # Pre-normalization
            phpcs_raw=phpcs_raws[i],
            verpo_raw=verpo_scores[i],
            judge_raw=raw_judge_scores[i],  # type: ignore[arg-type]
            judge_offset_applied=judge_offset_scores[i],
            security_fail=sec_fail,
            # Post-normalization
            phpcs_norm=phpcs_norm,
            verpo_norm=verpo_norm,
            judge_norm=judge_norm,
            # Composite pre-gate (real composite, not zeroed — gate only zeroes scalar)
            composite_pre_gate=float(composite_pre_gate),
            # VeRPO per-check data
            check_pass_rates=check_pass_rates,
            check_difficulties=check_difficulties,
            # Group stats
            group_size=G,
            group_phpcs_mean=group_phpcs_mean,
            group_phpcs_std=group_phpcs_std,
            group_judge_mean=group_judge_mean,
            group_judge_std=group_judge_std,
            # Parse-failure metadata (D-08-07)
            judge_parse_failure=judge_parse_failures[i],
            judge_imputed_from_group=judge_imputed_flags[i],
        )

        results.append(RewardResult(scalar=float(final_scalar), breakdown=breakdown))

    return results


def compute_reward(
    php_code: str,
    judge_client: object,
    judge_model: str,
) -> RewardResult:
    """Compute reward for a single PHP generation using a size-1 group.

    Convenience wrapper around compute_group_rewards for the single-sample case.
    Note: with a single sample, MO-GRPO normalization yields 0.0 for all signals
    (the sample IS the group mean). This is expected behavior for inference-time
    scoring; GRPO training always uses compute_group_rewards with G > 1.

    Args:
        php_code: PHP source code string.
        judge_client: Judge client object.
        judge_model: Model identifier string.

    Returns:
        RewardResult: (scalar, breakdown) for the single sample.
    """
    results = compute_group_rewards([php_code], judge_client, judge_model)
    return results[0]
