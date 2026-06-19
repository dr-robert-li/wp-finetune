"""Reward pipeline for Phase 8 GRPO training (Wave 1 — foundation).

This module provides:
  - _load_score_offset(path): injectable recalibration loader (D-08-02)
  - _SCORE_OFFSET: module-level singleton loaded at import time
  - _apply_offset_clip(raw_judge): applies offset then clips to [0, 100]

Wave 2 (08-02) adds: _mo_grpo_norm, VeRPO helpers, compute_reward,
compute_group_rewards, RewardBreakdown, RewardResult dataclasses.
Wave 3 (08-03) adds: _security_fail, full security gate, composite weights.

CRITICAL: LLM-checks are suppressed at module load (deterministic reward
signals only in training — Pitfall 6 from 08-PATTERNS.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

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
    in compute_group_rewards (08-02); this function only handles the
    recalibration step.

    Args:
        raw_judge: Raw overall_score from judge_score_single (0-100 range,
                   but may be outside that range before clipping).

    Returns:
        float: raw_judge + _SCORE_OFFSET, clipped to [0.0, 100.0].
    """
    offset_score = raw_judge + _SCORE_OFFSET
    return float(max(_SCORE_MIN, min(_SCORE_MAX, offset_score)))
