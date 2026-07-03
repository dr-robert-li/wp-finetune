"""Per-completion calibration reward vs a TRAIN-GT anchor set (Phase 08.2-03 / RVAL-02).

Parameterized pairwise rank-agreement-vs-teacher-GT calibration term for the judge
reward axis. The oracle proved fix_correctness is INVALID (Goodhart, corr -0.24) and
pairwise rank-agreement vs teacher GT is VALID (corr +0.70). This module builds the
reward-time version of that valid form.

DESIGN CONSTRAINTS (locked by advisor review):
  1. TRAIN GT ONLY. Both the per-item teacher_overall and the anchor set come from
     the TRAIN split (judge_gt_sidecar.jsonl, Plan 01). Val is the oracle's held-out
     target. Using val at reward time would make SC2 circular (T-09-LEAK guard).
     load_gt_anchor_set() asserts source=="train" on every row.

  2. PARAMETERIZED FORM. Three forms exposed:
     - "pairwise": fraction of anchor items j where sign(model-anchor_gt_j) ==
       sign(teacher-anchor_gt_j); pairs where anchor_gt_j == teacher skipped.
       Mirrors the oracle's pairwise_rank_agreement exactly.
     - "hybrid": pairwise concordance minus a small calibration-error term
       (|model-teacher|/100), so two near-identical saturated scores still differ.
       Intra-group gradient survives 90-100 saturation (08.1-MEASUREMENT.md).
     - "calibration": 1 - |model-teacher|/100. Dense but oracle rejected it alone
       (neg_abs_calibration corr +0.30, CI includes 0). Kept for the sweep.

  3. NaN SAFETY. Missing GT (completion's code_hash not in sidecar) yields NaN.
     augment_judge_scalar() guards the 0*NaN trap: at calib_weight=0 the calibration
     path is skipped entirely; at nonzero weight NaN calib_reward falls back to
     the unmodified judge_scalar (no-op for that item, not NaN propagation).

Exports:
  CALIB_FORMS         — frozenset of valid form names
  load_gt_anchor_set  — loads TRAIN-GT sidecar; returns (anchor_set, gt_map)
  calibration_reward  — per-completion reward scalar in [0, 1]
  augment_judge_scalar — blend judge_scalar with calibration term (NaN-safe)
  get_anchor_set       — module-level lazy singleton (load once per run)

CPU / $0. No GPU, no vLLM, no API.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Optional

# Valid form names
CALIB_FORMS: frozenset[str] = frozenset({"pairwise", "hybrid", "calibration"})

# Default sidecar path (Plan 01 output — TRAIN GT only). Overridable via
# REWARD_SIDECAR_PATH so the relabel-v1 sidecar v2 (B2 conditional-go 2026-07-04)
# can be adopted without touching every get_anchor_set() call site.
_DEFAULT_SIDECAR = Path(os.environ.get(
    "REWARD_SIDECAR_PATH",
    str(Path(__file__).resolve().parent.parent / "data/rl_probe/judge_gt_sidecar.jsonl")))

# Hybrid form: calibration-error weight subtracted from pairwise concordance.
# Small enough to preserve the primary pairwise signal while adding intra-group gradient.
# 0.10 chosen so a 10-point mis-calibration costs 0.01 (1% of the [0,1] range) —
# noticeable within a group but dominated by the pairwise rank signal.
_HYBRID_CALIB_WEIGHT: float = 0.10

# ---------------------------------------------------------------------------
# Canonical GT join key (single source of truth for BOTH sidecar build and
# reward time)
# ---------------------------------------------------------------------------


def normalized_code_hash(php_code: str | None) -> str | None:
    """SHA-256[:16] of whitespace-normalized PHP — the canonical GT JOIN KEY.

    This is the ONE hash function shared by build_reward_gt_sidecar.py (sidecar
    side) and rl_rollouts.py (reward side). The 2026-07-01 gated smoke was
    silently calib-dead because the reward side hashed RAW code while the
    sidecar hashed normalized code — 0/482 join, calib term never fired, the
    run trained on pure fix_correctness. Never fork this normalization.

    Returns None for empty/None input (unhashable — no PHP to key on).
    """
    if not php_code:
        return None
    normalized = re.sub(r"\s+", " ", php_code).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-step calib liveness telemetry (loud-fail wiring, 2026-07-02)
# ---------------------------------------------------------------------------
# Mirrors the rl_judge_dispatch.get_and_reset_score_hist pattern: rl_rollouts
# records one entry per judge completion during collect_rollouts; rl_train
# drains the accumulator once per step into rl_metrics.jsonl. A calib reward
# that never fires must be VISIBLE (and step-0 halt-worthy), never silent.

_CALIB_STATS_LOCK = threading.Lock()
_CALIB_STATS: dict[str, list[float]] = {"fired": [], "values": []}


def record_calib_stat(calib_reward: float) -> None:
    """Record one judge completion's calib outcome (NaN = did not fire)."""
    with _CALIB_STATS_LOCK:
        fired = not math.isnan(calib_reward)
        _CALIB_STATS["fired"].append(1.0 if fired else 0.0)
        if fired:
            _CALIB_STATS["values"].append(float(calib_reward))


def get_and_reset_calib_stats() -> dict[str, float | None]:
    """Drain the per-step calib accumulator.

    Returns {calib_fired_frac, calib_mean, calib_std, calib_n}; fired_frac is
    None when no judge completions were recorded this step (distinguishes
    "no judge batch" from "joined 0%").
    """
    with _CALIB_STATS_LOCK:
        fired = _CALIB_STATS["fired"]
        values = _CALIB_STATS["values"]
        _CALIB_STATS["fired"] = []
        _CALIB_STATS["values"] = []
    if not fired:
        return {"calib_fired_frac": None, "calib_mean": None, "calib_std": None, "calib_n": 0}
    n = len(values)
    mean = sum(values) / n if n else None
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n else None
    return {
        "calib_fired_frac": sum(fired) / len(fired),
        "calib_mean": mean,
        "calib_std": std,
        "calib_n": n,
    }


# ---------------------------------------------------------------------------
# Module-level lazy singleton (load anchor set once per run, keyed by path)
# ---------------------------------------------------------------------------

_ANCHOR_CACHE: dict[str, tuple[list, dict]] = {}
_ANCHOR_LOCK = threading.Lock()


def get_anchor_set(sidecar_path: Optional[str] = None) -> tuple[list, dict]:
    """Return (anchor_set, gt_map) for the given sidecar path, loading once per run.

    Thread-safe singleton keyed by sidecar_path string. On first call loads the file
    and caches; subsequent calls return the cached result.

    Args:
        sidecar_path: Path to a TRAIN-GT sidecar JSONL file. Defaults to
                      data/rl_probe/judge_gt_sidecar.jsonl (Plan 01 output).

    Returns:
        anchor_set: list of (None, teacher_gt_float) 2-tuples.
                    The first element is None (anchor-score placeholder); reward-time
                    pairwise compares sign(model_overall - anchor_gt) vs
                    sign(teacher_overall - anchor_gt) using only anchor_gt.
        gt_map:     dict[code_hash, teacher_overall_float] for per-completion GT lookup.
    """
    path = str(sidecar_path or _DEFAULT_SIDECAR)
    if path not in _ANCHOR_CACHE:
        with _ANCHOR_LOCK:
            if path not in _ANCHOR_CACHE:
                _ANCHOR_CACHE[path] = load_gt_anchor_set(path)
    return _ANCHOR_CACHE[path]


# ---------------------------------------------------------------------------
# GT sidecar loader (anti-leakage enforced)
# ---------------------------------------------------------------------------

def load_gt_anchor_set(sidecar_path: str) -> tuple[list, dict]:
    """Load the TRAIN-GT sidecar and return the anchor set + per-completion GT map.

    Anti-leakage invariant (T-082-08): every row in the sidecar MUST have
    source=="train". Any row with a different source raises AssertionError immediately.
    Val GT must NEVER appear here — val is the oracle's held-out target.

    Args:
        sidecar_path: Path to the TRAIN-GT sidecar JSONL
                      (data/rl_probe/judge_gt_sidecar.jsonl, Plan 01 output).

    Returns:
        anchor_set: list of (None, teacher_overall_float) 2-tuples.
                    None is the anchor-score placeholder; the actual anchor signal is
                    the teacher_overall (GT) used to define the correct ordering.
        gt_map:     dict[code_hash: str, teacher_overall: float] for per-completion
                    GT lookup via the content-hash join (T-082-02 / Plan 01 design).

    Raises:
        AssertionError: if any row has source != "train" (anti-leakage guard).
        FileNotFoundError: if sidecar_path does not exist.
    """
    rows = []
    with open(sidecar_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source", "")
            # Any train-split provenance is accepted (e.g. "train_relabel_v1" from
            # the 2026-07-03 relabel campaign); the protected invariant is that VAL
            # GT never enters the reward path.
            assert src == "train" or src.startswith("train_"), (
                f"Anti-leakage (T-082-08): row {lineno} in {sidecar_path!r} has "
                f"source={src!r} (expected 'train' or 'train_*'). Val GT must NEVER "
                f"enter the reward path — val is the oracle's held-out target "
                f"(T-09-LEAK). Row: {row}"
            )
            rows.append(row)

    # Build anchor_set: list of (None, teacher_overall) 2-tuples.
    # The None placeholder makes the reward-time API explicit: we never store a
    # model score in the anchor set (that would be a T-082-09 GT leakage surface).
    anchor_set = [(None, float(row["teacher_overall"])) for row in rows]

    # Build gt_map: code_hash -> teacher_overall for per-completion lookup.
    # Uses content-hash join (T-082-02 / Plan 01 decision) — reorder-proof.
    gt_map: dict[str, float] = {
        row["code_hash"]: float(row["teacher_overall"]) for row in rows
    }

    return anchor_set, gt_map


# ---------------------------------------------------------------------------
# Core reward function
# ---------------------------------------------------------------------------

def calibration_reward(
    model_overall: float,
    teacher_overall: float,
    anchor_set: list,
    form: str = "hybrid",
    calib_weight: float = 0.0,  # unused by the three forms (reserved for future composites)
) -> float:
    """Per-completion calibration reward vs a TRAIN-GT anchor set.

    Computes a scalar in [0, 1] measuring how well this completion's ranking
    relative to the TRAIN anchor population matches the teacher's ranking.

    Args:
        model_overall:   The judge model's overall score for this completion (0-100).
        teacher_overall: The teacher GT overall for this completion's prompt (0-100).
                         From the TRAIN sidecar (anti-leakage: val GT forbidden here).
        anchor_set:      List of (anchor_score_placeholder, anchor_gt_float) 2-tuples
                         from load_gt_anchor_set(). Each item is a different TRAIN
                         prompt's teacher GT; the anchor_score_placeholder is unused
                         (None) — the reward compares model_overall vs anchor_gt
                         directly, not vs a stored model score.
        form:            One of "pairwise", "hybrid", "calibration".
        calib_weight:    Unused by the three base forms (reserved parameter).

    Returns:
        float in [0, 1]. Returns float("nan") if no valid pairs can be formed
        (e.g. anchor_set empty or all anchor_gts equal teacher_overall).

    Raises:
        ValueError / KeyError: if form is not in CALIB_FORMS.

    Form semantics:

    "pairwise" (mirrors oracle's pairwise_rank_agreement across prompts):
        For each anchor j: is sign(model_overall - anchor_gt_j) the same as
        sign(teacher_overall - anchor_gt_j)? Anchors where anchor_gt_j ==
        teacher_overall are skipped (sign undefined at boundary). Returns
        concordant_pairs / total_valid_pairs. In [0, 1].

    "hybrid" (adds intra-group gradient under saturation):
        pairwise_concordance - _HYBRID_CALIB_WEIGHT * |model_overall - teacher_overall| / 100
        Clipped to [0, 1]. When two completions are both below every anchor (saturated
        90-100 rubric), their pairwise scores are equal — the hybrid term differentiates
        them by absolute calibration error. This avoids the 08.1 gradient-death failure.

    "calibration" (pure absolute calibration — oracle rejected alone, kept for sweep):
        1 - |model_overall - teacher_overall| / 100. In [0, 1].
        Dense but oracle's neg_abs_calibration (equivalent) had ci_lo=-0.454 (INVALID).
        Retained for Plan 04's sweep in case it augments another form.
    """
    if form not in CALIB_FORMS:
        raise ValueError(
            f"Unknown calibration form {form!r}. Valid forms: {sorted(CALIB_FORMS)}"
        )

    if form == "calibration":
        return max(0.0, min(1.0, 1.0 - abs(model_overall - teacher_overall) / 100.0))

    if form in ("pairwise", "hybrid"):
        concordant = 0
        total = 0
        for (_placeholder, anchor_gt) in anchor_set:
            # Skip pairs where anchor_gt == teacher_overall (sign undefined at boundary).
            # This mirrors the oracle's pairwise_rank_agreement: "if dg == 0: continue".
            if anchor_gt == teacher_overall:
                continue
            total += 1
            model_sign = math.copysign(1.0, model_overall - anchor_gt) if model_overall != anchor_gt else 0.0
            teacher_sign = math.copysign(1.0, teacher_overall - anchor_gt)  # non-zero (skipped above)
            # Concordant: both signs are the same AND model is not tied with the anchor.
            if model_sign != 0.0 and model_sign == teacher_sign:
                concordant += 1

        pairwise_score = (concordant / total) if total > 0 else float("nan")

        if form == "pairwise":
            return pairwise_score

        # "hybrid": subtract a small absolute calibration-error term for intra-group gradient
        if math.isnan(pairwise_score):
            return float("nan")
        calib_error = abs(model_overall - teacher_overall) / 100.0
        hybrid = pairwise_score - _HYBRID_CALIB_WEIGHT * calib_error
        return max(0.0, min(1.0, hybrid))

    # Unreachable (CALIB_FORMS check above), but keeps type checkers happy
    raise ValueError(f"Unhandled form: {form!r}")


# ---------------------------------------------------------------------------
# Blending helper (NaN-safe)
# ---------------------------------------------------------------------------

def augment_judge_scalar(
    judge_scalar: float,
    calib_reward: float,
    calib_weight: float,
) -> float:
    """Blend the existing judge_scalar with a calibration reward at a configurable weight.

    This is the ONLY place the calibration term enters the judge reward path.
    The blend stays inside [0, 1] (both inputs in [0, 1], convex combination).
    The MO-GRPO group normalization (advantage centering) is applied DOWNSTREAM —
    never re-normalized here (CR-05 constraint).

    Invariants:
      - calib_weight=0.0: returns judge_scalar exactly (byte-for-byte back-compat).
        The calibration path is SKIPPED, not multiplied by 0, to guard the NaN trap:
        `0 * NaN` propagates NaN in IEEE 754; skipping avoids it entirely.
      - calib_reward=NaN (missing GT for this completion's code_hash): falls back to
        judge_scalar unchanged (no-op for that item). Never propagates NaN downstream.
      - calib_weight > 0 with finite calib_reward: convex blend
        (1 - w) * judge_scalar + w * calib_reward.

    Args:
        judge_scalar:  Combined judge reward from combine_judge_reward() (fix_correctness
                       + consistency). Already in [0, 1].
        calib_reward:  Output of calibration_reward() for this completion. May be NaN
                       if the completion's code_hash is absent from the GT sidecar.
        calib_weight:  Fraction allocated to the calibration term. Default 0.0 (no change).
                       Plan 04 selects the winning weight after the offline sweep.

    Returns:
        float in [0, 1]: augmented judge scalar.
    """
    # Guard 1: calib_weight=0 — skip the calibration path entirely (0*NaN trap).
    if calib_weight == 0.0:
        return judge_scalar

    # Guard 2: NaN calib_reward (missing GT) — fall back to unmodified judge_scalar.
    if math.isnan(calib_reward):
        return judge_scalar

    # Convex blend — stays in [0, 1] for valid inputs.
    return (1.0 - calib_weight) * judge_scalar + calib_weight * calib_reward
