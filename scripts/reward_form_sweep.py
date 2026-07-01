"""Offline reward form/weight sweep (Phase 08.2-04 / RVAL-04).

Sweeps a grid of (form in CALIB_FORMS) x (calib_weight in WEIGHT_GRID) and
scores each config on THREE lenses:

  1. ORACLE-VALIDITY lens (VAL GT, 11-checkpoint captures):
     Composite reward = (1-w)*fix_correctness + w*calib_form_oracle_series.
     Scores via bootstrap Spearman CI against the teacher-Spearman target trajectory.
     Gate: CI-lower > 0 (the standing D-08.2 rule).
     NOTE: uses VAL-GT (not TRAIN-GT) because this is an offline diagnostic on the
     held-out oracle target — using TRAIN GT here would make SC2 circular.

  2. GRADIENT-DENSITY lens (TRAIN GT, probe corpus):
     Scores each config's per-completion reward over the 93-group probe corpus
     (data/rl_probe/judge_probe_corpus.jsonl).
     Join: probe_row.prompt_id -> sidecar.prompt_id -> teacher_overall (TRAIN GT).
     Reuses the 08.1 _compute_group_stats framework (frac_mid, group_reward_std_mean,
     frac_groups_all_zero, frac_groups_nonuniform).

  3. STRUCTURAL CODEGEN-PROXY lens (echo_reward):
     Computes composite reward on a small set of adversary completions (echo-adversary,
     verbose-padding, non-PHP). All must score <= 0.30.
     NOTE: the real codegen guarantee requires wp-bench serving (GPU). This is an
     OFFLINE STRUCTURAL PROXY only — the real check is Plan 05's smoke trip-wire.

Selection (select_config):
  Among valid oracle configs (ci_lo > 0) with echo_reward <= 0.30:
  maximize frac_mid, tie-break lowest frac_groups_all_zero, then highest ci_lo.
  If no config satisfies both gates: returns documented ESCALATION marker.

Exports: run_sweep, select_config

CPU / $0. No GPU, no vLLM, no API calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Imports from prior plans
# ---------------------------------------------------------------------------
from scripts._reward_validity_oracle import (   # noqa: E402
    FORMS,
    FIX_CORR,
    bootstrap_corr_lo,
    build_oracle_pipeline,
    pairwise_rank_agreement,
    neg_abs_calibration,
)
from scripts.reward_calibration import (        # noqa: E402
    CALIB_FORMS,
    calibration_reward,
    augment_judge_scalar,
    load_gt_anchor_set,
)
from scripts._probe_rl_reward import _compute_group_stats  # noqa: E402
from eval.output_parsers import parse_judge_scores          # noqa: E402
from eval.eval_judge import _derive_prose_overall           # noqa: E402
from eval.output_parsers import load_dim_map               # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROBE_PATH   = REPO / "data/rl_probe/judge_probe_corpus.jsonl"
SIDECAR_PATH = REPO / "data/rl_probe/judge_gt_sidecar.jsonl"

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
# 7 weights spanning 0.0 (pure fix_correctness) -> 1.0 (pure calibration).
WEIGHT_GRID: list[float] = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]

# ---------------------------------------------------------------------------
# Oracle form mapping (per advisor P3)
# Each CALIB_FORM maps to an oracle-level function for the trajectory lens.
# - "pairwise" -> pairwise_rank_agreement (exact mirror — validated VALID by Plan 01)
# - "calibration" -> neg_abs_calibration (equivalent to calibration_reward "calibration"
#   at checkpoint level; oracle had neg_abs_calibration corr=+0.30, CI includes 0)
# - "hybrid" -> pairwise_series - 0.10 * mean_calib_error_series
#   (per advisor: derive from pairwise and calibration oracle series already computed)
# ---------------------------------------------------------------------------
_ORACLE_FORM_MAP: dict[str, str] = {
    "pairwise":    "pairwise_rank_agreement",
    "calibration": "neg_abs_calibration",
    "hybrid":      "__hybrid__",   # derived below from pairwise + neg_abs_calibration
}

# ---------------------------------------------------------------------------
# Dimension weights for model_overall derivation (from oracle / Plan 01)
# ---------------------------------------------------------------------------
_dim_map = load_dim_map()
_weights = {k: v for k, v in _dim_map["dimension_weights"].items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Helpers: model_overall from probe raw_text
# ---------------------------------------------------------------------------

def _parse_model_overall(raw_text: str) -> Optional[float]:
    """Parse model_overall from a probe corpus raw_text (judge completion).

    Mirrors the oracle's model_overall_map() logic:
      parse_judge_scores(text, 'auto') -> overall or _derive_prose_overall(dims, weights).
    Returns None if the completion cannot be parsed to a numeric score.
    """
    p = parse_judge_scores(raw_text, "auto")
    if not p or not p.get("dimension_scores"):
        return None
    if "overall" in p:
        return float(p["overall"])
    mo = _derive_prose_overall(p["dimension_scores"], _weights)
    return float(mo) if mo is not None else None


# ---------------------------------------------------------------------------
# Helpers: TRAIN-GT join for probe corpus
# ---------------------------------------------------------------------------

def _load_sidecar_pid_map() -> dict[int, float]:
    """Load sidecar as {prompt_id: teacher_overall}.

    join: probe_row.prompt_id -> sidecar.prompt_id -> teacher_overall (TRAIN GT).
    This is reorder-proof because build_reward_gt_sidecar.py uses content-hash to
    match RL pool entries to TRAIN GT; the sidecar prompt_id equals the pool index.
    The sidecar was validated at build time (T-082-12 / Plan 01).
    """
    rows = [json.loads(l) for l in SIDECAR_PATH.open() if l.strip()]
    for row in rows:
        assert row.get("source") == "train", (
            f"Anti-leakage (T-082-08): sidecar row has source={row.get('source')!r}. "
            f"Val GT must never appear in reward-time GT (T-09-LEAK)."
        )
    return {int(row["prompt_id"]): float(row["teacher_overall"]) for row in rows}


def _load_probe_corpus() -> list[dict]:
    """Load probe corpus rows (raw_text, group_id, sample_idx, prompt_id, parseable)."""
    return [json.loads(l) for l in PROBE_PATH.open() if l.strip()]


# ---------------------------------------------------------------------------
# Compute per-completion reward on probe corpus for a given (form, calib_weight)
# ---------------------------------------------------------------------------

def _compute_probe_rewards(
    form: str,
    calib_weight: float,
    probe_rows: list[dict],
    gt_pid_map: dict[int, float],
    anchor_set: list,
) -> list[float]:
    """Compute per-completion composite reward for a (form, calib_weight) config.

    Composite = augment_judge_scalar(fix_correctness, calib_reward, calib_weight)
    where:
      fix_correctness = model_overall / 100 if parsed, else 0.0 (parse-cliff)
      calib_reward    = calibration_reward(model_overall, teacher_overall, anchor_set, form)
                        NaN if model_overall is None or teacher_overall not in sidecar.

    NaN handling: augment_judge_scalar() falls back to fix_correctness when calib_reward=NaN.
    Returns a flat list of composite scalars (one per probe row, in corpus order).
    """
    rewards: list[float] = []
    for row in probe_rows:
        model_overall = _parse_model_overall(row["raw_text"])
        fix_corr = (model_overall / 100.0) if model_overall is not None else 0.0

        # TRAIN GT join: probe prompt_id -> sidecar teacher_overall
        teacher_overall = gt_pid_map.get(int(row["prompt_id"]))

        if calib_weight == 0.0 or model_overall is None or teacher_overall is None:
            # Weight=0 path: pure fix_correctness, no calibration term.
            # Also falls back if GT or model_overall unavailable.
            rewards.append(fix_corr)
        else:
            cr = calibration_reward(model_overall, teacher_overall, anchor_set, form=form)
            blended = augment_judge_scalar(fix_corr, cr, calib_weight)
            rewards.append(blended)

    return rewards


# ---------------------------------------------------------------------------
# Oracle composite series for a given (form, calib_weight)
# ---------------------------------------------------------------------------

def _compute_oracle_composite(
    form: str,
    calib_weight: float,
    fix_series: dict,
    oracle_form_series: dict,
    present: list[str],
    target: dict,
) -> tuple[float, float, float, bool, int]:
    """Compute oracle validity for composite = (1-w)*fix_correctness + w*oracle_form.

    fix_correctness is only defined for step-50..step-500 (FIX_CORR keys, no warmstart).
    oracle_form_series is defined for all present checkpoints.
    We align to the intersection of all three: fix_series keys, oracle_form_series keys,
    and target keys (within present).

    Returns: (spearman_vs_target, ci_lo, ci_hi, valid, n_ckpts)
    """
    # Fix series keys (from FIX_CORR): no 'warmstart'
    ck_list = [
        n for n in present
        if n in fix_series and n in oracle_form_series and n in target
    ]
    if len(ck_list) < 4:
        return float("nan"), float("nan"), float("nan"), False, len(ck_list)

    xs: list[float] = []
    ys: list[float] = []
    for ck in ck_list:
        composite = (1.0 - calib_weight) * fix_series[ck] + calib_weight * oracle_form_series[ck]
        xs.append(composite)
        ys.append(target[ck])

    point, lo, hi = bootstrap_corr_lo(xs, ys)
    valid = bool(lo == lo and lo > 0)   # NaN-safe
    return point, lo, hi, valid, len(ck_list)


# ---------------------------------------------------------------------------
# Structural codegen-proxy lens: echo-adversary probe
# ---------------------------------------------------------------------------

# Echo-adversary and non-PHP/verbose completions.
# These must score <= 0.30 on the composite reward.
# The calibration term must NOT inflate them — this is the offline proxy for
# "no codegen penalty" (the real check is Plan 05's wp-bench smoke trip-wire).
#
# Design: an echo adversary echoes GT-like scores back verbatim.
# It should NOT get rewarded by pairwise/hybrid (the pairwise concordance of a
# made-up response vs the anchor population should not be systematically high).
# For 'calibration' form: model_overall ≈ teacher_overall -> calibration=1.0,
# so the calibration form WILL inflate echo adversaries at high weight.
# That's exactly the feature: the structural proxy rejects the invalid form.

# We build adversary completions with NO valid PHP judge_output (parse failure).
# fix_correctness = 0.0 (no corrected PHP block). calib_reward uses model_overall=None
# (no parse), so falls back to fix_correctness=0.0 regardless of form/weight.

# For completions WITH a parseable judge score (echo of scores in prose), the risk is
# that calibration reward inflates them. We test this case: a completion that echoes
# a high score in prose, so parse_judge_scores returns a high model_overall.

_ECHO_ADVERSARY_COMPLETIONS = [
    # Case 1: no PHP block at all, pure verbose padding — model_overall=None -> fix=0.0
    ("verbose_no_php",
     "This code is excellent. " * 80 + " All standards are met perfectly."),

    # Case 2: echo-scores prose WITH parseable judge_output (worst case for calibration)
    # parse_judge_scores will pick up the JSON -> model_overall=95
    # teacher_overall is unknown (no prompt_id join) -> GT fallback to fix_correctness.
    # But we test with an explicit teacher_overall to probe whether the calibration
    # term rewards the echo. We use a synthetic teacher_overall=95 (same as model).
    # calibration form: 1 - |95-95|/100 = 1.0 — high echo reward if form='calibration'.
    # pairwise form: sign(95-anchor_gt) concordant fraction — not systematically high.
    ("echo_judge_output_prose",
     ("WPCS Compliance: score 10/10 — Perfect.\n\nSQL Safety: score 10/10 — Perfect.\n\n"
      "Security: score 10/10 — Perfect.\n\nPerformance: score 10/10 — Perfect.\n\n"
      "WP API Usage: score 10/10 — Perfect.\n\nCode Quality: score 10/10 — Perfect.\n\n"
      "Dependency Integrity: score 10/10 — Perfect.\n\n"
      "[/REASONING]\n\n<judge_output>\n"
      '{"verdict": "PASS", "wpcs_compliance": 10, "sql_safety": 10, "security": 10, '
      '"performance": 10, "wp_api_usage": 10, "code_quality": 10, '
      '"dependency_integrity": 10, "overall_score": 95}\n</judge_output>\n\n'
      "<?php\nfunction test() { echo 'score: 99'; }")),

    # Case 3: non-PHP echo (no corrected PHP block, just score repetition)
    ("non_php_echo", "overall_score: 99\nscore: 99\nrating: 99\n" * 10),
]


def _compute_echo_reward(
    form: str,
    calib_weight: float,
    anchor_set: list,
) -> float:
    """Compute the worst-case composite reward on the adversary probe set.

    For each adversary, we parse model_overall and use a synthetic teacher_overall=95
    (the most generous case — model matches teacher). Returns the max reward over the
    adversary set. Gate: <= 0.30.

    STRUCTURAL PROXY CAVEAT: the real codegen guarantee (no PHP correctness reward for
    echo completions) requires wp-bench serving (GPU). This proxy only catches reward
    inflation on parsed-score completions. Real check is Plan 05's smoke trip-wire.
    """
    # Synthetic teacher_overall for the "parsed echo" case: same as model (worst case)
    # This maximizes calibration_reward for 'calibration' form (score=1.0).
    SYNTHETIC_TEACHER = 95.0

    max_reward = 0.0
    for _label, adv in _ECHO_ADVERSARY_COMPLETIONS:
        model_overall = _parse_model_overall(adv)
        fix_corr = (model_overall / 100.0) if model_overall is not None else 0.0

        if calib_weight == 0.0 or model_overall is None:
            composite = fix_corr
        else:
            # Use synthetic teacher_overall for worst-case echo analysis
            cr = calibration_reward(model_overall, SYNTHETIC_TEACHER, anchor_set, form=form)
            composite = augment_judge_scalar(fix_corr, cr, calib_weight)

        max_reward = max(max_reward, composite)

    return max_reward


# ---------------------------------------------------------------------------
# Gradient-density stats (reusing 08.1 framework)
# ---------------------------------------------------------------------------

def _compute_density_stats(rewards: list[float]) -> dict:
    """Compute 08.1 gradient-density stats on a flat list of per-completion rewards.

    Simulates G=4 groups by chunking (matching the probe corpus group_size=4).
    Reuses _compute_group_stats from _probe_rl_reward.py (the 08.1 framework).
    """
    return _compute_group_stats(rewards, group_size=4)


# ---------------------------------------------------------------------------
# Main sweep entry point
# ---------------------------------------------------------------------------

def run_sweep() -> list[dict]:
    """Sweep (form, calib_weight) grid and score each config on three lenses.

    Returns a list of dicts (one per config), each with keys:
      form, calib_weight,
      -- Oracle-validity lens --
      spearman_vs_target, ci_lo, ci_hi, valid, n_ckpts,
      -- Gradient-density lens --
      frac_mid, group_reward_std_mean, frac_groups_all_zero, frac_groups_nonuniform,
      n_samples, n_groups,
      -- Structural codegen-proxy lens --
      echo_reward,

    Grid: CALIB_FORMS x WEIGHT_GRID = 3 x 7 = 21 configs.
    """
    # ------------------------------------------------------------------
    # 1. Build oracle pipeline (11-checkpoint trajectory, VAL GT)
    #    Compute per-form component series ONCE (weight-independent).
    # ------------------------------------------------------------------
    target, reward_series, present, _n_common = build_oracle_pipeline()

    # FIX_CORR series (fix_correctness baseline trajectory, from READS_TALLY.md)
    fix_series: dict[str, float] = {
        n: v for n, v in FIX_CORR.items() if n in present
    }

    # Pre-compute oracle series for each calib form (weight-independent).
    # hybrid = pairwise_series[ck] - 0.10 * neg_abs_calib_error[ck]
    # neg_abs_calibration in FORMS is: 1 - mean|m-g|/100 (per checkpoint level).
    # hybrid oracle = pairwise_rank_agreement[ck] - 0.10 * (1 - neg_abs_calibration[ck])
    #   = pairwise[ck] - 0.10 * (mean|m-g|/100)
    # This mirrors calibration_reward "hybrid" = pairwise - 0.10*|m-t|/100.
    oracle_form_series: dict[str, dict[str, float]] = {}
    for calib_form in CALIB_FORMS:
        oracle_key = _ORACLE_FORM_MAP[calib_form]
        if oracle_key == "__hybrid__":
            # Derive hybrid from pairwise and neg_abs_calibration checkpoint series.
            # neg_abs_calibration[ck] = 1 - mean|m-g|/100  =>  mean|m-g|/100 = 1 - neg[ck]
            # hybrid[ck] = pairwise[ck] - 0.10 * (1 - neg_abs[ck])
            pairwise_s = reward_series.get("pairwise_rank_agreement", {})
            neg_abs_s = reward_series.get("neg_abs_calibration", {})
            hybrid_series: dict[str, float] = {}
            for ck in present:
                if ck in pairwise_s and ck in neg_abs_s:
                    pw = pairwise_s[ck]
                    na = neg_abs_s[ck]
                    # mean|m-g|/100 = 1 - neg_abs_calibration
                    calib_err = 1.0 - na
                    hybrid_val = pw - 0.10 * calib_err
                    hybrid_series[ck] = max(0.0, min(1.0, hybrid_val))
            oracle_form_series[calib_form] = hybrid_series
        else:
            oracle_form_series[calib_form] = reward_series.get(oracle_key, {})

    # ------------------------------------------------------------------
    # 2. Load probe corpus + TRAIN GT (gradient-density lens)
    # ------------------------------------------------------------------
    probe_rows = _load_probe_corpus()
    gt_pid_map = _load_sidecar_pid_map()

    # Load anchor set once (TRAIN GT, for calibration_reward per completion)
    anchor_set, _gt_map = load_gt_anchor_set(str(SIDECAR_PATH))

    # ------------------------------------------------------------------
    # 3. Pre-compute per-completion rewards per FORM (weight-independent)
    #    Then blend across weights.
    # ------------------------------------------------------------------
    # Per-row fix_correctness (weight=0 baseline) and calib_reward per form.
    fix_corr_per_row: list[float] = []
    calib_reward_per_form: dict[str, list[Optional[float]]] = {f: [] for f in CALIB_FORMS}

    for row in probe_rows:
        model_overall = _parse_model_overall(row["raw_text"])
        fix_corr = (model_overall / 100.0) if model_overall is not None else 0.0
        fix_corr_per_row.append(fix_corr)

        pid = int(row["prompt_id"])
        teacher_overall = gt_pid_map.get(pid)

        for calib_form in CALIB_FORMS:
            if model_overall is None or teacher_overall is None:
                calib_reward_per_form[calib_form].append(None)  # NaN / fallback
            else:
                cr = calibration_reward(
                    model_overall, teacher_overall, anchor_set, form=calib_form
                )
                calib_reward_per_form[calib_form].append(cr)

    # ------------------------------------------------------------------
    # 4. Sweep grid
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for calib_form in sorted(CALIB_FORMS):   # sorted for determinism
        for calib_weight in WEIGHT_GRID:
            # ---- Oracle-validity lens ----
            spearman, ci_lo, ci_hi, oracle_valid, n_ckpts = _compute_oracle_composite(
                calib_form, calib_weight,
                fix_series, oracle_form_series[calib_form],
                present, target,
            )

            # ---- Gradient-density lens: blend per-row rewards ----
            blended_rewards: list[float] = []
            for i, fix_corr in enumerate(fix_corr_per_row):
                if calib_weight == 0.0:
                    blended_rewards.append(fix_corr)
                else:
                    cr = calib_reward_per_form[calib_form][i]
                    # augment_judge_scalar handles None/NaN fallback
                    cr_val = cr if (cr is not None and not math.isnan(cr)) else float("nan")
                    blended = augment_judge_scalar(fix_corr, cr_val, calib_weight)
                    blended_rewards.append(blended)

            density = _compute_density_stats(blended_rewards)

            # ---- Structural codegen-proxy lens ----
            echo_reward = _compute_echo_reward(calib_form, calib_weight, anchor_set)

            row_dict: dict = {
                "form": calib_form,
                "calib_weight": calib_weight,
                # Oracle lens
                "spearman_vs_target": round(spearman, 4) if not math.isnan(spearman) else None,
                "ci_lo": round(ci_lo, 4) if not math.isnan(ci_lo) else None,
                "ci_hi": round(ci_hi, 4) if not math.isnan(ci_hi) else None,
                "valid": oracle_valid,
                "n_ckpts": n_ckpts,
                # Gradient-density lens
                "frac_mid": round(density["frac_mid"], 4) if density["frac_mid"] is not None else None,
                "group_reward_std_mean": round(density["group_reward_std_mean"], 4) if density["group_reward_std_mean"] is not None else None,
                "frac_groups_all_zero": round(density["frac_groups_all_zero"], 4) if density["frac_groups_all_zero"] is not None else None,
                "frac_groups_nonuniform": round(density["frac_groups_nonuniform"], 4) if density["frac_groups_nonuniform"] is not None else None,
                "n_samples": density["n_samples"],
                "n_groups": density["n_groups"],
                # Structural codegen-proxy lens
                "echo_reward": round(echo_reward, 4),
            }
            rows.append(row_dict)

    return rows


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_config(rows: list[dict]) -> Optional[dict]:
    """Select the best (form, calib_weight) config from the sweep rows.

    Selection criteria (dual-lens):
      1. oracle valid == True (ci_lo > 0) — the standing D-08.2 gate
      2. echo_reward <= 0.30 — the structural codegen-proxy gate (Plan 05 real check)
      Among candidates:
        - Primary sort: highest frac_mid (maximum gradient density)
        - Tie-break 1: lowest frac_groups_all_zero (fewer gradient-dead groups)
        - Tie-break 2: highest ci_lo (stronger oracle validity)

    If NO config satisfies both gates, returns None with a documented reason.
    The result is deterministic (sort is stable; all inputs are deterministically computed).

    Returns:
        dict with all sweep row fields for the selected config, or None if no
        config satisfies both gates (ESCALATION — do not proceed to Plan 05).
    """
    candidates = [
        r for r in rows
        if r["valid"] is True
        and r["ci_lo"] is not None
        and r["ci_lo"] > 0
        and r["echo_reward"] is not None
        and r["echo_reward"] <= 0.30
    ]

    if not candidates:
        # Escalation: document why no config qualifies.
        valid_rows = [r for r in rows if r["valid"]]
        echo_ok_rows = [r for r in rows if r["echo_reward"] is not None and r["echo_reward"] <= 0.30]
        return None  # documented by run_sweep caller (sweep_results.json selected=null)

    # Sort: maximize frac_mid, minimize frac_groups_all_zero, maximize ci_lo.
    def _sort_key(r: dict):
        frac_mid = r["frac_mid"] if r["frac_mid"] is not None else -1.0
        fgaz = r["frac_groups_all_zero"] if r["frac_groups_all_zero"] is not None else 1.0
        ci_lo = r["ci_lo"] if r["ci_lo"] is not None else -999.0
        return (-frac_mid, fgaz, -ci_lo)

    candidates_sorted = sorted(candidates, key=_sort_key)
    return candidates_sorted[0]


# ---------------------------------------------------------------------------
# CLI entry point (offline diagnostic / manual inspection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    print("Running reward form sweep (CPU / $0)...")
    sweep_rows = run_sweep()
    print(f"\n{'form':<12} {'weight':>6} {'ci_lo':>7} {'valid':>5} {'frac_mid':>8} {'fgaz':>7} {'echo':>6}")
    print("-" * 60)
    for r in sweep_rows:
        ci_lo_str = f"{r['ci_lo']:>7.3f}" if r["ci_lo"] is not None else "    N/A"
        fm_str = f"{r['frac_mid']:>8.3f}" if r["frac_mid"] is not None else "     N/A"
        fgaz_str = f"{r['frac_groups_all_zero']:>7.3f}" if r["frac_groups_all_zero"] is not None else "    N/A"
        print(f"{r['form']:<12} {r['calib_weight']:>6.2f} {ci_lo_str} {str(r['valid']):>5} {fm_str} {fgaz_str} {r['echo_reward']:>6.3f}")

    selected = select_config(sweep_rows)
    print("\nSelected config:", _json.dumps(selected, indent=2) if selected else "ESCALATION: no config satisfies both oracle-valid AND echo<=0.30")

    # Write results
    out_dir = REPO / "output/reward_validity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_results.json"
    ranked = sorted(
        sweep_rows,
        key=lambda r: (
            0 if r["valid"] else 1,
            -(r["frac_mid"] or -1.0),
            (r["frac_groups_all_zero"] or 1.0),
            -(r["ci_lo"] or -999.0),
        ),
    )
    with out_path.open("w") as f:
        _json.dump({"ranked": ranked, "selected": selected}, f, indent=2)
    print(f"\nResults written to {out_path}")
