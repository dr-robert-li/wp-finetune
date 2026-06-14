"""Concentration metrics for merged-model MoE router profiling (PROF-04).

Reads the JSONL output of profile_merged_model.py and computes:
  - Per-layer CV (coefficient of variation of expert counts)
  - Cumulative coverage curve (sorted descending cumsum / total)
  - Layer-depth skew (early vs late layer concentration ratio)
  - E_eff mean/max/variance per split (total/wp_gen/wp_judge)
  - E_eff delta vs base model (D-08 join on normalized "30_70" key)
  - Bootstrap CI for all aggregates (D-09 CI-aware disposition)
  - PROF-03 Jaccard CI gate: reads jaccard_stability.json, applies
    bootstrap_ci over the 48 per-layer Jaccard values, emits jaccard_ci_lower;
    FAILs gate when ci_lower < 0.94 (triggers D-06 re-profile-with-larger fallback)

Output:
  - output/profiling/reasoning-merged-v4/concentration_report.json

Usage:
    python -m scripts.compute_concentration \\
        --merged-jsonl output/profiling/reasoning-merged-v4/routing_report.jsonl \\
        --base-jsonl output/profiling/base_model_eeff.jsonl \\
        --jaccard-json output/profiling/reasoning-merged-v4/jaccard_stability.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from scripts.profile_base_model import compute_eeff, _nan_to_null


# ---------------------------------------------------------------------------
# Bootstrap CI (D-09)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of values.

    Uses np.random.choice + np.percentile. Symmetric alpha/2 tails.

    Args:
        values: 1-D array of values to resample.
        n_boot: Number of bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 -> 95% CI).

    Returns:
        (lo, hi): lower and upper CI bounds as floats.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng()
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# PROF-03 Jaccard CI gate (D-09 CI-aware disposition)
# ---------------------------------------------------------------------------


def jaccard_disposition(
    jaccard_array: np.ndarray,
    threshold: float = 0.94,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Compute CI-aware PROF-03 gate from per-layer Jaccard array.

    Applies bootstrap_ci over the 48 per-layer Jaccard values (resamples the
    48 layers -> CI of the across-layer mean). Uses ci_lower for the gate
    disposition (D-09: never bare point estimate >= bar).

    Args:
        jaccard_array: 1-D array of per-layer Jaccard values (typically 48 elements).
        threshold: Gate threshold (default 0.94, PROF-03 literal).
        n_boot: Bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 -> 95% CI).

    Returns:
        (ci_lower, passes): CI lower bound (float) and gate disposition (bool).
        passes = (ci_lower >= threshold).
        passes=False triggers D-06 re-profile-with-larger-subsample fallback.
    """
    lo, hi = bootstrap_ci(np.asarray(jaccard_array, dtype=float), n_boot=n_boot, alpha=alpha)
    ci_lower = float(lo)
    passes = bool(ci_lower >= threshold)
    return (ci_lower, passes)


# ---------------------------------------------------------------------------
# Concentration metrics (PROF-04)
# ---------------------------------------------------------------------------


def compute_cv(counts: np.ndarray) -> float:
    """Per-layer CV = counts.std() / counts.mean(). Returns 0.0 if mean == 0."""
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean()
    if mean == 0.0:
        return 0.0
    return float(counts.std() / mean)


def cumulative_coverage(counts: np.ndarray) -> np.ndarray:
    """Sorted-descending cumulative coverage curve.

    Args:
        counts: 1-D array of expert counts for one layer.

    Returns:
        np.ndarray of shape (n_experts,): cumsum(sorted_desc) / total.
        All values are in [0, 1]. Final value is 1.0 (within floating-point precision).
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0.0:
        return np.zeros(len(counts))
    sorted_desc = np.sort(counts)[::-1]
    return np.cumsum(sorted_desc) / total


def layer_depth_skew(cv_per_layer: np.ndarray) -> float:
    """Ratio of mean CV in early layers (0-15) vs late layers (32-47).

    Returns mean_early / mean_late. Returns 1.0 if either segment mean is 0.

    Args:
        cv_per_layer: 1-D array of per-layer CV values (length >= 48 expected).
    """
    cv_per_layer = np.asarray(cv_per_layer, dtype=float)
    early = cv_per_layer[:16]
    late = cv_per_layer[32:48]
    mean_early = float(early.mean()) if len(early) > 0 else 0.0
    mean_late = float(late.mean()) if len(late) > 0 else 0.0
    if mean_late == 0.0:
        return 1.0
    return mean_early / mean_late


def compute_eeff_delta(merged_eeff: float, base_eeff: float) -> float:
    """E_eff delta = merged_eeff - base_eeff.

    Positive: merged model is more diffuse (less concentrated) than base.
    Negative: merged model is more concentrated than base.
    """
    return merged_eeff - base_eeff


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def compute_concentration_report(
    merged_jsonl_path: str,
    base_jsonl_path: str,
    jaccard_json_path: str,
    output_path: str,
) -> dict:
    """Compute PROF-04 concentration metrics and write concentration_report.json.

    Args:
        merged_jsonl_path: JSONL from profile_merged_model.py (per-layer counts).
        base_jsonl_path: JSONL from profile_base_model.py (baseline E_eff).
        jaccard_json_path: JSON sidecar from profile_merged_model.py
            with "per_layer_jaccard" key (raw 48-element Jaccard array).
        output_path: Destination for concentration_report.json.

    Returns:
        The full report dict.
    """
    # --- Load merged JSONL ---
    merged_records = []
    with open(merged_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    merged_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed record: {e}")

    # --- Load base JSONL (D-08 delta join) ---
    base_records = []
    with open(base_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    base_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed base record: {e}")

    # Filter base to ratio "30_70" (normalize at join time — D-08)
    base_filtered = []
    for rec in base_records:
        raw_ratio = rec.get("ratio", "")
        normalized = raw_ratio.removeprefix("ratio_")
        if normalized == "30_70":
            base_filtered.append(rec)

    # Build base E_eff lookup by layer_idx
    base_eeff_by_layer: dict[int, float] = {}
    for rec in base_filtered:
        layer_idx = rec.get("layer_idx")
        eeff_total = rec.get("eeff_total")
        if layer_idx is not None and eeff_total is not None:
            base_eeff_by_layer[int(layer_idx)] = float(eeff_total)

    # Assert non-empty join (D-08 guard)
    matched_layers = [
        rec for rec in merged_records
        if rec.get("layer_idx") in base_eeff_by_layer
    ]
    assert len(matched_layers) > 0, (
        f"D-08 join yielded zero matched rows. "
        f"Merged records: {len(merged_records)}, "
        f"Base filtered (30_70): {len(base_filtered)}. "
        f"Check ratio key normalization."
    )

    # --- Load Jaccard sidecar ---
    with open(jaccard_json_path) as f:
        jaccard_data = json.load(f)
    raw_jaccards = np.array(jaccard_data["per_layer_jaccard"], dtype=float)

    # --- PROF-03 CI gate ---
    jaccard_ci_lower, jaccard_gate_passes = jaccard_disposition(raw_jaccards)
    jaccard_disposition_str = "PASS" if jaccard_gate_passes else "FAIL"

    # --- Per-layer metrics ---
    per_layer = []
    cv_list = []
    eeff_total_list = []
    eeff_wp_gen_list = []
    eeff_wp_judge_list = []
    eeff_delta_list = []

    for rec in sorted(merged_records, key=lambda r: r.get("layer_idx", 0)):
        layer_idx = int(rec.get("layer_idx", 0))

        # Build count arrays from the dicts
        total_dict = {int(k): v for k, v in rec.get("expert_counts_total", {}).items()}
        gen_dict = {int(k): v for k, v in rec.get("expert_counts_wp_gen", {}).items()}
        judge_dict = {int(k): v for k, v in rec.get("expert_counts_wp_judge", {}).items()}

        n_experts = 128
        counts_total = np.array([total_dict.get(e, 0) for e in range(n_experts)], dtype=float)
        counts_gen = np.array([gen_dict.get(e, 0) for e in range(n_experts)], dtype=float)
        counts_judge = np.array([judge_dict.get(e, 0) for e in range(n_experts)], dtype=float)

        cv = compute_cv(counts_total)
        cov_curve = cumulative_coverage(counts_total)

        eeff_total_val = compute_eeff(total_dict)
        eeff_gen_val = compute_eeff(gen_dict)
        eeff_judge_val = compute_eeff(judge_dict)

        # E_eff delta vs base (D-08)
        base_eeff_val = base_eeff_by_layer.get(layer_idx, float("nan"))
        if not math.isnan(eeff_total_val) and not math.isnan(base_eeff_val):
            delta = compute_eeff_delta(eeff_total_val, base_eeff_val)
        else:
            delta = float("nan")

        cv_list.append(cv)
        if not math.isnan(eeff_total_val):
            eeff_total_list.append(eeff_total_val)
        if not math.isnan(eeff_gen_val):
            eeff_wp_gen_list.append(eeff_gen_val)
        if not math.isnan(eeff_judge_val):
            eeff_wp_judge_list.append(eeff_judge_val)
        if not math.isnan(delta):
            eeff_delta_list.append(delta)

        per_layer.append({
            "layer_idx": layer_idx,
            "cv": _nan_to_null(cv),
            "eeff_total": _nan_to_null(eeff_total_val),
            "eeff_wp_gen": _nan_to_null(eeff_gen_val),
            "eeff_wp_judge": _nan_to_null(eeff_judge_val),
            "eeff_delta_vs_base": _nan_to_null(delta),
            "top1_coverage": float(cov_curve[0]) if len(cov_curve) > 0 else None,
            "top8_coverage": float(cov_curve[7]) if len(cov_curve) >= 8 else None,
        })

    # --- Aggregate statistics with bootstrap CI ---
    def _agg_with_ci(values: list[float], label: str) -> dict:
        if not values:
            return {"mean": None, "max": None, "variance": None, "ci_lower": None, "ci_upper": None}
        arr = np.array(values, dtype=float)
        lo, hi = bootstrap_ci(arr)
        return {
            "mean": _nan_to_null(float(np.nanmean(arr))),
            "max": _nan_to_null(float(np.nanmax(arr))),
            "variance": _nan_to_null(float(np.nanvar(arr))),
            "ci_lower": _nan_to_null(lo),
            "ci_upper": _nan_to_null(hi),
        }

    # Layer-depth skew
    depth_skew = layer_depth_skew(np.array(cv_list))

    # CV aggregate CI
    cv_arr = np.array(cv_list, dtype=float)
    cv_ci_lo, cv_ci_hi = bootstrap_ci(cv_arr) if len(cv_arr) > 0 else (float("nan"), float("nan"))

    # Assemble report
    report = {
        "analysis": "concentration_report",
        "model": "reasoning-merged-v4",
        "n_layers": 48,
        "n_experts": 128,
        # PROF-03 Jaccard CI gate
        "jaccard_ci_lower": _nan_to_null(jaccard_ci_lower),
        "jaccard_gate_disposition": jaccard_disposition_str,
        "jaccard_gate_passes": jaccard_gate_passes,
        "jaccard_mean": _nan_to_null(float(raw_jaccards.mean())),
        "jaccard_min": _nan_to_null(float(raw_jaccards.min())),
        # CV summary
        "cv_mean": _nan_to_null(float(cv_arr.mean())) if len(cv_arr) > 0 else None,
        "cv_ci_lower": _nan_to_null(cv_ci_lo),
        "cv_ci_upper": _nan_to_null(cv_ci_hi),
        # Layer-depth skew
        "layer_depth_skew_early_vs_late": _nan_to_null(depth_skew),
        # E_eff aggregates
        "eeff_total": _agg_with_ci(eeff_total_list, "total"),
        "eeff_wp_gen": _agg_with_ci(eeff_wp_gen_list, "wp_gen"),
        "eeff_wp_judge": _agg_with_ci(eeff_wp_judge_list, "wp_judge"),
        "eeff_delta_vs_base": _agg_with_ci(eeff_delta_list, "delta"),
        # Per-layer detail
        "per_layer": per_layer,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    if not jaccard_gate_passes:
        print(
            f"WARNING: PROF-03 gate FAILED — jaccard_ci_lower={jaccard_ci_lower:.4f} < 0.94. "
            f"D-06 fallback: re-run with larger subsample fraction."
        )
    else:
        print(f"PROF-03 gate PASS — jaccard_ci_lower={jaccard_ci_lower:.4f} >= 0.94")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute PROF-04 concentration metrics from merged-model profiling output"
    )
    parser.add_argument(
        "--merged-jsonl",
        default="output/profiling/reasoning-merged-v4/routing_report.jsonl",
        help="JSONL from profile_merged_model.py",
    )
    parser.add_argument(
        "--base-jsonl",
        default="output/profiling/base_model_eeff.jsonl",
        help="Baseline JSONL from profile_base_model.py (D-08 delta join)",
    )
    parser.add_argument(
        "--jaccard-json",
        default="output/profiling/reasoning-merged-v4/jaccard_stability.json",
        help="Jaccard sidecar JSON from profile_merged_model.py (PROF-03)",
    )
    parser.add_argument(
        "--output",
        default="output/profiling/reasoning-merged-v4/concentration_report.json",
        help="Output path for concentration_report.json",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    compute_concentration_report(
        merged_jsonl_path=str(project_root / args.merged_jsonl),
        base_jsonl_path=str(project_root / args.base_jsonl),
        jaccard_json_path=str(project_root / args.jaccard_json),
        output_path=str(project_root / args.output),
    )


if __name__ == "__main__":
    main()
