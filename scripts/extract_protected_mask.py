"""Protected expert mask extraction for MoE router profiling (D-03/D-04).

Implements the D-03 conservative co-activation rule: an expert is "protected"
(excluded from MoE-Sieve pruning) iff it fires above the per-layer mean for
BOTH wp_gen AND wp_judge token types.

D-04 sensitivity table computes three threshold variants alongside the chosen
mask (mean, median, top-K intersection) to quantify stability of the rule.

Output (default under output/profiling/reasoning-merged-v4/):
  - protected_expert_mask.npy         ([48, 128] bool array)
  - protected_expert_mask.json        ({layer_idx: [expert_ids]} sidecar)
  - sensitivity_table.json            (D-04 three-threshold comparison)
  - protected_mask_result.json        (full analysis result)

Usage:
    python -m scripts.extract_protected_mask \\
        --merged-jsonl output/profiling/reasoning-merged-v4/routing_report.jsonl \\
        --output-dir output/profiling/reasoning-merged-v4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.compute_concentration import bootstrap_ci
from scripts.sieve_arch import infer_dims_from_records


# ---------------------------------------------------------------------------
# D-03 Conservative Co-activation Mask
# ---------------------------------------------------------------------------


def extract_protected_mask(
    counts_wp_gen: np.ndarray,
    counts_wp_judge: np.ndarray,
) -> np.ndarray:
    """Extract D-03 conservative co-activation protected-expert mask.

    An expert e at layer l is protected iff:
      counts_wp_gen[l, e] > mean_gen[l]  AND
      counts_wp_judge[l, e] > mean_judge[l]

    Args:
        counts_wp_gen: [n_layers, n_experts] float array of wp_gen routing counts.
        counts_wp_judge: [n_layers, n_experts] float array of wp_judge routing counts.

    Returns:
        np.ndarray of shape (n_layers, n_experts), dtype bool.
        True = expert is protected (dual-purpose).
    """
    counts_wp_gen = np.asarray(counts_wp_gen, dtype=float)
    counts_wp_judge = np.asarray(counts_wp_judge, dtype=float)

    mean_gen = counts_wp_gen.mean(axis=1, keepdims=True)
    mean_judge = counts_wp_judge.mean(axis=1, keepdims=True)

    mask = (counts_wp_gen > mean_gen) & (counts_wp_judge > mean_judge)
    return mask.astype(bool)


def extract_protected_mask_single_task(counts_total: np.ndarray) -> np.ndarray:
    """Judge-only single-task specialization of the D-03 rule (T-25-02).

    The v4 judge has no <wp_gen>/<wp_judge> token split (sieve_arch.resolve_task_token_ids
    returns (None, None) for this tokenizer -- see 20-04/22-02), so a profile of it carries
    expert_counts_total only; expert_counts_wp_gen/wp_judge are always empty. The dual-task
    AND-rule in extract_protected_mask() would degenerate to an always-False mask against an
    all-zero counts_wp_gen/counts_wp_judge input. D-03's conservative co-activation rule is
    "fires above the per-layer mean for BOTH tasks" -- when there is only one task, both-tasks
    collapses to one task: an expert is protected iff it fires above its layer's mean count.

    Args:
        counts_total: [n_layers, n_experts] float array of total routing counts.

    Returns:
        np.ndarray of shape (n_layers, n_experts), dtype bool.
    """
    counts_total = np.asarray(counts_total, dtype=float)
    mean_total = counts_total.mean(axis=1, keepdims=True)
    return (counts_total > mean_total).astype(bool)


# ---------------------------------------------------------------------------
# D-04 Sensitivity Table
# ---------------------------------------------------------------------------


def sensitivity_table(
    counts_wp_gen: np.ndarray,
    counts_wp_judge: np.ndarray,
    top_k: int = 16,
) -> dict:
    """Compute D-04 sensitivity table: three threshold variants for the co-activation mask.

    Variants:
      - mean_threshold: D-03 conservative (expert > per-layer mean in both)
      - median_threshold: expert > per-layer median in both
      - topk_intersection_k{top_k}: expert in top-K by count in both splits

    Args:
        counts_wp_gen: [n_layers, n_experts] float array.
        counts_wp_judge: [n_layers, n_experts] float array.
        top_k: K for the top-K intersection variant (default 16 = top_k*2 from plan).

    Returns:
        dict with three keys, each mapping to:
          {"mask_size_per_layer": [int, ...], "total_protected": int}
    """
    counts_wp_gen = np.asarray(counts_wp_gen, dtype=float)
    counts_wp_judge = np.asarray(counts_wp_judge, dtype=float)
    n_layers, n_experts = counts_wp_gen.shape

    def _mask_stats(mask: np.ndarray) -> dict:
        per_layer = [int(mask[l].sum()) for l in range(n_layers)]
        return {
            "mask_size_per_layer": per_layer,
            "total_protected": int(mask.sum()),
        }

    # --- mean threshold (D-03) ---
    mean_gen = counts_wp_gen.mean(axis=1, keepdims=True)
    mean_judge = counts_wp_judge.mean(axis=1, keepdims=True)
    mask_mean = (counts_wp_gen > mean_gen) & (counts_wp_judge > mean_judge)

    # --- median threshold ---
    med_gen = np.median(counts_wp_gen, axis=1, keepdims=True)
    med_judge = np.median(counts_wp_judge, axis=1, keepdims=True)
    mask_median = (counts_wp_gen > med_gen) & (counts_wp_judge > med_judge)

    # --- top-K intersection ---
    actual_k = min(top_k, n_experts)
    topk_mask = np.zeros((n_layers, n_experts), dtype=bool)
    for layer in range(n_layers):
        gen_topk = set(np.argsort(counts_wp_gen[layer])[-actual_k:].tolist())
        judge_topk = set(np.argsort(counts_wp_judge[layer])[-actual_k:].tolist())
        for e in gen_topk & judge_topk:
            topk_mask[layer, e] = True

    return {
        "mean_threshold": _mask_stats(mask_mean),
        "median_threshold": _mask_stats(mask_median),
        f"topk_intersection_k{actual_k}": _mask_stats(topk_mask),
    }


def sensitivity_table_single_task(counts_total: np.ndarray, top_k: int = 16) -> dict:
    """Judge-only D-04 sensitivity table: three threshold variants on ONE distribution.

    Single-task specialization of sensitivity_table() -- see extract_protected_mask_single_task
    for the collapse rationale. All three variants (mean/median/top-K) are computed against the
    same counts_total; the "intersection" in topk_intersection is against itself (trivially the
    plain top-K set), kept for schema parity with the dual-task D-04 table.
    """
    counts_total = np.asarray(counts_total, dtype=float)
    n_layers, n_experts = counts_total.shape

    def _mask_stats(mask: np.ndarray) -> dict:
        per_layer = [int(mask[l].sum()) for l in range(n_layers)]
        return {"mask_size_per_layer": per_layer, "total_protected": int(mask.sum())}

    mean_total = counts_total.mean(axis=1, keepdims=True)
    mask_mean = counts_total > mean_total

    med_total = np.median(counts_total, axis=1, keepdims=True)
    mask_median = counts_total > med_total

    actual_k = min(top_k, n_experts)
    topk_mask = np.zeros((n_layers, n_experts), dtype=bool)
    for layer in range(n_layers):
        for e in np.argsort(counts_total[layer])[-actual_k:].tolist():
            topk_mask[layer, e] = True

    return {
        "mean_threshold": _mask_stats(mask_mean),
        "median_threshold": _mask_stats(mask_median),
        f"topk_intersection_k{actual_k}": _mask_stats(topk_mask),
    }


# ---------------------------------------------------------------------------
# E_eff report (Task 2, GATE4-03) -- per-stratum + overall E_eff, E_eff/n_experts
# ratio (the redundancy-headroom signal), and protected-experts-per-layer stats.
# Reuses strata_eeff already computed by profile_merged_model.py (jaccard_stability.json)
# rather than re-deriving per-stratum entropy here.
# ---------------------------------------------------------------------------


def build_eeff_report(
    records: list[dict],
    strata_eeff: dict,
    mask: np.ndarray,
    n_experts: int,
) -> dict:
    """Assemble eeff_report.json content: per-stratum + overall E_eff/n_experts + protected-per-layer.

    Args:
        records: Parsed routing_report.jsonl rows (for the "overall" eeff_total column).
        strata_eeff: The {"deltanet": {...}, "attention": {...}} dict from jaccard_stability.json.
        mask: [n_layers, n_experts] bool protected-expert mask.
        n_experts: Total expert count (denominator of the E_eff/n_experts ratio).

    Returns:
        dict with "deltanet", "attention" (each augmented with eeff_over_n_experts_ratio),
        "overall", "protected_per_layer", and "n_experts".
    """
    eeff_vals = [
        r["eeff_total"] for r in records
        if r.get("eeff_total") is not None
    ]
    arr = np.array(eeff_vals, dtype=float) if eeff_vals else np.array([])
    overall = {
        "mean": float(arr.mean()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "var": float(arr.var()) if arr.size else None,
        "n_layers": int(arr.size),
        "eeff_over_n_experts_ratio": float(arr.mean() / n_experts) if arr.size else None,
    }

    per_layer_protected = mask.sum(axis=1)
    protected_per_layer = {
        "mean": float(per_layer_protected.mean()),
        "min": int(per_layer_protected.min()),
        "max": int(per_layer_protected.max()),
    }

    report: dict = {}
    for stratum_name, stats in strata_eeff.items():
        stats = dict(stats)
        if stats.get("mean") is not None:
            stats["eeff_over_n_experts_ratio"] = stats["mean"] / n_experts
        report[stratum_name] = stats
    report["overall"] = overall
    report["protected_per_layer"] = protected_per_layer
    report["n_experts"] = n_experts
    return report


# ---------------------------------------------------------------------------
# Mask Export
# ---------------------------------------------------------------------------


def export_mask(
    mask: np.ndarray,
    out_dir: Path,
) -> None:
    """Export protected-expert mask as .npy and JSON sidecar.

    Args:
        mask: [n_layers, n_experts] bool array.
        out_dir: Directory to write files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write .npy
    np.save(out_dir / "protected_expert_mask.npy", mask)

    # Write JSON sidecar: {str(layer_idx): [int expert_ids]}
    sidecar = {
        str(layer_idx): [int(e) for e in np.where(mask[layer_idx])[0].tolist()]
        for layer_idx in range(mask.shape[0])
    }
    with open(out_dir / "protected_expert_mask.json", "w") as f:
        json.dump(sidecar, f, indent=2)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_and_report(
    merged_jsonl_path: str,
    output_dir: str,
    top_k_mask: int = 16,
    model_tag: str = "reasoning-merged-v4",
    stimulus: str = "data/final_dataset/ratio_30_70/openai_train.jsonl",
    jaccard_json_path: str | None = None,
) -> dict:
    """Load merged profiling JSONL, extract D-03 mask, write all outputs.

    Args:
        merged_jsonl_path: JSONL from profile_merged_model.py.
        output_dir: Directory for all output files.
        top_k_mask: K for top-K intersection sensitivity variant (default 16).
        model_tag: Recorded into result["model"] (metadata only).
        stimulus: Recorded into result["stimulus"] (metadata only).
        jaccard_json_path: If given, path to the sibling jaccard_stability.json (has
            strata_eeff from profile_merged_model.py). When provided, also writes
            eeff_report.json to out_dir (Task 2, GATE4-03).

    Returns:
        Full result dict (also written to protected_mask_result.json).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONL
    records = []
    with open(merged_jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed record: {e}")

    # Arch-derive dims from the JSONL itself (GATE4-02 SC1) -- (40, 256) for a v4
    # report, (48, 128) for a v3 report; both round-trip, no hardcoded dim.
    n_layers, n_experts = infer_dims_from_records(records)

    # Build [n_layers, n_experts] count arrays per split
    counts_total = np.zeros((n_layers, n_experts), dtype=float)
    counts_gen = np.zeros((n_layers, n_experts), dtype=float)
    counts_judge = np.zeros((n_layers, n_experts), dtype=float)

    for rec in records:
        layer_idx = int(rec.get("layer_idx", 0))
        if layer_idx >= n_layers:
            continue
        for k, v in rec.get("expert_counts_total", {}).items():
            e = int(k)
            if 0 <= e < n_experts:
                counts_total[layer_idx, e] = float(v)
        for k, v in rec.get("expert_counts_wp_gen", {}).items():
            e = int(k)
            if 0 <= e < n_experts:
                counts_gen[layer_idx, e] = float(v)
        for k, v in rec.get("expert_counts_wp_judge", {}).items():
            e = int(k)
            if 0 <= e < n_experts:
                counts_judge[layer_idx, e] = float(v)

    # T-25-02: a profile with no <wp_gen>/<wp_judge> task-token split (e.g. the v4
    # judge -- sieve_arch.resolve_task_token_ids returns (None, None) for its
    # tokenizer) carries expert_counts_total only; counts_gen/counts_judge are all
    # zero and the dual-task AND-rule below would degenerate to an always-False
    # mask. Detect that degenerate case and dispatch to the single-task rule
    # instead (the D-03 co-activation rule's single-task collapse).
    single_task = bool(counts_gen.sum() == 0 and counts_judge.sum() == 0 and counts_total.sum() > 0)
    rule_name = "D-03_single_task_mean_threshold" if single_task else "D-03_conservative_co_activation"

    if single_task:
        mask = extract_protected_mask_single_task(counts_total)
        sens = sensitivity_table_single_task(counts_total, top_k=top_k_mask)
    else:
        # Extract D-03 mask
        mask = extract_protected_mask(counts_gen, counts_judge)

        # D-04 sensitivity table
        sens = sensitivity_table(counts_gen, counts_judge, top_k=top_k_mask)

    # Bootstrap CI on mask size per layer (D-09 CI-aware reporting)
    mask_sizes = mask.sum(axis=1).astype(float)
    ci_lo, ci_hi = bootstrap_ci(mask_sizes)

    # Per-layer detail
    per_layer = []
    for layer_idx in range(n_layers):
        protected_ids = [int(e) for e in np.where(mask[layer_idx])[0].tolist()]
        per_layer.append({
            "layer_idx": layer_idx,
            "n_protected": len(protected_ids),
            "protected_expert_ids": protected_ids,
        })

    # Export mask files
    export_mask(mask, out_dir)

    # Write sensitivity table JSON
    with open(out_dir / "sensitivity_table.json", "w") as f:
        json.dump(sens, f, indent=2)

    # Assemble result JSON
    result = {
        "analysis": "protected_expert_mask_extraction",
        "model": model_tag,
        "stimulus": stimulus,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": 8,
        "rule": rule_name,
        "single_task": single_task,
        "total_protected": int(mask.sum()),
        "mean_protected_per_layer": float(mask.sum(axis=1).mean()),
        "mask_size_ci_lower": float(ci_lo),
        "mask_size_ci_upper": float(ci_hi),
        "sensitivity_table": sens,
        "per_layer": per_layer,
    }

    (out_dir / "protected_mask_result.json").write_text(json.dumps(result, indent=2))
    print(
        f"D-03 mask: {int(mask.sum())} protected experts across {n_layers} layers "
        f"(mean {float(mask.sum(axis=1).mean()):.1f}/layer)"
    )

    if jaccard_json_path is not None:
        with open(jaccard_json_path) as f:
            jaccard_data = json.load(f)
        strata_eeff = jaccard_data.get("strata_eeff", {})
        eeff_report = build_eeff_report(records, strata_eeff, mask, n_experts)
        (out_dir / "eeff_report.json").write_text(json.dumps(eeff_report, indent=2))
        print(f"eeff_report.json written: overall={eeff_report.get('overall')}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract D-03 protected-expert mask from merged-model profiling output"
    )
    parser.add_argument(
        "--merged-jsonl",
        default="output/profiling/reasoning-merged-v4/routing_report.jsonl",
        help="JSONL from profile_merged_model.py",
    )
    parser.add_argument(
        "--output-dir",
        default="output/profiling/reasoning-merged-v4",
        help="Directory for all output files",
    )
    parser.add_argument(
        "--top-k-mask",
        type=int,
        default=16,
        help="K for top-K intersection sensitivity variant (default 16)",
    )
    parser.add_argument("--model-tag", default="reasoning-merged-v4", help="Recorded into result['model']")
    parser.add_argument(
        "--stimulus",
        default="data/final_dataset/ratio_30_70/openai_train.jsonl",
        help="Recorded into result['stimulus']",
    )
    parser.add_argument(
        "--jaccard-json",
        default=None,
        help="Path to the sibling jaccard_stability.json (strata_eeff source). "
             "If given, also writes eeff_report.json to --output-dir.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    jaccard_path = str(project_root / args.jaccard_json) if args.jaccard_json else None

    extract_and_report(
        merged_jsonl_path=str(project_root / args.merged_jsonl),
        output_dir=str(project_root / args.output_dir),
        top_k_mask=args.top_k_mask,
        model_tag=args.model_tag,
        stimulus=args.stimulus,
        jaccard_json_path=jaccard_path,
    )


if __name__ == "__main__":
    main()
