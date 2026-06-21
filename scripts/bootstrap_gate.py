"""
scripts/bootstrap_gate.py

CI-aware bootstrap gates for RL comparative evaluation (Phase 10).

Functions
---------
check_dim_regression(candidate_scores, baseline_scores) -> dict
    Gate: CI lower-bound of candidate mean >= mean(baseline).
    Resamples candidate only (one-sided adequacy test).

bootstrap_spearman_improvement(pred_rl, gt, pred_baseline) -> dict
    Gate: pair-level Spearman-improvement CI lower bound > 0.
    Resamples (pred_rl[i], gt[i], pred_baseline[i]) pairs together.
    Returns dict with lo, hi, improved_beyond_noise.

check_wpbench_gate(candidate_overall, knowledge_subscore, execution_subscore) -> dict
    Gate: direct point comparison (NO bootstrap).
    candidate_overall >= 0.4616  (weighted overall from metadata.scores.overall)
    knowledge_subscore >= 0.45   (from metadata.scores.knowledge)
    execution_subscore >= 0.375  (from metadata.scores.correctness — field name in JSON)
    All three must pass conjunctively.

check_no_routing_collapse(rl_metrics: list[dict]) -> dict
    Gate: no step has halt_reason set, kl >= 0.3, or efrac < 0.5.

CLI
---
python scripts/bootstrap_gate.py --eval-gen eval_gen_results.jsonl \\
    --baseline eval_gen_baseline.jsonl \\
    --wp-bench wp_bench_results.json \\
    --rl-metrics rl_metrics.jsonl \\
    --out gate_result.json

All values serialized as Python-native float/bool (not np.float64/np.bool_).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Gate 1 — dim-level regression check
# ---------------------------------------------------------------------------

BASELINE_WP_BENCH_OVERALL = 0.4616  # weighted overall from output/04.4_wp_bench_results.json


def check_dim_regression(
    candidate_scores: list[float],
    baseline_scores: list[float],
    n_boot: int = 1000,
) -> dict[str, Any]:
    """CI lower bound of candidate mean >= mean(baseline).

    Only candidate side is resampled (one-sided adequacy test).
    Passes when lo_cand >= baseline_mean.
    """
    if not candidate_scores:
        raise ValueError("candidate_scores is empty")
    if not baseline_scores:
        raise ValueError("baseline_scores is empty")

    import numpy as np
    from scripts.compute_concentration import bootstrap_ci

    cand_arr = np.array(candidate_scores, dtype=float)
    base_arr = np.array(baseline_scores, dtype=float)

    lo_cand, hi_cand = bootstrap_ci(cand_arr, n_boot=n_boot)
    baseline_mean = float(base_arr.mean())

    passed = bool(lo_cand >= baseline_mean)

    return {
        "passed": passed,
        "lo_cand": float(lo_cand),
        "hi_cand": float(hi_cand),
        "candidate_mean": float(cand_arr.mean()),
        "baseline_mean": baseline_mean,
        "n_candidate": len(candidate_scores),
        "n_baseline": len(baseline_scores),
    }


# ---------------------------------------------------------------------------
# Gate 2 — pair-level Spearman improvement
# ---------------------------------------------------------------------------


def bootstrap_spearman_improvement(
    pred_rl: list[float],
    gt: list[float],
    pred_baseline: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Pair-level Spearman-improvement bootstrap.

    Resamples (pred_rl[i], gt[i], pred_baseline[i]) pairs jointly (same row
    indices for all three), then computes spearmanr per resample and takes
    delta = rho_rl - rho_baseline.  CI of delta; improved_beyond_noise = lo > 0.

    IMPORTANT: does NOT call bootstrap_ci(corr_array) — that would compute
    CI of a MEAN of correlations, which is mathematically wrong for this gate.
    """
    if len(pred_rl) != len(gt) or len(pred_rl) != len(pred_baseline):
        raise ValueError("pred_rl, gt, pred_baseline must have equal length")
    if len(pred_rl) < 2:
        raise ValueError("Need at least 2 pairs for Spearman bootstrap")

    import numpy as np
    from scipy.stats import spearmanr

    pred_rl_arr = np.array(pred_rl, dtype=float)
    gt_arr = np.array(gt, dtype=float)
    pred_base_arr = np.array(pred_baseline, dtype=float)

    n = len(pred_rl_arr)
    rng = np.random.default_rng()
    deltas = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho_rl = spearmanr(pred_rl_arr[idx], gt_arr[idx]).statistic
        rho_base = spearmanr(pred_base_arr[idx], gt_arr[idx]).statistic
        # nan-safe: if degenerate resample, delta = 0 (conservatively not-improved)
        if rho_rl != rho_rl or rho_base != rho_base:
            deltas[i] = 0.0
        else:
            deltas[i] = rho_rl - rho_base

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    improved_beyond_noise = bool(lo > 0)

    return {
        "improved_beyond_noise": improved_beyond_noise,
        "lo": lo,
        "hi": hi,
        "rho_rl_point": float(spearmanr(pred_rl_arr, gt_arr).statistic),
        "rho_baseline_point": float(spearmanr(pred_base_arr, gt_arr).statistic),
        "n_pairs": n,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Gate 3 — wp-bench aggregate gate (DIRECT point comparison, no bootstrap)
# ---------------------------------------------------------------------------


def check_wpbench_gate(
    candidate_overall: float,
    knowledge_subscore: float,
    execution_subscore: float,
    baseline_aggregate: float = BASELINE_WP_BENCH_OVERALL,
    knowledge_floor: float = 0.45,
    execution_floor: float = 0.375,
) -> dict[str, Any]:
    """Direct point comparison gate for wp-bench (D-10-03 compliant).

    candidate_overall: metadata.scores.overall (weighted aggregate from vLLM)
    knowledge_subscore: metadata.scores.knowledge
    execution_subscore: metadata.scores.correctness (note: field name is 'correctness')

    Gate:
      overall_gate_passed = candidate_overall >= baseline_aggregate (0.4616)
      knowledge_floor_passed = knowledge_subscore >= 0.45
      execution_floor_passed = execution_subscore >= 0.375
      passed = all three

    D-10-03 discriminating case (must fail):
      candidate_overall=0.44, knowledge=0.50, execution=0.38
      -> passed=False (0.44 < 0.4616), despite BOTH sub-floors passing.
      (simple per-task mean ~= 0.49 WOULD have passed under old flat-array logic)
    """
    overall_gate_passed = bool(candidate_overall >= baseline_aggregate)
    knowledge_floor_passed = bool(knowledge_subscore >= knowledge_floor)
    execution_floor_passed = bool(execution_subscore >= execution_floor)
    passed = bool(overall_gate_passed and knowledge_floor_passed and execution_floor_passed)

    return {
        "passed": passed,
        "overall_gate_passed": overall_gate_passed,
        "knowledge_floor_passed": knowledge_floor_passed,
        "execution_floor_passed": execution_floor_passed,
        "candidate_overall": float(candidate_overall),
        "knowledge_subscore": float(knowledge_subscore),
        "execution_subscore": float(execution_subscore),
        "baseline_aggregate": float(baseline_aggregate),
        "knowledge_floor": float(knowledge_floor),
        "execution_floor": float(execution_floor),
    }


# ---------------------------------------------------------------------------
# Gate 4 — no routing collapse
# ---------------------------------------------------------------------------

KL_HARD = 0.3
EFRAC_HARD = 0.5


def check_no_routing_collapse(
    rl_metrics: list[dict],
    kl_hard: float = KL_HARD,
    efrac_hard: float = EFRAC_HARD,
) -> dict[str, Any]:
    """Passes iff no step triggered a routing-collapse condition.

    Conditions checked per step:
    - halt_reason set (non-None, non-empty)
    - kl_sample_train_v1 >= kl_hard (0.3)
    - e_frac_with_tokens_mean < efrac_hard (0.5)

    Any failure = passed=False.
    """
    if not rl_metrics:
        raise ValueError("rl_metrics is empty — need at least one step")

    halt_triggered = False
    kl_violation_step = None
    efrac_violation_step = None
    failure_reason = None

    for step_data in rl_metrics:
        step = step_data.get("step", "?")
        halt = step_data.get("halt_reason")
        kl = step_data.get("kl_sample_train_v1")
        efrac = step_data.get("e_frac_with_tokens_mean")

        if halt is not None and halt != "":
            halt_triggered = True
            failure_reason = f"halt_reason set at step {step}: {halt!r}"
            break

        if kl is not None and kl >= kl_hard:
            kl_violation_step = step
            failure_reason = f"kl_sample_train_v1={kl} >= {kl_hard} at step {step}"
            break

        if efrac is not None and efrac < efrac_hard:
            efrac_violation_step = step
            failure_reason = f"e_frac_with_tokens_mean={efrac} < {efrac_hard} at step {step}"
            break

    passed = bool(
        not halt_triggered
        and kl_violation_step is None
        and efrac_violation_step is None
    )

    return {
        "passed": passed,
        "halt_triggered": halt_triggered,
        "kl_violation_step": kl_violation_step,
        "efrac_violation_step": efrac_violation_step,
        "failure_reason": failure_reason,
        "n_steps": len(rl_metrics),
        "kl_hard": float(kl_hard),
        "efrac_hard": float(efrac_hard),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 10 bootstrap gates for RL comparative evaluation"
    )
    parser.add_argument("--eval-gen", type=Path, help="RL eval_gen_results.jsonl (per-example)")
    parser.add_argument("--baseline", type=Path, help="Baseline eval_gen_results.jsonl")
    parser.add_argument("--wp-bench", type=Path, help="wp_bench_results.json")
    parser.add_argument("--rl-metrics", type=Path, help="rl_metrics.jsonl")
    parser.add_argument("--dim", default="reasoning_score", help="Score field to gate on (default: reasoning_score)")
    parser.add_argument("--out", type=Path, default=Path("gate_result.json"), help="Output JSON")
    args = parser.parse_args()

    results: dict[str, Any] = {}

    # Gate 1: dim regression
    if args.eval_gen and args.baseline:
        cand_records = _load_jsonl(args.eval_gen)
        base_records = _load_jsonl(args.baseline)
        cand_scores = [r["dimension_scores"][args.dim] for r in cand_records if args.dim in r.get("dimension_scores", {})]
        base_scores = [r["dimension_scores"][args.dim] for r in base_records if args.dim in r.get("dimension_scores", {})]
        results["dim_regression"] = check_dim_regression(cand_scores, base_scores)

    # Gate 2: Spearman improvement
    if args.eval_gen and args.baseline:
        cand_records = cand_records if "cand_records" in dir() else _load_jsonl(args.eval_gen)
        base_records = base_records if "base_records" in dir() else _load_jsonl(args.baseline)
        # Align by example_id
        cand_map = {r["example_id"]: r for r in cand_records if "example_id" in r}
        base_map = {r["example_id"]: r for r in base_records if "example_id" in r}
        common_ids = sorted(set(cand_map) & set(base_map))
        if common_ids:
            pred_rl = [cand_map[eid]["dimension_scores"].get(args.dim, 0.0) for eid in common_ids]
            pred_base = [base_map[eid]["dimension_scores"].get(args.dim, 0.0) for eid in common_ids]
            gt = [cand_map[eid].get("gt_score", cand_map[eid].get("label", 0)) for eid in common_ids]
            results["spearman_improvement"] = bootstrap_spearman_improvement(pred_rl, gt, pred_base)

    # Gate 3: wp-bench
    if args.wp_bench:
        wp = json.loads(args.wp_bench.read_text())
        scores = wp.get("metadata", {}).get("scores", {})
        results["wpbench"] = check_wpbench_gate(
            candidate_overall=scores.get("overall", 0.0),
            knowledge_subscore=scores.get("knowledge", 0.0),
            execution_subscore=scores.get("correctness", 0.0),  # field is "correctness" not "execution"
        )

    # Gate 4: routing collapse
    if args.rl_metrics:
        metrics = _load_jsonl(args.rl_metrics)
        results["no_routing_collapse"] = check_no_routing_collapse(metrics)

    all_passed = all(g["passed"] for g in results.values() if isinstance(g, dict) and "passed" in g)
    results["all_gates_passed"] = bool(all_passed)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Gate results written to {out_path}")
    print(f"all_gates_passed: {all_passed}")


if __name__ == "__main__":
    _cli_main()
