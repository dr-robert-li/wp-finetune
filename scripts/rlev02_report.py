"""
scripts/rlev02_report.py

RLEV-02 report generator and five-part conjunctive gate aggregator (Phase 10).

Functions
---------
check_antihack_gate(perturbed_rl_rewards, clean_v12_rewards) -> dict
    Anti-hack gate: hi_perturbed_rl < lo_clean_v12.
    Reuses compute_axis_gate from build_antihack_set.py directly.
    clean_v12_rewards MUST be a parameter — never hard-code fixture values.

check_jaccard_retention(rl_metric_steps, bar=0.85) -> dict
    Gate: mean(jaccard_protected across steps) >= bar.
    bar=0.85 is the provisional RL retention bar.
    NOTE: 0.9426 is SFT cross-run profiling stability (Phase 7) — different quantity.

apply_conjunctive_gate(gate_results: dict[str, dict]) -> dict
    Five-part conjunctive gate: ALL must pass.
    Gates: judge_spearman_improvement, wpbench_hard_gate,
           antihack_no_reward_hack, protected_expert_retention, no_routing_collapse.

build_report(gate_results, checkpoint_step, run_id) -> dict
    Assembles four report sections + conjunctive gate summary.
    Returns JSON-serializable dict (all values Python-native float/bool).

CLI
---
python scripts/rlev02_report.py \\
    --eval-gen eval_gen_results.jsonl \\
    --baseline eval_gen_baseline.jsonl \\
    --wp-bench wp_bench_results.json \\
    --rl-metrics rl_metrics.jsonl \\
    --antihack-perturbed antihack_perturbed_rl.jsonl \\
    --antihack-clean antihack_clean_v12.jsonl \\
    --checkpoint-step 200 \\
    --run-id rl-v3.0-candidate \\
    --out rlev02_report.json
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
# Anti-hack gate (reuses compute_axis_gate from build_antihack_set.py)
# ---------------------------------------------------------------------------


def check_antihack_gate(
    perturbed_rl_rewards: list[float],
    clean_v12_rewards: list[float],
    axis_name: str = "rl_vs_clean_v12",
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Anti-hack gate: hi_perturbed_rl < lo_clean_v12.

    Reuses compute_axis_gate from build_antihack_set.py directly.
    The clean_v12_rewards MUST come from a LIVE v1.2 SFT baseline — this
    function takes them as a parameter (never hard-codes fixture values like
    0.666/0.689 which are synthetic test bounds, not a real SFT distribution).

    Passes when hi_perturbed_rl < lo_clean_v12 (no reward hack detected).
    """
    from scripts.build_antihack_set import compute_axis_gate

    axis_result = compute_axis_gate(perturbed_rl_rewards, clean_v12_rewards, axis_name, n_boot=n_boot)

    # Map compute_axis_gate field names to RLEV-02 naming convention
    passed = bool(axis_result["gate_pass"])
    return {
        "passed": passed,
        "hi_perturbed_rl": float(axis_result["hi_perturbed"]),
        "lo_clean_v12": float(axis_result["lo_clean"]),
        "hi_clean_v12": float(axis_result["hi_clean"]),
        "lo_perturbed_rl": float(axis_result["lo_perturbed"]),
        "perturbed_mean": float(axis_result["perturbed_mean"]),
        "clean_mean": float(axis_result["clean_mean"]),
        "n_perturbed": axis_result["n_perturbed"],
        "n_clean": axis_result["n_clean"],
        "axis": axis_result["axis"],
    }


# ---------------------------------------------------------------------------
# Jaccard retention gate
# ---------------------------------------------------------------------------


def check_jaccard_retention(
    rl_metric_steps: list[dict],
    bar: float = 0.85,
) -> dict[str, Any]:
    """Gate: mean(jaccard_protected across RL steps) >= bar.

    bar=0.85 is the provisional RL per-step jaccard retention bar.
    This is NOT 0.9426 — that value is SFT cross-run profiling stability
    (Phase 7, a different quantity measuring cross-run expert routing consistency,
    not RL per-step protected-expert jaccard overlap).

    bar is configurable and will be confirmed at the D-10-04 human-review
    checkpoint after the Phase 9 live run produces real jaccard_protected traces.
    """
    if not rl_metric_steps:
        raise ValueError("rl_metric_steps is empty")

    jaccards = [
        float(step["jaccard_protected"])
        for step in rl_metric_steps
        if "jaccard_protected" in step
    ]

    if not jaccards:
        raise ValueError("No jaccard_protected values found in rl_metric_steps")

    mean_jaccard = float(sum(jaccards) / len(jaccards))
    passed = bool(mean_jaccard >= bar)

    return {
        "passed": passed,
        "mean_jaccard": mean_jaccard,
        "bar": float(bar),
        "n_steps": len(jaccards),
        "min_jaccard": float(min(jaccards)),
        "max_jaccard": float(max(jaccards)),
    }


# ---------------------------------------------------------------------------
# Five-part conjunctive gate
# ---------------------------------------------------------------------------

_FIVE_GATES = [
    "judge_spearman_improvement",
    "wpbench_hard_gate",
    "antihack_no_reward_hack",
    "protected_expert_retention",
    "no_routing_collapse",
]


def apply_conjunctive_gate(
    gate_results: dict[str, dict],
) -> dict[str, Any]:
    """Five-part conjunctive gate: ALL must pass.

    Each sub-gate's result dict must have a 'passed' key.
    Gates:
      1. judge_spearman_improvement  — pair-level Spearman CI lower > 0
      2. wpbench_hard_gate           — candidate_overall >= 0.4616 (D-10-03)
      3. antihack_no_reward_hack     — hi_perturbed_rl < lo_clean_v12
      4. protected_expert_retention  — mean(jaccard_protected) >= 0.85
      5. no_routing_collapse         — no halt/kl/efrac violation

    Returns dict with all_gates_passed and failing_gates list.
    """
    failing_gates = []
    for gate_name in _FIVE_GATES:
        gate_data = gate_results.get(gate_name, {})
        # Check "passed" key; also accept "improved_beyond_noise" for gate #1
        if gate_name == "judge_spearman_improvement":
            gate_passed = bool(
                gate_data.get("passed", gate_data.get("improved_beyond_noise", False))
            )
        else:
            gate_passed = bool(gate_data.get("passed", False))
        if not gate_passed:
            failing_gates.append(gate_name)

    all_gates_passed = bool(len(failing_gates) == 0)

    return {
        "all_gates_passed": all_gates_passed,
        "failing_gates": failing_gates,
        "gate_count": len(_FIVE_GATES),
        "passed_count": len(_FIVE_GATES) - len(failing_gates),
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    gate_results: dict[str, dict],
    checkpoint_step: int,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the RLEV-02 report dict.

    Four sections:
      1. metadata       — run_id, checkpoint_step, timestamp
      2. gate_details   — per-gate results verbatim
      3. conjunctive_gate — five-part gate summary (all_gates_passed)
      4. recommendation — human-readable verdict string

    Returns JSON-serializable dict.
    """
    import datetime

    conjunctive = apply_conjunctive_gate(gate_results)

    if conjunctive["all_gates_passed"]:
        recommendation = (
            "PROCEED: all five gates passed. "
            "Present to human reviewer for final D-10-04 sign-off before declaring v3.0."
        )
    else:
        failing = ", ".join(conjunctive["failing_gates"])
        recommendation = (
            f"BLOCKED: gate(s) failed: {failing}. "
            "Do not promote to v3.0. Investigate and remediate before re-evaluation."
        )

    return {
        "metadata": {
            "run_id": run_id,
            "checkpoint_step": checkpoint_step,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "report_version": "rlev02-v1",
        },
        "gate_details": gate_results,
        "conjunctive_gate": conjunctive,
        "recommendation": recommendation,
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
    from scripts.bootstrap_gate import (
        check_dim_regression,
        bootstrap_spearman_improvement,
        check_wpbench_gate,
        check_no_routing_collapse,
    )

    parser = argparse.ArgumentParser(description="Phase 10 RLEV-02 report generator")
    parser.add_argument("--eval-gen", type=Path, required=True, help="RL eval_gen_results.jsonl")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval_gen_results.jsonl")
    parser.add_argument("--wp-bench", type=Path, required=True, help="wp_bench_results.json")
    parser.add_argument("--rl-metrics", type=Path, required=True, help="rl_metrics.jsonl")
    parser.add_argument("--antihack-perturbed", type=Path, help="Perturbed RL rewards .jsonl (anti-hack axis)")
    parser.add_argument("--antihack-clean", type=Path, help="Clean v1.2 SFT rewards .jsonl (anti-hack baseline)")
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--run-id", default="rl-candidate")
    parser.add_argument("--dim", default="reasoning_score")
    parser.add_argument("--out", type=Path, default=Path("rlev02_report.json"))
    args = parser.parse_args()

    gate_results: dict[str, dict] = {}

    # --- Load eval-gen records ---
    cand_records = _load_jsonl(args.eval_gen)
    base_records = _load_jsonl(args.baseline)

    cand_scores = [
        r["dimension_scores"][args.dim]
        for r in cand_records
        if args.dim in r.get("dimension_scores", {})
    ]
    base_scores = [
        r["dimension_scores"][args.dim]
        for r in base_records
        if args.dim in r.get("dimension_scores", {})
    ]

    # Gate 1: Spearman improvement (pair-level)
    cand_map = {r["example_id"]: r for r in cand_records if "example_id" in r}
    base_map = {r["example_id"]: r for r in base_records if "example_id" in r}
    common_ids = sorted(set(cand_map) & set(base_map))
    if common_ids:
        pred_rl = [cand_map[eid]["dimension_scores"].get(args.dim, 0.0) for eid in common_ids]
        pred_base = [base_map[eid]["dimension_scores"].get(args.dim, 0.0) for eid in common_ids]
        gt = [cand_map[eid].get("gt_score", cand_map[eid].get("label", 0)) for eid in common_ids]
        spearman_result = bootstrap_spearman_improvement(pred_rl, gt, pred_base)
        # Normalize "passed" key for conjunctive gate
        spearman_result["passed"] = spearman_result["improved_beyond_noise"]
        gate_results["judge_spearman_improvement"] = spearman_result

    # Gate 2: wp-bench
    wp = json.loads(args.wp_bench.read_text())
    scores = wp.get("metadata", {}).get("scores", {})
    gate_results["wpbench_hard_gate"] = check_wpbench_gate(
        candidate_overall=scores.get("overall", 0.0),
        knowledge_subscore=scores.get("knowledge", 0.0),
        execution_subscore=scores.get("correctness", 0.0),
    )

    # Gate 3: anti-hack
    if args.antihack_perturbed and args.antihack_clean:
        perturbed_records = _load_jsonl(args.antihack_perturbed)
        clean_records = _load_jsonl(args.antihack_clean)
        perturbed_rewards = [r["reward"] for r in perturbed_records if "reward" in r]
        clean_rewards = [r["reward"] for r in clean_records if "reward" in r]
        gate_results["antihack_no_reward_hack"] = check_antihack_gate(
            perturbed_rl_rewards=perturbed_rewards,
            clean_v12_rewards=clean_rewards,
        )
    else:
        gate_results["antihack_no_reward_hack"] = {
            "passed": None,
            "note": "antihack-perturbed/clean not provided; gate skipped",
        }

    # Gate 4: jaccard retention
    rl_metrics = _load_jsonl(args.rl_metrics)
    gate_results["protected_expert_retention"] = check_jaccard_retention(rl_metrics)

    # Gate 5: routing collapse
    gate_results["no_routing_collapse"] = check_no_routing_collapse(rl_metrics)

    # Assemble report
    report = build_report(
        gate_results=gate_results,
        checkpoint_step=args.checkpoint_step,
        run_id=args.run_id,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    all_passed = report["conjunctive_gate"]["all_gates_passed"]
    print(f"RLEV-02 report written to {args.out}")
    print(f"all_gates_passed: {all_passed}")
    if not all_passed:
        print(f"failing: {report['conjunctive_gate']['failing_gates']}")


if __name__ == "__main__":
    _cli_main()
