"""Triage decision script for experiment elimination (GATE-02).

Reads per-experiment eval results from output/eval_triage/ and applies elimination
rules to select surviving experiments for Phase 7 fine-tuned adapter profiling.

Experiment directories are auto-discovered: any subdirectory of the eval triage
output that contains an ``eval_gen_results.json`` file is treated as an experiment.
This replaces the former hardcoded ``ratio_*`` naming convention while remaining
backward-compatible with it.

Elimination rules (D-12, D-13):
  1. Hard gates (strict > required; value AT threshold FAILS):
     - PHPCS pass rate > 0.95
     - Judge Spearman > 0.85
     - Security pass rate > 0.98
  2. 5pp rule: eliminated if (best_overall_score - experiment_score) > 0.05
     Exactly 5pp behind SURVIVES (low bar for continuation per D-13).

NO_SURVIVORS handling: if zero experiments pass all gates, returns status="NO_SURVIVORS"
with recommendation (does not crash).

Usage:
    python -m scripts.triage_ratios
    python -m scripts.triage_ratios --eval-dir output/eval_triage
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import namedtuple
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# All gates use strict > (not >=). A ratio AT the threshold FAILS.
# ---------------------------------------------------------------------------

PHPCS_GATE = 0.95       # must be strictly > 0.95
SPEARMAN_GATE = 0.85    # must be strictly > 0.85
SECURITY_GATE = 0.98    # must be strictly > 0.98
ELIMINATION_PP = 0.05   # eliminated if (best - experiment) > 0.05 (strictly greater)

# Deprecated: kept for backward compatibility with run_eval_triage.py and
# profile_base_model.py which import this constant.  New code should use
# discover_experiments() instead.
RATIO_ORDER = ["30_70", "40_60", "50_50", "60_40", "70_30"]

# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------


def discover_experiments(base_dir: Path) -> list[str]:
    """Auto-discover experiment directories under the eval triage output.

    Any subdirectory containing ``eval_gen_results.json`` is treated as an
    experiment.  Results are returned sorted by name for deterministic ordering.

    Args:
        base_dir: Path to the eval triage output directory.

    Returns:
        Sorted list of experiment directory names.
    """
    if not base_dir.exists():
        return []
    return sorted([
        d.name for d in base_dir.iterdir()
        if d.is_dir() and (d / "eval_gen_results.json").exists()
    ])

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class TriageResult(namedtuple(
    "_TriageResult",
    [
        "survivors",
        "eliminated",
        "best_experiment",
        "status",
        "wpbench_available",
        "triage_table",
        "gen_quality_scores",   # dict[experiment, float] — (phpcs + security) / 2
        "judge_calibrations",   # dict[experiment, float] — spearman correlation
    ],
)):
    """Result of GATE-02 triage elimination."""

    @property
    def best_ratio(self):
        """Deprecated alias for best_experiment (backward compatibility)."""
        return self.best_experiment

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_results(eval_triage_dir: str) -> dict:
    """Read per-experiment eval JSON files from eval_triage_dir.

    Auto-discovers experiment directories via :func:`discover_experiments`.
    For each experiment directory, looks for:
      {eval_triage_dir}/{experiment}/eval_gen_results.json
      {eval_triage_dir}/{experiment}/eval_judge_results.json
      {eval_triage_dir}/{experiment}/wp_bench_results.json   (optional)

    Extracts: phpcs_pass_rate, security_pass_rate, spearman (overall),
    overall_mean, wpbench_score (None if missing).

    Args:
        eval_triage_dir: Path to directory containing experiment subdirectories.

    Returns:
        Dict mapping experiment name -> eval result dict.
    """
    base = Path(eval_triage_dir)
    experiments = discover_experiments(base)
    results = {}

    for experiment in experiments:
        experiment_dir = base / experiment

        gen_path = experiment_dir / "eval_gen_results.json"
        judge_path = experiment_dir / "eval_judge_results.json"
        bench_path = experiment_dir / "wp_bench_results.json"

        if not gen_path.exists() or not judge_path.exists():
            logger.warning(f"Missing eval files for experiment {experiment} at {experiment_dir}")
            continue

        gen_data = json.loads(gen_path.read_text())
        judge_data = json.loads(judge_path.read_text())

        # Extract spearman -- support multiple key names and formats
        raw_spearman = (
            judge_data.get("overall_spearman")
            or judge_data.get("spearman_corr")
            or judge_data.get("spearman")
            or judge_data.get("spearman_overall")
            or 0.0
        )
        # overall_spearman may be a dict {"corr": float, "p_value": float, ...}
        if isinstance(raw_spearman, dict):
            spearman = float(raw_spearman.get("corr", 0.0))
        else:
            spearman = float(raw_spearman)

        # wp-bench score (optional)
        wpbench_score = None
        if bench_path.exists():
            bench_data = json.loads(bench_path.read_text())
            wpbench_score = (
                bench_data.get("overall_score")
                or bench_data.get("wpbench_score")
                or bench_data.get("score")
            )

        results[experiment] = {
            "phpcs_pass_rate": gen_data.get("phpcs_pass_rate", 0.0),
            "security_pass_rate": gen_data.get("security_pass_rate", 0.0),
            "spearman": spearman,
            "overall_mean": gen_data.get("overall_mean", 0.0),
            "wpbench_score": wpbench_score,
        }

    return results


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_gen_quality_score(phpcs_rate: float, security_rate: float) -> float:
    """Compute generation quality score from two proportion metrics.

    Both inputs are proportions (0.0-1.0), so averaging them is well-defined.
    This score is used for the 5pp elimination rule.

    Args:
        phpcs_rate: PHPCS pass rate (0.0-1.0).
        security_rate: Security pass rate (0.0-1.0).

    Returns:
        Generation quality score (0.0-1.0).
    """
    return (phpcs_rate + security_rate) / 2


def compute_overall_score(phpcs_rate: float, security_rate: float, spearman: float) -> float:
    """Deprecated: blended score mixing proportions and correlation.

    Kept for backward compatibility. New code should use compute_gen_quality_score()
    for generation quality ranking and treat spearman as a separate axis.

    Args:
        phpcs_rate: PHPCS pass rate (0.0-1.0).
        security_rate: Security pass rate (0.0-1.0).
        spearman: Overall Spearman correlation (0.0-1.0).

    Returns:
        Blended score (0.0-1.0). Do not use for ranking — use gen_quality_score instead.
    """
    return 0.6 * ((phpcs_rate + security_rate) / 2) + 0.4 * spearman


# ---------------------------------------------------------------------------
# Core triage logic
# ---------------------------------------------------------------------------


def triage_ratios(
    eval_results: dict,
    profiling_summary: Optional[dict] = None,
) -> TriageResult:
    """Apply GATE-02 elimination logic to select surviving experiments.

    Hard gates (strict > required -- value AT threshold FAILS):
      - PHPCS pass rate > PHPCS_GATE (0.95)
      - Judge Spearman > SPEARMAN_GATE (0.85)
      - Security pass rate > SECURITY_GATE (0.98)

    5pp rule (per D-13, low bar for continuation):
      - Among gate-passers, compute overall score via compute_overall_score()
      - Eliminated strictly if (best_score - experiment_score) > ELIMINATION_PP (0.05)
      - Exactly 5pp behind = NOT eliminated (D-13: only clearly failing are removed)

    NO_SURVIVORS: if no experiments pass all gates, returns status="NO_SURVIVORS" with
    recommendation. Does not crash.

    Args:
        eval_results: Dict mapping experiment name -> eval result dict.
            Required keys per experiment: phpcs_pass_rate, security_pass_rate, spearman.
        profiling_summary: Optional dict with E_eff data (informational only,
            not used for elimination).

    Returns:
        TriageResult namedtuple with:
            survivors: list[str] -- experiments that passed all gates and 5pp rule
            eliminated: list[dict] -- {experiment, reason} for each eliminated experiment
            best_experiment: str|None -- experiment with highest overall score (None if NO_SURVIVORS)
            status: str -- "OK" or "NO_SURVIVORS"
            wpbench_available: bool -- True if any experiment has a wpbench_score
            triage_table: str -- human-readable markdown summary table
    """
    survivors = []
    eliminated = []
    # gate_passers: experiment -> gen_quality_score (proportion, used for 5pp elimination)
    gate_passers = {}
    # Track both axes for all gate-passing experiments
    gen_quality_scores_all: dict = {}   # experiment -> gen_quality_score
    judge_calibrations_all: dict = {}   # experiment -> spearman

    # Check hard gates
    for experiment, data in eval_results.items():
        phpcs = data.get("phpcs_pass_rate", 0.0)
        security = data.get("security_pass_rate", 0.0)
        spearman = data.get("spearman", 0.0)

        # Hard gate checks (strictly greater than threshold)
        if phpcs <= PHPCS_GATE:
            eliminated.append({
                "experiment": experiment,
                "reason": (
                    f"PHPCS gate failed: {phpcs:.4f} not strictly > {PHPCS_GATE} "
                    f"(strict > {PHPCS_GATE} required)"
                ),
            })
            continue

        if spearman <= SPEARMAN_GATE:
            eliminated.append({
                "experiment": experiment,
                "reason": (
                    f"Spearman gate failed: {spearman:.4f} not strictly > {SPEARMAN_GATE} "
                    f"(strict > {SPEARMAN_GATE} required)"
                ),
            })
            continue

        if security <= SECURITY_GATE:
            eliminated.append({
                "experiment": experiment,
                "reason": (
                    f"Security gate failed: {security:.4f} not strictly > {SECURITY_GATE} "
                    f"(strict > {SECURITY_GATE} required)"
                ),
            })
            continue

        # Passed all hard gates — track both axes independently
        gen_q = compute_gen_quality_score(phpcs, security)
        gate_passers[experiment] = gen_q
        gen_quality_scores_all[experiment] = gen_q
        judge_calibrations_all[experiment] = spearman

    # 5pp elimination rule among gate-passers — uses gen_quality_score only
    # (proportions are comparable; spearman is a separate axis)
    if gate_passers:
        best_score = max(gate_passers.values())
        best_experiment = max(gate_passers, key=lambda r: gate_passers[r])

        for experiment, score in gate_passers.items():
            diff = best_score - score
            if diff > ELIMINATION_PP:
                eliminated.append({
                    "experiment": experiment,
                    "reason": (
                        f"5pp elimination: gen_quality_score {diff:.4f} behind best ({best_experiment}). "
                        f"Threshold: strictly > {ELIMINATION_PP}"
                    ),
                })
            else:
                survivors.append(experiment)
    else:
        best_experiment = None

    # NO_SURVIVORS handling
    if not survivors:
        status = "NO_SURVIVORS"
        best_experiment = None
    else:
        status = "OK"

    # wp-bench availability
    wpbench_available = any(
        data.get("wpbench_score") is not None
        for data in eval_results.values()
    )

    # Build triage table
    triage_table = _build_triage_table(
        eval_results=eval_results,
        gate_passers=gate_passers,
        gen_quality_scores=gen_quality_scores_all,
        judge_calibrations=judge_calibrations_all,
        survivors=survivors,
        eliminated=eliminated,
        best_experiment=best_experiment,
        status=status,
        wpbench_available=wpbench_available,
        profiling_summary=profiling_summary,
    )

    return TriageResult(
        survivors=survivors,
        eliminated=eliminated,
        best_experiment=best_experiment,
        status=status,
        wpbench_available=wpbench_available,
        triage_table=triage_table,
        gen_quality_scores=gen_quality_scores_all,
        judge_calibrations=judge_calibrations_all,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _build_triage_table(
    eval_results: dict,
    gate_passers: dict,
    gen_quality_scores: dict,
    judge_calibrations: dict,
    survivors: list,
    eliminated: list,
    best_experiment: Optional[str],
    status: str,
    wpbench_available: bool,
    profiling_summary: Optional[dict] = None,
) -> str:
    """Build a human-readable markdown triage decision table."""
    experiment_order = sorted(eval_results.keys())

    lines = [
        f"STATUS: {status}",
        "",
        "## Triage Decision",
        "",
        "### Hard Gate Results",
        "",
        "| Experiment | PHPCS Rate | Spearman | Security Rate | PHPCS Gate | Spearman Gate | Security Gate |",
        "|------------|------------|----------|---------------|------------|---------------|---------------|",
    ]

    for experiment in experiment_order:
        data = eval_results[experiment]
        phpcs = data.get("phpcs_pass_rate", 0.0)
        spearman = data.get("spearman", 0.0)
        security = data.get("security_pass_rate", 0.0)

        phpcs_pass = "PASS" if phpcs > PHPCS_GATE else "FAIL"
        spearman_pass = "PASS" if spearman > SPEARMAN_GATE else "FAIL"
        security_pass = "PASS" if security > SECURITY_GATE else "FAIL"

        lines.append(
            f"| {experiment} | {phpcs:.4f} | {spearman:.4f} | {security:.4f} "
            f"| {phpcs_pass} | {spearman_pass} | {security_pass} |"
        )

    if gate_passers:
        lines += [
            "",
            "### Generation Quality Ranking — Axis 1: gen_quality_score = (phpcs + security) / 2",
            "",
            "Both phpcs_rate and security_rate are proportions (0-1); averaging is well-defined.",
            "The 5pp elimination rule applies to this axis only.",
            "",
            "| Experiment | Gen Quality Score | Behind Best | Verdict |",
            "|------------|-------------------|-------------|---------|",
        ]
        best_score = max(gate_passers.values()) if gate_passers else 0.0
        for experiment in sorted(gate_passers, key=lambda r: gate_passers[r], reverse=True):
            score = gate_passers[experiment]
            diff = best_score - score
            verdict = "BEST" if experiment == best_experiment else (
                "ELIMINATED (>5pp)" if diff > ELIMINATION_PP else "SURVIVOR"
            )
            lines.append(
                f"| {experiment} | {score:.4f} | {diff:.4f} | {verdict} |"
            )

        lines += [
            "",
            "### Judge Calibration Ranking — Axis 2: Spearman correlation",
            "",
            "Spearman measures judge-GT agreement (rank correlation), not a proportion.",
            "Reported as a separate axis; not mixed into gen_quality_score.",
            "",
            "| Experiment | Spearman |",
            "|------------|----------|",
        ]
        for experiment in sorted(judge_calibrations, key=lambda r: judge_calibrations[r], reverse=True):
            spearman_val = judge_calibrations[experiment]
            lines.append(f"| {experiment} | {spearman_val:.4f} |")

    lines += [
        "",
        "### Survivors",
        "",
        f"Experiments proceeding to Phase 7: {', '.join(survivors) if survivors else 'NONE'}",
        "",
        "### Eliminated Experiments",
        "",
    ]
    for e in eliminated:
        lines.append(f"- **{e['experiment']}**: {e['reason']}")
    if not eliminated:
        lines.append("- None")

    # wp-bench section
    lines += ["", "### wp-bench Scores", ""]
    if wpbench_available:
        lines.append("| Experiment | wp-bench Score |")
        lines.append("|------------|----------------|")
        for experiment in experiment_order:
            score = eval_results[experiment].get("wpbench_score")
            if score is not None:
                lines.append(f"| {experiment} | {score:.1f} |")
    else:
        lines.append(
            "wp-bench was skipped (--skip-wpbench). "
            "Triage based on static eval gates only. "
            "wp-bench differentiation deferred."
        )

    # E_eff informational section (if profiling data provided)
    if profiling_summary:
        lines += ["", "### E_eff Summary (Informational)", ""]
        for experiment, eeff_data in profiling_summary.items():
            lines.append(f"- **{experiment}**: {eeff_data}")

    # NO_SURVIVORS recommendation
    if status == "NO_SURVIVORS":
        lines += [
            "",
            "### NO_SURVIVORS: Recommendation",
            "",
            "All experiments failed hard gates. Consider:",
            "1. Re-examine training data quality",
            "2. Investigate specific failure dimensions",
            "3. Lower gate thresholds if domain warrants",
        ]

    return "\n".join(lines)


def write_triage_decision(
    triage_result: TriageResult,
    profiling_summary: Optional[dict],
    out_path: str,
) -> None:
    """Write triage decision markdown to file.

    The file begins with a machine-parseable STATUS line.

    Args:
        triage_result: TriageResult from triage_ratios().
        profiling_summary: Optional E_eff summary dict.
        out_path: Path to write output markdown.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(triage_result.triage_table + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GATE-02 triage decision for experiment elimination")
    parser.add_argument(
        "--eval-dir",
        default="output/eval_triage",
        help="Directory containing experiment eval subdirectories (auto-discovered)",
    )
    parser.add_argument(
        "--profiling-dir",
        default="output/profiling",
        help="Directory containing base_model_eeff_summary.md",
    )
    parser.add_argument(
        "--output",
        default="output/triage_decision.md",
        help="Output path for triage decision markdown",
    )
    args = parser.parse_args()

    eval_results = load_eval_results(args.eval_dir)
    if not eval_results:
        print(f"ERROR: No eval results found in {args.eval_dir}")
        return 1

    triage_result = triage_ratios(eval_results)
    write_triage_decision(triage_result, profiling_summary=None, out_path=args.output)

    print(f"STATUS: {triage_result.status}")
    print(f"Survivors: {', '.join(triage_result.survivors) if triage_result.survivors else 'NONE'}")
    print(f"Best experiment: {triage_result.best_experiment or 'N/A'}")
    print(f"Triage decision written to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
