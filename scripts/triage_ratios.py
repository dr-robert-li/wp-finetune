"""Triage decision script for ratio elimination (GATE-02).

Reads per-ratio eval results from output/eval_triage/ and applies elimination
rules to select surviving ratios for Phase 7 fine-tuned adapter profiling.

Elimination rules (D-12, D-13):
  1. Hard gates (strict > required; value AT threshold FAILS):
     - PHPCS pass rate > 0.95
     - Judge Spearman > 0.85
     - Security pass rate > 0.98
  2. 5pp rule: eliminated if (best_overall_score - ratio_score) > 0.05
     Exactly 5pp behind SURVIVES (low bar for continuation per D-13).

NO_SURVIVORS handling: if zero ratios pass all gates, returns status="NO_SURVIVORS"
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
ELIMINATION_PP = 0.05   # eliminated if (best - ratio) > 0.05 (strictly greater)

RATIO_ORDER = ["30_70", "40_60", "50_50", "60_40", "70_30"]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

TriageResult = namedtuple(
    "TriageResult",
    ["survivors", "eliminated", "best_ratio", "status", "wpbench_available", "triage_table"],
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_results(eval_triage_dir: str) -> dict:
    """Read per-ratio eval JSON files from eval_triage_dir.

    Looks for:
      {eval_triage_dir}/ratio_{r}/eval_gen_results.json
      {eval_triage_dir}/ratio_{r}/eval_judge_results.json
      {eval_triage_dir}/ratio_{r}/wp_bench_results.json   (optional)

    Extracts: phpcs_pass_rate, security_pass_rate, spearman (overall),
    overall_mean, wpbench_score (None if missing).

    Args:
        eval_triage_dir: Path to directory containing ratio_* subdirectories.

    Returns:
        Dict mapping ratio string -> eval result dict.
    """
    base = Path(eval_triage_dir)
    results = {}

    for ratio in RATIO_ORDER:
        ratio_dir = base / f"ratio_{ratio}"
        if not ratio_dir.exists():
            continue

        gen_path = ratio_dir / "eval_gen_results.json"
        judge_path = ratio_dir / "eval_judge_results.json"
        bench_path = ratio_dir / "wp_bench_results.json"

        if not gen_path.exists() or not judge_path.exists():
            logger.warning(f"Missing eval files for ratio {ratio} at {ratio_dir}")
            continue

        gen_data = json.loads(gen_path.read_text())
        judge_data = json.loads(judge_path.read_text())

        # Extract spearman -- support multiple key names
        spearman = (
            judge_data.get("overall_spearman")
            or judge_data.get("spearman")
            or judge_data.get("spearman_overall")
            or 0.0
        )

        # wp-bench score (optional)
        wpbench_score = None
        if bench_path.exists():
            bench_data = json.loads(bench_path.read_text())
            wpbench_score = (
                bench_data.get("overall_score")
                or bench_data.get("wpbench_score")
                or bench_data.get("score")
            )

        results[ratio] = {
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


def compute_overall_score(phpcs_rate: float, security_rate: float, spearman: float) -> float:
    """Compute gen-weighted overall score per D-11.

    Formula: 0.6 * ((phpcs_rate + security_rate) / 2) + 0.4 * spearman

    PHPCS + security get more weight (gen quality is user-facing).
    Spearman is judge calibration (refined later via GRPO).

    Args:
        phpcs_rate: PHPCS pass rate (0.0-1.0).
        security_rate: Security pass rate (0.0-1.0).
        spearman: Overall Spearman correlation (0.0-1.0).

    Returns:
        Weighted overall score (0.0-1.0).
    """
    return 0.6 * ((phpcs_rate + security_rate) / 2) + 0.4 * spearman


# ---------------------------------------------------------------------------
# Core triage logic
# ---------------------------------------------------------------------------


def triage_ratios(
    eval_results: dict,
    profiling_summary: Optional[dict] = None,
) -> TriageResult:
    """Apply GATE-02 elimination logic to select surviving ratios.

    Hard gates (strict > required -- value AT threshold FAILS):
      - PHPCS pass rate > PHPCS_GATE (0.95)
      - Judge Spearman > SPEARMAN_GATE (0.85)
      - Security pass rate > SECURITY_GATE (0.98)

    5pp rule (per D-13, low bar for continuation):
      - Among gate-passers, compute overall score via compute_overall_score()
      - Eliminated strictly if (best_score - ratio_score) > ELIMINATION_PP (0.05)
      - Exactly 5pp behind = NOT eliminated (D-13: only clearly failing are removed)

    NO_SURVIVORS: if no ratios pass all gates, returns status="NO_SURVIVORS" with
    recommendation. Does not crash.

    Args:
        eval_results: Dict mapping ratio string -> eval result dict.
            Required keys per ratio: phpcs_pass_rate, security_pass_rate, spearman.
        profiling_summary: Optional dict with E_eff data (informational only,
            not used for elimination).

    Returns:
        TriageResult namedtuple with:
            survivors: list[str] -- ratios that passed all gates and 5pp rule
            eliminated: list[dict] -- {ratio, reason} for each eliminated ratio
            best_ratio: str|None -- ratio with highest overall score (None if NO_SURVIVORS)
            status: str -- "OK" or "NO_SURVIVORS"
            wpbench_available: bool -- True if any ratio has a wpbench_score
            triage_table: str -- human-readable markdown summary table
    """
    survivors = []
    eliminated = []
    gate_passers = {}  # ratio -> overall_score

    # Check hard gates
    for ratio, data in eval_results.items():
        phpcs = data.get("phpcs_pass_rate", 0.0)
        security = data.get("security_pass_rate", 0.0)
        spearman = data.get("spearman", 0.0)

        # Hard gate checks (strictly greater than threshold)
        if phpcs <= PHPCS_GATE:
            eliminated.append({
                "ratio": ratio,
                "reason": (
                    f"PHPCS gate failed: {phpcs:.4f} not strictly > {PHPCS_GATE} "
                    f"(strict > {PHPCS_GATE} required)"
                ),
            })
            continue

        if spearman <= SPEARMAN_GATE:
            eliminated.append({
                "ratio": ratio,
                "reason": (
                    f"Spearman gate failed: {spearman:.4f} not strictly > {SPEARMAN_GATE} "
                    f"(strict > {SPEARMAN_GATE} required)"
                ),
            })
            continue

        if security <= SECURITY_GATE:
            eliminated.append({
                "ratio": ratio,
                "reason": (
                    f"Security gate failed: {security:.4f} not strictly > {SECURITY_GATE} "
                    f"(strict > {SECURITY_GATE} required)"
                ),
            })
            continue

        # Passed all hard gates
        score = compute_overall_score(phpcs, security, spearman)
        gate_passers[ratio] = score

    # 5pp elimination rule among gate-passers
    if gate_passers:
        best_score = max(gate_passers.values())
        best_ratio = max(gate_passers, key=lambda r: gate_passers[r])

        for ratio, score in gate_passers.items():
            diff = best_score - score
            if diff > ELIMINATION_PP:
                eliminated.append({
                    "ratio": ratio,
                    "reason": (
                        f"5pp elimination: {diff:.4f} behind best ({best_ratio}) "
                        f"overall score. Threshold: strictly > {ELIMINATION_PP}"
                    ),
                })
            else:
                survivors.append(ratio)
    else:
        best_ratio = None

    # NO_SURVIVORS handling
    if not survivors:
        status = "NO_SURVIVORS"
        best_ratio = None
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
        survivors=survivors,
        eliminated=eliminated,
        best_ratio=best_ratio,
        status=status,
        wpbench_available=wpbench_available,
        profiling_summary=profiling_summary,
    )

    return TriageResult(
        survivors=survivors,
        eliminated=eliminated,
        best_ratio=best_ratio,
        status=status,
        wpbench_available=wpbench_available,
        triage_table=triage_table,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _build_triage_table(
    eval_results: dict,
    gate_passers: dict,
    survivors: list,
    eliminated: list,
    best_ratio: Optional[str],
    status: str,
    wpbench_available: bool,
    profiling_summary: Optional[dict] = None,
) -> str:
    """Build a human-readable markdown triage decision table."""
    lines = [
        f"STATUS: {status}",
        "",
        "## Triage Decision",
        "",
        "### Hard Gate Results",
        "",
        "| Ratio | PHPCS Rate | Spearman | Security Rate | PHPCS Gate | Spearman Gate | Security Gate |",
        "|-------|------------|----------|---------------|------------|---------------|---------------|",
    ]

    for ratio in RATIO_ORDER:
        if ratio not in eval_results:
            continue
        data = eval_results[ratio]
        phpcs = data.get("phpcs_pass_rate", 0.0)
        spearman = data.get("spearman", 0.0)
        security = data.get("security_pass_rate", 0.0)

        phpcs_pass = "PASS" if phpcs > PHPCS_GATE else "FAIL"
        spearman_pass = "PASS" if spearman > SPEARMAN_GATE else "FAIL"
        security_pass = "PASS" if security > SECURITY_GATE else "FAIL"

        lines.append(
            f"| {ratio} | {phpcs:.4f} | {spearman:.4f} | {security:.4f} "
            f"| {phpcs_pass} | {spearman_pass} | {security_pass} |"
        )

    if gate_passers:
        formula = "0.6 * ((phpcs + security) / 2) + 0.4 * spearman"
        lines += [
            "",
            f"### Overall Score Ranking ({formula})",
            "",
            "| Ratio | Overall Score | Behind Best | Verdict |",
            "|-------|---------------|-------------|---------|",
        ]
        best_score = max(gate_passers.values()) if gate_passers else 0.0
        for ratio in sorted(gate_passers, key=lambda r: gate_passers[r], reverse=True):
            score = gate_passers[ratio]
            diff = best_score - score
            verdict = "BEST" if ratio == best_ratio else (
                "ELIMINATED (>5pp)" if diff > ELIMINATION_PP else "SURVIVOR"
            )
            lines.append(
                f"| {ratio} | {score:.4f} | {diff:.4f} | {verdict} |"
            )

    lines += [
        "",
        "### Survivors",
        "",
        f"Ratios proceeding to Phase 7: {', '.join(survivors) if survivors else 'NONE'}",
        "",
        "### Eliminated Ratios",
        "",
    ]
    for e in eliminated:
        lines.append(f"- **{e['ratio']}**: {e['reason']}")
    if not eliminated:
        lines.append("- None")

    # wp-bench section
    lines += ["", "### wp-bench Scores", ""]
    if wpbench_available:
        lines.append("| Ratio | wp-bench Score |")
        lines.append("|-------|----------------|")
        for ratio in RATIO_ORDER:
            if ratio not in eval_results:
                continue
            score = eval_results[ratio].get("wpbench_score")
            if score is not None:
                lines.append(f"| {ratio} | {score:.1f} |")
    else:
        lines.append(
            "wp-bench was skipped (--skip-wpbench). "
            "Triage based on static eval gates only. "
            "wp-bench differentiation deferred."
        )

    # E_eff informational section (if profiling data provided)
    if profiling_summary:
        lines += ["", "### E_eff Summary (Informational)", ""]
        for ratio, eeff_data in profiling_summary.items():
            lines.append(f"- **{ratio}**: {eeff_data}")

    # NO_SURVIVORS recommendation
    if status == "NO_SURVIVORS":
        lines += [
            "",
            "### NO_SURVIVORS: Recommendation",
            "",
            "All ratios failed hard gates. Consider:",
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
    parser = argparse.ArgumentParser(description="GATE-02 triage decision for ratio elimination")
    parser.add_argument(
        "--eval-dir",
        default="output/eval_triage",
        help="Directory containing ratio_* eval subdirectories",
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
    print(f"Best ratio: {triage_result.best_ratio or 'N/A'}")
    print(f"Triage decision written to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
