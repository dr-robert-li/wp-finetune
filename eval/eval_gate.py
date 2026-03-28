"""Quality gate -- exits non-zero if any evaluation threshold fails.

Reads thresholds from config/train_config.yaml (eval section) and
result JSON files from output/. Prints pass/fail summary for each gate.

Supports:
  - Overall mean score target (gen results)
  - Per-dimension gen pass-rate targets (score >= 8/10)
  - Overall Spearman correlation target (judge results)
  - Per-dimension judge correlation targets
  - Legacy thresholds (phpcs, spearman, security) for backward compat

Usage:
    python -m eval.eval_gate [--results-dir PATH] [--config PATH]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from scripts.dgx_toolbox import get_toolbox

# Default config path (relative to project root)
DEFAULT_CONFIG_PATH = "config/train_config.yaml"

# Fallback thresholds if config does not have an eval section
_FALLBACK_THRESHOLDS = {
    "phpcs_pass_target": 0.95,
    "spearman_target": 0.85,
    "security_pass_target": 0.98,
    "overall_mean_target": 75.0,
    "overall_spearman_target": 0.80,
    "gen_dimension_targets": {},
    "judge_dimension_targets": {},
}


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------

def load_thresholds(config_path: Optional[str] = None) -> dict:
    """Load evaluation thresholds from config/train_config.yaml eval section.

    Args:
        config_path: Path to train_config.yaml. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        dict with overall, per-dimension, and legacy threshold keys.
        Falls back to _FALLBACK_THRESHOLDS if file missing or section absent.
    """
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return dict(_FALLBACK_THRESHOLDS)

    with path.open() as f:
        config = yaml.safe_load(f) or {}

    eval_section = config.get("eval", {})

    return {
        # Overall targets
        "overall_mean_target": eval_section.get(
            "overall_mean_target", _FALLBACK_THRESHOLDS["overall_mean_target"]
        ),
        "overall_spearman_target": eval_section.get(
            "overall_spearman_target", _FALLBACK_THRESHOLDS["overall_spearman_target"]
        ),
        # Per-dimension targets
        "gen_dimension_targets": eval_section.get(
            "gen_dimension_targets", _FALLBACK_THRESHOLDS["gen_dimension_targets"]
        ),
        "judge_dimension_targets": eval_section.get(
            "judge_dimension_targets", _FALLBACK_THRESHOLDS["judge_dimension_targets"]
        ),
        # Legacy (backward compat)
        "phpcs_pass_target": eval_section.get(
            "phpcs_pass_target", _FALLBACK_THRESHOLDS["phpcs_pass_target"]
        ),
        "spearman_target": eval_section.get(
            "spearman_target", _FALLBACK_THRESHOLDS["spearman_target"]
        ),
        "security_pass_target": eval_section.get(
            "security_pass_target", _FALLBACK_THRESHOLDS["security_pass_target"]
        ),
    }


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def check_gates(results: dict, thresholds: dict) -> tuple:
    """Check each evaluation metric against its threshold.

    Args:
        results: dict containing gen and judge metrics (overall + per-dimension).
        thresholds: dict from load_thresholds() with target values.

    Returns:
        (passed: bool, gate_rows: list[dict]) where each gate_row has keys:
            gate (str), target (float), actual (float), passed (bool).
        Overall passed is True iff all individual gates pass.
    """
    gate_rows = []

    def _check(gate_name: str, actual: float, target: float):
        gate_rows.append({
            "gate": gate_name,
            "target": target,
            "actual": actual,
            "passed": actual >= target,
        })

    # --- Overall gen mean score ---
    _check(
        "overall_mean_score",
        results.get("overall_mean", 0.0),
        thresholds["overall_mean_target"],
    )

    # --- Per-dimension gen pass rates ---
    gen_dim_targets = thresholds.get("gen_dimension_targets", {})
    gen_dim_actuals = results.get("gen_dimension_pass_rates", {})
    for dim, target in sorted(gen_dim_targets.items()):
        _check(
            f"gen_pass_rate/{dim}",
            gen_dim_actuals.get(dim, 0.0),
            target,
        )

    # --- Overall judge Spearman ---
    _check(
        "overall_spearman",
        results.get("overall_spearman", 0.0),
        thresholds["overall_spearman_target"],
    )

    # --- Per-dimension judge correlations ---
    judge_dim_targets = thresholds.get("judge_dimension_targets", {})
    judge_dim_actuals = results.get("judge_dimension_correlations", {})
    for dim, target in sorted(judge_dim_targets.items()):
        _check(
            f"judge_corr/{dim}",
            judge_dim_actuals.get(dim, 0.0),
            target,
        )

    # --- Legacy thresholds (backward compat) ---
    _check(
        "legacy/phpcs_pass_rate",
        results.get("phpcs_pass_rate", 0.0),
        thresholds["phpcs_pass_target"],
    )
    _check(
        "legacy/spearman_corr",
        results.get("spearman_corr", 0.0),
        thresholds["spearman_target"],
    )
    _check(
        "legacy/security_pass_rate",
        results.get("security_pass_rate", 0.0),
        thresholds["security_pass_target"],
    )

    all_passed = all(row["passed"] for row in gate_rows)
    return (all_passed, gate_rows)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(gate_rows: list, all_passed: bool) -> None:
    """Print a clear summary table of all gate results."""
    # Column widths
    max_gate = max(len(r["gate"]) for r in gate_rows)
    col_gate = max(max_gate, 4)  # min width for "Gate"

    print("=" * 60)
    print("EVALUATION QUALITY GATE")
    print("=" * 60)
    header = f"  {'Gate':<{col_gate}}  {'Target':>8}  {'Actual':>8}  {'Result':>6}"
    print(header)
    print("  " + "-" * (col_gate + 28))

    for row in gate_rows:
        status = "PASS" if row["passed"] else "FAIL"
        # Format: rates as percentage if <= 1.0, else as number
        def _fmt(v):
            if v <= 1.0:
                return f"{v:.3f}"
            return f"{v:.1f}"

        print(
            f"  {row['gate']:<{col_gate}}  {_fmt(row['target']):>8}  "
            f"{_fmt(row['actual']):>8}  {status:>6}"
        )

    print("  " + "-" * (col_gate + 28))
    if all_passed:
        print("  GATE: PASS - All evaluation thresholds met.")
    else:
        fail_count = sum(1 for r in gate_rows if not r["passed"])
        print(f"  GATE: FAIL - {fail_count} threshold(s) not met.")
    print()


# ---------------------------------------------------------------------------
# Main evaluation gate runner
# ---------------------------------------------------------------------------

def run_gate(
    results_dir: str = "output",
    config_path: Optional[str] = None,
) -> tuple:
    """Load results and thresholds, run all gates, return (passed, gate_rows).

    Args:
        results_dir: Directory containing eval_gen_results.json and eval_judge_results.json.
        config_path: Path to train_config.yaml. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        (passed: bool, gate_rows: list[dict])
    """
    # Resolve DGX toolbox (for consistency and future use)
    _dgx = get_toolbox()

    thresholds = load_thresholds(config_path)

    # Load eval_gen results
    gen_path = Path(results_dir) / "eval_gen_results.json"
    judge_path = Path(results_dir) / "eval_judge_results.json"

    combined = {}

    if gen_path.exists():
        with gen_path.open() as f:
            gen_results = json.load(f)
        combined["overall_mean"] = gen_results.get("overall_mean", 0.0)
        combined["gen_dimension_pass_rates"] = gen_results.get(
            "dimension_pass_rates", {}
        )
        # Legacy keys
        combined["phpcs_pass_rate"] = gen_results.get("phpcs_pass_rate", 0.0)
        combined["security_pass_rate"] = gen_results.get("security_pass_rate", 0.0)
    else:
        print(f"WARNING: {gen_path} not found -- treating as 0.0", file=sys.stderr)
        combined["overall_mean"] = 0.0
        combined["gen_dimension_pass_rates"] = {}
        combined["phpcs_pass_rate"] = 0.0
        combined["security_pass_rate"] = 0.0

    if judge_path.exists():
        with judge_path.open() as f:
            judge_results = json.load(f)
        combined["overall_spearman"] = judge_results.get("overall_spearman", 0.0)
        combined["judge_dimension_correlations"] = judge_results.get(
            "dimension_correlations", {}
        )
        # Legacy key
        combined["spearman_corr"] = judge_results.get("spearman_corr", 0.0)
    else:
        print(f"WARNING: {judge_path} not found -- treating as 0.0", file=sys.stderr)
        combined["overall_spearman"] = 0.0
        combined["judge_dimension_correlations"] = {}
        combined["spearman_corr"] = 0.0

    return check_gates(combined, thresholds)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quality gate: check eval thresholds and exit 0/1."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="output",
        help="Directory containing eval JSON results files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to train_config.yaml (default: config/train_config.yaml)",
    )
    args = parser.parse_args()

    passed, gate_rows = run_gate(
        results_dir=args.results_dir,
        config_path=args.config,
    )

    print_summary_table(gate_rows, passed)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
