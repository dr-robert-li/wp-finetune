"""Quality gate — exits non-zero if any evaluation threshold fails.

Reads thresholds from config/train_config.yaml (eval section) and
result JSON files from output/. Prints pass/fail summary for each gate.

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
}


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------

def load_thresholds(config_path: Optional[str] = None) -> dict:
    """Load evaluation thresholds from config/train_config.yaml eval section.

    Args:
        config_path: Path to train_config.yaml. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        dict with phpcs_pass_target, spearman_target, security_pass_target.
        Falls back to _FALLBACK_THRESHOLDS if file missing or section absent.
    """
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return dict(_FALLBACK_THRESHOLDS)

    with path.open() as f:
        config = yaml.safe_load(f) or {}

    eval_section = config.get("eval", {})

    return {
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
        results: dict containing phpcs_pass_rate, spearman_corr, security_pass_rate.
        thresholds: dict from load_thresholds() with target values.

    Returns:
        (passed: bool, failures: list[str]) where failures lists descriptions
        of any thresholds not met. passed is True iff failures is empty.
    """
    failures = []

    phpcs_pass_rate = results.get("phpcs_pass_rate", 0.0)
    phpcs_target = thresholds["phpcs_pass_target"]
    if phpcs_pass_rate < phpcs_target:
        failures.append(
            f"PHPCS pass rate {phpcs_pass_rate:.3f} < target {phpcs_target:.3f}"
        )

    spearman_corr = results.get("spearman_corr", 0.0)
    spearman_target = thresholds["spearman_target"]
    if spearman_corr < spearman_target:
        failures.append(
            f"Spearman correlation {spearman_corr:.3f} < target {spearman_target:.3f}"
        )

    security_pass_rate = results.get("security_pass_rate", 0.0)
    security_target = thresholds["security_pass_target"]
    if security_pass_rate < security_target:
        failures.append(
            f"Security pass rate {security_pass_rate:.3f} < target {security_target:.3f}"
        )

    return (len(failures) == 0, failures)


# ---------------------------------------------------------------------------
# Main evaluation gate runner
# ---------------------------------------------------------------------------

def run_gate(
    results_dir: str = "output",
    config_path: Optional[str] = None,
) -> tuple:
    """Load results and thresholds, run all gates, return (passed, failures).

    Args:
        results_dir: Directory containing eval_gen_results.json and eval_judge_results.json.
        config_path: Path to train_config.yaml. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        (passed: bool, failures: list[str])
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
        combined["phpcs_pass_rate"] = gen_results.get("phpcs_pass_rate", 0.0)
        combined["security_pass_rate"] = gen_results.get("security_pass_rate", 0.0)
    else:
        print(f"WARNING: {gen_path} not found — treating as 0.0", file=sys.stderr)
        combined["phpcs_pass_rate"] = 0.0
        combined["security_pass_rate"] = 0.0

    if judge_path.exists():
        with judge_path.open() as f:
            judge_results = json.load(f)
        combined["spearman_corr"] = judge_results.get("spearman_corr", 0.0)
    else:
        print(f"WARNING: {judge_path} not found — treating as 0.0", file=sys.stderr)
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

    thresholds = load_thresholds(args.config)
    passed, failures = run_gate(
        results_dir=args.results_dir,
        config_path=args.config,
    )

    # Print summary
    print("=" * 50)
    print("EVALUATION QUALITY GATE")
    print("=" * 50)
    print(f"  phpcs_pass_target:     {thresholds['phpcs_pass_target']:.2%}")
    print(f"  spearman_target:       {thresholds['spearman_target']:.2f}")
    print(f"  security_pass_target:  {thresholds['security_pass_target']:.2%}")
    print()

    if passed:
        print("GATE: PASS - All evaluation thresholds met.")
        sys.exit(0)
    else:
        print("GATE: FAIL - The following thresholds were NOT met:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
