#!/usr/bin/env python3
"""Phase 2, Step 1: Analyze coverage gaps in Phase 1 output.

Compares tag counts from passed functions against minimum_coverage
targets in config/taxonomy.yaml. Outputs a gap report that drives
synthetic generation.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.yaml"
GAP_REPORT_PATH = PROJECT_ROOT / "data" / "phase2_synthetic" / "gap_report.json"


def main():
    # Load taxonomy with minimum coverage targets.
    with open(TAXONOMY_PATH) as f:
        taxonomy = yaml.safe_load(f)

    minimums = taxonomy.get("minimum_coverage", {})

    # Count tags across all passed functions.
    tag_counts = Counter()
    total_functions = 0

    passed_files = list(PASSED_DIR.glob("*.json"))
    if not passed_files:
        print("No passed functions found. Run Phase 1 first.")
        sys.exit(1)

    for passed_file in passed_files:
        with open(passed_file) as f:
            functions = json.load(f)
        total_functions += len(functions)
        for func in functions:
            for tag in func.get("training_tags", []):
                tag_counts[tag] += 1

    # Calculate gaps.
    gaps = {}
    for tag, minimum in minimums.items():
        have = tag_counts.get(tag, 0)
        if have < minimum:
            gaps[tag] = {
                "have": have,
                "need": minimum,
                "deficit": minimum - have,
                "fill_pct": have / minimum * 100 if minimum > 0 else 100,
            }

    # Report.
    print(f"Phase 1 Coverage Report")
    print(f"{'='*60}")
    print(f"Total passed functions: {total_functions}")
    print(f"Unique tags present: {len(tag_counts)}")
    print()

    print(f"{'Tag':<40} {'Have':>6} {'Need':>6} {'Gap':>6} {'%':>6}")
    print(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for tag in sorted(minimums.keys()):
        have = tag_counts.get(tag, 0)
        need = minimums[tag]
        deficit = max(0, need - have)
        pct = have / need * 100 if need > 0 else 100
        marker = " <-- GAP" if deficit > 0 else ""
        print(f"{tag:<40} {have:>6} {need:>6} {deficit:>6} {pct:>5.0f}%{marker}")

    # Tags found but not in taxonomy (potential new categories).
    uncategorized = set(tag_counts.keys()) - set(minimums.keys())
    if uncategorized:
        print(f"\nTags found but not in taxonomy ({len(uncategorized)}):")
        for tag in sorted(uncategorized):
            print(f"  {tag}: {tag_counts[tag]}")

    # Save gap report for Phase 2 generation.
    report = {
        "total_passed_functions": total_functions,
        "tag_counts": dict(tag_counts),
        "gaps": gaps,
        "total_synthetic_needed": sum(g["deficit"] for g in gaps.values()),
    }

    GAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GAP_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gaps found: {len(gaps)} tags below minimum")
    print(f"Total synthetic examples needed: {report['total_synthetic_needed']}")
    print(f"Gap report saved to: {GAP_REPORT_PATH}")
    print(f"\nRun phase2_generate.py next.")


if __name__ == "__main__":
    main()
