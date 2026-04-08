"""Retroactive critique-then-fix quality auditor.

Audits a CtF pilot or bulk JSON file in place. For each example:
  - re-runs php_lint_check on corrected_code
  - re-runs check_critique_fix_alignment on (critique, defective_code, corrected_code)
  - flags whether corrected_code differs from defective_code

Emits a JSON report with aggregate pass rates and per-example detail.

Usage:
    python scripts/audit_critique_fix_quality.py <input.json> <output_audit.json>
"""
import json, sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_critique_then_fix import (
    php_lint_check, check_critique_fix_alignment, REQUIRED_DIMENSIONS, SEVERITY_LEVELS
)

def audit_file(input_path: Path, output_path: Path) -> dict:
    examples = json.loads(input_path.read_text())
    per_example = []
    lint_valid = 0
    lint_skipped = 0
    align_total_ratio = 0.0
    align_count = 0
    differs_count = 0
    all_dims_present_count = 0
    severities_seen = set()
    for i, ex in enumerate(examples):
        critique = ex.get("critique", {})
        corrected = ex.get("corrected_code", "") or ""
        defective = ex.get("defective_code", "") or ""

        lint = php_lint_check(corrected)
        is_skipped = "php not available" in (lint.get("errors") or "")
        if is_skipped:
            lint_skipped += 1
        elif lint.get("valid"):
            lint_valid += 1

        align = check_critique_fix_alignment(critique, defective, corrected)
        align_total_ratio += align.get("alignment_ratio", 1.0)
        align_count += 1

        differs = bool(corrected.strip()) and corrected.strip() != defective.strip()
        if differs:
            differs_count += 1

        dims = critique.get("dimensions", {})
        if all(d in dims for d in REQUIRED_DIMENSIONS):
            all_dims_present_count += 1
        for d in dims.values():
            sev = d.get("severity")
            if sev:
                severities_seen.add(sev)

        per_example.append({
            "index": i,
            "function_name": ex.get("function_name", ""),
            "source_file": ex.get("source_file", ""),
            "php_lint_valid": lint.get("valid"),
            "php_lint_skipped": is_skipped,
            "alignment_ratio": align.get("alignment_ratio"),
            "critical_high_issues": align.get("critical_high_issues", 0),
            "addressed_issues": align.get("addressed_issues", 0),
            "corrected_code_differs": differs,
            "all_dimensions_present": all(d in dims for d in REQUIRED_DIMENSIONS),
        })

    n = len(examples) or 1
    n_lintable = (n - lint_skipped) or 1
    report = {
        "input_file": str(input_path),
        "total_examples": len(examples),
        "php_lint_pass_rate": lint_valid / n_lintable,
        "php_lint_skipped": lint_skipped,
        "mean_alignment_ratio": align_total_ratio / align_count if align_count else 1.0,
        "corrected_code_differs_rate": differs_count / n,
        "all_dimensions_present_rate": all_dims_present_count / n,
        "severities_seen": sorted(severities_seen),
        "missing_severity_levels": sorted(set(SEVERITY_LEVELS) - severities_seen),
        "per_example": per_example,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--min-lint-pass-rate", type=float, default=0.80)
    ap.add_argument("--min-alignment-ratio", type=float, default=0.50)
    ap.add_argument("--min-differs-rate", type=float, default=0.95)
    args = ap.parse_args()
    report = audit_file(args.input, args.output)
    print(f"Audited {report['total_examples']} examples")
    print(f"  php_lint_pass_rate: {report['php_lint_pass_rate']:.3f} (threshold: {args.min_lint_pass_rate}, skipped: {report['php_lint_skipped']})")
    print(f"  mean_alignment_ratio: {report['mean_alignment_ratio']:.3f} (threshold: {args.min_alignment_ratio})")
    print(f"  corrected_code_differs_rate: {report['corrected_code_differs_rate']:.3f} (threshold: {args.min_differs_rate})")
    print(f"  all_dimensions_present_rate: {report['all_dimensions_present_rate']:.3f}")
    print(f"  severities_seen: {report['severities_seen']}")
    failures = []
    if report['php_lint_pass_rate'] < args.min_lint_pass_rate and report['php_lint_skipped'] < report['total_examples']:
        failures.append(f"lint_pass_rate {report['php_lint_pass_rate']:.3f} < {args.min_lint_pass_rate}")
    if report['mean_alignment_ratio'] < args.min_alignment_ratio:
        failures.append(f"alignment {report['mean_alignment_ratio']:.3f} < {args.min_alignment_ratio}")
    if report['corrected_code_differs_rate'] < args.min_differs_rate:
        failures.append(f"differs_rate {report['corrected_code_differs_rate']:.3f} < {args.min_differs_rate}")
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("PASS")
    return 0

if __name__ == "__main__":
    main()
