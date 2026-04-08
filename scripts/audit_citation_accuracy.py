"""Retroactive citation-accuracy auditor for deep judge CoT artifacts.

Reads a pilot or bulk CoT JSON array, runs verify_citation_accuracy()
against each example's source code, and emits a JSON audit report.

Usage:
    python scripts/audit_citation_accuracy.py <input.json> <output_audit.json>
    python scripts/audit_citation_accuracy.py <input.json> <output_audit.json> --min-validity-rate 0.70
"""
import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_deep_judge_cot import verify_citation_accuracy, CITATION_HALLUCINATION_THRESHOLD


def audit_file(input_path: Path, output_path: Path) -> dict:
    """Audit citation accuracy for all examples in a CoT JSON file.

    Args:
        input_path: Path to a JSON array of generated CoT examples.
        output_path: Path where the audit report JSON will be written.

    Returns:
        The audit report dict (same structure written to output_path).
    """
    examples = json.loads(input_path.read_text())
    per_example = []
    grounded = 0
    total_citations = 0
    total_hallucinated = 0
    examples_with_zero_citations = 0

    for i, ex in enumerate(examples):
        result = ex.get("reasoning", {})
        source = ex.get("code", "") or ex.get("body", "")
        ca = verify_citation_accuracy(result, source)
        per_example.append({
            "index": i,
            "function_name": ex.get("function_name", ""),
            "source_file": ex.get("source_file", ""),
            "total_citations": ca["total_citations"],
            "grounded_citations": ca["grounded_citations"],
            "hallucinated_citations": ca["hallucinated_citations"],
            "hallucination_ratio": ca["hallucination_ratio"],
            "passes_threshold": ca["hallucination_ratio"] < CITATION_HALLUCINATION_THRESHOLD,
        })
        total_citations += ca["total_citations"]
        total_hallucinated += len(ca["hallucinated_citations"])
        if ca["total_citations"] == 0:
            examples_with_zero_citations += 1
        if ca["hallucination_ratio"] < CITATION_HALLUCINATION_THRESHOLD:
            grounded += 1

    citation_validity_rate = grounded / len(examples) if examples else 0.0
    aggregate_hallucination_ratio = total_hallucinated / total_citations if total_citations else 0.0

    report = {
        "input_file": str(input_path),
        "total_examples": len(examples),
        "citation_validity_rate": citation_validity_rate,
        "aggregate_hallucination_ratio": aggregate_hallucination_ratio,
        "total_citations": total_citations,
        "total_hallucinated": total_hallucinated,
        "examples_with_zero_citations": examples_with_zero_citations,
        "threshold": CITATION_HALLUCINATION_THRESHOLD,
        "per_example": per_example,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(
        description="Retroactive citation-accuracy auditor for deep judge CoT artifacts."
    )
    ap.add_argument("input", type=Path, help="Input CoT JSON array file")
    ap.add_argument("output", type=Path, help="Output audit report JSON file")
    ap.add_argument(
        "--min-validity-rate",
        type=float,
        default=0.70,
        help="Minimum acceptable citation_validity_rate (fraction of examples with hallucination_ratio < 0.5)",
    )
    args = ap.parse_args()

    report = audit_file(args.input, args.output)
    print(f"Audited {report['total_examples']} examples")
    print(f"  citation_validity_rate: {report['citation_validity_rate']:.3f} (threshold: {args.min_validity_rate})")
    print(f"  aggregate_hallucination_ratio: {report['aggregate_hallucination_ratio']:.3f}")
    print(f"  examples_with_zero_citations: {report['examples_with_zero_citations']}")

    if report["citation_validity_rate"] < args.min_validity_rate:
        print(
            f"FAIL: citation_validity_rate {report['citation_validity_rate']:.3f} < {args.min_validity_rate}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("PASS")
    return 0


if __name__ == "__main__":
    main()
