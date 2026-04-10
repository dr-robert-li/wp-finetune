"""Merge reasoning batches with hardened acceptance gates, dedup, and contamination manifest.

Reads Claude Code agent batch outputs from data/phase4_reasoning/{stream}/batches/,
applies pass/fail quality gates per the bulk_acceptance_gate threshold table,
emits the merged bulk JSON, a bulk_acceptance_report.json (concern #3),
and an input function manifest for downstream eval contamination guard (concern #4).

Usage:
    python -m scripts.merge_reasoning_batches cot
    python -m scripts.merge_reasoning_batches ctf
    python -m scripts.merge_reasoning_batches both
"""
import json
import re
import sys
import hashlib
import argparse
import datetime
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Bogus-template blacklist gate (REVIEWS concern #2 followup, 2026-04-08)
# ---------------------------------------------------------------------------
# These patterns were inserted by the _generate_ctf_batches.py generator into
# nearly every corrected_code block as generic fallback boilerplate. They
# reference an undefined $args variable, pass `php -l` (syntax-only) but throw
# "Undefined variable $args" at runtime. Any example whose corrected_code matches
# one of these patterns is rejected outright regardless of other gate metrics.
CTF_BOGUS_PATTERNS = [
    re.compile(r"Defensive:\s*Added input sanitization per WPCS review"),
    re.compile(r"Input validation added per code quality review"),
    re.compile(r"if\s*\(\s*empty\(\s*\$args\s*\)\s*&&\s*func_num_args\("),
    re.compile(r"\$sanitized_input\s*=\s*array_map\(\s*['\"]sanitize_text_field['\"]"),
]


def has_bogus_template(corrected_code: str) -> bool:
    """Return True if corrected_code contains a templated bogus fix snippet."""
    return any(p.search(corrected_code) for p in CTF_BOGUS_PATTERNS)


# Acceptance thresholds (from review feedback HIGH concern #3)
COT_THRESHOLDS = {
    "parse_failure_rate_max": 0.02,
    "citation_validity_rate_min": 0.70,
    "duplicate_rate_max": 0.05,
    "accepted_count_min": 100,
    "max_na_dimensions": 2,
    "na_justification_min_chars": 20,
}
CTF_THRESHOLDS = {
    "parse_failure_rate_max": 0.02,
    "php_lint_pass_rate_min": 0.80,
    "mean_alignment_ratio_min": 0.50,
    "corrected_code_differs_rate_min": 0.95,
    "duplicate_rate_max": 0.05,
    "accepted_count_min": 50,
}


def normalize_code(code: str) -> str:
    """Strip comments and collapse whitespace for hashing."""
    if not code:
        return ""
    no_comments = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    no_comments = re.sub(r"//.*?$", "", no_comments, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", no_comments).strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def phase1_function_id(source_file: str, function_name: str) -> str:
    return hashlib.sha256(f"{source_file}:{function_name}".encode("utf-8")).hexdigest()


def is_dimension_na(dim_value: dict, min_just: int) -> bool:
    """A dimension is validly marked N/A if score is null AND the analysis
    provides at least ``min_just`` chars of justification.

    Previous version required the literal phrase 'not applicable' in the
    analysis, which was too brittle: agents wrote valid justifications like
    'No user-facing strings in this function' which should count as N/A but
    didn't. That bug let entries with 3+ null-score dims slip through the
    max_na_dimensions check (caught by gemini CoT v3 audit 2026-04-10).

    Valid N/A now requires only: score is None AND analysis is a real
    justification >= min_just characters."""
    if dim_value.get("score") is not None:
        return False
    analysis = (dim_value.get("analysis") or "").strip()
    return len(analysis) >= min_just


def cot_passes_full_gate(ex: dict, source_code: str) -> "tuple[bool, str]":
    """Returns (passed, reason). Reason is empty if passed."""
    from scripts.generate_deep_judge_cot import (
        REQUIRED_DIMENSIONS,
        verify_citation_accuracy,
        CITATION_HALLUCINATION_THRESHOLD,
    )

    result = ex.get("reasoning", {})
    if result.get("verdict") not in ("PASS", "FAIL"):
        return False, "verdict"
    if not isinstance(result.get("overall_score"), int) or not (0 <= result["overall_score"] <= 100):
        return False, "overall_score"
    da = result.get("dimension_analysis", {})
    missing = [d for d in REQUIRED_DIMENSIONS if d not in da]
    if missing:
        return False, f"missing_dims:{missing}"
    # N/A policy (concern #6): allow up to max_na_dimensions marked not_applicable with justification
    na_count = sum(
        1 for d in REQUIRED_DIMENSIONS
        if is_dimension_na(da[d], COT_THRESHOLDS["na_justification_min_chars"])
    )
    if na_count > COT_THRESHOLDS["max_na_dimensions"]:
        return False, f"too_many_na:{na_count}"
    # Score range on non-N/A dimensions
    for d in REQUIRED_DIMENSIONS:
        v = da[d]
        if v.get("score") is None:
            continue
        if not isinstance(v["score"], int) or not (1 <= v["score"] <= 10):
            return False, f"bad_score:{d}"
        if not v.get("analysis"):
            return False, f"empty_analysis:{d}"
    # Citation accuracy
    ca = verify_citation_accuracy(result, source_code)
    if ca["hallucination_ratio"] >= CITATION_HALLUCINATION_THRESHOLD:
        return False, f"citation_hallucination:{ca['hallucination_ratio']:.2f}"
    return True, ""


def wrap_class_method_for_lint(code: str, fn_name: str) -> str:
    """Wrap a PHP class method in a synthetic class scope so `php -l` accepts it.

    `php -l` rejects standalone `public function foo()` as a top-level parse
    error. Most CtF corrected_code blocks ARE class methods (the source
    functions came from WordPress plugin classes). This helper detects those
    and wraps them in `class Shim { ... }`. If the method uses `parent::`,
    we also add a no-op parent class to satisfy the scope.

    Strips any leading `<?php` tag from the input since php_lint_check adds
    its own.
    """
    stripped = re.sub(r"^\s*<\?php\s*", "", code)
    stripped = re.sub(r"\?>\s*$", "", stripped)

    needs_class_scope = bool(
        re.search(r"^\s*(public|private|protected|static|final|abstract)\s+function", stripped, re.MULTILINE)
        or ("::" in fn_name and "::" not in (fn_name.split("::", 1)[0] or ""))
    )
    if not needs_class_scope:
        return stripped

    cls_name = fn_name.split("::", 1)[0] if "::" in fn_name else "WpFtMergeShim"
    # Avoid reserved-keyword or invalid class names
    if cls_name.lower() in {"false", "true", "null", "self", "static", "parent", "class"} or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", cls_name):
        cls_name = "WpFtMergeShim"

    if "parent::" in stripped:
        return (
            f"class WpFtMergeShimParent {{\n"
            f"    public function __construct() {{}}\n"
            f"}}\n"
            f"class {cls_name} extends WpFtMergeShimParent {{\n"
            f"{stripped}\n"
            f"}}"
        )
    return f"class {cls_name} {{\n{stripped}\n}}"


def ctf_passes_full_gate(ex: dict) -> "tuple[bool, str, dict, dict]":
    """Returns (passed, reason, lint_meta, alignment_meta)."""
    from scripts.generate_critique_then_fix import (
        REQUIRED_DIMENSIONS,
        SEVERITY_LEVELS,
        php_lint_check,
        check_critique_fix_alignment,
    )

    critique = ex.get("critique", {})
    corrected = ex.get("corrected_code", "") or ""
    defective = ex.get("defective_code", "") or ""
    fn_name = ex.get("function_name", "") or ""
    if not critique.get("summary"):
        return False, "no_summary", {}, {}
    dims = critique.get("dimensions", {})
    missing = [d for d in REQUIRED_DIMENSIONS if d not in dims]
    if missing:
        return False, f"missing_dims:{missing}", {}, {}
    for d in REQUIRED_DIMENSIONS:
        v = dims[d]
        if v.get("severity") not in SEVERITY_LEVELS:
            return False, f"bad_severity:{d}", {}, {}
        if not v.get("issue") or not v.get("fix"):
            return False, f"empty_issue_or_fix:{d}", {}, {}
    if len(corrected.strip()) <= 20:
        return False, "corrected_too_short", {}, {}
    if normalize_code(corrected) == normalize_code(defective):
        return False, "corrected_identical_to_defective", {}, {}
    # Wrap class methods in a synthetic class scope so `php -l` accepts
    # `public/private/protected/static function ...` — without this wrap
    # the lint would false-fail ~35% of class-method corrected_code blocks.
    lint_code = wrap_class_method_for_lint(corrected, fn_name)
    lint = php_lint_check(lint_code)
    is_skipped = "php not available" in (lint.get("errors") or "")
    if not is_skipped and not lint.get("valid"):
        return False, "php_lint_invalid", lint, {}
    align = check_critique_fix_alignment(critique, defective, corrected)
    if align.get("critical_high_issues", 0) > 0 and align.get("alignment_ratio", 1.0) < 0.30:
        return False, f"alignment:{align['alignment_ratio']:.2f}", lint, align
    return True, "", lint, align


def merge_stream(stream: str) -> dict:
    """stream in {'cot', 'ctf'}. Returns the bulk_acceptance_report dict."""
    is_cot = stream == "cot"
    sub = "deep_judge_cot" if is_cot else "critique_then_fix"
    batches_dir = PROJECT_ROOT / "data" / "phase4_reasoning" / sub / "batches"
    output_path = PROJECT_ROOT / "data" / "phase4_reasoning" / sub / f"{sub}_bulk.json"
    report_path = PROJECT_ROOT / "data" / "phase4_reasoning" / sub / "bulk_acceptance_report.json"
    manifest_path = PROJECT_ROOT / "data" / "phase4_reasoning" / "manifests" / f"{stream}_input_function_ids.json"

    accepted = []
    rejection_reasons: Counter = Counter()
    parse_failures = 0
    dup_prompt = 0
    dup_output = 0
    seen_prompt_hashes: set = set()
    seen_output_hashes: set = set()
    align_ratios = []
    lint_valid_count = 0
    lint_skipped_count = 0
    differs_count = 0
    bogus_template_rejection_count = 0

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    if not batch_files:
        print(f"WARNING: no batch files in {batches_dir}", file=sys.stderr)

    for batch_file in batch_files:
        try:
            batch = json.loads(batch_file.read_text())
        except json.JSONDecodeError:
            parse_failures += 1
            continue
        if not isinstance(batch, list):
            parse_failures += 1
            continue
        for ex in batch:
            # Prompt-side dedup (concern #7)
            source_code = ex.get("code", "") if is_cot else ex.get("defective_code", "")
            prompt_hash = hash_text(normalize_code(source_code))
            if prompt_hash in seen_prompt_hashes:
                dup_prompt += 1
                rejection_reasons["dup_prompt"] += 1
                continue

            # Output-side dedup (concern #7)
            if is_cot:
                output_text = " ".join(
                    (v.get("analysis") or "")
                    for v in ex.get("reasoning", {}).get("dimension_analysis", {}).values()
                ).lower()
            else:
                output_text = " ".join(
                    (v.get("fix") or "") + " " + (v.get("issue") or "")
                    for v in ex.get("critique", {}).get("dimensions", {}).values()
                ).lower()
            output_hash = hash_text(re.sub(r"\s+", " ", output_text).strip())
            if output_hash in seen_output_hashes:
                dup_output += 1
                rejection_reasons["dup_output"] += 1
                continue

            # Quality gate
            if is_cot:
                ok, reason = cot_passes_full_gate(ex, source_code)
                if not ok:
                    rejection_reasons[reason] += 1
                    continue
                # Stamp citation accuracy metadata
                from scripts.generate_deep_judge_cot import verify_citation_accuracy
                ex["citation_accuracy"] = verify_citation_accuracy(ex.get("reasoning", {}), source_code)
            else:
                # Bogus-template blacklist gate (REVIEWS concern #2 followup)
                corrected = ex.get("corrected_code", "") or ""
                if has_bogus_template(corrected):
                    bogus_template_rejection_count += 1
                    rejection_reasons["bogus_template"] += 1
                    continue

                ok, reason, lint, align = ctf_passes_full_gate(ex)
                if not ok:
                    rejection_reasons[reason] += 1
                    continue
                ex["php_lint"] = lint
                ex["critique_fix_alignment"] = align
                if "php not available" in (lint.get("errors") or ""):
                    lint_skipped_count += 1
                elif lint.get("valid"):
                    lint_valid_count += 1
                align_ratios.append(align.get("alignment_ratio", 1.0))
                if normalize_code(ex.get("corrected_code", "")) != normalize_code(ex.get("defective_code", "")):
                    differs_count += 1

            seen_prompt_hashes.add(prompt_hash)
            seen_output_hashes.add(output_hash)
            accepted.append(ex)

    # Aggregate metrics
    n_accepted = len(accepted)
    n_total_attempted = n_accepted + sum(rejection_reasons.values()) + parse_failures
    parse_failure_rate = parse_failures / n_total_attempted if n_total_attempted else 0.0
    duplicate_rate = (dup_prompt + dup_output) / n_total_attempted if n_total_attempted else 0.0

    report: dict = {
        "stream": stream,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "accepted_count": n_accepted,
        "total_attempted": n_total_attempted,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failure_rate,
        "duplicate_rate": duplicate_rate,
        "dup_prompt": dup_prompt,
        "dup_output": dup_output,
        "rejection_reasons": dict(rejection_reasons),
    }

    if is_cot:
        from scripts.audit_citation_accuracy import audit_file

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(accepted, indent=2))
        # Run citation audit on accepted set
        audit_out = output_path.parent / "bulk_citation_audit.json"
        audit_report = audit_file(output_path, audit_out)
        report["citation_validity_rate"] = audit_report["citation_validity_rate"]
        report["aggregate_hallucination_ratio"] = audit_report["aggregate_hallucination_ratio"]
        thresholds = COT_THRESHOLDS
        report["thresholds"] = thresholds
        report["passed_acceptance_gate"] = (
            n_accepted >= thresholds["accepted_count_min"]
            and parse_failure_rate < thresholds["parse_failure_rate_max"]
            and report["citation_validity_rate"] >= thresholds["citation_validity_rate_min"]
            and duplicate_rate <= thresholds["duplicate_rate_max"]
        )
    else:
        n_lintable = (n_accepted - lint_skipped_count) or 1
        report["php_lint_pass_rate"] = lint_valid_count / n_lintable
        report["mean_alignment_ratio"] = sum(align_ratios) / len(align_ratios) if align_ratios else 1.0
        report["corrected_code_differs_rate"] = differs_count / n_accepted if n_accepted else 0.0
        report["bogus_template_rejection_count"] = bogus_template_rejection_count
        thresholds = CTF_THRESHOLDS
        report["thresholds"] = thresholds
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(accepted, indent=2))
        report["passed_acceptance_gate"] = (
            n_accepted >= thresholds["accepted_count_min"]
            and parse_failure_rate < thresholds["parse_failure_rate_max"]
            and report["php_lint_pass_rate"] >= thresholds["php_lint_pass_rate_min"]
            and report["mean_alignment_ratio"] >= thresholds["mean_alignment_ratio_min"]
            and report["corrected_code_differs_rate"] >= thresholds["corrected_code_differs_rate_min"]
            and duplicate_rate <= thresholds["duplicate_rate_max"]
        )

    # Contamination manifest (concern #4)
    function_ids = []
    for ex in accepted:
        sf = ex.get("source_file", "")
        fn = ex.get("function_name", "")
        function_ids.append({
            "source_file": sf,
            "function_name": fn,
            "source_dir": ex.get("source_dir", ""),
            "phase1_id": phase1_function_id(sf, fn),
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "phase": "04.1",
        "stream": stream,
        "generated_at": report["generated_at"],
        "function_ids": function_ids,
    }, indent=2))

    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== {stream.upper()} BULK MERGE ===")
    print(f"Accepted: {n_accepted} / {n_total_attempted}")
    print(f"Parse failures: {parse_failures} ({parse_failure_rate:.3%})")
    print(f"Duplicates: prompt={dup_prompt}, output={dup_output}, rate={duplicate_rate:.3%}")
    print(f"Rejection reasons: {dict(rejection_reasons)}")
    if is_cot:
        print(f"citation_validity_rate: {report['citation_validity_rate']:.3f}")
    else:
        print(f"php_lint_pass_rate: {report['php_lint_pass_rate']:.3f}")
        print(f"mean_alignment_ratio: {report['mean_alignment_ratio']:.3f}")
        print(f"corrected_code_differs_rate: {report['corrected_code_differs_rate']:.3f}")
        print(f"bogus_template_rejection_count: {report['bogus_template_rejection_count']}")
    print(f"PASSED ACCEPTANCE GATE: {report['passed_acceptance_gate']}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Merge reasoning batches with hardened acceptance gates."
    )
    parser.add_argument("stream", choices=["cot", "ctf", "both"])
    args = parser.parse_args()
    reports = []
    if args.stream in ("cot", "both"):
        reports.append(merge_stream("cot"))
    if args.stream in ("ctf", "both"):
        reports.append(merge_stream("ctf"))
    if not all(r["passed_acceptance_gate"] for r in reports):
        print("\nFAIL: one or more streams did not pass the bulk acceptance gate", file=sys.stderr)
        sys.exit(1)
    print("\nPASS: all streams passed the bulk acceptance gate")


# Compatibility aliases (Task 0 acceptance criteria)
def merge_cot_batches():
    return merge_stream("cot")


def merge_ctf_batches():
    return merge_stream("ctf")


if __name__ == "__main__":
    main()
