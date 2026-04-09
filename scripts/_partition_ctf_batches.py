#!/usr/bin/env python3
"""Partition Phase 1 failed functions into CtF input batches for Claude Code agents.

This script does ONLY data partitioning — NO generation, NO heuristics, NO LLM calls.
It loads defect-confirmed failed functions from Phase 1, filters out corrupted entries,
and writes _input_batch_NN.json files that Claude Code agents will process.

Upstream of this: scripts/phase1_extract.py produces data/phase1_extraction/output/failed/
Downstream of this: Claude Code agents spawned from the orchestrator read each input batch
and produce a matching batch_NNN.json with real critique-then-fix examples.

Usage:
    python scripts/_partition_ctf_batches.py partition [num_batches] [batch_size]

Replaces the old scripts/_generate_ctf_batches.py which was a heuristic string-substitution
pseudo-generator that masqueraded as a Claude Code agent (commit history reveals two failed
generations, 1/20 and 4/20 cross-AI audit pass rates respectively, before the root cause
was identified).
"""
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
BATCHES_DIR = PROJECT_ROOT / "data" / "phase4_reasoning" / "critique_then_fix" / "batches"

# Known PHP builtins and WP core functions that indicate a corrupted class_context
# from the phase1_extract.py ::class bug (fixed 2026-04-09 but existing corpus still
# contains corrupted entries from prior extraction runs).
CORRUPTED_CLASS_NAMES = {
    # PHP builtins
    "array_flip", "array_merge", "array_map", "array_filter", "explode", "implode",
    "strlen", "strpos", "str_replace", "preg_match", "preg_replace", "sprintf",
    "var_dump", "print_r", "error_log", "trim", "count", "is_array", "in_array",
    "array_keys", "array_values", "func_get_args", "func_num_args", "substr",
    # WP core hook/filter functions commonly seen in ::class corruption
    "add_shortcode", "_delete_post", "get_post", "wp_die", "esc_html", "__", "_e",
    "add_action", "add_filter", "do_action", "apply_filters", "register_post_type",
    "wp_verify_nonce", "current_user_can", "sanitize_text_field",
}

# Substantive critical failure strings — functions flagged with ONLY these
# deserve to be in the CtF pool (real fix work, not style issues).
SUBSTANTIVE_FAILURE_KEYWORDS = (
    "security_auto_fail",
    "Security auto-fail",
    "Uses extract()",
    "Uses eval()",
    "Use of extract() or eval()",
    "superglobal",
    "nonce",
    "SQL",
    "sql",
    "Unprepared",
    "Unsanitized",
    "Debug statements",
    "N+1",
    "Form inputs without associated labels",
    "Missing nonce verification",
    "XSS",
    "injection",
)

# Non-substantive failures to EXCLUDE entirely (style-only or should-be-dropped)
EXCLUDE_FAILURE_KEYWORDS = (
    "test/bin/docs code",
    "vendored third-party",
    "judge_error",
    "deprecated function",
    "legacy compatibility",
)


def has_substantive_failure(failures: list) -> bool:
    """True if at least one failure string matches a substantive defect keyword."""
    return any(
        any(kw.lower() in f.lower() for kw in SUBSTANTIVE_FAILURE_KEYWORDS)
        for f in failures
    )


def has_excluded_failure(failures: list) -> bool:
    """True if ANY failure string matches an excluded category (test/vendored/etc)."""
    return any(
        any(kw.lower() in f.lower() for kw in EXCLUDE_FAILURE_KEYWORDS)
        for f in failures
    )


# High-stakes dimensions for the low_score fallback path
HIGH_STAKES_DIMENSIONS = {"security", "sql_safety", "wp_api_usage", "code_quality"}


def is_corrupted_function_name(fn_name: str) -> bool:
    """Return True if function_name has corrupted class_context from the ::class bug."""
    if "::" not in fn_name:
        return False
    cls = fn_name.split("::", 1)[0]
    return cls in CORRUPTED_CLASS_NAMES


def load_failed_functions_filtered() -> list:
    """Load Phase 1 failed functions with concrete defects only.

    Source filter: Only include functions where Phase 1 assessment shows concrete
    defects — critical_failures is non-empty OR any dimension score is <= 5.

    Additional filter: Skip entries with corrupted class_context from the
    phase1_extract.py ::class bug (e.g., do_action::method, array_flip::method).
    Re-running phase1_extract.py would fix the corpus, but in the interim this
    filter keeps bad entries out of CtF generation.

    Returns only functions with body length >= 50 chars.
    """
    functions = []
    skipped_no_defect = 0
    skipped_corrupted = 0
    skipped_style_only = 0
    skipped_excluded = 0

    if not FAILED_DIR.exists():
        print(f"ERROR: {FAILED_DIR} not found", file=sys.stderr)
        return functions

    for f in sorted(FAILED_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                entries = json.load(fh)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                fn_name = entry.get("function_name", "unknown")

                # Skip corrupted entries (phase1_extract.py ::class bug aftermath)
                if is_corrupted_function_name(fn_name):
                    skipped_corrupted += 1
                    continue

                assessment = entry.get("assessment", {})
                critical_failures = assessment.get("critical_failures", [])
                scores = assessment.get("scores", {})

                # Drop test/vendored/deprecated/judge_error entries entirely
                if has_excluded_failure(critical_failures):
                    skipped_excluded += 1
                    continue

                # Require SUBSTANTIVE defect: either a substantive critical_failure
                # OR a high-stakes dimension scoring <= 5 (security, sql_safety,
                # wp_api_usage, code_quality). Style-only issues (PHPDoc, WPCS
                # threshold) without any high-stakes problem are skipped — the
                # model has nothing meaningful to fix.
                has_sub_failure = has_substantive_failure(critical_failures)
                high_stakes_low = [
                    dim for dim, v in scores.items()
                    if isinstance(v, (int, float)) and v <= 5
                    and dim in HIGH_STAKES_DIMENSIONS
                ]

                if not has_sub_failure and not high_stakes_low:
                    if not critical_failures and not any(
                        v <= 5 for v in scores.values() if isinstance(v, (int, float))
                    ):
                        skipped_no_defect += 1
                    else:
                        skipped_style_only += 1
                    continue

                body = entry.get("body", "")
                docblock = entry.get("docblock", "")
                code = f"{docblock}\n{body}".strip() if docblock else body
                if len(code) < 50:
                    continue

                functions.append({
                    "code": code,
                    "source_file": f.name,
                    "function_name": fn_name,
                    "critical_failures": critical_failures,
                    "low_score_dims": [
                        k for k, v in scores.items()
                        if isinstance(v, (int, float)) and v <= 5
                    ],
                    "substantive_failure_detected": has_sub_failure,
                    "high_stakes_low_dims": high_stakes_low,
                })
        except (json.JSONDecodeError, KeyError):
            continue

    print(
        f"Source filter: {len(functions)} functions with substantive defects\n"
        f"  Skipped — no defect at all:        {skipped_no_defect}\n"
        f"  Skipped — style/wpcs only:         {skipped_style_only}\n"
        f"  Skipped — test/vendored/deprecated:{skipped_excluded}\n"
        f"  Skipped — corrupted class_context: {skipped_corrupted}"
    )
    return functions


def partition_input_batches(num_batches: int = 10, batch_size: int = 20,
                             seed: int = 20260410) -> int:
    """Partition filtered Phase 1 failed functions into input batch files.

    Writes _input_batch_{NN:02d}.json to the CtF batches directory. Each file
    contains a list of function dicts ready to be processed by a Claude Code
    agent. The agent reads one file, produces a matching batch_{NNN:03d}.json
    with real critique-then-fix examples.
    """
    fns = load_failed_functions_filtered()
    if len(fns) < num_batches * batch_size:
        print(
            f"BLOCKED: Only {len(fns)} filtered functions available — "
            f"need at least {num_batches * batch_size}."
        )
        return 0

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Wipe any stale _input_batch_*.json files from prior runs
    for stale in BATCHES_DIR.glob("_input_batch_*.json"):
        stale.unlink()

    random.seed(seed)
    random.shuffle(fns)

    total_written = 0
    for i in range(num_batches):
        batch = fns[i * batch_size:(i + 1) * batch_size]
        if not batch:
            break
        out = BATCHES_DIR / f"_input_batch_{i:02d}.json"
        out.write_text(json.dumps(batch, indent=2))
        total_written += len(batch)
        print(f"Wrote input batch {i:02d}: {len(batch)} functions -> {out.name}")

    print(
        f"\nPartition complete: {num_batches} batches x {batch_size} = "
        f"{total_written} functions ready for Claude Code agents"
    )
    return total_written


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "partition":
        num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        partition_input_batches(num_batches=num_batches, batch_size=batch_size)
    else:
        print("Usage: python scripts/_partition_ctf_batches.py partition [num_batches] [batch_size]")
        sys.exit(1)
