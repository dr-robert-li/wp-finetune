#!/usr/bin/env python3
"""
Phase 2 Step 4: Generate judge training data from phase1 extraction output.

Samples 200 HIGH-quality (passed/) and 200 LOW-quality (failed/) functions,
converts 1-10 rubric scores to 0-100 scale, and produces judge training JSONL.
"""

import json
import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PASSED_DIR = REPO_ROOT / "data/phase1_extraction/output/passed"
FAILED_DIR = REPO_ROOT / "data/phase1_extraction/output/failed"
OUTPUT_DIR = REPO_ROOT / "data/phase2_synthetic/output/judge_training"
OUTPUT_FILE = OUTPUT_DIR / "judge_training.json"

SEED = 42
SAMPLE_HIGH = 200
SAMPLE_LOW = 200

# Dimension mapping: rubric names -> output names (task spec)
# The task spec uses 6 dimensions + overall; rubric has 9 dimensions
# Map: wpcs_compliance, security_score, performance_score, i18n_score,
#       accessibility_score, documentation_score
# We'll use the rubric scores and map them:
#   wpcs_compliance   <- wpcs_compliance
#   security_score    <- security (combined with sql_safety)
#   performance_score <- performance
#   i18n_score        <- i18n
#   accessibility_score <- accessibility
#   documentation_score <- code_quality (closest proxy, code docs)
# Weights for overall_score (out of 100):
DIMENSION_WEIGHTS = {
    "wpcs_compliance": 0.15,
    "security_score": 0.30,    # security matters most
    "performance_score": 0.15,
    "i18n_score": 0.10,
    "accessibility_score": 0.10,
    "documentation_score": 0.20,  # code_quality as documentation proxy
}

PASS_THRESHOLD = 70  # overall_score >= 70 passes


def scale_10_to_100(score: int | float) -> int:
    """Convert 1-10 rubric score to 0-100."""
    return round((score / 10) * 100)


def compute_scores(assessment: dict) -> dict:
    """Extract and convert rubric scores to task spec format (0-100).

    For FAIL verdicts, critical failures cause the affected dimension(s) to
    be penalised so the overall_score correctly lands in the 10-65 range.
    """
    raw = assessment.get("scores", {})
    verdict = assessment.get("verdict", "FAIL")
    critical_failures = assessment.get("critical_failures", [])
    has_critical = bool(critical_failures)

    # Security combines security + sql_safety (average)
    raw_security = raw.get("security", 5)
    raw_sql = raw.get("sql_safety", 10)
    combined_security = (raw_security + raw_sql) / 2

    wpcs = scale_10_to_100(raw.get("wpcs_compliance", 5))
    security = scale_10_to_100(combined_security)
    performance = scale_10_to_100(raw.get("performance", 5))
    i18n = scale_10_to_100(raw.get("i18n", 7))  # N/A default=7
    accessibility = scale_10_to_100(raw.get("accessibility", 7))  # N/A default=7
    documentation = scale_10_to_100(raw.get("code_quality", 5))

    # For FAIL verdict, apply critical-failure penalties to reflect true quality.
    # Critical failures indicate the code cannot pass despite possibly high
    # dimension sub-scores.  We lower the *most relevant* dimension(s) and
    # penalise the overall to ensure FAIL items score 10-65 overall.
    if verdict == "FAIL" and has_critical:
        cf_text = " ".join(critical_failures).lower()

        # Security-related critical failures
        if any(k in cf_text for k in ("nonce", "capability", "unescaped", "sql inject", "xss", "security")):
            security = min(security, 45)

        # Accessibility critical failures
        if any(k in cf_text for k in ("label", "alt text", "keyboard", "aria", "accessibility")):
            accessibility = min(accessibility, 45)

        # SQL safety critical failures
        if any(k in cf_text for k in ("sql", "prepare", "injection")):
            security = min(security, 30)

        # Code-quality / standards critical failures
        if any(k in cf_text for k in ("naming", "phpdoc", "standard", "dead code", "debug")):
            wpcs = min(wpcs, 50)
            documentation = min(documentation, 50)

        # Performance critical failures
        if any(k in cf_text for k in ("loop", "unbounded", "n+1", "performance")):
            performance = min(performance, 40)

    elif verdict == "FAIL" and not has_critical:
        # FAIL without explicit critical_failures: minor overall penalty.
        # The phase1 judge may have failed on dimension thresholds (all >= 8).
        # Identify the lowest-scoring dimension and penalise it further.
        dim_raw = {
            "security": combined_security,
            "wpcs": raw.get("wpcs_compliance", 5),
            "code_quality": raw.get("code_quality", 5),
        }
        lowest_dim = min(dim_raw, key=dim_raw.get)
        if dim_raw[lowest_dim] < 9:
            if lowest_dim == "security":
                security = min(security, 55)
            elif lowest_dim == "wpcs":
                wpcs = min(wpcs, 55)
            else:
                documentation = min(documentation, 55)

    scores = {
        "wpcs_compliance": wpcs,
        "security_score": security,
        "performance_score": performance,
        "i18n_score": i18n,
        "accessibility_score": accessibility,
        "documentation_score": documentation,
    }

    # Weighted overall
    overall = (
        wpcs * DIMENSION_WEIGHTS["wpcs_compliance"]
        + security * DIMENSION_WEIGHTS["security_score"]
        + performance * DIMENSION_WEIGHTS["performance_score"]
        + i18n * DIMENSION_WEIGHTS["i18n_score"]
        + accessibility * DIMENSION_WEIGHTS["accessibility_score"]
        + documentation * DIMENSION_WEIGHTS["documentation_score"]
    )
    scores["overall_score"] = round(overall)

    # Security auto-fail: if raw security < 5 overall is capped at 40
    if raw_security < 5:
        scores["overall_score"] = min(scores["overall_score"], 40)

    # Hard cap: FAIL items must not exceed 65 overall
    if verdict == "FAIL":
        scores["overall_score"] = min(scores["overall_score"], 65)

    return scores


def build_issues(assessment: dict, scores: dict) -> tuple[list[str], list[str]]:
    """Build must_fix_issues and suggested_improvements from assessment data."""
    must_fix = list(assessment.get("critical_failures", []))
    suggestions = []

    raw = assessment.get("scores", {})

    # Generate must-fix from low security scores
    if raw.get("security", 10) < 5:
        if not any("security" in m.lower() for m in must_fix):
            must_fix.append("Security critical failure: review nonce verification, output escaping, and capability checks")

    if raw.get("sql_safety", 10) < 5:
        must_fix.append("SQL safety: use $wpdb->prepare() for all dynamic SQL queries")

    # Generate suggestions from medium-low scores
    if raw.get("wpcs_compliance", 10) < 8:
        suggestions.append("Improve WordPress Coding Standards compliance: add PHPDoc blocks, fix naming conventions")

    if raw.get("performance", 10) < 8:
        suggestions.append("Add transient/object caching for expensive operations; avoid queries in loops")

    if raw.get("i18n", 10) < 7:
        suggestions.append("Wrap user-facing strings in translation functions (__(), _e(), esc_html__())")

    if raw.get("accessibility", 10) < 7:
        suggestions.append("Add proper ARIA attributes and ensure form inputs have associated labels")

    if raw.get("code_quality", 10) < 8:
        suggestions.append("Improve code quality: add error handling, reduce function complexity")

    if raw.get("wp_api_usage", 10) < 8:
        suggestions.append("Use WordPress APIs (WP_Query, Options API) instead of raw SQL or direct DB access")

    if not must_fix and not suggestions:
        notes = assessment.get("notes", "")
        if notes:
            suggestions.append(f"Minor improvements: {notes}")

    return must_fix, suggestions


def build_explanation(item: dict, assessment: dict, scores: dict, passes: bool) -> str:
    """Build a human-readable explanation string."""
    verdict = assessment.get("verdict", "FAIL")
    fn = item.get("function_name", "unknown")
    notes = assessment.get("notes", "")
    tags = item.get("training_tags", [])

    parts = [
        f"Function `{fn}` receives verdict {verdict}.",
        f"Overall score: {scores['overall_score']}/100.",
    ]

    if scores["security_score"] < 50:
        parts.append("Security auto-fail triggered: security score below 50.")

    if tags:
        parts.append(f"Demonstrates: {', '.join(tags[:5])}.")

    if notes:
        parts.append(notes)

    if passes:
        parts.append("Passes quality threshold for training inclusion.")
    else:
        parts.append("Does not pass quality threshold; excluded from training set.")

    return " ".join(parts)


def build_assistant_response(item: dict, assessment: dict, source: str) -> str:
    """Build the JSON assistant response string."""
    scores = compute_scores(assessment)
    passes = scores["overall_score"] >= PASS_THRESHOLD
    must_fix, suggestions = build_issues(assessment, scores)
    explanation = build_explanation(item, assessment, scores, passes)

    response = {
        "wpcs_compliance": scores["wpcs_compliance"],
        "security_score": scores["security_score"],
        "performance_score": scores["performance_score"],
        "i18n_score": scores["i18n_score"],
        "accessibility_score": scores["accessibility_score"],
        "documentation_score": scores["documentation_score"],
        "overall_score": scores["overall_score"],
        "must_fix_issues": must_fix,
        "suggested_improvements": suggestions,
        "passes_threshold": passes,
        "explanation": explanation,
    }

    return json.dumps(response, indent=2, ensure_ascii=False)


def load_samples(directory: Path, target_count: int, source_label: str) -> list[dict]:
    """Load and sample functions from a directory of JSON files."""
    all_items = []

    files = sorted([f for f in os.listdir(directory) if f.endswith(".json")])
    random.shuffle(files)

    for fname in files:
        fpath = directory / fname
        if fpath.stat().st_size < 10:
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            assessment = item.get("assessment")
            if not assessment:
                continue
            if not assessment.get("scores"):
                continue
            body = item.get("body", "").strip()
            if not body or len(body) < 50:
                continue
            item["_source_label"] = source_label
            item["_source_file"] = fname
            all_items.append(item)

    random.shuffle(all_items)
    return all_items[:target_count]


def build_training_example(item: dict) -> dict:
    """Build a single judge training example."""
    assessment = item["assessment"]
    source = item["_source_label"]
    fn_name = item.get("function_name", "unknown")
    file_path = item.get("source_file", "unknown")
    code = item.get("body", "").strip()

    # Prepend docblock if available
    docblock = item.get("docblock", "")
    if docblock:
        full_code = docblock.strip() + "\n" + code
    else:
        full_code = code

    user_content = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{full_code}\n```"
    assistant_content = build_assistant_response(item, assessment, source)

    # Parse scores for metadata
    scores = compute_scores(assessment)
    passes = scores["overall_score"] >= PASS_THRESHOLD

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "task_type": "judge",
            "function_name": fn_name,
            "file_path": file_path,
            "source_repo": item.get("source_repo", ""),
            "overall_score": scores["overall_score"],
            "passes_threshold": passes,
            "source": source,
        },
    }


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading HIGH quality samples from {PASSED_DIR}...")
    high_items = load_samples(PASSED_DIR, SAMPLE_HIGH, "phase1_passed")
    print(f"  Loaded {len(high_items)} high-quality functions")

    print(f"Loading LOW quality samples from {FAILED_DIR}...")
    low_items = load_samples(FAILED_DIR, SAMPLE_LOW, "phase1_failed")
    print(f"  Loaded {len(low_items)} low-quality functions")

    all_items = high_items + low_items
    random.shuffle(all_items)

    examples = []
    score_dist = {"high": [], "low": []}

    for item in all_items:
        try:
            example = build_training_example(item)
            examples.append(example)
            src = item["_source_label"]
            overall = example["metadata"]["overall_score"]
            if src == "phase1_passed":
                score_dist["high"].append(overall)
            else:
                score_dist["low"].append(overall)
        except Exception as e:
            print(f"  Warning: skipped {item.get('function_name', '?')}: {e}")

    print(f"\nGenerated {len(examples)} training examples")
    print(f"  HIGH quality: {len(score_dist['high'])} examples")
    if score_dist["high"]:
        print(f"    Score range: {min(score_dist['high'])}-{max(score_dist['high'])}, avg={sum(score_dist['high'])/len(score_dist['high']):.1f}")
    print(f"  LOW quality:  {len(score_dist['low'])} examples")
    if score_dist["low"]:
        print(f"    Score range: {min(score_dist['low'])}-{max(score_dist['low'])}, avg={sum(score_dist['low'])/len(score_dist['low']):.1f}")

    passes = sum(1 for e in examples if e["metadata"]["passes_threshold"])
    print(f"  Passes threshold: {passes}/{len(examples)} ({100*passes/len(examples):.1f}%)")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(examples)} examples to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
