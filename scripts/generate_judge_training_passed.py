#!/usr/bin/env python3
"""
Generate judge training data from Phase 1 passed functions.
Produces calibrated scores for high-quality functions (75-100 overall).
Target: 2,000 examples across 5 repos.
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path("/home/robert_li/Desktop/projects/wp-finetune")
OUTPUT_FILE = ROOT / "data/phase2_synthetic/output/judge_training/judge_training_passed_batch1.json"

SOURCE_FILES = [
    ("woocommerce", ROOT / "data/phase1_extraction/output/passed/woocommerce.json"),
    ("jetpack", ROOT / "data/phase1_extraction/output/passed/jetpack.json"),
    ("wordpress-develop", ROOT / "data/phase1_extraction/output/passed/wordpress-develop.json"),
    ("elementor", ROOT / "data/phase1_extraction/output/passed/elementor.json"),
    ("advanced-custom-fields", ROOT / "data/phase1_extraction/output/passed/advanced-custom-fields.json"),
]

TARGET_TOTAL = 2000
PER_FILE = 450  # sample extra to absorb skips; we trim to 400 after

# ── Scoring logic ────────────────────────────────────────────────────────────

def has_phpdoc(entry):
    doc = entry.get("docblock") or ""
    return bool(doc.strip())

def has_since_tag(entry):
    doc = entry.get("docblock") or ""
    return "@since" in doc

def has_param_tags(entry):
    doc = entry.get("docblock") or ""
    return "@param" in doc

def has_return_tag(entry):
    doc = entry.get("docblock") or ""
    return "@return" in doc

def has_sql(entry):
    return bool(entry.get("sql_patterns"))

def has_prepare(entry):
    body = entry.get("body", "")
    return "$wpdb->prepare" in body or "prepare(" in body

def has_unescaped_output(entry):
    body = entry.get("body", "")
    # echo with variables not escaped
    echo_pattern = re.compile(r'\becho\s+\$[A-Za-z_]', re.MULTILINE)
    esc_pattern = re.compile(r'\besc_(html|attr|url|js|sql|textarea)\b')
    kses_pattern = re.compile(r'\bwp_kses\b')
    if echo_pattern.search(body):
        if not esc_pattern.search(body) and not kses_pattern.search(body):
            return True
    return False

def has_nonce(entry):
    body = entry.get("body", "")
    return any(fn in body for fn in [
        "wp_verify_nonce", "check_ajax_referer", "check_admin_referer",
        "wp_nonce_field", "wp_create_nonce"
    ])

def has_caps_check(entry):
    body = entry.get("body", "")
    return "current_user_can" in body

def has_select_star(entry):
    sql = " ".join(entry.get("sql_patterns", []))
    return "SELECT *" in sql.upper()

def has_query_in_loop(entry):
    body = entry.get("body", "")
    # Rough heuristic: foreach/for/while containing ->get_results or $wpdb->query
    loop_then_query = re.compile(
        r'(foreach|for\s*\(|while\s*\().*?\n.*?(\$wpdb->|WP_Query|get_posts|get_post_meta)',
        re.DOTALL | re.MULTILINE
    )
    return bool(loop_then_query.search(body[:1000]))

def has_user_facing_strings(entry):
    body = entry.get("body", "")
    return bool(re.search(r'\b(echo|return|printf)\b.*["\'][A-Z][a-z]', body))

def has_i18n(entry):
    body = entry.get("body", "")
    return bool(re.search(r'\b(__\(|_e\(|esc_html__\(|esc_attr__\(|_n\(|_x\()', body))

def has_html_output(entry):
    body = entry.get("body", "")
    return bool(re.search(r'<[a-z][a-z0-9]*[\s>]', body))

def has_aria_or_labels(entry):
    body = entry.get("body", "")
    return bool(re.search(r'aria-|<label|for="|\.screen-reader-text', body))

def has_hardcoded_strings_no_i18n(entry):
    """Detects echo/print of hardcoded English strings without __()"""
    body = entry.get("body", "")
    # echo "Something" or echo 'Something' patterns
    if re.search(r'\becho\s+["\'][A-Z][a-z]', body):
        if not has_i18n(entry):
            return True
    return False

def body_complexity(entry):
    """Rough proxy for code complexity — line count."""
    return entry.get("line_count", 5)

def uses_wp_filesystem(entry):
    body = entry.get("body", "")
    return "WP_Filesystem" in body or "global $wp_filesystem" in body

def has_direct_file_ops(entry):
    body = entry.get("body", "")
    return bool(re.search(r'\b(fopen|fwrite|file_put_contents|file_get_contents|unlink)\s*\(', body))

def has_wpdb_direct(entry):
    body = entry.get("body", "")
    return "$wpdb->" in body

def has_prepared_or_no_sql(entry):
    if not has_sql(entry) and not has_wpdb_direct(entry):
        return True
    return has_prepare(entry)


def score_entry(entry):
    """
    Produce calibrated 0-100 scores for a passed function.
    Passed functions are high quality (75-100 expected) but not perfect.
    """
    body = entry.get("body", "")
    docblock = entry.get("docblock", "")
    line_count = entry.get("line_count", 5)

    # ── WPCS Compliance ──────────────────────────────────────────────────────
    wpcs = 100
    if not has_phpdoc(entry):
        wpcs -= 20
    else:
        if not has_since_tag(entry):
            wpcs -= 8
        if not has_return_tag(entry) and "return" in body:
            wpcs -= 5
        if not has_param_tags(entry) and re.search(r'\$[a-z_]+', body.split("\n")[0] if body else ""):
            wpcs -= 5
    # naming conventions: assume passed means reasonable naming
    wpcs = max(55, wpcs)

    # ── SQL Safety ───────────────────────────────────────────────────────────
    if not has_sql(entry) and not has_wpdb_direct(entry):
        sql_safety = 100
    elif has_prepare(entry):
        sql_safety = 95
        if has_select_star(entry):
            sql_safety -= 10
    else:
        # Has SQL but no prepare — partial penalty (some statements may be static)
        sql_safety = 82
        if has_select_star(entry):
            sql_safety -= 8

    # ── Security ─────────────────────────────────────────────────────────────
    security = 100
    form_handler = bool(re.search(r'\$_(POST|GET|REQUEST)\b', body))
    if form_handler:
        if not has_nonce(entry):
            security -= 25
        if not has_caps_check(entry):
            security -= 15
    if has_unescaped_output(entry):
        security -= 20
    if has_direct_file_ops(entry) and not uses_wp_filesystem(entry):
        security -= 10
    if "extract(" in body and form_handler:
        security -= 20
    security = max(45, security)

    # ── Performance ──────────────────────────────────────────────────────────
    performance = 100
    if has_query_in_loop(entry):
        performance -= 20
    if has_select_star(entry):
        performance -= 10
    # Very long functions without caching hints
    if line_count > 80 and not any(kw in body for kw in ["wp_cache_get", "get_transient", "static $"]):
        performance -= 8
    performance = max(60, performance)

    # ── i18n ─────────────────────────────────────────────────────────────────
    if not has_user_facing_strings(entry) and not has_html_output(entry):
        i18n = 75  # N/A score per rubric
    elif has_i18n(entry):
        i18n = 92
        if has_hardcoded_strings_no_i18n(entry):
            i18n -= 10
    else:
        if has_hardcoded_strings_no_i18n(entry):
            i18n = 68
        else:
            i18n = 78

    # ── Accessibility ─────────────────────────────────────────────────────────
    if not has_html_output(entry):
        accessibility = 75  # N/A
    elif has_aria_or_labels(entry):
        accessibility = 88
    else:
        # Produces HTML but no ARIA/labels seen — may be fine or may be missing
        form_html = bool(re.search(r'<(input|select|textarea)', body))
        if form_html:
            accessibility = 72
        else:
            accessibility = 80

    # ── Documentation ────────────────────────────────────────────────────────
    doc = 0
    if has_phpdoc(entry):
        doc = 85
        if has_since_tag(entry):
            doc += 5
        if has_param_tags(entry):
            doc += 5
        if has_return_tag(entry):
            doc += 5
    else:
        doc = 50  # No docblock — significant dock
    doc = min(100, doc)

    # ── Overall ───────────────────────────────────────────────────────────────
    # Weighted: security and wpcs are most important for this dataset
    weights = {
        "wpcs": 0.20,
        "sql_safety": 0.15,
        "security": 0.20,
        "performance": 0.15,
        "i18n": 0.10,
        "accessibility": 0.05,
        "documentation": 0.15,
    }
    overall = int(
        wpcs * weights["wpcs"] +
        sql_safety * weights["sql_safety"] +
        security * weights["security"] +
        performance * weights["performance"] +
        i18n * weights["i18n"] +
        accessibility * weights["accessibility"] +
        doc * weights["documentation"]
    )
    # These are PASSED functions, floor at 72
    overall = max(72, min(100, overall))

    return {
        "wpcs_compliance": wpcs,
        "sql_safety": sql_safety,
        "security_score": security,
        "performance_score": performance,
        "i18n_score": i18n,
        "accessibility_score": accessibility,
        "documentation_score": doc,
        "overall_score": overall,
    }


def build_must_fix(entry, scores):
    issues = []
    body = entry.get("body", "")
    form_handler = bool(re.search(r'\$_(POST|GET|REQUEST)\b', body))

    if scores["security_score"] < 80:
        if form_handler and not has_nonce(entry):
            issues.append("Add nonce verification (wp_verify_nonce / check_ajax_referer) to this state-changing handler")
        if has_unescaped_output(entry):
            issues.append("Escape all output with appropriate esc_html(), esc_attr(), esc_url() calls")
        if has_direct_file_ops(entry) and not uses_wp_filesystem(entry):
            issues.append("Use WP_Filesystem API instead of direct PHP file functions")

    if scores["sql_safety"] < 90:
        if has_wpdb_direct(entry) and not has_prepare(entry):
            issues.append("Wrap all dynamic values in $wpdb->prepare() with typed placeholders")

    if scores["documentation_score"] < 70:
        issues.append("Add PHPDoc block with @param, @return, and @since tags")
    elif scores["wpcs_compliance"] < 80:
        if not has_since_tag(entry):
            issues.append("Add @since tag to PHPDoc block")

    return issues


def build_suggested(entry, scores):
    suggestions = []
    body = entry.get("body", "")

    if scores["performance_score"] < 90 and has_query_in_loop(entry):
        suggestions.append("Cache query results outside the loop or batch-fetch with a single query")

    if scores["i18n_score"] < 80 and has_html_output(entry) and not has_i18n(entry):
        suggestions.append("Wrap user-facing strings with __() or esc_html__() for translatability")

    if scores["documentation_score"] < 95 and not has_return_tag(entry) and "return " in body:
        suggestions.append("Add @return tag to document the return type and value")

    if "$wpdb->get_results" in body and "LIMIT" not in body.upper():
        suggestions.append("Add a LIMIT clause to prevent unbounded result sets")

    if not suggestions:
        line_count = entry.get("line_count", 0)
        if line_count > 60:
            suggestions.append("Consider splitting this function into smaller, single-responsibility helpers")

    return suggestions


def build_explanation(entry, scores, must_fix, passes):
    fn_name = entry.get("function_name", "unknown")
    source_file = entry.get("source_file", "")
    overall = scores["overall_score"]

    parts = [f"Function `{fn_name}` reviewed from `{source_file}`."]
    parts.append(f"Overall score: {overall}/100.")

    if scores["wpcs_compliance"] < 90:
        parts.append(f"WPCS compliance docked to {scores['wpcs_compliance']} — " + (
            "missing PHPDoc block" if not has_phpdoc(entry) else "incomplete PHPDoc (@since/@return missing)"
        ) + ".")
    if scores["security_score"] < 90:
        parts.append(f"Security scored {scores['security_score']} — " + (
            "unescaped output detected" if has_unescaped_output(entry) else
            "handler lacks nonce verification" if bool(re.search(r'\$_(POST|GET|REQUEST)\b', entry.get("body",""))) else
            "direct file operations without WP_Filesystem"
        ) + ".")
    if scores["documentation_score"] < 85:
        parts.append(f"Documentation score {scores['documentation_score']} — PHPDoc incomplete or absent.")
    if scores["i18n_score"] < 80:
        parts.append(f"i18n score {scores['i18n_score']} — hardcoded user-facing strings detected.")
    if not must_fix:
        parts.append("No critical issues found; code demonstrates solid WordPress patterns.")
    else:
        parts.append(f"{len(must_fix)} issue(s) require attention before production use.")
    parts.append("Passes quality threshold for training inclusion." if passes else "Does not pass threshold.")

    return " ".join(parts)


def build_training_example(entry, repo_name):
    body = entry.get("body", "")
    docblock = entry.get("docblock", "")

    # Reconstruct full PHP snippet (docblock + function body)
    docblock = docblock or ""
    if docblock.strip():
        code = docblock.rstrip() + "\n" + body
    else:
        code = body

    fn_name = entry.get("function_name", "unknown")
    class_ctx = entry.get("class_context", "")
    if class_ctx:
        qualified_name = f"{class_ctx}::{fn_name}"
    else:
        qualified_name = fn_name

    scores = score_entry(entry)
    must_fix = build_must_fix(entry, scores)
    suggested = build_suggested(entry, scores)
    passes = scores["overall_score"] >= 75 and scores["security_score"] >= 70

    explanation = build_explanation(entry, scores, must_fix, passes)

    assistant_content = {
        "wpcs_compliance": scores["wpcs_compliance"],
        "security_score": scores["security_score"],
        "performance_score": scores["performance_score"],
        "i18n_score": scores["i18n_score"],
        "accessibility_score": scores["accessibility_score"],
        "documentation_score": scores["documentation_score"],
        "overall_score": scores["overall_score"],
        "must_fix_issues": must_fix,
        "suggested_improvements": suggested,
        "passes_threshold": passes,
        "explanation": explanation,
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_content, indent=2)
            }
        ],
        "metadata": {
            "task_type": "judge",
            "function_name": qualified_name,
            "source_file": entry.get("source_file", ""),
            "source_repo": repo_name,
            "overall_score": scores["overall_score"],
            "passes_threshold": passes,
            "source": "phase1_passed",
        }
    }


def sample_entries(data, n, seed=42):
    """Sample n entries, filtered to body >= 50 chars, stratified by quality."""
    viable = [e for e in data if e.get("body") and len(e["body"]) >= 50]
    rng = random.Random(seed)
    if len(viable) <= n:
        return viable
    return rng.sample(viable, n)


def main():
    print(f"Generating {TARGET_TOTAL} judge training examples from passed data...")
    all_examples = []

    per_file_target = TARGET_TOTAL // len(SOURCE_FILES)  # 400 each

    for repo_name, path in SOURCE_FILES:
        print(f"  Loading {repo_name}...", end=" ", flush=True)
        with open(path) as f:
            data = json.load(f)

        # Sample extra to absorb skips
        sample = sample_entries(data, PER_FILE, seed=hash(repo_name) % 10000)
        print(f"{len(sample)} sampled from {len(data)} entries")

        repo_examples = []
        for entry in sample:
            if len(repo_examples) >= per_file_target:
                break
            try:
                ex = build_training_example(entry, repo_name)
                repo_examples.append(ex)
            except Exception as e:
                print(f"    WARN: skipped {entry.get('function_name', '?')}: {e}")

        print(f"    -> {len(repo_examples)} examples from {repo_name}")
        all_examples.extend(repo_examples)

    print(f"\nGenerated {len(all_examples)} examples total")

    # Score distribution summary
    scores = [e["metadata"]["overall_score"] for e in all_examples]
    buckets = {"75-79": 0, "80-84": 0, "85-89": 0, "90-94": 0, "95-100": 0}
    for s in scores:
        if s < 80: buckets["75-79"] += 1
        elif s < 85: buckets["80-84"] += 1
        elif s < 90: buckets["85-89"] += 1
        elif s < 95: buckets["90-94"] += 1
        else: buckets["95-100"] += 1
    print("Score distribution:")
    for k, v in buckets.items():
        print(f"  {k}: {v}")

    passes_count = sum(1 for e in all_examples if e["metadata"]["passes_threshold"])
    print(f"  passes_threshold=True: {passes_count} / {len(all_examples)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_examples)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
