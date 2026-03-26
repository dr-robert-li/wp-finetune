#!/usr/bin/env python3
"""Agent-based function judging using static code analysis.

Applies the 9-dimension WordPress code quality rubric from config/judge_system.md
using heuristic analysis of PHP code patterns.

Usage:
    python3 scripts/agent_judge.py <repo1> [<repo2> ...]

Each repo must have an extracted file in phase1_extraction/output/extracted/.
Results are written to phase1_extraction/output/passed/ and failed/.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "failed"


# ---------------------------------------------------------------------------
# Rubric scoring heuristics
# ---------------------------------------------------------------------------

def score_wpcs_compliance(func: dict) -> tuple:
    """Dimension 1: WordPress Coding Standards compliance."""
    body = func.get("body", "") or ""
    docblock = func.get("docblock", "") or ""
    func_name = func.get("function_name", "")
    failures = []
    score = 9  # Start optimistic

    # Check naming convention: lowercase_with_underscores for functions
    # Class methods are exempt (ClassName::method_name)
    bare_name = func_name.split("::")[-1] if "::" in func_name else func_name
    if bare_name and not bare_name.startswith("__"):
        # Should be lowercase_with_underscores or camelCase (for OOP methods)
        if re.search(r"[A-Z]", bare_name) and "_" not in bare_name and "::" not in func_name:
            # Probably camelCase function (not method) — mild deduction
            score -= 1

    # Missing PHPDoc for non-trivial functions
    lines = body.count("\n") + 1
    if lines >= 5 and not docblock:
        score -= 2
        failures.append("missing_phpdoc")
    elif docblock and "@param" not in docblock and "@return" not in docblock:
        # Has docblock but missing param/return tags
        if lines >= 10:
            score -= 1

    # Debug statements in production code
    if re.search(r"\bvar_dump\b|\bprint_r\b|\bdie\(|\bexit\(", body):
        score -= 3
        failures.append("debug_statements")

    # Inline HTML style attributes (minor WPCS issue)
    if "style=" in body and ("echo" in body or "?>" in body):
        score -= 1

    score = max(1, min(10, score))
    return score, failures


def score_sql_safety(func: dict) -> tuple:
    """Dimension 2: SQL safety — prepared statements, no concatenation."""
    body = func.get("body", "") or ""
    sql_patterns = func.get("sql_patterns", []) or []
    failures = []
    score = 10

    # Check for raw SQL queries
    has_sql = (
        re.search(r"\$wpdb\s*->\s*(query|get_results|get_row|get_col|get_var)", body) or
        "SELECT " in body.upper() or
        "INSERT INTO" in body.upper() or
        "UPDATE " in body.upper() or
        "DELETE FROM" in body.upper()
    )

    if has_sql:
        has_prepare = re.search(r"\$wpdb\s*->\s*prepare\s*\(", body)
        # Look for string concatenation into SQL
        has_concat_sql = re.search(
            r'\$wpdb\s*->\s*(query|get_results|get_row|get_col|get_var)\s*\(\s*["\'].*?\$',
            body, re.DOTALL
        ) or re.search(
            r'\$wpdb\s*->\s*(query|get_results|get_row|get_col|get_var)\s*\(\s*["\'][^"\']*\s*\.\s*\$',
            body
        )

        if has_concat_sql and not has_prepare:
            score = 1
            failures.append("unprepared_query_with_concat")
        elif not has_prepare:
            # Has SQL but no prepare — check if it uses literals only
            if re.search(r"\$(?!wpdb\b)\w+", body.split("->query")[1] if "->query" in body else ""):
                score = 3
                failures.append("possible_unprepared_query")
            else:
                score -= 1  # Minor concern

        # Check for hardcoded wp_ prefix instead of $wpdb->prefix
        if re.search(r"['\"]\s*wp_\w+\s*['\"]", body) and "$wpdb->prefix" not in body:
            score -= 1

    return max(1, min(10, score)), failures


def score_security(func: dict) -> tuple:
    """Dimension 3: Security — nonces, capabilities, escaping, sanitization."""
    body = func.get("body", "") or ""
    hooks = func.get("hooks_used", []) or []
    failures = []
    score = 9  # Start optimistic

    # Check for form/AJAX handlers without nonce verification
    is_form_handler = (
        re.search(r"wp_insert_post|wp_update_post|update_option|delete_option|update_post_meta", body) or
        re.search(r"\$_POST|\$_GET|\$_REQUEST", body)
    )
    has_nonce = re.search(r"wp_verify_nonce|check_ajax_referer|check_admin_referer", body)
    has_capability = re.search(r"current_user_can\s*\(", body)

    if is_form_handler and not has_nonce:
        # State-changing operations without nonce verification
        if re.search(r"\$_POST|\$_REQUEST", body):
            score -= 4
            failures.append("missing_nonce_on_state_change")

    # Check unescaped output
    has_echo_output = re.search(r"\becho\b|\bprint\b|<\?=", body)
    has_escaping = re.search(r"esc_html|esc_attr|esc_url|wp_kses|esc_js|wp_kses_post", body)

    if has_echo_output and not has_escaping:
        # Check if the echo contains user-controlled data
        if re.search(r"echo.*\$_(GET|POST|REQUEST|COOKIE|SERVER)", body):
            score = 1
            failures.append("unescaped_user_controlled_output")
        elif re.search(r"echo.*\$(?!this->|self::)\w+", body):
            score -= 2
            failures.append("possible_unescaped_output")

    # Check for use of extract() on user data
    if re.search(r"\bextract\s*\(\s*\$_(GET|POST|REQUEST|COOKIE|FILES)", body):
        score = 1
        failures.append("extract_on_user_data")

    # Check for eval()
    if re.search(r"\beval\s*\(", body):
        score = 1
        failures.append("eval_usage")

    # Check for direct file operations
    if re.search(r"\bfopen\s*\(|\bfile_put_contents\s*\(|\bfile_get_contents\s*\(", body):
        if "WP_Filesystem" not in body and "wp_filesystem" not in body:
            score -= 2
            failures.append("direct_file_operations")

    # SQL injection vector check
    if re.search(r"\$wpdb.*query.*\$_(GET|POST|REQUEST)", body, re.DOTALL):
        score = 1
        failures.append("sql_injection_vector")

    score = max(1, min(10, score))
    return score, failures


def score_performance(func: dict) -> tuple:
    """Dimension 4: Performance — no N+1, caching, efficient queries."""
    body = func.get("body", "") or ""
    failures = []
    score = 9

    # N+1 query pattern: query inside a loop
    loop_patterns = re.findall(
        r"(foreach|for|while)\s*\([^)]+\)\s*\{[^}]*\$wpdb",
        body, re.DOTALL
    )
    if loop_patterns:
        score -= 3
        failures.append("query_in_loop")

    # WP_Query or get_posts inside a loop
    query_in_loop = re.search(
        r"(foreach|for|while).*?(new\s+WP_Query|get_posts\s*\(|wp_get_post_terms|get_post_meta)",
        body, re.DOTALL
    )
    if query_in_loop:
        score -= 2
        failures.append("wp_query_in_loop")

    # SELECT * without LIMIT on meta tables
    if re.search(r"SELECT\s+\*\s+FROM\s+\w*(meta|options)", body, re.IGNORECASE):
        if not re.search(r"LIMIT\s+\d+", body, re.IGNORECASE):
            score -= 2
            failures.append("select_star_meta_no_limit")

    # Unbounded loops with no break condition
    if re.search(r"while\s*\(\s*true\s*\)", body):
        score -= 1

    score = max(1, min(10, score))
    return score, failures


def score_wp_api_usage(func: dict) -> tuple:
    """Dimension 5: WordPress API usage — correct hooks, APIs, patterns."""
    body = func.get("body", "") or ""
    hooks = func.get("hooks_used", []) or []
    failures = []
    score = 9

    # Raw SQL for post queries instead of WP_Query
    if re.search(r"SELECT.*FROM.*wp_posts", body, re.IGNORECASE):
        if "WP_Query" not in body and "get_posts" not in body:
            score -= 2
            failures.append("raw_sql_instead_of_wp_query")

    # REST endpoints without permission_callback
    if re.search(r"register_rest_route\s*\(", body):
        if "permission_callback" not in body:
            score -= 3
            failures.append("rest_route_missing_permission_callback")

    # Incorrect Options API usage (storing large arrays without considering autoload)
    # This is too nuanced to detect reliably, skip

    # Custom taxonomy/post type registration — check for completeness
    if re.search(r"register_post_type\s*\(", body):
        if "labels" not in body:
            score -= 1

    score = max(1, min(10, score))
    return score, failures


def score_code_quality(func: dict) -> tuple:
    """Dimension 6: Code quality — single responsibility, error handling, no dead code."""
    body = func.get("body", "") or ""
    func_name = func.get("function_name", "")
    failures = []
    score = 9

    lines = body.count("\n") + 1

    # Very long functions may violate single responsibility
    if lines > 200:
        score -= 2
        failures.append("function_too_long")
    elif lines > 100:
        score -= 1

    # Commented-out code blocks
    commented_blocks = len(re.findall(r"//.*(?:TODO|FIXME|HACK|XXX|TEMP)", body, re.IGNORECASE))
    if commented_blocks > 3:
        score -= 1

    # Dead code: unreachable return
    if re.search(r"return\s*;.*return\s*;", body, re.DOTALL):
        score -= 1

    # Error swallowing: empty catch blocks
    if re.search(r"catch\s*\([^)]+\)\s*\{\s*\}", body):
        score -= 2
        failures.append("empty_catch_block")

    # Nested control flow depth (complexity indicator)
    if body.count("{") - body.count("}") > 0:
        score -= 1  # Unbalanced braces (truncated function body)

    # Check for use of global variables without checking
    global_uses = len(re.findall(r"\bglobal\s+\$\w+", body))
    if global_uses > 3:
        score -= 1

    score = max(1, min(10, score))
    return score, failures


def score_dependency_integrity(func: dict) -> tuple:
    """Dimension 7: Dependency chain integrity."""
    body = func.get("body", "") or ""
    dependencies = func.get("dependencies", []) or []
    failures = []
    score = 9

    # Direct require/include of vendor files (not through WordPress patterns)
    if re.search(r"\brequire_once?\s*\(\s*['\"](?!.*vendor\/autoload)", body):
        score -= 1
    if re.search(r"\brequire_once?\s*\(\s*['\"][^'\"]*vendor\/[^'\"]+['\"]", body):
        score -= 2
        failures.append("direct_vendor_require")

    # Circular dependency detection (function calling itself without recursion guard)
    bare_name = func.get("function_name", "").split("::")[-1]
    if bare_name and re.search(rf"\b{re.escape(bare_name)}\s*\(", body):
        # Self-call without obvious recursion pattern
        if not re.search(r"static\s+\$", body) and "$depth" not in body and "$level" not in body:
            score -= 1  # Could be intentional recursion

    score = max(1, min(10, score))
    return score, failures


def score_i18n(func: dict) -> tuple:
    """Dimension 8: Internationalization — translation functions, text domain."""
    body = func.get("body", "") or ""
    failures = []

    # Check if function has any user-facing string output
    has_output = re.search(r"\becho\b|\bprint\b|return\s+['\"]", body)
    has_html = re.search(r"<[a-zA-Z][^>]*>|<\?=", body)

    if not has_output and not has_html:
        # No user-facing output — N/A score
        return 7, []

    # Has output — check for translation functions
    has_translation = re.search(
        r"\b__\s*\(|\b_e\s*\(|\besc_html__\s*\(|\besc_html_e\s*\(|\besc_attr__\s*\(|\b_n\s*\(|\b_x\s*\(",
        body
    )

    # Check for hardcoded English strings in echo/print
    hardcoded_strings = re.findall(
        r'(?:echo|print)\s+["\'][A-Za-z][A-Za-z\s]{3,}["\']',
        body
    )

    if hardcoded_strings and not has_translation:
        score = 4
        failures.append("hardcoded_user_facing_strings")
    elif hardcoded_strings:
        score = 7  # Has some translation but also some hardcoded
    elif has_translation:
        score = 9
    else:
        score = 8  # Output but no obvious user-facing strings

    return max(1, min(10, score)), failures


def score_accessibility(func: dict) -> tuple:
    """Dimension 9: Accessibility — labels, ARIA, semantic HTML."""
    body = func.get("body", "") or ""
    failures = []

    # Check if function produces HTML output
    has_html = re.search(r"<[a-zA-Z][^>]*>|echo\s+.*<\w", body)

    if not has_html:
        # No HTML output — N/A score
        return 7, []

    # Has HTML — check accessibility patterns
    score = 8  # Default for HTML-producing functions

    # Form inputs without labels
    has_input = re.search(r"<input[^>]+>", body, re.IGNORECASE)
    has_label = re.search(r"<label[^>]*for=|<label[^>]*htmlFor=", body, re.IGNORECASE)
    has_aria_label = re.search(r"aria-label=|aria-labelledby=", body, re.IGNORECASE)

    if has_input and not has_label and not has_aria_label:
        # Check if it's a hidden input
        if not re.search(r'<input[^>]+type=["\']hidden["\']', body, re.IGNORECASE):
            score -= 2
            failures.append("form_input_without_label")

    # Images without alt text
    has_img = re.search(r"<img[^>]+>", body, re.IGNORECASE)
    has_alt = re.search(r"alt=", body, re.IGNORECASE)

    if has_img and not has_alt:
        score -= 2
        failures.append("image_without_alt_text")

    # ARIA attributes present — bonus
    if re.search(r"aria-|role=", body, re.IGNORECASE):
        score = min(10, score + 1)

    # Screen reader text patterns
    if "screen-reader-text" in body or "sr-only" in body:
        score = min(10, score + 1)

    return max(1, min(10, score)), failures


def assess_function(func: dict) -> dict:
    """Apply all 9 dimensions and produce an assessment dict."""
    wpcs_score, wpcs_failures = score_wpcs_compliance(func)
    sql_score, sql_failures = score_sql_safety(func)
    security_score, security_failures = score_security(func)
    perf_score, perf_failures = score_performance(func)
    api_score, api_failures = score_wp_api_usage(func)
    quality_score, quality_failures = score_code_quality(func)
    dep_score, dep_failures = score_dependency_integrity(func)
    i18n_score, i18n_failures = score_i18n(func)
    a11y_score, a11y_failures = score_accessibility(func)

    all_failures = (
        wpcs_failures + sql_failures + security_failures + perf_failures +
        api_failures + quality_failures + dep_failures + i18n_failures + a11y_failures
    )

    scores = {
        "wpcs_compliance": wpcs_score,
        "sql_safety": sql_score,
        "security": security_score,
        "performance": perf_score,
        "wp_api_usage": api_score,
        "code_quality": quality_score,
        "dependency_integrity": dep_score,
        "i18n": i18n_score,
        "accessibility": a11y_score,
    }

    # Determine verdict: all scores >= 8 AND no critical failures
    min_score = min(scores.values())
    verdict = "PASS" if min_score >= 8 and not all_failures else "FAIL"

    # Security auto-FAIL: security < 5 forces FAIL
    if security_score < 5:
        verdict = "FAIL"
        if "security_auto_fail" not in all_failures:
            all_failures.append("security_auto_fail")

    # Generate training tags
    training_tags = generate_training_tags(func, scores)

    # Generate notes
    if verdict == "PASS":
        notes = f"All dimensions >= 8 (min: {min_score}). Clean WordPress code."
    else:
        notes = f"Failed dimensions: {', '.join(all_failures) if all_failures else f'scores below 8 (min: {min_score})'}"

    return {
        "function_name": func.get("function_name", ""),
        "file_path": func.get("source_file", ""),
        "verdict": verdict,
        "scores": scores,
        "critical_failures": all_failures,
        "dependency_chain": func.get("dependencies", []),
        "training_tags": training_tags,
        "notes": notes,
    }


def generate_training_tags(func: dict, scores: dict) -> list:
    """Generate training_tags based on function content."""
    body = func.get("body", "") or ""
    hooks = func.get("hooks_used", []) or []
    sql_patterns = func.get("sql_patterns", []) or []
    tags = []

    # SQL tags
    if sql_patterns:
        if "prepared_query" in sql_patterns:
            tags.append("sql:prepared_statements")
        if "join" in sql_patterns:
            tags.append("sql:joins_across_meta")
        if "dbdelta" in sql_patterns:
            tags.append("sql:dbdelta_migrations")
        if any(p in sql_patterns for p in ["get_var", "get_col", "get_row"]):
            tags.append("sql:targeted_select")

    # Hook tags
    if any("add_action" in h for h in hooks):
        tags.append("hooks:action_registration")
    if any("add_filter" in h for h in hooks):
        tags.append("hooks:filter_registration")

    # Security tags
    body_lower = body.lower()
    if "wp_verify_nonce(" in body_lower or "check_ajax_referer(" in body_lower:
        tags.append("security:nonce_verification")
    if "current_user_can(" in body_lower:
        tags.append("security:capability_checks")
    if any(e in body_lower for e in ["esc_html(", "esc_attr(", "esc_url(", "wp_kses("]):
        tags.append("security:output_escaping")
    if any(s in body_lower for s in ["sanitize_text_field(", "sanitize_email(", "absint("]):
        tags.append("security:input_sanitization")

    # Data modeling tags
    if "register_post_type(" in body_lower:
        tags.append("data:custom_post_types")
    if "register_taxonomy(" in body_lower:
        tags.append("data:custom_taxonomies")
    if "register_rest_route(" in body_lower:
        tags.append("rest:route_registration")
    if "set_transient(" in body_lower or "get_transient(" in body_lower:
        tags.append("data:transients")
    if "wp_cache_set(" in body_lower or "wp_cache_get(" in body_lower:
        tags.append("data:object_cache")

    # Performance tags
    if "set_transient(" in body_lower or "wp_cache_set(" in body_lower:
        tags.append("perf:query_caching")
    if "wp_schedule_event(" in body_lower:
        tags.append("cron:scheduled_events")

    # Theme/enqueue tags
    if "wp_enqueue_script(" in body_lower or "wp_enqueue_style(" in body_lower:
        tags.append("theme:enqueue_scripts")
    if "register_block_pattern(" in body_lower:
        tags.append("theme:block_patterns")

    # Architecture tags
    if "register_activation_hook(" in body_lower:
        tags.append("arch:activation_hooks")
    if "register_deactivation_hook(" in body_lower:
        tags.append("arch:deactivation_hooks")

    # Multisite tags
    if "switch_to_blog(" in body_lower:
        tags.append("multisite:site_switching")

    # i18n tags
    if any(fn in body_lower for fn in ["__(", "_e(", "esc_html__(", "esc_html_e(", "esc_attr__("]):
        tags.append("i18n:translation_functions")
    if "_n(" in body_lower:
        tags.append("i18n:pluralization")

    # Accessibility tags
    if any(a in body_lower for a in ['aria-label', 'aria-describedby', 'role="', 'screen-reader-text']):
        tags.append("a11y:aria_attributes")
    if "<label" in body_lower and 'for="' in body_lower:
        tags.append("a11y:form_labels")

    # WP_Query usage
    if "wp_query" in body_lower or "new wp_query" in body_lower:
        tags.append("data:wp_query")

    # REST API
    if "wp_rest" in body_lower or "rest_api" in body_lower or "wp_rest_response" in body_lower:
        tags.append("rest:api_response")

    # AJAX handling
    if "wp_ajax_" in str(hooks) or "admin-ajax" in body_lower:
        tags.append("ajax:handler")

    return list(set(tags))


def judge_repo(repo_name: str) -> dict:
    """Judge all functions in a repo and write results to passed/failed dirs."""
    PASSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already judged
    passed_path = PASSED_DIR / f"{repo_name}.json"
    failed_path = FAILED_DIR / f"{repo_name}.json"
    if passed_path.exists() or failed_path.exists():
        print(f"  [{repo_name}] Already judged — skipping")
        return {"repo": repo_name, "status": "skipped"}

    extracted_path = EXTRACTED_DIR / f"{repo_name}.json"
    if not extracted_path.exists():
        print(f"  [{repo_name}] No extracted file — skipping")
        return {"repo": repo_name, "status": "missing_extracted"}

    with open(extracted_path) as f:
        functions = json.load(f)

    print(f"  [{repo_name}] Judging {len(functions)} functions...")

    passed = []
    failed = []

    for i, func in enumerate(functions):
        assessment = assess_function(func)
        func["assessment"] = assessment
        func["training_tags"] = assessment.get("training_tags", [])

        if assessment["verdict"] == "PASS":
            passed.append(func)
        else:
            failed.append(func)

        if (i + 1) % 100 == 0:
            print(f"  [{repo_name}] Progress: {i + 1}/{len(functions)} "
                  f"(passed: {len(passed)}, failed: {len(failed)})")

    # Write results
    if passed:
        with open(passed_path, "w") as f:
            json.dump(passed, f, indent=2)
    else:
        # Write empty passed file to mark repo as done
        with open(passed_path, "w") as f:
            json.dump([], f, indent=2)

    if failed:
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)

    pass_rate = len(passed) / len(functions) * 100 if functions else 0
    print(f"  [{repo_name}] Done: {len(passed)} passed ({pass_rate:.1f}%), {len(failed)} failed")
    return {
        "repo": repo_name,
        "status": "done",
        "total": len(functions),
        "passed": len(passed),
        "failed": len(failed),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/agent_judge.py <repo1> [<repo2> ...]")
        sys.exit(1)

    repos = sys.argv[1:]
    results = []

    for repo in repos:
        result = judge_repo(repo)
        results.append(result)

    # Summary
    print("\n=== Summary ===")
    total_passed = sum(r.get("passed", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)
    total_funcs = sum(r.get("total", 0) for r in results)
    for r in results:
        if r["status"] == "done":
            print(f"  {r['repo']}: {r['passed']}/{r['total']} passed")
        else:
            print(f"  {r['repo']}: {r['status']}")
    print(f"Total: {total_passed}/{total_funcs} passed ({total_failed} failed)")


if __name__ == "__main__":
    main()
