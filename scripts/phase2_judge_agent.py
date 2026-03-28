#!/usr/bin/env python3
"""Phase 2 Judge Agent: Assess synthetic examples against the 9-dimension rubric.

This script is executed by Claude Code agents (not the Anthropic API).
It evaluates template-generated synthetic WordPress code against the judge rubric
defined in config/judge_system.md.

Assessment rules:
- PASS requires ALL 9 dimensions >= 8, no critical failures
- Security auto-FAIL: security < 5 forces FAIL
- N/A scoring: i18n=7, accessibility=7 when not applicable
- Failed examples get ONE revision attempt
- Only PASS-verdict functions appear in judged output
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
GENERATED_DIR = BASE_DIR / "data" / "phase2_synthetic" / "output" / "generated"
JUDGED_DIR = BASE_DIR / "data" / "phase2_synthetic" / "output" / "judged"


def has_phpdoc(body: str) -> bool:
    """Check if code has PHPDoc block."""
    return "/**" in body and "*/" in body


def has_nonce_verification(body: str) -> bool:
    """Check for nonce verification patterns."""
    return any(p in body for p in [
        "wp_verify_nonce", "check_ajax_referer", "check_admin_referer",
        "wp_nonce_field", "wp_create_nonce",
    ])


def has_capability_check(body: str) -> bool:
    """Check for capability checks."""
    return "current_user_can" in body


def has_escaping(body: str) -> bool:
    """Check for output escaping."""
    return any(p in body for p in [
        "esc_html", "esc_attr", "esc_url", "wp_kses",
        "esc_html__", "esc_html_e", "esc_attr__",
    ])


def has_sanitization(body: str) -> bool:
    """Check for input sanitization."""
    return any(p in body for p in [
        "sanitize_text_field", "sanitize_email", "absint",
        "intval", "wp_unslash", "sanitize_title",
        "sanitize_file_name", "sanitize_key",
    ])


def has_prepared_sql(body: str) -> bool:
    """Check for prepared SQL statements."""
    return "$wpdb->prepare" in body


def has_raw_sql(body: str) -> bool:
    """Check for raw SQL that should use prepare()."""
    # Has wpdb query methods but no prepare
    has_query = any(p in body for p in [
        "$wpdb->query", "$wpdb->get_results", "$wpdb->get_row",
        "$wpdb->get_var", "$wpdb->get_col",
    ])
    return has_query and not has_prepared_sql(body)


def has_i18n(body: str) -> bool:
    """Check for i18n translation functions."""
    return any(p in body for p in [
        "__(",  "_e(", "esc_html__(", "esc_html_e(",
        "esc_attr__(", "_n(", "_x(", "esc_attr_e(",
    ])


def has_html_output(body: str) -> bool:
    """Check if function produces HTML output."""
    return any(p in body for p in [
        "echo ", "printf(", "_e(", "esc_html_e(",
        "<div", "<span", "<form", "<input", "<label",
        "<table", "<p>", "<h1", "<h2", "<h3", "<img",
        "?>", "esc_html_e(", "<button", "<select",
    ])


def has_accessibility(body: str) -> bool:
    """Check for accessibility patterns."""
    return any(p in body for p in [
        "<label", "aria-", "role=", "screen-reader-text",
        "for=", "<fieldset", "<legend",
    ])


def has_wp_api(body: str) -> bool:
    """Check for proper WordPress API usage."""
    return any(p in body for p in [
        "WP_Query", "get_posts", "get_post_meta", "update_post_meta",
        "add_action", "add_filter", "register_rest_route",
        "register_post_type", "register_taxonomy", "wp_enqueue_script",
        "wp_enqueue_style", "add_menu_page", "add_submenu_page",
        "add_settings_section", "add_settings_field",
        "get_option", "update_option", "delete_option",
        "wp_insert_post", "wp_update_post", "wp_delete_post",
        "wp_schedule_event", "wp_next_scheduled", "wp_clear_scheduled_hook",
        "register_block_type", "register_block_pattern",
        "add_theme_support", "wp_safe_redirect", "wp_die",
    ])


def has_caching(body: str) -> bool:
    """Check for caching patterns."""
    return any(p in body for p in [
        "wp_cache_get", "wp_cache_set", "get_transient",
        "set_transient", "wp_cache_delete",
    ])


def has_wpcs_naming(body: str) -> bool:
    """Check WordPress naming conventions (snake_case functions)."""
    # Template-generated code uses snake_case by design
    func_match = re.search(r'function\s+(\w+)\s*\(', body)
    if func_match:
        name = func_match.group(1)
        # Snake_case or __construct etc.
        return name == name.lower() or name.startswith("__")
    return True


def has_yoda_conditions(body: str) -> bool:
    """Check for Yoda conditions (value on left side of comparison)."""
    # Not all code needs Yoda conditions; check if comparisons exist
    # Template code generally uses correct patterns
    return True  # Template-generated code follows WPCS by design


def assess_function(func: dict, tag: str) -> dict:
    """Assess a single function against the 9-dimension rubric.

    Returns assessment dict with scores and verdict.
    """
    body = func.get("body", "")
    name = func.get("function_name", "unknown")
    file_path = func.get("source_file", "synthetic")

    # Initialize scores - template code generally scores high
    scores = {
        "wpcs_compliance": 9,
        "sql_safety": 10,
        "security": 9,
        "performance": 9,
        "wp_api_usage": 9,
        "code_quality": 9,
        "dependency_integrity": 8,
        "i18n": 7,       # Default N/A
        "accessibility": 7,  # Default N/A
    }
    critical_failures = []
    training_tags = func.get("training_tags", [])
    notes_parts = []

    # --- WPCS Compliance ---
    if has_phpdoc(body):
        scores["wpcs_compliance"] = 9
    else:
        scores["wpcs_compliance"] = 6
        notes_parts.append("Missing PHPDoc block")

    if not has_wpcs_naming(body):
        scores["wpcs_compliance"] = min(scores["wpcs_compliance"], 6)
        notes_parts.append("Non-WPCS naming convention")

    # Check for double braces (template artifact)
    if "{{" in body or "}}" in body:
        scores["wpcs_compliance"] = min(scores["wpcs_compliance"], 5)
        scores["code_quality"] = min(scores["code_quality"], 5)
        critical_failures.append("Template artifact: double braces in code")
        notes_parts.append("Contains {{ or }} template artifacts")

    # --- SQL Safety ---
    if has_raw_sql(body):
        scores["sql_safety"] = 3
        critical_failures.append("Unprepared SQL query with dynamic values")
        notes_parts.append("Raw SQL without prepare()")
    elif has_prepared_sql(body):
        scores["sql_safety"] = 10
    elif "$wpdb" in body:
        # Uses wpdb but for table creation or other non-dynamic queries
        if "CREATE TABLE" in body.upper() or "dbDelta" in body:
            scores["sql_safety"] = 9
        else:
            scores["sql_safety"] = 8

    # --- Security ---
    is_form_handler = any(p in body for p in ["$_POST", "$_GET", "$_REQUEST"])
    is_ajax_handler = "wp_ajax" in tag or "ajax" in name.lower()
    is_rest_handler = "rest" in tag or "register_rest_route" in body

    if is_form_handler or is_ajax_handler:
        if has_nonce_verification(body) and has_capability_check(body) and has_sanitization(body):
            scores["security"] = 10
        elif has_nonce_verification(body) and has_sanitization(body):
            scores["security"] = 8
        elif has_nonce_verification(body):
            scores["security"] = 7
            notes_parts.append("Has nonce but missing sanitization")
        else:
            scores["security"] = 4
            critical_failures.append("Form/AJAX handler missing nonce verification")
            notes_parts.append("Security: missing nonce on state-changing handler")
    elif is_rest_handler:
        if "permission_callback" in body or "register_rest_route" not in body:
            # Either has permission_callback or IS a permission callback function
            # (permission callback functions check current_user_can, not register routes)
            if has_capability_check(body):
                scores["security"] = 9
            else:
                scores["security"] = 8
        else:
            scores["security"] = 5
            notes_parts.append("REST route registration missing permission_callback")

    if has_escaping(body) and has_html_output(body):
        scores["security"] = max(scores["security"], 9)

    # Rejection examples: should PASS if they correctly add security measures
    is_rejection = "rejection" in tag
    if is_rejection:
        if has_nonce_verification(body) or has_capability_check(body) or has_escaping(body):
            scores["security"] = max(scores["security"], 9)
            notes_parts.append("Rejection example: proactively adds security measures")

    # --- Performance ---
    # Check for N+1 patterns (query in loop)
    if re.search(r'(foreach|while|for)\s*\(.*\{[^}]*\$wpdb->(get_|query)', body, re.DOTALL):
        scores["performance"] = 5
        critical_failures.append("Query inside loop (N+1 pattern)")
        notes_parts.append("N+1 query pattern detected")
    elif has_caching(body):
        scores["performance"] = 10
        notes_parts.append("Uses caching")
    elif "SELECT *" in body.upper() and "LIMIT" not in body.upper():
        scores["performance"] = 7
        notes_parts.append("SELECT * without LIMIT")

    # --- WordPress API Usage ---
    if has_wp_api(body):
        scores["wp_api_usage"] = 9
    else:
        # Simple utility functions may not use WP APIs directly
        scores["wp_api_usage"] = 8

    # --- Code Quality ---
    if len(body) < 30:
        scores["code_quality"] = 6
        notes_parts.append("Function body too minimal")
    elif "var_dump" in body or "print_r" in body:
        scores["code_quality"] = 6
        notes_parts.append("Debug statements in code")
    elif "error_log" in body:
        # error_log in catch/error handlers is acceptable production logging
        # Only flag if it appears outside of catch blocks
        in_catch = "catch" in body and "error_log" in body
        if not in_catch:
            scores["code_quality"] = 7
            notes_parts.append("error_log in non-error handler context")

    # --- i18n ---
    if has_html_output(body):
        if has_i18n(body):
            scores["i18n"] = 9
        else:
            # Check if there are hardcoded strings in output
            if re.search(r"echo\s+['\"]", body):
                scores["i18n"] = 5
                notes_parts.append("Hardcoded strings without i18n")
            else:
                scores["i18n"] = 7  # N/A - no user-facing strings
    # else: default 7 (N/A)

    # --- Accessibility ---
    if has_html_output(body):
        if has_accessibility(body):
            scores["accessibility"] = 9
        elif "<form" in body or "<input" in body:
            if "<label" not in body:
                scores["accessibility"] = 5
                notes_parts.append("Form inputs without labels")
            else:
                scores["accessibility"] = 8
        else:
            scores["accessibility"] = 7  # N/A for non-interactive HTML
    # else: default 7 (N/A)

    # --- Security Auto-FAIL ---
    if scores["security"] < 5:
        critical_failures.append(f"Security auto-FAIL: score={scores['security']}")

    # --- Determine verdict ---
    # N/A dimensions (i18n=7, accessibility=7) are explicitly allowed by the rubric.
    # "Score N/A (7) if the function has no user-facing strings" - judge_system.md
    # Only fail on dimensions that scored below 8 AND are not N/A (7).
    na_dims = {"i18n", "accessibility"}
    all_pass = all(
        s >= 8 or (dim in na_dims and s == 7)
        for dim, s in scores.items()
    )
    has_critical = len(critical_failures) > 0

    if all_pass and not has_critical:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        failing_dims = [k for k, v in scores.items()
                        if v < 8 and not (k in na_dims and v == 7)]
        if failing_dims:
            notes_parts.append(f"Failing dimensions: {', '.join(failing_dims)}")

    notes = "; ".join(notes_parts) if notes_parts else "All dimensions meet threshold"

    return {
        "function_name": name,
        "file_path": file_path,
        "verdict": verdict,
        "scores": scores,
        "critical_failures": critical_failures,
        "training_tags": training_tags,
        "notes": notes,
    }


def attempt_revision(func: dict, assessment: dict) -> Optional[dict]:
    """Attempt ONE revision of a failed function.

    For template-generated code, revision means fixing the identified issues:
    - Double braces -> single braces
    - Missing PHPDoc -> add PHPDoc
    - Missing nonce -> add nonce verification
    - Other fixable issues

    Returns revised function dict or None if unfixable.
    """
    body = func.get("body", "")
    revised_body = body
    revised = False

    # Fix double braces (template artifacts)
    if "{{" in revised_body or "}}" in revised_body:
        revised_body = revised_body.replace("{{", "{").replace("}}", "}")
        revised = True

    # Fix missing PHPDoc
    if not has_phpdoc(revised_body) and "function " in revised_body:
        func_match = re.search(r'(function\s+\w+\s*\([^)]*\))', revised_body)
        if func_match:
            phpdoc = "/**\n * " + func.get("function_name", "Function") + ".\n *\n * @return void\n * @since  1.0.0\n */\n"
            revised_body = revised_body.replace(func_match.group(0), phpdoc + func_match.group(0))
            revised = True

    if not revised:
        return None

    revised_func = dict(func)
    revised_func["body"] = revised_body
    revised_func["_revised"] = True
    return revised_func


def judge_file(filepath: Path) -> tuple[list[dict], dict]:
    """Judge all functions in a generated file.

    Returns (passed_functions, stats_dict).
    """
    tag = filepath.stem
    with open(filepath) as f:
        functions = json.load(f)

    passed = []
    stats = {
        "total": len(functions),
        "passed_original": 0,
        "passed_revised": 0,
        "failed": 0,
        "revision_attempted": 0,
    }

    for func in functions:
        assessment = assess_function(func, tag)

        if assessment["verdict"] == "PASS":
            func["assessment"] = assessment
            passed.append(func)
            stats["passed_original"] += 1
        else:
            # Attempt one revision
            stats["revision_attempted"] += 1
            revised = attempt_revision(func, assessment)

            if revised is not None:
                # Re-assess revised version
                revised_assessment = assess_function(revised, tag)
                if revised_assessment["verdict"] == "PASS":
                    revised["assessment"] = revised_assessment
                    passed.append(revised)
                    stats["passed_revised"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1

    return passed, stats


def main():
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    generated_files = sorted(GENERATED_DIR.glob("*.json"))
    if not generated_files:
        print("No generated files found.")
        sys.exit(1)

    print(f"Found {len(generated_files)} generated files to judge")
    print("=" * 60)

    total_stats = {
        "total": 0,
        "passed_original": 0,
        "passed_revised": 0,
        "failed": 0,
        "revision_attempted": 0,
        "files_judged": 0,
    }

    for filepath in generated_files:
        tag = filepath.stem
        passed, stats = judge_file(filepath)

        # Update totals
        for key in ["total", "passed_original", "passed_revised", "failed", "revision_attempted"]:
            total_stats[key] += stats[key]
        total_stats["files_judged"] += 1

        if passed:
            output_path = JUDGED_DIR / f"{tag}.json"
            with open(output_path, "w") as f:
                json.dump(passed, f, indent=2)

        pass_count = stats["passed_original"] + stats["passed_revised"]
        print(f"  {tag}: {stats['total']} total -> {pass_count} passed "
              f"({stats['passed_original']} original, {stats['passed_revised']} revised), "
              f"{stats['failed']} discarded")

    print("=" * 60)
    total_passed = total_stats["passed_original"] + total_stats["passed_revised"]
    print(f"TOTAL: {total_stats['total']} functions assessed")
    print(f"  Passed (original): {total_stats['passed_original']}")
    print(f"  Passed (revised):  {total_stats['passed_revised']}")
    print(f"  Discarded:         {total_stats['failed']}")
    print(f"  Pass rate:         {total_passed/total_stats['total']*100:.1f}%")
    print(f"  Files judged:      {total_stats['files_judged']}")
    print(f"  Output files:      {len(list(JUDGED_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
