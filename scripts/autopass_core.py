#!/usr/bin/env python3
"""Auto-pass WordPress Core (wordpress-develop) functions.

WordPress Core is the reference implementation — all functions are auto-passed
with tag-only assessment (no LLM judging needed).

Writes: data/phase1_extraction/output/passed/wordpress-develop.json
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"


def auto_tag_function(func: dict) -> list:
    """Assign taxonomy tags based on code content for core auto-pass."""
    tags = []
    body = func.get("body", "").lower()
    hooks = func.get("hooks_used", [])
    sql = func.get("sql_patterns", [])

    # SQL tags.
    if sql:
        if "prepared_query" in sql:
            tags.append("sql:prepared_statements")
        if "join" in sql:
            tags.append("sql:joins_across_meta")
        if "dbdelta" in sql:
            tags.append("sql:dbdelta_migrations")
        if any(p in sql for p in ["get_var", "get_col", "get_row"]):
            tags.append("sql:targeted_select")

    # Hook tags.
    if any("add_action" in h for h in hooks):
        tags.append("hooks:action_registration")
    if any("add_filter" in h for h in hooks):
        tags.append("hooks:filter_registration")

    # Security tags.
    if "wp_verify_nonce(" in body or "check_ajax_referer(" in body:
        tags.append("security:nonce_verification")
    if "current_user_can(" in body:
        tags.append("security:capability_checks")
    if any(esc in body for esc in ["esc_html(", "esc_attr(", "esc_url(", "wp_kses("]):
        tags.append("security:output_escaping")
    if any(s in body for s in ["sanitize_text_field(", "sanitize_email(", "absint("]):
        tags.append("security:input_sanitization")

    # Data modeling tags.
    if "register_post_type(" in body:
        tags.append("data:custom_post_types")
    if "register_taxonomy(" in body:
        tags.append("data:custom_taxonomies")
    if "register_rest_route(" in body:
        tags.append("rest:route_registration")
    if "set_transient(" in body or "get_transient(" in body:
        tags.append("data:transients")
    if "wp_cache_set(" in body or "wp_cache_get(" in body:
        tags.append("data:object_cache")

    # Performance tags.
    if "set_transient(" in body or "wp_cache_set(" in body:
        tags.append("perf:query_caching")
    if "wp_schedule_event(" in body:
        tags.append("cron:scheduled_events")

    # Theme tags.
    if "wp_enqueue_script(" in body or "wp_enqueue_style(" in body:
        tags.append("theme:enqueue_scripts")
    if "register_block_pattern(" in body:
        tags.append("theme:block_patterns")

    # Architecture tags.
    if "register_activation_hook(" in body:
        tags.append("arch:activation_hooks")
    if "register_deactivation_hook(" in body:
        tags.append("arch:deactivation_hooks")

    # Multisite tags.
    if "switch_to_blog(" in body:
        tags.append("multisite:site_switching")

    # i18n tags.
    if any(fn in body for fn in ["__(", "_e(", "esc_html__(", "esc_html_e(", "esc_attr__("]):
        tags.append("i18n:translation_functions")
    if "_n(" in body:
        tags.append("i18n:pluralization")

    # Accessibility tags.
    if any(a in body for a in ['aria-label', 'aria-describedby', 'role="', 'screen-reader-text']):
        tags.append("a11y:aria_attributes")
    if "<label" in body and 'for="' in body:
        tags.append("a11y:form_labels")

    return list(set(tags))


def main():
    PASSED_DIR.mkdir(parents=True, exist_ok=True)

    input_path = EXTRACTED_DIR / "wordpress-develop.json"
    output_path = PASSED_DIR / "wordpress-develop.json"

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        functions = json.load(f)

    print(f"Auto-passing {len(functions)} WordPress Core functions...")

    passed = []
    for i, func in enumerate(functions):
        training_tags = auto_tag_function(func)

        assessment = {
            "function_name": func["function_name"],
            "file_path": func.get("source_file", ""),
            "verdict": "PASS",
            "scores": {
                "wpcs_compliance": 10,
                "sql_safety": 10,
                "security": 10,
                "performance": 10,
                "wp_api_usage": 10,
                "code_quality": 10,
                "dependency_integrity": 10,
                "i18n": 10,
                "accessibility": 10,
            },
            "critical_failures": [],
            "dependency_chain": func.get("dependencies", []),
            "training_tags": training_tags,
            "notes": "WordPress Core — auto-passed",
        }

        func["assessment"] = assessment
        func["training_tags"] = training_tags
        passed.append(func)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(functions)}...")

    print(f"Writing {len(passed)} auto-passed functions to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(passed, f, indent=2)

    print(f"Done. wordpress-develop: {len(passed)} functions auto-passed.")


if __name__ == "__main__":
    main()
