#!/usr/bin/env python3
"""
Heuristic judge for WordPress PHP functions.
Applies the rubric from config/judge_system.md to extracted functions.
"""

import json
import os
import re
import sys
from typing import Optional

BASE = "/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output"
EXTRACTED = os.path.join(BASE, "extracted")
PASSED_DIR = os.path.join(BASE, "passed")
FAILED_DIR = os.path.join(BASE, "failed")

os.makedirs(PASSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

REPOS = [
    "woo-safepay-gateway",
    "woo-widget-product-slideshow",
    "woocommerce",
    "woocommerce-extra-checkout-fields-for-brazil",
    "woocommerce-gateway-stripe",
    "woocommerce-google-analytics-integration",
    "woocommerce-legacy-rest-api",
    "woocommerce-mercadopago",
    "woocommerce-payments",
    "woocommerce-paypal-payments",
    "woocommerce-services",
    "woocommerce-shipping",
    "woocommerce-square",
    "wordpress-develop",
    "wordpress-importer",
    "wordpress-seo",
    "wp-db-backup",
    "wp-frequently-searched-words",
    "wp-instantarticles",
    "wp-job-manager-companies",
]

# wordpress-develop is auto-passed (core tier)
CORE_REPOS = {"wordpress-develop"}

# ─── Pattern helpers ────────────────────────────────────────────────────────

def body(fn: dict) -> str:
    return (fn.get("body") or "").strip()

def doc(fn: dict) -> str:
    return (fn.get("docblock") or "").strip()

def has_body_pattern(fn: dict, *patterns) -> bool:
    b = body(fn)
    return any(re.search(p, b) for p in patterns)

def body_lower(fn: dict) -> str:
    return body(fn).lower()


# ─── Dimension scorers ───────────────────────────────────────────────────────

def score_wpcs(fn: dict) -> tuple[int, list[str]]:
    """1. WordPress Coding Standards"""
    failures = []
    score = 10

    b = body(fn)
    d = doc(fn)
    fname = fn.get("function_name", "")
    bare_name = fname.split("::")[-1] if "::" in fname else fname
    is_method = "::" in fname
    is_php_magic = bare_name.startswith("__")  # PHP magic methods are exempt from some rules

    # Tabs vs spaces check (needed early to inform combined scoring)
    lines = b.split("\n")
    indented_space = [l for l in lines[1:] if l.startswith("    ") and not l.startswith("\t")]
    tab_indented = [l for l in lines[1:] if l.startswith("\t")]
    has_space_indent = len(indented_space) > 3 and len(indented_space) > len(tab_indented)

    # Check for docblock presence - CRITICAL in WP standards
    if not d:
        score -= 5  # Missing PHPDoc is a serious violation (backup gives 3-5)
        failures.append("Missing PHPDoc block")
    else:
        # Missing @since - noted; compounded with other issues becomes more severe
        missing_since = "@since" not in d
        line_count = fn.get("line_count", 0)
        missing_params = (line_count > 10 and "@param" not in d and "@return" not in d and not is_php_magic)

        if missing_since and (missing_params or has_space_indent):
            # Multiple WPCS violations in docblock area
            score -= 3
            failures.append("Multiple PHPDoc/WPCS issues: missing @since" +
                          (" + missing @param/@return" if missing_params else "") +
                          (" + space indentation" if has_space_indent else ""))
        elif missing_since:
            score -= 1
            failures.append("Missing @since tag in PHPDoc block")
        elif missing_params:
            score -= 2
            failures.append("PHPDoc missing @param/@return tags")

    # Check naming conventions - WPCS requires snake_case for standalone functions
    # PHP magic methods (__construct, __toString, etc.) are always exempt
    # Class methods in PSR-4 style plugins can use camelCase - this is common/accepted
    # Only penalize camelCase for standalone functions (not class methods)
    if not is_php_magic and not is_method and re.search(r"[A-Z]", bare_name) and not bare_name[0].isupper():
        # camelCase standalone function - clear WPCS violation
        score -= 5
        failures.append("camelCase function naming violates WPCS (WordPress requires snake_case for functions)")

    # Check for debug statements
    if re.search(r"\b(var_dump|print_r|var_export)\s*\(", b):
        score -= 5
        failures.append("Debug output function (var_dump/print_r) found in production code")

    # error_log in non-debug context (mild)
    if re.search(r"\berror_log\s*\(", b):
        score -= 1

    # Space indentation is a WPCS violation (if not already captured above)
    if has_space_indent and d and "@since" in d:
        # Space indent but otherwise good docblock
        score -= 1
        failures.append("Space indentation instead of tabs (WPCS requires tabs)")
    elif has_space_indent and not d:
        # Already captured by missing PHPDoc, just note it
        failures.append("Space indentation instead of tabs (WPCS requires tabs)")

    return max(1, score), failures


def score_sql(fn: dict) -> tuple[int, list[str]]:
    """2. SQL Safety"""
    failures = []
    score = 10

    b = body(fn)
    sql_patterns = fn.get("sql_patterns") or []

    # Check for hardcoded wp_ prefix
    if re.search(r'["\']wp_\w+["\']', b):
        score -= 2
        failures.append("Hardcoded wp_ table prefix instead of $wpdb->prefix")

    # Check for unprepared queries with user-controlled dynamic values
    # Pattern: $wpdb->query/get_results etc with user superglobal input ($_ vars) directly
    # We must distinguish safe WP table refs ($wpdb->posts) from dangerous user input ($var from $_POST)

    # Check if prepare() is used
    has_prepare = bool(re.search(r'\$wpdb->prepare\s*\(', b))

    # Dangerous: user superglobal concatenated into query without prepare
    user_input_in_query = re.search(
        r'\$wpdb->\s*(query|get_results|get_row|get_var|get_col|insert|update|delete)\s*\([^)]*\$_(POST|GET|REQUEST|COOKIE)',
        b, re.DOTALL
    )
    # Also catch string concatenation of untrusted vars (non-wpdb, non-this, non-common-safe patterns)
    # More conservative: only flag if actual $_POST/$_GET appears in query string
    concat_user_input = re.search(
        r'\$wpdb->\s*(query|get_results|get_row|get_var|get_col)\s*\(\s*["\'][^"\']*["\'\s]*\.\s*\$(?!wpdb|this|post_type|post_status|post_id|table)',
        b
    )

    has_raw_dynamic = bool(user_input_in_query)

    if has_raw_dynamic and not has_prepare:
        score -= 6
        failures.append("CRITICAL: Unprepared dynamic SQL query with user input (SQL injection risk)")
    elif concat_user_input and not has_prepare:
        # String concatenation in query - moderate risk
        score -= 3
        failures.append("String concatenation in SQL query without prepare()")
    elif has_raw_dynamic and has_prepare:
        score -= 1

    # Check for SELECT * on potentially large tables (meta tables especially)
    if re.search(r'SELECT\s+\*\s+FROM', b, re.IGNORECASE):
        score -= 1

    # Bonus: uses prepare correctly
    if has_prepare:
        score = min(10, score + 1)

    return max(1, score), failures


def score_security(fn: dict) -> tuple[int, list[str]]:
    """3. Security"""
    failures = []
    score = 10

    b = body(fn)
    bl = body_lower(fn)

    # Check for eval()
    if re.search(r'\beval\s*\(', b):
        score -= 8
        failures.append("CRITICAL: Use of eval()")

    # Check for extract() on untrusted data
    if re.search(r'\bextract\s*\(\s*\$_(POST|GET|REQUEST|COOKIE)', b):
        score -= 7
        failures.append("CRITICAL: extract() on superglobal data")

    # Check for unescaped output of user-controlled data
    # echo $_POST, echo $_GET, echo $_REQUEST without escaping
    unescaped_output = re.search(
        r'\becho\s+\$_(POST|GET|REQUEST|COOKIE|SERVER)\b',
        b
    )
    if unescaped_output:
        score -= 7
        failures.append("CRITICAL: Unescaped output of superglobal data")

    # Check for direct file operations (should use WP_Filesystem)
    if re.search(r'\b(fopen|file_put_contents|file_get_contents|fwrite)\s*\(', b):
        score -= 2
        failures.append("Direct file operation instead of WP_Filesystem")

    # State-changing handlers: check for nonce verification
    # Look for form processing (handling $_POST) without nonce check
    has_post_processing = bool(re.search(r'\$_(POST|REQUEST)\s*\[', b))
    has_nonce_check = bool(re.search(
        r'(wp_verify_nonce|check_ajax_referer|check_admin_referer)\s*\(',
        b
    ))
    has_capability_check = bool(re.search(
        r'current_user_can\s*\(',
        b
    ))
    # phpcs:ignore for nonce means the team explicitly acknowledged and suppressed it
    has_nonce_ignore = bool(re.search(
        r'phpcs:ignore\s+WordPress\.Security\.Nonce',
        b
    ))

    # If it's an AJAX handler or form handler without nonce (and no explicit ignore)
    if has_post_processing and not has_nonce_check and not has_nonce_ignore:
        # This could legitimately have nonce in calling function, but rubric says assess as-is
        score -= 3
        failures.append("Missing nonce verification on POST data handler")

    # Check for unescaped echo of variables (not superglobals, but dynamic vars)
    # echo $var; without esc_* - moderate concern
    unescaped_var_echo = re.findall(r'\becho\s+(\$(?!this)[a-zA-Z_]\w*)\s*;', b)
    # Filter out known safe variables
    dangerous_echo = [v for v in unescaped_var_echo if not any(
        safe in v.lower() for safe in ['html', 'output', 'content', 'markup']
    )]
    if len(dangerous_echo) > 2:
        score -= 1

    # Missing permission check on capability-sensitive operations
    admin_operations = bool(re.search(r'(delete_posts|update_option|add_user|wp_insert_user)', b))
    if admin_operations and not has_capability_check:
        score -= 1

    return max(1, score), failures


def score_performance(fn: dict) -> tuple[int, list[str]]:
    """4. Performance"""
    failures = []
    score = 10

    b = body(fn)

    # Check for SELECT *
    if re.search(r'SELECT\s+\*\s+FROM\s+\w*_(post)?meta', b, re.IGNORECASE):
        score -= 3
        failures.append("SELECT * on meta table")

    # Check for queries inside loops (N+1)
    # Look for $wpdb-> or WP_Query or get_posts inside foreach/while/for
    loop_query = re.search(
        r'(foreach|for|while)\s*\([^{]+\)\s*\{[^}]*\$wpdb->',
        b, re.DOTALL
    )
    loop_wp_query = re.search(
        r'(foreach|for|while)\s*\([^{]+\)\s*\{[^}]*(new\s+WP_Query|get_posts|get_post_meta)\s*\(',
        b, re.DOTALL
    )
    if loop_query or loop_wp_query:
        score -= 4
        failures.append("CRITICAL: Database query inside a loop (N+1 pattern)")

    # Check for missing cache on expensive operations
    has_db_query = bool(re.search(r'\$wpdb->(query|get_results|get_row|get_var)', b))
    has_cache = bool(re.search(r'(wp_cache_get|wp_cache_set|get_transient|set_transient)', b))
    if has_db_query and not has_cache:
        score -= 1  # mild deduction, not all queries need caching

    # Unbounded queries (no LIMIT)
    if re.search(r'SELECT.*FROM', b, re.IGNORECASE) and not re.search(r'LIMIT\s+\d+', b, re.IGNORECASE):
        if has_db_query:
            score -= 1

    return max(1, score), failures


def score_wp_api(fn: dict) -> tuple[int, list[str]]:
    """5. WordPress API Usage"""
    failures = []
    score = 10

    b = body(fn)
    hooks = fn.get("hooks_used") or []

    # Raw SQL for post queries instead of WP_Query
    raw_post_query = re.search(
        r'\$wpdb->.*SELECT.*FROM.*wp_posts',
        b, re.IGNORECASE | re.DOTALL
    )
    wp_query_used = bool(re.search(r'new\s+WP_Query|get_posts\s*\(', b))
    if raw_post_query and not wp_query_used:
        score -= 3
        failures.append("Raw SQL for post queries instead of WP_Query")

    # REST endpoint missing permission_callback
    rest_route = re.search(r'register_rest_route\s*\(', b)
    if rest_route:
        if not re.search(r'permission_callback', b):
            score -= 5
            failures.append("CRITICAL: REST route registered without permission_callback")
        elif re.search(r"permission_callback.*__return_true", b):
            score -= 2
            failures.append("REST route uses __return_true permission_callback (no auth)")

    # Check hook argument count consistency (mild check)
    # add_action/add_filter with 4 args should match callback arg count
    for m in re.finditer(r'add_(action|filter)\s*\(\s*[\'"][^\'"]+[\'"]\s*,\s*[^\,]+,\s*(\d+)\s*,\s*(\d+)\s*\)', b):
        accepted_args = int(m.group(3))
        # Just noting - hard to verify without running the code

    # Uses deprecated functions
    deprecated = re.search(r'\b(wp_get_current_user|get_currentuserinfo|the_category_head|wp_specialchars)\s*\(', b)
    if deprecated:
        score -= 2
        failures.append("Uses deprecated WordPress function")

    return max(1, score), failures


def score_code_quality(fn: dict) -> tuple[int, list[str]]:
    """6. Code Quality"""
    failures = []
    score = 10

    b = body(fn)
    lines = b.split("\n")

    # Check for dead code / commented blocks
    comment_lines = [l for l in lines if re.match(r'\s*//.*', l) or re.match(r'\s*/\*.*', l)]
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("//") and not l.strip().startswith("*")]
    if len(comment_lines) > len(code_lines) * 0.5 and len(comment_lines) > 5:
        score -= 1  # heavy commenting could be docs or dead code

    # Check for var_dump / print_r debug
    if re.search(r'\b(var_dump|print_r|var_export)\s*\(', b):
        score -= 3
        failures.append("Debug statement in production code")

    # Check for error swallowing: empty catch blocks
    if re.search(r'catch\s*\([^)]+\)\s*\{\s*\}', b):
        score -= 2
        failures.append("Empty catch block (swallowing errors)")

    # Single responsibility: very long functions may be doing too much
    # (line_count is already tracked)
    line_count = fn.get("line_count", 0)
    if line_count > 200:
        score -= 1

    # Null/empty checks presence - hard to judge absence, skip

    return max(1, score), failures


def score_dependency(fn: dict) -> tuple[int, list[str]]:
    """7. Dependency Chain Integrity"""
    failures = []
    score = 8  # default to 8 (acceptable)

    b = body(fn)

    # Direct require/include of vendor files (not using WP patterns)
    if re.search(r'\b(require|include)(_once)?\s*\(', b):
        if re.search(r'(vendor|lib|library|node_modules)', b, re.IGNORECASE):
            score -= 2
            failures.append("Direct require of vendor/library files")

    # Circular dependency hard to detect statically, skip

    return max(1, score), failures


def score_i18n(fn: dict) -> tuple[int, list[str]]:
    """8. Internationalization"""
    failures = []
    score = 10

    b = body(fn)

    # Check if function outputs user-facing strings
    has_echo = bool(re.search(r'\becho\b', b))
    has_return_string = bool(re.search(r'\breturn\s+["\']', b))
    has_output = has_echo or has_return_string

    if not has_output:
        return 7, []  # N/A

    # Check for hardcoded English strings in echo without translation
    echo_strings = re.findall(r'\becho\s+["\']([^"\']{5,})["\']', b)
    # Filter out HTML attributes, structural tokens, CSS classes, etc.
    untranslated = [s for s in echo_strings if not any(
        kw in s.lower() for kw in ['<', '>', 'http', 'class=', 'id=', '{', '%',
                                    'selected', 'checked', 'disabled', ' =', '="',
                                    "='", 'href=', 'src=', 'type=', 'name=', 'value=']
    ) and re.search(r'[a-zA-Z ]{6,}', s)]  # must have meaningful words
    if untranslated:
        score -= 3
        failures.append(f"Hardcoded untranslated string(s) in echo: {untranslated[:2]}")

    # Check translation functions are used
    has_i18n = bool(re.search(
        r'\b(__\s*\(|_e\s*\(|esc_html__\s*\(|esc_html_e\s*\(|esc_attr__\s*\(|_n\s*\(|_x\s*\()',
        b
    ))

    # If echoing content but no i18n functions at all, mild deduction
    if has_echo and not has_i18n:
        # Check if it's purely HTML
        non_html_echo = [s for s in echo_strings if not re.match(r'^[\s<>/a-zA-Z0-9_=-]+$', s)]
        if non_html_echo:
            score -= 2

    # String concatenation with translated strings (instead of sprintf)
    concat_translation = re.search(
        r'(__\s*\([^)]+\))\s*\.\s*["\']', b
    )
    if concat_translation:
        score -= 1

    return max(1, score), failures


def score_accessibility(fn: dict) -> tuple[int, list[str]]:
    """9. Accessibility"""
    failures = []
    score = 10

    b = body(fn)

    # Check if function outputs HTML
    has_html_output = bool(re.search(r'\becho\s+.*<', b) or re.search(r'<(input|select|textarea|form)\b', b))

    if not has_html_output:
        return 7, []  # N/A

    # Form inputs without labels
    has_input = bool(re.search(r'<input\b', b, re.IGNORECASE))
    has_label = bool(re.search(r'<label\b', b, re.IGNORECASE))
    has_aria_label = bool(re.search(r'aria-label', b, re.IGNORECASE))

    if has_input and not has_label and not has_aria_label:
        score -= 3
        failures.append("Form input(s) without associated label or aria-label")

    # Images without alt
    img_without_alt = re.search(r'<img\b(?![^>]*\balt\s*=)[^>]*>', b, re.IGNORECASE)
    if img_without_alt:
        score -= 3
        failures.append("Image tag without alt attribute")

    return max(1, score), failures


# ─── Training tag detection ──────────────────────────────────────────────────

def detect_training_tags(fn: dict) -> list[str]:
    b = body(fn)
    tags = []

    if re.search(r'wp_verify_nonce|check_ajax_referer|check_admin_referer', b):
        tags.append("nonce-handling")
    if re.search(r'wp_nonce_url|wp_nonce_field|wp_create_nonce', b):
        tags.append("nonce-generation")
    if re.search(r'esc_html|esc_attr|esc_url|esc_js|wp_kses', b):
        tags.append("output-escaping")
    if re.search(r'\$wpdb->prepare', b):
        tags.append("prepared-statements")
    if re.search(r'\$wpdb->(query|get_results|get_row|get_var)', b):
        tags.append("direct-db-query")
    if re.search(r'new\s+WP_Query|get_posts\s*\(', b):
        tags.append("wp-query")
    if re.search(r'add_(action|filter)\s*\(', b):
        tags.append("hooks-actions-filters")
    if re.search(r'register_rest_route\s*\(', b):
        tags.append("rest-api")
    if re.search(r'wp_cache_get|wp_cache_set|get_transient|set_transient', b):
        tags.append("caching")
    if re.search(r'current_user_can\s*\(', b):
        tags.append("capability-checks")
    if re.search(r'register_post_type|register_taxonomy', b):
        tags.append("custom-post-types")
    if re.search(r'add_meta_box|get_post_meta|update_post_meta|delete_post_meta', b):
        tags.append("meta-boxes-post-meta")
    if re.search(r'get_option|update_option|add_option|delete_option', b):
        tags.append("options-api")
    if re.search(r'WP_List_Table|wp_list_table', b):
        tags.append("admin-list-table")
    if re.search(r'woocommerce|WC\(\)|WC_|wc_', b):
        tags.append("woocommerce-integration")
    if re.search(r'__\s*\(|_e\s*\(|esc_html__\s*\(|esc_attr__\s*\(|_n\s*\(', b):
        tags.append("i18n-translation-functions")
    if re.search(r'WP_Filesystem|WP_Filesystem\(\)', b):
        tags.append("wp-filesystem")
    if re.search(r'wp_enqueue_script|wp_enqueue_style|wp_register_script', b):
        tags.append("script-style-enqueue")
    if re.search(r'shortcode|add_shortcode', b):
        tags.append("shortcodes")
    if re.search(r'widget|WP_Widget', b):
        tags.append("widgets")
    if re.search(r'ajax|wp_ajax_', b, re.IGNORECASE):
        tags.append("ajax-handling")
    if re.search(r'sanitize_(text_field|email|url|int|key)', b):
        tags.append("data-sanitization")

    return list(set(tags))


# ─── Main judge function ─────────────────────────────────────────────────────

def judge_function(fn: dict) -> dict:
    """Judge a single function and return assessment dict."""
    fname = fn.get("function_name", "unknown")
    fpath = fn.get("source_file", "unknown")
    line_count = fn.get("line_count", 0)
    source_repo = fn.get("source_repo", "")
    quality_tier = fn.get("quality_tier", "assessed")

    # Skip: too short
    if line_count < 5:
        return None  # skip signal

    # Auto-PASS: WordPress core
    if source_repo in CORE_REPOS or quality_tier == "core":
        return {
            "function_name": fname,
            "file_path": fpath,
            "verdict": "PASS",
            "scores": {
                "wpcs_compliance": 10, "sql_safety": 10, "security": 10,
                "performance": 10, "wp_api_usage": 10, "code_quality": 10,
                "dependency_integrity": 10, "i18n": 10, "accessibility": 10
            },
            "critical_failures": [],
            "dependency_chain": fn.get("dependencies", []),
            "training_tags": detect_training_tags(fn),
            "notes": "Auto-PASS: WordPress core reference implementation."
        }

    # Score all dimensions
    all_failures = []

    wpcs_score, wpcs_failures = score_wpcs(fn)
    sql_score, sql_failures = score_sql(fn)
    sec_score, sec_failures = score_security(fn)
    perf_score, perf_failures = score_performance(fn)
    api_score, api_failures = score_wp_api(fn)
    cq_score, cq_failures = score_code_quality(fn)
    dep_score, dep_failures = score_dependency(fn)
    i18n_score, i18n_failures = score_i18n(fn)
    acc_score, acc_failures = score_accessibility(fn)

    all_failures = (
        wpcs_failures + sql_failures + sec_failures + perf_failures +
        api_failures + cq_failures + dep_failures + i18n_failures + acc_failures
    )

    scores = {
        "wpcs_compliance": wpcs_score,
        "sql_safety": sql_score,
        "security": sec_score,
        "performance": perf_score,
        "wp_api_usage": api_score,
        "code_quality": cq_score,
        "dependency_integrity": dep_score,
        "i18n": i18n_score,
        "accessibility": acc_score,
    }

    # N/A dimensions (score of 7 = not applicable, should not cause failure)
    na_dims = set()
    if i18n_score == 7 and not i18n_failures:
        na_dims.add("i18n")
    if acc_score == 7 and not acc_failures:
        na_dims.add("accessibility")

    # Determine verdict
    # Security auto-fail
    if sec_score < 5:
        verdict = "FAIL"
        notes = f"AUTO-FAIL: Security score {sec_score}/10. " + "; ".join(sec_failures[:2])
    elif any(v < 8 for k, v in scores.items() if k not in na_dims):
        verdict = "FAIL"
        failing_dims = [k for k, v in scores.items() if v < 8 and k not in na_dims]
        notes = f"FAIL: Dimensions below threshold: {', '.join(failing_dims)}. Issues: {'; '.join(all_failures[:3])}"
    else:
        verdict = "PASS"
        notes = f"Production-quality function from {fpath}."
        if detect_training_tags(fn):
            notes += f" Demonstrates: {', '.join(sorted(detect_training_tags(fn))[:4])}."

    training_tags = detect_training_tags(fn)

    return {
        "function_name": fname,
        "file_path": fpath,
        "verdict": verdict,
        "scores": scores,
        "critical_failures": all_failures,
        "dependency_chain": fn.get("dependencies", [])[:10],  # cap for size
        "training_tags": training_tags,
        "notes": notes,
    }


# ─── Process repos ───────────────────────────────────────────────────────────

def process_repo(repo: str) -> tuple[int, int, int]:
    """Returns (processed, passed, failed)."""
    extracted_path = os.path.join(EXTRACTED, f"{repo}.json")
    if not os.path.exists(extracted_path):
        print(f"  SKIP: {repo} - no extracted file")
        return 0, 0, 0

    with open(extracted_path) as f:
        functions = json.load(f)

    passed = []
    failed = []
    skipped = 0

    for fn in functions:
        assessment = judge_function(fn)
        if assessment is None:
            skipped += 1
            continue

        # Build output record
        record = dict(fn)  # copy all original fields
        record["assessment"] = assessment

        if assessment["verdict"] == "PASS":
            passed.append(record)
        else:
            failed.append(record)

    # Write output files
    passed_path = os.path.join(PASSED_DIR, f"{repo}.json")
    failed_path = os.path.join(FAILED_DIR, f"{repo}.json")

    with open(passed_path, "w") as f:
        json.dump(passed, f, indent=2)

    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)

    return len(functions), len(passed), len(failed)


def main():
    print("WordPress Function Judge")
    print("=" * 60)

    total_processed = 0
    total_passed = 0
    total_failed = 0

    for repo in REPOS:
        print(f"\nProcessing: {repo}")
        processed, passed, failed = process_repo(repo)
        print(f"  Processed: {processed}, Passed: {passed}, Failed: {failed}")
        total_processed += processed
        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_processed} processed, {total_passed} passed, {total_failed} failed")
    print(f"Pass rate: {total_passed/max(1,total_passed+total_failed)*100:.1f}%")


if __name__ == "__main__":
    main()
