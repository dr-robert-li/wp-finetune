#!/usr/bin/env python3
"""
WordPress Code Quality Judge
Applies the 9-dimension rubric from config/judge_system.md to extracted functions.
"""

import json
import re
import os
import sys

DEFAULT_REPOS = [
    "archive",
    "aruba-hispeed-cache",
    "athemes-starter-sites",
    "autoconvert-greeklish-permalinks",
    "auxin-elements",
    "backstage",
    "blocks-animation",
    "boldgrid-easy-seo",
    "bs-payone-woocommerce",
    "catch-themes-demo-import",
]

BASE = "/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output"
EXTRACTED_DIR = os.path.join(BASE, "extracted")
PASSED_DIR = os.path.join(BASE, "passed")
FAILED_DIR = os.path.join(BASE, "failed")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def body(fn):
    return fn.get("body", "") or ""

def docblock(fn):
    return fn.get("docblock", "") or ""

def full_text(fn):
    return (docblock(fn) + "\n" + body(fn)).lower()

def has_pattern(text, *patterns):
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def count_pattern(text, pattern):
    return len(re.findall(pattern, text))

def outputs_html(fn):
    b = body(fn)
    return bool(re.search(r'echo\s|print\s|printf\s*\(|<[a-zA-Z]', b))

def has_user_strings(fn):
    """Check if function has user-facing hardcoded strings."""
    b = body(fn)
    # echo/print with string literals that aren't already wrapped
    return bool(re.search(r'echo\s+["\']|print\s+["\']', b))

def uses_sql(fn):
    b = body(fn)
    return bool(re.search(r'\$wpdb\s*->|SELECT\s|INSERT\s|UPDATE\s|DELETE\s', b, re.IGNORECASE))

# ─── Dimension Scorers ────────────────────────────────────────────────────────

def score_wpcs(fn):
    """Dimension 1: WordPress Coding Standards"""
    score = 10
    notes = []
    b = body(fn)
    doc = docblock(fn)
    name = fn.get("function_name", "")

    # Check PHPDoc presence for non-trivial functions
    if fn.get("line_count", 0) >= 10 and not doc.strip():
        score -= 2
        notes.append("missing PHPDoc block")

    # Check for @param/@return in docblock
    if doc.strip() and fn.get("line_count", 0) >= 10:
        if "@param" not in doc and "@return" not in doc:
            # Allow if it's a void function with no clear params
            if re.search(r'\$[a-z]', b):  # has parameters used
                score -= 1
                notes.append("PHPDoc missing @param/@return")

    # Naming: should be snake_case or CamelCase for class methods
    if name and not re.match(r'^[a-z_][a-z0-9_]*$|^[A-Z][a-zA-Z0-9_]*$', name):
        score -= 1
        notes.append("non-standard function name")

    # camelCase function names (non-class) are a WP violation
    if name and re.match(r'^[a-z]+[A-Z]', name) and not fn.get("class_context"):
        score -= 1
        notes.append("camelCase function name (should be snake_case)")

    # Yoda conditions check (can't be comprehensive, but flag obvious violations)
    # e.g. if ($var == 'string') is non-Yoda
    if re.search(r'if\s*\(\s*\$[a-zA-Z_]+\s*==\s*[\'"]', b):
        score -= 1
        notes.append("non-Yoda condition detected")

    # var_dump/print_r debug statements
    if re.search(r'\bvar_dump\b|\bprint_r\b|\bdie\s*\(', b):
        score -= 2
        notes.append("debug statements in production code")

    # error_log in production paths
    if re.search(r'\berror_log\s*\(', b):
        score -= 1
        notes.append("error_log() in production path")

    return max(1, score), notes

def score_sql(fn):
    """Dimension 2: SQL Safety"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    if not uses_sql(fn):
        return 10, [], []

    # Check for prepared statements
    uses_prepare = bool(re.search(r'\$wpdb\s*->\s*prepare\s*\(', b))

    # Raw string interpolation in SQL - critical
    # Pattern: $wpdb->query/get_results/get_row/get_col/get_var with direct variable interpolation
    raw_query_pattern = re.search(
        r'\$wpdb\s*->\s*(query|get_results|get_row|get_col|get_var)\s*\(\s*["\'].*?\$[^"\']*["\']',
        b, re.DOTALL
    )

    # Also catch string concatenation in queries
    concat_in_query = re.search(
        r'\$wpdb\s*->\s*(query|get_results|get_row|get_col|get_var)\s*\([^)]*\.\s*\$',
        b, re.DOTALL
    )

    # Sprintf without prepare (common pattern that's OK if no user input, but risky)
    sprintf_query = re.search(r'sprintf\s*\([^)]*SELECT|sprintf\s*\([^)]*INSERT|sprintf\s*\([^)]*UPDATE', b, re.IGNORECASE)

    if raw_query_pattern or concat_in_query:
        # Check if it's actually a user input variable (more dangerous)
        user_input_vars = re.search(r'\$_(GET|POST|REQUEST|COOKIE|SERVER)', b)
        if user_input_vars:
            score = 1
            critical.append("SQL injection: user input concatenated directly into query")
        else:
            # Could be a fixed variable - still bad practice but less critical
            score -= 4
            notes.append("dynamic values concatenated into SQL without prepare()")
            if not uses_prepare:
                score -= 1

    # Hardcoded wp_ prefix instead of $wpdb->prefix
    if re.search(r'["\']wp_[a-z]', b) and not re.search(r'\$wpdb\s*->\s*prefix', b):
        score -= 2
        notes.append("hardcoded wp_ table prefix instead of $wpdb->prefix")

    # Using prepare() correctly is good
    if uses_prepare:
        # Check for correct placeholder types
        if re.search(r"prepare\s*\(\s*['\"].*?%[^sdF'\"]*", b):
            # may have wrong placeholders - minor deduction
            pass

    return max(1, score), notes, critical

def score_security(fn):
    """Dimension 3: Security"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # Check if this is a user-triggered form/AJAX handler
    is_ajax = bool(re.search(r'wp_ajax_|admin-ajax|wp_send_json|wp_die', b))
    is_form_handler = bool(re.search(r'\$_(POST|GET|REQUEST)\s*\[', b))
    # State-changing ops that are user-triggered (has form data or AJAX context)
    is_user_triggered_state_change = (is_ajax or is_form_handler) and bool(
        re.search(r'wp_insert_post|wp_update_post|wp_delete_post|update_option|delete_option|update_user_meta|delete_user_meta|wp_insert_user|wp_set_password', b)
    )

    # Nonce verification - only required when user triggers a state change
    has_nonce = bool(re.search(r'wp_verify_nonce|check_ajax_referer|check_admin_referer', b))

    if is_form_handler and not has_nonce:
        # Form handlers without nonce verification are a security issue
        score -= 2
        notes.append("form handler missing nonce verification")
        if is_ajax:
            score -= 1
            critical.append("AJAX handler accepts POST data without nonce verification")

    # Capability checks - required when user-triggered state changes happen
    has_cap_check = bool(re.search(r'current_user_can\s*\(', b))

    if is_user_triggered_state_change and not has_cap_check:
        score -= 2
        notes.append("user-triggered state-changing operation missing capability check")

    # Output escaping - check for direct echo of user input ($_GET/$_POST/$_REQUEST)
    has_escape = bool(re.search(r'esc_html|esc_attr|esc_url|wp_kses|esc_js|esc_textarea|absint\s*\(', b))

    # Critical: direct echo of superglobals without escaping
    if re.search(r'echo\s+\$_(GET|POST|REQUEST)|echo\s+.*\$_(GET|POST|REQUEST)', b):
        if not has_escape:
            score = 1
            critical.append("Direct echo of user input without escaping")
        else:
            # Has escaping but echoes user data - check if escape wraps the superglobal
            if not re.search(r'esc_html\s*\(\s*\$_(GET|POST|REQUEST)|esc_attr\s*\(\s*\$_(GET|POST|REQUEST)|esc_url\s*\(\s*\$_(GET|POST|REQUEST)', b):
                score -= 1
                notes.append("user input may not be properly escaped before echo")

    # Moderate: echo of variables when function is a form handler (higher risk context)
    elif is_form_handler and bool(re.search(r'echo\s+\$[a-zA-Z_]', b)) and not has_escape:
        score -= 1
        notes.append("echoing variables in form handler context without escaping")

    # extract() on user data
    if re.search(r'\bextract\s*\(\s*\$_(GET|POST|REQUEST)', b):
        score = 1
        critical.append("extract() used on user-supplied data")

    # eval() usage
    if re.search(r'\beval\s*\(', b):
        score = max(1, score - 4)
        critical.append("eval() usage detected")

    # file operations
    if re.search(r'\bfopen\s*\(|\bfile_put_contents\s*\(|\bfwrite\s*\(', b):
        if not re.search(r'WP_Filesystem|global\s+\$wp_filesystem', b):
            score -= 1
            notes.append("direct file operations instead of WP_Filesystem")

    # include/require with variables
    if re.search(r'\b(include|require)(_once)?\s*\(\s*\$', b):
        if re.search(r'\$_(GET|POST|REQUEST)', b):
            score = 1
            critical.append("remote file inclusion via user-controlled variable")
        else:
            score -= 1
            notes.append("include/require with variable path (potential LFI)")

    # sanitization of inputs
    if is_form_handler:
        has_sanitize = bool(re.search(r'sanitize_text_field|sanitize_email|absint\s*\(|intval\s*\(|floatval\s*\(|sanitize_key|sanitize_textarea_field|sanitize_file_name|wp_kses', b))
        if not has_sanitize and not has_nonce:
            score -= 1
            notes.append("form data used without sanitization")

    return max(1, score), notes, critical

def score_performance(fn):
    """Dimension 4: Performance"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # SELECT * check
    if re.search(r'SELECT\s+\*', b, re.IGNORECASE):
        score -= 2
        notes.append("SELECT * instead of specific columns")

    # Queries in loops
    has_loop = bool(re.search(r'\bforeach\b|\bfor\s*\(|\bwhile\s*\(', b))
    has_db_in_loop = has_loop and bool(re.search(r'\$wpdb\s*->|new\s+WP_Query|get_posts\s*\(|get_post_meta\s*\(', b))

    if has_db_in_loop:
        # More specific check: DB call inside loop body
        # Simple heuristic: if both are present, flag it
        loop_match = re.search(r'(foreach|for\s*\(|while\s*\()[^{]*\{(.*?)(?:\}[^}]|\Z)', b, re.DOTALL)
        if loop_match and re.search(r'\$wpdb\s*->|new\s+WP_Query|get_posts\s*\(|get_post_meta\s*\(', loop_match.group(2) if loop_match.lastindex >= 2 else ""):
            score -= 3
            critical.append("Database query inside loop (N+1 pattern)")
        else:
            score -= 1
            notes.append("potential query in loop (verify manually)")

    # Transient/cache usage for expensive ops
    has_transient = bool(re.search(r'get_transient|set_transient|wp_cache_get|wp_cache_set', b))
    has_expensive_op = bool(re.search(r'\$wpdb\s*->|new\s+WP_Query|get_posts\s*\(', b))

    # If there's an expensive DB op with no caching, minor deduction (not all need caching)
    # Only deduct if the function looks like it's called frequently
    if has_expensive_op and not has_transient:
        # Check if it's hooked to something frequently called
        pass  # Don't penalize - not all queries need caching

    # wp_remote_get/post without caching
    if re.search(r'wp_remote_get|wp_remote_post|wp_remote_request', b):
        if not has_transient:
            score -= 1
            notes.append("HTTP request without transient caching")

    # Unbounded queries
    if re.search(r"'posts_per_page'\s*=>\s*-1|'numberposts'\s*=>\s*-1", b):
        score -= 2
        notes.append("unbounded query (posts_per_page: -1) without limit")

    return max(1, score), notes, critical

def score_wp_api(fn):
    """Dimension 5: WordPress API Usage"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # REST endpoint without permission_callback
    rest_route = re.search(r'register_rest_route\s*\(', b)
    if rest_route:
        if not re.search(r'permission_callback', b):
            score -= 3
            critical.append("REST route registered without permission_callback")
        elif re.search(r"permission_callback['\"]?\s*=>\s*'__return_true'|permission_callback['\"]?\s*=>\s*\"__return_true\"", b):
            score -= 1
            notes.append("REST route uses __return_true as permission_callback (open to all)")

    # Using raw SQL instead of WP_Query for post queries
    if re.search(r"SELECT.*FROM.*wp_posts|SELECT.*FROM.*\$wpdb->posts", b, re.IGNORECASE):
        if not re.search(r'GROUP BY|JOIN|HAVING', b, re.IGNORECASE):
            score -= 2
            notes.append("raw SQL to query posts instead of WP_Query")

    # Using deprecated functions
    deprecated = [
        (r'\bquery_posts\s*\(', "query_posts() is deprecated"),
        (r'\bthe_post_thumbnail_url\s*\(', "the_post_thumbnail_url() - prefer get_the_post_thumbnail_url()"),
        (r'\bget_currentuserinfo\s*\(', "get_currentuserinfo() is deprecated, use wp_get_current_user()"),
        (r'\badd_option\s*\(.*autoload', "add_option with autoload may be suboptimal"),
    ]
    for pattern, msg in deprecated:
        if re.search(pattern, b):
            score -= 1
            notes.append(msg)

    # Hook argument count mismatch is hard to check statically - skip

    # Options API: storing large objects
    if re.search(r'update_option\s*\(.*serialize|add_option\s*\(.*serialize', b):
        score -= 1
        notes.append("serializing data for options storage (consider post meta or transients)")

    return max(1, score), notes, critical

def score_code_quality(fn):
    """Dimension 6: Code Quality"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # var_dump / print_r / die in production
    if re.search(r'\bvar_dump\b|\bprint_r\b', b):
        score -= 3
        critical.append("debug output (var_dump/print_r) in production code")

    # Hardcoded die() without context
    if re.search(r'\bdie\s*\(\s*["\']', b):
        score -= 1
        notes.append("hardcoded die() message instead of wp_die()")

    # TODO/FIXME/HACK comments
    if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', b, re.IGNORECASE):
        score -= 1
        notes.append("unresolved TODO/FIXME comments in code")

    # Empty catch blocks
    if re.search(r'catch\s*\([^)]*\)\s*\{\s*\}', b):
        score -= 2
        critical.append("empty catch block swallows errors silently")

    # Function length - very long functions lose a point
    lines = fn.get("line_count", 0)
    if lines > 150:
        score -= 1
        notes.append(f"very long function ({lines} lines) - consider refactoring")

    # Commented-out code blocks
    commented_code = len(re.findall(r'//\s*\$[a-zA-Z_]|//\s*[a-zA-Z_]+\s*\(', b))
    if commented_code > 5:
        score -= 1
        notes.append("large blocks of commented-out code")

    # @codingStandardsIgnore overuse
    if len(re.findall(r'phpcs:ignore|@codingStandardsIgnore', b)) > 3:
        score -= 1
        notes.append("excessive PHPCS ignore directives")

    # Null/empty checks before operations
    # If function uses an array/object, check it's validated
    if re.search(r'foreach\s*\(\s*\$[a-zA-Z_]+\s', b):
        if not re.search(r'empty\s*\(|is_array\s*\(|isset\s*\(|!\s*\$[a-zA-Z_]', b):
            score -= 1
            notes.append("foreach without empty/is_array check")

    return max(1, score), notes, critical

def score_dependency(fn):
    """Dimension 7: Dependency Chain Integrity"""
    score = 10  # Default - assume OK unless we detect violations
    notes = []
    b = body(fn)

    # Direct require/include of vendor files (not through WP patterns)
    if re.search(r'\b(require|include)(_once)?\s*\([^)]*vendor/', b):
        score -= 2
        notes.append("direct require of vendor files without autoloader")

    # Circular dependency indicators (hard to detect statically)

    # Using globals without declaring them
    if re.search(r'\$wpdb|\$wp_query|\$post\b|\$current_user', b):
        if not re.search(r'global\s+\$wpdb|global\s+\$wp_query|global\s+\$post\b|global\s+\$current_user', b):
            # Check if it's a class method context (where globals might not need declaration)
            if not fn.get("class_context"):
                # Only flag if actually using global vars outside class context
                if re.search(r'^\s*\$wpdb->', b, re.MULTILINE):
                    score -= 1
                    notes.append("using $wpdb without global declaration")

    return max(1, score), notes

def score_i18n(fn):
    """Dimension 8: Internationalization"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # Check if function outputs user-facing text (actual readable text, not just HTML structure)
    # A function has user-facing strings if it echoes readable English words (not just tags/attributes)
    has_echo = bool(re.search(r'\becho\b|\bprint\b', b))

    if not has_echo:
        return 10, [], []  # N/A - no echo output

    # Check for translation function usage
    has_i18n = bool(re.search(r'\b__\s*\(|\b_e\s*\(|\besc_html__\s*\(|\besc_html_e\s*\(|\besc_attr__\s*\(|\b_n\s*\(|\b_x\s*\(|\b_nx\s*\(', b))

    # Find hardcoded English text strings (multi-word readable text, not HTML tags/attributes/CSS)
    # Match strings with actual English words (letters, spaces, punctuation) - not just HTML/CSS markup
    hardcoded_echo = re.findall(r'echo\s+["\']([A-Za-z][a-z]{2,}\s+[A-Za-z][a-z]{2,}[^"\']*)["\']', b)

    # Filter out things that look like HTML attribute values, CSS classes, PHP class names
    real_hardcoded = [s for s in hardcoded_echo
                      if not re.match(r'^[a-z-_]+$', s)  # pure CSS class
                      and not re.match(r'^[A-Z][a-zA-Z_]+$', s)  # class name
                      and len(s) > 5]

    if real_hardcoded and not has_i18n:
        score -= 3
        critical.append(f"hardcoded user-facing strings without translation wrappers: {real_hardcoded[:2]}")
    elif real_hardcoded and has_i18n:
        # Mixed - some translated, some not
        score -= 1
        notes.append("some strings may not be wrapped in translation functions")

    # Concatenation with translated strings (should use sprintf)
    if re.search(r'__\s*\([^)]+\)\s*\.\s*["\']|["\'].*\.\s*__\s*\(', b):
        score -= 1
        notes.append("string concatenation with translated strings (use sprintf instead)")

    return max(1, score), notes, critical

def score_accessibility(fn):
    """Dimension 9: Accessibility"""
    score = 10
    notes = []
    critical = []
    b = body(fn)

    # Only relevant if outputting HTML with form elements or images
    # If no form elements or images, accessibility concerns are minimal
    has_form_or_img = bool(re.search(r'<input\b|<select\b|<textarea\b|<img\b|<button\b', b, re.IGNORECASE))
    if not has_form_or_img:
        return 10, [], []  # N/A - no form elements or images that need accessibility treatment

    # Check for form inputs without labels
    has_input = bool(re.search(r'<input\b', b, re.IGNORECASE))
    has_label = bool(re.search(r'<label\b|for=["\']', b, re.IGNORECASE))
    has_aria_label = bool(re.search(r'aria-label|aria-labelledby', b, re.IGNORECASE))

    if has_input and not has_label and not has_aria_label:
        score -= 2
        critical.append("form input without associated label or aria-label")

    # Images without alt
    has_img = bool(re.search(r'<img\b', b, re.IGNORECASE))
    has_alt = bool(re.search(r'\balt=["\']', b, re.IGNORECASE))

    if has_img and not has_alt:
        score -= 2
        critical.append("img element missing alt attribute")

    # Interactive elements
    has_button = bool(re.search(r'<button\b|<a\b', b, re.IGNORECASE))
    # Can't fully verify keyboard accessibility statically

    # Screen reader text for icon-only buttons
    if re.search(r'dashicons|fa-', b) and has_button:
        if not re.search(r'screen-reader-text|aria-label|aria-hidden', b, re.IGNORECASE):
            score -= 1
            notes.append("icon-only interactive element may lack screen reader text")

    return max(1, score), notes, critical


# ─── Main Judge ───────────────────────────────────────────────────────────────

def judge_function(fn):
    """Apply all 9 dimensions and return assessment dict."""

    # Skip short functions
    if fn.get("line_count", 0) < 5:
        return None

    # Auto-pass core
    if fn.get("quality_tier") == "core":
        return {
            "verdict": "PASS",
            "scores": {d: 10 for d in ["wpcs_compliance","sql_safety","security","performance","wp_api_usage","code_quality","dependency_integrity","i18n","accessibility"]},
            "critical_failures": [],
            "dependency_chain": [],
            "training_tags": ["wordpress-core"],
            "notes": "WordPress core - auto-passed"
        }

    # Score all dimensions
    wpcs_score, wpcs_notes = score_wpcs(fn)
    sql_score, sql_notes, sql_critical = score_sql(fn)
    sec_score, sec_notes, sec_critical = score_security(fn)
    perf_score, perf_notes, perf_critical = score_performance(fn)
    api_score, api_notes, api_critical = score_wp_api(fn)
    cq_score, cq_notes, cq_critical = score_code_quality(fn)
    dep_score, dep_notes = score_dependency(fn)
    i18n_score, i18n_notes, i18n_critical = score_i18n(fn)
    acc_score, acc_notes, acc_critical = score_accessibility(fn)

    all_critical = sql_critical + sec_critical + perf_critical + api_critical + cq_critical + i18n_critical + acc_critical

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

    # Determine verdict
    verdict = "PASS"
    fail_reasons = []

    # Security auto-fail
    if sec_score < 5:
        verdict = "FAIL"
        fail_reasons.append(f"security score {sec_score} < 5 (auto-fail)")

    # All dimensions must be >= 8
    for dim, s in scores.items():
        if s < 8:
            verdict = "FAIL"
            fail_reasons.append(f"{dim} score {s} < 8")

    # Any critical failure = FAIL
    if all_critical:
        verdict = "FAIL"

    # Generate training tags
    training_tags = derive_training_tags(fn)

    # Build notes
    all_notes = wpcs_notes + sql_notes + sec_notes + perf_notes + api_notes + cq_notes + dep_notes + i18n_notes + acc_notes
    if verdict == "PASS":
        note_str = f"Production-quality WordPress code. Demonstrates: {', '.join(training_tags) if training_tags else 'general WordPress patterns'}."
    else:
        parts = []
        if fail_reasons:
            parts.append("Low scores: " + "; ".join(fail_reasons[:3]))
        if all_critical:
            parts.append("Critical: " + "; ".join(all_critical[:2]))
        note_str = "FAIL - " + ". ".join(parts) if parts else "FAIL"

    # Dependency chain from function's dependencies list
    deps = fn.get("dependencies", [])
    custom_deps = [d for d in (deps or []) if not d.startswith(('wp_', 'get_', 'the_', 'is_', 'has_', 'add_', 'remove_', 'do_', 'apply_', 'register_', 'update_', 'delete_', 'sanitize_', 'esc_', 'check_', 'current_'))]

    fn_name = fn.get("function_name", "unknown")
    file_path = fn.get("source_file", "")
    repo = fn.get("source_repo", "")
    idx = fn.get("_idx", 0)
    custom_id = f"{repo}_{idx}_{fn_name}"

    return {
        "function_name": fn_name,
        "file_path": file_path,
        "verdict": verdict,
        "scores": scores,
        "critical_failures": all_critical,
        "dependency_chain": custom_deps[:10],
        "training_tags": training_tags,
        "notes": note_str,
        "_custom_id": custom_id,
    }


def derive_training_tags(fn):
    """Infer relevant WordPress training tags from function content."""
    b = body(fn)
    tags = []

    tag_patterns = [
        (r'add_action|add_filter|do_action|apply_filters', "hooks"),
        (r'wp_enqueue_script|wp_enqueue_style|wp_register_script|wp_register_style', "asset-enqueuing"),
        (r'register_block_type|register_block_style|block_editor', "block-registration"),
        (r'wp_localize_script', "script-localization"),
        (r'\$wpdb\s*->', "wpdb-queries"),
        (r'\$wpdb\s*->\s*prepare', "sql-prepare"),
        (r'new\s+WP_Query|WP_Query\s*\(', "wp-query"),
        (r'get_posts\s*\(|query_posts\s*\(', "post-query"),
        (r'register_post_type\s*\(', "custom-post-types"),
        (r'register_taxonomy\s*\(', "taxonomies"),
        (r'register_rest_route\s*\(', "rest-api"),
        (r'wp_verify_nonce|check_ajax_referer|check_admin_referer', "nonce-verification"),
        (r'current_user_can\s*\(', "capability-checks"),
        (r'esc_html|esc_attr|esc_url|wp_kses|esc_js', "output-escaping"),
        (r'sanitize_text_field|sanitize_email|absint\s*\(|wp_kses', "input-sanitization"),
        (r'get_transient|set_transient|delete_transient', "transients"),
        (r'wp_cache_get|wp_cache_set', "object-cache"),
        (r'add_settings_field|register_setting|settings_fields|do_settings_sections', "settings-api"),
        (r'add_meta_box|get_post_meta|update_post_meta|delete_post_meta', "meta-boxes"),
        (r'add_submenu_page|add_menu_page|add_options_page|add_theme_page', "admin-menus"),
        (r'wp_ajax_|wp_doing_ajax\s*\(', "ajax-handlers"),
        (r'wp_mail\s*\(', "email"),
        (r'WP_Filesystem|global\s+\$wp_filesystem', "filesystem-api"),
        (r'wp_remote_get|wp_remote_post', "http-api"),
        (r'WooCommerce|woocommerce|WC\(\)', "woocommerce"),
        (r'wc_get_order|WC_Order|get_order', "woocommerce-orders"),
        (r'__\s*\(|_e\s*\(|esc_html__', "i18n"),
        (r'screen-reader-text|aria-label|aria-describedby', "accessibility"),
        (r'wp_nonce_field|nonce_field', "nonce-field"),
        (r'get_option|update_option|add_option|delete_option', "options-api"),
        (r'WP_Error|is_wp_error', "error-handling"),
        (r'update_user_meta|get_user_meta|delete_user_meta', "user-meta"),
        (r'wp_schedule_event|wp_next_scheduled|wp_unschedule_event', "cron"),
        (r'shortcode_atts|add_shortcode', "shortcodes"),
        (r'get_template_part|locate_template|load_template', "template-hierarchy"),
        (r'widgets_init|register_sidebar|WP_Widget', "widgets"),
        (r'walker_|Walker_|extends\s+Walker', "walker-classes"),
        (r'plugin_action_links|plugin_row_meta', "plugin-admin"),
        (r'activation_hook|deactivation_hook|register_activation_hook', "plugin-lifecycle"),
        (r'sprintf\s*\(.*__\s*\(|printf\s*\(.*__\s*\(', "i18n-sprintf"),
        (r'wp_insert_post|wp_update_post|wp_delete_post', "post-crud"),
    ]

    for pattern, tag in tag_patterns:
        if re.search(pattern, b):
            tags.append(tag)

    # Deduplicate
    seen = set()
    unique_tags = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)

    return unique_tags


# ─── Runner ───────────────────────────────────────────────────────────────────

def process_repo(repo):
    extracted_path = os.path.join(EXTRACTED_DIR, f"{repo}.json")
    if not os.path.exists(extracted_path):
        print(f"  SKIP: {repo}.json not found")
        return 0, 0, 0

    with open(extracted_path) as f:
        functions = json.load(f)

    passed = []
    failed = []
    skipped = 0

    for _idx, fn in enumerate(functions):
        fn["_idx"] = _idx
        assessment = judge_function(fn)
        if assessment is None:
            skipped += 1
            continue

        # Merge assessment into function record
        fn_out = dict(fn)
        fn_out["assessment"] = assessment
        fn_out["training_tags"] = assessment["training_tags"]

        if assessment["verdict"] == "PASS":
            passed.append(fn_out)
        else:
            failed.append(fn_out)

    # Write outputs
    passed_path = os.path.join(PASSED_DIR, f"{repo}.json")
    failed_path = os.path.join(FAILED_DIR, f"{repo}.json")

    with open(passed_path, "w") as f:
        json.dump(passed, f, indent=2)

    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)

    return len(passed), len(failed), skipped


def main():
    # Use CLI args if provided, else use DEFAULT_REPOS
    repos = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_REPOS

    print("WordPress Code Quality Judge")
    print("=" * 60)
    print(f"Repos to judge: {repos}")

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for repo in repos:
        print(f"\nProcessing: {repo}")
        p, f, s = process_repo(repo)
        print(f"  PASS: {p}  FAIL: {f}  SKIP (< 5 lines): {s}")
        total_pass += p
        total_fail += f
        total_skip += s

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print(f"Pass rate: {total_pass/(total_pass+total_fail)*100:.1f}%" if (total_pass+total_fail) > 0 else "")


if __name__ == "__main__":
    main()
