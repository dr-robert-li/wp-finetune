#!/usr/bin/env python3
"""
WordPress Code Quality Judge
Applies rubric from config/judge_system.md to all functions in target repos.
Writes passed/{repo}.json and failed/{repo}.json
"""
import json
import re
import os
import sys

BASE = "/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output"
EXTRACTED = f"{BASE}/extracted"
PASSED_DIR = f"{BASE}/passed"
FAILED_DIR = f"{BASE}/failed"

REPOS = [
    "integromat-connector",
    "jet-style-manager",
    "jetpack",
    "jupiterx-core",
    "kafkai",
    "kelkoogroup-sales-tracking",
    "kliken-ads-pixel-for-meta",
    "kliken-marketing-for-google",
    "komoju-japanese-payments",
    "learnpress-bbpress",
    "learnpress-buddypress",
    "learnpress-course-review",
    "learnpress-wishlist",
    "list-category-posts",
    "liveblog",
    "lmsace-connect",
    "loginradius-for-wordpress",
    "mailchimp",
    "masvideos",
    "mautic-for-fluent-forms",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def body(fn):
    return fn.get("body", "") or ""

def docblock(fn):
    return fn.get("docblock", "") or ""

def full_text(fn):
    return docblock(fn) + "\n" + body(fn)

def has(text, pattern, flags=re.IGNORECASE):
    return bool(re.search(pattern, text, flags))

def count_matches(text, pattern):
    return len(re.findall(pattern, text))


# ── WPCS (dimension 1) ───────────────────────────────────────────────────────

def score_wpcs(fn):
    b = body(fn)
    d = docblock(fn)
    score = 10
    notes = []

    # PHPDoc check
    has_doc = bool(d and "@param" in d or "@return" in d or "@since" in d)
    has_any_doc = bool(d and len(d.strip()) > 5)
    # Methods/functions should have docblocks
    if not has_any_doc:
        score -= 2
        notes.append("missing docblock")
    elif not has_doc:
        score -= 1
        notes.append("docblock missing @param/@return/@since")

    # Yoda conditions: check for obvious non-Yoda (var == 'string' patterns)
    non_yoda = re.findall(r'\$\w+\s*==\s*[\'"]', b)
    if non_yoda:
        score -= 1
        notes.append("non-Yoda conditions")

    # camelCase function names (should be snake_case for WP)
    fname = fn.get("function_name", "")
    if re.search(r'[a-z][A-Z]', fname) and "::" not in fname and "->" not in fname:
        # camelCase method in non-class context
        if not fn.get("class_context"):
            score -= 1
            notes.append("camelCase function name")

    # debug leftovers
    if has(b, r'\b(var_dump|print_r|error_log|console\.log)\s*\('):
        score -= 2
        notes.append("debug statement in code")

    # hardcoded wp_ prefix in SQL (handled in sql_safety but note here too)
    if re.search(r'["\']wp_\w+["\']', b):
        score -= 1
        notes.append("hardcoded wp_ table prefix")

    return max(1, score), notes


# ── SQL Safety (dimension 2) ─────────────────────────────────────────────────

def score_sql_safety(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    sql_pats = fn.get("sql_patterns", []) or []
    has_sql = bool(sql_pats) or has(b, r'\$wpdb\s*->\s*(query|get_results|get_row|get_var|get_col)\s*\(')

    if not has_sql:
        return 10, [], []

    # Check for $wpdb->prepare usage
    has_prepare = has(b, r'\$wpdb\s*->\s*prepare\s*\(')
    # Dynamic values concatenated directly
    heredoc_sql = has(b, r'(?:query|get_results|get_row|get_var)\s*\(\s*<<<')

    # Detect dangerous SQL interpolation: variables that are NOT $wpdb->property (table names)
    # Pattern: wpdb query call with a string that contains $variable (not $wpdb->...)
    # We look for queries where interpolated values come from non-wpdb sources
    def has_dangerous_sql_interp(body_text):
        # Find SQL query strings
        sql_calls = re.finditer(
            r'\$wpdb\s*->\s*(?:query|get_results|get_row|get_var|get_col)\s*\(\s*"([^"]*)"',
            body_text, re.DOTALL
        )
        for m in sql_calls:
            sql_str = m.group(1)
            # Find interpolated variables in the SQL string
            vars_in_sql = re.findall(r'\{?\$(?!wpdb\b)(\w+)', sql_str)
            if vars_in_sql:
                return True
        return False

    if has_dangerous_sql_interp(b):
        score = 2
        critical.append("SQL query with direct variable interpolation (SQL injection risk)")
    elif has_sql and not has_prepare:
        # Could be using static SQL or other pattern
        static_only = not re.search(r'\$(?!wpdb)\w+', b[b.find('$wpdb'):b.find('$wpdb')+300] if '$wpdb' in b else '')
        if not static_only:
            score -= 3
            notes.append("SQL present without $wpdb->prepare()")

    # Hardcoded wp_ prefix
    if re.search(r'["\']wp_\w+["\']', b):
        score -= 1
        notes.append("hardcoded wp_ table prefix instead of $wpdb->prefix")

    return max(1, score), notes, critical


# ── Security (dimension 3) ───────────────────────────────────────────────────

NONCE_FUNCS = r'(wp_verify_nonce|check_ajax_referer|check_admin_referer)'
CAP_FUNCS = r'current_user_can\s*\('
OUTPUT_ESC = r'(esc_html|esc_attr|esc_url|esc_js|wp_kses|esc_textarea|absint|intval|floatval|sanitize_\w+)\s*\('
NONCE_CREATE = r'(wp_nonce_field|wp_create_nonce|nonce_field)\s*\('

# State-changing handlers (POST/AJAX handlers that need nonce)
AJAX_HANDLER = r'add_action\s*\(\s*[\'"]wp_ajax_'
POST_HANDLER = r'\$_POST\s*\[|\$_REQUEST\s*\['
FILE_OPS_DIRECT = r'\b(fopen|fwrite|file_put_contents|file_get_contents|unlink|mkdir|rmdir)\s*\('
EVAL_USE = r'\beval\s*\('
EXTRACT_USE = r'\bextract\s*\('

def score_security(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    # eval() is always bad
    if has(b, EVAL_USE):
        score = 1
        critical.append("uses eval()")

    # extract() on untrusted data
    if has(b, EXTRACT_USE):
        if has(b, r'extract\s*\(\s*\$_(POST|GET|REQUEST|COOKIE)'):
            score = min(score, 2)
            critical.append("extract() on superglobal data")
        else:
            score -= 2
            notes.append("uses extract() - risky pattern")

    # Direct file operations (not using WP_Filesystem)
    if has(b, FILE_OPS_DIRECT):
        if not has(b, r'WP_Filesystem|global\s+\$wp_filesystem'):
            score -= 2
            notes.append("direct file operations without WP_Filesystem")

    # Check for unescaped output of user data
    # echo/print with $_POST/$_GET without escaping
    unsafe_echo = re.findall(r'(?:echo|print)\s+(?!esc_|wp_kses).*?\$_(POST|GET|REQUEST|COOKIE)', b)
    if unsafe_echo:
        score = min(score, 3)
        critical.append("unescaped output of user-controlled superglobal data")

    # HTML output: check if function echoes/prints HTML
    echoes_html = has(b, r'(?:echo|print|printf|vprintf)\s*[(\s]') or has(b, r'<\s*(?:input|form|select|textarea|button|a\s|div|span|p\b|ul|ol|li|table|tr|td)\b')

    # Check for output without escaping
    if echoes_html:
        # Look for echo of variables without escaping
        bare_echo = re.findall(r'echo\s+\$(?!_SERVER|_SESSION)(\w+)', b)
        if bare_echo:
            score -= 1
            notes.append(f"unescaped echo of variable(s): {bare_echo[:3]}")

    # AJAX/POST handler nonce check
    # If function is a POST/AJAX handler, should verify nonce
    is_state_changer = (
        has(b, POST_HANDLER) and
        has(b, r'(wp_insert|wp_update|wp_delete|update_option|delete_option|update_post_meta|delete_post_meta|wp_insert_post|wp_update_post)')
    )
    if is_state_changer:
        if not has(b, NONCE_FUNCS):
            score -= 2
            notes.append("state-changing function missing nonce verification")

    # Capability check for admin operations
    if has(b, r'(update_option|delete_option|wp_insert_post|wp_delete_post)\s*\('):
        if not has(b, CAP_FUNCS) and not has(b, NONCE_FUNCS):
            score -= 1
            notes.append("administrative operation without capability check")

    return max(1, score), notes, critical


# ── Performance (dimension 4) ────────────────────────────────────────────────

def score_performance(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    # SELECT * detection
    select_star = re.findall(r'SELECT\s+\*', b, re.IGNORECASE)
    if select_star:
        score -= 1
        notes.append("SELECT * query")

    # Query in a loop
    # Look for wpdb calls inside foreach/while/for
    loop_query = re.search(
        r'(?:foreach|while|for)\s*\([^{]*\{[^}]*\$wpdb\s*->',
        b, re.DOTALL
    )
    if loop_query:
        score -= 3
        critical.append("database query inside a loop (N+1 pattern)")

    # Unbounded queries (get_results without limit and with large tables)
    if has(b, r'\$wpdb\s*->\s*get_results\s*\(') and not has(b, r'LIMIT\s+\d+'):
        score -= 1
        notes.append("unbounded query - no LIMIT clause")

    # WP_Query in a loop
    wp_query_loop = re.search(
        r'(?:foreach|while|for)\s*\([^{]*\{[^}]*new\s+WP_Query',
        b, re.DOTALL
    )
    if wp_query_loop:
        score -= 3
        critical.append("WP_Query inside a loop")

    # get_post_meta inside a loop
    meta_loop = re.search(
        r'(?:foreach|while|for)\s*\([^{]*\{[^}]*get_(?:post|term|user)_meta\s*\(',
        b, re.DOTALL
    )
    if meta_loop:
        score -= 2
        notes.append("get_*_meta() inside a loop")

    # Transient/cache usage (positive signal - no deduction)

    return max(1, score), notes, critical


# ── WP API Usage (dimension 5) ───────────────────────────────────────────────

def score_wp_api(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    # REST route without permission_callback
    rest_route = re.findall(r'register_rest_route\s*\(', b)
    if rest_route:
        if not has(b, r'permission_callback'):
            score = 2
            critical.append("register_rest_route() missing permission_callback")

    # Raw SQL for post queries
    if has(b, r'\$wpdb.*?SELECT.*?FROM.*?wp_posts', re.IGNORECASE):
        if not has(b, r'WP_Query|get_posts\s*\(|get_post\s*\('):
            score -= 2
            notes.append("raw SQL for post queries instead of WP_Query")

    # add_action/add_filter argument count
    # Basic check: if hooks_used contains add_filter, verify arg counts
    hooks = fn.get("hooks_used", []) or []

    # Deprecated functions
    deprecated = [
        'query_posts', 'wp_get_sites', 'get_currentuserinfo', 'wp_specialchars',
        'the_content_rss', 'get_bloginfo_rss', 'comments_rss_link',
        'wp_convert_bytes_to_hr', 'wp_nav_menu_locations_meta_box',
        'get_user_metavalues', 'dbDelta',  # dbDelta is fine, skip
    ]
    deprecated.remove('dbDelta')
    for dep in deprecated:
        if has(b, r'\b' + dep + r'\s*\('):
            score -= 2
            notes.append(f"uses deprecated function: {dep}")
            break

    # Using $_GET/$_POST directly without sanitization for WP operations
    if has(b, r'\$_(GET|POST|REQUEST)\s*\[') and not has(b, r'sanitize_\w+\s*\(|absint\s*\(|intval\s*\(|wp_unslash\s*\('):
        score -= 1
        notes.append("unsanitized superglobal input")

    return max(1, score), notes, critical


# ── Code Quality (dimension 6) ───────────────────────────────────────────────

def score_code_quality(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    lines = b.strip().split('\n')
    line_count = fn.get("line_count", len(lines))

    # debug statements
    if has(b, r'\b(var_dump|print_r|error_log|die\s*\(\s*[\'"]debug|var_export)\s*\('):
        score -= 2
        notes.append("debug/development statements present")

    # Very long function (complexity signal)
    if line_count > 150:
        score -= 2
        notes.append(f"function very long ({line_count} lines) - may violate single responsibility")
    elif line_count > 80:
        score -= 1
        notes.append(f"function long ({line_count} lines)")

    # Commented-out code blocks
    commented_code = re.findall(r'//\s*\$\w+|//\s*(?:echo|return|if|foreach|while)', b)
    if len(commented_code) > 2:
        score -= 1
        notes.append("commented-out code blocks")

    # TODO/FIXME/HACK in body
    if has(b, r'//\s*(TODO|FIXME|HACK|XXX)\b'):
        score -= 1
        notes.append("TODO/FIXME/HACK comments in production code")

    # Empty catch blocks
    if has(b, r'catch\s*\([^)]+\)\s*\{\s*\}'):
        score -= 2
        critical.append("empty catch block - swallows errors silently")

    # Silent error suppression
    suppressed = re.findall(r'@\$\w+|@\w+\s*\(', b)
    if suppressed:
        score -= 1
        notes.append("uses @ error suppression operator")

    return max(1, score), notes, critical


# ── Dependency Integrity (dimension 7) ───────────────────────────────────────

def score_dependency(fn):
    b = body(fn)
    score = 8  # Default N/A-ish since we can't fully trace
    notes = []

    # Direct require/include of vendor files (not autoloader)
    if has(b, r'require(?:_once)?\s*\(\s*[\'"](?!ABSPATH|dirname|plugin_dir_path)'):
        score -= 1
        notes.append("direct require of file path (should use autoloader)")

    # global keyword usage (excluding $wpdb, $post, $wp_query which are standard WP globals)
    WP_STANDARD_GLOBALS = {'wpdb', 'post', 'wp_query', 'wp_filesystem', 'current_user', 'pagenow', 'wp', 'wp_rewrite'}
    globals_used = re.findall(r'\bglobal\s+\$(\w+)', b)
    non_wp_globals = [g for g in globals_used if g.lower() not in WP_STANDARD_GLOBALS]
    if non_wp_globals:
        score -= 1
        notes.append(f"uses non-standard global variables: {non_wp_globals[:3]}")

    return max(1, score), notes


# ── i18n (dimension 8) ───────────────────────────────────────────────────────

I18N_FUNCS = r'(?:__|_e|esc_html__|esc_html_e|esc_attr__|esc_attr_e|_n|_x|_nx|_ex|printf|sprintf)\s*\('
I18N_OUTPUT = r'(?:echo|print|printf|return)\s+[\'"][A-Z][a-zA-Z\s,\.!?:;\-]+[\'"]'

def score_i18n(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    # Check if function has user-facing output
    has_output = has(b, r'(?:echo|print|_e|printf|vprintf)\s*[(\s]') or \
                 has(b, r'return\s+[\'"][A-Za-z]')

    if not has_output:
        # No user-facing strings needed
        return 7, [], []

    # Check for hardcoded English strings being output
    # Look for echo "string" or echo 'string' with readable text
    hardcoded_output = re.findall(
        r'(?:echo|print)\s+[\'"]([A-Za-z][A-Za-z\s,\.!?:;\-]{5,})[\'"]',
        b
    )
    # Also catch: echo "label: " or return "Message"
    hardcoded_return = re.findall(
        r'return\s+[\'"]([A-Za-z][A-Za-z\s,\.!?:;\-]{5,})[\'"]',
        b
    )

    has_i18n = has(b, I18N_FUNCS)

    if hardcoded_output and not has_i18n:
        score = 3
        critical.append(f"hardcoded user-facing strings without i18n wrapping: {hardcoded_output[:2]}")
    elif hardcoded_output:
        score -= 1
        notes.append("some hardcoded strings alongside i18n usage")

    if not has_i18n and has_output:
        # Check if the output is just variable content (not hardcoded strings)
        if not hardcoded_output and not hardcoded_return:
            score = 7  # N/A - outputs variables, not literal strings
        elif hardcoded_return and not has_i18n:
            score -= 2
            notes.append("returns hardcoded strings that may be user-facing")

    return max(1, score), notes, critical


# ── Accessibility (dimension 9) ──────────────────────────────────────────────

HTML_OUTPUT = r'<\s*(?:input|form|select|textarea|button|img|label|fieldset|legend)\b'
HAS_LABEL = r'<\s*label\b'
HAS_FOR = r'\bfor\s*=\s*["\']'
HAS_ALT = r'\balt\s*=\s*["\']'
IMG_TAG = r'<\s*img\b'
FORM_INPUTS = r'<\s*(?:input|select|textarea)\b'

def score_accessibility(fn):
    b = body(fn)
    score = 10
    notes = []
    critical = []

    has_html = has(b, HTML_OUTPUT) or has(b, r'<div|<span|<p\b|<ul|<table')

    if not has_html:
        return 7, [], []

    # img tags should have alt
    img_tags = re.findall(IMG_TAG, b)
    alt_attrs = re.findall(HAS_ALT, b)
    if img_tags and len(alt_attrs) < len(img_tags):
        score -= 2
        critical.append("img tag(s) missing alt attribute")

    # form inputs should have labels
    inputs = re.findall(FORM_INPUTS, b, re.IGNORECASE)
    labels = re.findall(HAS_LABEL, b, re.IGNORECASE)
    for_attrs = re.findall(HAS_FOR, b, re.IGNORECASE)

    if inputs and not labels:
        score -= 2
        critical.append("form inputs without associated label elements")
    elif inputs and labels and not for_attrs:
        score -= 1
        notes.append("labels present but missing 'for' attributes")

    return max(1, score), notes, critical


# ── Training Tags ─────────────────────────────────────────────────────────────

def get_training_tags(fn):
    b = body(fn)
    d = docblock(fn)
    tags = []

    # OOP
    if fn.get("class_context"):
        tags.append("oop")

    # Hooks
    if has(b, r'add_action\s*\(|add_filter\s*\('):
        tags.append("hooks")
    if has(b, r'do_action\s*\(|apply_filters\s*\('):
        tags.append("hooks")
    if has(b, r'remove_action\s*\(|remove_filter\s*\('):
        tags.append("hooks")

    # Security patterns
    if has(b, NONCE_FUNCS):
        tags.append("nonce-verification")
    if has(b, CAP_FUNCS):
        tags.append("capability-check")
    if has(b, OUTPUT_ESC):
        tags.append("output-escaping")
    if has(b, r'sanitize_\w+\s*\(|wp_unslash\s*\('):
        tags.append("input-sanitization")

    # WP APIs
    if has(b, r'WP_Query\b|new\s+WP_Query'):
        tags.append("wp-query")
    if has(b, r'get_posts\s*\('):
        tags.append("wp-query")
    if has(b, r'register_post_type\s*\('):
        tags.append("custom-post-type")
    if has(b, r'register_taxonomy\s*\('):
        tags.append("taxonomy")
    if has(b, r'register_rest_route\s*\('):
        tags.append("rest-api")
    if has(b, r'wp_enqueue_script\s*\(|wp_enqueue_style\s*\('):
        tags.append("asset-enqueueing")
    if has(b, r'add_shortcode\s*\('):
        tags.append("shortcodes")
    if has(b, r'add_meta_box\s*\('):
        tags.append("meta-boxes")
    if has(b, r'register_setting\s*\(|add_settings_section\s*\(|add_settings_field\s*\('):
        tags.append("settings-api")
    if has(b, r'get_option\s*\(|update_option\s*\(|delete_option\s*\('):
        tags.append("options-api")
    if has(b, r'get_(?:post|term|user)_meta\s*\(|update_(?:post|term|user)_meta\s*\('):
        tags.append("metadata-api")
    if has(b, r'get_transient\s*\(|set_transient\s*\(|delete_transient\s*\('):
        tags.append("transients")
    if has(b, r'wp_cache_get\s*\(|wp_cache_set\s*\('):
        tags.append("object-cache")
    if has(b, r'\$wpdb\s*->'):
        tags.append("wpdb")
    if has(b, r'\$wpdb\s*->\s*prepare\s*\('):
        tags.append("sql-safety")
    if has(b, r'wp_send_json|wp_send_json_success|wp_send_json_error'):
        tags.append("ajax")
    if has(b, r'check_ajax_referer\s*\(|wp_ajax_'):
        tags.append("ajax")
    if has(b, r'register_widget\s*\(|WP_Widget\b'):
        tags.append("widgets")
    if has(b, r'WP_Filesystem\b|global\s+\$wp_filesystem'):
        tags.append("wp-filesystem")
    if has(b, r'add_menu_page\s*\(|add_submenu_page\s*\(|add_options_page\s*\('):
        tags.append("admin-menus")
    if has(b, r'wc_\w+|WooCommerce|WC\(\)|WC_\w+'):
        tags.append("woocommerce")
    if has(b, r'__\s*\(|_e\s*\(|esc_html__\s*\(|_n\s*\('):
        tags.append("i18n")
    if has(b, r'wp_mail\s*\('):
        tags.append("email")
    if has(b, r'wp_schedule_event\s*\(|wp_cron\b'):
        tags.append("cron")
    if has(b, r'the_content|setup_postdata|have_posts|the_loop'):
        tags.append("template-tags")
    if has(b, r'walker_\w+|Walker\b'):
        tags.append("walkers")

    return list(set(tags))


# ── Core Tier Auto-pass ───────────────────────────────────────────────────────

def is_core_tier(fn):
    return fn.get("quality_tier") == "core"


# ── Main Judge ────────────────────────────────────────────────────────────────

def judge_function(fn):
    """Apply all rubric dimensions and return assessment dict."""
    b = body(fn)
    line_count = fn.get("line_count", 0)
    fname = fn.get("function_name", "")
    fpath = fn.get("source_file", "")

    # Skip too-short functions
    if line_count < 5:
        return None  # skip

    # Core tier auto-pass
    if is_core_tier(fn):
        return {
            "function_name": fname,
            "file_path": fpath,
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
            "dependency_chain": fn.get("dependencies", []) or [],
            "training_tags": get_training_tags(fn),
            "notes": "WordPress core code - auto-passed as reference implementation.",
        }

    # Score each dimension
    wpcs_score, wpcs_notes = score_wpcs(fn)
    sql_score, sql_notes, sql_critical = score_sql_safety(fn)
    sec_score, sec_notes, sec_critical = score_security(fn)
    perf_score, perf_notes, perf_critical = score_performance(fn)
    api_score, api_notes, api_critical = score_wp_api(fn)
    qual_score, qual_notes, qual_critical = score_code_quality(fn)
    dep_score, dep_notes = score_dependency(fn)
    i18n_score, i18n_notes, i18n_critical = score_i18n(fn)
    a11y_score, a11y_notes, a11y_critical = score_accessibility(fn)

    scores = {
        "wpcs_compliance": wpcs_score,
        "sql_safety": sql_score,
        "security": sec_score,
        "performance": perf_score,
        "wp_api_usage": api_score,
        "code_quality": qual_score,
        "dependency_integrity": dep_score,
        "i18n": i18n_score,
        "accessibility": a11y_score,
    }

    all_critical = sql_critical + sec_critical + perf_critical + api_critical + qual_critical + i18n_critical + a11y_critical
    all_notes = wpcs_notes + sql_notes + sec_notes + perf_notes + api_notes + qual_notes + dep_notes + i18n_notes + a11y_notes

    # Determine verdict
    # PASS = ALL dimensions >= 8, no critical failures
    # Security auto-fail if < 5
    verdict = "PASS"
    fail_reasons = []

    if sec_score < 5:
        verdict = "FAIL"
        fail_reasons.append(f"security score {sec_score} < 5 (auto-fail)")

    if all_critical:
        verdict = "FAIL"
        fail_reasons.extend(all_critical)

    # N/A dimensions: i18n=7 and accessibility=7 mean "not applicable" - they do NOT block PASS
    # dependency_integrity=7 is also N/A when no unusual dependencies are observed
    NA_EXEMPT_DIMS = {"i18n", "accessibility", "dependency_integrity"}
    for dim, s in scores.items():
        if s < 8:
            # Score of 7 on N/A-eligible dimensions is exempt from the >= 8 requirement
            if dim in NA_EXEMPT_DIMS and s == 7:
                continue
            verdict = "FAIL"
            fail_reasons.append(f"{dim} score {s} < 8")

    # Build notes string
    if verdict == "PASS":
        tags = get_training_tags(fn)
        notes_str = f"Well-structured code following WordPress standards. Demonstrates: {', '.join(tags) if tags else 'general WP patterns'}."
    else:
        notes_str = "; ".join(fail_reasons[:5])
        if all_notes:
            notes_str += f". Issues: {'; '.join(all_notes[:5])}"

    training_tags = get_training_tags(fn)

    custom_id = f"{fn.get('source_repo','unknown')}_{fn.get('start_line',0)}_{fname.replace('::', '--').replace('->', '__')}"

    return {
        "function_name": fname,
        "file_path": fpath,
        "verdict": verdict,
        "scores": scores,
        "critical_failures": list(set(all_critical)),
        "dependency_chain": fn.get("dependencies", []) or [],
        "training_tags": training_tags,
        "notes": notes_str,
        "_custom_id": custom_id,
    }


# ── Process all repos ─────────────────────────────────────────────────────────

def process_repo(repo):
    src = f"{EXTRACTED}/{repo}.json"
    if not os.path.exists(src):
        print(f"  SKIP (missing): {repo}")
        return 0, 0, 0

    with open(src) as f:
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
        record = dict(fn)
        record["assessment"] = assessment
        record["training_tags"] = assessment["training_tags"]

        if assessment["verdict"] == "PASS":
            passed.append(record)
        else:
            failed.append(record)

    # Write outputs
    passed_path = f"{PASSED_DIR}/{repo}.json"
    failed_path = f"{FAILED_DIR}/{repo}.json"

    with open(passed_path, "w") as f:
        json.dump(passed, f, indent=2)

    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)

    return len(passed), len(failed), skipped


def main():
    os.makedirs(PASSED_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    print("WordPress Code Quality Judge")
    print("=" * 60)

    for repo in REPOS:
        print(f"Judging: {repo}...", end="", flush=True)
        p, f, s = process_repo(repo)
        total_passed += p
        total_failed += f
        total_skipped += s
        total = p + f
        pass_rate = (p / total * 100) if total > 0 else 0
        print(f" {p} PASS, {f} FAIL, {s} skip — {pass_rate:.0f}% pass rate")

    print("=" * 60)
    print(f"TOTAL: {total_passed} PASS, {total_failed} FAIL, {total_skipped} skipped")
    total = total_passed + total_failed
    if total > 0:
        print(f"Overall pass rate: {total_passed/total*100:.1f}%")


if __name__ == "__main__":
    main()
