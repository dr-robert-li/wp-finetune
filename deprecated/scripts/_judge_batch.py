#!/usr/bin/env python3
"""
WordPress Code Quality Judge - Deterministic Rubric Engine
Applies the judge_system.md rubric heuristically to extracted PHP functions.
"""

import json
import re
import os
import sys
from pathlib import Path

# ─── Rubric helpers ────────────────────────────────────────────────────────────

def has_pattern(body, *patterns):
    return any(re.search(p, body, re.IGNORECASE) for p in patterns)

def count_pattern(body, pattern):
    return len(re.findall(pattern, body, re.IGNORECASE))

def score_wpcs(fn):
    body = fn.get('body', '')
    docblock = fn.get('docblock', '') or ''
    name = fn.get('function_name', '')
    score = 10
    issues = []

    # Name must be lowercase_underscore or ClassName::method
    if not re.match(r'^[a-z_][a-z0-9_]*$|^[A-Z][a-zA-Z0-9_]*::[a-z_]', name):
        score -= 2
        issues.append('non-standard function name')

    # PHPDoc present - CRITICAL per rubric
    if not docblock or len(docblock.strip()) < 5:
        # Missing PHPDoc = critical WPCS fail (score 6 = below 8 threshold)
        score -= 4
        issues.append('missing PHPDoc block')
    elif len(docblock.strip()) < 20:
        # Minimal docblock (just comment, no tags)
        score -= 2
        issues.append('minimal PHPDoc - missing @param/@return/@since')
    elif '@param' not in docblock and '@return' not in docblock and '@since' not in docblock:
        # Has docblock but no proper tags on non-trivial functions
        if fn.get('line_count', 0) > 8:
            score -= 2
            issues.append('PHPDoc lacks @param/@return/@since tags')
        else:
            score -= 1
            issues.append('PHPDoc lacks @param/@return tags')

    # Yoda conditions check - look for assignments in conditions (anti-pattern)
    if re.search(r'if\s*\(\s*\$\w+\s*==\s*[\'"]', body):
        score -= 1
        issues.append('non-Yoda comparison')

    # var_dump / print_r debug statements
    if has_pattern(body, r'\bvar_dump\b', r'\bprint_r\b', r'\bdie\s*\(', r'\bvar_export\b'):
        score -= 2
        issues.append('debug statement in production code')

    # error_log in production paths
    if re.search(r'\berror_log\s*\(', body):
        score -= 1
        issues.append('error_log call')

    return max(1, score), issues

def score_sql_safety(fn):
    body = fn.get('body', '')
    sql_patterns = fn.get('sql_patterns', [])
    score = 10
    issues = []
    critical = []

    if not sql_patterns and not has_pattern(body, r'\$wpdb\b', r'->query\b', r'->get_results\b', r'->get_row\b', r'->get_var\b', r'->prepare\b'):
        return 10, [], []  # No SQL at all

    # Unprepared queries with dynamic values
    # Look for $wpdb->query/get_results/etc WITHOUT prepare()
    raw_query_patterns = [
        r'\$wpdb->(query|get_results|get_row|get_var|get_col)\s*\(\s*["\'].*\$',
        r'\$wpdb->(query|get_results|get_row|get_var|get_col)\s*\(\s*["\'].*\{',
        r'\$wpdb->(query|get_results|get_row|get_var|get_col)\s*\(\s*\$(?!wpdb)',
    ]
    for p in raw_query_patterns:
        if re.search(p, body):
            score -= 5
            critical.append('unprepared SQL query with dynamic values')
            break

    # Hardcoded wp_ table prefix
    if re.search(r'["\']wp_\w+["\']', body) and not re.search(r'\$wpdb->prefix', body):
        score -= 2
        issues.append('hardcoded wp_ table prefix')

    # Good: uses prepare()
    if re.search(r'\$wpdb->prepare\s*\(', body):
        score = min(10, score + 1)

    return max(1, score), issues, critical

def score_security(fn):
    body = fn.get('body', '')
    hooks = fn.get('hooks_used', []) or []
    score = 10
    issues = []
    critical = []

    # eval() usage
    if re.search(r'\beval\s*\(', body):
        score -= 5
        critical.append('eval() usage')

    # extract() on $_POST/$_GET/$_REQUEST
    if re.search(r'\bextract\s*\(\s*\$_(POST|GET|REQUEST|COOKIE|SERVER)', body):
        score -= 5
        critical.append('extract() on superglobal')

    # State-changing AJAX/form handler without nonce check
    is_state_changing = any(x in body for x in [
        'wp_insert_post', 'wp_update_post', 'wp_delete_post',
        'update_option', 'add_option', 'delete_option',
        'update_user_meta', 'delete_user_meta',
        '$wpdb->insert', '$wpdb->update', '$wpdb->delete', '$wpdb->query',
    ])
    is_ajax = any('ajax' in h.lower() for h in hooks) or re.search(r'wp_ajax_', body)
    has_nonce = has_pattern(body, r'wp_verify_nonce', r'check_ajax_referer', r'check_admin_referer', r'nonce_field')

    if is_state_changing and is_ajax and not has_nonce:
        score -= 3
        critical.append('state-changing AJAX handler missing nonce check')

    # Direct $_POST/$_GET usage without sanitization
    unsanitized = re.findall(r'\$_(POST|GET|REQUEST|COOKIE)\s*\[', body)
    sanitize_calls = count_pattern(body, r'\b(sanitize_\w+|absint|intval|floatval|wp_kses|esc_\w+|wp_unslash)\s*\(')
    if unsanitized and sanitize_calls == 0:
        score -= 2
        issues.append('user input used without sanitization')

    # Unescaped output
    echo_matches = re.findall(r'\becho\s+[^;]+', body)
    for match in echo_matches:
        # Check if the echo contains user-controlled data without escaping
        if re.search(r'\$_(POST|GET|REQUEST|COOKIE)', match):
            if not re.search(r'\besc_\w+\s*\(|\bwp_kses\s*\(|\bintval\s*\(|\babsint\s*\(', match):
                score -= 3
                critical.append('unescaped user-controlled output')
                break

    # Shell execution with potential injection
    if has_pattern(body, r'\bpassthru\s*\(', r'\bexec\s*\(', r'\bshell_exec\s*\(', r'\bproc_open\s*\('):
        # Check if arguments are escaped
        if not has_pattern(body, r'escapeshellarg\s*\(', r'escapeshellcmd\s*\('):
            score -= 3
            critical.append('shell command execution without escapeshellarg/escapeshellcmd')
        else:
            score -= 1
            issues.append('shell command execution (verify args are escaped)')

    # file operations - direct PHP file functions
    if has_pattern(body, r'\bfopen\s*\(', r'\bfile_put_contents\s*\(', r'\bfwrite\s*\('):
        if not has_pattern(body, r'WP_Filesystem', r'wp_filesystem'):
            score -= 2
            issues.append('direct file operations instead of WP_Filesystem')

    # Missing capability check on admin functions
    is_admin_action = any('admin' in h.lower() for h in hooks) or 'admin_post_' in body
    has_cap_check = has_pattern(body, r'current_user_can\s*\(')
    if is_admin_action and is_state_changing and not has_cap_check:
        score -= 2
        issues.append('admin action missing capability check')

    return max(1, score), issues, critical

def score_performance(fn):
    body = fn.get('body', '')
    score = 10
    issues = []
    critical = []

    # Query inside loop - distinguish cached vs uncached DB operations
    loop_patterns = [r'\bforeach\b', r'\bfor\s*\(', r'\bwhile\s*\(']
    # Truly expensive: new WP_Query, get_posts, raw $wpdb queries
    expensive_db_patterns = [r'\bnew\s+WP_Query\b', r'\bget_posts\s*\(', r'\$wpdb->(query|get_results|get_row|get_var)\b']
    # WP internally cached (object cache): get_post_meta, get_post_status, get_term_meta, get_option
    cached_db_patterns = [r'\bget_post_meta\s*\(', r'\bget_term_meta\s*\(', r'\bget_post_status\s*\(', r'\bget_user_meta\s*\(']

    has_loop = any(re.search(p, body) for p in loop_patterns)

    if has_loop:
        # Find first loop position
        loop_start = -1
        for p in loop_patterns:
            m = re.search(p, body)
            if m and (loop_start == -1 or m.start() < loop_start):
                loop_start = m.start()

        # Check for expensive DB calls after loop start
        for p in expensive_db_patterns:
            m = re.search(p, body)
            if m and m.start() > loop_start:
                score -= 3
                critical.append('expensive query inside loop (N+1 pattern)')
                break

        # Check for WP-cached DB calls after loop start (less severe)
        if score == 10:  # only if no expensive pattern found
            for p in cached_db_patterns:
                m = re.search(p, body)
                if m and m.start() > loop_start:
                    score -= 1
                    issues.append('potentially repeated DB meta calls in loop (WP-cached but consider batch loading)')
                    break

    # SELECT * on meta/post tables
    if re.search(r'SELECT\s+\*', body, re.IGNORECASE):
        score -= 2
        issues.append('SELECT * query')

    # Shell commands - not performance but caught here as resource concern
    if has_pattern(body, r'\bpassthru\s*\(', r'\bexec\s*\(', r'\bshell_exec\s*\(', r'\bproc_open\s*\('):
        score -= 1
        issues.append('shell command execution')

    return max(1, score), issues, critical

def score_wp_api(fn):
    body = fn.get('body', '')
    score = 10
    issues = []
    critical = []

    # REST route without permission callback
    if re.search(r'register_rest_route\s*\(', body):
        if not re.search(r'permission_callback', body):
            score -= 4
            critical.append('REST route missing permission_callback')
        elif re.search(r'permission_callback.*__return_true', body):
            score -= 1
            issues.append('REST route uses __return_true permission callback')

    # Raw SQL for what WP_Query could handle
    if re.search(r'\$wpdb->get_results.*FROM.*posts\b', body, re.IGNORECASE):
        if not re.search(r'JOIN|GROUP BY|HAVING|UNION', body, re.IGNORECASE):
            score -= 2
            issues.append('raw SQL for post query (should use WP_Query)')

    # Using deprecated functions
    deprecated = ['get_currentuserinfo', 'get_userdatabylogin', 'get_user_by_email',
                  'wp_get_post_categories', 'update_usermeta', 'get_usermeta',
                  'the_category_ID', 'wp_cache_reset']
    for dep in deprecated:
        if re.search(r'\b' + dep + r'\b', body):
            score -= 2
            issues.append(f'deprecated function: {dep}')
            break

    return max(1, score), issues, critical

def score_code_quality(fn):
    body = fn.get('body', '')
    line_count = fn.get('line_count', 0)
    score = 10
    issues = []
    critical = []

    # Commented-out code blocks
    commented_code = re.findall(r'//.*\$\w+\s*=|//.*function\s+|//.*echo\s+', body)
    if len(commented_code) > 3:
        score -= 1
        issues.append('excessive commented-out code')

    # Too many responsibilities (too long without clear structure)
    if line_count > 200:
        score -= 3
        issues.append(f'function too long ({line_count} lines) - almost certainly violates SRP')
    elif line_count > 120:
        score -= 2
        issues.append(f'function too long ({line_count} lines) - likely violates SRP')
    elif line_count > 80:
        score -= 1
        issues.append(f'function quite long ({line_count} lines)')

    # Error handling
    if has_pattern(body, r'\$wpdb->', r'wp_remote_get', r'wp_remote_post'):
        if not has_pattern(body, r'is_wp_error\s*\(', r'WP_Error', r'if\s*\(\s*false', r'!== false'):
            score -= 1
            issues.append('missing error handling for DB/HTTP calls')

    # Silent error swallowing
    if re.search(r'@\$\w+|@\w+\s*\(', body):
        score -= 1
        issues.append('PHP error suppression operator @')

    return max(1, score), issues, critical

def score_dependency_integrity(fn):
    score = 9  # Default reasonable score - we can't verify all deps
    issues = []
    return score, issues

def score_i18n(fn):
    body = fn.get('body', '')
    score = 10
    issues = []
    critical = []

    # Check if function has HTML output or user-facing strings
    has_output = has_pattern(body, r'\becho\b', r'\bprint\b', r'<[a-z]+', r'printf\b')
    if not has_output:
        return None, [], []  # N/A - no user-facing strings

    # Look for hardcoded English strings in output
    # Find echo/print statements
    echo_lines = re.findall(r'(?:echo|print)\s+[^;]+', body)
    has_hardcoded = False
    for line in echo_lines:
        # Check if there are literal strings that look like English words
        literals = re.findall(r'["\'][A-Za-z][A-Za-z\s,!?]{4,}["\']', line)
        for lit in literals:
            # If no translation wrapper nearby
            if not re.search(r'__\s*\(|_e\s*\(|esc_html__\s*\(|_x\s*\(|_n\s*\(', line):
                has_hardcoded = True
                break

    if has_hardcoded:
        score -= 3
        critical.append('hardcoded English strings in output without translation wrappers')

    # Check for proper text domain usage
    td_matches = re.findall(r'__\s*\([^,]+,\s*[\'"]([^\'"]+)[\'"]', body)
    if td_matches:
        domains = set(td_matches)
        if len(domains) > 1:
            score -= 1
            issues.append('inconsistent text domains')

    return max(1, score), issues, critical

def score_accessibility(fn):
    body = fn.get('body', '')
    score = 10
    issues = []
    critical = []

    # Check if function outputs HTML
    has_html_output = has_pattern(body, r'<input', r'<select', r'<textarea', r'<button', r'<img\b', r'<form\b')
    if not has_html_output:
        return None, [], []  # N/A - no HTML output

    # Form inputs without labels
    input_count = len(re.findall(r'<input\b(?![^>]*type=["\']hidden["\'])', body, re.IGNORECASE))
    label_count = len(re.findall(r'<label\b', body, re.IGNORECASE))
    aria_label_count = len(re.findall(r'aria-label', body, re.IGNORECASE))

    if input_count > 0 and label_count == 0 and aria_label_count == 0:
        score -= 3
        critical.append('form inputs without labels or aria-label')

    # Images without alt
    img_matches = re.findall(r'<img\b[^>]*>', body, re.IGNORECASE)
    for img in img_matches:
        if 'alt=' not in img.lower():
            score -= 3
            critical.append('img element missing alt attribute')
            break

    return max(1, score), issues, critical

def get_training_tags(fn):
    body = fn.get('body', '')
    hooks = fn.get('hooks_used', []) or []
    tags = []

    patterns = {
        'asset-enqueue': [r'wp_enqueue_script', r'wp_enqueue_style', r'wp_register_script'],
        'hooks': [r'add_action\s*\(', r'add_filter\s*\(', r'do_action\s*\(', r'apply_filters\s*\('],
        'nonce-verification': [r'wp_verify_nonce', r'check_ajax_referer', r'check_admin_referer'],
        'capability-check': [r'current_user_can\s*\('],
        'wpdb-query': [r'\$wpdb->'],
        'options-api': [r'get_option\s*\(', r'update_option\s*\(', r'add_option\s*\('],
        'post-meta': [r'get_post_meta\s*\(', r'update_post_meta\s*\(', r'add_post_meta\s*\('],
        'user-meta': [r'get_user_meta\s*\(', r'update_user_meta\s*\('],
        'transients': [r'get_transient\s*\(', r'set_transient\s*\(', r'delete_transient\s*\('],
        'wp-query': [r'\bnew\s+WP_Query\b', r'\bget_posts\s*\('],
        'rest-api': [r'register_rest_route\s*\(', r'WP_REST_Response', r'WP_REST_Request'],
        'ajax-handler': [r'wp_ajax_', r'wp_send_json', r'wp_die\s*\('],
        'shortcode': [r'add_shortcode\s*\(', r'do_shortcode\s*\('],
        'cpt-registration': [r'register_post_type\s*\(', r'register_taxonomy\s*\('],
        'sanitization': [r'sanitize_\w+\s*\(', r'wp_kses\s*\('],
        'escaping': [r'esc_html\s*\(', r'esc_attr\s*\(', r'esc_url\s*\('],
        'i18n': [r'__\s*\(', r'_e\s*\(', r'esc_html__\s*\('],
        'caching': [r'wp_cache_get\s*\(', r'wp_cache_set\s*\(', r'get_transient\s*\('],
        'admin-ui': [r'add_menu_page\s*\(', r'add_submenu_page\s*\(', r'add_settings_\w+\s*\('],
        'settings-api': [r'register_setting\s*\(', r'add_settings_field\s*\(', r'add_settings_section\s*\('],
        'woocommerce': [r'WC\(\)', r'wc_\w+\s*\(', r'WooCommerce', r'woocommerce_'],
        'block-editor': [r'register_block_type\s*\(', r'wp_set_script_translations\s*\(', r'gutenberg'],
        'filesystem-api': [r'WP_Filesystem', r'global\s+\$wp_filesystem'],
        'http-api': [r'wp_remote_get\s*\(', r'wp_remote_post\s*\(', r'wp_safe_remote_'],
        'error-handling': [r'is_wp_error\s*\(', r'new\s+WP_Error\s*\(', r'WP_Error'],
    }

    for tag, pats in patterns.items():
        if any(re.search(p, body, re.IGNORECASE) for p in pats):
            tags.append(tag)

    return tags

def is_test_code(fn):
    """Detect test code by file path or class context."""
    source_file = fn.get('source_file', '') or ''
    class_context = fn.get('class_context', '') or ''
    function_name = fn.get('function_name', '') or ''
    body = fn.get('body', '') or ''

    # Test file paths
    test_path_patterns = ['/test/', '/tests/', '/phpunit/', '/spec/', '/fixtures/',
                          'test.php', '-test.php', '_test.php', '.test.php',
                          'TestCase', 'test-case']
    for p in test_path_patterns:
        if p.lower() in source_file.lower():
            return True

    # Test class patterns
    if re.search(r'Test(?:Case)?$|^Test|_Test$', class_context):
        return True

    # PHPUnit method names
    if re.match(r'^(setUp|tearDown|setUpBeforeClass|tearDownAfterClass|test_|test[A-Z])', function_name):
        return True

    # PHPUnit assertions in body
    if re.search(r'\$this->assert(?:Equals|True|False|Null|NotNull|Contains|Same|Count)\s*\(', body):
        return True

    return False


def judge_function(fn):
    """Apply all rubric dimensions and return assessment dict."""
    body = fn.get('body', '')
    line_count = fn.get('line_count', 0)

    # Skip tiny functions
    if line_count < 5:
        return None  # Will be skipped

    # Detect and auto-fail test code
    if is_test_code(fn):
        fail_scores = {k: 1 for k in ['wpcs_compliance','sql_safety','security','performance','wp_api_usage','code_quality','dependency_integrity','i18n','accessibility']}
        return {
            'verdict': 'FAIL',
            'scores': fail_scores,
            'critical_failures': ['test code, excluded from training data'],
            'training_tags': [],
            'notes': 'Test code, excluded from training data.'
        }

    # Auto-PASS for core tier (shouldn't appear, but handle gracefully)
    if fn.get('quality_tier') == 'core':
        return {
            'verdict': 'PASS',
            'scores': {k: 10 for k in ['wpcs_compliance','sql_safety','security','performance','wp_api_usage','code_quality','dependency_integrity','i18n','accessibility']},
            'critical_failures': [],
            'training_tags': get_training_tags(fn),
            'notes': 'Auto-PASS: WordPress core tier'
        }

    # Score each dimension
    wpcs_score, wpcs_issues = score_wpcs(fn)
    sql_score, sql_issues, sql_critical = score_sql_safety(fn)
    sec_score, sec_issues, sec_critical = score_security(fn)
    perf_score, perf_issues, perf_critical = score_performance(fn)
    api_score, api_issues, api_critical = score_wp_api(fn)
    cq_score, cq_issues, cq_critical = score_code_quality(fn)
    dep_score, dep_issues = score_dependency_integrity(fn)
    i18n_score, i18n_issues, i18n_critical = score_i18n(fn)
    a11y_score, a11y_issues, a11y_critical = score_accessibility(fn)

    # N/A dimensions use None and are stored as 7 in output but don't gate PASS
    i18n_display = i18n_score if i18n_score is not None else 7
    a11y_display = a11y_score if a11y_score is not None else 7

    scores = {
        'wpcs_compliance': wpcs_score,
        'sql_safety': sql_score,
        'security': sec_score,
        'performance': perf_score,
        'wp_api_usage': api_score,
        'code_quality': cq_score,
        'dependency_integrity': dep_score,
        'i18n': i18n_display,
        'accessibility': a11y_display,
    }

    all_critical = sql_critical + sec_critical + perf_critical + api_critical + cq_critical + i18n_critical + a11y_critical

    # Gating dimensions - must ALL be >= 8 (excluding N/A dims which score 7)
    gating_scores = {
        'wpcs_compliance': wpcs_score,
        'sql_safety': sql_score,
        'security': sec_score,
        'performance': perf_score,
        'wp_api_usage': api_score,
        'code_quality': cq_score,
        'dependency_integrity': dep_score,
    }
    # Only gate on i18n/a11y if they have real (non-N/A) scores
    if i18n_score is not None:
        gating_scores['i18n'] = i18n_score
    if a11y_score is not None:
        gating_scores['accessibility'] = a11y_score

    # Security auto-fail
    if sec_score < 5:
        verdict = 'FAIL'
        all_critical.append(f'security score {sec_score} < 5 triggers auto-FAIL')
    elif all(v >= 8 for v in gating_scores.values()) and not all_critical:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'

    all_issues = wpcs_issues + sql_issues + sec_issues + perf_issues + api_issues + cq_issues + dep_issues + i18n_issues + a11y_issues

    notes_parts = []
    if verdict == 'PASS':
        tags = get_training_tags(fn)
        notes_parts.append(f"Production-quality code. Demonstrates: {', '.join(tags) if tags else 'general WordPress patterns'}.")
    else:
        # Only flag truly failing gating dimensions (not N/A ones)
        failing = [k for k, v in gating_scores.items() if v < 8]
        if failing:
            notes_parts.append(f"Fails dimensions: {', '.join(failing)}.")
        if all_critical:
            notes_parts.append(f"Critical: {'; '.join(all_critical[:3])}.")
        if all_issues:
            notes_parts.append(f"Issues: {'; '.join(all_issues[:3])}.")

    training_tags = get_training_tags(fn)

    return {
        'verdict': verdict,
        'scores': scores,
        'critical_failures': all_critical,
        'training_tags': training_tags,
        'notes': ' '.join(notes_parts) or 'No specific notes.',
    }

def process_repo(repo_name, extracted_dir, passed_dir, failed_dir):
    input_path = os.path.join(extracted_dir, repo_name + '.json')
    if not os.path.exists(input_path):
        print(f"  SKIP {repo_name}: file not found")
        return 0, 0, 0

    with open(input_path) as f:
        functions = json.load(f)

    passed = []
    failed = []
    skipped = 0

    for fn in functions:
        line_count = fn.get('line_count', 0)
        if line_count < 5:
            skipped += 1
            continue

        assessment = judge_function(fn)
        if assessment is None:
            skipped += 1
            continue

        # Build output record
        record = dict(fn)
        record['assessment'] = {
            'function_name': fn.get('function_name', ''),
            'file_path': fn.get('source_file', ''),
            'verdict': assessment['verdict'],
            'scores': assessment['scores'],
            'critical_failures': assessment['critical_failures'],
            'dependency_chain': fn.get('dependencies', []),
            'training_tags': assessment['training_tags'],
            'notes': assessment['notes'],
            '_custom_id': f"{repo_name}_{fn.get('start_line', 0)}_{fn.get('function_name', '')}",
        }
        record['training_tags'] = assessment['training_tags']

        if assessment['verdict'] == 'PASS':
            passed.append(record)
        else:
            failed.append(record)

    # Write outputs
    os.makedirs(passed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    with open(os.path.join(passed_dir, repo_name + '.json'), 'w') as f:
        json.dump(passed, f, indent=2)

    with open(os.path.join(failed_dir, repo_name + '.json'), 'w') as f:
        json.dump(failed, f, indent=2)

    return len(passed), len(failed), skipped

def main():
    repos = [
        'facebook-for-woocommerce',
        'fonto',
        'full-site-editing',
        'geidea-online-payments',
        'gf-form-locator',
        'global-payments-woocommerce',
        'google-listings-and-ads',
        'google-site-kit',
        'google-video-sitemap-feed-with-multisite-support',
        'gridable',
        'grigora-kit',
        'gui-for-lcp',
        'gutenberg',
        'header-and-footer-scripts',
        'hello-elementor',
        'hello-plus',
        'html-post-editor-new',
        'image-comparison',
        'image-prioritizer',
        'insert-pages',
    ]

    base = '/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output'
    extracted_dir = os.path.join(base, 'extracted')
    passed_dir = os.path.join(base, 'passed')
    failed_dir = os.path.join(base, 'failed')

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    print(f"Processing {len(repos)} repos...")
    for repo in repos:
        p, f, s = process_repo(repo, extracted_dir, passed_dir, failed_dir)
        total_passed += p
        total_failed += f
        total_skipped += s
        pass_rate = p / (p + f) * 100 if (p + f) > 0 else 0
        print(f"  {repo}: {p} PASS, {f} FAIL, {s} SKIP ({pass_rate:.1f}% pass rate)")

    print(f"\nTOTAL: {total_passed} PASS, {total_failed} FAIL, {total_skipped} SKIP")
    print(f"Overall pass rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")

if __name__ == '__main__':
    main()
