#!/usr/bin/env python3
"""
WordPress Code Quality Judge - Batch Processor
Judges all functions in specified repos and writes passed/failed JSON files.
"""

import json
import os
import re
import sys

EXTRACTED_DIR = '/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output/extracted'
PASSED_DIR = '/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output/passed'
FAILED_DIR = '/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output/failed'

os.makedirs(PASSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

REPOS = [
    '360-video',
    'a3-responsive-slider',
    'absolute-relative-urls',
    'acf-page-builder',
    'ads-txt',
    'advanced-custom-fields',
    'advanced-custom-fields-markdown',
    'algori-image-video-slider',
    'amp',
    'aqua-page-builder',
]


# ─── helpers ────────────────────────────────────────────────────────────────

def has_pattern(body, *patterns):
    return any(re.search(p, body) for p in patterns)

def count_pattern(body, pattern):
    return len(re.findall(pattern, body))


# ─── dimension scorers ──────────────────────────────────────────────────────

def score_wpcs(fn):
    """WordPress Coding Standards compliance."""
    body = fn.get('body', '')
    doc  = fn.get('docblock', '') or ''
    name = fn.get('function_name', '')

    score = 10
    failures = []

    # naming: should be snake_case for functions
    if re.search(r'function\s+[a-z][a-zA-Z]*[A-Z][a-zA-Z]*\s*\(', body):
        # camelCase function name — minor deduction
        score -= 1

    # missing PHPDoc — only deduct for public/non-trivial functions
    if not doc.strip():
        # Check if function has params (signature has parameters)
        sig_match = re.match(r'.*?function\s+\w+\s*\(([^)]*)\)', body[:300], re.DOTALL)
        has_params = sig_match and sig_match.group(1).strip()
        has_return = re.search(r'\breturn\s+\S', body)
        if has_params or has_return:
            score -= 2
            failures.append('Missing PHPDoc block on function with params/return')
        else:
            score -= 1  # Minor: simple function without params/return still benefits from a doc
    else:
        # PHPDoc quality
        if '@param' not in doc and re.search(r'\$\w+', body.split('{')[0] if '{' in body else body):
            score -= 1  # params exist but not documented
        if '@return' not in doc and re.search(r'\breturn\s+\S', body):
            score -= 1  # has return but not documented

    # Space after control structures — spot check
    if re.search(r'\b(if|while|for|foreach|switch)\(', body):
        score -= 1
        failures.append('Missing space after control structure keyword')

    # Yoda conditions: look for assignments in conditions (anti-Yoda is OK but missing Yoda is minor)
    # Don't penalise absence of Yoda — many modern WP plugins skip this

    # Tabs vs spaces — cannot easily check without seeing actual file, skip

    score = max(1, score)
    return score, failures


def score_sql(fn):
    """SQL safety — critical if dynamic values without prepare()."""
    body = fn.get('body', '')
    sql_patterns = fn.get('sql_patterns', []) or []

    score = 10
    failures = []

    # Raw $wpdb->query / get_results / get_var / get_row with string that contains variables
    raw_sql_re = re.compile(
        r'\$wpdb->(query|get_results|get_var|get_row|get_col)\s*\(\s*"[^"]*\$|'
        r'\$wpdb->(query|get_results|get_var|get_row|get_col)\s*\(\s*\'[^\']*\$|'
        r'\$wpdb->(query|get_results|get_var|get_row|get_col)\s*\(\s*\$[a-zA-Z_]',
        re.DOTALL
    )
    # Concatenation into SQL
    concat_sql_re = re.compile(
        r'\$wpdb->(query|get_results|get_var|get_row|get_col)\s*\([^)]*\.[^)]*\)',
        re.DOTALL
    )

    if raw_sql_re.search(body):
        # Check if it's wrapped in prepare
        if 'prepare' not in body:
            score = 1
            failures.append('CRITICAL: Unprepared SQL query with dynamic values (SQL injection risk)')
            return score, failures
        else:
            # has prepare somewhere, check if the prepare wraps the query
            score = max(score - 2, 5)

    if concat_sql_re.search(body):
        if 'prepare' not in body:
            score = 1
            failures.append('CRITICAL: SQL built by string concatenation without prepare()')
            return score, failures

    # Hardcoded wp_ table prefix
    if re.search(r'[\'"]wp_[a-z]', body):
        score -= 2
        failures.append('Hardcoded wp_ table prefix instead of $wpdb->prefix')

    # Uses prepare correctly — bonus
    if sql_patterns and 'prepare' in body:
        score = min(10, score + 1)

    # No SQL at all — neutral high score
    if not sql_patterns and not re.search(r'\$wpdb', body):
        score = 10

    score = max(1, score)
    return score, failures


def score_security(fn):
    """Security: nonces, caps, escaping, no eval/extract on user data."""
    body = fn.get('body', '')
    hooks = fn.get('hooks_used', []) or []
    name  = fn.get('function_name', '') or ''

    score = 10
    failures = []

    # Check if function is a form/ajax handler
    is_ajax_handler = (
        has_pattern(body, r'wp_ajax_') or
        any('ajax' in h.lower() for h in hooks) or
        re.search(r'action.*ajax|ajax.*action', name, re.I)
    )
    is_form_handler = has_pattern(body, r'\$_POST|\$_GET|\$_REQUEST')

    if is_ajax_handler or is_form_handler:
        # Must have nonce check
        if not has_pattern(body, r'wp_verify_nonce|check_ajax_referer|check_admin_referer'):
            score -= 4
            failures.append('CRITICAL: State-changing handler missing nonce verification')

        # Should check capabilities for sensitive ops
        if has_pattern(body, r'delete_|update_|insert_|wp_insert|wp_update|wp_delete'):
            if not has_pattern(body, r'current_user_can|is_admin\b'):
                score -= 2
                failures.append('Missing capability check for privileged operation')

    # Unescaped output
    echo_re = re.compile(r'\becho\s+(?!esc_|wp_kses|intval|absint|sanitize_)(\$_|\$\w+\[)', re.I)
    if echo_re.search(body):
        score -= 3
        failures.append('CRITICAL: Possible unescaped output of user-controlled data')

    # eval() usage
    if re.search(r'\beval\s*\(', body):
        score = 1
        failures.append('CRITICAL: eval() usage')
        return score, failures

    # extract() on user data
    if re.search(r'\bextract\s*\(\s*\$_(POST|GET|REQUEST|COOKIE)', body):
        score = 1
        failures.append('CRITICAL: extract() on user superglobal data')
        return score, failures

    # Direct file operations
    if has_pattern(body, r'\bfopen\s*\(', r'\bfile_put_contents\s*\(', r'\bfwrite\s*\('):
        if not has_pattern(body, r'WP_Filesystem|wp_filesystem'):
            score -= 2
            failures.append('Direct file I/O without WP_Filesystem')

    score = max(1, score)
    return score, failures


def score_performance(fn):
    """Performance: no N+1 queries, caching, no SELECT *."""
    body = fn.get('body', '')
    sql_patterns = fn.get('sql_patterns', []) or []

    score = 10
    failures = []

    # Queries inside loops
    loop_re = re.compile(
        r'\b(for|foreach|while)\b[^{]*\{[^}]*\$wpdb->(query|get_results|get_var|get_row)',
        re.DOTALL
    )
    if loop_re.search(body):
        score -= 4
        failures.append('CRITICAL: Database query inside loop (N+1 pattern)')

    # SELECT * on potentially wide tables
    if re.search(r'SELECT\s+\*', body, re.I):
        # Not critical unless on meta tables or without LIMIT
        if re.search(r'postmeta|usermeta|options', body, re.I):
            score -= 3
            failures.append('SELECT * on meta/options table without column specification')
        else:
            score -= 1

    # Expensive operations without caching
    if sql_patterns:
        if not has_pattern(body, r'wp_cache_get|get_transient|wp_cache_set|set_transient|static\s+\$'):
            # Only deduct if function seems to run on every request (not a CLI/cron utility)
            if has_pattern(body, r'add_action|add_filter'):
                score -= 1  # Minor: could benefit from caching

    score = max(1, score)
    return score, failures


def score_wp_api(fn):
    """WordPress API usage."""
    body = fn.get('body', '')
    hooks = fn.get('hooks_used', []) or []

    score = 10
    failures = []

    # Raw SQL for things WP_Query could handle
    if re.search(r'\$wpdb.*SELECT.*FROM.*posts\b', body, re.I | re.DOTALL):
        if not re.search(r'WP_Query|get_posts|query_posts', body):
            score -= 2
            failures.append('Raw SQL query on posts table; WP_Query preferred')

    # REST endpoint without permission_callback
    if re.search(r'register_rest_route', body):
        if not re.search(r'permission_callback', body):
            score -= 4
            failures.append('CRITICAL: REST route missing permission_callback')

    # add_action/add_filter arg count mismatch is hard to check statically — skip

    # Deprecated functions
    deprecated = [
        r'\bquery_posts\s*\(',
        r'\bthe_permalink\s*\(',  # Actually not deprecated, skip
        r'\bget_currentuserinfo\s*\(',
        r'\bcreate_function\s*\(',
    ]
    for dep in deprecated:
        if re.search(dep, body):
            score -= 2
            failures.append(f'Deprecated WordPress function: {dep}')
            break

    score = max(1, score)
    return score, failures


def score_code_quality(fn):
    """Code quality: single responsibility, error handling, no debug code."""
    body = fn.get('body', '')
    line_count = fn.get('line_count', 0)

    score = 10
    failures = []

    # Debug statements
    debug_patterns = [r'\bvar_dump\s*\(', r'\bprint_r\s*\(', r'\bdie\s*\(', r'\bexit\s*\(']
    # die/exit are sometimes legitimate, only flag if clearly debug-style
    if re.search(r'\bvar_dump\s*\(|\bprint_r\s*\(', body):
        score -= 4
        failures.append('Debug output (var_dump/print_r) in production code')

    # error_log in non-debug context
    if re.search(r'\berror_log\s*\(', body):
        score -= 1  # Minor warning: could be intentional logging

    # Very long functions doing too much
    if line_count > 150:
        score -= 2
        failures.append('Extremely long function (>150 lines) — likely violates single responsibility')
    elif line_count > 80:
        score -= 1

    # Swallowing errors (empty catch blocks)
    if re.search(r'catch\s*\([^)]*\)\s*\{\s*\}', body):
        score -= 2
        failures.append('Empty catch block — swallowed exception')

    # Global state dependency without checks
    global_deps = count_pattern(body, r'\bglobal\s+\$')
    if global_deps > 3:
        score -= 1

    score = max(1, score)
    return score, failures


def score_dependency_integrity(fn):
    """Dependency chain integrity."""
    body = fn.get('body', '')
    deps = fn.get('dependencies', []) or []

    score = 8  # Default neutral

    # Check for direct vendor requires (non-WP pattern)
    if re.search(r'\brequire\s+[\'"][^\'"]*(vendor|lib|libs)/', body):
        score -= 2

    # Circular dependency is hard to detect statically — skip
    score = max(1, min(10, score))
    return score, []


def score_i18n(fn):
    """Internationalization."""
    body = fn.get('body', '')

    # Does function output user-visible strings?
    has_echo = has_pattern(body, r'\becho\b', r'\bprintf\b', r'<[a-z]+[^>]*>')
    has_strings = re.search(r'"[A-Za-z][A-Za-z\s]{3,}"', body) or re.search(r"'[A-Za-z][A-Za-z\s]{3,}'", body)

    if not has_echo and not has_strings:
        return 7, []  # N/A

    score = 10
    failures = []

    # Has hardcoded strings in echo without translation
    hardcoded_re = re.compile(
        r'\becho\s+["\'][A-Za-z][A-Za-z\s,!.\']{3,}["\']',
        re.I
    )
    if hardcoded_re.search(body):
        # Check if there are any translation function calls at all
        if not has_pattern(body, r'\b__\s*\(', r'\b_e\s*\(', r'\besc_html__\s*\(', r'\besc_html_e\s*\(', r'\b_n\s*\(', r'\b_x\s*\('):
            score -= 3
            failures.append('CRITICAL: Hardcoded English strings output without translation wrappers')

    # String concatenation with translated strings (should use sprintf)
    if re.search(r'__\s*\([^)]+\)\s*\.\s*\$', body) or re.search(r'\$\w+\s*\.\s*__\s*\(', body):
        score -= 1
        failures.append('String concatenation with translation instead of sprintf')

    score = max(1, score)
    return score, failures


def score_accessibility(fn):
    """Accessibility for HTML output."""
    body = fn.get('body', '')

    # Does function output HTML?
    has_html = has_pattern(body, r'<input|<select|<textarea|<button|<form|<img')

    if not has_html:
        return 7, []  # N/A

    score = 10
    failures = []

    # Inputs without labels
    if re.search(r'<input\b[^>]+type=["\'](?!hidden)[^"\']+["\']', body, re.I):
        if not re.search(r'<label\b|aria-label|aria-labelledby', body, re.I):
            score -= 3
            failures.append('CRITICAL: Form input without associated label')

    # Images without alt
    if re.search(r'<img\b[^>]*>', body, re.I):
        img_tags = re.findall(r'<img\b[^>]*>', body, re.I)
        for img in img_tags:
            if 'alt=' not in img.lower():
                score -= 2
                failures.append('Image tag missing alt attribute')
                break

    score = max(1, score)
    return score, failures


# ─── training tag extractor ─────────────────────────────────────────────────

def extract_training_tags(fn):
    body = fn.get('body', '')
    hooks = fn.get('hooks_used', []) or []
    sql_patterns = fn.get('sql_patterns', []) or []
    tags = set()

    if has_pattern(body, r'\bwp_enqueue_script\b', r'\bwp_enqueue_style\b'):
        tags.add('asset-enqueuing')
    if has_pattern(body, r'\bregister_block_type\b', r'\bregister_block_type_from_metadata\b'):
        tags.add('block-registration')
    if has_pattern(body, r'\badd_action\b', r'\badd_filter\b'):
        tags.add('hooks')
    if has_pattern(body, r'\bWP_Query\b', r'\bget_posts\b'):
        tags.add('wp-query')
    if has_pattern(body, r'\$wpdb'):
        tags.add('wpdb')
    if sql_patterns:
        tags.add('sql')
    if has_pattern(body, r'\bwp_verify_nonce\b', r'\bcheck_ajax_referer\b', r'\bcheck_admin_referer\b'):
        tags.add('nonce-verification')
    if has_pattern(body, r'\bcurrent_user_can\b'):
        tags.add('capability-check')
    if has_pattern(body, r'\besc_html\b', r'\besc_attr\b', r'\besc_url\b', r'\bwp_kses\b'):
        tags.add('output-escaping')
    if has_pattern(body, r'\bsanitize_text_field\b', r'\bsanitize_email\b', r'\bwp_unslash\b', r'\bintval\b', r'\babsint\b'):
        tags.add('input-sanitization')
    if has_pattern(body, r'\bregister_rest_route\b'):
        tags.add('rest-api')
    if has_pattern(body, r'\bwp_cache_get\b', r'\bget_transient\b', r'\bwp_cache_set\b', r'\bset_transient\b'):
        tags.add('caching')
    if has_pattern(body, r'\bwp_localize_script\b'):
        tags.add('script-localization')
    if has_pattern(body, r'\bregister_post_type\b'):
        tags.add('custom-post-type')
    if has_pattern(body, r'\bregister_taxonomy\b'):
        tags.add('taxonomy')
    if has_pattern(body, r'\badd_shortcode\b'):
        tags.add('shortcode')
    if has_pattern(body, r'\bget_option\b', r'\bupdate_option\b', r'\badd_option\b'):
        tags.add('options-api')
    if has_pattern(body, r'\b__\s*\(', r'\b_e\s*\(', r'\besc_html__\s*\('):
        tags.add('i18n')
    if has_pattern(body, r'\bWP_Filesystem\b', r'\bwp_filesystem\b'):
        tags.add('wp-filesystem')
    if has_pattern(body, r'\bwp_mail\b'):
        tags.add('email')
    if has_pattern(body, r'\bwp_ajax_'):
        tags.add('ajax-handler')
    if has_pattern(body, r'\bget_post_meta\b', r'\bupdate_post_meta\b', r'\badd_post_meta\b'):
        tags.add('post-meta')
    if has_pattern(body, r'\bwp_register_style\b', r'\bwp_register_script\b'):
        tags.add('asset-registration')
    if has_pattern(body, r'\bget_user_meta\b', r'\bupdate_user_meta\b'):
        tags.add('user-meta')

    return sorted(tags)


# ─── main judge ─────────────────────────────────────────────────────────────

def judge_function(fn, idx):
    body = fn.get('body', '') or ''
    source_file = fn.get('source_file', '') or ''
    name = fn.get('function_name', '') or ''
    quality_tier = fn.get('quality_tier', '') or ''
    line_count = fn.get('line_count', 0) or 0

    # Auto-pass core
    if quality_tier == 'core':
        assessment = {
            'function_name': name,
            'file_path': source_file,
            'verdict': 'PASS',
            'scores': {
                'wpcs_compliance': 10, 'sql_safety': 10, 'security': 10,
                'performance': 10, 'wp_api_usage': 10, 'code_quality': 10,
                'dependency_integrity': 10, 'i18n': 10, 'accessibility': 10,
            },
            'critical_failures': [],
            'dependency_chain': fn.get('dependencies', []) or [],
            'training_tags': extract_training_tags(fn),
            'notes': 'WordPress core - auto-passed',
            '_custom_id': f"{fn.get('source_repo','')}_{ idx}_{name}",
        }
        return 'PASS', assessment

    # Score dimensions
    wpcs_score, wpcs_fails = score_wpcs(fn)
    sql_score, sql_fails = score_sql(fn)
    sec_score, sec_fails = score_security(fn)
    perf_score, perf_fails = score_performance(fn)
    api_score, api_fails = score_wp_api(fn)
    cq_score, cq_fails = score_code_quality(fn)
    dep_score, dep_fails = score_dependency_integrity(fn)
    i18n_score, i18n_fails = score_i18n(fn)
    a11y_score, a11y_fails = score_accessibility(fn)

    all_failures = wpcs_fails + sql_fails + sec_fails + perf_fails + api_fails + cq_fails + dep_fails + i18n_fails + a11y_fails

    scores = {
        'wpcs_compliance': wpcs_score,
        'sql_safety': sql_score,
        'security': sec_score,
        'performance': perf_score,
        'wp_api_usage': api_score,
        'code_quality': cq_score,
        'dependency_integrity': dep_score,
        'i18n': i18n_score,
        'accessibility': a11y_score,
    }

    # Verdict determination
    verdict = 'PASS'
    fail_reasons = []

    # Security auto-fail
    if sec_score < 5:
        verdict = 'FAIL'
        fail_reasons.append(f'Security score {sec_score} < 5 (auto-fail threshold)')

    # Any critical failure strings
    critical_keywords = ['CRITICAL:']
    critical_failures = [f for f in all_failures if 'CRITICAL' in f]
    if critical_failures:
        verdict = 'FAIL'

    # All dimensions must be >= 8 (N/A dimensions score 7 — exempt from the threshold)
    # N/A applies when: i18n=7 with no i18n failures, accessibility=7 with no a11y failures
    na_dims = set()
    if i18n_score == 7 and not i18n_fails:
        na_dims.add('i18n')
    if a11y_score == 7 and not a11y_fails:
        na_dims.add('accessibility')

    if verdict == 'PASS':
        for dim, sc in scores.items():
            if dim in na_dims:
                continue  # N/A dimension does not block PASS
            if sc < 8:
                verdict = 'FAIL'
                fail_reasons.append(f'{dim} score {sc} < 8 (minimum threshold)')
                break

    notes_parts = []
    if verdict == 'PASS':
        tags = extract_training_tags(fn)
        notes_parts.append(f"Production-quality WordPress code suitable for training.")
        if tags:
            notes_parts.append(f"Demonstrates: {', '.join(tags)}.")
    else:
        if critical_failures:
            notes_parts.append('; '.join(critical_failures[:2]))
        if fail_reasons:
            notes_parts.append('; '.join(fail_reasons[:2]))

    assessment = {
        'function_name': name,
        'file_path': source_file,
        'verdict': verdict,
        'scores': scores,
        'critical_failures': critical_failures,
        'dependency_chain': fn.get('dependencies', []) or [],
        'training_tags': extract_training_tags(fn),
        'notes': ' '.join(notes_parts),
        '_custom_id': f"{fn.get('source_repo','')}_{ idx}_{name}",
    }
    return verdict, assessment


# ─── process repos ──────────────────────────────────────────────────────────

def process_repo(repo):
    path = os.path.join(EXTRACTED_DIR, f'{repo}.json')
    if not os.path.exists(path):
        print(f'  SKIP {repo}: file not found')
        return

    with open(path) as f:
        functions = json.load(f)

    passed = []
    failed = []
    skipped = 0

    for idx, fn in enumerate(functions):
        line_count = fn.get('line_count', 0) or 0

        if line_count < 5:
            skipped += 1
            continue

        verdict, assessment = judge_function(fn, idx)

        entry = dict(fn)
        entry['assessment'] = assessment
        entry['training_tags'] = assessment['training_tags']

        if verdict == 'PASS':
            passed.append(entry)
        else:
            failed.append(entry)

    # Write results
    passed_path = os.path.join(PASSED_DIR, f'{repo}.json')
    failed_path = os.path.join(FAILED_DIR, f'{repo}.json')

    with open(passed_path, 'w') as f:
        json.dump(passed, f, indent=2)

    with open(failed_path, 'w') as f:
        json.dump(failed, f, indent=2)

    total = len(functions)
    judged = total - skipped
    print(f'  {repo}: {total} total, {skipped} skipped (<5 lines), {judged} judged -> {len(passed)} PASS / {len(failed)} FAIL')
    return len(passed), len(failed)


def main():
    print('WordPress Code Quality Judge')
    print('=' * 60)
    total_pass = 0
    total_fail = 0
    for repo in REPOS:
        print(f'\nProcessing: {repo}')
        result = process_repo(repo)
        if result:
            total_pass += result[0]
            total_fail += result[1]

    print('\n' + '=' * 60)
    print(f'TOTAL: {total_pass} PASS / {total_fail} FAIL')
    print(f'Pass rate: {total_pass/(total_pass+total_fail)*100:.1f}%' if (total_pass+total_fail) else 'No functions judged')


if __name__ == '__main__':
    main()
