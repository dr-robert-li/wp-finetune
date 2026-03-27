#!/usr/bin/env python3
"""
Judge all functions in amp-wp.json against the WordPress code quality rubric.
Splits into passed and failed JSON files.
"""

import json
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted" / "amp-wp.json"
PASSED_FILE = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed" / "amp-wp.json"
FAILED_FILE = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed" / "amp-wp.json"


def has_test_code(func):
    """Detect test/bin/docs code that should be excluded."""
    source_file = func.get('source_file', '')
    # Test files
    if '/test' in source_file.lower() or 'test-' in source_file.lower() or '-test.' in source_file.lower():
        return True
    if source_file.endswith('Test.php') or 'Tests.php' in source_file:
        return True
    # PHPUnit test methods
    body = func.get('body', '')
    if 'PHPUnit' in body or 'extends WP_UnitTestCase' in body or 'extends TestCase' in body:
        return True
    # bin/ scripts - tooling/dev scripts, not plugin code
    if source_file.startswith('bin/'):
        return True
    # docs/ scripts - documentation generation CLI tools, not WordPress plugin code
    if source_file.startswith('docs/'):
        return True
    return False


def is_deprecated_function(func):
    """Check if function is deprecated."""
    docblock = func.get('docblock', '') or ''
    return '@deprecated' in docblock


def score_wpcs(func):
    """Score WordPress Coding Standards compliance."""
    score = 10
    body = func.get('body', '')
    docblock = func.get('docblock', '') or ''
    function_name = func.get('function_name', '')
    class_ctx = func.get('class_context')

    # Check for PHPDoc
    if not docblock or len(docblock.strip()) < 10:
        # Missing docblock entirely = major WPCS fail for public API
        score -= 3
    else:
        # Check for @param tags when function has parameters
        has_params = bool(re.search(r'function\s+\w+\s*\([^)]+\)', body))
        if has_params and '@param' not in docblock:
            score -= 1

        # Check for @return tag
        if '@return' not in docblock and '@void' not in docblock.lower() and 'void' not in docblock:
            # Only penalize if function has a return statement
            if re.search(r'\breturn\s+(?!;)', body):
                score -= 1

        # @since is REQUIRED for standalone public functions (not for methods)
        if not class_ctx:
            if '@since' not in docblock:
                score -= 2  # Strong penalty for missing @since on public functions
        else:
            # For methods, @since is strongly recommended
            if '@since' not in docblock:
                score -= 1

    # Check naming convention (snake_case for functions)
    if function_name and not re.match(r'^[a-z_][a-z0-9_]*$', function_name):
        # Could be a method in camelCase - only penalize standalone functions
        if not class_ctx:
            score -= 2

    # Check for camelCase variable names (WP uses snake_case)
    camel_vars = re.findall(r'\$([a-z]+[A-Z][a-zA-Z]*)\b', body)
    if len(camel_vars) > 3:
        score -= 1

    # Check for proper spacing around operators (basic check)
    # Missing spaces like $a=$b
    bad_spacing = re.findall(r'[a-zA-Z0-9_]\=[a-zA-Z0-9_\'"]', body)
    # Filter out => (array arrow) and == patterns
    bad_spacing = [b for b in bad_spacing if '=>' not in b and '==' not in b]
    if len(bad_spacing) > 2:
        score -= 1

    return max(1, min(10, round(score)))


def score_sql_safety(func):
    """Score SQL safety."""
    body = func.get('body', '')
    sql_patterns = func.get('sql_patterns', [])

    # No SQL patterns - score 10
    if not sql_patterns and 'query' not in body.lower() and '$wpdb' not in body:
        return 10

    score = 10

    # Check for $wpdb->query/get_results etc. with non-wpdb variable interpolation
    # Only flag actual dynamic user-supplied data, not $wpdb->table_name references
    # Pattern: query contains a variable that is NOT $wpdb->something
    dangerous_queries = re.findall(
        r'\$wpdb->\w+\s*\(\s*["\'][^"\']*\$(?!wpdb->)[a-zA-Z_\(]',
        body
    )

    if dangerous_queries:
        # Check if they use prepare()
        has_prepare = 'prepare(' in body or '$wpdb->prepare' in body
        if not has_prepare:
            # Check if it's actually user input (more severe)
            has_user_input = re.search(r'\$_(GET|POST|REQUEST|COOKIE)\s*\[', body)
            if has_user_input:
                return 1  # Critical fail - user input in query without prepare
            else:
                score -= 3  # Dynamic but not necessarily user input

    # Check for prepare() usage when there is actual user input
    if '$wpdb' in body:
        has_user_input = re.search(r'\$_(GET|POST|REQUEST|COOKIE)\s*\[', body)
        has_dynamic_query = re.search(r'\$wpdb->(?:query|get_results|get_row|get_var|get_col|insert|update|delete)\s*\(', body)
        if has_dynamic_query and has_user_input and 'prepare(' not in body:
            return 1  # Critical fail - user input in query without prepare

    # Check if prepare is used correctly
    if 'prepare(' in body:
        # Good usage
        score = 10

    # Hardcoded wp_ prefix (should use $wpdb->prefix)
    hardcoded_prefix = re.findall(r'["\']wp_[a-z_]+["\']', body)
    if hardcoded_prefix:
        score -= 2

    return max(1, min(10, round(score)))


def score_security(func):
    """Score security."""
    body = func.get('body', '')
    function_name = func.get('function_name', '')

    score = 10
    critical_failures = []

    # Check for nonce verification in form handlers / AJAX handlers
    is_ajax_handler = 'wp_ajax' in str(func.get('hooks_used', []))
    is_form_handler = any(pattern in body for pattern in [
        '$_POST', '$_GET', '$_REQUEST', 'wp_verify_nonce', 'check_ajax_referer'
    ])

    if is_form_handler:
        has_nonce = any(pattern in body for pattern in [
            'wp_verify_nonce', 'check_ajax_referer', 'check_admin_referer'
        ])
        if '$_POST' in body and not has_nonce and 'nonce' not in body.lower():
            # Check if it's just reading, not state-changing
            state_changing = any(p in body for p in ['update_option', 'wp_insert_post', 'wp_update_post',
                                                       'delete_option', 'wp_delete_post', 'update_user_meta'])
            if state_changing:
                score -= 4
                critical_failures.append('Missing nonce verification on state-changing handler')

    # Check for capability checks
    has_capability_check = any(p in body for p in ['current_user_can', 'is_admin', 'is_super_admin'])
    if is_ajax_handler and not has_capability_check:
        score -= 2

    # Check for unescaped output
    has_echo = 'echo ' in body or 'print ' in body or '?>' in body
    if has_echo:
        # Check for escaping
        has_escaping = any(p in body for p in ['esc_html', 'esc_attr', 'esc_url', 'wp_kses', 'esc_js',
                                                 'esc_textarea', 'intval(', 'absint(', 'number_format'])
        unescaped_vars = re.findall(r'echo\s+\$(?!wpdb)', body)
        if unescaped_vars and not has_escaping:
            score -= 3
            critical_failures.append('Unescaped output of variable data')

    # Check for extract() on untrusted data
    if 'extract(' in body:
        if '$_POST' in body or '$_GET' in body or '$_REQUEST' in body:
            score -= 5
            critical_failures.append('extract() used on untrusted user input')
        else:
            score -= 1

    # Check for eval()
    if re.search(r'\beval\s*\(', body):
        score -= 6
        critical_failures.append('eval() usage detected')

    # Check for direct file operations
    if re.search(r'\b(fopen|fwrite|file_put_contents|file_get_contents)\s*\(', body):
        if 'WP_Filesystem' not in body and 'wp_filesystem' not in body.lower():
            # Not necessarily a fail if reading only
            if any(p in body for p in ['fwrite', 'file_put_contents']):
                score -= 3

    # Check for SQL injection via direct concatenation
    direct_sql_concat = re.findall(
        r'(?:query|get_results|get_row|get_var)\s*\(\s*["\'][^"\']*\'\s*\.\s*\$(?!wpdb->prepare)',
        body
    )
    if direct_sql_concat:
        score -= 4
        critical_failures.append('Potential SQL injection via string concatenation')

    return max(1, min(10, round(score))), critical_failures


def score_performance(func):
    """Score performance."""
    body = func.get('body', '')
    score = 10
    critical_failures = []

    # Check for queries in loops
    has_loop = bool(re.search(r'\b(?:foreach|while|for)\s*\(', body))
    has_query_in_loop = False

    if has_loop:
        # Simple heuristic: check if db calls appear after loop start
        loop_pos = re.search(r'\b(?:foreach|while|for)\s*\(', body)
        if loop_pos:
            after_loop = body[loop_pos.start():]
            db_in_loop = re.search(r'\$wpdb->|get_post_meta\(|get_option\(|WP_Query\(|new WP_Query', after_loop)
            if db_in_loop:
                # More context needed - check if it's inside the loop body
                loop_body_match = re.search(r'\b(?:foreach|while|for)\s*\([^)]+\)\s*\{(.+?)(?=\}[^}]*(?:foreach|while|for|\Z))', after_loop, re.DOTALL)
                if loop_body_match:
                    loop_body = loop_body_match.group(1)
                    if re.search(r'\$wpdb->|get_post_meta\(|WP_Query\(|new WP_Query', loop_body):
                        score -= 3
                        critical_failures.append('Database query inside loop (N+1 pattern)')
                        has_query_in_loop = True

    # Check for SELECT * (but be lenient for non-meta tables)
    if re.search(r'SELECT\s+\*\s+FROM', body, re.IGNORECASE):
        score -= 1

    # Check for caching of expensive operations
    # (Give credit if they use transients or object cache)
    has_caching = any(p in body for p in ['get_transient', 'set_transient', 'wp_cache_get', 'wp_cache_set'])

    # If there are db calls but no caching - mild penalty for repeated-call functions
    db_calls = re.findall(r'\$wpdb->|new WP_Query', body)
    if len(db_calls) > 2 and not has_caching:
        score -= 1

    return max(1, min(10, round(score))), critical_failures


def score_wp_api(func):
    """Score WordPress API usage."""
    body = func.get('body', '')
    score = 10
    critical_failures = []

    # Check for raw SQL where WP_Query could be used
    raw_post_query = re.search(
        r'\$wpdb->(?:get_results|get_row|query).*?(?:FROM\s+["\']?\{?\$wpdb->prefix\}?posts)',
        body, re.IGNORECASE | re.DOTALL
    )
    if raw_post_query and 'WP_Query' not in body:
        score -= 2

    # Check for proper REST endpoint registration
    if 'register_rest_route' in body:
        if 'permission_callback' not in body:
            score -= 4
            critical_failures.append('REST route missing permission_callback')

    # Check hooks have correct argument counts
    hook_registrations = re.findall(
        r'add_(?:action|filter)\s*\(\s*["\'][^"\']+["\']\s*,\s*[^,\)]+(?:,\s*\d+)?(?:,\s*(\d+))?\s*\)',
        body
    )

    # Check for deprecated functions
    deprecated = ['get_currentuserinfo', 'get_profile', 'the_attachment_link', 'wp_get_post_tags']
    for dep in deprecated:
        if dep in body:
            score -= 2
            break

    # Check for autoload consideration with large option values
    if 'add_option' in body or 'update_option' in body:
        # If storing arrays/objects, should consider autoload
        if re.search(r'(?:add|update)_option\s*\([^,]+,\s*(?:array|\[|\$)', body):
            # Large data storage - mild concern
            if 'autoload' not in body and 'false' not in body:
                score -= 0.5

    return max(1, min(10, round(score))), critical_failures


def score_code_quality(func):
    """Score code quality."""
    body = func.get('body', '')
    function_name = func.get('function_name', '')
    score = 10
    critical_failures = []

    # Check for debug statements
    debug_patterns = ['var_dump(', 'var_export(', 'print_r(', 'die(', 'exit(']
    has_debug = False
    for pat in debug_patterns:
        if pat in body:
            # Check if it's in a conditional debug context
            # Allow die() if it's in wp_send_json pattern or similar
            if pat == 'die(' and 'wp_die(' in body:
                continue
            if pat != 'die(' and pat != 'exit(':
                score -= 2
                has_debug = True
                break

    # Check for error_log() in production paths
    if 'error_log(' in body:
        score -= 1

    # Check for commented-out code blocks
    commented_code = re.findall(r'//\s*(?:echo|var_dump|\$\w+\s*=|\$\w+->)', body)
    if len(commented_code) > 2:
        score -= 1

    # Check for null/empty checks
    has_complex_logic = len(body) > 200
    has_null_checks = any(p in body for p in ['isset(', 'empty(', 'is_null(', '=== null', '!== null',
                                               '=== false', '!== false', 'is_wp_error('])
    if has_complex_logic and not has_null_checks:
        score -= 1

    # Check function length - very long functions do too many things
    line_count = func.get('line_count', 0)
    if line_count > 150:
        score -= 1
    if line_count > 300:
        score -= 1

    # Check for magic numbers/strings without explanation
    magic_numbers = re.findall(r'\b(?<!\$)(?<!\->)(?<!::)(\d{4,})\b', body)
    if len(magic_numbers) > 3:
        score -= 0.5

    # Check for silent error swallowing - @ operator
    error_suppression = re.findall(r'@\w+\s*\(', body)
    if len(error_suppression) > 2:
        score -= 1

    return max(1, min(10, round(score))), critical_failures


def score_dependency_integrity(func):
    """Score dependency chain integrity."""
    dependencies = func.get('dependencies', [])
    body = func.get('body', '')
    score = 10

    # Remove self-reference from dependencies
    func_name = func.get('function_name', '')
    external_deps = [d for d in dependencies if d != func_name]

    # Check for direct require/include of vendor files
    if re.search(r'(?:require|include)(?:_once)?\s*\([^)]*vendor', body):
        score -= 2

    # Check for circular dependency patterns (function calling itself without recursion guard)
    if func_name in body:
        # Self-reference
        recursive_calls = re.findall(r'\b' + re.escape(func_name) + r'\s*\(', body)
        if len(recursive_calls) > 0:
            # Check for recursion guards
            has_guard = any(p in body for p in ['static $', 'return;', '$depth', '$recursion'])
            if not has_guard and len(recursive_calls) > 0:
                score -= 1  # Mild concern, might be intentional recursion

    # Many custom dependencies without clear documentation
    custom_deps = [d for d in external_deps if not d.startswith('wp_') and
                   not d.startswith('get_') and not d.startswith('the_') and
                   not d.startswith('is_') and not d.startswith('has_') and
                   not d.startswith('esc_') and not d.startswith('add_') and
                   not d.startswith('do_') and not d.startswith('apply_') and
                   not d.startswith('__') and not d.startswith('_e') and
                   not d.startswith('_n') and not d.startswith('_x')]

    return max(1, min(10, round(score)))


def score_i18n(func):
    """Score internationalization."""
    body = func.get('body', '')

    # Check if function has any user-facing output
    has_output = 'echo ' in body or 'print ' in body or '?>' in body or 'return' in body
    has_strings = bool(re.search(r'["\'][A-Z][a-z\s]+["\']', body))

    # If no output or strings, N/A score of 7
    if not has_output:
        return 7

    score = 10

    # Check for hardcoded English strings in output
    # Find echo statements with string literals
    echo_strings = re.findall(r'echo\s+["\']([A-Z][^"\']*)["\']', body)
    if echo_strings:
        score -= 3

    # Check for translation functions
    has_translation = any(p in body for p in ['__(', '_e(', 'esc_html__(', 'esc_html_e(',
                                               '_n(', '_x(', '_nx(', 'esc_attr__('])

    # Check text domain consistency
    text_domains = re.findall(r'["\'](?:__)\s*\(\s*[^,]+,\s*["\']([^"\']+)["\']', body)
    amp_text_domains = re.findall(r',\s*["\']amp["\']', body)
    other_text_domains = [td for td in text_domains if td != 'amp']

    if other_text_domains and not amp_text_domains:
        score -= 1  # Inconsistent text domain

    # Check for sprintf with translated strings
    has_sprintf = 'sprintf(' in body or 'printf(' in body
    string_concat = re.findall(r'["\'][^"\']*\'\s*\.\s*\$', body)
    if string_concat and has_output:
        # Concatenation instead of sprintf might lose translation ability
        score -= 1

    # If output but no translation at all for string content
    if has_strings and not has_translation and has_output:
        # Check if strings are things like HTML attributes, not user-facing
        if re.search(r'echo\s+["\'][A-Z]', body):
            score -= 2

    return max(1, min(10, round(score)))


def score_accessibility(func):
    """Score accessibility."""
    body = func.get('body', '')

    # If no HTML output, N/A score of 7
    has_html = bool(re.search(r'<(?:div|span|input|form|button|label|select|textarea|img|a)\b', body, re.IGNORECASE))
    if not has_html:
        return 7

    score = 10
    critical_failures = []

    # Check for form inputs without labels
    inputs = re.findall(r'<input[^>]*>', body, re.IGNORECASE)
    labels = re.findall(r'<label[^>]*>', body, re.IGNORECASE)

    # If there are inputs but no labels
    if inputs:
        text_inputs = [i for i in inputs if 'type="hidden"' not in i and 'type="submit"' not in i
                      and 'type="button"' not in i and 'type="reset"' not in i]
        if text_inputs and not labels:
            # Check for aria-label
            if 'aria-label' not in body and 'aria-labelledby' not in body:
                score -= 3
                critical_failures.append('Form inputs without labels')

    # Check for images without alt
    imgs = re.findall(r'<img[^>]*>', body, re.IGNORECASE)
    if imgs:
        for img in imgs:
            if 'alt=' not in img:
                score -= 2
                critical_failures.append('Image without alt attribute')
                break

    # Check for interactive elements keyboard accessibility
    if re.search(r'<button|<a\s', body, re.IGNORECASE):
        # Generally OK if using proper HTML elements
        pass

    # Check for screen reader text
    if re.search(r'class=["\'].*screen-reader-text', body):
        score = min(10, score + 0.5)

    # Check for aria attributes on custom interactive elements
    custom_interactive = re.search(r'<div[^>]*(?:onclick|onkeypress)', body, re.IGNORECASE)
    if custom_interactive:
        if 'role=' not in body:
            score -= 2

    return max(1, min(10, round(score))), critical_failures


def assess_function(func):
    """Full assessment of a single function."""
    function_name = func.get('function_name', 'unknown')
    source_file = func.get('source_file', '')
    body = func.get('body', '')

    # Test/bin/docs code check
    if has_test_code(func):
        return {
            "function_name": function_name,
            "file_path": source_file,
            "verdict": "FAIL",
            "scores": {
                "wpcs_compliance": 1,
                "sql_safety": 7,
                "security": 7,
                "performance": 7,
                "wp_api_usage": 7,
                "code_quality": 1,
                "dependency_integrity": 7,
                "i18n": 7,
                "accessibility": 7
            },
            "critical_failures": ["test/bin/docs code, excluded from training data"],
            "dependency_chain": [],
            "training_tags": [],
            "notes": "Test, bin script, or docs tooling code, excluded from training data per rubric."
        }

    # Deprecated function check - rubric: legacy code gets no special treatment
    if is_deprecated_function(func):
        return {
            "function_name": function_name,
            "file_path": source_file,
            "verdict": "FAIL",
            "scores": {
                "wpcs_compliance": 4,
                "sql_safety": 7,
                "security": 7,
                "performance": 7,
                "wp_api_usage": 4,
                "code_quality": 5,
                "dependency_integrity": 5,
                "i18n": 7,
                "accessibility": 7
            },
            "critical_failures": ["deprecated function - legacy compatibility code fails per rubric"],
            "dependency_chain": func.get('dependencies', []),
            "training_tags": ["deprecated"],
            "notes": "Function marked @deprecated. Per rubric, legacy compatibility code gets no special treatment and fails."
        }

    # Score all dimensions
    wpcs = score_wpcs(func)
    sql = score_sql_safety(func)
    security, sec_failures = score_security(func)
    perf, perf_failures = score_performance(func)
    wp_api, api_failures = score_wp_api(func)
    quality, qual_failures = score_code_quality(func)
    dep_int = score_dependency_integrity(func)
    i18n = score_i18n(func)

    # Accessibility scoring
    acc_result = score_accessibility(func)
    if isinstance(acc_result, tuple):
        acc, acc_failures = acc_result
    else:
        acc = acc_result
        acc_failures = []

    all_critical_failures = sec_failures + perf_failures + api_failures + qual_failures + acc_failures

    # Check for SQL critical failure (score 1 = auto-fail)
    sql_critical = sql == 1

    scores = {
        "wpcs_compliance": wpcs,
        "sql_safety": sql,
        "security": security,
        "performance": perf,
        "wp_api_usage": wp_api,
        "code_quality": quality,
        "dependency_integrity": dep_int,
        "i18n": i18n,
        "accessibility": acc
    }

    # PASS requires ALL >= 8 and no critical failures, security >= 5
    # NOTE: i18n and accessibility use N/A score of 7 when not applicable - treat 7 as passing for those dims
    security_auto_fail = security < 5
    sql_auto_fail = sql_critical
    na_dims = {'i18n', 'accessibility'}
    all_pass_threshold = all(
        v >= 8 or (k in na_dims and v == 7)
        for k, v in scores.items()
    )
    has_critical = len(all_critical_failures) > 0

    if security_auto_fail:
        verdict = "FAIL"
    elif sql_auto_fail:
        verdict = "FAIL"
        all_critical_failures.insert(0, "SQL injection vector - automatic FAIL")
    elif has_critical:
        verdict = "FAIL"
    elif not all_pass_threshold:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    # Build training tags
    training_tags = []
    if '$wpdb' in body:
        training_tags.append('wpdb')
    if 'WP_Query' in body:
        training_tags.append('wp-query')
    if 'get_option' in body or 'update_option' in body:
        training_tags.append('options-api')
    if 'add_action' in body or 'add_filter' in body:
        training_tags.append('hooks')
    if 'register_rest_route' in body:
        training_tags.append('rest-api')
    if 'wp_enqueue_script' in body or 'wp_enqueue_style' in body:
        training_tags.append('enqueue')
    if 'get_transient' in body or 'set_transient' in body:
        training_tags.append('transients')
    if 'wp_cache_get' in body or 'wp_cache_set' in body:
        training_tags.append('object-cache')
    if any(p in body for p in ['esc_html', 'esc_attr', 'esc_url', 'wp_kses']):
        training_tags.append('output-escaping')
    if 'wp_verify_nonce' in body or 'check_ajax_referer' in body:
        training_tags.append('nonces')
    if 'current_user_can' in body:
        training_tags.append('capabilities')
    if '__(' in body or '_e(' in body or 'esc_html__' in body:
        training_tags.append('i18n')
    if 'register_post_type' in body:
        training_tags.append('custom-post-types')
    if 'register_taxonomy' in body:
        training_tags.append('taxonomies')
    if 'WP_Error' in body:
        training_tags.append('wp-error')
    if func.get('class_context'):
        training_tags.append('oop')

    # Build dependency chain
    func_deps = func.get('dependencies', [])
    dep_chain = [d for d in func_deps if d != function_name]

    # Build notes
    failing_dims = [k for k, v in scores.items() if v < 8]
    if verdict == 'PASS':
        notes = f"High-quality function demonstrating good WordPress practices. "
        if training_tags:
            notes += f"Covers: {', '.join(training_tags[:5])}."
    else:
        notes = f"FAIL - "
        if security_auto_fail:
            notes += f"Security auto-fail (score: {security}). "
        if sql_auto_fail:
            notes += "SQL injection risk. "
        if all_critical_failures:
            notes += f"Critical: {'; '.join(all_critical_failures[:2])}. "
        if failing_dims:
            notes += f"Low scores in: {', '.join(failing_dims)}."

    return {
        "function_name": function_name,
        "file_path": source_file,
        "verdict": verdict,
        "scores": scores,
        "critical_failures": all_critical_failures,
        "dependency_chain": dep_chain[:20],  # Limit to 20 deps
        "training_tags": training_tags,
        "notes": notes.strip()
    }


def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        functions = json.load(f)

    print(f"Processing {len(functions)} functions...")

    passed = []
    failed = []

    for i, func in enumerate(functions):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(functions)}...")

        assessment = assess_function(func)

        # Add assessment to the function object
        func_copy = dict(func)
        func_copy['assessment'] = assessment

        if assessment['verdict'] == 'PASS':
            passed.append(func_copy)
        else:
            failed.append(func_copy)

    print(f"\nResults:")
    print(f"  PASSED: {len(passed)}")
    print(f"  FAILED: {len(failed)}")
    print(f"  Pass rate: {len(passed)/len(functions)*100:.1f}%")

    print(f"\nWriting passed functions to {PASSED_FILE}...")
    with open(PASSED_FILE, 'w', encoding='utf-8') as f:
        json.dump(passed, f, indent=2, ensure_ascii=False)

    print(f"Writing failed functions to {FAILED_FILE}...")
    with open(FAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    print("\nDone!")

    # Show sample of failures by category
    fail_reasons = {}
    for func in failed:
        assessment = func['assessment']
        for cf in assessment['critical_failures']:
            fail_reasons[cf] = fail_reasons.get(cf, 0) + 1
        # Check for dimension fails
        for dim, score in assessment['scores'].items():
            if score < 8:
                key = f"Low {dim} ({score})"
                # Just track dims that drop below threshold without critical
                if not assessment['critical_failures']:
                    fail_reasons[f"Dim: {dim}<8"] = fail_reasons.get(f"Dim: {dim}<8", 0) + 1

    print("\nTop failure reasons:")
    sorted_reasons = sorted(fail_reasons.items(), key=lambda x: -x[1])
    for reason, count in sorted_reasons[:15]:
        print(f"  {count:4d}  {reason}")


if __name__ == '__main__':
    main()
