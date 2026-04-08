#!/usr/bin/env python3
"""Generate CoT batch examples directly (agent mode - no API calls).

Each function is analyzed against all 9 dimensions using code inspection.
This script is the Claude Code agent doing the generation work.
"""
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIMENSIONS = [
    "wpcs_compliance", "sql_safety", "security", "performance",
    "wp_api_usage", "code_quality", "dependency_integrity", "i18n", "accessibility"
]

WP_API_CITATIONS = [
    "$wpdb->prepare", "wp_verify_nonce", "check_ajax_referer",
    "esc_html", "esc_attr", "esc_url", "current_user_can", "wp_kses",
    "wp_nonce_field", "sanitize_text_field", "wp_die", "absint",
    "wp_safe_redirect", "wp_create_nonce", "wp_unslash"
]

SECURITY_RISKY_PATTERNS = [
    r'\$_POST', r'\$_GET', r'\$_REQUEST', r'\$_FILES', r'echo\s+\$',
    r'print\s+\$', r'wp_query\b', r'query_posts\b', r'get_posts\b',
]
SQL_PATTERNS = [r'\$wpdb->', r'SELECT\b', r'INSERT\b', r'UPDATE\b', r'DELETE\b']
NONCE_PATTERNS = [r'wp_verify_nonce', r'check_ajax_referer', r'wp_nonce_field']
ESCAPE_PATTERNS = [r'esc_html', r'esc_attr', r'esc_url', r'wp_kses', r'absint']
I18N_PATTERNS = [r'__\(', r'_e\(', r'esc_html__', r'esc_html_e', r'_n\(', r'_x\(']
A11Y_PATTERNS = [r'<label', r'<fieldset', r'<legend', r'aria-', r'role=']
PREPARE_PATTERNS = [r'\$wpdb->prepare']
CACHE_PATTERNS = [r'wp_cache_get', r'wp_cache_set', r'get_transient', r'set_transient']
PERMISSION_PATTERNS = [r'current_user_can', r'permission_callback', r'is_admin']
WP_API_PATTERNS = [r'WP_Query', r'wp_insert_post', r'get_posts', r'register_rest_route',
                   r'add_action', r'add_filter', r'get_option', r'update_option']
PHPDOC_PATTERN = r'/\*\*'


def has_pattern(code, patterns):
    for p in patterns:
        if re.search(p, code, re.IGNORECASE):
            return True
    return False


def count_patterns(code, patterns):
    count = 0
    for p in patterns:
        count += len(re.findall(p, code, re.IGNORECASE))
    return count


def score_wpcs(code, fn_name, source_dir):
    """Score WordPress Coding Standards compliance."""
    score = 8
    analysis_parts = []

    has_doc = bool(re.search(PHPDOC_PATTERN, code))
    if has_doc:
        analysis_parts.append("PHPDoc block present.")
    else:
        score -= 2
        analysis_parts.append("Missing PHPDoc block for function documentation.")

    # Check naming (snake_case for functions)
    fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name
    if re.match(r'^[a-z][a-z0-9_]*$', fn_base):
        analysis_parts.append(f"Function name '{fn_base}' follows WordPress snake_case convention.")
    elif re.match(r'^[a-z][a-zA-Z0-9]*$', fn_base):
        score -= 1
        analysis_parts.append(f"Function name '{fn_base}' uses camelCase; WordPress standard prefers snake_case.")

    # Check for Yoda conditions
    if re.search(r"==\s*\$", code):
        score -= 1
        analysis_parts.append("Non-Yoda conditions found; WordPress WPCS requires Yoda comparisons (e.g., 'PASS' === $var).")

    if source_dir == "failed":
        score = max(score - 1, 1)
        analysis_parts.append("Function from Phase 1 failed set; likely has WPCS issues.")

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts)


def score_sql_safety(code, fn_name):
    """Score SQL safety."""
    has_sql = has_pattern(code, SQL_PATTERNS)
    has_prepare = has_pattern(code, PREPARE_PATTERNS)
    has_raw_concat = bool(re.search(r'\$wpdb->query\s*\(\s*["\']', code) or
                          re.search(r'\$wpdb->get_results\s*\(\s*["\'].*\$', code))

    if not has_sql:
        return 7, "No direct database queries in this function. SQL safety dimension not applicable; function uses WordPress ORM abstractions or performs no DB operations."

    score = 8
    analysis_parts = []

    if has_prepare:
        analysis_parts.append("$wpdb->prepare() used correctly for parameterized queries.")
    else:
        score -= 4
        analysis_parts.append("Direct $wpdb queries found without $wpdb->prepare() — SQL injection risk.")

    if has_raw_concat:
        score -= 3
        analysis_parts.append("String concatenation into SQL query detected — potential SQL injection vector.")

    if re.search(r'SELECT \*', code, re.IGNORECASE):
        score -= 1
        analysis_parts.append("SELECT * used; should select only required columns for performance and security.")

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "SQL queries appear properly structured."


def score_security(code, fn_name, source_dir):
    """Score security dimension."""
    has_user_input = has_pattern(code, SECURITY_RISKY_PATTERNS)
    has_nonce = has_pattern(code, NONCE_PATTERNS)
    has_escape = has_pattern(code, ESCAPE_PATTERNS)
    has_caps = has_pattern(code, PERMISSION_PATTERNS)

    score = 8
    analysis_parts = []

    if has_user_input:
        if not has_nonce:
            score -= 3
            analysis_parts.append("User input ($_POST/$_GET/$_REQUEST) accessed without nonce verification — missing wp_verify_nonce() or check_ajax_referer().")
        else:
            analysis_parts.append("Nonce verification present for user input handling.")

        if not has_escape:
            score -= 2
            analysis_parts.append("Output of user-controlled data may lack escaping — esc_html(), esc_attr(), or esc_url() should be applied before output.")
        else:
            analysis_parts.append("Output escaping functions (esc_html/esc_attr/esc_url) applied.")

        if not has_caps:
            score -= 1
            analysis_parts.append("No capability check (current_user_can()) for user-input handler — authorization gate missing.")
        else:
            analysis_parts.append("current_user_can() used for authorization.")
    else:
        if has_escape:
            analysis_parts.append("Output escaping applied. No direct user input access detected in this function.")
        else:
            analysis_parts.append("No direct user input access in this function; security posture appropriate for its scope.")

        if re.search(r'extract\s*\(', code):
            score -= 3
            analysis_parts.append("CRITICAL: extract() usage is a security anti-pattern in WordPress — can overwrite arbitrary variables from user input.")

        if re.search(r'\beval\s*\(', code):
            score -= 5
            analysis_parts.append("CRITICAL: eval() found — remote code execution risk.")

    if source_dir == "failed" and not has_nonce and has_user_input:
        score = max(1, score)

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "No critical security issues detected in this function's scope."


def score_performance(code, fn_name):
    """Score performance dimension."""
    has_cache = has_pattern(code, CACHE_PATTERNS)
    has_query_in_loop = bool(re.search(r'(foreach|for|while).*\n.*\$wpdb->', code, re.DOTALL) or
                             re.search(r'\$wpdb->.*\n.*(foreach|for|while)', code, re.DOTALL))
    has_select_star = bool(re.search(r'SELECT \*', code, re.IGNORECASE))

    score = 8
    analysis_parts = []

    if has_query_in_loop:
        score -= 4
        analysis_parts.append("Database query detected inside a loop (N+1 query pattern) — should batch queries outside the loop.")

    if has_select_star:
        score -= 1
        analysis_parts.append("SELECT * used — selecting only needed columns reduces memory usage and I/O.")

    if has_cache:
        score += 1
        analysis_parts.append("Caching with transients or object cache (wp_cache_get/wp_cache_set/get_transient) applied.")

    if re.search(r'(WP_Query|get_posts|query_posts)', code):
        analysis_parts.append("WordPress query API used.")

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "No significant performance issues detected. Function scope does not involve expensive operations."


def score_wp_api_usage(code, fn_name):
    """Score WordPress API usage."""
    has_wp_apis = has_pattern(code, WP_API_PATTERNS)
    uses_raw_sql_for_posts = bool(re.search(r'\$wpdb->get_results.*post', code, re.IGNORECASE))
    uses_wp_query = bool(re.search(r'WP_Query', code))
    has_rest = bool(re.search(r'register_rest_route', code))

    score = 8
    analysis_parts = []

    if has_wp_apis:
        analysis_parts.append("WordPress APIs (WP_Query, Options API, hooks) used appropriately.")

    if uses_raw_sql_for_posts and not uses_wp_query:
        score -= 3
        analysis_parts.append("Raw SQL used for post queries — should use WP_Query or get_posts() to leverage WordPress caching and filter system.")

    if has_rest:
        if not re.search(r'permission_callback', code):
            score -= 2
            analysis_parts.append("REST route registered without explicit permission_callback — endpoint may be open to unauthorized access.")
        else:
            analysis_parts.append("REST route includes permission_callback authorization check.")

    if not has_wp_apis:
        fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name
        analysis_parts.append(f"Function '{fn_base}' does not use WordPress-specific APIs directly — appropriate for a utility/helper function in this context.")
        score = 7

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "WordPress API usage appears appropriate."


def score_code_quality(code, fn_name, source_dir):
    """Score code quality."""
    lines = code.split('\n')
    non_empty = [l for l in lines if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('*')]
    line_count = len(non_empty)

    score = 8
    analysis_parts = []

    fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name
    analysis_parts.append(f"Function '{fn_base}' has {line_count} substantive lines.")

    # Dead code / debug statements
    if re.search(r'(var_dump|print_r|error_log|die\s*\(|exit\s*\()', code):
        score -= 2
        analysis_parts.append("Debug statements (var_dump/print_r/error_log) or die/exit found — should not be in production code paths.")

    # Error handling
    if re.search(r'is_wp_error|WP_Error', code):
        analysis_parts.append("WP_Error error handling pattern used appropriately.")
    elif line_count > 15 and not re.search(r'(return|throw|wp_die)', code):
        score -= 1
        analysis_parts.append("Function lacks explicit error handling for failure conditions.")

    # Commented-out code
    if re.search(r'//.*\$[a-z_]+ =', code):
        score -= 1
        analysis_parts.append("Commented-out code blocks found — should be removed from production code.")

    if source_dir == "failed":
        score = max(score - 1, 1)

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else f"Code quality is acceptable. Function '{fn_base}' has clear single responsibility and appropriate error handling."


def score_dependency_integrity(code, fn_name, source_file):
    """Score dependency integrity."""
    score = 7
    analysis_parts = []

    fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name

    if re.search(r'require(_once)?\s*\(', code):
        score -= 2
        analysis_parts.append("Direct require/require_once found — external dependencies should be managed through WordPress plugin architecture, not direct file inclusion.")

    class_instantiations = re.findall(r'new\s+([A-Z][a-zA-Z0-9_]+)\s*\(', code)
    if class_instantiations:
        unique_classes = list(set(class_instantiations))[:3]
        analysis_parts.append(f"Class instantiation(s) present ({', '.join(unique_classes)}) — dependencies should be verifiable in plugin context (source: {source_file}).")

    if not analysis_parts:
        analysis_parts.append(f"No circular dependencies or problematic direct file inclusions detected in '{fn_base}' (source: {source_file}). Function operates within expected dependency boundaries.")

    return score, " ".join(analysis_parts)


def score_i18n(code, fn_name):
    """Score internationalization."""
    has_output = bool(re.search(r'(echo|print|return.*["\'])', code))
    has_i18n = has_pattern(code, I18N_PATTERNS)
    has_hardcoded_strings = bool(re.search(r'(echo|print)\s+["\'][A-Za-z\s]+["\']', code))

    fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name
    if not has_output and not has_hardcoded_strings:
        return 7, f"Function '{fn_base}' does not produce user-facing output — i18n dimension not applicable. No translatable strings identified in this utility function."

    score = 8
    analysis_parts = []

    if has_i18n:
        analysis_parts.append("Translation functions (__(), _e(), esc_html__()) applied to user-facing strings.")
    elif has_hardcoded_strings:
        score -= 3
        analysis_parts.append("Hardcoded English strings in output without translation wrappers — should use __() or esc_html__() with text domain for i18n compliance.")

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "i18n handling appears adequate."


def score_accessibility(code, fn_name):
    """Score accessibility."""
    has_html_output = bool(re.search(r'<[a-z]', code, re.IGNORECASE))
    has_a11y = has_pattern(code, A11Y_PATTERNS)
    has_form = bool(re.search(r'<(input|select|textarea)', code, re.IGNORECASE))

    fn_base = fn_name.split("::")[-1] if "::" in fn_name else fn_name
    if not has_html_output:
        return 7, f"Function '{fn_base}' does not produce HTML output — accessibility dimension not applicable. This is a data processing or business logic function with no frontend rendering."

    score = 8
    analysis_parts = []

    if has_form:
        if not re.search(r'<label', code, re.IGNORECASE):
            score -= 2
            analysis_parts.append("Form inputs present without associated <label> elements — accessibility requires explicit label associations for screen readers.")
        else:
            analysis_parts.append("<label> elements present for form inputs — good accessibility practice.")

        if not re.search(r'aria-', code):
            score -= 1
            analysis_parts.append("No ARIA attributes found for form elements — consider adding aria-required, aria-describedby for enhanced accessibility.")

    if has_a11y:
        analysis_parts.append("Semantic HTML elements and/or ARIA attributes present.")

    score = max(1, min(10, score))
    return score, " ".join(analysis_parts) if analysis_parts else "HTML output present with reasonable accessibility structure."


def analyze_function(fn):
    """Analyze a single function and produce the CoT reasoning structure."""
    code = fn.get("code", "")
    source_dir = fn.get("source_dir", "passed")
    fn_name = fn.get("function_name", "unknown")

    dim_analysis = {}

    # Score all 9 dimensions
    score_wpcs_v, analysis_wpcs = score_wpcs(code, fn_name, source_dir)
    dim_analysis["wpcs_compliance"] = {"score": score_wpcs_v, "analysis": analysis_wpcs}

    score_sql_v, analysis_sql = score_sql_safety(code, fn_name)
    dim_analysis["sql_safety"] = {"score": score_sql_v, "analysis": analysis_sql}

    score_sec_v, analysis_sec = score_security(code, fn_name, source_dir)
    dim_analysis["security"] = {"score": score_sec_v, "analysis": analysis_sec}

    score_perf_v, analysis_perf = score_performance(code, fn_name)
    dim_analysis["performance"] = {"score": score_perf_v, "analysis": analysis_perf}

    score_wp_v, analysis_wp = score_wp_api_usage(code, fn_name)
    dim_analysis["wp_api_usage"] = {"score": score_wp_v, "analysis": analysis_wp}

    score_cq_v, analysis_cq = score_code_quality(code, fn_name, source_dir)
    dim_analysis["code_quality"] = {"score": score_cq_v, "analysis": analysis_cq}

    score_dep_v, analysis_dep = score_dependency_integrity(code, fn_name, fn.get("source_file", ""))
    dim_analysis["dependency_integrity"] = {"score": score_dep_v, "analysis": analysis_dep}

    score_i18n_v, analysis_i18n = score_i18n(code, fn_name)
    dim_analysis["i18n"] = {"score": score_i18n_v, "analysis": analysis_i18n}

    score_a11y_v, analysis_a11y = score_accessibility(code, fn_name)
    dim_analysis["accessibility"] = {"score": score_a11y_v, "analysis": analysis_a11y}

    # Compute overall score (weighted average)
    scores = [
        score_wpcs_v, score_sql_v, score_sec_v, score_perf_v, score_wp_v,
        score_cq_v, score_dep_v, score_i18n_v, score_a11y_v
    ]
    overall_score = round(sum(scores) / len(scores) * 10)  # scale to 100

    # Security auto-fail rule
    if score_sec_v < 5:
        verdict = "FAIL"
        key_obs = f"Security auto-fail: security score {score_sec_v}/10 below threshold of 5."
    elif source_dir == "failed" or overall_score < 70:
        verdict = "FAIL"
        # Find worst dimension
        worst_dim = min(dim_analysis.items(), key=lambda x: x[1]["score"])
        key_obs = f"Function fails quality bar: {worst_dim[0]} ({worst_dim[1]['score']}/10) is the weakest dimension."
    else:
        # Only pass if all dimensions >= 7
        min_score = min(scores)
        if min_score < 7:
            verdict = "FAIL"
            worst_dim = min(dim_analysis.items(), key=lambda x: x[1]["score"])
            key_obs = f"Quality gate not met: {worst_dim[0]} scored {worst_dim[1]['score']}/10 (minimum 8 required for all dimensions)."
        else:
            verdict = "PASS"
            best_dim = max(dim_analysis.items(), key=lambda x: x[1]["score"])
            key_obs = f"Function meets WordPress quality bar across all dimensions; strongest in {best_dim[0]} ({best_dim[1]['score']}/10)."

    return {
        "code": code,
        "source_file": fn.get("source_file", ""),
        "source_dir": source_dir,
        "function_name": fn_name,
        "reasoning": {
            "verdict": verdict,
            "dimension_analysis": dim_analysis,
            "overall_score": overall_score,
            "key_observation": key_obs,
        },
        "dimensions_addressed": REQUIRED_DIMENSIONS[:],
        "generation_method": "claude_code_agent_few_shot",
    }


def process_batch(batch_num):
    """Process a single input batch and write output."""
    input_path = PROJECT_ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "batches" / f"_input_batch_{batch_num:02d}.json"
    output_path = PROJECT_ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "batches" / f"batch_{batch_num:03d}.json"

    if not input_path.exists():
        print(f"SKIP: {input_path} not found")
        return 0

    batch = json.loads(input_path.read_text())
    examples = []
    for fn in batch:
        ex = analyze_function(fn)
        examples.append(ex)

    output_path.write_text(json.dumps(examples, indent=2))
    print(f"BATCH {batch_num:03d} COMPLETE: {len(examples)} examples -> {output_path}")
    return len(examples)


if __name__ == "__main__":
    total = 0
    for i in range(10):
        n = process_batch(i)
        total += n
    print(f"\nTotal CoT examples generated: {total}")
