#!/usr/bin/env python3
"""
WordPress Code Quality Judge - Batch Runner
Judges all functions in the target repos and writes passed/failed JSON files.
"""

import json
import os
import re

BASE = "/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/output"
EXTRACTED = os.path.join(BASE, "extracted")
PASSED_DIR = os.path.join(BASE, "passed")
FAILED_DIR = os.path.join(BASE, "failed")

REPOS = [
    "custom-post-type-rewrite",
    "custom-post-type-ui",
    "customizer-search",
    "default-quantity-for-woocommerce",
    "description-list-block",
    "directorist-wpml-integration",
    "discord-display",
    "disqus-comment-system",
    "doqrcode",
    "dpo-group-for-woocommerce",
]

# ---------------------------------------------------------------------------
# Helper patterns for body analysis
# ---------------------------------------------------------------------------

def has_pattern(body, *patterns):
    for p in patterns:
        if re.search(p, body):
            return True
    return False


def body_outputs_html(body):
    return has_pattern(body, r'echo\s', r'\?>', r'printf\s*\(', r'print\s*\(')


def body_has_unescaped_output(body):
    """Detect echo of raw variables without escaping."""
    # echo $var or echo $obj->prop without esc_ wrapper
    return bool(re.search(r'echo\s+\$(?!wpdb)', body) and not re.search(r'esc_(?:html|attr|url|js)\s*\(', body))


def body_has_sql(body):
    return has_pattern(body, r'\$wpdb->', r'SELECT\s+', r'INSERT\s+INTO', r'UPDATE\s+', r'DELETE\s+FROM')


def body_has_raw_sql(body):
    """Detect SQL concatenation without prepare()."""
    # Has SQL but no prepare()
    if body_has_sql(body) and not re.search(r'\$wpdb->prepare\s*\(', body):
        # Check for variable interpolation in query
        if re.search(r'\$wpdb->query\s*\(\s*"[^"]*\$', body) or \
           re.search(r'\$wpdb->query\s*\(\s*\'[^\']*\$', body) or \
           re.search(r'\"[^\"]*SELECT[^\"]*\$[^\"]*\"', body):
            return True
    return False


def body_has_nonce_check(body):
    return has_pattern(body, r'wp_verify_nonce', r'check_admin_referer', r'check_ajax_referer')


def body_has_capability_check(body):
    return has_pattern(body, r'current_user_can', r'is_admin\s*\(', r'manage_options')


def body_has_i18n(body):
    return has_pattern(body, r'__\s*\(', r'_e\s*\(', r'esc_html__\s*\(', r'esc_html_e\s*\(', r'esc_attr__\s*\(', r'_n\s*\(', r'_x\s*\(')


def body_has_hardcoded_strings(body):
    """Detect echo of plain English strings without i18n."""
    return bool(re.search(r'echo\s+[\'"][A-Z][a-zA-Z\s]+[\'"]', body))


def body_has_wp_query(body):
    return has_pattern(body, r'new\s+WP_Query', r'get_posts\s*\(', r'query_posts\s*\(')


def body_has_debug(body):
    return has_pattern(body, r'\bvar_dump\b', r'\bprint_r\b', r'\bdie\s*\(', r'\bdd\s*\(')


def body_has_hooks(body):
    return has_pattern(body, r'add_action\s*\(', r'add_filter\s*\(', r'apply_filters\s*\(', r'do_action\s*\(')


def body_has_rest_api(body):
    return has_pattern(body, r'register_rest_route\s*\(', r'WP_REST_Request', r'WP_REST_Response')


def body_has_transient(body):
    return has_pattern(body, r'get_transient\s*\(', r'set_transient\s*\(', r'wp_cache_get\s*\(')


def body_has_error_reporting(body):
    return has_pattern(body, r'error_reporting\s*\(', r'error_log\s*\(')


def detect_training_tags(fn):
    body = fn.get("body", "")
    tags = []
    if body_has_hooks(body):
        tags.append("hooks")
    if body_has_sql(body):
        tags.append("sql")
    if has_pattern(body, r'\$wpdb->prepare'):
        tags.append("wpdb-prepare")
    if body_has_nonce_check(body):
        tags.append("nonce-verification")
    if body_has_capability_check(body):
        tags.append("capability-check")
    if body_has_i18n(body):
        tags.append("i18n")
    if body_has_rest_api(body):
        tags.append("rest-api")
    if body_has_wp_query(body):
        tags.append("wp-query")
    if body_outputs_html(body):
        tags.append("html-output")
    if has_pattern(body, r'register_post_type\s*\('):
        tags.append("cpt-registration")
    if has_pattern(body, r'register_taxonomy\s*\('):
        tags.append("taxonomy-registration")
    if has_pattern(body, r'wp_enqueue_script|wp_enqueue_style|wp_register'):
        tags.append("asset-enqueue")
    if has_pattern(body, r'singleton|self::\$instance|static::\$instance'):
        tags.append("singleton-pattern")
    if has_pattern(body, r'WC_Payment_Gateway|woocommerce|WC_Order'):
        tags.append("woocommerce-integration")
    if has_pattern(body, r'wp_localize_script'):
        tags.append("script-localization")
    if body_has_transient(body):
        tags.append("caching")
    if has_pattern(body, r'update_option|get_option'):
        tags.append("options-api")
    if has_pattern(body, r'update_post_meta|get_post_meta|add_post_meta'):
        tags.append("post-meta")
    if has_pattern(body, r'WP_CLI'):
        tags.append("wp-cli")
    if has_pattern(body, r'register_block_type|block_init'):
        tags.append("gutenberg-blocks")
    if has_pattern(body, r'load_plugin_textdomain|load_textdomain'):
        tags.append("textdomain-loading")
    if has_pattern(body, r'esc_html|esc_attr|esc_url|wp_kses'):
        tags.append("output-escaping")
    if has_pattern(body, r'sanitize_text_field|sanitize_key|absint|intval|wp_unslash'):
        tags.append("input-sanitization")
    if has_pattern(body, r'wp_remote_get|wp_remote_post'):
        tags.append("http-api")
    if has_pattern(body, r'SimpleXML|DOMDocument|xml'):
        tags.append("xml-handling")
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result if result else ["utility"]


# ---------------------------------------------------------------------------
# Core judging logic - per-repo heuristics
# ---------------------------------------------------------------------------

def make_assessment(fn, repo):
    name = fn.get("function_name", "")
    source_file = fn.get("source_file", "")
    body = fn.get("body", "")
    lc = fn.get("line_count", 0)
    qt = fn.get("quality_tier", "assessed")
    hooks = fn.get("hooks_used", [])
    sql_patterns = fn.get("sql_patterns", [])
    deps = fn.get("dependencies", [])

    # Auto-pass WordPress core
    if qt == "core":
        return {
            "verdict": "PASS",
            "scores": {k: 10 for k in [
                "wpcs_compliance","sql_safety","security","performance",
                "wp_api_usage","code_quality","dependency_integrity","i18n","accessibility"
            ]},
            "critical_failures": [],
            "dependency_chain": deps,
            "training_tags": ["wordpress-core"],
            "notes": "WordPress core - auto-passed",
        }

    # ---------------------------------------------------------------------------
    # Score each dimension
    # ---------------------------------------------------------------------------
    scores = {
        "wpcs_compliance": 8,
        "sql_safety": 10,
        "security": 8,
        "performance": 8,
        "wp_api_usage": 8,
        "code_quality": 8,
        "dependency_integrity": 8,
        "i18n": 7,  # N/A default
        "accessibility": 7,  # N/A default
    }
    critical_failures = []
    training_tags = detect_training_tags(fn)

    # --- WPCS Compliance ---
    docblock = fn.get("docblock", "")
    if not docblock or len(docblock.strip()) < 10:
        scores["wpcs_compliance"] -= 2  # Missing PHPDoc
    if re.search(r'camelCase\s+function\s+\w[A-Z]', body):
        scores["wpcs_compliance"] -= 1
    # Check for PHP 8 typed returns (modern, ok)
    # Check missing spaces around operators etc - hard to detect, assume ok unless obvious
    if body_has_debug(body):
        scores["wpcs_compliance"] -= 2
        scores["code_quality"] -= 3
    if has_pattern(body, r'error_reporting\s*\('):
        scores["wpcs_compliance"] -= 2
        scores["code_quality"] -= 2
        critical_failures.append("error_reporting() call in production code - suppresses PHP errors")

    # --- SQL Safety ---
    if "direct_query" in sql_patterns and "prepared_query" not in sql_patterns:
        # Check if it's truly unsafe (interpolation) or just a DROP/schema query
        if body_has_raw_sql(body) or re.search(r'\$wpdb->query\s*\(\s*\$sql\b', body):
            scores["sql_safety"] = 2
            critical_failures.append("Unprepared query with dynamic SQL variable")
        elif re.search(r'drop table|CREATE TABLE|ALTER TABLE', body, re.IGNORECASE):
            # Schema queries are often ok without prepare
            scores["sql_safety"] = 7
    elif "direct_query" in sql_patterns:
        scores["sql_safety"] = 8  # Has prepare too, inspect more

    # Special case: gdpo_delete_dbo_custom_order_table - has $dpo_table_name interpolated in SQL
    if name == "gdpo_delete_dbo_custom_order_table":
        # String interpolation with $wpdb->prefix is technically safe (controlled value)
        # but the pattern is still flagged
        scores["sql_safety"] = 7  # Minor - table name from wpdb->prefix is safe
        scores["code_quality"] = 7  # Could use $wpdb->prefix safely but direct query

    # --- Security ---
    if body_outputs_html(body):
        # Check for unescaped output
        if re.search(r'echo\s+\$(?!wpdb|post\b)', body) and not re.search(r'esc_(?:html|attr|url)\s*\(.*?\)', body, re.DOTALL):
            scores["security"] -= 2
        # Check for hardcoded strings echoed without esc
        if re.search(r'echo\s+[\'"]<(?!--)', body) and not re.search(r'esc_html', body):
            # Raw HTML output is sometimes ok if no user data
            pass

    # Form handlers that save data should have nonce checks
    is_save_handler = has_pattern(body, r'update_option|update_post_meta|update_user_meta|wp_insert|wp_update') and \
                      has_pattern(body, r'\$_POST|\$_GET|\$_REQUEST')
    if is_save_handler and not body_has_nonce_check(body):
        scores["security"] -= 3
        critical_failures.append("State-changing handler processes POST/GET data without nonce verification")

    # AJAX handlers (wp_send_json) should verify nonce
    if has_pattern(body, r'wp_send_json') and has_pattern(body, r'\$_POST|\$_REQUEST') and not body_has_nonce_check(body):
        scores["security"] -= 3
        critical_failures.append("AJAX handler processes user input without nonce verification")

    # Check unescaped user output (specific patterns)
    if re.search(r'echo\s+\$_(?:GET|POST|REQUEST)', body):
        scores["security"] = min(scores["security"], 2)
        critical_failures.append("Direct echo of superglobal without escaping")

    # --- Performance ---
    # Check for SELECT * patterns
    if re.search(r'SELECT\s+\*', body, re.IGNORECASE):
        scores["performance"] -= 2
    # N+1 queries (query inside loop)
    if re.search(r'foreach.*\n.*\$wpdb->|while.*\n.*\$wpdb->', body):
        scores["performance"] -= 3
        critical_failures.append("Database query inside loop (N+1 pattern)")

    # --- WordPress API Usage ---
    # Using raw SQL for post queries when WP_Query would be better
    if re.search(r'SELECT.*FROM.*\bposts\b', body, re.IGNORECASE) and not has_pattern(body, r'WP_Query|get_posts'):
        scores["wp_api_usage"] -= 2

    # REST endpoints without permission_callback
    if body_has_rest_api(body) and re.search(r'register_rest_route', body):
        if not re.search(r'permission_callback', body):
            scores["wp_api_usage"] = min(scores["wp_api_usage"], 4)
            critical_failures.append("REST endpoint registered without permission_callback")

    # --- Code Quality ---
    # Large functions doing too many things (heuristic: >150 lines)
    if lc > 200:
        scores["code_quality"] -= 1  # Could be doing too much
    if lc > 400:
        scores["code_quality"] -= 1  # Likely doing too much

    # Check for commented-out code blocks
    if re.search(r'//.*echo|//.*var_dump|/\*.*\*/', body):
        pass  # Minimal deduction - comments are informative

    # strip_tags is not a full sanitizer
    if re.search(r'strip_tags\s*\(', body) and has_pattern(body, r'update_option|update_post_meta'):
        scores["security"] -= 1  # strip_tags is weaker than sanitize_text_field

    # --- i18n ---
    if body_outputs_html(body) or re.search(r'echo\s+[\'"][A-Z]', body):
        if body_has_i18n(body):
            scores["i18n"] = 9
        elif re.search(r'echo\s+[\'"][A-Z][a-zA-Z\s]+[\'"]', body):
            scores["i18n"] = 4
            critical_failures.append("Hardcoded English strings in output without i18n wrapper")
        else:
            scores["i18n"] = 7  # N/A - no user-facing strings

    # --- Accessibility ---
    if body_outputs_html(body):
        has_form_inputs = has_pattern(body, r'<input|<textarea|<select')
        has_labels = has_pattern(body, r'<label', r'for="')
        has_aria = has_pattern(body, r'aria-', r'role="')
        if has_form_inputs:
            if has_labels or has_aria:
                scores["accessibility"] = 9
            else:
                scores["accessibility"] = 6
                critical_failures.append("Form inputs present without associated labels")
        else:
            scores["accessibility"] = 7  # N/A

    # --- Dependency Integrity ---
    # Check for bare require_once of non-wp paths (could break)
    if re.search(r'require_once\s+(?!ABSPATH|plugin_dir)', body):
        scores["dependency_integrity"] -= 1

    # --- Repo-specific overrides ---
    _apply_repo_overrides(fn, repo, scores, critical_failures)

    # Clamp all scores 1-10
    for k in scores:
        scores[k] = max(1, min(10, scores[k]))

    # --- Verdict ---
    has_critical_auto_fail = any(
        cf for cf in critical_failures
        if any(kw in cf.lower() for kw in [
            "unprepared query", "direct echo of superglobal", "rest endpoint registered without permission",
            "nonce verification", "form inputs present without associated labels",
            "hardcoded english strings",
        ])
    )
    security_auto_fail = scores["security"] < 5
    # N/A dimensions (i18n=7 meaning no strings, accessibility=7 meaning no HTML output)
    # are treated as passing per the rubric and the sample output format.
    NA_SCORE = 7
    def is_passing_score(dim, val):
        if dim in ("i18n", "accessibility") and val == NA_SCORE:
            return True  # N/A exemption
        return val >= 8
    all_pass = all(is_passing_score(k, v) for k, v in scores.items())

    # A function PASSES if ALL dimensions >= 8 and no critical security failures
    if all_pass and not security_auto_fail and not has_critical_auto_fail:
        verdict = "PASS"
        notes = "All dimensions meet the >= 8 threshold (with N/A exemptions at 7). " + (
            f"Critical issues: {'; '.join(critical_failures)}" if critical_failures else "No critical failures."
        )
    else:
        verdict = "FAIL"
        failing_dims = [f"{k}={v}" for k, v in scores.items() if not is_passing_score(k, v)]
        notes = f"Failing dimensions: {', '.join(failing_dims)}. " + (
            f"Critical issues: {'; '.join(critical_failures)}" if critical_failures else ""
        )

    custom_id = f"{repo}_{name.replace('::', '--').replace(' ', '_')}"

    return {
        "verdict": verdict,
        "scores": scores,
        "critical_failures": critical_failures,
        "dependency_chain": deps,
        "training_tags": training_tags,
        "notes": notes.strip(),
        "_custom_id": custom_id,
    }


def _apply_repo_overrides(fn, repo, scores, critical_failures):
    """Apply repo and function-specific score adjustments."""
    name = fn.get("function_name", "")
    body = fn.get("body", "")
    lc = fn.get("line_count", 0)

    # -----------------------------------------------------------------------
    # custom-post-type-rewrite
    # -----------------------------------------------------------------------
    if repo == "custom-post-type-rewrite":
        if name == "Custom_Post_Type_Rewrite::set_rewrite":
            # Full function - checks $wp_rewrite, uses API correctly
            scores["wpcs_compliance"] = 9
            scores["wp_api_usage"] = 9
            scores["code_quality"] = 8
            scores["i18n"] = 7  # No user-facing strings

    # -----------------------------------------------------------------------
    # discord-display
    # -----------------------------------------------------------------------
    if repo == "discord-display":
        # Missing PHPDoc across the board
        scores["wpcs_compliance"] = max(scores["wpcs_compliance"] - 1, 5)

        if name == "Discord_Display::load_textdomain":
            # Uses deprecated apply_filters('plugin_locale') - should use get_locale()
            scores["wp_api_usage"] = 7
            if "deprecated pattern" not in " ".join(critical_failures):
                pass  # Not critical but not best practice

        if name == "discord_display_widget::widget":
            # Missing escaping in widget output
            if not re.search(r'esc_html|esc_attr|wp_kses', body):
                scores["security"] = 6
                if "Unescaped output in widget" not in " ".join(critical_failures):
                    critical_failures.append("Widget outputs HTML with potential unescaped dynamic data")

        if name == "discord_display_widget::update":
            # Uses strip_tags instead of sanitize_text_field
            scores["security"] = 7
            scores["code_quality"] = 7

        if name == "discord_display_widget::form":
            # Uses _e() without esc_ prefix (not best practice but acceptable in forms)
            scores["i18n"] = 8
            scores["accessibility"] = 8

        if name == "Discord_API::__construct":
            # Returns false from constructor - PHP anti-pattern
            scores["code_quality"] = 6
            if "Constructor returns false" not in " ".join(critical_failures):
                critical_failures.append("Constructor returns false - PHP anti-pattern, constructors cannot return values")

        if name == "discord_display_scripts":
            # Loads Font Awesome from CDN with specific version (not best practice)
            scores["wp_api_usage"] = 7
            scores["performance"] = 7

    # -----------------------------------------------------------------------
    # customizer-search (trusted tier - high quality Astra notices library)
    # -----------------------------------------------------------------------
    if repo == "customizer-search":
        qt = fn.get("quality_tier", "")
        if qt == "trusted":
            # Generally well-written code, boost scores
            scores["wpcs_compliance"] = max(scores["wpcs_compliance"], 9)
            scores["code_quality"] = max(scores["code_quality"], 9)

        if name == "Astra_Notices::dismiss_notice":
            # Checks nonce, checks capability - excellent security
            scores["security"] = 10
            scores["wpcs_compliance"] = 9

        if name == "Astra_Notices::show_notices":
            # Large function but well-structured
            scores["code_quality"] = 8

        if name == "Astra_Notices::enqueue_scripts":
            scores["wp_api_usage"] = 9
            scores["security"] = 9  # Creates nonce

        if name == "is_expired":
            # Uses get_transient correctly
            scores["performance"] = 9
            scores["wp_api_usage"] = 9

    # -----------------------------------------------------------------------
    # default-quantity-for-woocommerce
    # -----------------------------------------------------------------------
    if repo == "default-quantity-for-woocommerce":
        if name == "DefaultQuantityForWoocommerce::__construct":
            # Has error_reporting(E_ALL ^ E_DEPRECATED) - production code
            scores["wpcs_compliance"] = 5
            scores["code_quality"] = 5
            # Already added critical failure from main logic

        if name == "Settings::dqfwc_save_taxonomy_custom_meta":
            # No nonce verification before saving
            scores["security"] = 5
            if "State-changing" not in " ".join(critical_failures):
                critical_failures.append("Taxonomy meta save handler lacks nonce verification")

        if name == "Settings::dqfwc_taxonomy_add_new_meta_field":
            scores["accessibility"] = 9  # Has label with for attribute
            scores["i18n"] = 9
            scores["security"] = 9

        if name == "Settings::dqfwc_taxonomy_edit_meta_field":
            scores["accessibility"] = 9
            scores["i18n"] = 9
            scores["security"] = 9

        if name == "Settings::dqfwc_product_default_quantity_meta":
            scores["wp_api_usage"] = 9  # Uses woocommerce_wp_text_input
            scores["i18n"] = 9
            scores["security"] = 9

        if name == "PluginMeta::plugin_meta_links":
            scores["i18n"] = 9
            scores["security"] = 9

    # -----------------------------------------------------------------------
    # description-list-block (too short)
    # -----------------------------------------------------------------------
    if repo == "description-list-block":
        pass  # Will be skipped by line_count < 5

    # -----------------------------------------------------------------------
    # directorist-wpml-integration
    # -----------------------------------------------------------------------
    if repo == "directorist-wpml-integration":
        # Generally modern OOP code
        scores["wpcs_compliance"] = max(scores["wpcs_compliance"], 8)

        if name == "Get_Directory_Type_Translations::create_directory_type_translation":
            # Uses directorist_verify_nonce() - custom nonce check (ok if the function wraps check_ajax_referer)
            scores["security"] = 9
            scores["wpcs_compliance"] = 8

        if name == "WPML_Helper::set_post_translation":
            scores["code_quality"] = 9
            scores["i18n"] = 8

        if name == "Block_Widget_Translation::translate_block_attributes":
            scores["code_quality"] = 9
            scores["wp_api_usage"] = 9

        if name == "Category_Directory_Sync::sync_category_directory":
            scores["code_quality"] = 9
            scores["wp_api_usage"] = 9

        if name == "Query_Filtering::filter_query_results":
            scores["code_quality"] = 9
            scores["performance"] = 8

        if name == "Search_Form_Filter::translate_tax_query_terms":
            scores["code_quality"] = 9
            scores["wp_api_usage"] = 9

    # -----------------------------------------------------------------------
    # disqus-comment-system
    # -----------------------------------------------------------------------
    if repo == "disqus-comment-system":
        # Generally well-written plugin
        scores["wpcs_compliance"] = max(scores["wpcs_compliance"], 8)

        if name == "Disqus_Rest_Api::register_endpoints":
            # All routes have permission_callback
            scores["wp_api_usage"] = 10
            scores["security"] = 10

        if name == "Disqus_Rest_Api::rest_admin_only_permission_callback":
            # Properly checks both cookie auth and HMAC
            scores["security"] = 10
            scores["code_quality"] = 9

        if name == "Disqus_Admin::dsq_dismiss_ads_notice":
            # Verifies nonce + capability check
            scores["security"] = 10

        if name == "Disqus_Admin::enqueue_scripts":
            # Creates nonce for REST API
            scores["security"] = 9
            scores["wp_api_usage"] = 9

        if name == "Disqus_Admin::dsq_construct_admin_bar":
            # Checks capability before modifying admin bar
            # But has raw '<span class="ab-icon"></span>Disqus' without i18n for "Disqus"
            scores["security"] = 9
            scores["i18n"] = 7  # Brand name doesn't need i18n

        if name == "Disqus_Rest_Api::generate_export_wxr":
            # Large (206L) but well-structured XML generation
            scores["code_quality"] = 8
            scores["wp_api_usage"] = 8

        if name == "Disqus_Rest_Api::comment_data_from_post":
            scores["wp_api_usage"] = 9
            scores["code_quality"] = 8

    # -----------------------------------------------------------------------
    # doqrcode - pure PHP library wrapped for WP (non-WP code)
    # -----------------------------------------------------------------------
    if repo == "doqrcode":
        # The QR code library classes (QRspec, QRmask, etc.) are not WordPress-idiomatic
        # They're a PHP library that happens to be in a WP plugin
        is_qr_lib = any(name.startswith(cls) for cls in [
            "QRtools", "QRspec", "QRimage", "QRinputItem", "QRinput",
            "QRbitstream", "QRsplit", "QRrsItem", "QRrs", "QRmask",
            "QRrsblock", "QRrawcode", "QRcode", "FrameFiller", "QRencode",
            "QRvect", "qrstr"
        ])
        if is_qr_lib:
            # These are library code, not WordPress idioms
            # They lack PHPDoc, don't use WP APIs, have non-WP naming
            scores["wpcs_compliance"] = 5
            scores["wp_api_usage"] = 5  # Library code, not using WP APIs (expected)
            scores["i18n"] = 7  # No user strings expected in library
            scores["accessibility"] = 7  # Library generates images/SVG
            # Library code but security-safe (no SQL, no user input without sanitization)

        if name == "DoQRCode::shortcode_handler":
            # This IS WP code
            scores["wpcs_compliance"] = 8
            scores["wp_api_usage"] = 9

        if name == "QRtools::log":
            # Uses file_put_contents (should use WP_Filesystem)
            scores["security"] = 6
            scores["wp_api_usage"] = 5
            if "WP_Filesystem" not in " ".join(critical_failures):
                critical_failures.append("Uses file_put_contents directly instead of WP_Filesystem")

        if name == "QRtools::save":
            scores["security"] = 6
            scores["wp_api_usage"] = 5
            if any("file_put_contents" in cf for cf in critical_failures):
                pass
            else:
                critical_failures.append("Uses file_put_contents/imagepng directly instead of WP_Filesystem")

        if name == "QRtools::buildCache":
            scores["security"] = 6
            scores["wp_api_usage"] = 5

    # -----------------------------------------------------------------------
    # dpo-group-for-woocommerce
    # -----------------------------------------------------------------------
    if repo == "dpo-group-for-woocommerce":
        if name == "gdpo_woocommerce_dpo_init":
            # Nested function inside another function - poor practice
            scores["code_quality"] = 6
            if "Nested function" not in " ".join(critical_failures):
                critical_failures.append("Function defines a nested named function (gdpo_woocommerce_add_gateway_dpo) which pollutes global scope on each call")

        if name == "WCGatewayDPO::check_dpo_notify":
            # Webhook handler - no nonce (that's fine, it's a server-to-server call)
            # But has proper HMAC verification via the API
            scores["security"] = 8
            scores["code_quality"] = 7  # Complex, hard to follow
            scores["i18n"] = 7  # echo 'OK' is server-to-server API response, not user-facing
            # Remove false positive nonce/hardcoded-string failures
            critical_failures[:] = [cf for cf in critical_failures
                                     if "nonce" not in cf.lower() and "hardcoded" not in cf.lower()]

        if name == "WCGatewayDPO::updateResponseOrderStatus":
            # Uses __(constant_string + variable) - strings with concat in __() don't work in i18n
            scores["i18n"] = 5
            if "i18n string concatenation" not in " ".join(critical_failures):
                critical_failures.append("Translatable string contains concatenation inside __() which breaks i18n tools")

        if name == "WCGatewayDPO::updatePostMeta":
            # Uses __(string + variable) pattern
            scores["i18n"] = 5
            if "i18n string concatenation" not in " ".join(critical_failures):
                critical_failures.append("Translatable string contains concatenation inside __() which breaks i18n tools")

        if name == "WCGatewayDPO::admin_options":
            # Hard-coded non-translated table header "Settings" and "DPO Pay"
            scores["i18n"] = 5
            if "Hardcoded" not in " ".join(critical_failures):
                critical_failures.append("Hard-coded English strings in admin table headers without i18n wrappers")

        if name == "WCGatewayDPO::check_dpo_response":
            # Accesses $_GET['TransactionToken'] without isset check first
            scores["security"] = 7
            scores["code_quality"] = 7

        if name == "logData":
            # Uses $logger->add() (deprecated WC logger API) instead of $logger->info()
            scores["wp_api_usage"] = 7

        if name == "gdpo_delete_dbo_custom_order_table":
            # Uses string interpolation in SQL for table name
            # $wpdb->prefix is trusted but pattern is still not ideal
            scores["sql_safety"] = 7
            # Already handled above

        if name == "DpoPaySettings::get_form_fields":
            scores["i18n"] = 9
            scores["wp_api_usage"] = 9
            scores["code_quality"] = 9

        if name == "WCGatewayDPO::before_payment":
            # 105 line function doing too many things
            scores["code_quality"] = 7

        if name == "WCGatewayDPO::updateOrderStatus":
            # i18n strings with self::constants - ok pattern
            scores["i18n"] = 7  # Constants in __() are ok since they're defined elsewhere
            scores["code_quality"] = 8

        if name == "gdpo_custom_tab_options":
            # Uses $woothemes as text domain variable - unusual
            scores["i18n"] = 7
            scores["wpcs_compliance"] = 7

        if name == "gdpo_process_product_meta_custom_tab":
            # Has nonce verification, sanitizes input - good
            scores["security"] = 9
            scores["code_quality"] = 8
            # Remove false positive
            critical_failures[:] = [cf for cf in critical_failures if "nonce" not in cf.lower()]


def judge_function(fn, repo):
    """Judge a single function and return the function dict with assessment added."""
    lc = fn.get("line_count", 0)
    qt = fn.get("quality_tier", "assessed")
    name = fn.get("function_name", "")

    # Skip functions that are too short
    if lc < 5:
        return None

    # Core auto-pass
    if qt == "core":
        assessment = make_assessment(fn, repo)
        assessment["function_name"] = name
        assessment["file_path"] = fn.get("source_file", "")
        result = dict(fn)
        result["assessment"] = assessment
        result["training_tags"] = assessment["training_tags"]
        return result

    assessment = make_assessment(fn, repo)
    assessment["function_name"] = name
    assessment["file_path"] = fn.get("source_file", "")

    result = dict(fn)
    result["assessment"] = assessment
    result["training_tags"] = assessment["training_tags"]
    return result


def process_repo(repo):
    extracted_path = os.path.join(EXTRACTED, f"{repo}.json")
    if not os.path.exists(extracted_path):
        print(f"  SKIP: {repo} (no extracted file)")
        return 0, 0

    with open(extracted_path) as f:
        functions = json.load(f)

    passed = []
    failed = []
    skipped = 0

    for fn in functions:
        result = judge_function(fn, repo)
        if result is None:
            skipped += 1
            continue
        verdict = result["assessment"]["verdict"]
        if verdict == "PASS":
            passed.append(result)
        else:
            failed.append(result)

    # Write results
    passed_path = os.path.join(PASSED_DIR, f"{repo}.json")
    failed_path = os.path.join(FAILED_DIR, f"{repo}.json")

    with open(passed_path, "w") as f:
        json.dump(passed, f, indent=2)

    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)

    print(f"  {repo}: {len(passed)} PASS / {len(failed)} FAIL / {skipped} skipped (< 5 lines)")
    return len(passed), len(failed)


def main():
    total_pass = 0
    total_fail = 0
    for repo in REPOS:
        p, f = process_repo(repo)
        total_pass += p
        total_fail += f
    print(f"\nTotal: {total_pass} PASS / {total_fail} FAIL")


if __name__ == "__main__":
    main()
