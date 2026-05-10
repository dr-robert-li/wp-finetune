"""
Static rubric data for WordPress PHP Code Quality Scoring.

Source: docs/eval/wp_code_quality_rubric.md v1.0 (March 2026)
Supporting research:
  - docs/eval/research_wpcs_standards.md (PHPCS sniff names)
  - docs/eval/research_wp_security_sql_perf.md (security/SQL/perf patterns)

This module is pure data -- importing it has no side effects.
All 193 check IDs (83 positive + 110 negative) are registered in CHECK_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckDef:
    """A single rubric check definition.

    Attributes:
        id: Unique check identifier (e.g. 'WPCS-P01', 'SEC-N03').
        dimension: Internal dimension key (e.g. 'D1_wpcs').
        polarity: 'positive' or 'negative'.
        weight: Integer weight (positive checks) or penalty magnitude
                (negative checks, stored as positive int -- caller negates).
        method: Primary detection method.  One of:
                phpcs, phpstan, regex, regex+, llm, file, dom.
        tool_detail: Sniff name, regex pattern string, or description of
                     what the LLM/file/dom check looks for.
    """

    id: str
    dimension: str
    polarity: str  # "positive" | "negative"
    weight: int
    method: str  # phpcs | phpstan | regex | regex+ | llm | file | dom
    tool_detail: str


# ---------------------------------------------------------------------------
# 1. DIMENSION_WEIGHTS  (must sum to 1.0)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "D1_wpcs": 0.10,
    "D2_security": 0.20,
    "D3_sql": 0.15,
    "D4_perf": 0.10,
    "D5_wp_api": 0.10,
    "D6_i18n": 0.10,
    "D7_a11y": 0.08,
    "D8_errors": 0.10,
    "D9_structure": 0.07,
}
"""Dimension weights from rubric Section D. Sum = 1.0."""

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# 2. DIMENSION_MAX_POSITIVE  (Section C formulas for each dimension)
# ---------------------------------------------------------------------------

DIMENSION_MAX_POSITIVE: Dict[str, int] = {
    "D1_wpcs": 15,
    "D2_security": 25,
    "D3_sql": 14,
    "D4_perf": 16,
    "D5_wp_api": 17,
    "D6_i18n": 15,
    "D7_a11y": 18,
    "D8_errors": 19,
    "D9_structure": 23,
}
"""Max positive point total per dimension (from rubric Section C formulas)."""


# ---------------------------------------------------------------------------
# 3. CHECK_REGISTRY  -- all 193 checks
# ---------------------------------------------------------------------------

CHECK_REGISTRY: Dict[str, CheckDef] = {}
"""Master registry of all 193 rubric checks, keyed by check ID."""


def _reg(
    id: str,
    dimension: str,
    polarity: str,
    weight: int,
    method: str,
    tool_detail: str,
) -> None:
    """Helper to register a check."""
    CHECK_REGISTRY[id] = CheckDef(
        id=id,
        dimension=dimension,
        polarity=polarity,
        weight=weight,
        method=method,
        tool_detail=tool_detail,
    )


# ---- Dimension 1: WPCS Compliance (12 positive, 20 negative = 32) --------

# Positive
_reg("WPCS-P01", "D1_wpcs", "positive", 2, "regex+",
     r"^\s{4,}")  # lines starting with 4+ spaces (not in strings/heredoc)
_reg("WPCS-P02", "D1_wpcs", "positive", 2, "phpcs",
     "WordPress.NamingConventions.ValidFunctionName, WordPress.NamingConventions.ValidVariableName")
_reg("WPCS-P03", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.NamingConventions.ValidFunctionName (class context)")
_reg("WPCS-P04", "D1_wpcs", "positive", 1, "regex",
     r"define\s*\(\s*'[A-Z][A-Z0-9_]+'")
_reg("WPCS-P05", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.PHP.YodaConditions")
_reg("WPCS-P06", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.WhiteSpace.ControlStructureSpacing")
_reg("WPCS-P07", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.WhiteSpace.OperatorSpacing")
_reg("WPCS-P08", "D1_wpcs", "positive", 2, "phpcs",
     "WordPress.NamingConventions.PrefixAllGlobals")
_reg("WPCS-P09", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.Files.FileName")
_reg("WPCS-P10", "D1_wpcs", "positive", 1, "phpcs",
     "WordPress.PHP.StrictInArray")
_reg("WPCS-P11", "D1_wpcs", "positive", 1, "phpcs",
     "Generic.ControlStructures.InlineControlStructure")
_reg("WPCS-P12", "D1_wpcs", "positive", 1, "regex",
     r"else\s+if\b")  # negative: absent means elseif used correctly

# Negative
_reg("WPCS-N01", "D1_wpcs", "negative", 3, "phpcs",
     "Generic.PHP.DisallowShortOpenTag")
_reg("WPCS-N02", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.NamingConventions.ValidFunctionName")
_reg("WPCS-N03", "D1_wpcs", "negative", 3, "phpcs",
     "WordPress.NamingConventions.PrefixAllGlobals")
_reg("WPCS-N04", "D1_wpcs", "negative", 2, "phpcs",
     "Squiz.Scope.MemberVarScope")
_reg("WPCS-N05", "D1_wpcs", "negative", 2, "phpcs",
     "Squiz.Scope.MethodScope")
_reg("WPCS-N06", "D1_wpcs", "negative", 1, "regex",
     r"\?>\s*$")
_reg("WPCS-N07", "D1_wpcs", "negative", 1, "regex",
     r"\?:")  # Elvis operator in expression context
_reg("WPCS-N08", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.PHP.StrictInArray")
_reg("WPCS-N09", "D1_wpcs", "negative", 1, "phpcs",
     "WordPress.PHP.StrictComparisons")
_reg("WPCS-N10", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.PHP.DontExtract")
_reg("WPCS-N11", "D1_wpcs", "negative", 3, "phpcs",
     "WordPress.PHP.RestrictedPHPFunctions")
_reg("WPCS-N12", "D1_wpcs", "negative", 5, "phpcs",
     "WordPress.PHP.RestrictedPHPFunctions")  # eval()
_reg("WPCS-N13", "D1_wpcs", "negative", 3, "regex",
     r"\bgoto\b")
_reg("WPCS-N14", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.PHP.NoSilencedErrors")
_reg("WPCS-N15", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.PHP.DevelopmentFunctions")
_reg("WPCS-N16", "D1_wpcs", "negative", 1, "phpcs",
     "WordPress.DateTime.CurrentTimeTimestamp")
_reg("WPCS-N17", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.CodeAnalysis.AssignmentInTernaryCondition")
_reg("WPCS-N18", "D1_wpcs", "negative", 1, "phpcs",
     "WordPress.CodeAnalysis.EscapedNotTranslated")
_reg("WPCS-N19", "D1_wpcs", "negative", 1, "phpcs",
     "WordPress.PHP.PregQuoteDelimiter")
_reg("WPCS-N20", "D1_wpcs", "negative", 2, "phpcs",
     "WordPress.WP.GlobalVariablesOverride")


# ---- Dimension 2: Security (13 positive, 20 negative = 33) ---------------

# Positive
_reg("SEC-P01", "D2_security", "positive", 3, "phpcs",
     "WordPress.Security.EscapeOutput (absence of violations)")
_reg("SEC-P02", "D2_security", "positive", 2, "regex+",
     r"wp_unslash\s*\(.*sanitize|sanitize.*wp_unslash")
_reg("SEC-P03", "D2_security", "positive", 3, "phpcs",
     "WordPress.Security.ValidatedSanitizedInput (absence of violations)")
_reg("SEC-P04", "D2_security", "positive", 3, "phpcs",
     "WordPress.Security.NonceVerification (absence of violations)")
_reg("SEC-P05", "D2_security", "positive", 3, "llm",
     "current_user_can() before privileged write operation")
_reg("SEC-P06", "D2_security", "positive", 1, "phpcs",
     "WordPress.Security.SafeRedirect (absence of violations)")
_reg("SEC-P07", "D2_security", "positive", 2, "regex",
     r"wp_check_filetype_and_ext\s*\(")
_reg("SEC-P08", "D2_security", "positive", 1, "regex",
     r"wp_handle_upload\s*\(")
_reg("SEC-P09", "D2_security", "positive", 1, "regex",
     r"is_email\s*\(")
_reg("SEC-P10", "D2_security", "positive", 1, "llm",
     "Safelist pattern for ORDER BY / non-preparable SQL fragments")
_reg("SEC-P11", "D2_security", "positive", 1, "regex",
     r"absint\s*\(")
_reg("SEC-P12", "D2_security", "positive", 1, "regex",
     r"esc_html__\s*\(|esc_attr__\s*\(")
_reg("SEC-P13", "D2_security", "positive", 2, "llm",
     "Output context matched to correct escaping function")

# Sum of positive weights: 3+2+3+3+3+1+2+1+1+1+1+1+2 = 24 ... rubric says 25
# NOTE: The rubric Section C states max_positive=25 for D2.
# The table sums to 24. We faithfully store both the per-check weights from the
# table AND the max_positive=25 from the formula. The scorer should use
# DIMENSION_MAX_POSITIVE for the denominator.

# Negative
_reg("SEC-N01", "D2_security", "negative", 5, "phpcs",
     "WordPress.Security.EscapeOutput")
_reg("SEC-N02", "D2_security", "negative", 3, "phpcs",
     "WordPress.Security.ValidatedSanitizedInput")
_reg("SEC-N03", "D2_security", "negative", 4, "phpcs",
     "WordPress.Security.NonceVerification")
_reg("SEC-N04", "D2_security", "negative", 4, "llm",
     "Privileged action without current_user_can() check")
_reg("SEC-N05", "D2_security", "negative", 3, "phpcs",
     "WordPress.Security.SafeRedirect")
_reg("SEC-N06", "D2_security", "negative", 4, "phpcs",
     "Security.BadFunctions.PHPInternalFunctions")  # unserialize on user input
_reg("SEC-N07", "D2_security", "negative", 2, "phpcs",
     "Security.BadFunctions.PHPInternalFunctions")  # base64_decode on user input
_reg("SEC-N08", "D2_security", "negative", 5, "phpcs",
     "Security.BadFunctions.FilesystemFunctions")  # include/require with user input
_reg("SEC-N09", "D2_security", "negative", 3, "regex",
     r"\$_FILES\[.*\]\[.type.\]")  # browser-set MIME type as sole validation
_reg("SEC-N10", "D2_security", "negative", 3, "llm",
     "HTML <form> present without wp_nonce_field")
_reg("SEC-N11", "D2_security", "negative", 4, "regex",
     r"wp_ajax_nopriv_")  # without check_ajax_referer
_reg("SEC-N12", "D2_security", "negative", 2, "regex",
     r"current_user_can\s*\(\s*['\"](?:administrator|editor|author|contributor|subscriber)['\"]")
_reg("SEC-N13", "D2_security", "negative", 2, "llm",
     "is_admin() used as security gate for data modification")
_reg("SEC-N14", "D2_security", "negative", 1, "phpcs",
     "WordPress.Security.PluginMenuSlug")
_reg("SEC-N15", "D2_security", "negative", 2, "llm",
     "esc_html() used on URL context (wrong escaping function)")
_reg("SEC-N16", "D2_security", "negative", 2, "llm",
     "esc_attr() used on URL (does not validate scheme)")
_reg("SEC-N17", "D2_security", "negative", 1, "llm",
     "Double-escaping: value escaped then escaped again")
_reg("SEC-N18", "D2_security", "negative", 1, "regex",
     r"\$_REQUEST\b")
_reg("SEC-N19", "D2_security", "negative", 5, "phpcs",
     "Security.BadFunctions.SystemExecFunctions")
_reg("SEC-N20", "D2_security", "negative", 5, "regex",
     r"preg_replace\s*\(.*\/e['\"]")


# ---- Dimension 3: SQL Safety (9 positive, 17 negative = 26) --------------

# Positive
_reg("SQL-P01", "D3_sql", "positive", 3, "phpcs",
     "WordPress.DB.PreparedSQL (absence of violations)")
_reg("SQL-P02", "D3_sql", "positive", 2, "regex",
     r"\$wpdb->(insert|update|delete)\s*\(")
_reg("SQL-P03", "D3_sql", "positive", 2, "regex+",
     r"esc_like\s*\(")  # context: before LIKE %s in prepare
_reg("SQL-P04", "D3_sql", "positive", 2, "regex",
     r"new\s+WP_Query\s*\(|get_posts\s*\(")
_reg("SQL-P05", "D3_sql", "positive", 1, "regex",
     r"prepare\s*\(.*%i")
_reg("SQL-P06", "D3_sql", "positive", 1, "regex",
     r'".*wp_[a-z_]+.*"')  # NEGATIVE match: absence of hardcoded wp_ table names
_reg("SQL-P07", "D3_sql", "positive", 1, "llm",
     "ORDER BY direction validated against safelist")
_reg("SQL-P08", "D3_sql", "positive", 1, "llm",
     "LIMIT clause present on all direct $wpdb->get_results() calls")
_reg("SQL-P09", "D3_sql", "positive", 1, "regex",
     r"\$wpdb->last_error")

# Negative
_reg("SQL-N01", "D3_sql", "negative", 5, "phpcs",
     "WordPress.DB.PreparedSQL")
_reg("SQL-N02", "D3_sql", "negative", 4, "regex",
     r'["\'].*\.\s*\$[a-z_].*["\']')  # string concat in $wpdb-> argument
_reg("SQL-N03", "D3_sql", "negative", 5, "phpcs",
     "WordPress.DB.PreparedSQL, WordPress.Security.ValidatedSanitizedInput")
_reg("SQL-N04", "D3_sql", "negative", 3, "regex+",
     r"LIKE\s+%s")  # without preceding esc_like
_reg("SQL-N05", "D3_sql", "negative", 2, "phpcs",
     "WordPress.DB.PreparedSQLPlaceholders")
_reg("SQL-N06", "D3_sql", "negative", 2, "phpcs",
     "WordPress.DB.PreparedSQLPlaceholders")
_reg("SQL-N07", "D3_sql", "negative", 2, "regex",
     r"->escape\s*\(")
_reg("SQL-N08", "D3_sql", "negative", 2, "regex",
     r"prepare\s*\(.*esc_sql")
_reg("SQL-N09", "D3_sql", "negative", 4, "phpcs",
     "WordPress.DB.RestrictedFunctions, WordPress.DB.RestrictedClasses")
_reg("SQL-N10", "D3_sql", "negative", 3, "phpcs",
     "WordPress.WP.DiscouragedFunctions")  # query_posts()
_reg("SQL-N11", "D3_sql", "negative", 2, "regex",
     r'"[^"]*wp_[a-z_]+[^"]*"')  # hardcoded table name in SQL context
_reg("SQL-N12", "D3_sql", "negative", 2, "phpcs",
     "WordPress.WP.PostsPerPage")
_reg("SQL-N13", "D3_sql", "negative", 2, "regex",
     r"'orderby'\s*=>\s*'rand'")
_reg("SQL-N14", "D3_sql", "negative", 1, "regex",
     r"'suppress_filters'\s*=>\s*true")
_reg("SQL-N15", "D3_sql", "negative", 1, "regex",
     r"SELECT\s+\*\s+FROM")  # in $wpdb->get_results
_reg("SQL-N16", "D3_sql", "negative", 2, "llm",
     "No LIMIT on direct SQL get_results() returning unbounded rows")
_reg("SQL-N17", "D3_sql", "negative", 5, "regex",
     r"prepare\s*\(\s*\$_(GET|POST|REQUEST)")


# ---- Dimension 4: Performance (12 positive, 13 negative = 25) ------------

# Positive
_reg("PERF-P01", "D4_perf", "positive", 3, "regex+",
     r"wp_cache_get\s*\(")  # also requires wp_cache_set nearby
_reg("PERF-P02", "D4_perf", "positive", 2, "regex",
     r"set_transient\s*\(\s*[^,]+,\s*[^,]+,\s*[^)]+\)")  # 3 arguments
_reg("PERF-P03", "D4_perf", "positive", 1, "regex",
     r"false\s*===?\s*get_transient|get_transient.*!==?\s*false")
_reg("PERF-P04", "D4_perf", "positive", 2, "llm",
     "Remote HTTP response cached in transients (wp_remote_get + set_transient in same scope)")
_reg("PERF-P05", "D4_perf", "positive", 1, "regex",
     r"'no_found_rows'\s*=>\s*true")
_reg("PERF-P06", "D4_perf", "positive", 1, "regex",
     r"'fields'\s*=>\s*'ids'")
_reg("PERF-P07", "D4_perf", "positive", 1, "regex",
     r"'update_post_meta_cache'\s*=>\s*false|'update_post_term_cache'\s*=>\s*false")
_reg("PERF-P08", "D4_perf", "positive", 1, "regex",
     r"add_option\s*\(.*,.*,.*,\s*false|update_option\s*\(.*,.*,\s*false")
_reg("PERF-P09", "D4_perf", "positive", 1, "phpcs",
     "WordPress.WP.EnqueuedResourceParameters")
_reg("PERF-P10", "D4_perf", "positive", 1, "llm",
     "Conditional script/style enqueueing (loads only where needed)")
_reg("PERF-P11", "D4_perf", "positive", 1, "regex",
     r"wp_remote_get\s*\(.*'timeout'")
_reg("PERF-P12", "D4_perf", "positive", 1, "regex",
     r"wp_remote_retrieve_response_code\s*\(")

# Negative
_reg("PERF-N01", "D4_perf", "negative", 3, "llm",
     "N+1 pattern: get_post_meta/get_the_terms/get_user_by inside foreach/while loop")
_reg("PERF-N02", "D4_perf", "negative", 3, "regex",
     r"\$wpdb->")  # inside foreach|while body (context-sensitive)
_reg("PERF-N03", "D4_perf", "negative", 3, "regex",
     r"set_transient\s*\(\s*['\"\w]+\s*,\s*[^,)]+\s*\)")  # 2-arg form, no expiration
_reg("PERF-N04", "D4_perf", "negative", 3, "regex",
     r"wp_cache_flush\s*\(\s*\)")  # in hook callback
_reg("PERF-N05", "D4_perf", "negative", 2, "llm",
     "Uncached wp_remote_get/post on every page load without transient/cache wrapper")
_reg("PERF-N06", "D4_perf", "negative", 2, "phpcs",
     "WordPress.WP.AlternativeFunctions")  # file_get_contents for HTTP
_reg("PERF-N07", "D4_perf", "negative", 2, "regex",
     r"curl_exec\s*\(")
_reg("PERF-N08", "D4_perf", "negative", 2, "phpcs",
     "WordPress.WP.EnqueuedResources")
_reg("PERF-N09", "D4_perf", "negative", 2, "regex",
     r"add_action\s*\(\s*'init'.*flush_rewrite_rules")
_reg("PERF-N10", "D4_perf", "negative", 1, "llm",
     "get_option() called repeatedly inside loop for same option")
_reg("PERF-N11", "D4_perf", "negative", 2, "llm",
     "Large array stored as autoloaded option without false")
_reg("PERF-N12", "D4_perf", "negative", 1, "regex",
     r"wp_remote_get\s*\(")  # without nearby is_wp_error check
_reg("PERF-N13", "D4_perf", "negative", 2, "regex",
     r"'orderby'\s*=>\s*'rand'")


# ---- Dimension 5: WP API Usage (13 positive, 14 negative = 27) -----------

# Positive
_reg("WAPI-P01", "D5_wp_api", "positive", 2, "phpcs",
     "WordPress.WP.AlternativeFunctions")
_reg("WAPI-P02", "D5_wp_api", "positive", 1, "phpcs",
     "WordPress.WP.AlternativeFunctions")
_reg("WAPI-P03", "D5_wp_api", "positive", 1, "phpcs",
     "WordPress.WP.AlternativeFunctions")
_reg("WAPI-P04", "D5_wp_api", "positive", 1, "phpcs",
     "WordPress.WP.AlternativeFunctions")
_reg("WAPI-P05", "D5_wp_api", "positive", 1, "phpcs",
     "WordPress.DateTime.RestrictedFunctions")
_reg("WAPI-P06", "D5_wp_api", "positive", 2, "phpcs",
     "WordPress.WP.EnqueuedResources, WordPress.WP.EnqueuedResourceParameters")
_reg("WAPI-P07", "D5_wp_api", "positive", 2, "regex",
     r"WP_Filesystem\(\)|\$wp_filesystem->")
_reg("WAPI-P08", "D5_wp_api", "positive", 2, "llm",
     "Initialization code wrapped in add_action/add_filter hooks")
_reg("WAPI-P09", "D5_wp_api", "positive", 1, "regex",
     r"add_action\s*\(\s*'init'.*register_post_type|register_taxonomy")
_reg("WAPI-P10", "D5_wp_api", "positive", 1, "regex",
     r"wp_localize_script\s*\(|wp_add_inline_script\s*\(")
_reg("WAPI-P11", "D5_wp_api", "positive", 1, "regex",
     r"plugins_url\s*\(|plugin_dir_url\s*\(")
_reg("WAPI-P12", "D5_wp_api", "positive", 1, "regex",
     r"query_posts\s*\(")  # NEGATIVE match: absence means using WP_Query correctly
_reg("WAPI-P13", "D5_wp_api", "positive", 1, "phpcs",
     "WordPress.WP.DeprecatedFunctions, WordPress.WP.DeprecatedClasses (absence of violations)")

# Negative
_reg("WAPI-N01", "D5_wp_api", "negative", 2, "phpcs",
     "WordPress.WP.AlternativeFunctions")  # file_get_contents for HTTP
_reg("WAPI-N02", "D5_wp_api", "negative", 2, "regex",
     r"curl_init\s*\(|curl_exec\s*\(|curl_setopt\s*\(")
_reg("WAPI-N03", "D5_wp_api", "negative", 1, "phpcs",
     "WordPress.WP.AlternativeFunctions")  # json_encode
_reg("WAPI-N04", "D5_wp_api", "negative", 1, "phpcs",
     "WordPress.WP.AlternativeFunctions")  # rand/mt_rand
_reg("WAPI-N05", "D5_wp_api", "negative", 1, "phpcs",
     "WordPress.DateTime.RestrictedFunctions")
_reg("WAPI-N06", "D5_wp_api", "negative", 3, "phpcs",
     "WordPress.WP.DiscouragedFunctions")  # query_posts
_reg("WAPI-N07", "D5_wp_api", "negative", 2, "phpcs",
     "WordPress.WP.DeprecatedFunctions")
_reg("WAPI-N08", "D5_wp_api", "negative", 2, "phpcs",
     "WordPress.WP.DeprecatedClasses")
_reg("WAPI-N09", "D5_wp_api", "negative", 2, "regex",
     r"require.*phpmailer|wp_enqueue_script.*jquery.*http")
_reg("WAPI-N10", "D5_wp_api", "negative", 2, "regex",
     r"\bfopen\s*\(|file_put_contents\s*\(|\bfwrite\s*\(")
_reg("WAPI-N11", "D5_wp_api", "negative", 2, "regex",
     r"session_start\s*\(|\$_SESSION")
_reg("WAPI-N12", "D5_wp_api", "negative", 2, "llm",
     "Initialization code runs at file include time, not deferred to hook")
_reg("WAPI-N13", "D5_wp_api", "negative", 1, "llm",
     "add_menu_page/add_submenu_page called without capability parameter restriction")
_reg("WAPI-N14", "D5_wp_api", "negative", 1, "phpcs",
     "WordPress.WP.DeprecatedFunctions")  # get_page_by_title


# ---- Dimension 6: i18n / l10n (10 positive, 13 negative = 23) ------------

# Positive
_reg("I18N-P01", "D6_i18n", "positive", 3, "llm",
     "All user-visible strings wrapped in translation function")
_reg("I18N-P02", "D6_i18n", "positive", 2, "phpcs",
     "WordPress.WP.I18n")
_reg("I18N-P03", "D6_i18n", "positive", 2, "regex",
     r"_n\s*\(|_nx\s*\(")
_reg("I18N-P04", "D6_i18n", "positive", 1, "regex",
     r"_x\s*\(|_ex\s*\(")
_reg("I18N-P05", "D6_i18n", "positive", 2, "regex",
     r"esc_html__\s*\(|esc_html_e\s*\(")
_reg("I18N-P06", "D6_i18n", "positive", 1, "regex",
     r"esc_attr__\s*\(|esc_attr_e\s*\(")
_reg("I18N-P07", "D6_i18n", "positive", 1, "regex",
     r"%1\$s.*%2\$s")  # positional placeholders in translated strings
_reg("I18N-P08", "D6_i18n", "positive", 1, "regex",
     r"/\*.*translators.*\*/")  # translators comment before placeholder strings
_reg("I18N-P09", "D6_i18n", "positive", 1, "regex",
     r"number_format_i18n\s*\(")
_reg("I18N-P10", "D6_i18n", "positive", 1, "regex",
     r"date_i18n\s*\(")

# Negative
_reg("I18N-N01", "D6_i18n", "negative", 2, "phpcs",
     "WordPress.WP.I18n")
_reg("I18N-N02", "D6_i18n", "negative", 2, "phpcs",
     "WordPress.WP.I18n")
_reg("I18N-N03", "D6_i18n", "negative", 3, "regex",
     r'__\s*\(\s*"[^"]*\$[a-z_]')  # variable interpolated in translation string
_reg("I18N-N04", "D6_i18n", "negative", 2, "regex",
     r"__\s*\(.*\)\s*\.\s*.*\.\s*__\s*\(")  # sentence split across multiple __()
_reg("I18N-N05", "D6_i18n", "negative", 1, "regex",
     r"%s[^']*%s|%d[^']*%d")  # multiple placeholders without positional numbering
_reg("I18N-N06", "D6_i18n", "negative", 2, "regex",
     r"if\s*\(\s*\$\w+\s*===?\s*1\s*\).*__\(")  # manual plural branching
_reg("I18N-N07", "D6_i18n", "negative", 1, "regex",
     r"__\s*\(.*%s|%d|%\d+\$")  # placeholder in __() without translators comment
_reg("I18N-N08", "D6_i18n", "negative", 1, "regex",
     r"echo.*number_format\s*\(")
_reg("I18N-N09", "D6_i18n", "negative", 1, "regex",
     r"echo.*\bdate\s*\(|echo.*\bgmdate\s*\(")
_reg("I18N-N10", "D6_i18n", "negative", 1, "regex",
     r"__\s*\(\s*'[^']*<[a-z]")  # HTML tags inside translated string
_reg("I18N-N11", "D6_i18n", "negative", 1, "regex",
     r"__\s*\(\s*''")  # empty string translation
_reg("I18N-N12", "D6_i18n", "negative", 2, "llm",
     "_e() used inside HTML attribute context (should use esc_attr_e)")
_reg("I18N-N13", "D6_i18n", "negative", 2, "llm",
     "User-visible English string echoed without translation wrapper")


# ---- Dimension 7: Accessibility (12 positive, 13 negative = 25) ----------

# Positive
_reg("A11Y-P01", "D7_a11y", "positive", 3, "regex",
     r"<label\s+for=")  # with matching id= on input
_reg("A11Y-P02", "D7_a11y", "positive", 2, "regex",
     r"screen-reader-text.*skip|skip.*screen-reader-text")
_reg("A11Y-P03", "D7_a11y", "positive", 2, "regex",
     r"aria-label=|aria-labelledby=")  # on interactive elements
_reg("A11Y-P04", "D7_a11y", "positive", 2, "regex",
     r'role="alert"|aria-live=')
_reg("A11Y-P05", "D7_a11y", "positive", 2, "regex",
     r"<img")  # all instances have alt= (context check)
_reg("A11Y-P06", "D7_a11y", "positive", 1, "regex",
     r"<fieldset.*<legend")
_reg("A11Y-P07", "D7_a11y", "positive", 1, "regex",
     r"<main\b|<nav\b|<aside\b|<header\b|<footer\b")
_reg("A11Y-P08", "D7_a11y", "positive", 1, "llm",
     "Custom focus styles provided when outline:none is present")
_reg("A11Y-P09", "D7_a11y", "positive", 1, "regex",
     r"autocomplete=")  # on personal data form fields
_reg("A11Y-P10", "D7_a11y", "positive", 1, "regex",
     r'aria-required="true".*required|required.*aria-required')
_reg("A11Y-P11", "D7_a11y", "positive", 1, "regex",
     r"aria-describedby=")  # on input elements
_reg("A11Y-P12", "D7_a11y", "positive", 1, "regex",
     r"aria-expanded=|aria-controls=|aria-haspopup=")

# Negative
_reg("A11Y-N01", "D7_a11y", "negative", 4, "llm",
     "Form input lacks any associated label (no <label for>, no aria-label, no title)")
_reg("A11Y-N02", "D7_a11y", "negative", 3, "regex",
     r"<img\s(?![^>]*\balt=)")
_reg("A11Y-N03", "D7_a11y", "negative", 2, "regex",
     r'alt="(?:image|photo|img|picture|icon)"')
_reg("A11Y-N04", "D7_a11y", "negative", 3, "regex",
     r"outline:\s*(?:none|0)")  # without replacement focus indicator
_reg("A11Y-N05", "D7_a11y", "negative", 4, "llm",
     "Icon-only button/link with no accessible name")
_reg("A11Y-N06", "D7_a11y", "negative", 3, "regex",
     r"onclick=")  # on <div>|<span> without role="button" and keyboard handler
_reg("A11Y-N07", "D7_a11y", "negative", 2, "regex",
     r'tabindex="[1-9][0-9]*"')
_reg("A11Y-N08", "D7_a11y", "negative", 3, "llm",
     "Placeholder text used as only accessible label for input")
_reg("A11Y-N09", "D7_a11y", "negative", 1, "regex",
     r'class="notice(?!.*role=)')
_reg("A11Y-N10", "D7_a11y", "negative", 3, "regex",
     r"\.skip-link.*display:\s*none|display:\s*none.*skip")
_reg("A11Y-N11", "D7_a11y", "negative", 2, "llm",
     "Heading elements used out of logical order")
_reg("A11Y-N12", "D7_a11y", "negative", 2, "llm",
     "Color used as sole conveyor of information")
_reg("A11Y-N13", "D7_a11y", "negative", 2, "regex",
     r"<button.*\xc3\x97|<button.*\xe2\x9c\x95")  # dismiss button x/X without aria-label


# ---- Dimension 8: Error Handling (11 positive, 12 negative = 23) ---------

# Positive
_reg("ERR-P01", "D8_errors", "positive", 3, "regex",
     r"return\s+new\s+WP_Error\s*\(")
_reg("ERR-P02", "D8_errors", "positive", 3, "regex",
     r"is_wp_error\s*\(")
_reg("ERR-P03", "D8_errors", "positive", 1, "regex",
     r"new\s+WP_Error\s*\(\s*'[a-z_]+'")
_reg("ERR-P04", "D8_errors", "positive", 1, "regex",
     r"new\s+WP_Error\s*\(.*array\s*\(.*'status'")
_reg("ERR-P05", "D8_errors", "positive", 2, "phpstan",
     "Type declarations on function parameters")
_reg("ERR-P06", "D8_errors", "positive", 2, "phpstan",
     "Return type declarations on functions")
_reg("ERR-P07", "D8_errors", "positive", 1, "regex",
     r"declare\s*\(\s*strict_types\s*=\s*1\s*\)")
_reg("ERR-P08", "D8_errors", "positive", 2, "regex",
     r"try\s*\{")
_reg("ERR-P09", "D8_errors", "positive", 2, "llm",
     "Input validated before use (type check, range check, safelist check)")
_reg("ERR-P10", "D8_errors", "positive", 1, "regex",
     r"wp_die\s*\(.*,.*,\s*array\s*\(.*'response'")
_reg("ERR-P11", "D8_errors", "positive", 1, "regex",
     r"\$wpdb->last_error")

# Negative
_reg("ERR-N01", "D8_errors", "negative", 3, "phpstan",
     "szepeviktor/phpstan-wordpress: WP_Error return used without is_wp_error()")
_reg("ERR-N02", "D8_errors", "negative", 3, "phpcs",
     "WordPress.PHP.NoSilencedErrors")
_reg("ERR-N03", "D8_errors", "negative", 2, "regex",
     r"\bdie\s*\(|\bexit\s*\(")  # outside ABSPATH/WP_UNINSTALL guards
_reg("ERR-N04", "D8_errors", "negative", 2, "regex",
     r"trigger_error\s*\(")
_reg("ERR-N05", "D8_errors", "negative", 1, "phpstan",
     "Missing type hints on function parameters or return type")
_reg("ERR-N06", "D8_errors", "negative", 1, "llm",
     "is_wp_error() check present but error silently discarded")
_reg("ERR-N07", "D8_errors", "negative", 1, "llm",
     "Functions return mixed types without documentation")
_reg("ERR-N08", "D8_errors", "negative", 1, "phpcs",
     "WordPress.PHP.StrictComparisons")
_reg("ERR-N09", "D8_errors", "negative", 2, "regex",
     r"catch\s*\([^)]+\)\s*\{\s*\}")  # empty catch block
_reg("ERR-N10", "D8_errors", "negative", 1, "regex",
     r"wp_send_json_(success|error)\s*\(")  # without wp_die after
_reg("ERR-N11", "D8_errors", "negative", 2, "llm",
     "I/O call without error checking")
_reg("ERR-N12", "D8_errors", "negative", 2, "regex",
     r"mysql_error\s*\(|mysqli_error\s*\(")


# ---- Dimension 9: Code Structure (13 positive, 14 negative = 27) ---------

# Positive
_reg("STR-P01", "D9_structure", "positive", 3, "llm",
     "All filter callbacks return a value (never implicit null)")
_reg("STR-P02", "D9_structure", "positive", 2, "regex",
     r"add_action\s*\(.*,\s*'[a-z_]+'")  # named function, not closure
_reg("STR-P03", "D9_structure", "positive", 2, "phpcs",
     "WordPress.NamingConventions.PrefixAllGlobals, WordPress.NamingConventions.ValidHookName")
_reg("STR-P04", "D9_structure", "positive", 1, "regex",
     r'do_action\s*\(\s*\$\w+\s*\.\s*[\'"]')  # NEGATIVE: absence = correct interpolation
_reg("STR-P05", "D9_structure", "positive", 2, "regex",
     r"register_activation_hook\s*\(")
_reg("STR-P06", "D9_structure", "positive", 1, "regex",
     r"register_deactivation_hook\s*\(")
_reg("STR-P07", "D9_structure", "positive", 2, "file",
     "uninstall.php with WP_UNINSTALL_PLUGIN constant check")
_reg("STR-P08", "D9_structure", "positive", 1, "regex",
     r"add_action\s*\(\s*'rest_api_init'")
_reg("STR-P09", "D9_structure", "positive", 3, "regex",
     r"'permission_callback'")  # in register_rest_route arg array
_reg("STR-P10", "D9_structure", "positive", 2, "regex",
     r"'args'\s*=>\s*array.*'type'\s*=>.*'sanitize_callback'|'validate_callback'")
_reg("STR-P11", "D9_structure", "positive", 2, "regex",
     r"extends\s+WP_REST_Controller")
_reg("STR-P12", "D9_structure", "positive", 1, "llm",
     "Classes have single responsibility; one class per file")
_reg("STR-P13", "D9_structure", "positive", 1, "regex",
     r"rest_ensure_response\s*\(|new\s+WP_REST_Response\s*\(")

# Negative
_reg("STR-N01", "D9_structure", "negative", 5, "llm",
     "Filter callback returns nothing (implicit null) -- corrupts filtered value")
_reg("STR-N02", "D9_structure", "negative", 2, "regex",
     r"add_action\s*\(\s*'[^']+'\s*,\s*function\s*\(")  # closure callback
_reg("STR-N03", "D9_structure", "negative", 3, "phpcs",
     "WordPress.NamingConventions.PrefixAllGlobals")
_reg("STR-N04", "D9_structure", "negative", 1, "llm",
     "Hook priority > 100 without code comment explaining why")
_reg("STR-N05", "D9_structure", "negative", 3, "llm",
     "echo/print inside filter callback (corrupts output buffer)")
_reg("STR-N06", "D9_structure", "negative", 3, "llm",
     "User data deleted in deactivation hook (should be in uninstall only)")
_reg("STR-N07", "D9_structure", "negative", 3, "file",
     "uninstall.php missing WP_UNINSTALL_PLUGIN guard")
_reg("STR-N08", "D9_structure", "negative", 2, "llm",
     "register_rest_route called outside rest_api_init hook")
_reg("STR-N09", "D9_structure", "negative", 5, "regex",
     r"register_rest_route\s*\(")  # presence + absence of permission_callback
_reg("STR-N10", "D9_structure", "negative", 4, "regex",
     r"'methods'.*CREATABLE|EDITABLE|DELETABLE.*__return_true|__return_true.*methods.*POST|PUT|DELETE")
_reg("STR-N11", "D9_structure", "negative", 3, "regex",
     r"wp_send_json\s*\(|die\s*\(")  # inside register_rest_route callback
_reg("STR-N12", "D9_structure", "negative", 1, "regex",
     r"add_action\s*\(\s*'wp_ajax_")  # handler without wp_die()
_reg("STR-N13", "D9_structure", "negative", 2, "regex",
     r"add_action\s*\(\s*'init'.*flush_rewrite_rules\s*\(")
_reg("STR-N14", "D9_structure", "negative", 1, "regex",
     r"do_action\s*\(\s*\$\w+\s*\.\s*['\"]")  # hook name via concatenation


# ---------------------------------------------------------------------------
# Verify total count
# ---------------------------------------------------------------------------

_positive_count = sum(1 for c in CHECK_REGISTRY.values() if c.polarity == "positive")
_negative_count = sum(1 for c in CHECK_REGISTRY.values() if c.polarity == "negative")
assert _positive_count == 105, f"Expected 105 positive checks, got {_positive_count}"
assert _negative_count == 136, f"Expected 136 negative checks, got {_negative_count}"
assert len(CHECK_REGISTRY) == 241, f"Expected 241 total checks, got {len(CHECK_REGISTRY)}"


# ---------------------------------------------------------------------------
# 4. SNIFF_TO_CHECKS  (Section F Step 3 automation mapping)
# ---------------------------------------------------------------------------

SNIFF_TO_CHECKS: Dict[str, List[str]] = {
    # WordPress.Security.*
    "WordPress.Security.EscapeOutput": ["SEC-N01", "SEC-N15", "SEC-N16"],
    "WordPress.Security.ValidatedSanitizedInput": ["SEC-N02"],
    "WordPress.Security.NonceVerification": ["SEC-N03"],
    "WordPress.Security.SafeRedirect": ["SEC-N05"],
    "WordPress.Security.PluginMenuSlug": ["SEC-N14"],

    # WordPress.DB.*
    "WordPress.DB.PreparedSQL": ["SQL-N01", "SQL-N03"],
    "WordPress.DB.PreparedSQLPlaceholders": ["SQL-N05", "SQL-N06"],
    "WordPress.DB.RestrictedFunctions": ["SQL-N09"],
    "WordPress.DB.RestrictedClasses": ["SQL-N09"],
    "WordPress.DB.SlowDBQuery": ["PERF-N01"],

    # WordPress.WP.*
    "WordPress.WP.DiscouragedFunctions": ["SQL-N10", "WAPI-N06"],
    "WordPress.WP.DeprecatedFunctions": ["WAPI-N07", "WAPI-N14"],
    "WordPress.WP.DeprecatedClasses": ["WAPI-N08"],
    "WordPress.WP.AlternativeFunctions": [
        "WAPI-N01", "WAPI-N02", "WAPI-N03", "WAPI-N04", "WAPI-N05",
    ],
    "WordPress.WP.EnqueuedResources": ["PERF-N08", "WAPI-P06"],
    "WordPress.WP.EnqueuedResourceParameters": ["PERF-P09"],
    "WordPress.WP.PostsPerPage": ["SQL-N12"],
    "WordPress.WP.I18n": ["I18N-N01", "I18N-N02", "I18N-N03"],
    "WordPress.WP.GlobalVariablesOverride": ["WPCS-N20"],

    # WordPress.NamingConventions.*
    "WordPress.NamingConventions.PrefixAllGlobals": ["WPCS-N03", "STR-N03"],
    "WordPress.NamingConventions.ValidFunctionName": ["WPCS-N02"],
    "WordPress.NamingConventions.ValidHookName": ["STR-N03"],

    # WordPress.PHP.*
    "WordPress.PHP.YodaConditions": ["WPCS-P05"],
    "WordPress.PHP.StrictInArray": ["WPCS-N08"],
    "WordPress.PHP.StrictComparisons": ["WPCS-N09", "ERR-N08"],
    "WordPress.PHP.DontExtract": ["WPCS-N10"],
    "WordPress.PHP.RestrictedPHPFunctions": ["WPCS-N11", "WPCS-N12"],
    "WordPress.PHP.NoSilencedErrors": ["WPCS-N14", "ERR-N02"],
    "WordPress.PHP.DevelopmentFunctions": ["WPCS-N15"],
    "WordPress.PHP.PregQuoteDelimiter": ["WPCS-N19"],

    # WordPress.DateTime.*
    "WordPress.DateTime.CurrentTimeTimestamp": ["WPCS-N16"],
    "WordPress.DateTime.RestrictedFunctions": ["WAPI-N05"],

    # WordPress.CodeAnalysis.*
    "WordPress.CodeAnalysis.AssignmentInTernaryCondition": ["WPCS-N17"],
    "WordPress.CodeAnalysis.EscapedNotTranslated": ["WPCS-N18"],

    # WordPress.WhiteSpace.*
    "WordPress.WhiteSpace.ControlStructureSpacing": ["WPCS-P06"],
    "WordPress.WhiteSpace.OperatorSpacing": ["WPCS-P07"],

    # WordPress.Files.*
    "WordPress.Files.FileName": ["WPCS-P09"],

    # Generic (bundled with WPCS)
    "Generic.PHP.DisallowShortOpenTag": ["WPCS-N01"],
    "Generic.ControlStructures.InlineControlStructure": ["WPCS-P11"],

    # Squiz (bundled with WPCS)
    "Squiz.Scope.MemberVarScope": ["WPCS-N04"],
    "Squiz.Scope.MethodScope": ["WPCS-N05"],

    # WordPressVIPMinimum (VIP standard)
    # These map to the same checks already covered above via WordPress.*
    # but trigger through the VIP standard run.

    # Security audit (pheromone/phpcs-security-audit)
    "Security.BadFunctions.EasyXSS": ["SEC-N01"],
    "Security.BadFunctions.PHPInternalFunctions": ["SEC-N06", "SEC-N07"],
    "Security.BadFunctions.FilesystemFunctions": ["SEC-N08"],
    "Security.BadFunctions.SystemExecFunctions": ["SEC-N19"],
}
"""Maps PHPCS sniff source prefixes to lists of check IDs they can trigger.

Covers WordPress, WordPressVIPMinimum, Generic, Squiz, and Security standards
as defined in rubric Section F Step 3.
"""


# ---------------------------------------------------------------------------
# 5. REGEX_PATTERNS  (all regex / regex+ checks)
# ---------------------------------------------------------------------------

REGEX_PATTERNS: Dict[str, str] = {
    # --- D1 WPCS ---
    "WPCS-P01": r"^ {4,}",  # space-only indentation (not tab)
    "WPCS-P04": r"define\s*\(\s*'[A-Z][A-Z0-9_]+'",
    "WPCS-P12": r"else\s+if\b",  # negative: presence means NOT using elseif
    "WPCS-N06": r"\?>\s*$",
    "WPCS-N07": r"\?:",  # Elvis operator in expression context
    "WPCS-N13": r"\bgoto\b",

    # --- D2 Security ---
    "SEC-P02": r"wp_unslash\s*\(.*sanitize|sanitize.*wp_unslash",
    "SEC-P07": r"wp_check_filetype_and_ext\s*\(",
    "SEC-P08": r"wp_handle_upload\s*\(",
    "SEC-P09": r"is_email\s*\(",
    "SEC-P11": r"absint\s*\(",
    "SEC-P12": r"esc_html__\s*\(|esc_attr__\s*\(",
    "SEC-N09": r"\$_FILES\[.*\]\[.type.\]",
    "SEC-N11": r"wp_ajax_nopriv_",
    "SEC-N12": r"current_user_can\s*\(\s*['\"](?:administrator|editor|author|contributor|subscriber)['\"]",
    "SEC-N18": r"\$_REQUEST\b",
    "SEC-N20": r"preg_replace\s*\(.*\/e['\"]",

    # --- D3 SQL ---
    "SQL-P02": r"\$wpdb->(insert|update|delete)\s*\(",
    "SQL-P03": r"esc_like\s*\(",
    "SQL-P04": r"new\s+WP_Query\s*\(|get_posts\s*\(",
    "SQL-P05": r"prepare\s*\(.*%i",
    "SQL-P06": r'".*wp_[a-z_]+.*"',  # negative match: absence means using $wpdb->tablename
    "SQL-P09": r"\$wpdb->last_error",
    "SQL-N01": r'\$wpdb->.*\(.*"\s*.*\$[a-z_]+',  # direct variable interpolation in SQL
    "SQL-N02": r'["\'].*\.\s*\$[a-z_].*["\']',  # string concat in SQL
    "SQL-N04": r"LIKE\s+%s",  # without preceding esc_like
    "SQL-N07": r"->escape\s*\(",
    "SQL-N08": r"prepare\s*\(.*esc_sql",
    "SQL-N11": r'"[^"]*wp_[a-z_]+[^"]*"',  # hardcoded table name
    "SQL-N12": r"'posts_per_page'\s*=>\s*-1",
    "SQL-N13": r"'orderby'\s*=>\s*'rand'",
    "SQL-N14": r"'suppress_filters'\s*=>\s*true",
    "SQL-N15": r"SELECT\s+\*\s+FROM",
    "SQL-N17": r"prepare\s*\(\s*\$_(GET|POST|REQUEST)",

    # --- D4 Performance ---
    "PERF-P01": r"wp_cache_get\s*\(",
    "PERF-P02": r"set_transient\s*\(\s*[^,]+,\s*[^,]+,\s*[^)]+\)",
    "PERF-P03": r"false\s*===?\s*get_transient|get_transient.*!==?\s*false",
    "PERF-P05": r"'no_found_rows'\s*=>\s*true",
    "PERF-P06": r"'fields'\s*=>\s*'ids'",
    "PERF-P07": r"'update_post_meta_cache'\s*=>\s*false|'update_post_term_cache'\s*=>\s*false",
    "PERF-P08": r"add_option\s*\(.*,.*,.*,\s*false|update_option\s*\(.*,.*,\s*false",
    "PERF-P11": r"wp_remote_get\s*\(.*'timeout'",
    "PERF-P12": r"wp_remote_retrieve_response_code\s*\(",
    "PERF-N02": r"\$wpdb->",  # inside foreach|while body (needs context check)
    "PERF-N03": r"set_transient\s*\(\s*['\"\w]+\s*,\s*[^,)]+\s*\)",  # 2-arg form
    "PERF-N04": r"wp_cache_flush\s*\(\s*\)",
    "PERF-N07": r"curl_exec\s*\(",
    "PERF-N09": r"add_action\s*\(\s*'init'.*flush_rewrite_rules",
    "PERF-N12": r"wp_remote_get\s*\(",  # without nearby is_wp_error
    "PERF-N13": r"'orderby'\s*=>\s*'rand'",

    # --- D5 WP API ---
    "WAPI-P07": r"WP_Filesystem\(\)|\$wp_filesystem->",
    "WAPI-P09": r"add_action\s*\(\s*'init'.*register_post_type|register_taxonomy",
    "WAPI-P10": r"wp_localize_script\s*\(|wp_add_inline_script\s*\(",
    "WAPI-P11": r"plugins_url\s*\(|plugin_dir_url\s*\(",
    "WAPI-P12": r"query_posts\s*\(",  # negative match: absence = correct
    "WAPI-N02": r"curl_init\s*\(|curl_exec\s*\(|curl_setopt\s*\(",
    "WAPI-N09": r"require.*phpmailer|wp_enqueue_script.*jquery.*http",
    "WAPI-N10": r"\bfopen\s*\(|file_put_contents\s*\(|\bfwrite\s*\(",
    "WAPI-N11": r"session_start\s*\(|\$_SESSION",

    # --- D6 i18n ---
    "I18N-P03": r"_n\s*\(|_nx\s*\(",
    "I18N-P04": r"_x\s*\(|_ex\s*\(",
    "I18N-P05": r"esc_html__\s*\(|esc_html_e\s*\(",
    "I18N-P06": r"esc_attr__\s*\(|esc_attr_e\s*\(",
    "I18N-P07": r"%1\$s.*%2\$s",
    "I18N-P08": r"/\*.*translators.*\*/",
    "I18N-P09": r"number_format_i18n\s*\(",
    "I18N-P10": r"date_i18n\s*\(",
    "I18N-N03": r'__\s*\(\s*"[^"]*\$[a-z_]',
    "I18N-N04": r"__\s*\(.*\)\s*\.\s*.*\.\s*__\s*\(",
    "I18N-N05": r"%s[^']*%s|%d[^']*%d",
    "I18N-N06": r"if\s*\(\s*\$\w+\s*===?\s*1\s*\).*__\(",
    "I18N-N07": r"__\s*\(.*%s|%d|%\d+\$",
    "I18N-N08": r"echo.*number_format\s*\(",
    "I18N-N09": r"echo.*\bdate\s*\(|echo.*\bgmdate\s*\(",
    "I18N-N10": r"__\s*\(\s*'[^']*<[a-z]",
    "I18N-N11": r"__\s*\(\s*''",

    # --- D7 Accessibility ---
    "A11Y-P01": r"<label\s+for=",
    "A11Y-P02": r"screen-reader-text.*skip|skip.*screen-reader-text",
    "A11Y-P03": r"aria-label=|aria-labelledby=",
    "A11Y-P04": r'role="alert"|aria-live=',
    "A11Y-P05": r"<img",  # all instances should have alt= (context check)
    "A11Y-P06": r"<fieldset.*<legend",
    "A11Y-P07": r"<main\b|<nav\b|<aside\b|<header\b|<footer\b",
    "A11Y-P09": r"autocomplete=",
    "A11Y-P10": r'aria-required="true".*required|required.*aria-required',
    "A11Y-P11": r"aria-describedby=",
    "A11Y-P12": r"aria-expanded=|aria-controls=|aria-haspopup=",
    "A11Y-N02": r"<img\s(?![^>]*\balt=)",
    "A11Y-N03": r'alt="(?:image|photo|img|picture|icon)"',
    "A11Y-N04": r"outline:\s*(?:none|0)",
    "A11Y-N06": r"onclick=",  # on div/span without role="button"|onkeydown
    "A11Y-N07": r'tabindex="[1-9][0-9]*"',
    "A11Y-N09": r'class="notice(?!.*role=)',
    "A11Y-N10": r"\.skip-link.*display:\s*none|display:\s*none.*skip",
    "A11Y-N13": r"<button.*\xc3\x97|<button.*\xe2\x9c\x95",

    # --- D8 Error Handling ---
    "ERR-P01": r"return\s+new\s+WP_Error\s*\(",
    "ERR-P02": r"is_wp_error\s*\(",
    "ERR-P03": r"new\s+WP_Error\s*\(\s*'[a-z_]+'",
    "ERR-P04": r"new\s+WP_Error\s*\(.*array\s*\(.*'status'",
    "ERR-P07": r"declare\s*\(\s*strict_types\s*=\s*1\s*\)",
    "ERR-P08": r"try\s*\{",
    "ERR-P10": r"wp_die\s*\(.*,.*,\s*array\s*\(.*'response'",
    "ERR-P11": r"\$wpdb->last_error",
    "ERR-N03": r"\bdie\s*\(|\bexit\s*\(",
    "ERR-N04": r"trigger_error\s*\(",
    "ERR-N09": r"catch\s*\([^)]+\)\s*\{\s*\}",
    "ERR-N10": r"wp_send_json_(success|error)\s*\(",
    "ERR-N12": r"mysql_error\s*\(|mysqli_error\s*\(",

    # --- D9 Code Structure ---
    "STR-P02": r"add_action\s*\(.*,\s*'[a-z_]+'",
    "STR-P04": r"do_action\s*\(\s*\$\w+\s*\.\s*['\"]",  # negative: absence = correct
    "STR-P05": r"register_activation_hook\s*\(",
    "STR-P06": r"register_deactivation_hook\s*\(",
    "STR-P08": r"add_action\s*\(\s*'rest_api_init'",
    "STR-P09": r"'permission_callback'",
    "STR-P10": r"'args'\s*=>\s*array.*'type'\s*=>.*'sanitize_callback'|'validate_callback'",
    "STR-P11": r"extends\s+WP_REST_Controller",
    "STR-P13": r"rest_ensure_response\s*\(|new\s+WP_REST_Response\s*\(",
    "STR-N02": r"add_action\s*\(\s*'[^']+'\s*,\s*function\s*\(",
    "STR-N09": r"register_rest_route\s*\(",  # + absence of permission_callback
    "STR-N10": r"'methods'.*CREATABLE|EDITABLE|DELETABLE.*__return_true|__return_true.*methods.*POST|PUT|DELETE",
    "STR-N11": r"wp_send_json\s*\(|die\s*\(",  # inside REST callback
    "STR-N12": r"add_action\s*\(\s*'wp_ajax_",
    "STR-N13": r"add_action\s*\(\s*'init'.*flush_rewrite_rules\s*\(",
    "STR-N14": r"do_action\s*\(\s*\$\w+\s*\.\s*['\"]",
}
"""Maps check IDs to raw regex pattern strings for all regex and regex+ checks.

Patterns are raw strings suitable for ``re.compile()``.  For regex+ checks,
the caller must apply additional context awareness (e.g. scope, proximity).
"""


# ---------------------------------------------------------------------------
# 6. CRITICAL_FLOOR_RULES
# ---------------------------------------------------------------------------

CRITICAL_FLOOR_RULES: List[Tuple[str, float, List[str]]] = [
    (
        "D2_security",
        3.0,
        ["SEC-N01", "SEC-N03", "SEC-N04", "SEC-N06", "SEC-N08", "SEC-N19", "SEC-N20"],
    ),
    (
        "D3_sql",
        2.0,
        ["SQL-N01", "SQL-N03", "SQL-N17"],
    ),
    (
        "D9_structure",
        4.0,
        ["STR-N01", "STR-N09"],
    ),
]
"""Floor rules: (dimension_key, max_score_when_triggered, trigger_check_ids).

If ANY trigger check is present, the dimension score cannot exceed max_score.
Source: rubric critical floor rules in Sections 2C, 3C, 9C.
"""


# ---------------------------------------------------------------------------
# 7. GRADE_BANDS
# ---------------------------------------------------------------------------

GRADE_BANDS: List[Tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Acceptable"),
    (40.0, "Poor"),
    (20.0, "Bad"),
    (0.0, "Failing"),
]
"""Score interpretation bands: (min_score_inclusive, label).

Iterate from top; first band where ``score >= min_score`` is the grade.
Source: rubric Section D Score Interpretation Bands.
"""


# ---------------------------------------------------------------------------
# 8. NA_DETECTION_HINTS
# ---------------------------------------------------------------------------

NA_DETECTION_HINTS: Dict[str, str] = {
    "D1_wpcs": r"<\?php|function\s+\w+|class\s+\w+|\$\w+",  # any PHP-like content
    "D2_security": r"\$_(GET|POST|REQUEST|COOKIE|FILES|SERVER)|echo|print|wp_send_json|return\s+new\s+WP_|\$this->|wp_remote|\$wpdb",
    "D3_sql": r"\$wpdb|WP_Query|get_posts\s*\(|query_posts\s*\(",
    "D4_perf": r"\$wpdb|wp_remote_get|wp_remote_post|wp_enqueue_script|wp_enqueue_style|WP_Query|get_posts",
    "D5_wp_api": r"add_action|add_filter|wp_enqueue|register_post_type|wp_remote|WP_Filesystem|WP_Query|\$wpdb|get_posts|do_action|apply_filters|register_rest_route|wp_\w+\s*\(|add_meta_box|wp_send_json|register_meta",
    "D6_i18n": r"echo\s+['\"]|_e\s*\(|__\s*\(|_n\s*\(|_x\s*\(|esc_html__\s*\(|esc_attr__\s*\(|print\s+['\"]",
    "D7_a11y": r"<(input|select|textarea|button|img|form|a\s|div\s|span\s)|echo\s+['\"]<|->render\s*\(",
    "D8_errors": r"function\s+\w+|wp_remote|wpdb|\$_|try\s*\{|throw\s+new|is_wp_error",
    "D9_structure": r"add_action|add_filter|register_rest_route|register_activation_hook|class\s+\w+|interface\s+\w+|trait\s+\w+|function\s+\w+|public\s+function|private\s+function|protected\s+function",
}
"""Regex patterns to detect whether a dimension IS applicable to a code sample.

If the pattern does NOT match, the dimension may be scored N/A.
The scorer should confirm with additional heuristics before marking N/A.
Source: rubric N/A rules per dimension.
"""


# ---------------------------------------------------------------------------
# 9. DIM_NAME_MAP
# ---------------------------------------------------------------------------

DIM_NAME_MAP: Dict[str, str] = {
    # model output field name -> internal dimension key
    "wpcs_compliance": "D1_wpcs",
    "security_score": "D2_security",
    "sql_safety": "D3_sql",
    "performance": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "i18n_l10n": "D6_i18n",
    "accessibility": "D7_a11y",
    "error_handling": "D8_errors",
    "code_structure": "D9_structure",
    # Also support reverse lookups by including dim key -> field name
    "D1_wpcs": "wpcs_compliance",
    "D2_security": "security_score",
    "D3_sql": "sql_safety",
    "D4_perf": "performance",
    "D5_wp_api": "wp_api_usage",
    "D6_i18n": "i18n_l10n",
    "D7_a11y": "accessibility",
    "D8_errors": "error_handling",
    "D9_structure": "code_structure",
}
"""Bidirectional mapping between model output field names and internal dimension keys."""
