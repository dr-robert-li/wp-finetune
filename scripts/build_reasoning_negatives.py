#!/usr/bin/env python
"""Build invalid-PHP / fabricated-API should_fail TRAINING negatives (cot format).

Corrective branch for the REVL-05 judge-quality critical: the P4 model false-passed
4/24 invalid-PHP sentinel rows (a real `fn()->` parse error @51, a fabricated `wp_*`
call scored 100). Root cause = the SFT mix lacked syntactic-invalidity / fabricated-API
coverage in the SHORT-isolated-snippet regime + a lenient PASS boundary.

These negatives teach: short obviously-broken snippet -> low relevant-dim scores ->
low overall_score -> verdict FAIL. Emitted in the **cot** target format (prose
`Dim: score X/10 — reason` + [/REASONING] + judge_output{dims, overall_score, verdict})
because that is the format the model emits on these `<wp_judge>` prompts (the sentinel
false-passes all carried an overall_score).

LEAKAGE DISCIPLINE: every snippet here is a DISTINCT concrete instance from the 24
held-out sentinel rows (`build_invalid_php_sentinel.py`) — same defect CLASSES, different
code — so the sentinel tests generalization, not memorization.

Verdict policy (v1.2): PASS iff overall_score >= 70 AND no auto-FAIL defect class
(syntax/parse error, fabricated/non-existent WP API, out-of-context fatal `$this`/`self`,
unsanitized SQL/XSS). All rows here are auto-FAIL classes -> verdict FAIL, overall < 40.

Usage: python scripts/build_reasoning_negatives.py --out data/reasoning_dataset/negatives_train.jsonl
"""
import argparse
import json
import os

DIM_ORDER = ["wpcs_compliance", "sql_safety", "security", "performance", "wp_api_usage",
             "code_quality", "dependency_integrity", "i18n", "accessibility"]
DIM_LABELS = {
    "wpcs_compliance": "WPCS Compliance", "sql_safety": "SQL Safety", "security": "Security",
    "performance": "Performance", "wp_api_usage": "WP API Usage", "code_quality": "Code Quality",
    "dependency_integrity": "Dependency Integrity", "i18n": "i18n", "accessibility": "Accessibility",
}

# Default (non-defect) dim fillers — teacher-style. Respect the max-2-N/A rule: only
# sql_safety + i18n use None; accessibility takes a low real score when inapplicable.
DEFAULTS = {
    "wpcs_compliance": (4, "No PHPDoc block and the snippet shows formatting/style gaps; below WPCS expectations."),
    "sql_safety": (None, "No $wpdb usage or SQL strings — N/A applies."),
    "security": (6, "No direct user-input handling in this snippet; no specific injection sink observed."),
    "performance": (8, "Trivial control flow; performance is not a concern here."),
    "wp_api_usage": (5, "Minimal WordPress API integration shown in the snippet."),
    "code_quality": (4, "Sparse, context-light snippet with structural problems (see summary)."),
    "dependency_integrity": (6, "No external dependencies beyond core are referenced."),
    "i18n": (None, "No user-facing translatable strings — N/A applies."),
    "accessibility": (3, "No HTML output; accessibility is not applicable to this code."),
}

# Each case: code, summary line, overall_score, and DEFECT-specific dim overrides.
# overall < 40 (clear FAIL). Snippets are DISTINCT from the sentinel set.
CASES = [
    # ---- SYNTAX / PARSE ERRORS (auto-FAIL) ----
    ("syntax_missing_close_brace",
     "function lp_register_routes() {\n    register_rest_route( 'lp/v1', '/items', array(\n        'methods'  => 'GET',\n        'callback' => 'lp_get_items',\n    ) );\n    foreach ( $routes as $r ) {\n        do_action( 'lp_route', $r );\n",
     16, {"code_quality": (1, "The function body is missing its closing brace and `$routes` is never defined — this does not parse; PHP raises a fatal `syntax error, unexpected end of file`."),
          "wpcs_compliance": (2, "Unparseable code cannot satisfy WPCS; the unterminated block is a hard error.")}),
    ("syntax_missing_semicolon2",
     "function get_site_slug() {\n    $slug = sanitize_title( get_bloginfo( 'name' ) )\n    return $slug;\n}",
     20, {"code_quality": (1, "Missing semicolon after the `sanitize_title(...)` assignment — PHP raises `syntax error, unexpected 'return'`. The function never parses."),
          "wpcs_compliance": (3, "A parse error precludes any standards compliance.")}),
    ("syntax_unterminated_heredoc",
     "function render_box() {\n    $html = <<<HTML\n    <div class=\"box\">Hello</div>\n    return $html;\n}",
     15, {"code_quality": (1, "The heredoc opened with `<<<HTML` is never closed by a `HTML;` terminator at column 0 — fatal parse error; everything after is swallowed into the string."),
          "security": (4, "Even if it parsed, the heredoc emits markup without escaping.")}),
    ("syntax_mismatched_brackets",
     "add_filter( 'body_class', function( $classes ) {\n    $classes[] = 'custom';\n    return $classes;\n);",
     18, {"code_quality": (1, "The closure's closing `}` is missing before `);` — bracket mismatch produces a fatal parse error."),
          "wpcs_compliance": (3, "Unbalanced brackets cannot pass linting.")}),
    ("syntax_bad_cast_operator",
     "function to_int( $v ) {\n    return (integer)) $v;\n}",
     17, {"code_quality": (1, "`(integer)) $v` has a stray extra parenthesis after the cast — `syntax error, unexpected ')'`. Does not compile."),
          "wpcs_compliance": (3, "Parse failure; standards are moot.")}),
    ("syntax_double_arrow_in_call",
     "function build_query() {\n    return new WP_Query( array( 'post_type' => => 'post' ) );\n}",
     18, {"wp_api_usage": (3, "WP_Query is the right API, but the array literal has a doubled `=>` token."),
          "code_quality": (1, "`'post_type' => => 'post'` is a duplicated `=>` — fatal parse error; the array never builds.")}),
    ("syntax_unclosed_comment",
     "function init_plugin() {\n    /* bootstrap the plugin\n    add_action( 'init', 'plugin_setup' );\n}",
     16, {"code_quality": (1, "The block comment `/*` is never closed with `*/`, so the rest of the file is commented out and the function body/closing brace are consumed — fatal."),
          "wpcs_compliance": (3, "Unterminated comment cannot lint.")}),
    ("syntax_missing_function_keyword",
     "register_widget() {\n    return new My_Widget();\n}",
     19, {"code_quality": (1, "Missing the `function` keyword before `register_widget()` — PHP parses this as a call followed by an unexpected `{` and errors."),
          "wpcs_compliance": (3, "Not a valid function declaration.")}),

    # ---- FABRICATED / NON-EXISTENT WP APIs (auto-FAIL; the @100 blind spot) ----
    ("fabricated_get_current_theme_meta",
     "function theme_tagline() {\n    return wp_get_current_theme_meta( 'tagline' );\n}",
     22, {"wp_api_usage": (1, "`wp_get_current_theme_meta()` is not a WordPress function and does not exist in core. Calling it throws a fatal `Call to undefined function`. The real API is `wp_get_theme()->get( 'Description' )`."),
          "code_quality": (3, "Built on a hallucinated API; the function cannot run."),
          "dependency_integrity": (2, "Depends on a non-existent core function.")}),
    ("fabricated_register_admin_widget",
     "function add_dashboard() {\n    register_admin_dashboard_widget( 'stats', 'Stats', 'render_stats' );\n}",
     23, {"wp_api_usage": (1, "`register_admin_dashboard_widget()` does not exist; the correct API is `wp_add_dashboard_widget()`. This is a fabricated function and fatals at runtime."),
          "code_quality": (3, "Relies on an invented API."),
          "dependency_integrity": (2, "Hard dependency on a function that is not defined anywhere.")}),
    ("fabricated_sanitize_html_deep",
     "function clean_input( $data ) {\n    return wp_sanitize_html_deep( $data );\n}",
     24, {"wp_api_usage": (1, "`wp_sanitize_html_deep()` is not a real WordPress function. Recursive sanitization is done with `map_deep( $data, 'sanitize_text_field' )` or `wp_kses_post`. As written it fatals."),
          "security": (3, "Intends to sanitize but calls a non-existent sanitizer, so no sanitization actually occurs."),
          "code_quality": (3, "Built on a fabricated helper.")}),
    ("fabricated_get_user_capability",
     "function can_edit( $uid ) {\n    return wp_get_user_capability( $uid, 'edit_posts' );\n}",
     25, {"wp_api_usage": (1, "`wp_get_user_capability()` does not exist. Capability checks use `user_can( $uid, 'edit_posts' )` or `current_user_can()`. The call is a hallucination and fatals."),
          "security": (2, "A broken capability check means the gate never actually runs — authorization is effectively absent."),
          "code_quality": (3, "Invented API.")}),
    ("fabricated_enqueue_inline",
     "function add_css() {\n    wp_enqueue_inline_style( 'main', '.a{color:red}' );\n}",
     24, {"wp_api_usage": (1, "`wp_enqueue_inline_style()` is not a core function. Inline CSS uses `wp_add_inline_style()` attached to an enqueued handle. This fabricated call fatals."),
          "code_quality": (3, "Relies on a non-existent enqueue helper.")}),
    ("fabricated_query_meta_like",
     "function find_by_meta( $key ) {\n    return wp_query_posts_by_meta_like( $key, '%draft%' );\n}",
     23, {"wp_api_usage": (1, "`wp_query_posts_by_meta_like()` is invented; meta queries use `WP_Query` with a `meta_query` `LIKE` compare. The call does not exist and fatals."),
          "code_quality": (3, "Hallucinated query helper."),
          "dependency_integrity": (2, "Depends on a function absent from core and plugins.")}),

    # ---- $this / self OUTSIDE CLASS (auto-FAIL fatal) ----
    ("this_in_function_render",
     "function render_template() {\n    echo $this->get_html();\n}",
     21, {"code_quality": (1, "`$this` is referenced inside a plain function with no enclosing class — PHP fatals with `Using $this when not in object context`. This is not a valid standalone function."),
          "security": (4, "It also echoes `get_html()` output without escaping, but the fatal `$this` use dominates.")}),
    ("self_const_outside_class",
     "function api_base() {\n    return self::API_ROOT . '/v2';\n}",
     22, {"code_quality": (1, "`self::` is used outside any class context — fatal `Cannot access self:: when no class scope is active`. The function cannot execute."),
          "wp_api_usage": (4, "No WordPress API is involved; the scope error is the blocker.")}),
    ("parent_call_outside_class",
     "function setup() {\n    parent::__construct();\n    add_action( 'init', 'go' );\n}",
     23, {"code_quality": (1, "`parent::__construct()` appears in a free function with no parent class — fatal scope error. The hook registration never runs."),
          "wpcs_compliance": (4, "Beyond the fatal, there is no docblock or prefix.")}),

    # ---- UNSANITIZED SQL (auto-FAIL critical) ----
    ("sqli_orderby_concat",
     "function list_items() {\n    global $wpdb;\n    $col = $_GET['sort'];\n    return $wpdb->get_results( \"SELECT * FROM {$wpdb->posts} ORDER BY $col\" );\n}",
     20, {"sql_safety": (1, "`$_GET['sort']` is concatenated straight into the ORDER BY clause with no allowlist or `$wpdb->prepare()` — a textbook SQL injection. ORDER BY cannot be parameterized, so it must be validated against a fixed column allowlist."),
          "security": (1, "Unauthenticated SQL injection sink."),
          "code_quality": (4, "No input validation at all.")}),
    ("sqli_in_clause",
     "function posts_in() {\n    global $wpdb;\n    $ids = $_POST['ids'];\n    return $wpdb->get_results( \"SELECT * FROM {$wpdb->posts} WHERE ID IN ($ids)\" );\n}",
     19, {"sql_safety": (1, "`$_POST['ids']` is interpolated directly into the IN() list — SQL injection. The safe pattern builds placeholders and uses `$wpdb->prepare()` with the id array."),
          "security": (1, "Direct injection from request body."),
          "code_quality": (4, "No sanitization or integer casting of ids.")}),
    ("sqli_like_search",
     "function search_terms() {\n    global $wpdb;\n    $q = $_GET['q'];\n    return $wpdb->get_col( \"SELECT name FROM {$wpdb->terms} WHERE name LIKE '%$q%'\" );\n}",
     20, {"sql_safety": (1, "User input `$_GET['q']` is concatenated into a LIKE pattern unprepared — injectable. Use `$wpdb->prepare( '... LIKE %s', '%' . $wpdb->esc_like( $q ) . '%' )`."),
          "security": (1, "Reflected SQL injection sink."),
          "code_quality": (4, "Missing esc_like and prepare.")}),

    # ---- UNESCAPED OUTPUT / XSS (auto-FAIL critical) ----
    ("xss_admin_notice",
     "function notice() {\n    echo '<div class=\"notice\">' . $_GET['msg'] . '</div>';\n}",
     21, {"security": (1, "`$_GET['msg']` is echoed into HTML with no escaping — reflected XSS. Output must pass through `esc_html()` and ideally a nonce-gated context."),
          "wpcs_compliance": (3, "WPCS requires late escaping on all output; none is present."),
          "code_quality": (4, "Raw superglobal echoed directly.")}),
    ("xss_shortcode_attr",
     "function sc_handler( $atts ) {\n    return '<span title=\"' . $atts['label'] . '\">x</span>';\n}",
     24, {"security": (2, "Shortcode attribute `$atts['label']` is placed into an HTML attribute unescaped — stored/reflected XSS. Use `esc_attr()` for attribute context."),
          "wpcs_compliance": (3, "No `shortcode_atts()` defaults and no attribute escaping.")}),
    ("xss_setting_field",
     "function field() {\n    printf( '<input name=\"opt\" value=\"%s\">', get_option( 'opt' ) );\n}",
     26, {"security": (2, "The stored option is printed into a value attribute with no `esc_attr()`; a tampered option yields attribute-breakout XSS in wp-admin."),
          "wpcs_compliance": (3, "Output escaping is mandatory even for stored data.")}),

    # ---- MISSING NONCE / CAPABILITY ON STATE CHANGE (auto-FAIL) ----
    ("no_nonce_form_handler",
     "add_action( 'admin_post_save_opt', function() {\n    update_option( 'mode', $_POST['mode'] );\n    wp_redirect( admin_url() );\n} );",
     23, {"security": (1, "The handler writes an option from `$_POST` with no nonce verification (`check_admin_referer`) and no `current_user_can()` capability check — CSRF + privilege-escalation vector. Input is also unsanitized."),
          "wp_api_usage": (4, "Uses admin_post correctly but omits the required nonce/cap guards."),
          "code_quality": (4, "No validation of `mode`.")}),
    ("no_cap_ajax_delete",
     "add_action( 'wp_ajax_rm', function() {\n    wp_delete_post( (int) $_POST['id'], true );\n    wp_die();\n} );",
     22, {"security": (1, "An AJAX handler that force-deletes a post with no `check_ajax_referer()` nonce and no capability check — any logged-in user can delete arbitrary posts. Authorization is missing."),
          "wp_api_usage": (4, "Correct AJAX hook, but the security guards required for a destructive action are absent.")}),

    # ---- INFINITE RECURSION / LOGIC FATALS (auto-FAIL) ----
    ("infinite_recursion_filter",
     "add_filter( 'the_title', function( $t ) {\n    return apply_filters( 'the_title', $t );\n} );",
     24, {"code_quality": (1, "The `the_title` filter callback re-applies `the_title` on itself — unbounded recursion ending in a stack-overflow fatal on every title render."),
          "performance": (1, "Infinite recursion; the request never completes."),
          "wp_api_usage": (4, "Misuses the filter API by recursing into the same hook.")}),
    ("undefined_var_use",
     "function tax_links() {\n    foreach ( $terms as $t ) {\n        echo esc_html( $t->name );\n    }\n}",
     27, {"code_quality": (2, "`$terms` is never defined or fetched (no `get_terms()` call) — the loop iterates over null, emitting a warning and producing nothing. The snippet is non-functional as given."),
          "wp_api_usage": (4, "Expected a `get_terms()`/`wp_get_post_terms()` call to populate `$terms`; none is present.")}),
    ("wrong_callback_arity",
     "add_action( 'save_post', 'log_save' );\nfunction log_save( $post_id, $post, $update ) {\n    error_log( $update ? 'upd' : 'new' );\n}",
     30, {"wp_api_usage": (2, "`save_post` passes 3 args only if `add_action` declares `$accepted_args = 3`. Registered with the default 1, `$post` and `$update` are null/undefined — the handler misbehaves and may warn."),
          "code_quality": (4, "Signature/registration mismatch.")}),

    # ---- TRUNCATED / INCOMPLETE CONTEXT (auto-FAIL: not evaluable as correct) ----
    ("truncated_switch",
     "function status_label( $s ) {\n    switch ( $s ) {\n        case 'on': return 'Active';\n        case 'off':",
     18, {"code_quality": (1, "The snippet is truncated mid-`case` — no body, no closing of the switch or function. It does not parse and cannot be judged as complete, correct code."),
          "wpcs_compliance": (3, "Incomplete block; nothing to lint meaningfully.")}),
    ("truncated_method_chain",
     "    ->where( 'status', 'publish' )\n        ->orderBy( 'date' )\n        ->get();\n}",
     19, {"code_quality": (1, "A dangling method-chain fragment with no receiver, no opening, and an unmatched closing brace — not valid standalone PHP and impossible to evaluate as correct."),
          "wp_api_usage": (4, "Uses a non-core fluent query builder with no visible WordPress integration.")}),
]


def _prose_line(label, score, reason):
    s = "None" if score is None else str(score)
    return f"{label}: score {s}/10 — {reason}"


def build_assistant(summary, overall, overrides):
    dims = {}
    for d in DIM_ORDER:
        dims[d] = overrides[d] if d in overrides else DEFAULTS[d]
    # enforce max-2-N/A: if >2 None, demote extras to a low real score
    nones = [d for d in DIM_ORDER if dims[d][0] is None]
    for d in nones[2:]:
        dims[d] = (3, dims[d][1].replace("N/A applies", "scored low as inapplicable"))
    lines = [summary, ""]
    for d in DIM_ORDER:
        score, reason = dims[d]
        lines.append(_prose_line(DIM_LABELS[d], score, reason))
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n\n[/REASONING]\n\n"
    jo = {"verdict": "FAIL"}
    for d in DIM_ORDER:
        if dims[d][0] is not None:
            jo[d] = dims[d][0]
    jo["overall_score"] = overall
    body += "<judge_output>\n" + json.dumps(jo, indent=2) + "\n</judge_output>"
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reasoning_dataset/negatives_train.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for cat, code, overall, overrides in CASES:
            assert overall < 70, f"{cat}: overall must be FAIL (<70)"
            summary = SUMMARIES.get(cat, f"This snippet exhibits a critical defect ({cat}) that makes it unsafe or non-functional as WordPress code.")
            prompt = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
            assistant = build_assistant(summary, overall, overrides)
            row = {
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": assistant}],
                "metadata": {"stream": "cot", "format": "cot", "source_dir": "negatives_corrective",
                             "should_fail": True, "defect_category": cat,
                             "dimensions_addressed": DIM_ORDER},
            }
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} should_fail training negatives -> {args.out}")
    return 0


# Per-case opening summary lines (1-2 sentences naming the defect), teacher-style.
SUMMARIES = {
    "syntax_missing_close_brace": "lp_register_routes() is missing its closing brace and references an undefined $routes variable, so the file does not parse.",
    "syntax_missing_semicolon2": "get_site_slug() omits the semicolon after the sanitize_title() assignment, producing a fatal parse error before the return.",
    "syntax_unterminated_heredoc": "render_box() opens a heredoc that is never terminated, so the parser consumes the rest of the function as string content — a fatal error.",
    "syntax_mismatched_brackets": "The body_class closure is missing its closing brace before );, leaving brackets unbalanced and the file unparseable.",
    "syntax_bad_cast_operator": "to_int() has a stray extra parenthesis after the (integer) cast, which is a hard syntax error.",
    "syntax_double_arrow_in_call": "build_query() passes a WP_Query args array containing a doubled => token, which fails to parse.",
    "syntax_unclosed_comment": "init_plugin() opens a block comment that is never closed, commenting out the remainder of the function and breaking the file.",
    "syntax_missing_function_keyword": "register_widget() is declared without the function keyword, so it is not a valid function definition.",
    "fabricated_get_current_theme_meta": "theme_tagline() calls wp_get_current_theme_meta(), which is not a real WordPress function and fatals at runtime.",
    "fabricated_register_admin_widget": "add_dashboard() calls the non-existent register_admin_dashboard_widget(); the real API is wp_add_dashboard_widget().",
    "fabricated_sanitize_html_deep": "clean_input() relies on wp_sanitize_html_deep(), a fabricated function, so no sanitization actually happens and the call fatals.",
    "fabricated_get_user_capability": "can_edit() calls wp_get_user_capability(), which does not exist; capability checks use user_can()/current_user_can().",
    "fabricated_enqueue_inline": "add_css() calls wp_enqueue_inline_style(), which is not a core function; inline CSS uses wp_add_inline_style().",
    "fabricated_query_meta_like": "find_by_meta() calls the invented wp_query_posts_by_meta_like(); meta LIKE queries use WP_Query meta_query.",
    "this_in_function_render": "render_template() uses $this inside a plain function with no class, which fatals with 'Using $this when not in object context'.",
    "self_const_outside_class": "api_base() references self::API_ROOT outside any class scope, a fatal scope error.",
    "parent_call_outside_class": "setup() calls parent::__construct() in a free function with no parent class — a fatal scope error.",
    "sqli_orderby_concat": "list_items() concatenates $_GET['sort'] into an ORDER BY clause, a SQL injection with no allowlist or prepare.",
    "sqli_in_clause": "posts_in() interpolates $_POST['ids'] into an IN() list unprepared — SQL injection.",
    "sqli_like_search": "search_terms() concatenates $_GET['q'] into a LIKE pattern with no prepare or esc_like — injectable.",
    "xss_admin_notice": "notice() echoes $_GET['msg'] into HTML with no escaping — reflected XSS.",
    "xss_shortcode_attr": "sc_handler() places a shortcode attribute into an HTML attribute unescaped — XSS via esc_attr omission.",
    "xss_setting_field": "field() prints a stored option into a value attribute without esc_attr() — attribute-breakout XSS.",
    "no_nonce_form_handler": "The admin_post handler writes an option from $_POST with no nonce or capability check — CSRF and missing authorization.",
    "no_cap_ajax_delete": "The AJAX handler force-deletes a post with no nonce or capability check — any logged-in user can delete arbitrary posts.",
    "infinite_recursion_filter": "The the_title callback re-applies the_title on itself, causing unbounded recursion and a stack overflow.",
    "undefined_var_use": "tax_links() loops over an undefined $terms variable that is never fetched, so the function is non-functional.",
    "wrong_callback_arity": "log_save() expects three save_post arguments but is registered with the default arg count, so $post and $update are missing.",
    "truncated_switch": "status_label() is truncated mid-case with no closing of the switch or function — it does not parse.",
    "truncated_method_chain": "This is a dangling method-chain fragment with no receiver and an unmatched brace — not valid standalone PHP.",
}


if __name__ == "__main__":
    raise SystemExit(main())
