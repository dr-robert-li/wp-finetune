#!/usr/bin/env python
"""P4 REVL-05 invalid-PHP tail sentinel (REOPEN-PLAN Step 7).

REVL-05 was rejected partly because the judge PASSED syntactically-invalid PHP. This
builds ~24 `should_fail:true` rows — genuine syntax errors (the critical) plus clear
semantic defects (XSS, SQLi, out-of-class `$this->`, fabricated wp_* calls, truncated
context) — each wrapped in the exact `<wp_judge>` prompt format the model judges.

Gate (checker, separate): ZERO false-passes — the model-judge must NOT emit verdict
PASS (nor a high overall_score) on any should_fail row.

Usage: python scripts/build_invalid_php_sentinel.py --out data/reasoning_dataset/invalid_php_sentinel.jsonl
"""
import argparse
import json

# (defect_category, php_code). `should_fail:true` for every row.
CASES = [
    # --- SYNTAX ERRORS (the REVL-05 critical: judge must not PASS invalid PHP) ---
    ("syntax_missing_brace",
     "function wp_get_total( $items ) {\n    $sum = 0;\n    foreach ( $items as $i ) {\n        $sum += $i;\n    return $sum;\n}"),
    ("syntax_missing_semicolon",
     "function register_my_widget() {\n    $title = get_option( 'my_title' )\n    return $title;\n}"),
    ("syntax_unclosed_string",
     "function my_notice() {\n    echo '<div class=\"notice\">Hello;\n}"),
    ("syntax_unclosed_paren",
     "add_action( 'init', function() {\n    register_post_type( 'book', array( 'public' => true );\n} );"),
    ("syntax_stray_token",
     "function my_calc( $a, $b ) {\n    return $a +* $b;\n}"),
    ("syntax_unclosed_array",
     "$args = array(\n    'post_type' => 'page',\n    'posts_per_page' => 10,\nreturn new WP_Query( $args );"),
    ("syntax_bad_arrow",
     "function my_map( $arr ) {\n    return array_map( fn($x) -> $x * 2, $arr );\n}"),
    ("syntax_double_semicolon_func",
     "function;; my_init() {\n    do_action( 'my_init' );\n}"),

    # --- SECURITY: XSS (unescaped output) ---
    ("xss_unescaped_echo",
     "function show_user_name() {\n    echo '<h1>Welcome ' . $_GET['name'] . '</h1>';\n}"),
    ("xss_unescaped_attr",
     "function render_link() {\n    echo '<a href=\"' . $_REQUEST['url'] . '\">Click</a>';\n}"),
    ("xss_printf_raw",
     "function my_title() {\n    printf( '<title>%s</title>', $_GET['t'] );\n}"),

    # --- SECURITY: SQL injection (no prepare) ---
    ("sqli_direct_concat",
     "function get_user_posts() {\n    global $wpdb;\n    $id = $_GET['uid'];\n    return $wpdb->get_results( \"SELECT * FROM {$wpdb->posts} WHERE post_author = $id\" );\n}"),
    ("sqli_query_interp",
     "function delete_meta() {\n    global $wpdb;\n    $wpdb->query( \"DELETE FROM {$wpdb->postmeta} WHERE meta_key = '\" . $_POST['k'] . \"'\" );\n}"),

    # --- SECURITY: missing nonce / capability on state change ---
    ("no_nonce_delete",
     "add_action( 'admin_post_del', function() {\n    wp_delete_post( intval( $_GET['id'] ), true );\n} );"),
    ("no_cap_check_option",
     "function save_settings() {\n    update_option( 'site_mode', $_POST['mode'] );\n}"),

    # --- CORRECTNESS: out-of-class `$this->` in a standalone function ---
    ("this_outside_class",
     "function get_widget_title() {\n    return $this->title;\n}"),
    ("self_outside_class",
     "function build() {\n    return self::$instance->render();\n}"),

    # --- CORRECTNESS: fabricated / non-existent WP API ---
    ("fabricated_wp_fn",
     "function cache_user( $id ) {\n    return wp_store_user_cache_forever( $id, true );\n}"),
    ("fabricated_wp_fn2",
     "function get_theme_color() {\n    return wp_get_active_theme_primary_color();\n}"),

    # --- CORRECTNESS: truncated snippet, missing context ---
    ("truncated_dangling",
     "function process_order( $order_id ) {\n    $order = wc_get_order( $order_id );\n    foreach ( $order->get_items() as $item ) {\n        $total += $item->get_total();\n        // ... handling continues"),
    ("truncated_undeclared_var",
     "    $result[] = sanitize_text_field( $row->value );\n    }\n    return $result;\n}"),

    # --- CORRECTNESS: misc clearly-broken ---
    ("undefined_constant_call",
     "function boot() {\n    if ( MY_PLUGIN_DEBUG ) {\n        error_log( 'debug' );\n    }\n}"),
    ("wrong_hook_signature",
     "add_filter( 'the_content', function() {\n    return strtoupper( get_the_content() );\n} );"),
    ("infinite_recursion",
     "function get_count() {\n    return get_count() + 1;\n}"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reasoning_dataset/invalid_php_sentinel.jsonl")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for cat, code in CASES:
            prompt = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
            row = {
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {"should_fail": True, "defect_category": cat, "stream": "sentinel"},
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(CASES)} should_fail sentinel rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
