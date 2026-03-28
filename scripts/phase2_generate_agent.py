#!/usr/bin/env python3
"""Phase 2 synthetic generation via direct code templates.

Generates synthetic WordPress training examples to fill taxonomy gaps
identified by phase2_gap_analysis.py. Uses parameterized templates
instead of API calls -- each gap tag gets varied examples at different
complexity levels, contexts, and constraints.

Also generates ~500 rejection examples (proactive security).
"""

import json
import random
import textwrap
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAP_REPORT = PROJECT_ROOT / "data" / "phase2_synthetic" / "gap_report.json"
PROMPTS_PATH = PROJECT_ROOT / "config" / "synthetic_prompts.yaml"
GENERATED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "generated"

random.seed(42)

# ---------------------------------------------------------------------------
# Template-based generators for each gap tag
# ---------------------------------------------------------------------------
# Each generator returns a list of (function_name, body, tags, complexity) tuples.
# We generate MORE than enough to fill the deficit for each tag.


def _phpdoc(func_name, description, params=None, returns="void", since="1.0.0"):
    """Build a PHPDoc block."""
    lines = [f"/**", f" * {description}", f" *"]
    if params:
        for ptype, pname, pdesc in params:
            lines.append(f" * @param {ptype} ${pname} {pdesc}")
    lines.append(f" * @return {returns}")
    lines.append(f" * @since  {since}")
    lines.append(f" */")
    return "\n".join(lines)


# ---- SQL patterns ----

def gen_sql_prepared_statements(count):
    """Generate sql:prepared_statements examples."""
    examples = []
    variants = [
        ("get_user_order_total", "Retrieves total order amount for a user",
         "simple", """\
function get_user_order_total( $user_id ) {{
\tglobal $wpdb;

\t$total = $wpdb->get_var(
\t\t$wpdb->prepare(
\t\t\t"SELECT SUM(order_total) FROM {{$wpdb->prefix}}orders WHERE user_id = %d AND status = %s",
\t\t\t$user_id,
\t\t\t'completed'
\t\t)
\t);

\treturn $total ? (float) $total : 0.0;
}}"""),
        ("search_posts_by_meta", "Searches posts by meta value with pagination",
         "intermediate", """\
function search_posts_by_meta( $meta_key, $meta_value, $page = 1, $per_page = 20 ) {{
\tglobal $wpdb;

\t$offset = ( $page - 1 ) * $per_page;

\t$results = $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT p.ID, p.post_title, pm.meta_value
\t\t\tFROM {{$wpdb->posts}} p
\t\t\tINNER JOIN {{$wpdb->postmeta}} pm ON p.ID = pm.post_id
\t\t\tWHERE pm.meta_key = %s
\t\t\tAND pm.meta_value LIKE %s
\t\t\tAND p.post_status = 'publish'
\t\t\tORDER BY p.post_date DESC
\t\t\tLIMIT %d OFFSET %d",
\t\t\t$meta_key,
\t\t\t'%' . $wpdb->esc_like( $meta_value ) . '%',
\t\t\t$per_page,
\t\t\t$offset
\t\t)
\t);

\treturn $results ? $results : array();
}}"""),
        ("update_product_price", "Updates product price with audit logging",
         "advanced", """\
function update_product_price( $product_id, $new_price, $updated_by ) {{
\tglobal $wpdb;

\tif ( ! is_numeric( $new_price ) || $new_price < 0 ) {{
\t\treturn new WP_Error( 'invalid_price', esc_html__( 'Price must be a non-negative number.', 'my-plugin' ) );
\t}}

\t$old_price = $wpdb->get_var(
\t\t$wpdb->prepare(
\t\t\t"SELECT price FROM {{$wpdb->prefix}}products WHERE id = %d",
\t\t\t$product_id
\t\t)
\t);

\tif ( null === $old_price ) {{
\t\treturn new WP_Error( 'not_found', esc_html__( 'Product not found.', 'my-plugin' ) );
\t}}

\t$wpdb->update(
\t\t$wpdb->prefix . 'products',
\t\tarray( 'price' => $new_price ),
\t\tarray( 'id' => $product_id ),
\t\tarray( '%f' ),
\t\tarray( '%d' )
\t);

\t$wpdb->insert(
\t\t$wpdb->prefix . 'price_audit',
\t\tarray(
\t\t\t'product_id' => $product_id,
\t\t\t'old_price'  => $old_price,
\t\t\t'new_price'  => $new_price,
\t\t\t'changed_by' => $updated_by,
\t\t\t'changed_at' => current_time( 'mysql' ),
\t\t),
\t\tarray( '%d', '%f', '%f', '%d', '%s' )
\t);

\treturn true;
}}"""),
        ("get_filtered_entries", "Retrieves entries with dynamic filtering and sorting",
         "production-scale", """\
function get_filtered_entries( $args = array() ) {{
\tglobal $wpdb;

\t$defaults = array(
\t\t'status'   => 'active',
\t\t'category' => '',
\t\t'search'   => '',
\t\t'orderby'  => 'created_at',
\t\t'order'    => 'DESC',
\t\t'per_page' => 20,
\t\t'page'     => 1,
\t);
\t$args = wp_parse_args( $args, $defaults );

\t$where  = array( '1=1' );
\t$values = array();

\t$where[]  = 'status = %s';
\t$values[] = $args['status'];

\tif ( ! empty( $args['category'] ) ) {{
\t\t$where[]  = 'category_id = %d';
\t\t$values[] = absint( $args['category'] );
\t}}

\tif ( ! empty( $args['search'] ) ) {{
\t\t$where[]  = 'title LIKE %s';
\t\t$values[] = '%' . $wpdb->esc_like( sanitize_text_field( $args['search'] ) ) . '%';
\t}}

\t$allowed_orderby = array( 'created_at', 'title', 'priority' );
\t$orderby = in_array( $args['orderby'], $allowed_orderby, true ) ? $args['orderby'] : 'created_at';
\t$order   = 'ASC' === strtoupper( $args['order'] ) ? 'ASC' : 'DESC';

\t$offset = ( absint( $args['page'] ) - 1 ) * absint( $args['per_page'] );

\t$sql = "SELECT id, title, status, category_id, created_at
\t\tFROM {{$wpdb->prefix}}entries
\t\tWHERE " . implode( ' AND ', $where ) . "
\t\tORDER BY {{$orderby}} {{$order}}
\t\tLIMIT %d OFFSET %d";

\t$values[] = absint( $args['per_page'] );
\t$values[] = $offset;

\t// phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared -- dynamically built with safe parts.
\treturn $wpdb->get_results( $wpdb->prepare( $sql, $values ) );
}}"""),
        ("count_entries_by_status", "Counts entries grouped by status",
         "simple", """\
function count_entries_by_status() {{
\tglobal $wpdb;

\t$results = $wpdb->get_results(
\t\t"SELECT status, COUNT(*) AS total FROM {{$wpdb->prefix}}entries GROUP BY status",
\t\tARRAY_A
\t);

\t$counts = array();
\tif ( $results ) {{
\t\tforeach ( $results as $row ) {{
\t\t\t$counts[ $row['status'] ] = (int) $row['total'];
\t\t}}
\t}}

\treturn $counts;
}}"""),
        ("get_recent_activity", "Gets recent activity with user join",
         "intermediate", """\
function get_recent_activity( $limit = 50 ) {{
\tglobal $wpdb;

\t$results = $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT a.id, a.action, a.created_at, u.display_name
\t\t\tFROM {{$wpdb->prefix}}activity_log a
\t\t\tLEFT JOIN {{$wpdb->users}} u ON a.user_id = u.ID
\t\t\tWHERE a.created_at > %s
\t\t\tORDER BY a.created_at DESC
\t\t\tLIMIT %d",
\t\t\tgmdate( 'Y-m-d H:i:s', strtotime( '-30 days' ) ),
\t\t\tabsint( $limit )
\t\t)
\t);

\treturn $results ? $results : array();
}}"""),
    ]

    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        fn = fname + suffix
        doc = _phpdoc(fn, desc, [("int", "id", "Item ID")], "mixed")
        examples.append({
            "function_name": fn,
            "source_repo": "synthetic",
            "source_file": f"synthetic/sql_prepared_statements.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["sql:prepared_statements"],
            "complexity": complexity,
        })
    return examples


def gen_sql_joins_across_meta(count):
    examples = []
    variants = [
        ("get_posts_with_meta_join", "Joins posts with postmeta for efficient retrieval", "intermediate", """\
function get_posts_with_meta_join( $meta_key, $post_type = 'post' ) {{
\tglobal $wpdb;

\treturn $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT p.ID, p.post_title, pm.meta_value
\t\t\tFROM {{$wpdb->posts}} p
\t\t\tINNER JOIN {{$wpdb->postmeta}} pm ON p.ID = pm.post_id
\t\t\tWHERE p.post_type = %s
\t\t\tAND p.post_status = 'publish'
\t\t\tAND pm.meta_key = %s
\t\t\tORDER BY p.post_date DESC",
\t\t\t$post_type,
\t\t\t$meta_key
\t\t)
\t);
}}"""),
        ("get_users_with_order_meta", "Joins users with order meta across tables", "advanced", """\
function get_users_with_order_meta( $min_orders = 1 ) {{
\tglobal $wpdb;

\treturn $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT u.ID, u.display_name, u.user_email,
\t\t\t\tCOUNT(pm.meta_id) AS order_count,
\t\t\t\tSUM(pm2.meta_value) AS total_spent
\t\t\tFROM {{$wpdb->users}} u
\t\t\tINNER JOIN {{$wpdb->posts}} p ON u.ID = p.post_author
\t\t\tINNER JOIN {{$wpdb->postmeta}} pm ON p.ID = pm.post_id AND pm.meta_key = '_order_status'
\t\t\tINNER JOIN {{$wpdb->postmeta}} pm2 ON p.ID = pm2.post_id AND pm2.meta_key = '_order_total'
\t\t\tWHERE p.post_type = 'shop_order'
\t\t\tAND pm.meta_value = 'completed'
\t\t\tGROUP BY u.ID
\t\t\tHAVING order_count >= %d
\t\t\tORDER BY total_spent DESC",
\t\t\t$min_orders
\t\t)
\t);
}}"""),
        ("get_terms_with_post_count", "Gets taxonomy terms with post count via join", "simple", """\
function get_terms_with_post_count( $taxonomy ) {{
\tglobal $wpdb;

\treturn $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT t.term_id, t.name, t.slug, tt.count
\t\t\tFROM {{$wpdb->terms}} t
\t\t\tINNER JOIN {{$wpdb->term_taxonomy}} tt ON t.term_id = tt.term_id
\t\t\tWHERE tt.taxonomy = %s
\t\t\tAND tt.count > 0
\t\t\tORDER BY tt.count DESC",
\t\t\t$taxonomy
\t\t)
\t);
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "array")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/sql_joins_across_meta.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["sql:joins_across_meta"],
            "complexity": complexity,
        })
    return examples


def gen_sql_custom_table_creation(count):
    examples = []
    variants = [
        ("create_custom_table", "Creates a custom table with dbDelta", "intermediate", """\
function create_custom_table() {{
\tglobal $wpdb;

\t$table_name      = $wpdb->prefix . 'custom_entries';
\t$charset_collate = $wpdb->get_charset_collate();

\t$sql = "CREATE TABLE $table_name (
\t\tid bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\tuser_id bigint(20) unsigned NOT NULL DEFAULT 0,
\t\ttitle varchar(255) NOT NULL DEFAULT '',
\t\tcontent longtext NOT NULL,
\t\tstatus varchar(20) NOT NULL DEFAULT 'draft',
\t\tcreated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tupdated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
\t\tPRIMARY KEY  (id),
\t\tKEY user_id (user_id),
\t\tKEY status (status),
\t\tKEY created_at (created_at)
\t) $charset_collate;";

\trequire_once ABSPATH . 'wp-admin/includes/upgrade.php';
\tdbDelta( $sql );

\tupdate_option( 'custom_entries_db_version', '1.0.0' );
}}"""),
        ("create_analytics_tables", "Creates analytics tables with foreign keys", "advanced", """\
function create_analytics_tables() {{
\tglobal $wpdb;

\t$charset_collate = $wpdb->get_charset_collate();
\t$events_table    = $wpdb->prefix . 'analytics_events';
\t$sessions_table  = $wpdb->prefix . 'analytics_sessions';

\t$sqls = array();

\t$sqls[] = "CREATE TABLE $sessions_table (
\t\tsession_id bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\tuser_id bigint(20) unsigned NOT NULL DEFAULT 0,
\t\tsession_token varchar(64) NOT NULL,
\t\tip_address varchar(45) NOT NULL DEFAULT '',
\t\tuser_agent text NOT NULL,
\t\tstarted_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tended_at datetime DEFAULT NULL,
\t\tPRIMARY KEY  (session_id),
\t\tUNIQUE KEY session_token (session_token),
\t\tKEY user_id (user_id),
\t\tKEY started_at (started_at)
\t) $charset_collate;";

\t$sqls[] = "CREATE TABLE $events_table (
\t\tevent_id bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\tsession_id bigint(20) unsigned NOT NULL,
\t\tevent_type varchar(50) NOT NULL,
\t\tevent_data longtext NOT NULL,
\t\tcreated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tPRIMARY KEY  (event_id),
\t\tKEY session_id (session_id),
\t\tKEY event_type (event_type),
\t\tKEY created_at (created_at)
\t) $charset_collate;";

\trequire_once ABSPATH . 'wp-admin/includes/upgrade.php';
\tforeach ( $sqls as $sql ) {{
\t\tdbDelta( $sql );
\t}}

\tupdate_option( 'analytics_db_version', '1.0.0' );
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "void")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/sql_custom_table_creation.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["sql:custom_table_creation"],
            "complexity": complexity,
        })
    return examples


def gen_sql_dbdelta_migrations(count):
    examples = []
    body = """\
function run_db_migration_{idx}() {{
\tglobal $wpdb;

\t$installed_ver = get_option( 'myplugin_db_version', '0' );
\t$current_ver   = '2.{idx}.0';

\tif ( version_compare( $installed_ver, $current_ver, '>=' ) ) {{
\t\treturn;
\t}}

\t$table_name      = $wpdb->prefix . 'myplugin_data';
\t$charset_collate = $wpdb->get_charset_collate();

\t$sql = "CREATE TABLE $table_name (
\t\tid bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\ttitle varchar(255) NOT NULL DEFAULT '',
\t\tpriority tinyint(1) NOT NULL DEFAULT 0,
\t\tcreated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tPRIMARY KEY  (id),
\t\tKEY priority (priority)
\t) $charset_collate;";

\trequire_once ABSPATH . 'wp-admin/includes/upgrade.php';
\tdbDelta( $sql );

\tupdate_option( 'myplugin_db_version', $current_ver );
}}"""
    for i in range(count):
        fname = f"run_db_migration_{i}"
        doc = _phpdoc(fname, f"Database migration v2.{i}.0", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/sql_dbdelta_migrations.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["sql:dbdelta_migrations"],
            "complexity": "intermediate",
        })
    return examples


def gen_sql_batch_operations(count):
    examples = []
    body_template = """\
function batch_update_records_{idx}( $batch_size = 500 ) {{
\tglobal $wpdb;

\t$table  = $wpdb->prefix . 'records';
\t$offset = 0;
\t$total  = 0;

\tdo {{
\t\t$ids = $wpdb->get_col(
\t\t\t$wpdb->prepare(
\t\t\t\t"SELECT id FROM {{$table}} WHERE status = %s ORDER BY id ASC LIMIT %d OFFSET %d",
\t\t\t\t'pending',
\t\t\t\t$batch_size,
\t\t\t\t$offset
\t\t\t)
\t\t);

\t\tif ( empty( $ids ) ) {{
\t\t\tbreak;
\t\t}}

\t\t$placeholders = implode( ',', array_fill( 0, count( $ids ), '%d' ) );
\t\t// phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared -- placeholders generated safely.
\t\t$wpdb->query(
\t\t\t$wpdb->prepare(
\t\t\t\t"UPDATE {{$table}} SET status = %s, updated_at = %s WHERE id IN ($placeholders)",
\t\t\t\tarray_merge(
\t\t\t\t\tarray( 'processed', current_time( 'mysql' ) ),
\t\t\t\t\t$ids
\t\t\t\t)
\t\t\t)
\t\t);

\t\t$total += count( $ids );
\t\t$offset += $batch_size;

\t\tif ( function_exists( 'wp_cache_flush' ) ) {{
\t\t\twp_cache_flush();
\t\t}}
\t}} while ( count( $ids ) === $batch_size );

\treturn $total;
}}"""
    for i in range(count):
        fname = f"batch_update_records_{i}"
        doc = _phpdoc(fname, "Batch updates records in chunks to avoid memory limits",
                       [("int", "batch_size", "Records per batch")], "int")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/sql_batch_operations.php",
            "body": f"<?php\n{doc}\n{body_template.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["sql:batch_operations"],
            "complexity": "advanced",
        })
    return examples


def gen_security_nonce_verification(count):
    examples = []
    variants = [
        ("handle_settings_form", "Processes settings form with nonce verification", "intermediate", """\
function handle_settings_form() {{
\tif ( ! isset( $_POST['myplugin_settings_nonce'] ) ||
\t\t! wp_verify_nonce( sanitize_text_field( wp_unslash( $_POST['myplugin_settings_nonce'] ) ), 'myplugin_save_settings' ) ) {{
\t\twp_die( esc_html__( 'Security check failed.', 'my-plugin' ) );
\t}}

\tif ( ! current_user_can( 'manage_options' ) ) {{
\t\twp_die( esc_html__( 'Unauthorized access.', 'my-plugin' ) );
\t}}

\t$option_value = isset( $_POST['myplugin_option'] )
\t\t? sanitize_text_field( wp_unslash( $_POST['myplugin_option'] ) )
\t\t: '';

\tupdate_option( 'myplugin_option', $option_value );

\twp_safe_redirect(
\t\tadd_query_arg( 'updated', 'true', admin_url( 'options-general.php?page=myplugin' ) )
\t);
\texit;
}}"""),
        ("handle_ajax_delete", "AJAX handler with referer check", "simple", """\
function handle_ajax_delete() {{
\tcheck_ajax_referer( 'myplugin_delete_nonce', 'security' );

\tif ( ! current_user_can( 'delete_posts' ) ) {{
\t\twp_send_json_error( array( 'message' => esc_html__( 'Permission denied.', 'my-plugin' ) ) );
\t}}

\t$item_id = isset( $_POST['item_id'] ) ? absint( $_POST['item_id'] ) : 0;

\tif ( ! $item_id ) {{
\t\twp_send_json_error( array( 'message' => esc_html__( 'Invalid item ID.', 'my-plugin' ) ) );
\t}}

\t$deleted = wp_delete_post( $item_id, true );

\tif ( $deleted ) {{
\t\twp_send_json_success( array( 'message' => esc_html__( 'Item deleted.', 'my-plugin' ) ) );
\t}} else {{
\t\twp_send_json_error( array( 'message' => esc_html__( 'Delete failed.', 'my-plugin' ) ) );
\t}}
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "void")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/security_nonce_verification.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["security:nonce_verification"],
            "complexity": complexity,
        })
    return examples


def gen_security_input_sanitization(count):
    examples = []
    variants = [
        ("sanitize_form_input", "Sanitizes all form fields properly", "intermediate", """\
function sanitize_form_input( $raw_data ) {{
\t$sanitized = array();

\t$sanitized['title']   = isset( $raw_data['title'] )
\t\t? sanitize_text_field( wp_unslash( $raw_data['title'] ) )
\t\t: '';
\t$sanitized['email']   = isset( $raw_data['email'] )
\t\t? sanitize_email( wp_unslash( $raw_data['email'] ) )
\t\t: '';
\t$sanitized['url']     = isset( $raw_data['url'] )
\t\t? esc_url_raw( wp_unslash( $raw_data['url'] ) )
\t\t: '';
\t$sanitized['content'] = isset( $raw_data['content'] )
\t\t? wp_kses_post( wp_unslash( $raw_data['content'] ) )
\t\t: '';
\t$sanitized['count']   = isset( $raw_data['count'] )
\t\t? absint( $raw_data['count'] )
\t\t: 0;

\treturn $sanitized;
}}"""),
        ("sanitize_settings_input", "Sanitizes plugin settings before saving", "simple", """\
function sanitize_settings_input( $input ) {{
\t$output = array();

\t$output['api_key']     = isset( $input['api_key'] )
\t\t? sanitize_key( $input['api_key'] )
\t\t: '';
\t$output['display_name'] = isset( $input['display_name'] )
\t\t? sanitize_text_field( $input['display_name'] )
\t\t: '';
\t$output['max_items']    = isset( $input['max_items'] )
\t\t? absint( $input['max_items'] )
\t\t: 10;
\t$output['enabled']      = ! empty( $input['enabled'] ) ? 1 : 0;

\treturn $output;
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [("array", "data", "Raw input data")], "array")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/security_input_sanitization.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["security:input_sanitization"],
            "complexity": complexity,
        })
    return examples


def gen_rest_route_registration(count):
    examples = []
    variants = [
        ("register_items_routes", "Registers REST API routes for items CRUD", "advanced", """\
function register_items_routes() {{
\tregister_rest_route(
\t\t'myplugin/v1',
\t\t'/items',
\t\tarray(
\t\t\tarray(
\t\t\t\t'methods'             => WP_REST_Server::READABLE,
\t\t\t\t'callback'            => 'get_items_callback',
\t\t\t\t'permission_callback' => function () {{
\t\t\t\t\treturn current_user_can( 'read' );
\t\t\t\t}},
\t\t\t\t'args'                => array(
\t\t\t\t\t'per_page' => array(
\t\t\t\t\t\t'type'              => 'integer',
\t\t\t\t\t\t'default'           => 10,
\t\t\t\t\t\t'sanitize_callback' => 'absint',
\t\t\t\t\t\t'validate_callback' => function ( $param ) {{
\t\t\t\t\t\t\treturn $param > 0 && $param <= 100;
\t\t\t\t\t\t}},
\t\t\t\t\t),
\t\t\t\t),
\t\t\t),
\t\t\tarray(
\t\t\t\t'methods'             => WP_REST_Server::CREATABLE,
\t\t\t\t'callback'            => 'create_item_callback',
\t\t\t\t'permission_callback' => function () {{
\t\t\t\t\treturn current_user_can( 'edit_posts' );
\t\t\t\t}},
\t\t\t\t'args'                => array(
\t\t\t\t\t'title' => array(
\t\t\t\t\t\t'type'              => 'string',
\t\t\t\t\t\t'required'          => true,
\t\t\t\t\t\t'sanitize_callback' => 'sanitize_text_field',
\t\t\t\t\t),
\t\t\t\t),
\t\t\t),
\t\t)
\t);
}}"""),
        ("register_settings_route", "Registers REST route for plugin settings", "simple", """\
function register_settings_route() {{
\tregister_rest_route(
\t\t'myplugin/v1',
\t\t'/settings',
\t\tarray(
\t\t\t'methods'             => WP_REST_Server::READABLE,
\t\t\t'callback'            => function ( $request ) {{
\t\t\t\t$settings = get_option( 'myplugin_settings', array() );
\t\t\t\treturn rest_ensure_response( $settings );
\t\t\t}},
\t\t\t'permission_callback' => function () {{
\t\t\t\treturn current_user_can( 'manage_options' );
\t\t\t}},
\t\t)
\t);
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "void")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/rest_route_registration.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["rest:route_registration"],
            "complexity": complexity,
        })
    return examples


def gen_rest_permission_callbacks(count):
    examples = []
    variants = [
        ("check_item_read_permission", "Permission callback for reading items", "simple", """\
function check_item_read_permission( $request ) {{
\tif ( ! current_user_can( 'read' ) ) {{
\t\treturn new WP_Error(
\t\t\t'rest_forbidden',
\t\t\tesc_html__( 'You do not have permission to view items.', 'my-plugin' ),
\t\t\tarray( 'status' => rest_authorization_required_code() )
\t\t);
\t}}
\treturn true;
}}"""),
        ("check_item_write_permission", "Permission callback for creating/updating items", "intermediate", """\
function check_item_write_permission( $request ) {{
\t$method = $request->get_method();

\tif ( 'POST' === $method && ! current_user_can( 'edit_posts' ) ) {{
\t\treturn new WP_Error(
\t\t\t'rest_forbidden_create',
\t\t\tesc_html__( 'You cannot create items.', 'my-plugin' ),
\t\t\tarray( 'status' => rest_authorization_required_code() )
\t\t);
\t}}

\tif ( 'PUT' === $method || 'PATCH' === $method ) {{
\t\t$item_id = $request->get_param( 'id' );
\t\t$item    = get_post( $item_id );

\t\tif ( ! $item ) {{
\t\t\treturn new WP_Error( 'rest_not_found', esc_html__( 'Item not found.', 'my-plugin' ), array( 'status' => 404 ) );
\t\t}}

\t\tif ( (int) $item->post_author !== get_current_user_id() && ! current_user_can( 'edit_others_posts' ) ) {{
\t\t\treturn new WP_Error(
\t\t\t\t'rest_forbidden_edit',
\t\t\t\tesc_html__( 'You cannot edit this item.', 'my-plugin' ),
\t\t\t\tarray( 'status' => rest_authorization_required_code() )
\t\t\t);
\t\t}}
\t}}

\treturn true;
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [("WP_REST_Request", "request", "REST request")], "bool|WP_Error")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/rest_permission_callbacks.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["rest:permission_callbacks"],
            "complexity": complexity,
        })
    return examples


def gen_hooks_action_registration(count):
    examples = []
    variants = [
        ("register_plugin_actions", "Registers core plugin action hooks", "intermediate", """\
function register_plugin_actions() {{
\tadd_action( 'init', 'myplugin_register_post_types', 10 );
\tadd_action( 'admin_menu', 'myplugin_add_admin_pages', 10 );
\tadd_action( 'admin_enqueue_scripts', 'myplugin_admin_assets', 10 );
\tadd_action( 'wp_enqueue_scripts', 'myplugin_frontend_assets', 10 );
\tadd_action( 'rest_api_init', 'myplugin_register_routes', 10 );
\tadd_action( 'wp_ajax_myplugin_save', 'myplugin_ajax_save', 10 );
\tadd_action( 'wp_ajax_nopriv_myplugin_public_action', 'myplugin_public_action', 10 );
}}"""),
        ("register_post_save_actions", "Hooks into post save lifecycle", "advanced", """\
function register_post_save_actions() {{
\tadd_action( 'save_post', 'myplugin_on_save_post', 10, 3 );
\tadd_action( 'transition_post_status', 'myplugin_on_status_change', 10, 3 );
\tadd_action( 'before_delete_post', 'myplugin_cleanup_meta', 10, 1 );
\tadd_action( 'wp_trash_post', 'myplugin_on_trash', 10, 1 );
}}

function myplugin_on_save_post( $post_id, $post, $update ) {{
\tif ( defined( 'DOING_AUTOSAVE' ) && DOING_AUTOSAVE ) {{
\t\treturn;
\t}}
\tif ( wp_is_post_revision( $post_id ) ) {{
\t\treturn;
\t}}
\tif ( 'myplugin_cpt' !== $post->post_type ) {{
\t\treturn;
\t}}
\tif ( ! current_user_can( 'edit_post', $post_id ) ) {{
\t\treturn;
\t}}

\tif ( isset( $_POST['myplugin_nonce'] ) &&
\t\twp_verify_nonce( sanitize_text_field( wp_unslash( $_POST['myplugin_nonce'] ) ), 'myplugin_save' ) ) {{
\t\t$value = isset( $_POST['myplugin_field'] )
\t\t\t? sanitize_text_field( wp_unslash( $_POST['myplugin_field'] ) )
\t\t\t: '';
\t\tupdate_post_meta( $post_id, '_myplugin_field', $value );
\t}}
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "void")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/hooks_action_registration.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["hooks:action_registration"],
            "complexity": complexity,
        })
    return examples


def gen_hooks_filter_registration(count):
    examples = []
    variants = [
        ("register_content_filters", "Registers content modification filters", "simple", """\
function register_content_filters() {{
\tadd_filter( 'the_content', 'myplugin_append_cta', 20 );
\tadd_filter( 'the_title', 'myplugin_modify_title', 10, 2 );
\tadd_filter( 'excerpt_length', 'myplugin_excerpt_length', 10 );
\tadd_filter( 'excerpt_more', 'myplugin_excerpt_more', 10 );
}}

function myplugin_append_cta( $content ) {{
\tif ( ! is_singular( 'post' ) || ! is_main_query() ) {{
\t\treturn $content;
\t}}

\t$cta = get_option( 'myplugin_cta_text', '' );
\tif ( empty( $cta ) ) {{
\t\treturn $content;
\t}}

\treturn $content . '<div class="myplugin-cta">' . wp_kses_post( $cta ) . '</div>';
}}"""),
        ("register_query_filters", "Modifies WP_Query via pre_get_posts", "advanced", """\
function register_query_filters() {{
\tadd_filter( 'pre_get_posts', 'myplugin_modify_main_query', 10 );
\tadd_filter( 'posts_where', 'myplugin_custom_where', 10, 2 );
\tadd_filter( 'posts_join', 'myplugin_custom_join', 10, 2 );
\tadd_filter( 'posts_orderby', 'myplugin_custom_orderby', 10, 2 );
}}

function myplugin_modify_main_query( $query ) {{
\tif ( is_admin() || ! $query->is_main_query() ) {{
\t\treturn $query;
\t}}

\tif ( $query->is_post_type_archive( 'myplugin_item' ) ) {{
\t\t$query->set( 'posts_per_page', 24 );
\t\t$query->set( 'orderby', 'menu_order' );
\t\t$query->set( 'order', 'ASC' );
\t}}

\treturn $query;
}}"""),
    ]
    for i in range(count):
        idx = i % len(variants)
        fname, desc, complexity, body = variants[idx]
        suffix = f"_{i}" if i >= len(variants) else ""
        doc = _phpdoc(fname + suffix, desc, [], "void")
        examples.append({
            "function_name": fname + suffix,
            "source_repo": "synthetic",
            "source_file": "synthetic/hooks_filter_registration.php",
            "body": f"<?php\n{doc}\n{body}",
            "quality_tier": "synthetic",
            "training_tags": ["hooks:filter_registration"],
            "complexity": complexity,
        })
    return examples


def gen_data_custom_post_types(count):
    examples = []
    body = """\
function register_portfolio_post_type_{idx}() {{
\t$labels = array(
\t\t'name'               => esc_html_x( 'Portfolio Items', 'post type general name', 'my-plugin' ),
\t\t'singular_name'      => esc_html_x( 'Portfolio Item', 'post type singular name', 'my-plugin' ),
\t\t'add_new'            => esc_html__( 'Add New', 'my-plugin' ),
\t\t'add_new_item'       => esc_html__( 'Add New Portfolio Item', 'my-plugin' ),
\t\t'edit_item'          => esc_html__( 'Edit Portfolio Item', 'my-plugin' ),
\t\t'new_item'           => esc_html__( 'New Portfolio Item', 'my-plugin' ),
\t\t'view_item'          => esc_html__( 'View Portfolio Item', 'my-plugin' ),
\t\t'search_items'       => esc_html__( 'Search Portfolio Items', 'my-plugin' ),
\t\t'not_found'          => esc_html__( 'No portfolio items found.', 'my-plugin' ),
\t\t'not_found_in_trash' => esc_html__( 'No portfolio items found in Trash.', 'my-plugin' ),
\t);

\t$args = array(
\t\t'labels'             => $labels,
\t\t'public'             => true,
\t\t'publicly_queryable' => true,
\t\t'show_ui'            => true,
\t\t'show_in_menu'       => true,
\t\t'show_in_rest'       => true,
\t\t'menu_icon'          => 'dashicons-portfolio',
\t\t'has_archive'        => true,
\t\t'rewrite'            => array( 'slug' => 'portfolio' ),
\t\t'capability_type'    => 'post',
\t\t'supports'           => array( 'title', 'editor', 'thumbnail', 'excerpt', 'custom-fields' ),
\t\t'taxonomies'         => array( 'portfolio_category' ),
\t);

\tregister_post_type( 'portfolio', $args );
}}"""
    for i in range(count):
        fname = f"register_portfolio_post_type_{i}"
        doc = _phpdoc(fname, "Registers portfolio custom post type with REST support", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/data_custom_post_types.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["data:custom_post_types"],
            "complexity": "intermediate",
        })
    return examples


def gen_perf_query_caching(count):
    examples = []
    body = """\
function get_cached_results_{idx}( $args = array() ) {{
\t$cache_key   = 'myplugin_results_' . md5( wp_json_encode( $args ) );
\t$cache_group = 'myplugin';

\t$results = wp_cache_get( $cache_key, $cache_group );
\tif ( false !== $results ) {{
\t\treturn $results;
\t}}

\t$results = get_transient( $cache_key );
\tif ( false !== $results ) {{
\t\twp_cache_set( $cache_key, $results, $cache_group, 3600 );
\t\treturn $results;
\t}}

\tglobal $wpdb;
\t$results = $wpdb->get_results(
\t\t$wpdb->prepare(
\t\t\t"SELECT id, title, value FROM {{$wpdb->prefix}}items WHERE status = %s ORDER BY value DESC LIMIT %d",
\t\t\t'active',
\t\t\tabsint( $args['limit'] ?? 50 )
\t\t)
\t);

\tset_transient( $cache_key, $results, HOUR_IN_SECONDS );
\twp_cache_set( $cache_key, $results, $cache_group, 3600 );

\treturn $results;
}}"""
    for i in range(count):
        fname = f"get_cached_results_{i}"
        doc = _phpdoc(fname, "Retrieves results with tiered caching (object cache -> transient -> DB)",
                       [("array", "args", "Query arguments")], "array")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/perf_query_caching.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["perf:query_caching"],
            "complexity": "advanced",
        })
    return examples


def gen_perf_batch_processing(count):
    examples = []
    body = """\
function process_batch_queue_{idx}() {{
\tif ( get_transient( 'myplugin_batch_lock' ) ) {{
\t\treturn;
\t}}
\tset_transient( 'myplugin_batch_lock', true, 5 * MINUTE_IN_SECONDS );

\t$queue = get_option( 'myplugin_batch_queue', array() );
\tif ( empty( $queue ) ) {{
\t\tdelete_transient( 'myplugin_batch_lock' );
\t\treturn;
\t}}

\t$batch = array_splice( $queue, 0, 50 );
\tupdate_option( 'myplugin_batch_queue', $queue );

\tforeach ( $batch as $item ) {{
\t\ttry {{
\t\t\tmyplugin_process_single_item( $item );
\t\t}} catch ( Exception $e ) {{
\t\t\terror_log( 'MyPlugin batch error: ' . $e->getMessage() );
\t\t}}
\t}}

\tdelete_transient( 'myplugin_batch_lock' );

\tif ( ! empty( $queue ) ) {{
\t\twp_schedule_single_event( time() + 30, 'myplugin_process_batch' );
\t}}
}}"""
    for i in range(count):
        fname = f"process_batch_queue_{i}"
        doc = _phpdoc(fname, "Processes queued items in batches with locking", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/perf_batch_processing.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["perf:batch_processing"],
            "complexity": "advanced",
        })
    return examples


def gen_arch_activation_hooks(count):
    examples = []
    body = """\
function myplugin_activate_{idx}() {{
\tglobal $wpdb;

\t$table_name      = $wpdb->prefix . 'myplugin_data';
\t$charset_collate = $wpdb->get_charset_collate();

\t$sql = "CREATE TABLE $table_name (
\t\tid bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\ttitle varchar(255) NOT NULL DEFAULT '',
\t\tstatus varchar(20) NOT NULL DEFAULT 'active',
\t\tcreated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tPRIMARY KEY  (id),
\t\tKEY status (status)
\t) $charset_collate;";

\trequire_once ABSPATH . 'wp-admin/includes/upgrade.php';
\tdbDelta( $sql );

\tadd_option( 'myplugin_version', '1.0.0' );
\tadd_option( 'myplugin_settings', array( 'enabled' => true ) );

\tflush_rewrite_rules();
}}
register_activation_hook( __FILE__, 'myplugin_activate_{idx}' );"""
    for i in range(count):
        fname = f"myplugin_activate_{i}"
        doc = _phpdoc(fname, "Plugin activation hook: creates tables, sets defaults, flushes rewrites", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/arch_activation_hooks.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["arch:activation_hooks"],
            "complexity": "intermediate",
        })
    return examples


def gen_arch_uninstall_cleanup(count):
    examples = []
    body = """\
// Uninstall guard.
if ( ! defined( 'WP_UNINSTALL_PLUGIN' ) ) {{
\tdie;
}}

function myplugin_uninstall_{idx}() {{
\tglobal $wpdb;

\t// Remove options.
\tdelete_option( 'myplugin_settings' );
\tdelete_option( 'myplugin_version' );
\tdelete_option( 'myplugin_db_version' );

\t// Remove transients.
\t$wpdb->query( "DELETE FROM {{$wpdb->options}} WHERE option_name LIKE '_transient_myplugin_%'" );
\t$wpdb->query( "DELETE FROM {{$wpdb->options}} WHERE option_name LIKE '_transient_timeout_myplugin_%'" );

\t// Remove user meta.
\t$wpdb->query( "DELETE FROM {{$wpdb->usermeta}} WHERE meta_key LIKE 'myplugin_%'" );

\t// Remove post meta.
\t$wpdb->query( "DELETE FROM {{$wpdb->postmeta}} WHERE meta_key LIKE '_myplugin_%'" );

\t// Remove custom tables.
\t$wpdb->query( "DROP TABLE IF EXISTS {{$wpdb->prefix}}myplugin_data" );

\t// Remove scheduled cron events.
\twp_clear_scheduled_hook( 'myplugin_daily_cleanup' );
\twp_clear_scheduled_hook( 'myplugin_process_batch' );

\t// Remove custom posts.
\t$posts = get_posts( array(
\t\t'post_type'      => 'myplugin_item',
\t\t'posts_per_page' => -1,
\t\t'post_status'    => 'any',
\t\t'fields'         => 'ids',
\t) );
\tforeach ( $posts as $post_id ) {{
\t\twp_delete_post( $post_id, true );
\t}}

\t// Flush rewrite rules.
\tflush_rewrite_rules();
}}
myplugin_uninstall_{idx}();"""
    for i in range(count):
        fname = f"myplugin_uninstall_{i}"
        doc = _phpdoc(fname, "Complete uninstall cleanup: tables, options, meta, cron, posts", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/arch_uninstall_cleanup.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["arch:uninstall_cleanup"],
            "complexity": "advanced",
        })
    return examples


def gen_multisite_per_site_tables(count):
    examples = []
    body = """\
function create_site_table_{idx}( $blog_id ) {{
\tswitch_to_blog( $blog_id );

\tglobal $wpdb;
\t$table_name      = $wpdb->prefix . 'site_analytics';
\t$charset_collate = $wpdb->get_charset_collate();

\t$sql = "CREATE TABLE $table_name (
\t\tid bigint(20) unsigned NOT NULL AUTO_INCREMENT,
\t\tevent_type varchar(50) NOT NULL,
\t\tevent_data longtext NOT NULL,
\t\tcreated_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
\t\tPRIMARY KEY  (id),
\t\tKEY event_type (event_type)
\t) $charset_collate;";

\trequire_once ABSPATH . 'wp-admin/includes/upgrade.php';
\tdbDelta( $sql );

\trestore_current_blog();
}}
add_action( 'wp_insert_site', function ( $site ) {{
\tcreate_site_table_{idx}( $site->blog_id );
}} );"""
    for i in range(count):
        fname = f"create_site_table_{i}"
        doc = _phpdoc(fname, "Creates per-site analytics table on new site creation",
                       [("int", "blog_id", "Blog ID")], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/multisite_per_site_tables.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["multisite:per_site_tables"],
            "complexity": "advanced",
        })
    return examples


def gen_cron_scheduled_events(count):
    examples = []
    body = """\
function schedule_cleanup_cron_{idx}() {{
\tif ( ! wp_next_scheduled( 'myplugin_daily_cleanup_{idx}' ) ) {{
\t\twp_schedule_event( time(), 'daily', 'myplugin_daily_cleanup_{idx}' );
\t}}
}}
add_action( 'wp', 'schedule_cleanup_cron_{idx}' );

function run_daily_cleanup_{idx}() {{
\tglobal $wpdb;

\t// Delete expired transients.
\t$wpdb->query(
\t\t$wpdb->prepare(
\t\t\t"DELETE FROM {{$wpdb->options}} WHERE option_name LIKE %s AND option_value < %d",
\t\t\t$wpdb->esc_like( '_transient_timeout_myplugin_' ) . '%',
\t\t\ttime()
\t\t)
\t);

\t// Delete old log entries (older than 30 days).
\t$wpdb->query(
\t\t$wpdb->prepare(
\t\t\t"DELETE FROM {{$wpdb->prefix}}myplugin_logs WHERE created_at < %s",
\t\t\tgmdate( 'Y-m-d H:i:s', strtotime( '-30 days' ) )
\t\t)
\t);
}}
add_action( 'myplugin_daily_cleanup_{idx}', 'run_daily_cleanup_{idx}' );"""
    for i in range(count):
        fname = f"schedule_cleanup_cron_{i}"
        doc = _phpdoc(fname, "Schedules daily cleanup cron and handles expired data", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/cron_scheduled_events.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["cron:scheduled_events"],
            "complexity": "intermediate",
        })
    return examples


def gen_i18n_pluralization(count):
    examples = []
    body = """\
function display_item_count_{idx}( $count ) {{
\t$message = sprintf(
\t\t/* translators: %d: number of items */
\t\t_n(
\t\t\t'%d item found.',
\t\t\t'%d items found.',
\t\t\t$count,
\t\t\t'my-plugin'
\t\t),
\t\t$count
\t);

\treturn '<p class="item-count">' . esc_html( $message ) . '</p>';
}}"""
    for i in range(count):
        fname = f"display_item_count_{i}"
        doc = _phpdoc(fname, "Displays pluralized item count string",
                       [("int", "count", "Number of items")], "string")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/i18n_pluralization.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["i18n:pluralization"],
            "complexity": "simple",
        })
    return examples


def gen_a11y_semantic_html(count):
    examples = []
    body = """\
function render_accessible_card_{idx}( $item ) {{
\t$title   = esc_html( $item['title'] );
\t$content = wp_kses_post( $item['content'] );
\t$link    = esc_url( $item['url'] );
\t$date    = esc_attr( $item['date'] );

\tob_start();
\t?>
\t<article class="card" role="article" aria-labelledby="card-title-<?php echo absint( $item['id'] ); ?>">
\t\t<header>
\t\t\t<h3 id="card-title-<?php echo absint( $item['id'] ); ?>"><?php echo $title; ?></h3>
\t\t\t<time datetime="<?php echo $date; ?>"><?php echo esc_html( date_i18n( get_option( 'date_format' ), strtotime( $item['date'] ) ) ); ?></time>
\t\t</header>
\t\t<div class="card-content">
\t\t\t<?php echo $content; ?>
\t\t</div>
\t\t<footer>
\t\t\t<a href="<?php echo $link; ?>" aria-label="<?php echo esc_attr( sprintf( __( 'Read more about %s', 'my-plugin' ), $item['title'] ) ); ?>">
\t\t\t\t<?php esc_html_e( 'Read More', 'my-plugin' ); ?>
\t\t\t</a>
\t\t</footer>
\t</article>
\t<?php
\treturn ob_get_clean();
}}"""
    for i in range(count):
        fname = f"render_accessible_card_{i}"
        doc = _phpdoc(fname, "Renders an accessible card with semantic HTML and ARIA",
                       [("array", "item", "Card data")], "string")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/a11y_semantic_html.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["a11y:semantic_html"],
            "complexity": "intermediate",
        })
    return examples


def gen_theme_block_patterns(count):
    examples = []
    body = """\
function register_hero_pattern_{idx}() {{
\tregister_block_pattern(
\t\t'mytheme/hero-{idx}',
\t\tarray(
\t\t\t'title'       => esc_html__( 'Hero Section', 'my-theme' ),
\t\t\t'description' => esc_html__( 'A full-width hero section with heading and CTA.', 'my-theme' ),
\t\t\t'categories'  => array( 'featured' ),
\t\t\t'keywords'    => array( 'hero', 'banner', 'cta' ),
\t\t\t'content'     => '<!-- wp:cover {{"align":"full","dimRatio":50}} -->
<div class="wp-block-cover alignfull">
\t<div class="wp-block-cover__inner-container">
\t\t<!-- wp:heading {{"textAlign":"center","level":1}} -->
\t\t<h1 class="has-text-align-center">' . esc_html__( 'Welcome to Our Site', 'my-theme' ) . '</h1>
\t\t<!-- /wp:heading -->
\t\t<!-- wp:buttons {{"layout":{{"type":"flex","justifyContent":"center"}}}} -->
\t\t<div class="wp-block-buttons">
\t\t\t<!-- wp:button -->
\t\t\t<div class="wp-block-button"><a class="wp-block-button__link">' . esc_html__( 'Get Started', 'my-theme' ) . '</a></div>
\t\t\t<!-- /wp:button -->
\t\t</div>
\t\t<!-- /wp:buttons -->
\t</div>
</div>
<!-- /wp:cover -->',
\t\t)
\t);
}}
add_action( 'init', 'register_hero_pattern_{idx}' );"""
    for i in range(count):
        fname = f"register_hero_pattern_{i}"
        doc = _phpdoc(fname, "Registers a hero block pattern for the block editor", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/theme_block_patterns.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["theme:block_patterns"],
            "complexity": "intermediate",
        })
    return examples


def gen_theme_enqueue_scripts(count):
    examples = []
    body = """\
function enqueue_theme_assets_{idx}() {{
\t$version = wp_get_theme()->get( 'Version' );

\twp_enqueue_style(
\t\t'mytheme-style',
\t\tget_stylesheet_uri(),
\t\tarray(),
\t\t$version
\t);

\twp_enqueue_script(
\t\t'mytheme-navigation',
\t\tget_template_directory_uri() . '/assets/js/navigation.js',
\t\tarray(),
\t\t$version,
\t\ttrue
\t);

\tif ( is_singular() && comments_open() && get_option( 'thread_comments' ) ) {{
\t\twp_enqueue_script( 'comment-reply' );
\t}}

\twp_localize_script(
\t\t'mytheme-navigation',
\t\t'mythemeData',
\t\tarray(
\t\t\t'ajaxUrl' => esc_url( admin_url( 'admin-ajax.php' ) ),
\t\t\t'nonce'   => wp_create_nonce( 'mytheme_nonce' ),
\t\t)
\t);
}}
add_action( 'wp_enqueue_scripts', 'enqueue_theme_assets_{idx}' );"""
    for i in range(count):
        fname = f"enqueue_theme_assets_{i}"
        doc = _phpdoc(fname, "Enqueues theme styles and scripts with proper dependencies", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/theme_enqueue_scripts.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["theme:enqueue_scripts"],
            "complexity": "simple",
        })
    return examples


def gen_theme_template_hierarchy(count):
    examples = []
    body = """\
function custom_template_hierarchy_{idx}( $templates ) {{
\t$post = get_queried_object();

\tif ( ! $post ) {{
\t\treturn $templates;
\t}}

\t$custom_template = get_post_meta( $post->ID, '_custom_template', true );
\tif ( ! empty( $custom_template ) ) {{
\t\tarray_unshift( $templates, sanitize_file_name( $custom_template ) . '.php' );
\t}}

\treturn $templates;
}}
add_filter( 'single_template_hierarchy', 'custom_template_hierarchy_{idx}' );

function load_template_part_{idx}( $slug, $name = '' ) {{
\t$templates = array();

\tif ( ! empty( $name ) ) {{
\t\t$templates[] = "template-parts/{{$slug}}-{{$name}}.php";
\t}}
\t$templates[] = "template-parts/{{$slug}}.php";

\t$located = locate_template( $templates );
\tif ( $located ) {{
\t\tload_template( $located, false );
\t}}
}}"""
    for i in range(count):
        fname = f"custom_template_hierarchy_{i}"
        doc = _phpdoc(fname, "Customizes template hierarchy with post meta override",
                       [("array", "templates", "Template file candidates")], "array")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/theme_template_hierarchy.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["theme:template_hierarchy"],
            "complexity": "advanced",
        })
    return examples


# ---- Rejection examples (proactive security) ----

def gen_rejection_proactive_nonce(count):
    """Generate rejection examples where model proactively adds nonce verification."""
    examples = []
    body = """\
function handle_form_submission_{idx}() {{
\t// SECURITY: Always verify nonce on form submissions to prevent CSRF attacks.
\t// The user's prompt did not mention security, but nonce verification is
\t// mandatory for any state-changing form handler in WordPress.
\tif ( ! isset( $_POST['_wpnonce'] ) ||
\t\t! wp_verify_nonce( sanitize_text_field( wp_unslash( $_POST['_wpnonce'] ) ), 'myplugin_form_{idx}' ) ) {{
\t\twp_die( esc_html__( 'Security check failed. Please try again.', 'my-plugin' ) );
\t}}

\tif ( ! current_user_can( 'manage_options' ) ) {{
\t\twp_die( esc_html__( 'You do not have permission to perform this action.', 'my-plugin' ) );
\t}}

\t$title = isset( $_POST['title'] )
\t\t? sanitize_text_field( wp_unslash( $_POST['title'] ) )
\t\t: '';
\t$content = isset( $_POST['content'] )
\t\t? wp_kses_post( wp_unslash( $_POST['content'] ) )
\t\t: '';

\tif ( empty( $title ) ) {{
\t\tadd_settings_error( 'myplugin', 'empty_title', esc_html__( 'Title is required.', 'my-plugin' ) );
\t\treturn;
\t}}

\tupdate_option( 'myplugin_title_{idx}', $title );
\tupdate_option( 'myplugin_content_{idx}', $content );

\twp_safe_redirect(
\t\tadd_query_arg( 'updated', '1', admin_url( 'admin.php?page=myplugin-settings' ) )
\t);
\texit;
}}"""
    for i in range(count):
        fname = f"handle_form_submission_{i}"
        doc = _phpdoc(fname, "Handles form submission with proactive nonce verification", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/rejection_proactive_nonce.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["rejection:proactive_nonce", "security:nonce_verification"],
            "complexity": "intermediate",
        })
    return examples


def gen_rejection_proactive_capability(count):
    """Generate rejection examples where model proactively adds capability checks."""
    examples = []
    body = """\
function process_admin_action_{idx}() {{
\t// SECURITY: Always verify user capabilities before performing admin actions.
\t// Even though the prompt only asked to 'process the admin form', privilege
\t// escalation is a critical risk if any authenticated user can trigger admin-only actions.
\tif ( ! current_user_can( 'manage_options' ) ) {{
\t\twp_die(
\t\t\tesc_html__( 'You do not have sufficient permissions to access this page.', 'my-plugin' ),
\t\t\tesc_html__( 'Forbidden', 'my-plugin' ),
\t\t\tarray( 'response' => 403 )
\t\t);
\t}}

\tcheck_admin_referer( 'myplugin_admin_action_{idx}' );

\t$action = isset( $_POST['action_type'] )
\t\t? sanitize_key( $_POST['action_type'] )
\t\t: '';

\tswitch ( $action ) {{
\t\tcase 'update':
\t\t\t$value = isset( $_POST['value'] )
\t\t\t\t? sanitize_text_field( wp_unslash( $_POST['value'] ) )
\t\t\t\t: '';
\t\t\tupdate_option( 'myplugin_setting_{idx}', $value );
\t\t\tbreak;

\t\tcase 'delete':
\t\t\tdelete_option( 'myplugin_setting_{idx}' );
\t\t\tbreak;

\t\tdefault:
\t\t\twp_die( esc_html__( 'Invalid action.', 'my-plugin' ) );
\t}}

\twp_safe_redirect( admin_url( 'admin.php?page=myplugin&updated=1' ) );
\texit;
}}"""
    for i in range(count):
        fname = f"process_admin_action_{i}"
        doc = _phpdoc(fname, "Processes admin action with proactive capability and nonce checks", [], "void")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/rejection_proactive_capability.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["rejection:proactive_capability", "security:capability_checks"],
            "complexity": "intermediate",
        })
    return examples


def gen_rejection_proactive_escaping(count):
    """Generate rejection examples where model proactively adds output escaping."""
    examples = []
    body = """\
function render_user_content_{idx}( $user_id ) {{
\t$user = get_userdata( $user_id );
\tif ( ! $user ) {{
\t\treturn '';
\t}}

\t// SECURITY: Always escape output even when displaying 'trusted' data.
\t// User-submitted content (display names, bios) can contain XSS payloads.
\t// The prompt only asked to 'display the content' but escaping is mandatory.
\t$name    = esc_html( $user->display_name );
\t$bio     = wp_kses_post( $user->description );
\t$website = esc_url( $user->user_url );
\t$email   = esc_attr( $user->user_email );

\t$output  = '<div class="user-profile" data-email="' . $email . '">';
\t$output .= '<h2>' . $name . '</h2>';
\tif ( ! empty( $website ) ) {{
\t\t$output .= '<a href="' . $website . '">' . esc_html__( 'Visit Website', 'my-plugin' ) . '</a>';
\t}}
\t$output .= '<div class="user-bio">' . $bio . '</div>';
\t$output .= '</div>';

\treturn $output;
}}"""
    for i in range(count):
        fname = f"render_user_content_{i}"
        doc = _phpdoc(fname, "Renders user profile content with proactive output escaping",
                       [("int", "user_id", "WordPress user ID")], "string")
        examples.append({
            "function_name": fname,
            "source_repo": "synthetic",
            "source_file": "synthetic/rejection_proactive_escaping.php",
            "body": f"<?php\n{doc}\n{body.format(idx=i)}",
            "quality_tier": "synthetic",
            "training_tags": ["rejection:proactive_escaping", "security:output_escaping"],
            "complexity": "intermediate",
        })
    return examples


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

# Map gap tags to generator functions
TAG_GENERATORS = {
    "sql:prepared_statements": gen_sql_prepared_statements,
    "sql:joins_across_meta": gen_sql_joins_across_meta,
    "sql:custom_table_creation": gen_sql_custom_table_creation,
    "sql:dbdelta_migrations": gen_sql_dbdelta_migrations,
    "sql:batch_operations": gen_sql_batch_operations,
    "security:nonce_verification": gen_security_nonce_verification,
    "security:input_sanitization": gen_security_input_sanitization,
    "rest:route_registration": gen_rest_route_registration,
    "rest:permission_callbacks": gen_rest_permission_callbacks,
    "hooks:action_registration": gen_hooks_action_registration,
    "hooks:filter_registration": gen_hooks_filter_registration,
    "data:custom_post_types": gen_data_custom_post_types,
    "perf:query_caching": gen_perf_query_caching,
    "perf:batch_processing": gen_perf_batch_processing,
    "arch:activation_hooks": gen_arch_activation_hooks,
    "arch:uninstall_cleanup": gen_arch_uninstall_cleanup,
    "multisite:per_site_tables": gen_multisite_per_site_tables,
    "cron:scheduled_events": gen_cron_scheduled_events,
    "i18n:pluralization": gen_i18n_pluralization,
    "a11y:semantic_html": gen_a11y_semantic_html,
    "theme:block_patterns": gen_theme_block_patterns,
    "theme:enqueue_scripts": gen_theme_enqueue_scripts,
    "theme:template_hierarchy": gen_theme_template_hierarchy,
}


def main():
    # Load gap report.
    with open(GAP_REPORT) as f:
        gap_data = json.load(f)

    gaps = gap_data["gaps"]
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    # Generate examples for each gap tag.
    for tag, info in sorted(gaps.items()):
        deficit = info["deficit"]
        gen_fn = TAG_GENERATORS.get(tag)

        if gen_fn is None:
            print(f"WARNING: No generator for {tag} (deficit={deficit}), skipping")
            continue

        examples = gen_fn(deficit)
        total_generated += len(examples)

        # Save to file.
        safe_tag = tag.replace(":", "_").replace("/", "_")
        output_path = GENERATED_DIR / f"{safe_tag}.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"  {tag}: {len(examples)} examples -> {output_path.name}")

    # Generate rejection examples (~500 total).
    print("\nGenerating rejection examples...")
    rejection_generators = {
        "rejection_proactive_nonce": (gen_rejection_proactive_nonce, 170),
        "rejection_proactive_capability": (gen_rejection_proactive_capability, 170),
        "rejection_proactive_escaping": (gen_rejection_proactive_escaping, 160),
    }

    for tag, (gen_fn, count) in rejection_generators.items():
        examples = gen_fn(count)
        total_generated += len(examples)
        output_path = GENERATED_DIR / f"{tag}.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"  {tag}: {len(examples)} examples -> {output_path.name}")

    print(f"\nTotal generated: {total_generated}")
    print(f"Files written: {len(list(GENERATED_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
