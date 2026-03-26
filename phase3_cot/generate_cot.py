#!/usr/bin/env python3
"""
Generate Chain-of-Thought instruction-completion pairs from passed synthetic examples.
Samples 40 diverse examples across gap tags (34 normal + 6 rejection).
"""

import json
import random
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------

with open("phase2_synthetic/output/judged/passed_synthetic_A.json") as f:
    a = json.load(f)
with open("phase2_synthetic/output/judged/passed_synthetic_B.json") as f:
    b = json.load(f)
with open("phase2_synthetic/output/judged/passed_synthetic_C.json") as f:
    c = json.load(f)

all_examples = a + b + c

normal = [e for e in all_examples if "rejection_explanation" not in e]
rejections = [e for e in all_examples if "rejection_explanation" in e]

# ---------------------------------------------------------------------------
# Sampling strategy: 34 normal (1-2 per gap_tag) + 6 rejection (2 per type)
# ---------------------------------------------------------------------------

normal_by_tag = defaultdict(list)
for e in normal:
    normal_by_tag[e.get("gap_tag", "unknown")].append(e)

rejection_by_tag = defaultdict(list)
for e in rejections:
    rejection_by_tag[e.get("gap_tag", "unknown")].append(e)

# Tags with enough variety to contribute 2 examples
big_tags = {
    "a11y:semantic_html",
    "cron:scheduled_events",
    "data:custom_post_types",
    "hooks:filter_registration",
    "multisite:per_site_tables",
    "sql:batch_operations",
    "sql:custom_table_creation",
    "theme:template_hierarchy",
    "arch:activation_hooks",
    "perf:batch_processing",
    "sql:dbdelta_migrations",
    "theme:block_patterns",
}

normal_sampled = []
for tag, examples in sorted(normal_by_tag.items()):
    count = 2 if tag in big_tags else 1
    normal_sampled.extend(random.sample(examples, min(count, len(examples))))

rejection_sampled = []
for tag, examples in sorted(rejection_by_tag.items()):
    rejection_sampled.extend(random.sample(examples, min(2, len(examples))))

assert len(normal_sampled) == 34, f"Expected 34 normal, got {len(normal_sampled)}"
assert len(rejection_sampled) == 6, f"Expected 6 rejection, got {len(rejection_sampled)}"

# ---------------------------------------------------------------------------
# Complexity heuristic: rough line-count of function body
# ---------------------------------------------------------------------------

def infer_complexity(body: str) -> str:
    lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("*") and not l.strip().startswith("//")]
    if len(lines) < 25:
        return "simple"
    elif len(lines) < 60:
        return "medium"
    return "complex"


# ---------------------------------------------------------------------------
# Reverse-engineer instructions for normal examples that lack an instruction field
# ---------------------------------------------------------------------------

INSTRUCTIONS = {
    # a11y:semantic_html
    "wpft_render_accessible_table": (
        "Write a WordPress template function that renders an accessible HTML data table "
        "with proper WCAG 1.3.1 markup: caption, thead with scope attributes, tbody, "
        "and ARIA description support. Escape all cell content."
    ),
    "wpft_render_accessible_navigation": (
        "Create a WordPress template function that outputs an accessible site navigation "
        "region with a skip link, aria-label, a custom nav walker that marks the current "
        "page with aria-current='page', and toggle buttons with aria-expanded/aria-controls "
        "for sub-menus."
    ),
    # arch:activation_hooks
    "wpft_plugin_activate": (
        "Write a plugin activation hook callback that creates a custom database table with "
        "dbDelta, sets default plugin options using add_option, and flushes rewrite rules "
        "so custom post type slugs resolve immediately."
    ),
    "wpft_plugin_activate_set_options": (
        "Create a plugin activation callback that stores structured default options in a "
        "single serialised option key (to reduce autoloaded rows), merges new defaults over "
        "any existing values using wp_parse_args, and records activation history with a "
        "timestamp and the activating user ID."
    ),
    # arch:uninstall_cleanup
    "wpft_uninstall_remove_cron_schedules": (
        "Write an uninstall routine that removes all scheduled cron events for a plugin, "
        "including any per-post events, by scanning the full cron array for hooks prefixed "
        "with 'wpft_'. Guard with the WP_UNINSTALL_PLUGIN constant."
    ),
    # cron:scheduled_events
    "wpft_reschedule_on_blog_switch": (
        "Write a WordPress multisite-aware function that reschedules a per-site cron event "
        "when a blog is activated or deactivated in the network. Use switch_to_blog and "
        "restore_current_blog to scope the scheduling calls to each blog."
    ),
    "wpft_cron_hourly_transient_purge": (
        "Create a WordPress cron job that runs hourly to purge expired plugin transients "
        "directly from the options table (for setups with a persistent object cache). "
        "Register and unregister the event on plugin activation and deactivation."
    ),
    # data:custom_post_types
    "wpft_register_portfolio_post_type": (
        "Register a 'wpft_portfolio' custom post type with a full WP 5.0+ label set, "
        "REST API support, a custom archive slug, map_meta_cap with capability_type array, "
        "and fine-grained rewrite configuration."
    ),
    "wpft_register_testimonial_cpt": (
        "Register a Testimonial custom post type that has no public archive, is excluded "
        "from search, and is not shown in nav menus, but is still accessible by direct URL "
        "and via the REST API for use in shortcodes and blocks."
    ),
    # hooks:action_registration
    "wpft_register_post_save_actions": (
        "Write a function that registers all post-save action hooks for a plugin: cache "
        "invalidation and taxonomy sync on save_post, with autosave and revision guards "
        "to prevent unnecessary processing."
    ),
    # hooks:filter_registration
    "wpft_filter_upload_mimes": (
        "Create a WordPress filter callback for 'upload_mimes' that adds SVG, WebP, and a "
        "custom .wpft data format to the allowed upload types. Gate SVG uploads behind the "
        "manage_options capability to prevent stored XSS from untrusted contributors."
    ),
    "wpft_filter_kses_allowed_html": (
        "Write a WordPress filter callback for 'wp_kses_allowed_html' that adds the "
        "<details>, <summary>, and <mark> HTML5 elements with safe attributes to the "
        "'post' context, so authors can use them in post content without content being stripped."
    ),
    # multisite:per_site_tables
    "wpft_migrate_per_site_table_schema": (
        "Write a WordPress multisite migration function that runs a dbDelta schema update "
        "on every site's per-site plugin table, checking and updating each site's stored "
        "DB version independently using switch_to_blog and restore_current_blog."
    ),
    "wpft_create_per_site_table": (
        "Create a WordPress function that creates a plugin's per-site custom database table "
        "using dbDelta. It should work for both single-site and multisite networks, be "
        "called on plugin activation and when a new blog is created, and conditionally "
        "use switch_to_blog when a specific blog ID is provided."
    ),
    # perf:batch_processing
    "wpft_batch_update_postmeta": (
        "Write a WordPress function that updates a post meta key across many posts using "
        "direct SQL batching instead of update_post_meta, for high-performance bulk "
        "migrations. Use prepared statements and bust the object cache for each post."
    ),
    "wpft_memory_aware_batch_loop": (
        "Create a WordPress batch processing loop that monitors PHP memory usage after "
        "each item and pauses when approaching the memory limit, saving progress to an "
        "option so subsequent runs can resume from where it left off."
    ),
    # perf:query_caching
    "wpft_get_featured_posts_cached": (
        "Write a WordPress function that retrieves featured posts using a transient cache, "
        "running WP_Query only on cache miss. Automatically invalidate the cache when any "
        "post is saved or deleted. Use no_found_rows and targeted cache options for performance."
    ),
    # rest:permission_callbacks
    "wpft_rest_admin_only_permission": (
        "Write a REST API permission callback that restricts an endpoint to site "
        "administrators. Return a descriptive WP_Error with correct 401 vs 403 HTTP status "
        "codes so REST clients can distinguish unauthenticated from unauthorised requests."
    ),
    # rest:route_registration
    "wpft_register_products_rest_route": (
        "Register a read-only WP REST API endpoint at /wp-json/wpft/v1/products that "
        "supports pagination, category filtering, and price/date sorting. Provide full "
        "JSON Schema for all parameters, use WP_Query for the data source, and include "
        "X-WP-Total and X-WP-TotalPages response headers."
    ),
    # security:input_sanitization
    "wpft_sanitize_import_csv_row": (
        "Write a WordPress function that sanitizes a single row of CSV import data, "
        "applying the most appropriate sanitization function to each field type: "
        "text fields, URLs, slugs, prices, and status values with an allowlist."
    ),
    # security:nonce_verification
    "wpft_process_settings_update": (
        "Write a WordPress admin settings form handler that verifies the admin referer, "
        "checks manage_options capability, sanitizes all submitted fields, and persists "
        "settings to the database before redirecting back with a success indicator."
    ),
    # sql:batch_operations
    "wpft_batch_migrate_user_meta_to_custom_table": (
        "Write a WordPress function that migrates legacy user meta entries to a dedicated "
        "plugin table in chunks. Use a LEFT JOIN to find unmigrated users, insert with "
        "$wpdb->insert, sanitize all values, and delete the legacy meta only after a "
        "successful insert."
    ),
    "wpft_batch_delete_expired_sessions": (
        "Create a WordPress function that deletes expired session rows from a plugin "
        "session table in small batches to avoid table-lock issues. Use prepared DELETE "
        "statements, a max-passes guard, and log the last cleanup time."
    ),
    # sql:custom_table_creation
    "wpft_create_event_log_table": (
        "Write a WordPress plugin function that creates or upgrades a custom event log "
        "table using dbDelta. Include proper charset collation, a PRIMARY KEY, and "
        "multiple secondary indexes. Store the DB schema version after creation."
    ),
    "wpft_create_multisite_analytics_tables": (
        "Create a WordPress function that creates per-site analytics tables for a "
        "multisite-aware plugin using dbDelta. Use switch_to_blog/restore_current_blog "
        "and verify success by checking SHOW TABLES LIKE after creation."
    ),
    # sql:dbdelta_migrations
    "wpft_migrate_1_2_0": (
        "Write a WordPress DB migration function for version 1.2.0 that adds a new "
        "LONGTEXT meta column to an existing table and creates two new tables — including "
        "a junction table with a compound UNIQUE KEY — all using dbDelta."
    ),
    "wpft_migrate_add_fulltext_index": (
        "Write a WordPress DB migration function that adds a FULLTEXT index to a plugin "
        "table. Since dbDelta cannot add FULLTEXT indexes, query INFORMATION_SCHEMA to "
        "guard idempotency and use ALTER TABLE directly with a prepared statement."
    ),
    # sql:joins_across_meta
    "wpft_get_users_with_post_meta_summary": (
        "Write a WordPress function that returns users along with aggregated statistics "
        "for a numeric post meta key: post count, SUM, and AVG. Join wp_users, wp_posts, "
        "and wp_postmeta. Use GROUP BY, HAVING, and CAST to DECIMAL for numeric accuracy."
    ),
    # sql:prepared_statements
    "wpft_get_term_post_count_by_date": (
        "Write a WordPress function that counts published posts assigned to a specific "
        "taxonomy term, with optional date-range filtering. Build clauses dynamically and "
        "pass all values through $wpdb->prepare with a spread operator."
    ),
    # theme:block_patterns
    "wpft_register_newsletter_signup_pattern": (
        "Register a WordPress block pattern for a newsletter sign-up section using a "
        "two-column layout with a descriptive text column and a subscription form column. "
        "Wrap all translatable strings with esc_html__ and use _x() where translators "
        "need context."
    ),
    "wpft_register_pricing_table_pattern": (
        "Register a three-column WordPress block pattern for a pricing table using only "
        "core blocks. Each column should include a heading, price display, feature list, "
        "and CTA button. Use a helper closure to keep repetitive column markup DRY."
    ),
    # theme:enqueue_scripts
    "wpft_enqueue_frontend_assets": (
        "Write a WordPress function that enqueues a plugin's frontend scripts and styles "
        "with explicit dependency arrays, version-constant-based cache busting, and "
        "wp_localize_script exposing ajaxUrl, a WP nonce, and the REST API URL. Use the "
        "WP 6.3+ strategy/in_footer array syntax for defer/async."
    ),
    # theme:template_hierarchy
    "wpft_register_template_hierarchy_filters": (
        "Write a WordPress function that uses the WP 4.7+ {$type}_template_hierarchy "
        "filters to inject plugin-specific templates into the template lookup chain for "
        "custom post types and taxonomies, without overriding template_include."
    ),
    "wpft_taxonomy_template_loader": (
        "Create a WordPress template_include filter callback that loads the correct "
        "template for plugin-registered taxonomy archive pages, mirroring WordPress "
        "core's taxonomy template hierarchy and falling back to a plugin-bundled template."
    ),
}


# ---------------------------------------------------------------------------
# CoT reasoning builders
# ---------------------------------------------------------------------------

def build_normal_cot(e: dict) -> str:
    fn = e["function_name"]
    gap = e.get("gap_tag", "")
    notes = e.get("assessment", {}).get("notes", "")
    tags = e.get("assessment", {}).get("training_tags", e.get("training_tags", []))

    # Per-tag reasoning templates
    if gap == "a11y:semantic_html":
        if fn == "wpft_render_accessible_table":
            return (
                "Step-by-step explanation:\n"
                "1. WCAG 1.3.1 (Info and Relationships) requires that tabular data use "
                "proper markup so assistive technology can associate data cells with their "
                "headers; a bare <table> without <caption> or scope attributes fails this "
                "criterion.\n"
                "2. A <caption> element provides a visible and programmatic title for the "
                "table — preferred over aria-label because it is also displayed visually.\n"
                "3. scope='col' on <th> elements inside <thead> tells screen readers which "
                "column each header describes; scope='row' can optionally be added for row "
                "headers in the first <td> or <th> of each row.\n"
                "4. An explicit <tbody> wrapper satisfies browsers that require it and "
                "makes the DOM structure unambiguous for AT.\n"
                "5. aria-describedby links the table element to a summary paragraph, giving "
                "screen-reader users a quick overview without having to navigate the cells.\n"
                "6. esc_html() wraps every dynamic cell value to neutralise stored XSS "
                "payloads that could be lurking in post meta or imported data."
            )
        else:  # wpft_render_accessible_navigation
            return (
                "Step-by-step explanation:\n"
                "1. A <nav> element with a distinct aria-label is required when the page "
                "contains multiple navigation landmarks; without the label, screen readers "
                "announce only 'navigation' with no way to distinguish them.\n"
                "2. A visually-hidden skip link at the top of the nav lets keyboard users "
                "bypass the menu and jump straight to main content, satisfying WCAG 2.4.1 "
                "(Bypass Blocks).\n"
                "3. A custom Walker class sets aria-current='page' on the active menu item; "
                "this is the ARIA pattern screen readers rely on to identify the current "
                "location inside a nav.\n"
                "4. Toggle buttons for sub-menus use aria-expanded (true/false) and "
                "aria-controls (pointing to the sub-menu's ID) so AT users know the state "
                "and can navigate the relationship without a mouse.\n"
                "5. sanitize_html_class() is applied to IDs generated from menu item slugs "
                "to prevent malformed markup from untrusted menu item data.\n"
                "6. wp_nav_menu() is used with container=false because the outer <nav> is "
                "already the landmark container; a redundant <div> wrapper would create "
                "unnecessary nesting."
            )

    if gap == "arch:activation_hooks":
        if fn == "wpft_plugin_activate":
            return (
                "Step-by-step explanation:\n"
                "1. register_activation_hook() is the correct way to run setup code on "
                "activation — it fires before any redirect, unlike init or admin_init hooks "
                "which run on every request.\n"
                "2. dbDelta() is the WordPress-sanctioned function for schema creation and "
                "upgrades; it diffs the desired CREATE TABLE statement against the live "
                "schema and applies only necessary changes, making it safe to call repeatedly.\n"
                "3. $wpdb->get_charset_collate() ensures the table uses the same character "
                "set as the rest of the database, preventing collation mismatches on "
                "multi-byte strings.\n"
                "4. add_option() (not update_option()) is used for defaults so that "
                "re-activating the plugin does not overwrite settings the admin has already "
                "customised.\n"
                "5. The CPT registration function is called before flush_rewrite_rules() "
                "because WordPress needs the CPT in the rewrite map before it can flush it; "
                "reversing the order would leave CPT URLs broken until the next page load.\n"
                "6. The DB version constant is stored so future upgrades can check whether "
                "a migration has already been applied."
            )
        else:  # wpft_plugin_activate_set_options
            return (
                "Step-by-step explanation:\n"
                "1. Consolidating all plugin settings into a single option key reduces the "
                "number of autoloaded rows on every page load; WordPress loads every option "
                "with autoload=yes on init, so many small options have a measurable "
                "performance cost.\n"
                "2. wp_parse_args($existing, $defaults) merges new defaults over existing "
                "values rather than replacing them, so re-activation after an upgrade "
                "preserves the admin's configuration.\n"
                "3. update_option() with autoload=false on a large serialised option "
                "prevents the full settings blob from being loaded on every frontend "
                "request, where it is only needed in admin.\n"
                "4. Recording an activation history entry (timestamp, plugin version, "
                "user ID) provides an audit trail that is useful for support debugging and "
                "understanding when settings may have changed.\n"
                "5. get_current_user_id() is used rather than a hardcoded value because the "
                "plugin can be activated by different admins; the activating user is the "
                "relevant actor for the audit log."
            )

    if gap == "arch:uninstall_cleanup":
        return (
            "Step-by-step explanation:\n"
            "1. The WP_UNINSTALL_PLUGIN constant guard is mandatory at the top of any "
            "uninstall routine. WordPress defines this constant before including uninstall.php; "
            "if the file is accessed directly the constant is absent and the guard prevents "
            "accidental data deletion.\n"
            "2. wp_clear_scheduled_hook() removes every occurrence of a named cron event "
            "regardless of its scheduled arguments, which is the correct way to clear "
            "standard plugin cron hooks.\n"
            "3. Some plugins schedule per-object cron events (e.g. one event per post) whose "
            "hook names share a common prefix. These are not known at uninstall time, so the "
            "entire _get_cron_array() is scanned and any hook starting with 'wpft_' is "
            "unscheduled individually using wp_unschedule_event().\n"
            "4. str_starts_with() (PHP 8.0+) provides a readable, allocation-free prefix "
            "check without a regex or substr compare.\n"
            "5. Cron cleanup must happen before deleting plugin options, because the cron "
            "cleanup logic may need to read option values to know which hooks to remove."
        )

    if gap == "cron:scheduled_events":
        if fn == "wpft_reschedule_on_blog_switch":
            return (
                "Step-by-step explanation:\n"
                "1. On multisite, wp_schedule_event() is per-blog: each blog has its own "
                "cron table. A cron event scheduled on blog 1 will not fire for blog 5. "
                "This means the event must be (re-)scheduled in the context of each blog.\n"
                "2. switch_to_blog($blog_id) shifts $wpdb->prefix and relevant globals to "
                "the target blog, so wp_next_scheduled() and wp_schedule_event() operate on "
                "that blog's cron data. restore_current_blog() must always be called "
                "afterwards to avoid corrupting global state.\n"
                "3. A while loop with wp_unschedule_event() is used rather than "
                "wp_clear_scheduled_hook() because the event may have been scheduled with "
                "arguments; clearing by hook name removes all occurrences cleanly.\n"
                "4. wp_next_scheduled() guards against double-scheduling: if the event is "
                "already registered (e.g. on a re-activation), the schedule call is skipped.\n"
                "5. The recurrence interval is filterable via apply_filters(), allowing "
                "child plugins or tests to override it without modifying source code."
            )
        else:  # wpft_cron_hourly_transient_purge
            return (
                "Step-by-step explanation:\n"
                "1. On installations using a persistent object cache (e.g. Redis, Memcached), "
                "WordPress does not automatically purge expired transients from the database "
                "because the object cache shadows them. This cron job ensures the options "
                "table does not grow unboundedly.\n"
                "2. $wpdb->esc_like() is used to escape the wildcard prefix before passing "
                "it to a LIKE clause inside $wpdb->prepare(); without it, literal underscores "
                "or percent signs in the prefix would be interpreted as SQL wildcards.\n"
                "3. The DELETE query targets both the transient value (_transient_*) and its "
                "timeout sibling (_transient_timeout_*) in one pass to avoid orphaned "
                "timeout rows.\n"
                "4. The scheduling and cleanup hooks are registered as a matched pair — "
                "wp_schedule_event on activation, wp_clear_scheduled_hook on deactivation — "
                "so no ghost cron events are left behind when the plugin is deactivated.\n"
                "5. wp_next_scheduled() prevents duplicate scheduling if the activation hook "
                "fires more than once (e.g. network-activate then site-activate)."
            )

    if gap == "data:custom_post_types":
        if fn == "wpft_register_portfolio_post_type":
            return (
                "Step-by-step explanation:\n"
                "1. 'capability_type' => array('portfolio','portfolios') with 'map_meta_cap' "
                "=> true enables granular capability names (edit_portfolio, delete_portfolios, "
                "etc.) that roles and plugins can assign independently instead of sharing "
                "the generic 'post' capabilities.\n"
                "2. 'show_in_rest' => true is required for the block editor (Gutenberg) to "
                "work with this CPT; without it the classic editor is forced. rest_base sets "
                "the URL segment (/wp-json/wp/v2/<rest_base>) to something human-readable.\n"
                "3. WP 5.0+ extended label keys (item_published, item_scheduled, "
                "item_updated, etc.) populate the block editor's notices and give a "
                "polished, contextual UX instead of generic 'Post published' messages.\n"
                "4. The rewrite slug is set explicitly so the public archive URL is "
                "human-readable and predictable regardless of changes to the CPT label.\n"
                "5. The hook is add_action('init', ...) because register_post_type() must "
                "be called before the rewrite rules are built on init, not in plugins_loaded "
                "or admin_init."
            )
        else:  # wpft_register_testimonial_cpt
            return (
                "Step-by-step explanation:\n"
                "1. 'has_archive' => false prevents WordPress from generating a /testimonials/ "
                "archive URL, since testimonials are designed to be embedded via shortcode or "
                "block rather than browsed as a list.\n"
                "2. 'exclude_from_search' => true keeps testimonial posts out of the "
                "WordPress search index so they do not appear in front-end search results "
                "where they would be out of context.\n"
                "3. 'show_in_nav_menus' => false hides the CPT from the Appearance > Menus "
                "screen, preventing editors from accidentally linking to a non-existent "
                "archive or individual testimonial URL.\n"
                "4. 'publicly_queryable' => true is kept because individual testimonial "
                "URLs may still be needed for REST API access or canonical links; this "
                "allows direct access while keeping the browsing surfaces hidden.\n"
                "5. 'show_in_rest' => true enables the block editor and REST API access, "
                "which is required for the Gutenberg testimonial block to query and display "
                "entries via the REST endpoint."
            )

    if gap == "hooks:action_registration":
        return (
            "Step-by-step explanation:\n"
            "1. Grouping all add_action() calls in a dedicated registration function keeps "
            "the plugin bootstrap lean and makes hook dependencies explicit in one place; "
            "callers only invoke one function instead of sprinkling add_action calls across "
            "multiple files.\n"
            "2. The autosave guard (did_action('autosave_post') or checking "
            "DOING_AUTOSAVE) is critical on save_post: WordPress fires save_post during "
            "autosaves, and running cache invalidation or taxonomy sync on every autosave "
            "creates unnecessary load and can corrupt data.\n"
            "3. The revision guard (wp_is_post_revision()) prevents the callbacks from "
            "running when WordPress saves a revision, which would create misleading cache "
            "or taxonomy entries for the parent post.\n"
            "4. A current_user_can() check inside the callback that does the actual "
            "database write ensures that programmatic saves from WP-CLI or background "
            "processes still pass, while blocking saves from users without the requisite "
            "capability."
        )

    if gap == "hooks:filter_registration":
        if fn == "wpft_filter_upload_mimes":
            return (
                "Step-by-step explanation:\n"
                "1. SVG files can contain embedded JavaScript and are therefore an XSS "
                "risk when served from the same origin as the WordPress site. Restricting "
                "SVG uploads to manage_options (administrators) prevents contributors or "
                "authors from uploading a malicious SVG.\n"
                "2. The WP_User instanceof check handles the case where "
                "wp_get_current_user() returns an empty WP_User object (not logged in); "
                "calling ->has_cap() on a non-object would throw a PHP error.\n"
                "3. WebP is added for older WordPress versions (pre-5.8) that do not "
                "include it in the default mime list, ensuring compatibility across WP "
                "versions without duplicating existing entries.\n"
                "4. The custom .wpft extension is mapped to a safe application/octet-stream "
                "MIME type so it is handled as a binary download rather than parsed as code "
                "by the browser."
            )
        else:  # wpft_filter_kses_allowed_html
            return (
                "Step-by-step explanation:\n"
                "1. The filter checks that $context === 'post' before making any changes; "
                "wp_kses_allowed_html is called for multiple contexts (strip, data, etc.) "
                "and adding elements to the wrong context could weaken sanitization in "
                "security-sensitive areas.\n"
                "2. Only safe, non-scripting attributes (class, id, tabindex, title, "
                "open for <details>) are whitelisted. Event handler attributes (onclick, "
                "onchange, etc.) are deliberately excluded.\n"
                "3. The existing $tags array is merged non-destructively so other plugins "
                "that have already extended the allowlist are not overwritten; the filter "
                "chain remains cooperative.\n"
                "4. <details>/<summary> require special treatment because the 'open' "
                "attribute on <details> controls visibility state; including it allows "
                "authors to set the default-open/closed state in content."
            )

    if gap == "multisite:per_site_tables":
        if fn == "wpft_migrate_per_site_table_schema":
            return (
                "Step-by-step explanation:\n"
                "1. get_sites() with number=0 retrieves all sites in the network. The "
                "per-site iteration is necessary because each blog has its own options table "
                "with an independent DB version record.\n"
                "2. switch_to_blog($blog_id) shifts $wpdb->prefix to the target blog (e.g. "
                "wp_5_), so subsequent $wpdb calls read from and write to that blog's "
                "tables. restore_current_blog() reverts this after each iteration.\n"
                "3. get_option('wpft_db_version') inside the switch_to_blog context reads "
                "from the target blog's options table, not the main site's. This is why the "
                "version check must happen after the switch.\n"
                "4. dbDelta() is idempotent: if the schema is already up to date it does "
                "nothing, making the migration safe to run multiple times.\n"
                "5. update_option('wpft_db_version', WPFT_DB_VERSION) inside the blog "
                "context stores the version in that blog's options table so the migration "
                "is not re-run on the next upgrade check."
            )
        else:  # wpft_create_per_site_table
            return (
                "Step-by-step explanation:\n"
                "1. $wpdb->prefix already incorporates the blog prefix on multisite (e.g. "
                "'wp_5_'), so using $wpdb->prefix directly — rather than hardcoding 'wp_' "
                "— ensures the table is created in the correct blog's namespace without any "
                "extra string manipulation.\n"
                "2. dbDelta() requires the CREATE TABLE statement to follow a very specific "
                "format: two spaces before each column definition, PRIMARY KEY on its own "
                "line. Deviating from this format causes dbDelta to fail silently or "
                "misparse the diff.\n"
                "3. $wpdb->get_charset_collate() returns the correct CHARACTER SET and "
                "COLLATE clause for the database, preventing utf8/utf8mb4 collation "
                "mismatch errors on multilingual installs.\n"
                "4. is_plugin_active_for_network() guards the wpmu_new_blog handler so "
                "the table is only created if the plugin is network-activated; a "
                "per-site activation handles its own case separately.\n"
                "5. switch_to_blog/restore_current_blog wraps the table creation only when "
                "a non-current blog ID is provided, avoiding an unnecessary context switch "
                "for the current blog."
            )

    if gap == "perf:batch_processing":
        if fn == "wpft_batch_update_postmeta":
            return (
                "Step-by-step explanation:\n"
                "1. update_post_meta() fires multiple hooks (updated_post_meta, "
                "update_post_metadata, etc.) and runs an extra SELECT before each UPDATE. "
                "For bulk migrations across thousands of posts, this overhead is "
                "unacceptable; direct SQL batching is orders of magnitude faster.\n"
                "2. array_fill() generates the correct number of %d placeholders "
                "dynamically so that the IN() clause matches exactly the number of post "
                "IDs in the batch, and $wpdb->prepare() can type-check them all.\n"
                "3. maybe_serialize() is used on the new value to match WordPress's "
                "internal storage convention: arrays and objects are serialised before "
                "being written to the meta table.\n"
                "4. After the SQL UPDATE, wp_cache_delete() is called for each post_id "
                "to bust the WordPress object cache; otherwise, subsequent code reading "
                "post meta would receive the stale cached value until the cache TTL expired.\n"
                "5. The meta_key is passed through sanitize_key() before being interpolated "
                "into the SQL template so that only valid, safe key characters are used."
            )
        else:  # wpft_memory_aware_batch_loop
            return (
                "Step-by-step explanation:\n"
                "1. Long-running WP-Cron or CLI scripts run in a single PHP process and can "
                "exhaust the memory_limit before completing. Checking memory after each item "
                "and pausing early prevents fatal errors on large datasets.\n"
                "2. memory_get_usage(true) returns real memory allocated by PHP (not just "
                "usage reported by the runtime allocator), which is the correct metric for "
                "comparing against memory_limit.\n"
                "3. The threshold (default 0.85) is clamped between 0.5 and 0.95 so callers "
                "cannot set a value that would trigger before meaningful work is done or "
                "that would be too close to the limit to exit safely.\n"
                "4. Progress is stored in an option (update_option) after each batch "
                "completion so a scheduled follow-up run — or a CLI re-invocation — can "
                "resume from the last saved offset rather than reprocessing from the start.\n"
                "5. The helper that converts ini shorthand (K, M, G) to bytes handles the "
                "-1 (unlimited) case explicitly; wp_convert_hr_to_bytes does not handle "
                "the -1 value correctly in all WP versions."
            )

    if gap == "perf:query_caching":
        return (
            "Step-by-step explanation:\n"
            "1. WP_Query is expensive when run on every page load: it builds and executes "
            "at least one SQL query, processes the results, and populates the global "
            "$wp_query object. A transient wraps this behind a single options-table lookup "
            "on cache hits.\n"
            "2. The transient key is derived from a serialised hash of the query arguments "
            "so that different argument combinations get their own cache buckets, preventing "
            "one set of arguments from poisoning the cache for another.\n"
            "3. 'no_found_rows' => true disables the SQL_CALC_FOUND_ROWS trick that "
            "WP_Query uses for pagination totals; it is safe here because pagination is "
            "not needed for a fixed featured-posts block.\n"
            "4. Cache invalidation is hooked to save_post and deleted_post with guards for "
            "autosaves and revisions, ensuring stale data is not served after a post is "
            "edited or deleted, while avoiding unnecessary cache busts during autosave.\n"
            "5. An early exit if $post_type !== 'post' in the invalidation hook prevents "
            "unrelated CPT saves from triggering a cache flush."
        )

    if gap == "rest:permission_callbacks":
        return (
            "Step-by-step explanation:\n"
            "1. Returning false from a permission callback causes WordPress REST API to "
            "respond with a generic 403 Forbidden, giving the client no information about "
            "whether the request was unauthenticated or simply unauthorised. Returning a "
            "WP_Error with a status code allows REST clients to distinguish the two.\n"
            "2. A 401 Unauthorized response is returned when no user is authenticated "
            "(is_user_logged_in() returns false). This tells REST clients to retry with "
            "credentials before concluding the user lacks the capability.\n"
            "3. A 403 Forbidden response is returned when the user is authenticated but "
            "lacks the required capability (current_user_can('manage_options') is false). "
            "This tells the client the request will never succeed for this user and "
            "there is no point retrying with different authentication.\n"
            "4. WP_Error status codes are set via the 'status' key in the data array, "
            "not as a third constructor argument, which is the correct WP REST API pattern.\n"
            "5. Object-level permission callbacks use current_user_can('edit_post', $id) "
            "which checks the map_meta_cap expansion and respects custom capability types "
            "registered with register_post_type()."
        )

    if gap == "rest:route_registration":
        return (
            "Step-by-step explanation:\n"
            "1. register_rest_route() is called inside rest_api_init (not init) so the "
            "route is only registered when the REST API is actually being initialised, "
            "avoiding unnecessary work on non-API requests.\n"
            "2. WP_REST_Server::READABLE (= 'GET') is used instead of the string literal "
            "to make intent explicit and guard against typos.\n"
            "3. Full JSON Schema (type, minimum, maximum, enum, default) is supplied for "
            "every parameter so the REST API can validate and sanitize incoming values "
            "automatically before the callback is invoked.\n"
            "4. The permission_callback is set to '__return_true' for a public read-only "
            "endpoint; this is explicit and searchable, unlike an anonymous closure that "
            "returns true.\n"
            "5. X-WP-Total and X-WP-TotalPages headers mirror the WP core REST "
            "collections API contract, enabling REST clients to implement pagination "
            "without custom logic.\n"
            "6. WP_Query is used for data retrieval instead of raw SQL, ensuring "
            "compatibility with post status visibility rules and multisite table prefixes."
        )

    if gap == "security:input_sanitization":
        return (
            "Step-by-step explanation:\n"
            "1. Different field types require different sanitization functions. Using "
            "sanitize_text_field on a URL would corrupt it; using esc_url_raw on a free-text "
            "field would strip valid characters. Matching the sanitizer to the data type is "
            "the core principle.\n"
            "2. Status fields must be validated against an allowlist (in_array with strict "
            "mode) rather than sanitized, because sanitize_text_field would happily pass "
            "through an invalid status like 'deleted' that could corrupt business logic.\n"
            "3. filter_var($url, FILTER_VALIDATE_URL) validates URL structure before "
            "esc_url_raw cleans it; a URL that fails validation is replaced with an empty "
            "string rather than stored as a broken URL.\n"
            "4. SKU fields are sanitized with a specific regex that restricts to "
            "alphanumeric plus hyphens/underscores and caps the length, preventing both "
            "injection attempts and excessively long strings that could overflow the column.\n"
            "5. Price fields are cast to float and passed through abs() to guarantee a "
            "non-negative number; negative prices stored in the database would cause "
            "checkout calculation errors."
        )

    if gap == "security:nonce_verification":
        return (
            "Step-by-step explanation:\n"
            "1. check_admin_referer() verifies both the nonce value and the HTTP referer "
            "header, providing CSRF protection without a separate wp_verify_nonce() call "
            "and die() — it encapsulates the full check-and-die pattern.\n"
            "2. current_user_can('manage_options') must be checked after the nonce to "
            "avoid leaking capability information to unauthenticated or low-privilege "
            "attackers who might probe the endpoint.\n"
            "3. sanitize_key() on select/radio values ensures only valid key characters "
            "are stored; sanitize_text_field() on free-text fields strips tags and "
            "normalises whitespace; absint() on numeric fields prevents negative values "
            "and non-numeric input.\n"
            "4. wp_safe_redirect() is used rather than wp_redirect() to prevent open "
            "redirect attacks; it restricts the destination to the same host and a list "
            "of allowed hosts.\n"
            "5. add_query_arg('updated', '1', ...) appends a success flag to the redirect "
            "URL so the settings page can display an admin notice without storing state in "
            "a transient or session."
        )

    if gap == "sql:batch_operations":
        if fn == "wpft_batch_migrate_user_meta_to_custom_table":
            return (
                "Step-by-step explanation:\n"
                "1. A LEFT JOIN between wp_users and the new wpft_profiles table finds "
                "users who have the legacy meta but no corresponding profile row, so the "
                "query correctly selects only unmigrated users without a separate "
                "EXISTS/NOT EXISTS subquery.\n"
                "2. Processing in chunks (LIMIT $chunk_size) prevents the query result set "
                "from consuming unbounded memory; for large user bases a single SELECT "
                "could return millions of rows.\n"
                "3. $wpdb->insert() with an explicit format array (%s, %s, %s, %d) is used "
                "instead of raw SQL so WordPress handles escaping and type coercion "
                "correctly for each column.\n"
                "4. The legacy meta row is deleted only after $wpdb->insert() reports "
                "success (return value !== false). Deleting first and then failing on "
                "insert would cause data loss.\n"
                "5. The function returns 0 when no rows are processed, signalling to the "
                "caller (e.g. a cron job) that the migration is complete and should not "
                "be re-queued."
            )
        else:  # wpft_batch_delete_expired_sessions
            return (
                "Step-by-step explanation:\n"
                "1. Deleting a large number of rows in a single DELETE statement can lock "
                "the table for seconds, causing front-end requests that read sessions to "
                "queue up. Chunked deletes keep each statement small, holding the lock "
                "only briefly.\n"
                "2. DATE_SUB(NOW(), INTERVAL 1 DAY) is evaluated by MySQL, not PHP, "
                "ensuring the expiry cutoff is in the database server's timezone and "
                "avoids PHP/MySQL timezone drift issues.\n"
                "3. The for-loop includes a $max_passes guard to prevent an infinite loop "
                "if new sessions are inserted faster than old ones are deleted; after "
                "$max_passes iterations the function returns how many rows it deleted so "
                "far and lets the cron job reschedule.\n"
                "4. The break condition checks for false === $rows (a DB error) separately "
                "from 0 === $rows (no rows to delete), allowing callers to distinguish "
                "clean completion from failure.\n"
                "5. update_option('wpft_last_session_cleanup', current_time('mysql')) "
                "records the cleanup time so an admin dashboard or health check can detect "
                "if the cron job has stopped running."
            )

    if gap == "sql:custom_table_creation":
        if fn == "wpft_create_event_log_table":
            return (
                "Step-by-step explanation:\n"
                "1. dbDelta() is the only correct way to create or upgrade WordPress plugin "
                "tables; it diffs the desired schema against the live table and applies "
                "only the necessary ALTER TABLE statements, making it idempotent and safe "
                "to run on every activation.\n"
                "2. The CREATE TABLE statement must follow dbDelta's exact formatting rules: "
                "two spaces before each field definition, PRIMARY KEY on its own line. "
                "Violating these rules causes dbDelta to misparse the diff and skip needed "
                "changes silently.\n"
                "3. $wpdb->get_charset_collate() returns the correct CHARACTER SET / "
                "COLLATE for the server, preventing utf8/utf8mb4 mismatch errors on "
                "emoji-enabled installs.\n"
                "4. Secondary indexes (user_id, action, logged_at) are added for the "
                "query patterns the plugin uses. An unindexed table is fast to create but "
                "slow to query as row count grows.\n"
                "5. The DB version constant is stored with update_option after creation "
                "so that future upgrades can check whether a migration has already been "
                "applied without inspecting the live schema."
            )
        else:  # wpft_create_multisite_analytics_tables
            return (
                "Step-by-step explanation:\n"
                "1. On multisite, $wpdb->prefix already includes the blog prefix (e.g. "
                "'wp_5_') when called inside a switch_to_blog() context, so the table "
                "name is automatically scoped to the correct site without any manual "
                "string manipulation.\n"
                "2. switch_to_blog($blog_id) followed by restore_current_blog() is the "
                "correct pattern for operating on a specific site's tables. Forgetting "
                "restore_current_blog() would leave subsequent code executing in the wrong "
                "blog context for the remainder of the request.\n"
                "3. After dbDelta(), a SHOW TABLES LIKE check using $wpdb->prepare() "
                "verifies that the table was actually created. dbDelta can fail silently "
                "(e.g. due to insufficient MySQL privileges) so the explicit check provides "
                "a reliable success signal.\n"
                "4. The URL column is VARCHAR(2083) — the maximum URL length for IE11 "
                "compatibility — and session_id is CHAR(36) to store UUID v4 values "
                "at a fixed width, which is more efficient than VARCHAR for constant-length "
                "strings."
            )

    if gap == "sql:dbdelta_migrations":
        if fn == "wpft_migrate_1_2_0":
            return (
                "Step-by-step explanation:\n"
                "1. dbDelta() compares the full CREATE TABLE statement against the live "
                "schema and adds any columns that are present in the statement but absent "
                "in the table; this allows adding the new 'meta' LONGTEXT column by "
                "simply including it in the table definition.\n"
                "2. The new wpft_tags lookup table and the wpft_record_tags junction table "
                "are created in the same dbDelta call; dbDelta processes multiple "
                "CREATE TABLE blocks separated by semicolons.\n"
                "3. The junction table (wpft_record_tags) has a compound UNIQUE KEY on "
                "(record_id, tag_id) to prevent duplicate associations; a plain INDEX "
                "would not enforce uniqueness.\n"
                "4. BIGINT(20) UNSIGNED NOT NULL DEFAULT 0 on foreign key columns matches "
                "the data type of WordPress auto-increment primary keys, ensuring correct "
                "JOIN behaviour without implicit type coercion.\n"
                "5. update_option('wpft_db_version', '1.2.0') is called at the end of the "
                "migration so the migration dispatcher knows not to re-run it on subsequent "
                "page loads."
            )
        else:  # wpft_migrate_add_fulltext_index
            return (
                "Step-by-step explanation:\n"
                "1. dbDelta() silently ignores FULLTEXT index directives because its diff "
                "algorithm does not handle FULLTEXT. The migration therefore falls back to "
                "a direct ALTER TABLE statement.\n"
                "2. An idempotency guard queries INFORMATION_SCHEMA.STATISTICS to check "
                "whether the index already exists before attempting to add it. Without "
                "this check, re-running the migration (e.g. after a failed upgrade) would "
                "produce a MySQL error 1061 (Duplicate key name).\n"
                "3. $wpdb->prepare() is used for the INFORMATION_SCHEMA query with %s "
                "placeholders for DB_NAME, the table name, and the index name. Although "
                "these values come from constants and the plugin (not user input), prepare() "
                "is used for consistency with WPCS rules.\n"
                "4. A phpcs:ignore comment is added on the ALTER TABLE line because WPCS "
                "flags direct $wpdb->query() calls; the comment documents that the "
                "suppression is intentional and the value is trusted (from a constant).\n"
                "5. error_log() records failure details if $wpdb->last_error is set after "
                "the ALTER, providing a debuggable trace without surfacing database errors "
                "to front-end users."
            )

    if gap == "sql:joins_across_meta":
        return (
            "Step-by-step explanation:\n"
            "1. A single JOIN query across wp_users, wp_posts, and wp_postmeta retrieves "
            "aggregated statistics in one database round-trip; the alternative N+1 pattern "
            "(one query per user to fetch their post meta) scales linearly with the number "
            "of users.\n"
            "2. CAST(pm.meta_value AS DECIMAL(10,2)) is necessary because meta_value is "
            "stored as a VARCHAR; arithmetic on VARCHAR values uses implicit string-to-number "
            "conversion that silently returns 0 for non-numeric strings. CAST makes the "
            "conversion explicit and predictable.\n"
            "3. GROUP BY u.ID allows SUM() and AVG() to aggregate per user, while COUNT("
            "DISTINCT p.ID) counts unique posts per user rather than rows (which could be "
            "inflated by multiple meta entries per post).\n"
            "4. HAVING COUNT(DISTINCT p.ID) >= %d filters out users with fewer than the "
            "minimum post count at the SQL level, reducing the result set before PHP "
            "processes it.\n"
            "5. sanitize_key() is applied to the meta_key parameter before it is passed "
            "to $wpdb->prepare() as an additional defence; meta keys should only contain "
            "letters, numbers, hyphens, and underscores."
        )

    if gap == "sql:prepared_statements":
        return (
            "Step-by-step explanation:\n"
            "1. $wpdb->prepare() is the mandatory mechanism for parameterised queries in "
            "WordPress; it prevents SQL injection by using PDO-style typed placeholders "
            "(%s, %d, %f) and escaping values appropriately for the MySQL context.\n"
            "2. The date filter clauses are built dynamically: a $values array is "
            "initialised with the always-present parameters, and extra placeholders and "
            "values are pushed to their respective arrays only when the optional dates are "
            "provided. The final prepare() call uses spread ($values) to match placeholders "
            "to values without positional mismatch risk.\n"
            "3. COUNT(DISTINCT p.ID) is used rather than COUNT(*) because the JOIN with "
            "wp_term_relationships and wp_term_taxonomy can produce multiple rows per post "
            "when a post is assigned to multiple terms.\n"
            "4. sanitize_key($taxonomy) sanitizes the taxonomy slug before it is "
            "interpolated into the SQL fragment (used to build the JOIN condition), since "
            "it is not a value placeholder but a table/column identifier pattern."
        )

    if gap == "theme:block_patterns":
        if fn == "wpft_register_newsletter_signup_pattern":
            return (
                "Step-by-step explanation:\n"
                "1. register_block_pattern() is called inside init (not wp_loaded or "
                "admin_init) so the pattern is available to both the block editor and "
                "the REST API pattern endpoint.\n"
                "2. All user-facing strings inside the block grammar are wrapped with "
                "esc_html__() or esc_attr__() rather than __(), ensuring that translatable "
                "strings are also output-escaped at the point they are embedded in HTML "
                "attribute or text contexts.\n"
                "_x() is used for strings where translators need context (e.g. a button "
                "label that reads differently in other languages depending on surrounding "
                "copy).\n"
                "3. The pattern uses only core blocks (core/columns, core/group, "
                "core/paragraph, core/buttons) ensuring it renders correctly on any "
                "block-theme or classic-theme with Gutenberg without depending on a "
                "custom block that may not be available.\n"
                "4. The pattern content is a raw block grammar string (HTML comments with "
                "JSON attributes) that Gutenberg parses; PHP variables are interpolated "
                "using sprintf() to keep the string readable and the dynamic values "
                "properly escaped before injection."
            )
        else:  # wpft_register_pricing_table_pattern
            return (
                "Step-by-step explanation:\n"
                "1. A private helper closure is used to generate the repeated per-tier "
                "column markup. Duplicating the block grammar for each of three pricing "
                "tiers would create a large, error-prone string; the closure keeps the "
                "code DRY and makes it easy to add or change tiers.\n"
                "2. esc_html() is applied to all feature strings and tier names that are "
                "passed as PHP variables into the block grammar, preventing XSS if the "
                "values are ever sourced from translatable strings or dynamic data.\n"
                "3. All text uses __() with the plugin text domain so that pricing tier "
                "names, feature labels, and button copy can be translated.\n"
                "4. Only core blocks are used (core/columns, core/group, core/heading, "
                "core/list, core/buttons, core/button) to ensure the pattern works on any "
                "WordPress installation without a plugin dependency.\n"
                "5. The pattern is categorised under 'wpft-patterns' which is registered "
                "separately with register_block_pattern_category(), grouping plugin-provided "
                "patterns separately from theme patterns in the inserter."
            )

    if gap == "theme:enqueue_scripts":
        return (
            "Step-by-step explanation:\n"
            "1. wp_enqueue_scripts is the correct hook for frontend asset enqueueing; "
            "using wp_head or init would either miss the queueing mechanism or enqueue "
            "scripts on every admin page too.\n"
            "2. The WPFT_VERSION constant is used as the version parameter for both "
            "scripts and styles. WordPress appends this as ?ver=X to the URL, busting "
            "browser caches automatically on plugin updates.\n"
            "3. wp_register_script() without wp_enqueue_script() is used for utility "
            "scripts that are only conditionally needed; this registers the handle for "
            "use as a dependency without immediately outputting a <script> tag.\n"
            "4. The WP 6.3+ array syntax for the $args parameter ('strategy' => 'defer', "
            "'in_footer' => true) is used to load the main script deferred, improving "
            "Time-to-Interactive without requiring async handling in the script itself.\n"
            "5. wp_localize_script() exposes ajaxUrl, a CSRF nonce (wp_create_nonce), and "
            "the REST API root URL so the frontend JavaScript can make both AJAX and REST "
            "requests securely without hardcoding URLs or nonces in template files.\n"
            "6. esc_url_raw() is applied to rest_url() output because esc_url() would "
            "encode the trailing slash and colon in ways that break the REST endpoint URL "
            "when consumed by JavaScript."
        )

    if gap == "theme:template_hierarchy":
        if fn == "wpft_register_template_hierarchy_filters":
            return (
                "Step-by-step explanation:\n"
                "1. The {$type}_template_hierarchy filters (e.g. single_template_hierarchy, "
                "archive_template_hierarchy) were introduced in WordPress 4.7 and are "
                "preferred over template_include because they operate on the candidate list "
                "before WordPress resolves it, preserving theme compatibility and the "
                "full hierarchy lookup.\n"
                "2. Static closures (static function() use (...)) are used to avoid "
                "accidentally capturing $this or other heap references; they are slightly "
                "more memory-efficient and signal that no object context is needed.\n"
                "3. Plugin template candidates are prepended (array_merge($plugin_templates, "
                "$templates)) so that the plugin's templates take priority over the theme, "
                "allowing the plugin to ship sensible defaults that themes can override.\n"
                "4. The plugin template directory is filterable via apply_filters() so "
                "that child plugins or themes can redirect the lookup path without "
                "modifying the plugin source.\n"
                "5. is_singular() and is_post_type_archive() guards ensure the filters "
                "only inject templates when the query matches the plugin's registered CPT, "
                "leaving all other template lookups unaffected."
            )
        else:  # wpft_taxonomy_template_loader
            return (
                "Step-by-step explanation:\n"
                "1. The template_include filter is the entry point for custom template "
                "resolution; the callback receives the path WordPress has already resolved "
                "and can replace it with a more specific one.\n"
                "2. is_tax() with the plugin's taxonomy slugs guards the callback so it "
                "only runs on the plugin's taxonomy archive pages, leaving all other "
                "taxonomy and CPT templates undisturbed.\n"
                "3. The candidate list mirrors WordPress core's taxonomy hierarchy order: "
                "taxonomy-{taxonomy}-{term}.php → taxonomy-{taxonomy}.php → taxonomy.php. "
                "Following this order ensures theme authors can predict which file to "
                "create to override the plugin's default.\n"
                "4. sanitize_html_class() is applied to both the taxonomy slug and the "
                "term slug when building candidate filenames to prevent directory traversal "
                "or filename injection via malformed taxonomy/term slugs.\n"
                "5. locate_template() searches the child theme first, then the parent "
                "theme, before the plugin falls back to its bundled template; this "
                "respects the WordPress theme hierarchy without duplicating its logic."
            )

    # Generic fallback (should not be reached with current sample set)
    return (
        f"Step-by-step explanation:\n"
        f"1. The function follows standard WordPress API conventions for {gap.replace(':', ' — ')}.\n"
        f"2. Input validation and sanitization are applied before any database operations.\n"
        f"3. WordPress-native functions are preferred over raw PHP equivalents for portability.\n"
        f"4. Error handling returns sensible defaults so callers do not need to handle null.\n"
        f"5. Assessment notes: {notes}"
    )


def build_rejection_cot(e: dict) -> str:
    fn = e["function_name"]
    gap = e.get("gap_tag", "")
    instruction = e.get("instruction", "")
    rejection_explanation = e.get("rejection_explanation", "")

    if gap == "rejection_proactive_capability":
        if fn == "wpft_render_system_info_page":
            return (
                "The prompt asks for an admin page displaying server environment info "
                "(PHP version, memory limit, active plugins) but omits any access control. "
                "A responsible implementation must include current_user_can('manage_options') "
                "because this data is a significant security risk. Here's why:\n"
                "1. Vulnerability explanation: PHP version, server software, memory limits, "
                "and the list of active plugins form a precise fingerprint of the "
                "installation. An attacker who can access this page can immediately identify "
                "outdated software with known CVEs and target their exploit accordingly. "
                "Without access control, any logged-in user — including subscribers — can "
                "reach this page.\n"
                "2. How the added code prevents it: current_user_can('manage_options') "
                "restricts the page to administrators only (users with the 'administrator' "
                "role on single-site or network super admins). The check is placed at the "
                "very start of the render function before any output is produced, so an "
                "unauthorised user is redirected or shown a permission error before any "
                "server details are echoed.\n"
                "3. All dynamic output (PHP_VERSION, ini_get() values, plugin file paths) "
                "is wrapped with esc_html() to prevent stored or reflected XSS even within "
                "the admin context, where users may attempt to exploit other admin users."
            )
        else:  # wpft_view_debug_log_page
            return (
                "The prompt asks for an admin sub-page that reads and displays the plugin's "
                "debug log file but omits any capability gate. A responsible implementation "
                "must include current_user_can('manage_options') because debug logs "
                "contain highly sensitive information. Here's why:\n"
                "1. Vulnerability explanation: debug logs routinely contain API keys, OAuth "
                "tokens, database query details, internal file paths, and user data captured "
                "during error conditions. Without access control, any editor or author — who "
                "has access to wp-admin — could read these secrets, or an attacker who "
                "compromises a low-privilege account could use this page to escalate access.\n"
                "2. How the added code prevents it: current_user_can('manage_options') is "
                "checked at the top of the render callback and immediately returns (or calls "
                "wp_die()) for any non-administrator, before file_get_contents or file() is "
                "called. This means the file is never read, let alone output.\n"
                "3. Memory-safe file reading: file() + array_slice(-200) loads only the "
                "last 200 lines into memory, preventing the page from exhausting PHP memory "
                "on a large log file. The file_exists() guard prevents PHP warnings if the "
                "log has not been created yet.\n"
                "4. esc_html() wraps every log line before output to prevent XSS if a "
                "log message contains HTML characters from user input that was logged during "
                "an error."
            )

    if gap == "rejection_proactive_escaping":
        if fn == "wpft_display_user_bio_widget":
            return (
                "The prompt asks for a sidebar widget displaying the current user's display "
                "name and biographical info but omits output escaping. A responsible "
                "implementation must escape every user-controlled value before output "
                "because profile fields are a stored XSS vector. Here's why:\n"
                "1. Vulnerability explanation: WordPress user profile fields (display_name, "
                "description) can be edited by the account holder via their profile page or "
                "via the REST API. A user who sets their display_name to "
                "'<script>alert(document.cookie)</script>' would have that script execute in "
                "every visitor's browser when the widget renders — a classic stored XSS attack.\n"
                "2. How the added code prevents it: esc_html() on display_name converts "
                "HTML special characters (&, <, >, ', \") to their HTML entity equivalents, "
                "neutralising any script or markup injection in the name field. "
                "wp_kses_post() on the bio allows a safe subset of HTML (the same tags "
                "permitted in post content) while stripping dangerous elements and "
                "event handlers.\n"
                "3. Context-appropriate escaping: the avatar URL is passed through esc_url() "
                "to prevent javascript: pseudo-protocol URLs from being injected as the src "
                "attribute. The alt attribute is wrapped with esc_attr() to prevent "
                "attribute-context injection."
            )
        else:  # wpft_render_taxonomy_term_archive_title
            return (
                "The prompt asks for a template function displaying a taxonomy term name "
                "and description on archive pages but omits output escaping. A responsible "
                "implementation must escape both fields because taxonomy data is editable "
                "through the admin UI and the REST API. Here's why:\n"
                "1. Vulnerability explanation: Term names and descriptions can be set by "
                "any user with the edit_terms capability (typically editors and above). A "
                "term name containing '<script>alert(1)</script>' would execute JavaScript "
                "in every visitor's browser on the archive page — a stored XSS attack. "
                "Term descriptions also accept HTML, making them a richer injection surface.\n"
                "2. How the added code prevents it: esc_html() on the term name converts "
                "all HTML special characters to entities, rendering any injected markup "
                "as literal text. wp_kses_post() on the description allows the safe HTML "
                "subset (headings, paragraphs, links with safe attributes) while stripping "
                "script tags, event handlers, and other dangerous constructs.\n"
                "3. The instanceof WP_Term guard checks that get_queried_object() returned "
                "a valid term before accessing its properties, preventing PHP notices on "
                "non-taxonomy pages where this template part might accidentally be loaded."
            )

    if gap == "rejection_proactive_nonce":
        if fn == "wpft_handle_ajax_option_update":
            return (
                "The prompt asks for an AJAX handler that updates a plugin option when a "
                "toggle is switched in the admin UI but omits nonce verification. A "
                "responsible implementation must include check_ajax_referer() because AJAX "
                "endpoints are reachable by any script in the user's browser session. "
                "Here's why:\n"
                "1. Vulnerability explanation: WordPress AJAX endpoints (admin-ajax.php) "
                "are accessible to any JavaScript running in the browser, including scripts "
                "on other tabs or in injected ad content. Without nonce verification, a "
                "malicious website visited by a logged-in admin can silently send a crafted "
                "fetch() request to toggle the plugin option — a Cross-Site Request Forgery "
                "(CSRF) attack that requires no special privileges beyond being logged in.\n"
                "2. How the added code prevents it: check_ajax_referer('wpft_toggle_option', "
                "'_ajax_nonce') verifies a cryptographic token that was generated server-side "
                "and embedded in the admin page when it was rendered. The token is tied to "
                "the user session and expires after 24 hours. A forged request from another "
                "origin does not have this token and will fail verification, triggering "
                "wp_send_json_error with a 403 response.\n"
                "3. current_user_can('manage_options') is checked after the nonce to ensure "
                "only administrators can toggle plugin settings, even if a valid nonce is "
                "somehow obtained by a lower-privilege user.\n"
                "4. filter_var($value, FILTER_VALIDATE_BOOLEAN) converts the incoming "
                "toggle value to a proper boolean rather than storing the raw string '1' "
                "or 'true', preventing unexpected truthy comparisons in option readers."
            )
        else:  # wpft_save_settings_page_handler
            return (
                "The prompt asks for a form handler that saves a custom field value from a "
                "settings page but omits nonce verification. A responsible implementation "
                "must include wp_verify_nonce() because form handlers without CSRF "
                "protection allow attackers to silently modify site configuration. "
                "Here's why:\n"
                "1. Vulnerability explanation: HTTP POST requests can be forged. Any web "
                "page the administrator visits could contain a hidden form that auto-submits "
                "to the settings handler URL. Without nonce verification the handler cannot "
                "distinguish a legitimate form submission from a forged one, and the "
                "attacker can overwrite any option the handler touches — including security "
                "settings — just by getting an admin to visit a malicious URL.\n"
                "2. How the added code prevents it: wp_verify_nonce() checks a "
                "cryptographically signed token (embedded in the form as a hidden field) "
                "that is unique to the user session and action name. The nonce is extracted "
                "with sanitize_key(wp_unslash($_POST[...])) before verification to meet "
                "WPCS requirements. On failure, wp_die() with a 403 response terminates "
                "the request immediately.\n"
                "3. sanitize_text_field(wp_unslash($_POST['wpft_custom_field'])) sanitizes "
                "the submitted value before update_option, preventing stored XSS via an "
                "option value that is later echoed in the admin.\n"
                "4. wp_safe_redirect() after a successful save prevents open-redirect "
                "attacks; add_query_arg('updated', '1', wp_get_referer()) appends a "
                "success flag so the settings page can display an admin notice."
            )

    # Fallback
    return (
        f"The prompt asks for {instruction} but omits a critical security measure. "
        f"A responsible implementation must add the missing protection. Here's why:\n"
        f"1. Vulnerability explanation: {rejection_explanation}\n"
        f"2. How the added code prevents the attack by validating every untrusted input "
        f"before processing."
    )


# ---------------------------------------------------------------------------
# Reverse-engineer instructions for rejection examples
# ---------------------------------------------------------------------------

REJECTION_INSTRUCTIONS = {
    "wpft_render_system_info_page": (
        "Add an admin page that displays server environment info like PHP version, "
        "memory limit, and active plugins"
    ),
    "wpft_view_debug_log_page": (
        "Create an admin sub-page that reads and displays the plugin's debug log file"
    ),
    "wpft_display_user_bio_widget": (
        "Create a sidebar widget that outputs the current user's display name and "
        "biographical info"
    ),
    "wpft_render_taxonomy_term_archive_title": (
        "Write a template function that displays the current taxonomy term name and "
        "description on archive pages"
    ),
    "wpft_handle_ajax_option_update": (
        "Create an AJAX handler that updates a plugin option when a toggle is switched "
        "in the admin UI"
    ),
    "wpft_save_settings_page_handler": (
        "Create a form handler that saves a custom field value from a settings page"
    ),
}


# ---------------------------------------------------------------------------
# Build CoT records
# ---------------------------------------------------------------------------

output = []

for e in normal_sampled:
    fn = e["function_name"]
    instruction_text = INSTRUCTIONS.get(fn)
    if not instruction_text:
        instruction_text = f"Implement {fn.replace('_', ' ')} in WordPress"

    # Merge training tags from both top-level and assessment
    tags = list(e.get("training_tags", []))
    assessment_tags = e.get("assessment", {}).get("training_tags", [])
    for t in assessment_tags:
        if t not in tags:
            tags.append(t)

    record = {
        "task_type": "wp_gen",
        "instruction": f"<wp_gen> {instruction_text}",
        "response": e["body"],
        "cot_reasoning": build_normal_cot(e),
        "source_repo": "synthetic",
        "training_tags": tags,
        "complexity": infer_complexity(e["body"]),
    }
    output.append(record)

for e in rejection_sampled:
    fn = e["function_name"]
    instruction_text = e.get("instruction") or REJECTION_INSTRUCTIONS.get(fn) or f"Implement {fn.replace('_', ' ')}"

    tags = list(e.get("training_tags", []))
    assessment_tags = e.get("assessment", {}).get("training_tags", [])
    for t in assessment_tags:
        if t not in tags:
            tags.append(t)

    record = {
        "task_type": "wp_gen",
        "instruction": f"<wp_gen> {instruction_text}",
        "response": e["body"],
        "cot_reasoning": build_rejection_cot(e),
        "source_repo": "synthetic",
        "training_tags": tags,
        "complexity": infer_complexity(e["body"]),
    }
    output.append(record)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

import os
os.makedirs("phase3_cot/output", exist_ok=True)

with open("phase3_cot/output/cot_synthetic.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Written {len(output)} CoT records to phase3_cot/output/cot_synthetic.json")

# Verify tag coverage
from collections import Counter
gap_counts = Counter(
    e.get("gap_tag", "unknown") for e in normal_sampled + rejection_sampled
)
print("\nGap tag coverage:")
for tag, count in sorted(gap_counts.items()):
    print(f"  {tag}: {count}")

# Complexity distribution
complexity_counts = Counter(r["complexity"] for r in output)
print("\nComplexity distribution:")
for k, v in sorted(complexity_counts.items()):
    print(f"  {k}: {v}")
PYEOF
