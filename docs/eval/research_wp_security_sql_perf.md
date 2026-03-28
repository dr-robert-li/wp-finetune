# WordPress Security, SQL & Performance Best Practices
## Research for Code Quality Rubric Scoring Criteria

> **Purpose**: Specific, automatable patterns (positive and negative) for use in a code quality rubric.  
> **Sources**: WordPress Developer Documentation, WordPress VIP Documentation, Wordfence, Make WordPress Core, PHPStan/Psalm project docs.  
> **Last Updated**: March 2026

---

## Table of Contents

1. [WordPress SQL Safety](#1-wordpress-sql-safety)
   - [1.1 $wpdb Methods Reference](#11-wpdb-methods-reference)
   - [1.2 $wpdb->prepare() Patterns](#12-wpdb-prepare-patterns)
   - [1.3 WP_Query vs Direct SQL](#13-wp_query-vs-direct-sql)
   - [1.4 SQL Injection Prevention Patterns](#14-sql-injection-prevention-patterns)
   - [1.5 Common SQL Anti-Patterns](#15-common-sql-anti-patterns)
2. [WordPress Security](#2-wordpress-security)
   - [2.1 Sanitization Functions](#21-sanitization-functions)
   - [2.2 Escaping Functions](#22-escaping-functions)
   - [2.3 Nonce Patterns](#23-nonce-patterns)
   - [2.4 Capability Checks](#24-capability-checks)
   - [2.5 Data Validation Patterns](#25-data-validation-patterns)
   - [2.6 File Upload Security](#26-file-upload-security)
   - [2.7 CSRF / XSS / SQLi Prevention Summary](#27-csrf--xss--sqli-prevention-summary)
   - [2.8 AJAX Security Patterns](#28-ajax-security-patterns)
3. [WordPress Performance Patterns](#3-wordpress-performance-patterns)
   - [3.1 Object Caching & Transients](#31-object-caching--transients)
   - [3.2 WP_Query Optimization Arguments](#32-wp_query-optimization-arguments)
   - [3.3 N+1 Query Detection Patterns](#33-n1-query-detection-patterns)
   - [3.4 Autoload Option Management](#34-autoload-option-management)
   - [3.5 Remote HTTP Request Patterns](#35-remote-http-request-patterns)
   - [3.6 Script/Style Enqueueing](#36-scriptstyle-enqueueing)
   - [3.7 Database Query Optimization](#37-database-query-optimization)
4. [Static Analysis Tools](#4-static-analysis-tools)
   - [4.1 PHP_CodeSniffer + WordPressCS](#41-php_codesniffer--wordpresscs)
   - [4.2 PHPStan for WordPress](#42-phpstan-for-wordpress)
   - [4.3 Psalm for WordPress](#43-psalm-for-wordpress)
   - [4.4 Phan for PHP](#44-phan-for-php)
   - [4.5 phpcs-security-audit](#45-phpcs-security-audit)
   - [4.6 SonarQube PHP Rules](#46-sonarqube-php-rules)
   - [4.7 WordPress-Specific Tools](#47-wordpress-specific-tools)
5. [Scoring Criteria Summary Tables](#5-scoring-criteria-summary-tables)

---

## 1. WordPress SQL Safety

### 1.1 $wpdb Methods Reference

Source: [WordPress Developer Reference – wpdb class](https://developer.wordpress.org/reference/classes/wpdb/)

| Method | Purpose | Escaping Handled? | Returns |
|--------|---------|-------------------|---------|
| `$wpdb->get_var( $sql )` | Fetch single scalar value | **No** – must use `prepare()` | string\|null |
| `$wpdb->get_row( $sql, $output_type )` | Fetch single row | **No** | object\|array\|null |
| `$wpdb->get_col( $sql, $col_offset )` | Fetch single column as array | **No** | array |
| `$wpdb->get_results( $sql, $output_type )` | Fetch multiple rows | **No** | array |
| `$wpdb->query( $sql )` | Execute arbitrary SQL | **No** | int\|bool |
| `$wpdb->insert( $table, $data, $format )` | Insert row | **Yes** – auto-escapes via format | int\|false |
| `$wpdb->update( $table, $data, $where, $format, $where_format )` | Update rows | **Yes** – auto-escapes via format | int\|false |
| `$wpdb->delete( $table, $where, $where_format )` | Delete rows | **Yes** – auto-escapes via format | int\|false |
| `$wpdb->replace( $table, $data, $format )` | Insert or replace row | **Yes** – auto-escapes via format | int\|false |
| `$wpdb->prepare( $sql, ...$args )` | Prepare SQL with placeholders | **Yes** – this IS the escaping | string |
| `$wpdb->esc_like( $text )` | Escape `%` and `_` for LIKE clauses | **Yes (partial)** – must still call `prepare()` after | string |
| `$wpdb->escape()` | **DEPRECATED** | — | — |
| `$wpdb->_real_escape()` | Internal method only | — | string |

**Output types for `get_row()` / `get_results()`:**
- `OBJECT` (default) – stdClass object
- `OBJECT_K` – keyed by first column value
- `ARRAY_A` – associative array
- `ARRAY_N` – numerically indexed array

**Key property:**
- `$wpdb->insert_id` – last AUTO_INCREMENT ID after `insert()` or `replace()`
- `$wpdb->last_error` – last DB error string
- `$wpdb->num_rows` – rows returned by last query
- `$wpdb->rows_affected` – rows affected by last `insert/update/delete`

---

### 1.2 $wpdb->prepare() Patterns

Source: [WordPress Developer Reference](https://developer.wordpress.org/reference/classes/wpdb/), [WordPress VIP Documentation](https://docs.wpvip.com/databases/optimize-queries/database-queries/)

#### Placeholder Types

| Placeholder | Type | Notes |
|-------------|------|-------|
| `%s` | String | Auto-quoted in output |
| `%d` | Integer | |
| `%f` | Float | Locale-unaware (`%F` internally) |
| `%i` | Identifier (table/column name) | Added in WP 6.2; backtick-quoted |
| `%%` | Literal `%` | Escape in LIKE patterns outside of `$wpdb->esc_like()` |

#### CORRECT Usage Patterns

```php
// Basic string/int placeholders
$wpdb->get_var( $wpdb->prepare(
    "SELECT COUNT(*) FROM $wpdb->posts WHERE post_author = %d AND post_status = %s",
    $author_id,
    'publish'
) );

// INSERT (alternative: use $wpdb->insert() instead)
$wpdb->query( $wpdb->prepare(
    "INSERT INTO $wpdb->postmeta (post_id, meta_key, meta_value) VALUES (%d, %s, %s)",
    $post_id, $meta_key, $meta_value
) );

// LIKE clause – must call esc_like() THEN prepare()
$search = 'foo_bar';
$like   = '%' . $wpdb->esc_like( $search ) . '%';
$wpdb->get_results( $wpdb->prepare(
    "SELECT ID FROM $wpdb->posts WHERE post_title LIKE %s",
    $like
) );

// Identifier placeholder (WP 6.2+) – for dynamic table/column names
$wpdb->prepare( "SELECT * FROM %i WHERE %i = %s", $table, $column, $value );

// Array arguments (vsprintf style)
$wpdb->query( $wpdb->prepare(
    "DELETE FROM $wpdb->postmeta WHERE post_id = %d AND meta_key = %s",
    array( $post_id, $meta_key )
) );

// Use $wpdb->insert() for inserts – cleaner, auto-escapes
$wpdb->insert(
    $wpdb->postmeta,
    array( 'post_id' => $post_id, 'meta_key' => $key, 'meta_value' => $value ),
    array( '%d', '%s', '%s' )
);
```

#### INCORRECT / ANTI-PATTERN Usage

```php
// ❌ CRITICAL: Direct variable interpolation – SQL injection
$wpdb->get_results( "SELECT * FROM $wpdb->posts WHERE ID = $post_id" );

// ❌ CRITICAL: String concatenation without prepare()
$wpdb->query( "DELETE FROM wp_options WHERE option_name = '" . $name . "'" );

// ❌ CRITICAL: prepare() called with no placeholders (does nothing useful)
$wpdb->query( $wpdb->prepare( "SELECT * FROM $wpdb->posts", $id ) );
// ^ WP will trigger _doing_it_wrong() notice because no placeholder matched

// ❌ WRONG: Quoting %s manually – breaks prepare() escaping
$wpdb->prepare( "WHERE name = '%s'", $name );  // The quotes are added by prepare(); don't add them

// ❌ WRONG: Using esc_sql() inside a prepare() call (double-escaping)
$wpdb->prepare( "WHERE id = %s", esc_sql( $id ) );

// ❌ CRITICAL: LIKE clause without esc_like() – user can inject wildcards
$wpdb->prepare( "WHERE title LIKE %s", '%' . $search . '%' );

// ❌ WRONG: Passing user-controlled table/column names without %i (pre-6.2 workaround missing)
$wpdb->prepare( "SELECT * FROM $user_table WHERE $user_column = %s", $value );

// ❌ DEPRECATED: $wpdb->escape() – removed in modern WP
$safe = $wpdb->escape( $value );

// ❌ WRONG: Using $wpdb->_real_escape() directly – internal use only
```

---

### 1.3 WP_Query vs Direct SQL

Source: [WordPress VIP Documentation](https://docs.wpvip.com/databases/optimize-queries/database-queries/), [WP_Query Reference](https://developer.wordpress.org/reference/classes/wp_query/)

#### Decision Matrix

| Scenario | Use |
|----------|-----|
| Querying posts, pages, CPTs | `WP_Query` or `get_posts()` |
| Single post by ID/slug | `get_post()`, `get_page_by_path()` |
| List of posts without loop | `get_posts()` |
| Counting posts | `WP_Query` with `fields => 'ids'`, `no_found_rows => false` |
| Taxonomy, meta, date filters | `WP_Query` |
| Need pagination | `WP_Query` (has `found_posts`) |
| Cross-table custom query no WP abstraction | `$wpdb` with `prepare()` |
| Aggregate (SUM, AVG, COUNT) | `$wpdb->get_var()` with `prepare()` |
| Reporting/analytics queries | `$wpdb` with `prepare()` |
| Modifying WordPress's own query | `pre_get_posts` filter, not `query_posts()` |

#### `query_posts()` is ALWAYS WRONG

```php
// ❌ NEVER use query_posts() – corrupts global $wp_query, breaks pagination
query_posts( 'posts_per_page=5' );

// ✅ Correct: use pre_get_posts to modify main query
add_action( 'pre_get_posts', function( $query ) {
    if ( ! is_admin() && $query->is_main_query() && is_home() ) {
        $query->set( 'posts_per_page', 5 );
    }
} );

// ✅ Correct: secondary queries use WP_Query
$my_query = new WP_Query( array( 'post_type' => 'product', 'posts_per_page' => 5 ) );
```

---

### 1.4 SQL Injection Prevention Patterns

Source: [WordPress Developer Reference](https://developer.wordpress.org/reference/classes/wpdb/), [Wordfence SQL Injection Guide](https://www.wordfence.com/blog/2025/08/how-to-find-sql-injection-vulnerabilities-in-wordpress-plugins-and-themes/), [ircmaxell WordPress WPDB disclosure](https://blog.ircmaxell.com/2017/10/disclosure-wordpress-wpdb-sql-injection-technical.html)

**Core Rule**: All untrusted values in SQL must be passed through `$wpdb->prepare()`. "Untrusted" means any value from: `$_GET`, `$_POST`, `$_REQUEST`, `$_COOKIE`, `$_SERVER`, database, external APIs, user meta.

```php
// ✅ All external input goes through prepare()
$user_id = absint( $_GET['user_id'] );  // validate first
$results = $wpdb->get_results(
    $wpdb->prepare( "SELECT * FROM $wpdb->users WHERE ID = %d", $user_id )
);

// ✅ IN clause with dynamic array – build placeholders dynamically
$ids      = array_map( 'absint', $_POST['ids'] );
$placeholders = implode( ',', array_fill( 0, count( $ids ), '%d' ) );
$sql      = $wpdb->prepare( "SELECT * FROM $wpdb->posts WHERE ID IN ($placeholders)", $ids );

// ✅ ORDER BY with safelist (cannot use prepare() for column names pre-6.2)
$allowed_columns = array( 'post_title', 'post_date', 'post_author' );
$orderby = in_array( $_GET['orderby'], $allowed_columns, true ) ? $_GET['orderby'] : 'post_date';
// Then use $orderby directly (it's now trusted from safelist)

// ✅ ORDER BY direction safelist
$order = in_array( strtoupper( $_GET['order'] ), array( 'ASC', 'DESC' ), true ) ? $_GET['order'] : 'DESC';

// ✅ For dynamic table/column names in WP 6.2+
$sql = $wpdb->prepare( "SELECT * FROM %i WHERE %i = %s", $table_name, $col_name, $value );
```

**Never pass user input as the query template to `prepare()`**:
```php
// ❌ CRITICAL: User input in $query position = SQL injection (CVE-2017 style)
$wpdb->prepare( $_GET['query'], $value );
```

---

### 1.5 Common SQL Anti-Patterns

Source: [WordPress VIP Documentation](https://docs.wpvip.com/databases/optimize-queries/database-queries/)

| Anti-Pattern | Description | Negative Score Trigger |
|---|---|---|
| Raw `$_GET`/`$_POST` in SQL | Direct concatenation without `prepare()` | `$wpdb->query("... $_{GET\|POST\|REQUEST\|COOKIE}`)` |
| Missing `esc_like()` | LIKE clause where user input is not run through `esc_like()` before `prepare()` | `LIKE %s` with direct `$_*` variable |
| Unbounded queries | `posts_per_page => -1` or `LIMIT` absent on direct SQL | `-1` in `posts_per_page`; no `LIMIT` in `$wpdb->get_results()` |
| `SELECT *` on large tables | Fetches unnecessary columns | `SELECT \*` in `$wpdb->get_results()` |
| `DISTINCT` / `GROUP BY` without indexes | Causes temp table creation | `DISTINCT` or `GROUP BY` without confirming index |
| Cross-table JOIN with huge datasets | e.g., negating taxonomy with `-cat` | negating taxonomy parameters in WP_Query |
| PHP logic instead of SQL LIMIT | Fetching 1000 rows then using `array_slice()` | Pattern of `get_results()` + `array_slice()` |
| `$wpdb->escape()` (deprecated) | Removed function – does nothing safely | Any usage of `$wpdb->escape()` |
| Hardcoded table names | `wp_posts` instead of `$wpdb->posts` | Regex: `"wp_[a-z_]+"` in SQL strings |
| No error handling | Ignoring `$wpdb->last_error` | Missing error check after critical queries |
| `query_posts()` | Corrupts global query | Any use of `query_posts()` |
| `suppress_filters => true` in WP_Query | Bypasses caching layers and plugin integration | `'suppress_filters' => true` |
| `orderby => 'rand'` | Full table scan on every request | `'orderby' => 'rand'` in WP_Query |

---

## 2. WordPress Security

### 2.1 Sanitization Functions

Source: [WordPress Sanitizing Data](https://developer.wordpress.org/apis/security/sanitizing/)

**Principle**: Sanitize on INPUT (before storing or using). Validate is preferred over sanitize when possible.

| Function | Use When | What It Does |
|---|---|---|
| `sanitize_text_field( $str )` | Single-line text input from users | Strips tags, removes invalid UTF-8, strips octets, removes extra whitespace |
| `sanitize_textarea_field( $str )` | Multi-line textarea input | Like `sanitize_text_field` but preserves newlines |
| `sanitize_email( $email )` | Email address fields | Strips invalid characters; use `is_email()` to validate |
| `sanitize_url( $url )` | URL fields (formerly `esc_url_raw()` for storage) | Strips invalid URL characters |
| `sanitize_key( $key )` | Option names, keys, slugs | Lowercase alphanumeric + dashes/underscores only |
| `sanitize_title( $title )` | Post slug generation | URL-safe string, lowercase |
| `sanitize_title_for_query( $title )` | Before using title in DB query | Query-safe version of `sanitize_title` |
| `sanitize_title_with_dashes( $title )` | Human-readable URL slugs | Replaces spaces with dashes |
| `sanitize_user( $username, $strict )` | Usernames | Strips disallowed characters |
| `sanitize_file_name( $filename )` | File names before saving | Removes dangerous characters |
| `sanitize_html_class( $class )` | CSS class names | A-Z, a-z, 0-9, `-` only |
| `sanitize_hex_color( $color )` | Color pickers | Returns valid hex color or empty string |
| `sanitize_hex_color_no_hash( $color )` | Color without `#` prefix | Hex digits only |
| `sanitize_mime_type( $mime )` | MIME type strings | Strips invalid chars |
| `sanitize_option( $option, $value )` | WordPress options via Settings API | Context-aware per option name |
| `sanitize_meta( $meta_key, $value, $object_type )` | Post/user/term meta | Applies registered sanitize callbacks |
| `sanitize_sql_orderby( $orderby )` | ORDER BY column names from user input | Returns empty string if not safe columns |
| `sanitize_term( $term, $taxonomy )` | Taxonomy term objects | Full term object sanitization |
| `sanitize_term_field( $field, $value, $term_id, $taxonomy, $context )` | Individual term fields | Context-aware field sanitization |
| `wp_kses( $html, $allowed_html )` | Rich HTML with allowed tags | Strip all except explicitly allowed tags/attributes |
| `wp_kses_post( $html )` | Post content rich HTML | Allows all tags permitted in posts |
| `wp_kses_data( $html )` | Comment content | Allows only comment-permitted tags |
| `wp_unslash( $value )` | Any `$_POST`/`$_GET` data | Removes magic-quotes-style slashing (always use before sanitizing) |
| `absint( $value )` | Positive integer IDs | Absolute integer; use for IDs instead of `intval()` |
| `(int)`, `intval()` | Integer values | Cast to integer |

**Key pattern**: Always apply `wp_unslash()` BEFORE sanitize functions on `$_POST`/`$_GET` data:
```php
// ✅ Correct order
$value = sanitize_text_field( wp_unslash( $_POST['field'] ) );

// ❌ Wrong order – slashes may corrupt sanitization
$value = wp_unslash( sanitize_text_field( $_POST['field'] ) );
```

---

### 2.2 Escaping Functions

Source: [WordPress Escaping Data](https://developer.wordpress.org/apis/security/escaping/)

**Principle**: Escape as LATE as possible, at point of output. Never escape early and store escaped data.

| Function | Context | What It Does |
|---|---|---|
| `esc_html( $text )` | HTML element content | Converts `<`, `>`, `&`, `"`, `'` to entities. **Strips all HTML.** |
| `esc_attr( $text )` | HTML attribute values | Same as `esc_html` but for attributes |
| `esc_url( $url )` | `href`, `src`, and all URL attributes | Validates URL scheme, encodes chars |
| `esc_url_raw( $url )` | Storing URL in DB | Like `esc_url` but no HTML encoding (safe for DB, not output) |
| `esc_js( $text )` | Inline JavaScript strings | Escapes for JS context |
| `esc_textarea( $text )` | Inside `<textarea>` tags | Converts entities for textarea context |
| `esc_xml( $text )` | XML output | Escapes for XML context |
| `esc_sql( $text )` | Direct SQL string (rarely needed) | Wraps `mysqli_real_escape_string`; **prefer `prepare()`** |
| `wp_json_encode( $data )` | JSON output in `<script>` blocks | Safe JSON encoding |
| `wp_kses( $html, $allowed )` | Allowing limited HTML in output | Strips disallowed tags/attrs |
| `wp_kses_post( $html )` | Post content output | Allows post-permitted HTML |

**Combined localization + escaping helpers**:

| Function | Equivalent |
|---|---|
| `esc_html__( $text, $domain )` | `esc_html( __( $text, $domain ) )` |
| `esc_html_e( $text, $domain )` | `echo esc_html( __( $text, $domain ) )` |
| `esc_html_x( $text, $ctx, $domain )` | Translatable + HTML-escaped |
| `esc_attr__( $text, $domain )` | `esc_attr( __( $text, $domain ) )` |
| `esc_attr_e( $text, $domain )` | `echo esc_attr( __( $text, $domain ) )` |
| `esc_attr_x( $text, $ctx, $domain )` | Translatable + attr-escaped |

#### Context–Function Mapping (Critical for Scoring)

```php
// ✅ HTML element content
echo '<h1>' . esc_html( $title ) . '</h1>';

// ✅ HTML attribute
echo '<div class="' . esc_attr( $class ) . '">';

// ✅ URL in href/src
echo '<a href="' . esc_url( $url ) . '">';

// ✅ Inline JS
echo '<script>var x = "' . esc_js( $value ) . '";</script>';

// ✅ data-* attribute with JSON
echo '<div data-config="' . esc_attr( wp_json_encode( $config ) ) . '">';

// ✅ textarea
echo '<textarea>' . esc_textarea( $content ) . '</textarea>';

// ✅ Integer output – cast, no esc function needed
echo (int) $count;
echo absint( $count );

// ❌ WRONG: esc_html() on a URL (breaks URL characters)
echo '<a href="' . esc_html( $url ) . '">';

// ❌ WRONG: esc_attr() on a URL (doesn't validate scheme, allows javascript:)
echo '<a href="' . esc_attr( $url ) . '">';

// ❌ WRONG: Double-escaping
$url_escaped = esc_url( $url );
echo '<a href="' . esc_attr( $url_escaped ) . '">';

// ❌ WRONG: No escaping on echo
echo '<div>' . $user_input . '</div>';

// ❌ WRONG: Escaping the whole concatenated attribute separately
echo '<div id="', esc_attr( $prefix ), '-box', esc_attr( $id ), '">';
// ✅ CORRECT: Escape the whole string as one unit
echo '<div id="' . esc_attr( $prefix . '-box' . $id ) . '">';
```

---

### 2.3 Nonce Patterns

Source: [WordPress Nonces API](https://developer.wordpress.org/apis/security/nonces/)

**Principle**: Nonces are CSRF tokens, not authentication. Always combine with `current_user_can()`.

#### Creating Nonces

```php
// For URLs
$url = wp_nonce_url( $bare_url, 'action-name_' . $object_id );

// For forms (outputs hidden field)
wp_nonce_field( 'action-name_' . $post_id );
// Generates: <input type="hidden" name="_wpnonce" value="..." />
//            <input type="hidden" name="_wp_http_referer" value="..." />

// Arbitrary nonce value
$nonce = wp_create_nonce( 'action-name_' . $object_id );
```

#### Verifying Nonces

```php
// Admin page/form handler
check_admin_referer( 'action-name_' . $post_id );
// ^ Dies with 403 on failure; also checks HTTP referer

// AJAX handler
check_ajax_referer( 'action-name' );
// ^ Dies on failure; does NOT check referer

// Manual verification (for custom contexts)
if ( ! wp_verify_nonce( $_POST['my_nonce'], 'action-name_' . $post_id ) ) {
    wp_die( 'Security check failed.' );
}
```

#### Anti-Patterns

```php
// ❌ Missing nonce check on form submission
add_action( 'save_post', function( $post_id ) {
    if ( isset( $_POST['my_field'] ) ) {
        update_post_meta( $post_id, 'my_field', sanitize_text_field( $_POST['my_field'] ) );
        // Missing: wp_verify_nonce() check
    }
} );

// ❌ Missing nonce on AJAX handler
add_action( 'wp_ajax_my_action', function() {
    // Missing: check_ajax_referer( 'my-action' );
    $data = sanitize_text_field( $_POST['data'] );
    // ...
    wp_die();
} );

// ❌ Using nonce as authentication (nonces can be guessed/reused within 24h)
if ( wp_verify_nonce( $_POST['nonce'], 'delete-user' ) ) {
    delete_user( $id ); // Missing: current_user_can() check
}

// ❌ Overly generic nonce action string (low specificity = higher collision risk)
wp_nonce_field( 'my_plugin' ); // Should be: 'my_plugin_action_' . $object_id

// ❌ Missing nonce output in form
// Form with no wp_nonce_field() call

// ❌ Using $_REQUEST instead of $_POST/$_GET (cookie injection risk)
$nonce = $_REQUEST['_wpnonce'];
```

---

### 2.4 Capability Checks

Source: [current_user_can() Reference](https://developer.wordpress.org/reference/functions/current_user_can/), [WordPress Security Handbook](https://developer.wordpress.org/apis/security/)

#### Common Capabilities by Role

| Capability | Minimum Role | Typical Use |
|---|---|---|
| `manage_options` | Administrator | Plugin settings pages |
| `edit_posts` | Editor+ | Post editing features |
| `publish_posts` | Author+ | Publishing |
| `edit_others_posts` | Editor+ | Editing other users' posts |
| `delete_posts` | Editor+ | Post deletion |
| `upload_files` | Author+ | Media uploads |
| `manage_categories` | Editor+ | Taxonomy management |
| `edit_users` | Administrator | User management |
| `install_plugins` | Administrator | Plugin installation |
| `activate_plugins` | Administrator | Plugin activation |
| `unfiltered_html` | Administrator+ | Raw HTML in content |
| `create_users` | Administrator | User creation |
| `list_users` | Administrator | User listing |

#### Correct Patterns

```php
// ✅ Always check capability before processing sensitive actions
add_action( 'admin_post_my_action', function() {
    if ( ! current_user_can( 'manage_options' ) ) {
        wp_die( esc_html__( 'You do not have permission.', 'text-domain' ), 403 );
    }
    check_admin_referer( 'my-action' );
    // ... process
} );

// ✅ Object-level capability check (meta capability)
if ( ! current_user_can( 'edit_post', $post_id ) ) {
    wp_die( 'Forbidden', 403 );
}

// ✅ AJAX handler – always check both capability AND nonce
add_action( 'wp_ajax_update_data', function() {
    check_ajax_referer( 'update-data-nonce' );
    if ( ! current_user_can( 'edit_posts' ) ) {
        wp_send_json_error( array( 'message' => 'Insufficient permissions.' ), 403 );
    }
    // ... process
    wp_die();
} );

// ✅ Check before rendering admin UI
if ( current_user_can( 'manage_options' ) ) {
    add_options_page( ... );
}
```

#### Anti-Patterns

```php
// ❌ No capability check on admin page callback
add_submenu_page( 'options-general.php', 'My Plugin', 'My Plugin', 'manage_options', 'my-plugin', 'my_plugin_page' );
function my_plugin_page() {
    // Missing: if ( ! current_user_can( 'manage_options' ) ) { ... }
    echo '<form>...'; // Dangerous
}

// ❌ Role check instead of capability check
if ( wp_get_current_user()->roles[0] === 'administrator' ) { ... }
// ✅ Should be: current_user_can( 'manage_options' )

// ❌ Missing capability check on nopriv AJAX that performs privileged action
add_action( 'wp_ajax_nopriv_delete_item', function() {
    // Missing capability check – any logged-out user can trigger this
    $wpdb->delete( 'wp_items', array( 'id' => absint( $_POST['id'] ) ) );
    wp_die();
} );

// ❌ Checking is_admin() as a security check (checks context, not permissions)
if ( is_admin() ) {
    // Not a capability check
}
```

---

### 2.5 Data Validation Patterns

Source: [WordPress Data Validation](https://developer.wordpress.org/apis/security/data-validation/)

```php
// ✅ Safelist validation for controlled inputs
$allowed_statuses = array( 'publish', 'draft', 'pending' );
$status = sanitize_key( wp_unslash( $_POST['status'] ) );
if ( ! in_array( $status, $allowed_statuses, true ) ) {
    wp_die( 'Invalid status.' );
}

// ✅ Integer validation
$page = absint( $_GET['paged'] );
if ( $page < 1 ) $page = 1;

// ✅ Email validation
$email = sanitize_email( wp_unslash( $_POST['email'] ) );
if ( ! is_email( $email ) ) {
    wp_die( 'Invalid email.' );
}

// ✅ URL validation
$url = esc_url_raw( wp_unslash( $_POST['url'] ) );
if ( empty( $url ) ) {
    wp_die( 'Invalid URL.' );
}

// ✅ Strict type comparison for safelists
$input = '1 malicious string';
if ( in_array( $input, array( 1, 2, 3 ), true ) ) { ... }
// ^ true enables strict type check; '1 malicious string' !== 1

// ✅ ORDER BY validation (cannot use prepare() for column names)
$allowed = array( 'title', 'date', 'author' );
$orderby = sanitize_key( $_GET['orderby'] ?? 'date' );
$orderby = in_array( $orderby, $allowed, true ) ? $orderby : 'date';
```

---

### 2.6 File Upload Security

Source: [Wordfence file upload vulnerability disclosure](https://www.wordfence.com/blog/2025/07/100000-wordpress-sites-affected-by-arbitrary-file-upload-vulnerability-in-ai-engine-wordpress-plugin/)

#### Core Functions

| Function | Purpose |
|---|---|
| `wp_handle_upload( $file, $overrides )` | Full upload handling (validates type, moves file, returns URL) |
| `wp_check_filetype( $filename, $mimes )` | Check extension against allowed MIME types |
| `wp_check_filetype_and_ext( $file, $filename, $mimes )` | Check BOTH extension AND real file content type |
| `wp_get_mime_types()` | Get list of all allowed MIME types in WordPress |
| `wp_upload_dir()` | Get upload directory path and URL |

#### Correct Patterns

```php
// ✅ Use wp_handle_upload() for form uploads – handles all security checks
$upload = wp_handle_upload( $_FILES['my_file'], array( 'test_form' => false ) );
if ( isset( $upload['error'] ) ) {
    wp_die( esc_html( $upload['error'] ) );
}
$file_url = $upload['url'];

// ✅ Validate file type explicitly
$allowed_types = array( 'image/jpeg', 'image/png', 'image/gif' );
$file_info = wp_check_filetype_and_ext( $_FILES['file']['tmp_name'], $_FILES['file']['name'] );
if ( ! in_array( $file_info['type'], $allowed_types, true ) ) {
    wp_die( 'File type not allowed.' );
}

// ✅ Use nonce + capability check before processing upload
if ( ! current_user_can( 'upload_files' ) ) {
    wp_die( 'Insufficient permissions.' );
}
check_admin_referer( 'upload-nonce' );
```

#### Anti-Patterns

```php
// ❌ Checking extension only (bypass: upload file.php.jpg)
$ext = pathinfo( $_FILES['file']['name'], PATHINFO_EXTENSION );
if ( $ext === 'php' ) { die(); }  // Trivially bypassed

// ❌ Trusting $_FILES['type'] (set by browser, not validated)
if ( $_FILES['file']['type'] !== 'image/jpeg' ) { die(); }

// ❌ Not using wp_check_filetype_and_ext() to check real MIME type
move_uploaded_file( $_FILES['file']['tmp_name'], $upload_dir . $_FILES['file']['name'] );

// ❌ Uploading to web-accessible directory without type check
$upload_path = ABSPATH . 'wp-content/uploads/' . $_FILES['file']['name'];
move_uploaded_file( $_FILES['file']['tmp_name'], $upload_path );

// ❌ No capability check before upload processing
add_action( 'wp_ajax_handle_upload', function() {
    // Missing: current_user_can( 'upload_files' ) check
    $upload = wp_handle_upload( $_FILES['file'], array() );
} );
```

---

### 2.7 CSRF / XSS / SQLi Prevention Summary

| Attack | Prevention Method | Anti-Pattern to Detect |
|---|---|---|
| **CSRF** | `wp_nonce_field()` + `check_admin_referer()` / `check_ajax_referer()` | Form handlers without nonce check |
| **Reflected XSS** | `esc_html()`, `esc_attr()`, `esc_url()`, `esc_js()` at output | `echo $_GET[...]` without escaping |
| **Stored XSS** | `wp_kses_post()` or `sanitize_text_field()` on save; escape on output | Storing raw HTML input; outputting stored data without esc_* |
| **SQL Injection** | `$wpdb->prepare()` with typed placeholders | Concatenation of `$_*` variables into SQL strings |
| **LIKE injection** | `$wpdb->esc_like()` before `prepare()` | `LIKE %s` with raw user input |
| **Path traversal** | `validate_file()`, `realpath()`, avoid user-controlled paths | `include $_GET['file']`, `file_get_contents( $user_path )` |
| **PHP object injection** | Avoid `unserialize()` on untrusted data | `unserialize( $_POST[...] )` or `unserialize( get_option(...) )` when option is user-writable |
| **Open redirect** | `wp_safe_redirect()` (limits to local), `wp_validate_redirect()` | `wp_redirect( $_GET['url'] )` |
| **Privilege escalation** | `current_user_can()` before any privileged action | AJAX handlers without capability check |

---

### 2.8 AJAX Security Patterns

Source: [WordPress Plugin Handbook – AJAX](https://developer.wordpress.org/plugins/javascript/enqueuing/)

```php
// ✅ Complete secure AJAX handler pattern
add_action( 'wp_ajax_my_action', 'my_ajax_handler' );
// add_action( 'wp_ajax_nopriv_my_action', 'my_ajax_handler' ); // Only if guests need this

function my_ajax_handler() {
    // 1. Verify nonce FIRST
    check_ajax_referer( 'my-action-nonce', 'nonce' );

    // 2. Check capability
    if ( ! current_user_can( 'edit_posts' ) ) {
        wp_send_json_error( array( 'message' => 'Forbidden' ), 403 );
    }

    // 3. Sanitize inputs
    $data = sanitize_text_field( wp_unslash( $_POST['data'] ?? '' ) );

    // 4. Process & respond
    wp_send_json_success( array( 'result' => $data ) );

    // 5. Always die() at end (wp_send_json_* handles this)
}

// ✅ Localize nonce to JavaScript
wp_localize_script( 'my-script', 'myAjax', array(
    'ajax_url' => admin_url( 'admin-ajax.php' ),
    'nonce'    => wp_create_nonce( 'my-action-nonce' ),
) );
```

**AJAX Anti-Patterns**:

```php
// ❌ Missing nonce check
add_action( 'wp_ajax_delete_item', function() {
    $id = absint( $_POST['id'] );
    $wpdb->delete( 'my_table', array( 'id' => $id ) );
    wp_die();
} );

// ❌ Sending response without wp_die() (output buffering issues)
add_action( 'wp_ajax_my_action', function() {
    echo json_encode( array( 'ok' => true ) );
    // Missing: wp_die();
} );

// ❌ Using $_REQUEST (cookie injection)
$nonce = $_REQUEST['nonce']; // Should use $_POST['nonce']

// ❌ nopriv handler performing privileged operation without capability check
add_action( 'wp_ajax_nopriv_submit_form', function() {
    // Any logged-out user can trigger this with no restriction
    wp_insert_post( array( 'post_status' => 'publish', ... ) );
    wp_die();
} );
```

---

## 3. WordPress Performance Patterns

### 3.1 Object Caching & Transients

Source: [WordPress Transients API](https://developer.wordpress.org/news/2024/06/an-introduction-to-the-transients-api/), [WordPress VIP Autoloaded Options](https://docs.wpvip.com/wordpress-on-vip/autoloaded-options/), [Pressable Object Cache Guide](https://pressable.com/knowledgebase/using-wordpress-object-cache-for-query-results/)

#### Object Cache API (`wp_cache_*`)

```php
// ✅ Standard cache read-through pattern
function get_expensive_data( $key ) {
    $cached = wp_cache_get( $key, 'my-plugin-group' );
    if ( false !== $cached ) {
        return $cached;
    }
    $data = /* expensive computation or query */;
    wp_cache_set( $key, $data, 'my-plugin-group', HOUR_IN_SECONDS );
    return $data;
}

// wp_cache_* functions
wp_cache_get( $key, $group, $force, &$found );
wp_cache_set( $key, $data, $group, $expire );  // $expire = 0 = non-expiring (within request)
wp_cache_add( $key, $data, $group, $expire );  // Only sets if key doesn't exist
wp_cache_delete( $key, $group );
wp_cache_flush();                              // Flush entire cache (avoid in production)
wp_cache_flush_group( $group );               // Flush specific group (WP 6.1+)
wp_cache_replace( $key, $data, $group, $expire ); // Only updates existing key
wp_cache_incr( $key, $offset, $group );
wp_cache_decr( $key, $offset, $group );
```

#### Transients API (Persistent Cache)

```php
// ✅ Correct transient pattern with expiration
function get_remote_data() {
    $data = get_transient( 'my_plugin_remote_data' );
    if ( false === $data ) {
        $response = wp_remote_get( 'https://api.example.com/data' );
        if ( ! is_wp_error( $response ) ) {
            $data = wp_remote_retrieve_body( $response );
            set_transient( 'my_plugin_remote_data', $data, HOUR_IN_SECONDS );
        }
    }
    return $data;
}

// ✅ Invalidate on relevant events (cache priming)
add_action( 'save_post', function( $post_id ) {
    delete_transient( 'my_plugin_post_list' );
} );
```

**Anti-Patterns**:

```php
// ❌ Transient without expiration – stored as autoloaded option forever
set_transient( 'my_data', $data ); // Missing expiration = autoloaded = performance hit

// ❌ Not checking for false before using cached value
$data = get_transient( 'key' );
process( $data ); // $data could be false (expired/unset)

// ❌ Using wp_cache_flush() in production hooks
add_action( 'save_post', function() {
    wp_cache_flush(); // Flushes ALL cache for all users
} );

// ❌ Caching too granularly (defeats purpose)
foreach ( $posts as $post ) {
    wp_cache_set( 'post_' . $post->ID, $post ); // Should cache the whole collection
}

// ❌ No caching on expensive repeated queries
function get_all_products() {
    return $wpdb->get_results( "SELECT * FROM $wpdb->posts WHERE post_type = 'product'" );
    // Called 10 times per request = 10 queries
}
```

---

### 3.2 WP_Query Optimization Arguments

Source: [WordPress VIP WP_Query Performance](https://wpvip.com/blog/wp-query-performance/), [Spacedmonkey WP_Query Performance](https://www.spacedmonkey.com/2025/04/14/enhancing-wp_query-performance-in-wordpress/)

| Argument | Default | Optimization | When to Apply |
|---|---|---|---|
| `posts_per_page` | 10 | Set to exact count needed; never `-1` | Any query with known count |
| `no_found_rows` | false | Set `true` | Non-paginated queries (widgets, feeds, API) |
| `fields` | `''` (all) | `'ids'` or `'id=>parent'` | When only IDs needed |
| `update_post_meta_cache` | true | Set `false` | When post meta not accessed in loop |
| `update_post_term_cache` | true | Set `false` | When taxonomy terms not needed |
| `update_menu_item_cache` | false | Leave false unless querying nav_menu_item | Only for nav menus |
| `ignore_sticky_posts` | false | Set `true` | Non-main queries (widgets, API responses) |
| `cache_results` | true | Keep `true` | Generally never disable |
| `suppress_filters` | false | Keep `false` | Never set true – breaks caching & plugins |
| `orderby` | `'date'` | Avoid `'rand'` | `'rand'` causes full table scan |

```php
// ✅ Optimized non-paginated query (e.g., sidebar widget)
$query = new WP_Query( array(
    'post_type'              => 'post',
    'posts_per_page'         => 5,
    'no_found_rows'          => true,    // Skip SQL_CALC_FOUND_ROWS
    'ignore_sticky_posts'    => true,    // Skip sticky post logic
    'update_post_meta_cache' => false,   // Skip meta preload if not needed
    'update_post_term_cache' => false,   // Skip term preload if not needed
    'fields'                 => 'ids',   // Only return IDs
) );

// ✅ Getting only IDs for a bulk operation
$post_ids = get_posts( array(
    'post_type'      => 'product',
    'posts_per_page' => 100,
    'fields'         => 'ids',
    'no_found_rows'  => true,
) );

// ❌ Fetching all posts – unbounded query
$all_posts = get_posts( array( 'posts_per_page' => -1 ) );

// ❌ Random order – full table scan every request
$posts = new WP_Query( array( 'orderby' => 'rand', 'posts_per_page' => 5 ) );

// ❌ Suppress filters breaks caching and plugin integration
$query = new WP_Query( array( 'suppress_filters' => true ) );
```

#### meta_query Performance

```php
// ❌ meta_query on non-indexed key with LIKE (very slow on large tables)
$query = new WP_Query( array(
    'meta_query' => array(
        array( 'key' => 'product_description', 'value' => $search, 'compare' => 'LIKE' )
    )
) );

// ✅ meta_query with EXISTS check (better for sparse meta)
$query = new WP_Query( array(
    'meta_query' => array(
        array( 'key' => 'featured', 'compare' => 'EXISTS' )
    )
) );
```

---

### 3.3 N+1 Query Detection Patterns

**N+1 pattern**: Fetching a list, then querying for related data on each item in a loop.

```php
// ❌ N+1: 1 query to get posts + N queries for meta
$posts = get_posts( array( 'post_type' => 'product', 'posts_per_page' => 50 ) );
foreach ( $posts as $post ) {
    $price = get_post_meta( $post->ID, '_price', true ); // 50 separate queries!
    echo $price;
}

// ✅ Fix: WP_Query pre-loads meta cache automatically when update_post_meta_cache=true
$query = new WP_Query( array(
    'post_type'              => 'product',
    'posts_per_page'         => 50,
    'update_post_meta_cache' => true,  // Default – preloads all meta in ONE query
) );
while ( $query->have_posts() ) {
    $query->the_post();
    $price = get_post_meta( get_the_ID(), '_price', true ); // Uses cache – no query
}

// ❌ N+1: Fetching term names in a loop
$posts = get_posts( array( 'post_type' => 'product', 'posts_per_page' => 50 ) );
foreach ( $posts as $post ) {
    $categories = get_the_terms( $post->ID, 'category' ); // 50 queries!
}

// ✅ Fix: use update_post_term_cache=true (default) with WP_Query
$query = new WP_Query( array(
    'post_type'              => 'product',
    'posts_per_page'         => 50,
    'update_post_term_cache' => true,  // Default – preloads all terms in ONE query
) );
// get_the_terms() inside loop uses cache

// ❌ N+1: get_user_by() inside a post loop
foreach ( $posts as $post ) {
    $author = get_user_by( 'id', $post->post_author ); // N queries
}
// ✅ Fix: Use update_post_author_cache (WP 6.0+) or cache_users() before loop
```

**Detection Pattern**: Look for WP/database functions inside `foreach`/`while` loops over post arrays. Specifically flag:
- `get_post_meta()` inside loop when `update_post_meta_cache` is false
- `get_the_terms()`, `get_the_category()` inside loop when `update_post_term_cache` is false
- `get_user_by()`, `get_userdata()` inside loop
- `$wpdb->get_*()` inside loop
- `get_option()` inside loop (use wp_load_alloptions() or cache beforehand)

---

### 3.4 Autoload Option Management

Source: [WordPress VIP Autoloaded Options](https://docs.wpvip.com/wordpress-on-vip/autoloaded-options/), [Pressable Autoloaded Data Guide](https://pressable.com/knowledgebase/speed-up-your-wordpress-site-by-optimizing-autoloaded-data/)

**Problem**: Every page load fetches ALL autoloaded options into memory. Excessive autoloaded data (>1MB on VIP) causes performance degradation and 503 errors.

```php
// ❌ Adding large data as autoloaded option (default behavior)
add_option( 'my_plugin_cache', $large_array );  // Autoloads by default
update_option( 'my_plugin_cache', $large_array ); // Also autoloads by default (before WP 6.6)

// ✅ Explicitly set autoload=false for data not needed on every page
add_option( 'my_plugin_cache', $large_array, '', false ); // WP < 6.6: 4th param is deprecated_message, use update_option
update_option( 'my_plugin_cache', $large_array, false );  // WP 6.6+: $autoload parameter

// ✅ WP 6.6+ explicit autoload=false
add_option( 'my_large_option', $data, autoload: false );
update_option( 'my_large_option', $data, autoload: false );

// ✅ Only autoload options needed on every single page request
// Bad examples of autoloaded data: large serialized arrays, report caches, per-post data

// ✅ For frequently changed or large data, use transients (with expiry to avoid autoload)
set_transient( 'my_plugin_data', $large_data, DAY_IN_SECONDS );
```

**Scoring signals**:
- `add_option( ... )` or `update_option( ... )` without explicit `false` for autoload on non-essential options
- Storing serialized arrays >10KB in options
- `set_transient( $key, $data )` without expiration time
- Multiple `get_option()` calls that could be batched with `wp_load_alloptions()`

---

### 3.5 Remote HTTP Request Patterns

Source: [WordPress HTTP API](https://developer.wordpress.org/plugins/http-api/)

```php
// ✅ Basic GET with timeout
$response = wp_remote_get( $url, array( 'timeout' => 10 ) );

// ✅ Always check for WP_Error
if ( is_wp_error( $response ) ) {
    error_log( 'Request failed: ' . $response->get_error_message() );
    return false;
}

// ✅ Check HTTP status code
$code = wp_remote_retrieve_response_code( $response );
if ( 200 !== $code ) {
    return false;
}

// ✅ Cache remote responses with transients
function get_external_data( $url ) {
    $cache_key = 'ext_' . md5( $url );
    $cached = get_transient( $cache_key );
    if ( false !== $cached ) {
        return $cached;
    }
    $response = wp_remote_get( $url, array( 'timeout' => 10 ) );
    if ( is_wp_error( $response ) || 200 !== wp_remote_retrieve_response_code( $response ) ) {
        return false;
    }
    $data = json_decode( wp_remote_retrieve_body( $response ), true );
    set_transient( $cache_key, $data, HOUR_IN_SECONDS );
    return $data;
}

// ✅ POST request
$response = wp_remote_post( $url, array(
    'body'    => array( 'key' => 'value' ),
    'timeout' => 15,
    'headers' => array( 'Authorization' => 'Bearer ' . $token ),
) );
```

**Anti-Patterns**:

```php
// ❌ No timeout set (default is 5s – may be insufficient or too long)
$response = wp_remote_get( $url );

// ❌ No WP_Error check
$body = wp_remote_retrieve_body( wp_remote_get( $url ) );
// ^ If request fails, this returns empty string silently

// ❌ Making HTTP request on every page load without caching
add_action( 'wp_head', function() {
    $data = wp_remote_get( 'https://api.example.com/data' );
    // Called on EVERY page load – should be cached
} );

// ❌ Blocking request on non-critical path
$response = wp_remote_get( $url, array( 'blocking' => true ) ); // blocks page render

// ❌ Using PHP's curl_* or file_get_contents() for HTTP (bypasses WP filters/proxy)
$data = file_get_contents( $url );
curl_exec( $ch );
```

---

### 3.6 Script/Style Enqueueing

Source: [WordPress Plugin Handbook – Enqueuing](https://developer.wordpress.org/plugins/javascript/enqueuing/)

```php
// ✅ Always enqueue scripts via hook, never directly
add_action( 'wp_enqueue_scripts', 'my_plugin_enqueue_scripts' );
function my_plugin_enqueue_scripts() {
    wp_enqueue_script(
        'my-plugin-script',
        plugins_url( '/js/main.js', __FILE__ ),
        array( 'jquery' ),
        '1.0.0',
        array( 'in_footer' => true )  // Load in footer
    );
    wp_enqueue_style(
        'my-plugin-style',
        plugins_url( '/css/main.css', __FILE__ ),
        array(),
        '1.0.0'
    );
}

// ✅ Conditional loading – only enqueue where needed
add_action( 'admin_enqueue_scripts', function( $hook ) {
    if ( 'my-plugin_page_settings' !== $hook ) {
        return;
    }
    wp_enqueue_script( 'my-settings-script', ... );
} );

// ✅ Use wp_localize_script() to pass PHP data to JS safely
wp_localize_script( 'my-plugin-script', 'myPluginData', array(
    'ajaxUrl' => admin_url( 'admin-ajax.php' ),
    'nonce'   => wp_create_nonce( 'my-plugin-nonce' ),
) );

// ✅ Deferred loading strategy (WP 6.3+)
wp_enqueue_script( 'my-script', $url, array(), '1.0', array(
    'strategy'  => 'defer',
    'in_footer' => true,
) );
```

**Anti-Patterns**:

```php
// ❌ Hardcoded script tags in HTML
echo '<script src="/wp-content/plugins/my-plugin/js/main.js"></script>';

// ❌ Enqueuing outside of proper hook
my_plugin_enqueue_scripts(); // Called directly, not via hook

// ❌ Loading scripts on all pages when only needed on specific pages
add_action( 'wp_enqueue_scripts', function() {
    wp_enqueue_script( 'heavy-library', $url, array(), '1.0', true );
    // No conditional check – loaded on every page
} );

// ❌ Using hardcoded URL instead of plugins_url() / get_template_directory_uri()
wp_enqueue_script( 'script', '/wp-content/plugins/my-plugin/js/script.js' );

// ❌ Registering the same script multiple times
wp_register_script( 'jquery', ... ); // Overrides WordPress's bundled jQuery

// ❌ Inline <script> in templates without wp_add_inline_script()
echo '<script>var data = ' . json_encode( $data ) . ';</script>';
// ✅ Use: wp_add_inline_script( 'handle', $js_code );
```

---

### 3.7 Database Query Optimization

Source: [WordPress VIP Database Queries](https://docs.wpvip.com/databases/optimize-queries/database-queries/), [Spacedmonkey WP_Query Performance](https://www.spacedmonkey.com/2025/04/14/enhancing-wp_query-performance-in-wordpress/)

```php
// ✅ EXPLAIN queries during development
$wpdb->get_results( "EXPLAIN SELECT * FROM $wpdb->posts WHERE post_status = 'publish'" );

// ✅ Use indexes – meta_query with indexed keys
// wp_postmeta has an index on meta_key; ensure custom tables have appropriate indexes

// ✅ Avoid cross-table negative queries
// ❌ Slow: -cat in WP_Query (large temporary tables)
$q = new WP_Query( array( 'cat' => '-5', 'posts_per_page' => 20 ) );

// ✅ Avoid DISTINCT/GROUP BY that create temp tables
// ❌ Avoid:
$wpdb->get_results( "SELECT DISTINCT post_author FROM $wpdb->posts" );

// ✅ Keep calculations in PHP, not DB
// ❌ DB-side:
$wpdb->get_var( "SELECT COUNT(*) * 2 FROM $wpdb->posts" );
// ✅ PHP-side:
$count = $wpdb->get_var( "SELECT COUNT(*) FROM $wpdb->posts" ) * 2;

// ✅ Use LIMIT on all direct SQL queries
$wpdb->get_results( $wpdb->prepare(
    "SELECT ID FROM $wpdb->posts WHERE post_type = %s LIMIT %d",
    'post', 100
) );

// ✅ Prefer $wpdb->posts, $wpdb->options etc. over hardcoded table names
// ❌ $wpdb->query( "SELECT * FROM wp_posts" );
// ✅ $wpdb->query( "SELECT * FROM $wpdb->posts" );
```

---

## 4. Static Analysis Tools

### 4.1 PHP_CodeSniffer + WordPressCS

Source: [WordPress Coding Standards GitHub](https://github.com/wordpress/wordpress-coding-standards), [Make WordPress Core – WordPressCS 3.0](https://make.wordpress.org/core/2023/08/21/wordpresscs-3-0-0-is-now-available/)

**Package**: `squizlabs/php_codesniffer` + `wp-coding-standards/wpcs`  
**Ruleset names**: `WordPress`, `WordPress-Core`, `WordPress-Docs`, `WordPress-Extra`

#### Key Security/Quality Sniffs (WordPressCS)

| Sniff | Category | What It Flags |
|---|---|---|
| `WordPress.Security.EscapeOutput` | XSS | `echo`/`print` of variables without escaping function |
| `WordPress.Security.NonceVerification` | CSRF | Processing `$_POST`/`$_GET` without nonce check |
| `WordPress.Security.ValidatedSanitizedInput` | Input | Using superglobal input without `sanitize_*` |
| `WordPress.DB.PreparedSQL` | SQLi | `$wpdb->query/get_*` with unescaped input |
| `WordPress.DB.PreparedSQLPlaceholders` | SQLi | Incorrect placeholder usage in `prepare()` |
| `WordPress.DB.DirectDatabaseQuery` | DB | Direct `$wpdb` calls where WP API exists |
| `WordPress.DB.SlowDBQuery` | Performance | `meta_value` queries, `LIKE %value`, `posts_per_page=-1` |
| `WordPress.WP.EnqueuedResources` | Assets | Hardcoded `<script>`/`<link>` in PHP |
| `WordPress.WP.DiscouragedFunctions` | Deprecated | `query_posts()`, `get_page_by_title()`, deprecated functions |
| `WordPress.WP.DiscouragedPHPFunctions` | Security | `eval()`, `base64_decode()`, `unserialize()`, `extract()`, `create_function()` |
| `WordPress.PHP.DiscouragedPHPFunctions` | Security | As above |
| `WordPress.PHP.StrictInArray` | Logic | `in_array()` without `true` strict parameter |
| `WordPress.PHP.StrictComparisons` | Logic | `==` instead of `===` in comparisons |
| `WordPress.Security.PluginMenuSlug` | Security | Non-unique menu slugs |
| `WordPress.WP.Capabilities` | Auth | Custom/misspelled capabilities |

**Installation & usage**:
```bash
composer require --dev squizlabs/php_codesniffer wp-coding-standards/wpcs
./vendor/bin/phpcs --standard=WordPress path/to/plugin
```

#### WordPress VIP Standard (WordPress-VIP-Go)

Source: [WordPress VIP PHPCS Analysis](https://docs.wpvip.com/vip-code-analysis-bot/phpcs-analysis/)

Additional sniffs on top of WordPressCS:
- Stricter caching requirements (direct `$wpdb` queries without caching flagged as warning)
- Remote HTTP calls without caching
- `sleep()` or `usleep()` in production code
- `var_dump()`, `print_r()`, `error_log()` left in code
- Direct file system operations without WP Filesystem API

---

### 4.2 PHPStan for WordPress

Source: [szepeviktor/phpstan-wordpress GitHub](https://github.com/szepeviktor/phpstan-wordpress), [Pascal Birchler PHPStan Guide](https://pascalbirchler.com/phpstan-wordpress/), [Make WordPress Core PHPStan Proposal](https://make.wordpress.org/core/2025/07/11/proposal-phpstan-in-the-wordpress-core-development-workflow/)

**Packages**:
- `phpstan/phpstan` – core tool
- `szepeviktor/phpstan-wordpress` – WordPress extension (includes stubs)
- `php-stubs/wordpress-stubs` – Type declarations for WP core functions
- `phpstan/phpstan-strict-rules` – Extra strict rules
- `phpstan/phpstan-phpunit` – PHPUnit assertions

```bash
composer require --dev szepeviktor/phpstan-wordpress phpstan/extension-installer
```

**Rule Levels**: 0 (minimal) to 9 (strictest). Recommended starting point: Level 5, then increase incrementally.

#### What PHPStan Catches for WordPress

| Category | Examples |
|---|---|
| Type errors | Passing string where WP function expects int (`post_id` parameter) |
| Return type mismatches | Assuming `get_post()` returns `WP_Post` when it can return `null` |
| Undefined variables | `$wpdb` used without `global $wpdb` declaration |
| `is_wp_error()` guard missing | Using return value of function that can return `WP_Error` without checking |
| `apply_filters()` type consistency | Filter return type inconsistent with first `@param` in docblock |
| Dead code | Code after `wp_die()`, `exit`, unreachable branches |
| Deprecated function calls | Functions marked `@deprecated` in WP stubs |
| Wrong argument count | Too many/few args to WP functions |

**PHPStan phpstan.neon configuration for WordPress**:
```yaml
parameters:
    level: 5
    paths:
        - src/
    bootstrapFiles:
        - vendor/php-stubs/wordpress-stubs/wordpress-stubs.php
```

---

### 4.3 Psalm for WordPress

Source: [Psalm Plugins Directory](https://psalm.dev/plugins), [humanmade/psalm-plugin-wordpress GitHub], [Psalm Taint Analysis](https://psalm.dev/docs/security_analysis/custom_taint_sinks/)

**Packages**:
- `vimeo/psalm` – core tool
- `humanmade/psalm-plugin-wordpress` – WordPress stubs and taint sinks
- `psalm-taint-sink` annotations for custom WordPress sinks

```bash
composer require --dev vimeo/psalm humanmade/psalm-plugin-wordpress
./vendor/bin/psalm-plugin enable humanmade/psalm-plugin-wordpress
```

#### Psalm Taint Analysis for WordPress

Psalm's taint analysis traces data flow from **sources** (user input) to **sinks** (dangerous operations) and reports paths where tainted data reaches a sink without sanitization.

**Default sources**: `$_GET`, `$_POST`, `$_COOKIE`, `$_REQUEST`, `$_SERVER['HTTP_*']`, `$_FILES`

**Default sinks**: SQL execution, HTML output, shell commands, file operations, `unserialize()`

**Custom WordPress sinks** (via psalm-plugin-wordpress or custom stubs):
```php
// Annotating $wpdb->query as a SQL taint sink
/**
 * @psalm-taint-sink sql $query
 */
function wpdb_query( string $query ) {}
```

**What Psalm catches**:
- Tainted data reaching `echo`/`print` without `esc_html()`/`esc_attr()` etc.
- Tainted data in SQL queries without `$wpdb->prepare()`
- Tainted data in `unserialize()`
- Tainted data in file operations (`file_get_contents()`, `include`, `require`)
- `== 0` instead of `=== false` for `WP_Error` checks (type issues)

---

### 4.4 Phan for PHP

Source: [Phan GitHub](https://github.com/phan/phan), [Phan WordPress Setup](https://www.youtube.com/watch?v=Ysuny5Zkyx8)

**Packages**:
- `phan/phan` – core tool  
- `szepeviktor/phpstan-wordpress` stubs work as Phan stubs too  
- `phan/phan` + `phan-wordpress` (community stub package)

**Philosophy**: Minimize false positives; prefers proving code is WRONG vs. proving it's correct.

**What Phan catches**:
- Undefined functions/methods/variables
- Type incompatibilities
- Incorrect argument types
- Return type violations
- Unused code and imports
- Potential null dereferences
- Access to undefined properties

**Phan config for WordPress** (`phan.config.php`):
```php
return [
    'target_php_version' => '8.0',
    'directory_list'     => [ 'src/', 'vendor/php-stubs/wordpress-stubs' ],
    'exclude_analysis_directory_list' => [ 'vendor/' ],
    'plugins' => [ 'AlwaysReturnPlugin', 'DuplicateArrayKeyPlugin', 'PregRegexCheckerPlugin' ],
];
```

---

### 4.5 phpcs-security-audit

Source: [FloeDesignTechnologies/phpcs-security-audit GitHub](https://github.com/FloeDesignTechnologies/phpcs-security-audit)

**Package**: `pheromone/phpcs-security-audit`

A PHPCS ruleset specifically for security vulnerabilities. Not WordPress-specific but highly applicable.

#### Key Sniffs

| Sniff | What It Flags |
|---|---|
| `Security.BadFunctions.EasyXSS` | `echo`/`print` with direct `$_*` superglobal input |
| `Security.BadFunctions.PregReplace` | `preg_replace()` with `/e` modifier (RCE) |
| `Security.BadFunctions.SQLFunctions` | Raw `mysql_query()`, `mysqli_query()` with user input |
| `Security.BadFunctions.FilesystemFunctions` | `include`/`require` with user input (LFI) |
| `Security.BadFunctions.CallbackFunctions` | `call_user_func()` with user-controlled name |
| `Security.BadFunctions.SystemExecFunctions` | `exec()`, `shell_exec()`, `system()`, `passthru()` with user input |
| `Security.BadFunctions.PHPInternalFunctions` | `eval()`, `unserialize()`, `base64_decode()` |
| `Security.BadFunctions.EasyRFI` | Remote file inclusion via user input |
| `Security.BadFunctions.CurlFunctions` | `curl_setopt( CURLOPT_URL, $user_input )` (SSRF) |
| `Security.Misc.BadPHPFunctions` | `extract()`, `compact()` with user data |
| `Security.CVE.20131899` | Specific CVE patterns |

---

### 4.6 SonarQube PHP Rules

Source: [SonarSource PHP Vulnerability Rules](https://rules.sonarsource.com/php/type/vulnerability/)

**Notable WordPress-relevant SonarQube rules**:

| Rule | Description |
|---|---|
| `S6345` | WordPress external HTTP requests should be security-sensitive |
| `S2631` | RegEx should not be vulnerable to Denial of Service (ReDoS) |
| `S4784` | Wildcard file inclusion (RFI risk) |
| `S2076` | OS commands should not be vulnerable to command injection |
| `S3649` | DB queries should not be vulnerable to injection |
| `S2083` | File paths should not be injectable |
| `S5334` | Dynamic code execution should not be vulnerable to code injection |
| `S2245` | Using pseudorandom number generators (PRNGs) is security-sensitive |
| `S5131` | Disabling WordPress admin file editing (`DISALLOW_FILE_EDIT`) |

**Integration**: SonarQube Community Edition supports PHP with SAST (Security Application Security Testing) rules via the `sonar-php-plugin`. Runs OWASP Top 10 checks, SANS Top 25, and CWE mappings.

---

### 4.7 WordPress-Specific Tools

#### WordPress Plugin Check (PCP)

Source: [Make WordPress Plugins – PCP Update](https://make.wordpress.org/plugins/2025/10/29/plugin-check-plugin-now-creates-automatic-security-reports-update/)

- **Official tool** from the WordPress Plugins Team
- Runs on all plugin submissions and updates to WordPress.org since October 2024
- Checks: nonce usage, sanitization, escaping, capability checks, deprecated functions, direct DB queries without caching, enqueued resources
- Available as a standalone plugin: `wordpress/plugin-check`
- Combines PHPCS WordPressCS rules with dynamic checks

```bash
# Run locally
wp plugin check my-plugin
```

#### WPScan (Vulnerability Database)

- Vulnerability database used by WordPress VIP
- Identifies plugins/themes with known CVEs
- API: `https://wpscan.com/api`

#### Query Monitor (Development Tool)

- Runtime query profiling
- Shows duplicate queries, slow queries, N+1 patterns
- Ideal for catching performance issues PHPCS can't detect

#### WordPress VIP Code Analysis Bot

Source: [WordPress VIP PHPCS Analysis](https://docs.wpvip.com/vip-code-analysis-bot/phpcs-analysis/)

- Runs `WordPress-VIP-Go` PHPCS standard on all PRs
- Includes: `PHPCompatibilityWP` standard
- Flags: slow queries, missing caching, remote HTTP without caching, direct DB access, `sleep()`, debug output

---

## 5. Scoring Criteria Summary Tables

### 5.1 SQL Safety – Automatable Scoring Patterns

| Pattern | Score | Detection Regex/Method |
|---|---|---|
| `$wpdb->prepare()` used with user input | +2 | `prepare\s*\(.*\$_(GET\|POST\|REQUEST\|COOKIE)` |
| `$wpdb->insert/update/delete` used instead of raw query | +1 | `\$wpdb->(insert\|update\|delete)\s*\(` |
| `esc_like()` before LIKE in prepare() | +2 | `esc_like.*prepare\|prepare.*LIKE` |
| `%i` identifier placeholder used (WP 6.2+) | +1 | `prepare.*%i` |
| Direct variable interpolation in SQL string | -3 | `".*\$(_(GET\|POST\|REQUEST\|COOKIE)\|[a-z_]+).*"` passed to `$wpdb->` |
| String concatenation in SQL query | -2 | `".*\.\s*\$.*"` in wpdb query argument |
| `LIKE %s` without esc_like() | -2 | `LIKE %s` with direct `$` variable |
| Missing LIMIT on direct SQL | -1 | `get_results\("SELECT` without `LIMIT` |
| `posts_per_page => -1` | -1 | `'posts_per_page'\s*=>\s*-1` |
| Hardcoded table name `wp_*` | -1 | `".*wp_[a-z_]+.*"` in SQL strings |
| `$wpdb->escape()` (deprecated) | -2 | `->escape\s*\(` |
| `query_posts()` | -2 | `query_posts\s*\(` |
| `orderby => 'rand'` | -1 | `'orderby'\s*=>\s*'rand'` |
| `suppress_filters => true` | -1 | `'suppress_filters'\s*=>\s*true` |

### 5.2 Security – Automatable Scoring Patterns

| Pattern | Score | Detection |
|---|---|---|
| `sanitize_*()` on all `$_POST/$_GET` input | +2 | PHPCS `ValidatedSanitizedInput` |
| `wp_unslash()` before sanitize | +1 | `wp_unslash.*sanitize\|sanitize.*wp_unslash` |
| `esc_*()` functions at point of output | +2 | PHPCS `EscapeOutput` |
| Nonce check in form/AJAX handler | +2 | `check_admin_referer\|check_ajax_referer\|wp_verify_nonce` |
| `current_user_can()` before privileged actions | +2 | `current_user_can\s*\(` before data mutation |
| `is_email()` validation for email fields | +1 | `is_email\s*\(` |
| `wp_safe_redirect()` for redirects | +1 | `wp_safe_redirect\s*\(` |
| `wp_check_filetype_and_ext()` for uploads | +2 | `wp_check_filetype_and_ext\s*\(` |
| Echo/print of unescaped `$_*` variable | -3 | `echo\s+\$_(GET\|POST\|REQUEST\|COOKIE)` |
| Echo of unescaped variable | -2 | `echo\s+\$[a-z_]` without esc function |
| `$_REQUEST` usage | -1 | `\$_REQUEST` |
| No nonce in form handling | -3 | `$_POST` processing without `wp_verify_nonce` |
| No capability check | -3 | `$_POST` processing without `current_user_can` |
| `unserialize()` on user input or option | -3 | `unserialize\s*\(\s*\$_(GET\|POST)` |
| `eval()` usage | -3 | `eval\s*\(` |
| `extract()` on user input | -2 | `extract\s*\(\s*\$_(GET\|POST\|REQUEST)` |
| `wp_redirect()` without validation | -2 | `wp_redirect\s*\(.*\$_(GET\|POST)` |
| `include`/`require` with user input | -3 | `(include\|require).*\$_(GET\|POST)` |
| Trusting `$_FILES['type']` | -2 | `\$_FILES\[.*\]\['type'\]` in conditionals |
| `base64_decode()` on unsanitized input | -2 | `base64_decode\s*\(\s*\$_` |

### 5.3 Performance – Automatable Scoring Patterns

| Pattern | Score | Detection |
|---|---|---|
| `wp_cache_get/set` read-through pattern | +2 | `wp_cache_get` before query + `wp_cache_set` after |
| `set_transient` with expiration | +1 | `set_transient\s*\(.*,.*,\s*[0-9]` |
| `no_found_rows => true` on non-paginated queries | +1 | `'no_found_rows'\s*=>\s*true` |
| `update_post_meta_cache => false` when not needed | +1 | `'update_post_meta_cache'\s*=>\s*false` |
| `fields => 'ids'` when only IDs needed | +1 | `'fields'\s*=>\s*'ids'` |
| `is_wp_error()` check on HTTP response | +1 | `is_wp_error\s*\(\s*\$response` |
| HTTP response cached in transient | +2 | `wp_remote_get.*transient\|transient.*wp_remote_get` |
| `add_option` with autoload false | +1 | `add_option\s*\(.*,.*,.*,\s*false` |
| `wp_enqueue_script/style` in proper hook | +1 | inside `add_action.*enqueue_scripts` |
| Conditional enqueueing | +1 | hook check inside enqueue callback |
| `set_transient` without expiration | -2 | `set_transient\s*\(\s*['"]\w+['"]\s*,.*\)` with only 2 args |
| `wp_remote_get/post` without caching | -1 | `wp_remote_get\s*\(` not within transient pattern |
| WP function called inside loop (N+1) | -2 | `get_post_meta\|get_the_terms\|get_user_by` inside `foreach\|while` |
| `wp_cache_flush()` in production hooks | -2 | `wp_cache_flush\s*\(\s*\)` in `add_action` hooks |
| Hardcoded `<script src=` or `<link rel="stylesheet"` | -1 | `echo.*<script\s+src\|echo.*<link\s+rel.*stylesheet` |
| `file_get_contents()` for HTTP | -1 | `file_get_contents\s*\(\s*'?https?://` |
| `curl_exec()` directly | -1 | `curl_exec\s*\(` in plugin code |

### 5.4 Static Analysis Tool Configuration Checklist

| Tool | Configuration | Scoring Signal |
|---|---|---|
| PHPCS WordPressCS | `.phpcs.xml` present, `WordPress` or `WordPress-Extra` standard | +1 for having config; -1 for missing |
| PHPStan | `phpstan.neon` with level ≥ 5 + `szepeviktor/phpstan-wordpress` | +2 for level 5+; +1 for any level |
| Psalm | `psalm.xml` with `--taint-analysis` enabled + `humanmade/psalm-plugin-wordpress` | +2 for taint analysis |
| PCP | `wp plugin check` results clean | +2 for zero critical issues |
| Phan | `.phan/config.php` present | +1 |

---

## Appendix: Key WordPress Time Constants

```php
MINUTE_IN_SECONDS  // 60
HOUR_IN_SECONDS    // 3600
DAY_IN_SECONDS     // 86400
WEEK_IN_SECONDS    // 604800
MONTH_IN_SECONDS   // 2592000 (30 days)
YEAR_IN_SECONDS    // 31536000 (365 days)
```

## Appendix: WordPress Capability Reference for Checks

```php
// Common capabilities in priority order for scoring:
'manage_options'     // Admin settings
'activate_plugins'   // Plugin management
'edit_plugins'       // Plugin file editing
'install_plugins'    // Plugin installation
'edit_theme_options' // Customizer, widgets
'manage_categories'  // Taxonomy management
'edit_others_posts'  // Edit any posts
'publish_posts'      // Publish posts
'edit_posts'         // Edit own posts
'upload_files'       // Media library
'read'               // Basic logged-in capability

// Meta capabilities (require object ID as 2nd arg)
'edit_post'          // Edit specific post
'delete_post'        // Delete specific post
'edit_user'          // Edit specific user
```

---

*Sources*:
- [WordPress wpdb Class Reference](https://developer.wordpress.org/reference/classes/wpdb/)
- [WordPress Security APIs Handbook](https://developer.wordpress.org/apis/security/)
- [WordPress Sanitizing Data](https://developer.wordpress.org/apis/security/sanitizing/)
- [WordPress Escaping Data](https://developer.wordpress.org/apis/security/escaping/)
- [WordPress Nonces API](https://developer.wordpress.org/apis/security/nonces/)
- [WordPress Data Validation](https://developer.wordpress.org/apis/security/data-validation/)
- [WordPress HTTP API](https://developer.wordpress.org/plugins/http-api/)
- [WordPress VIP Database Query Best Practices](https://docs.wpvip.com/databases/optimize-queries/database-queries/)
- [WordPress VIP WP_Query Performance](https://wpvip.com/blog/wp-query-performance/)
- [WordPress VIP Autoloaded Options](https://docs.wpvip.com/wordpress-on-vip/autoloaded-options/)
- [WordPress VIP PHPCS Analysis](https://docs.wpvip.com/vip-code-analysis-bot/phpcs-analysis/)
- [szepeviktor/phpstan-wordpress](https://github.com/szepeviktor/phpstan-wordpress)
- [php-stubs/wordpress-stubs](https://github.com/php-stubs/wordpress-stubs)
- [FloeDesignTechnologies/phpcs-security-audit](https://github.com/FloeDesignTechnologies/phpcs-security-audit)
- [WordPress/WordPress-Coding-Standards](https://github.com/wordpress/wordpress-coding-standards)
- [Make WordPress Core – PHPStan Proposal](https://make.wordpress.org/core/2025/07/11/proposal-phpstan-in-the-wordpress-core-development-workflow/)
- [Make WordPress Plugins – Plugin Check Update](https://make.wordpress.org/plugins/2025/10/29/plugin-check-plugin-now-creates-automatic-security-reports-update/)
- [WordPress Plugin Review Team 2025 Summary](https://make.wordpress.org/plugins/2026/01/07/a-year-in-the-plugins-team-2025/)
- [Wordfence – SQL Injection in WordPress Plugins](https://www.wordfence.com/blog/2025/08/how-to-find-sql-injection-vulnerabilities-in-wordpress-plugins-and-themes/)
- [ircmaxell – WordPress WPDB SQLi Technical Disclosure](https://blog.ircmaxell.com/2017/10/disclosure-wordpress-wpdb-sql-injection-technical.html)
- [Spacedmonkey – Enhancing WP_Query Performance](https://www.spacedmonkey.com/2025/04/14/enhancing-wp_query-performance-in-wordpress/)
- [WordPress Developer Blog – Transients API](https://developer.wordpress.org/news/2024/06/an-introduction-to-the-transients-api/)
- [Pressable – Autoloaded Data Optimization](https://pressable.com/knowledgebase/speed-up-your-wordpress-site-by-optimizing-autoloaded-data/)
- [Psalm Security Analysis / Custom Taint Sinks](https://psalm.dev/docs/security_analysis/custom_taint_sinks/)
- [SonarSource PHP Vulnerability Rules](https://rules.sonarsource.com/php/type/vulnerability/)
- [Phan PHP Static Analyzer](https://github.com/phan/phan)
