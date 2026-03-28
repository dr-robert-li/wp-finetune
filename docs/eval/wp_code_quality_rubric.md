# WordPress PHP Code Quality Scoring Rubric

**Version:** 1.0  
**Last Updated:** March 2026  
**Purpose:** Ground-truth scoring for `eval_gen.py` (code generation quality) and `eval_judge.py` (judge model training). Two independent reviewers applying this rubric to the same PHP function/file should produce scores within ±5 points on the 0–100 scale.

**Sources:**
- [WordPress/WordPress-Coding-Standards](https://github.com/WordPress/WordPress-Coding-Standards) — PHPCS sniff reference
- [WordPress Developer Coding Standards Handbook](https://developer.wordpress.org/coding-standards/)
- [WordPress Developer Reference](https://developer.wordpress.org/reference/)
- [WordPress Security APIs Handbook](https://developer.wordpress.org/apis/security/)
- [WordPress Sanitizing Data](https://developer.wordpress.org/apis/security/sanitizing/)
- [WordPress Escaping Data](https://developer.wordpress.org/apis/security/escaping/)
- [WordPress Nonces API](https://developer.wordpress.org/apis/security/nonces/)
- [WordPress Internationalization Guidelines](https://developer.wordpress.org/apis/internationalization/internationalization-guidelines/)
- [WordPress Accessibility Coding Standards](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/)
- [WordPress REST API Handbook](https://developer.wordpress.org/rest-api/)
- [WordPress VIP Coding Standards](https://github.com/Automattic/VIP-Coding-Standards)
- [WordPress VIP Documentation](https://docs.wpvip.com/)
- [Plugin Review Team Guidelines](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)
- [szepeviktor/phpstan-wordpress](https://github.com/szepeviktor/phpstan-wordpress)

---

## Scoring Overview

Each of the 9 dimensions is scored **0–10**. The overall score is a weighted sum normalized to **0–100**.

| Dimension | Weight |
|---|---|
| 1. WPCS Compliance | 10% |
| 2. Security | 20% |
| 3. SQL Safety | 15% |
| 4. Performance | 10% |
| 5. WP API Usage | 10% |
| 6. i18n / l10n | 10% |
| 7. Accessibility | 8% |
| 8. Error Handling | 10% |
| 9. Code Structure | 7% |
| **Total** | **100%** |

**Overall score formula:**

```
overall = (D1×10 + D2×20 + D3×15 + D4×10 + D5×10 + D6×10 + D7×8 + D8×10 + D9×7) / 10
```

Where D1–D9 are the 0–10 scores for each dimension.

**Dimension not applicable (N/A) handling:** If a dimension is entirely not applicable to the code under review (e.g., no database calls → SQL Safety is N/A; no output HTML → Accessibility is N/A), exclude it and redistribute its weight proportionally across remaining applicable dimensions. Score N/A only when there is genuinely no code surface for that dimension.

---

## Dimension 1: WPCS Compliance

**Scope:** WordPress PHP Coding Standards formatting, naming conventions, whitespace, and language construct usage.  
**Authority:** [WordPress PHP Coding Standards](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/php/), [WordPress/WordPress-Coding-Standards](https://github.com/WordPress/WordPress-Coding-Standards)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **WPCS-P01** | Indentation uses real tabs, not spaces | Regex: lines starting with 4+ spaces (not in strings/heredoc) | +2 |
| **WPCS-P02** | Functions and variables use `lowercase_with_underscores` naming | PHPCS `WordPress.NamingConventions.ValidFunctionName`, `ValidVariableName` | +2 |
| **WPCS-P03** | Classes use `UpperCamelCase_With_Underscores` (e.g., `WP_My_Class`) | PHPCS `WordPress.NamingConventions.ValidFunctionName` (class scope) | +1 |
| **WPCS-P04** | Constants use `ALL_UPPERCASE_WITH_UNDERSCORES` | Regex: `define\s*\(\s*'[A-Z][A-Z0-9_]+'` | +1 |
| **WPCS-P05** | Yoda conditions used for `==`, `!=`, `===`, `!==` comparisons | PHPCS `WordPress.PHP.YodaConditions` | +1 |
| **WPCS-P06** | Control structures use proper WP spacing: `if ( $cond )` with spaces inside parens | PHPCS `WordPress.WhiteSpace.ControlStructureSpacing` | +1 |
| **WPCS-P07** | Spaces on both sides of all binary operators | PHPCS `WordPress.WhiteSpace.OperatorSpacing` | +1 |
| **WPCS-P08** | All globals (functions, classes, hooks, constants) prefixed with plugin/theme slug | PHPCS `WordPress.NamingConventions.PrefixAllGlobals` | +2 |
| **WPCS-P09** | File named with lowercase-hyphens; class files prefixed `class-` | PHPCS `WordPress.Files.FileName` | +1 |
| **WPCS-P10** | `in_array()` always called with `true` as third argument (strict comparison) | PHPCS `WordPress.PHP.StrictInArray` | +1 |
| **WPCS-P11** | Braces used for all control structure blocks (including single-line) | Regex: `if\s*\(.*\)\s+[^{]` (single-statement if without brace) | +1 |
| **WPCS-P12** | `elseif` used (not `else if`) | Regex: `else\s+if\b` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **WPCS-N01** | Short PHP open tags (`<?` or `<?=`) used | PHPCS `Generic.PHP.DisallowShortOpenTag` | −3 |
| **WPCS-N02** | camelCase function or variable names (PSR-style in non-class context) | PHPCS `WordPress.NamingConventions.ValidFunctionName` | −2 |
| **WPCS-N03** | No global prefix — functions/classes/hooks in global namespace without prefix | PHPCS `WordPress.NamingConventions.PrefixAllGlobals` | −3 |
| **WPCS-N04** | `var` used for class property declaration (instead of `public`/`protected`/`private`) | PHPCS (Squiz/PSR standards); `WordPress.PHP.*` | −2 |
| **WPCS-N05** | Explicit visibility (`public`/`protected`/`private`) absent on class method or property | PHPStan / PHPCS PSR rules | −2 |
| **WPCS-N06** | Closing PHP tag `?>` present at end of pure PHP file | Regex: `\?>\s*$` at end of file | −1 |
| **WPCS-N07** | Short ternary `?:` (Elvis operator) used | Regex: `\?:` in expression context | −1 |
| **WPCS-N08** | `in_array()` without strict `true` third argument | PHPCS `WordPress.PHP.StrictInArray` | −2 |
| **WPCS-N09** | Loose comparison `==` / `!=` used where strict comparison is possible | PHPCS `WordPress.PHP.StrictComparisons` | −1 |
| **WPCS-N10** | `extract()` used | PHPCS `WordPress.PHP.DontExtract` | −2 |
| **WPCS-N11** | `create_function()` used | PHPCS `WordPress.PHP.RestrictedPHPFunctions` | −3 |
| **WPCS-N12** | `eval()` used | PHPCS `WordPress.PHP.RestrictedPHPFunctions` | −5 |
| **WPCS-N13** | `goto` statement used | PHPCS (various); pattern: `\bgoto\b` | −3 |
| **WPCS-N14** | `@` error suppression operator used | PHPCS `WordPress.PHP.NoSilencedErrors` | −2 |
| **WPCS-N15** | Debug functions (`var_dump`, `var_export`, `print_r`, `error_log`) left in code | PHPCS `WordPress.PHP.DevelopmentFunctions` | −2 |
| **WPCS-N16** | `current_time('timestamp')` or `current_time('U')` for Unix timestamp | PHPCS `WordPress.DateTime.CurrentTimeTimestamp` | −1 |
| **WPCS-N17** | Assignment inside ternary condition (`$a = fn()` where comparison intended) | PHPCS `WordPress.CodeAnalysis.AssignmentInTernaryCondition` | −2 |
| **WPCS-N18** | `esc_html()` wrapping a plain string literal that should also call `__()` | PHPCS `WordPress.CodeAnalysis.EscapedNotTranslated` | −1 |
| **WPCS-N19** | `preg_quote()` called without `$delimiter` second argument | PHPCS `WordPress.PHP.PregQuoteDelimiter` | −1 |
| **WPCS-N20** | WordPress global variables overwritten (`$post`, `$wp_query`, `$wpdb`, `$current_user`) | PHPCS `WordPress.WP.GlobalVariablesOverride` | −2 |

### C. Dimension 1 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 15   # sum of all positive weights
D1 = clamp(raw / max_positive * 10, 0, 10)
```

**Calibration:** Code with zero style violations and correct naming throughout: 9–10. Minor spacing/naming issues only: 6–8. Multiple naming violations + no prefix: 2–4. Critical forbidden constructs (`eval`, `extract`, `create_function`): 0–2.

---

## Dimension 2: Security

**Scope:** Output escaping, input sanitization, nonce verification, capability checks, CSRF/XSS/injection prevention.  
**Authority:** [WordPress Security Handbook](https://developer.wordpress.org/apis/security/), [WordPress Security Sniff Reference](https://github.com/WordPress/WordPress-Coding-Standards), [OWASP Top 10](https://owasp.org/www-project-top-ten/)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **SEC-P01** | All `echo`/`print` of variables wrapped in appropriate `esc_html()`, `esc_attr()`, `esc_url()`, `esc_js()`, `esc_textarea()`, `wp_kses()`, or `wp_kses_post()` | PHPCS `WordPress.Security.EscapeOutput` | +3 |
| **SEC-P02** | `wp_unslash()` called before sanitize function on `$_POST`/`$_GET` data | Regex: `wp_unslash\s*\(.*sanitize\|sanitize.*wp_unslash`; order check | +2 |
| **SEC-P03** | All superglobal input (`$_GET`, `$_POST`, `$_REQUEST`, `$_COOKIE`, `$_FILES`, `$_SERVER`) sanitized before use | PHPCS `WordPress.Security.ValidatedSanitizedInput` | +3 |
| **SEC-P04** | Nonce verified with `wp_verify_nonce()`, `check_admin_referer()`, or `check_ajax_referer()` before processing form/AJAX input | PHPCS `WordPress.Security.NonceVerification`; regex for nonce functions | +3 |
| **SEC-P05** | `current_user_can()` called before any privileged write operation or admin action | Regex: `current_user_can\s*\(` preceding `$_POST` processing, `$wpdb` writes, or `wp_insert_post` | +3 |
| **SEC-P06** | `wp_safe_redirect()` used instead of `wp_redirect()` | PHPCS `WordPress.Security.SafeRedirect` | +1 |
| **SEC-P07** | `wp_check_filetype_and_ext()` used for file upload validation | Regex: `wp_check_filetype_and_ext\s*\(` | +2 |
| **SEC-P08** | `wp_handle_upload()` used for processing file uploads | Regex: `wp_handle_upload\s*\(` | +1 |
| **SEC-P09** | `is_email()` used to validate email inputs | Regex: `is_email\s*\(` | +1 |
| **SEC-P10** | Safelist pattern used for `ORDER BY` or other non-preparable SQL fragments | Regex: `in_array.*$_.*allowed\|$allowed.*in_array` for orderby/order | +1 |
| **SEC-P11** | Integer IDs cast with `absint()` before use | Regex: `absint\s*\(` | +1 |
| **SEC-P12** | `esc_html__()` / `esc_attr__()` combined escape+translate used where applicable | Regex: `esc_html__\s*\(\|esc_attr__\s*\(` | +1 |
| **SEC-P13** | Output context matched to correct escaping function (URL→`esc_url`, attr→`esc_attr`, HTML→`esc_html`, JS→`esc_js`) | LLM or AST context analysis | +2 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **SEC-N01** | Raw `$_GET`/`$_POST`/`$_REQUEST` used in `echo`/`print` without escaping | PHPCS `WordPress.Security.EscapeOutput`; regex: `echo\s+\$_(GET\|POST\|REQUEST)` | −5 |
| **SEC-N02** | Any unsanitized superglobal used in code (not just output) | PHPCS `WordPress.Security.ValidatedSanitizedInput` | −3 |
| **SEC-N03** | Form or AJAX handler processes `$_POST`/`$_GET` without nonce check | PHPCS `WordPress.Security.NonceVerification` | −4 |
| **SEC-N04** | Privileged action executed without `current_user_can()` check | Regex: `wp_insert_post\|update_option\|delete_option\|$wpdb->` without preceding `current_user_can` | −4 |
| **SEC-N05** | `wp_redirect()` used with user-supplied URL without `wp_safe_redirect()` or `wp_validate_redirect()` | PHPCS `WordPress.Security.SafeRedirect`; regex: `wp_redirect\s*\(.*\$_(GET\|POST)` | −3 |
| **SEC-N06** | `unserialize()` called on user input or on an option that users can write | Regex: `unserialize\s*\(\s*\$_(GET\|POST\|REQUEST\|COOKIE)` | −4 |
| **SEC-N07** | `base64_decode()` used on user input without subsequent sanitization | Regex: `base64_decode\s*\(\s*\$_` | −2 |
| **SEC-N08** | `include`/`require` with user-controlled variable | Regex: `(include|require)(_once)?\s+.*\$_(GET\|POST\|REQUEST)` | −5 |
| **SEC-N09** | `$_FILES['*']['type']` used as sole MIME type validation (browser-set, untrustworthy) | Regex: `$_FILES\[.*\]\['type'\]` in if condition | −3 |
| **SEC-N10** | `wp_nonce_field()` absent from HTML form that processes POST data | LLM/template analysis: `<form` present without `wp_nonce_field` | −3 |
| **SEC-N11** | Nonce verification omitted on `nopriv` AJAX handler that modifies data | Regex: `wp_ajax_nopriv_` without `check_ajax_referer` | −4 |
| **SEC-N12** | Role name used instead of capability string in `current_user_can()` | Regex: `current_user_can\s*\(\s*['"]administrator\|editor\|author\|contributor\|subscriber['"]` | −2 |
| **SEC-N13** | `is_admin()` used as a security gate (checks context, not permissions) | LLM analysis: `if ( is_admin() )` protecting data modification | −2 |
| **SEC-N14** | `__FILE__` used as menu slug in `add_menu_page()` / `add_submenu_page()` | PHPCS `WordPress.Security.PluginMenuSlug` | −1 |
| **SEC-N15** | `esc_html()` used on a URL context (wrong escaping function) | LLM/AST: `href="'. esc_html(` | −2 |
| **SEC-N16** | `esc_attr()` used on a URL (does not validate scheme, allows `javascript:`) | LLM/AST: `href="'. esc_attr(` | −2 |
| **SEC-N17** | Double-escaping: value escaped then escaped again | LLM analysis: `esc_attr( esc_url(`, `esc_html( esc_attr(` | −1 |
| **SEC-N18** | `$_REQUEST` used instead of specific `$_POST` or `$_GET` (cookie injection risk) | Regex: `\$_REQUEST\b` | −1 |
| **SEC-N19** | `exec()`, `shell_exec()`, `system()`, `passthru()` used with user-derived input | phpcs-security-audit `BadFunctions.SystemExecFunctions` | −5 |
| **SEC-N20** | `preg_replace()` with `/e` modifier (code execution) | Regex: `preg_replace\s*\(.*\/e['"` | −5 |

### C. Dimension 2 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 25   # sum of all positive weights
D2 = clamp(raw / max_positive * 10, 0, 10)
```

**Critical floor rule:** If SEC-N01, SEC-N03, SEC-N04, SEC-N06, SEC-N08, SEC-N19, or SEC-N20 is present (catastrophic vulnerability), D2 **cannot exceed 3** regardless of positive signals. These represent exploitable security holes that override stylistic compliance.

**Calibration:** Perfect escaping + sanitization + nonce + capability: 9–10. Minor escaping gap (static string not escaped): 6–8. Missing nonce on one handler: 3–5. Unescaped `$_GET` in output: 0–2.

---

## Dimension 3: SQL Safety

**Scope:** `$wpdb` usage, prepared statements, `WP_Query` vs raw SQL, injection prevention.  
**Authority:** [WordPress wpdb Class Reference](https://developer.wordpress.org/reference/classes/wpdb/), [WordPress VIP Database Query Docs](https://docs.wpvip.com/databases/optimize-queries/database-queries/), PHPCS `WordPress.DB.*`

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **SQL-P01** | `$wpdb->prepare()` used with typed placeholders (`%d`, `%s`, `%f`, `%i`) for all custom queries containing variables | PHPCS `WordPress.DB.PreparedSQL` (absence of violations) | +3 |
| **SQL-P02** | `$wpdb->insert()`, `$wpdb->update()`, `$wpdb->delete()` used instead of raw `INSERT`/`UPDATE`/`DELETE` via `$wpdb->query()` | Regex: `$wpdb->(insert\|update\|delete)\s*\(` | +2 |
| **SQL-P03** | `$wpdb->esc_like()` used before `prepare()` for LIKE clauses | Regex: `esc_like\s*\(.*\).*prepare\|prepare.*LIKE.*esc_like` | +2 |
| **SQL-P04** | `WP_Query` or `get_posts()` used instead of raw SQL for post/meta/taxonomy queries | Regex: `new WP_Query\s*\(\|get_posts\s*\(` in preference to `$wpdb->get_results` | +2 |
| **SQL-P05** | `%i` identifier placeholder used for dynamic table/column names (WP 6.2+) | Regex: `prepare\s*\(.*%i` | +1 |
| **SQL-P06** | `$wpdb->posts`, `$wpdb->options` etc. used instead of hardcoded `wp_*` table names | Absence of regex: `".*wp_[a-z_]+.*"` in SQL strings | +1 |
| **SQL-P07** | ORDER BY direction validated against safelist before use in query | Regex: `in_array.*('ASC','DESC')\|in_array.*$order.*true` | +1 |
| **SQL-P08** | `LIMIT` clause present on all direct `$wpdb->get_results()` / `$wpdb->query()` calls | LLM / regex: absence of `get_results("SELECT` without `LIMIT` | +1 |
| **SQL-P09** | `$wpdb->last_error` checked after critical write queries | Regex: `$wpdb->last_error` following `$wpdb->query\|insert\|update\|delete` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **SQL-N01** | Direct variable interpolation in SQL string passed to `$wpdb` method | PHPCS `WordPress.DB.PreparedSQL`; regex: `$wpdb->.*\(.*"\s*.*\$[a-z_]+` without prepare wrapper | −5 |
| **SQL-N02** | String concatenation used to build SQL query with variables | Regex: `["'].*\.\s*\$[a-z_].*["']` in `$wpdb->` argument | −4 |
| **SQL-N03** | `$_GET`/`$_POST`/`$_REQUEST`/`$_COOKIE` variable used directly in SQL (without prepare or absint) | PHPCS `WordPress.DB.PreparedSQL`; regex: `\$_(GET\|POST\|REQUEST\|COOKIE)` inside SQL string | −5 |
| **SQL-N04** | LIKE clause without `esc_like()` (wildcard injection) | Regex: `LIKE\s+%s` without preceding `esc_like` | −3 |
| **SQL-N05** | `%s` placeholder manually quoted in `prepare()` string (`'%s'`) — double-escaping bug | PHPCS `WordPress.DB.PreparedSQLPlaceholders`; regex: `'%s'` inside prepare string | −2 |
| **SQL-N06** | Incorrect placeholder count in `prepare()` (more/fewer args than placeholders) | PHPCS `WordPress.DB.PreparedSQLPlaceholders` | −2 |
| **SQL-N07** | `$wpdb->escape()` used (deprecated, non-functional in modern WP) | Regex: `->escape\s*\(` | −2 |
| **SQL-N08** | `esc_sql()` used inside `prepare()` (double-escaping, corrupts values) | Regex: `prepare\s*\(.*esc_sql` | −2 |
| **SQL-N09** | Raw PHP database functions used (`mysql_*`, `mysqli_*`, `PDO`) | PHPCS `WordPress.DB.RestrictedFunctions`, `WordPress.DB.RestrictedClasses` | −4 |
| **SQL-N10** | `query_posts()` used (corrupts global `$wp_query`, breaks pagination) | PHPCS `WordPress.WP.DiscouragedFunctions`; regex: `query_posts\s*\(` | −3 |
| **SQL-N11** | Hardcoded table name `wp_*` in SQL strings instead of `$wpdb->tablename` | Regex: `".*wp_[a-z_]+"` in SQL context | −2 |
| **SQL-N12** | `posts_per_page => -1` or `numberposts => -1` in `WP_Query`/`get_posts` | PHPCS `WordPress.WP.PostsPerPage`; regex: `'posts_per_page'\s*=>\s*-1` | −2 |
| **SQL-N13** | `orderby => 'rand'` in `WP_Query` (full table scan on every request) | Regex: `'orderby'\s*=>\s*'rand'` | −2 |
| **SQL-N14** | `suppress_filters => true` in `WP_Query` (bypasses caching layers) | Regex: `'suppress_filters'\s*=>\s*true` | −1 |
| **SQL-N15** | `SELECT *` in direct SQL query (fetches unnecessary columns) | Regex: `SELECT\s+\*\s+FROM` in `$wpdb->get_results` | −1 |
| **SQL-N16** | No LIMIT on direct SQL `get_results()` that could return unbounded rows | LLM / regex: `get_results.*SELECT.*FROM` without LIMIT | −2 |
| **SQL-N17** | User input passed as the query template string to `prepare()` (not just as argument) | Regex: `prepare\s*\(\s*\$_(GET\|POST\|REQUEST)` | −5 |

### C. Dimension 3 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 14   # sum of all positive weights
D3 = clamp(raw / max_positive * 10, 0, 10)
```

**Critical floor rule:** If SQL-N01, SQL-N03, or SQL-N17 is present (direct SQL injection path), D3 **cannot exceed 2**.

**N/A rule:** If the code contains no `$wpdb` usage and no `WP_Query`, score this dimension N/A and redistribute weight. A single `$wpdb->prepare()` call without violations = 7+.

---

## Dimension 4: Performance

**Scope:** Caching strategy, N+1 query prevention, `WP_Query` optimization arguments, autoload management, remote HTTP request hygiene.  
**Authority:** [WordPress VIP Performance Docs](https://docs.wpvip.com/databases/optimize-queries/), [WordPress Transients API](https://developer.wordpress.org/news/2024/06/an-introduction-to-the-transients-api/), [WP_Query Performance Guide](https://wpvip.com/blog/wp-query-performance/)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **PERF-P01** | `wp_cache_get()` / `wp_cache_set()` read-through pattern used around expensive queries | Regex: `wp_cache_get\s*\(` before query + `wp_cache_set\s*\(` after | +3 |
| **PERF-P02** | `set_transient()` called with explicit expiration time (3rd argument) | Regex: `set_transient\s*\([^,]+,[^,]+,\s*[^)]+\)` — 3 arguments | +2 |
| **PERF-P03** | `get_transient()` result checked against `false` before use | Regex: `false\s*===?\s*get_transient\|get_transient.*!==?\s*false` | +1 |
| **PERF-P04** | Remote HTTP responses cached in transients | Regex: `wp_remote_get\|wp_remote_post` within same scope as `set_transient` | +2 |
| **PERF-P05** | `no_found_rows => true` in non-paginated `WP_Query` | Regex: `'no_found_rows'\s*=>\s*true` | +1 |
| **PERF-P06** | `fields => 'ids'` used when only post IDs are needed | Regex: `'fields'\s*=>\s*'ids'` | +1 |
| **PERF-P07** | `update_post_meta_cache => false` or `update_post_term_cache => false` when meta/terms not needed in loop | Regex: `'update_post_meta_cache'\s*=>\s*false\|'update_post_term_cache'\s*=>\s*false` | +1 |
| **PERF-P08** | `add_option()` / `update_option()` called with explicit `false` for autoload on non-essential options | Regex: `add_option\s*\(.*,.*,.*,\s*false\|update_option\s*\(.*,.*,\s*false` | +1 |
| **PERF-P09** | Scripts/styles enqueued via `wp_enqueue_scripts` or `admin_enqueue_scripts` hook with version parameter set | PHPCS `WordPress.WP.EnqueuedResourceParameters`; regex: `wp_enqueue_script.*,.*,.*,.*[^,)]+` (4+ args) | +1 |
| **PERF-P10** | Conditional script/style enqueueing (only loads where needed) | LLM/regex: conditional check (`if`, early return) inside enqueue callback | +1 |
| **PERF-P11** | `wp_remote_get()` called with explicit `timeout` parameter | Regex: `wp_remote_get\s*\(.*'timeout'` | +1 |
| **PERF-P12** | `wp_remote_retrieve_response_code()` checked after HTTP call | Regex: `wp_remote_retrieve_response_code\s*\(` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **PERF-N01** | `get_post_meta()`, `get_the_terms()`, or `get_user_by()` called inside `foreach`/`while` loop over posts (N+1 pattern) | LLM / regex: `get_post_meta\|get_the_terms\|get_the_category\|get_user_by` inside `foreach\|while` | −3 |
| **PERF-N02** | `$wpdb->get_results()` or similar called inside loop (N+1 SQL) | Regex: `$wpdb->` inside `foreach\|while` body | −3 |
| **PERF-N03** | `set_transient()` called without expiration (becomes autoloaded option forever) | Regex: `set_transient\s*\(\s*['"]\w+['"]\s*,\s*[^,)]+\s*\)` — only 2 args | −3 |
| **PERF-N04** | `wp_cache_flush()` called inside a production hook (flushes all cache for all users) | Regex: `wp_cache_flush\s*\(\s*\)` in `add_action` callback | −3 |
| **PERF-N05** | `wp_remote_get()` / `wp_remote_post()` called on every page load without caching | LLM: `wp_remote_get` in hook callback without transient/cache wrapper | −2 |
| **PERF-N06** | `file_get_contents()` used for HTTP requests (bypasses WP HTTP API, proxies, filters) | PHPCS `WordPress.WP.AlternativeFunctions`; regex: `file_get_contents\s*\(\s*'?https?://` | −2 |
| **PERF-N07** | `curl_exec()` used directly in plugin code | Regex: `curl_exec\s*\(` | −2 |
| **PERF-N08** | Hardcoded `<script src=` or `<link rel="stylesheet"` echoed in PHP templates | PHPCS `WordPress.WP.EnqueuedResources`; regex: `echo.*<script\s+src\|echo.*<link.*stylesheet` | −2 |
| **PERF-N09** | `flush_rewrite_rules()` called on `init` hook on every request (not just activation) | Regex: `add_action\s*\(\s*'init'.*flush_rewrite_rules` | −2 |
| **PERF-N10** | `get_option()` called repeatedly in a loop for same option (should pre-fetch) | LLM / regex: `get_option\s*\(` inside `foreach\|while` | −1 |
| **PERF-N11** | Large array stored as autoloaded option (`add_option`/`update_option` without `false` on non-essential data) | LLM / regex: `add_option\|update_option` with `array(` or serialized data, without `false` | −2 |
| **PERF-N12** | `wp_remote_get` / `wp_remote_post` called without `is_wp_error()` check | Regex: `wp_remote_get\s*\(` without nearby `is_wp_error` | −1 |
| **PERF-N13** | `orderby => 'rand'` in `WP_Query` (counted also in SQL Safety; score in whichever dimension applies) | Regex: `'orderby'\s*=>\s*'rand'` | −2 |

### C. Dimension 4 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 16   # sum of all positive weights
D4 = clamp(raw / max_positive * 10, 0, 10)
```

**N/A rule:** If code contains no database calls, no HTTP calls, and no enqueueing, mark N/A.

---

## Dimension 5: WP API Usage

**Scope:** Using WordPress functions instead of raw PHP equivalents, proper use of hooks, template hierarchy, asset management, date/time functions.  
**Authority:** [WordPress Plugin Handbook](https://developer.wordpress.org/plugins/), [WordPress WP Alternatives Sniff](https://github.com/WordPress/WordPress-Coding-Standards), [Plugin Review Team Guidelines](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **WAPI-P01** | `wp_remote_get()` / `wp_remote_post()` used instead of `file_get_contents()` or `curl_*` for HTTP | PHPCS `WordPress.WP.AlternativeFunctions` | +2 |
| **WAPI-P02** | `wp_json_encode()` used instead of `json_encode()` | PHPCS `WordPress.WP.AlternativeFunctions` | +1 |
| **WAPI-P03** | `wp_rand()` used instead of `rand()` or `mt_rand()` | PHPCS `WordPress.WP.AlternativeFunctions` | +1 |
| **WAPI-P04** | `wp_strip_all_tags()` used instead of `strip_tags()` | PHPCS `WordPress.WP.AlternativeFunctions` | +1 |
| **WAPI-P05** | `wp_date()` used instead of `date()` for user-facing date display | PHPCS `WordPress.DateTime.RestrictedFunctions` | +1 |
| **WAPI-P06** | `wp_enqueue_script()` / `wp_enqueue_style()` used with version parameter and proper hook | PHPCS `WordPress.WP.EnqueuedResources`, `WordPress.WP.EnqueuedResourceParameters` | +2 |
| **WAPI-P07** | WordPress Filesystem API (`WP_Filesystem`, `$wp_filesystem`) used instead of raw `fopen`/`file_put_contents` | Regex: `WP_Filesystem\(\)\|$wp_filesystem->` | +2 |
| **WAPI-P08** | `add_action()` / `add_filter()` used for all initialization (not direct function calls at file load time) | LLM: initialization code wrapped in hooks | +2 |
| **WAPI-P09** | Custom post types / taxonomies registered via `init` hook | Regex: `add_action\s*\(\s*'init'.*register_post_type\|register_taxonomy` | +1 |
| **WAPI-P10** | `wp_localize_script()` or `wp_add_inline_script()` used to pass PHP data to JS (not inline `<script>`) | Regex: `wp_localize_script\s*\(\|wp_add_inline_script\s*\(` | +1 |
| **WAPI-P11** | `plugins_url()` or `plugin_dir_url()` used for plugin asset URLs (not hardcoded paths) | Regex: `plugins_url\s*\(\|plugin_dir_url\s*\(` | +1 |
| **WAPI-P12** | `WP_Query` or `get_posts()` used instead of `query_posts()` | Absence of `query_posts\s*\(` | +1 |
| **WAPI-P13** | Deprecated WP functions absent from code | PHPCS `WordPress.WP.DeprecatedFunctions`, `WordPress.WP.DeprecatedClasses` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **WAPI-N01** | `file_get_contents()` used for HTTP requests | PHPCS `WordPress.WP.AlternativeFunctions` | −2 |
| **WAPI-N02** | `curl_*` functions used directly (not via WP HTTP API) | Regex: `curl_init\|curl_exec\|curl_setopt` | −2 |
| **WAPI-N03** | `json_encode()` used (should use `wp_json_encode()` for safety in WP context) | PHPCS `WordPress.WP.AlternativeFunctions` | −1 |
| **WAPI-N04** | `rand()` / `mt_rand()` used for randomness (not cryptographically suitable; use `wp_rand()`) | PHPCS `WordPress.WP.AlternativeFunctions` | −1 |
| **WAPI-N05** | `date()` used for user-facing output (not timezone-aware; use `wp_date()`) | PHPCS `WordPress.DateTime.RestrictedFunctions` | −1 |
| **WAPI-N06** | `query_posts()` used | PHPCS `WordPress.WP.DiscouragedFunctions` | −3 |
| **WAPI-N07** | Deprecated WordPress functions used | PHPCS `WordPress.WP.DeprecatedFunctions` | −2 |
| **WAPI-N08** | Deprecated WordPress classes used | PHPCS `WordPress.WP.DeprecatedClasses` | −2 |
| **WAPI-N09** | WordPress bundled library re-included separately (jQuery, PHPMailer, SimplePie) | Regex: `require.*phpmailer\|wp_enqueue_script.*jquery.*http` | −2 |
| **WAPI-N10** | Raw PHP filesystem functions (`fopen`, `file_put_contents`, `fwrite`) used without WP Filesystem API | Regex: `\bfopen\s*\(\|file_put_contents\s*\(\|fwrite\s*\(` | −2 |
| **WAPI-N11** | PHP session functions used (`session_start()`, `$_SESSION`) | Regex: `session_start\s*\(\|\$_SESSION` | −2 |
| **WAPI-N12** | Initialization code runs at file include time (not deferred to a hook) | LLM: `register_post_type\|add_menu_page\|wp_enqueue_script` called outside any `add_action` | −2 |
| **WAPI-N13** | `add_menu_page()` / `add_submenu_page()` called without capability parameter restriction | LLM: menu registration without verifying minimum capability in page callback | −1 |
| **WAPI-N14** | `get_page_by_title()` used (deprecated WP 6.2) | PHPCS `WordPress.WP.DeprecatedFunctions` | −1 |

### C. Dimension 5 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 17   # sum of all positive weights
D5 = clamp(raw / max_positive * 10, 0, 10)
```

---

## Dimension 6: i18n / l10n

**Scope:** Translation function usage, text domains, escape+translate combos, plural handling, placeholder formatting.  
**Authority:** [WordPress Internationalization Guidelines](https://developer.wordpress.org/apis/internationalization/internationalization-guidelines/), [How to Internationalize Your Plugin](https://developer.wordpress.org/plugins/internationalization/how-to-internationalize-your-plugin/), PHPCS `WordPress.WP.I18n`

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **I18N-P01** | All user-visible strings wrapped in a translation function (`__()`, `_e()`, `_x()`, etc.) | LLM / PHPCS `WordPress.WP.I18n`; regex: `echo\s+'[A-Za-z]` without translation wrapper | +3 |
| **I18N-P02** | Every translation function call includes the correct text domain as a string literal (not variable/constant) | PHPCS `WordPress.WP.I18n`; regex: missing second arg or `$var` as last arg | +2 |
| **I18N-P03** | `_n()` or `_nx()` used for all plural forms (not `if ($n == 1)` manual branching) | Regex: `_n\s*\(\|_nx\s*\(`; absence of `if.*===\s*1.*__(` pattern | +2 |
| **I18N-P04** | `_x()` or `_ex()` used for contextually ambiguous strings | Regex: `_x\s*\(\|_ex\s*\(` | +1 |
| **I18N-P05** | `esc_html__()` / `esc_html_e()` used for HTML context translations | Regex: `esc_html__\s*\(\|esc_html_e\s*\(` | +2 |
| **I18N-P06** | `esc_attr__()` / `esc_attr_e()` used for attribute context translations | Regex: `esc_attr__\s*\(\|esc_attr_e\s*\(` | +1 |
| **I18N-P07** | Positional placeholders (`%1$s`, `%2$s`) used when translated string has 2+ format args | Regex: `%1\$s.*%2\$s` inside `__()` strings | +1 |
| **I18N-P08** | `/* translators: */` comment present on the line immediately before any translated string with placeholders | Regex: `\/\*.*translators.*\*\/` before `__\|_e\|_n` with `%s\|%d` | +1 |
| **I18N-P09** | `number_format_i18n()` used instead of `number_format()` for user-facing numbers | Regex: `number_format_i18n\s*\(` | +1 |
| **I18N-P10** | `date_i18n()` used instead of `date()` for user-facing dates | Regex: `date_i18n\s*\(` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **I18N-N01** | Translation function called without text domain argument | PHPCS `WordPress.WP.I18n`; regex: `__\s*\(\s*'[^']+'\s*\)` (single arg) | −2 |
| **I18N-N02** | Variable or constant used as text domain (prevents static string extraction) | PHPCS `WordPress.WP.I18n`; regex: `__\s*\(.*,\s*\$[a-z_]+\s*\)\|__\s*\(.*,\s*[A-Z_]+\s*\)` | −2 |
| **I18N-N03** | PHP variable interpolated inside a translation string literal | PHPCS `WordPress.WP.I18n`; regex: `__\s*\(\s*"[^"]*\$[a-z_]` | −3 |
| **I18N-N04** | String concatenation used to split a translated sentence across multiple `__()` calls | Regex: `__\s*\(.*\)\s*\.\s*.*\.\s*__\s*\(` | −2 |
| **I18N-N05** | Multiple `%s` / `%d` placeholders without positional numbering (`%1$s`) | Regex: `%s[^']*%s\|%d[^']*%d` inside `__()` strings without positional markers | −1 |
| **I18N-N06** | Manual singular/plural branching with `if ($n == 1)` instead of `_n()` | Regex: `if\s*\(\s*\$\w+\s*===?\s*1\s*\).*__\(` with else `__\(` | −2 |
| **I18N-N07** | Missing `/* translators: */` comment for placeholder-containing string | Regex: `__\s*\(.*%s\|%d\|%\d+\$` without preceding translators comment | −1 |
| **I18N-N08** | `number_format()` used for user-facing numeric output | Regex: `echo.*number_format\s*\(` | −1 |
| **I18N-N09** | `date()` or `gmdate()` used for user-facing date output | Regex: `echo.*date\s*\(\|echo.*gmdate\s*\(` | −1 |
| **I18N-N10** | HTML markup (tags like `<strong>`, `<a>`) embedded inside translated string | Regex: `__\s*\(\s*'[^']*<[a-z]` | −1 |
| **I18N-N11** | Translation function called with empty string `''` | Regex: `__\s*\(\s*''\s*` | −1 |
| **I18N-N12** | `_e()` used inside HTML attribute (unescaped output in attribute context; should use `esc_attr_e()`) | LLM / regex: `_e\s*\(` inside `echo '.*="`  | −2 |
| **I18N-N13** | User-visible English string echoed without any translation wrapper | LLM / regex: `echo\s+['"][A-Z][a-zA-Z\s]+['"]\s*[;.]` | −2 |

### C. Dimension 6 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 15   # sum of all positive weights
D6 = clamp(raw / max_positive * 10, 0, 10)
```

**N/A rule:** If the code file contains no user-facing output and no string literals intended for display, mark N/A.

---

## Dimension 7: Accessibility

**Scope:** ARIA usage, form labels, screen reader text, focus management, color contrast intent, keyboard operability.  
**Authority:** [WordPress Accessibility Coding Standards](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/), [WCAG 2.2 Level AA](https://www.w3.org/TR/WCAG22/), [Make WordPress Accessible Handbook](https://make.wordpress.org/accessibility/handbook/)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **A11Y-P01** | All form `<input>`, `<select>`, `<textarea>` elements have an associated `<label for="id">` with matching `id` | Regex / DOM: `<label\s+for=` with matching `id=` on input | +3 |
| **A11Y-P02** | Skip link present as first focusable element, uses `.screen-reader-text` class, and `href` targets content area | Regex: `screen-reader-text.*skip\|skip.*screen-reader-text` with `href="#` | +2 |
| **A11Y-P03** | `aria-label` or `aria-labelledby` on interactive elements without visible text (icon-only buttons) | Regex: `aria-label=\|aria-labelledby=` on `<button\|<a` | +2 |
| **A11Y-P04** | `role="alert"` or `aria-live` used on dynamically injected notices and error messages | Regex: `role="alert"\|aria-live=` | +2 |
| **A11Y-P05** | All `<img>` elements have `alt` attribute (descriptive text or empty string for decorative) | Regex: `<img` → all instances have `alt=` | +2 |
| **A11Y-P06** | `<fieldset>` and `<legend>` used for radio button and checkbox groups | Regex: `<fieldset.*<legend` | +1 |
| **A11Y-P07** | Semantic HTML5 elements (`<main>`, `<nav>`, `<aside>`, `<header>`, `<footer>`) used for page regions | Regex: `<main\b\|<nav\b\|<aside\b\|<header\b\|<footer\b` | +1 |
| **A11Y-P08** | Custom focus styles provided (not just removed) when `outline: none` is present | Regex: `outline.*none.*outline\|outline.*0.*outline` with replacement style | +1 |
| **A11Y-P09** | `autocomplete` attribute set on personal data form fields (name, email, phone) | Regex: `autocomplete=` on appropriate `<input>` fields | +1 |
| **A11Y-P10** | `aria-required="true"` and HTML `required` attribute both present on required fields | Regex: `aria-required="true".*required\|required.*aria-required` | +1 |
| **A11Y-P11** | Error messages linked to inputs via `aria-describedby` | Regex: `aria-describedby=` on input elements | +1 |
| **A11Y-P12** | `aria-expanded`, `aria-controls`, `aria-haspopup` updated by JavaScript for interactive widgets | Regex: `aria-expanded=\|aria-controls=\|aria-haspopup=` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **A11Y-N01** | Form input lacks any associated label (no `<label for>`, no `aria-label`, no `aria-labelledby`, no `title`) | Regex / DOM: `<input` without any of these labeling mechanisms | −4 |
| **A11Y-N02** | `<img>` element missing `alt` attribute entirely | Regex: `<img\s` without `alt=` | −3 |
| **A11Y-N03** | `<img>` with non-descriptive `alt` text: `alt="image"`, `alt="photo"`, `alt="img"` | Regex: `alt="(image\|photo\|img\|picture\|icon)"` | −2 |
| **A11Y-N04** | `outline: none` or `outline: 0` without a replacement focus indicator | Regex: `outline:\s*(none\|0)` without subsequent `outline:` in same selector | −3 |
| **A11Y-N05** | Icon-only button or link with no accessible name (no text, no `aria-label`, no `.screen-reader-text`) | LLM / regex: `<button` or `<a` with only an icon class and no text content or ARIA | −4 |
| **A11Y-N06** | Click-only handler (`onclick`) on a `<div>` or `<span>` without `role="button"` and keyboard handler | Regex: `onclick=` on `<div\|<span` without `role="button"\|onkeydown\|onkeypress` | −3 |
| **A11Y-N07** | `tabindex` value greater than 0 (disrupts natural tab order) | Regex: `tabindex="[1-9][0-9]*"` | −2 |
| **A11Y-N08** | Placeholder text used as the only accessible label for an input | LLM / regex: `<input.*placeholder=` without any `<label` association | −3 |
| **A11Y-N09** | Admin notice output without `role="alert"` or `aria-live` | Regex: `class="notice` without `role=` attribute | −1 |
| **A11Y-N10** | `display: none` applied to a skip link (hides from screen readers entirely) | Regex: `.skip-link.*display:\s*none\|display:\s*none.*skip` | −3 |
| **A11Y-N11** | Heading elements used out of logical order (e.g., `<h4>` without preceding `<h3>`) | LLM / DOM: heading hierarchy analysis | −2 |
| **A11Y-N12** | Color used as the sole conveyor of information (no icon, text, or pattern alternative) | LLM analysis of CSS + HTML | −2 |
| **A11Y-N13** | Dismiss button for notice uses only `×` or `✕` without `aria-label` | Regex: `<button.*×\|×.*<button` without `aria-label=` | −2 |

### C. Dimension 7 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 18   # sum of all positive weights
D7 = clamp(raw / max_positive * 10, 0, 10)
```

**N/A rule:** If the code contains no HTML output whatsoever (pure logic, no templates), mark N/A.

---

## Dimension 8: Error Handling

**Scope:** `WP_Error` usage, `is_wp_error()` guards, graceful failure, validation before use, type safety, no silent failures.  
**Authority:** [WP_Error Reference](https://developer.wordpress.org/reference/classes/wp_error/), [WordPress PHP Coding Standards](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/php/), [szepeviktor/phpstan-wordpress](https://github.com/szepeviktor/phpstan-wordpress)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **ERR-P01** | `WP_Error` returned (not thrown as exception) to signal errors from functions | Regex: `return new WP_Error\s*\(` | +3 |
| **ERR-P02** | `is_wp_error()` called before using return value of any WP function that can return `WP_Error` | PHPCS / PHPStan; regex: `is_wp_error\s*\(` before using result | +3 |
| **ERR-P03** | `WP_Error` includes machine-readable error code (first arg) and human-readable message (second arg) | Regex: `new WP_Error\s*\(\s*'[a-z_]+'` | +1 |
| **ERR-P04** | `WP_Error` includes `status` in data array for REST/HTTP contexts | Regex: `new WP_Error\s*\(.*array\s*\(.*'status'` | +1 |
| **ERR-P05** | PHP type declarations on all function parameters | PHPStan / PHPCS; regex: `function \w+\s*\(\s*\w+\s+\$` | +2 |
| **ERR-P06** | PHP return type declarations on all functions | PHPStan; regex: `function.*\):\s*(string\|int\|bool\|array\|void\|WP_Error\|WP_Post)` | +2 |
| **ERR-P07** | `declare(strict_types=1)` present in files with typed logic | Regex: `declare\s*\(\s*strict_types\s*=\s*1\s*\)` | +1 |
| **ERR-P08** | `try`/`catch` used for third-party library calls that may throw exceptions | Regex: `try\s*\{.*catch\s*\(` | +2 |
| **ERR-P09** | Input validated before use (type check, range check, safelist check) — not only sanitized | LLM: `is_array`, `is_numeric`, `is_email`, `in_array` checks before use | +2 |
| **ERR-P10** | `wp_die()` used with appropriate HTTP status code on permission/error failure | Regex: `wp_die\s*\(.*,.*,\s*array\s*\(.*'response'` or numeric status | +1 |
| **ERR-P11** | `$wpdb->last_error` or `$wpdb->show_errors()` used appropriately in development | Regex: `$wpdb->last_error` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **ERR-N01** | Return value of function that can return `WP_Error` used without `is_wp_error()` check | PHPStan `szepeviktor/phpstan-wordpress`; LLM analysis | −3 |
| **ERR-N02** | `@` error suppression operator used (silences errors without handling them) | PHPCS `WordPress.PHP.NoSilencedErrors` | −3 |
| **ERR-N03** | `die()` / `exit()` used in plugin code outside of direct-file-access guards or `uninstall.php` | Regex: `\bdie\s*\(\|\bexit\s*\(` not in `if.*ABSPATH\|WP_UNINSTALL` context | −2 |
| **ERR-N04** | `trigger_error()` used instead of `WP_Error` or exceptions | Regex: `trigger_error\s*\(` | −2 |
| **ERR-N05** | No type declarations on function parameters or return type | PHPStan level 0+ | −1 |
| **ERR-N06** | `is_wp_error()` check present but error silently discarded (not propagated or logged) | LLM: `if ( is_wp_error(…) ) { return; }` with no logging or error passing | −1 |
| **ERR-N07** | Functions return mixed types without documentation (e.g., returns `false` on error, `int` on success, `array` on another state, no docblock) | LLM / PHPStan | −1 |
| **ERR-N08** | Type-unsafe comparison: `== false` or `== null` instead of `=== false` / `=== null` | PHPCS `WordPress.PHP.StrictComparisons`; regex: `==\s*false\|==\s*null` | −1 |
| **ERR-N09** | `catch ( \Exception $e ) {}` empty catch block (swallows exception silently) | Regex: `catch\s*\([^)]+\)\s*\{\s*\}` | −2 |
| **ERR-N10** | `wp_send_json_success()` / `wp_send_json_error()` used without `wp_die()` after (in non-REST AJAX context) | Regex: `wp_send_json.*success\|wp_send_json.*error` without `wp_die\s*\(\)` | −1 |
| **ERR-N11** | `file_get_contents()`, `wp_remote_get()`, or other I/O without error checking | LLM: I/O call without subsequent error check | −2 |
| **ERR-N12** | PHP 5-style `mysql_*` error handling used | Regex: `mysql_error\s*\(\|mysqli_error\s*\(` | −2 |

### C. Dimension 8 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 19   # sum of all positive weights
D8 = clamp(raw / max_positive * 10, 0, 10)
```

---

## Dimension 9: Code Structure

**Scope:** Hook patterns, plugin lifecycle (activation/deactivation/uninstall), REST API registration patterns, OOP patterns, separation of concerns.  
**Authority:** [WordPress Plugin Handbook — Hooks](https://developer.wordpress.org/plugins/hooks/), [WordPress REST API Handbook](https://developer.wordpress.org/rest-api/), [WordPress Plugin Handbook — Activation/Deactivation](https://developer.wordpress.org/plugins/plugin-basics/activation-deactivation-hooks/)

### A. Positive Signals

| ID | Description | Detection Method | Weight |
|---|---|---|---|
| **STR-P01** | All filter callbacks return a value (never implicit `null`) | PHPStan; LLM: `add_filter` callbacks all have `return` statement | +3 |
| **STR-P02** | Named functions (not closures) used as hook callbacks for removability | Regex: `add_action\s*\(.*,\s*'[a-z_]+'` or `array( $obj, 'method' )` — not `function()` | +2 |
| **STR-P03** | Custom hook names prefixed with plugin slug and use `snake_case` | PHPCS `WordPress.NamingConventions.PrefixAllGlobals`, `WordPress.NamingConventions.ValidHookName`; regex: `do_action\s*\(\s*'[a-z]+_` | +2 |
| **STR-P04** | Dynamic hook names use string interpolation not concatenation: `"{$var}_suffix"` not `$var . '_suffix'` | Regex: absence of `do_action\s*\(\s*\$\w+\s*\.\s*'` | +1 |
| **STR-P05** | `register_activation_hook()` present and flushes rewrite rules / sets defaults | Regex: `register_activation_hook\s*\(` with `flush_rewrite_rules\|add_option` | +2 |
| **STR-P06** | `register_deactivation_hook()` present and clears scheduled events / flushes rewrites | Regex: `register_deactivation_hook\s*\(` with `wp_clear_scheduled_hook\|flush_rewrite_rules` | +1 |
| **STR-P07** | `uninstall.php` exists with `WP_UNINSTALL_PLUGIN` constant check; OR `register_uninstall_hook()` used | Regex in uninstall.php: `defined\s*\(\s*'WP_UNINSTALL_PLUGIN'\s*\)` | +2 |
| **STR-P08** | REST API routes registered via `rest_api_init` hook | Regex: `add_action\s*\(\s*'rest_api_init'` | +1 |
| **STR-P09** | REST API endpoints have `permission_callback` defined | Regex: `'permission_callback'` in `register_rest_route` arg array | +3 |
| **STR-P10** | REST API `args` array with `type`, `sanitize_callback`, `validate_callback` on all parameters | Regex: `'args'\s*=>\s*array.*'type'\s*=>.*'sanitize_callback'\|'validate_callback'` | +2 |
| **STR-P11** | `WP_REST_Controller` subclass pattern used for REST endpoints | Regex: `class\s+\w+\s+extends\s+WP_REST_Controller` | +2 |
| **STR-P12** | Classes have single responsibility; one class per file | LLM: class structure and file scope analysis | +1 |
| **STR-P13** | `rest_ensure_response()` or `WP_REST_Response` used to return REST data | Regex: `rest_ensure_response\s*\(\|new WP_REST_Response\s*\(` | +1 |

### B. Negative Signals

| ID | Description | Detection Method | Penalty |
|---|---|---|---|
| **STR-N01** | Filter callback returns nothing (implicit `null`) — corrupts filtered value | PHPStan; LLM: `add_filter` callback without `return` | −5 |
| **STR-N02** | Closure registered as hook callback (cannot be removed with `remove_action/filter`) | Regex: `add_action\s*\(\s*'[^']+'\s*,\s*function\s*\(` | −2 |
| **STR-N03** | Custom hook name not prefixed or uses reserved `wp_` / `wordpress_` prefix | PHPCS `WordPress.NamingConventions.PrefixAllGlobals`; regex: `do_action\s*\(\s*'wp_\|apply_filters\s*\(\s*'wp_` | −3 |
| **STR-N04** | Hook priority > 100 without code comment explaining why | LLM: `add_action\s*\(.*,.*,\s*[1-9][0-9]{2,}\s*\)` without inline comment | −1 |
| **STR-N05** | `echo`/`print` inside a filter callback (corrupts output buffer) | LLM: `echo\|print` statement inside `add_filter` callback body | −3 |
| **STR-N06** | User data deleted in `register_deactivation_hook()` (should be in uninstall only) | LLM: `delete_option\|DROP TABLE\|$wpdb->delete` in deactivation callback | −3 |
| **STR-N07** | `uninstall.php` missing `WP_UNINSTALL_PLUGIN` guard (direct execution possible) | Regex in uninstall.php: absence of `WP_UNINSTALL_PLUGIN` check | −3 |
| **STR-N08** | REST route registered outside `rest_api_init` hook | LLM: `register_rest_route\s*\(` not inside `add_action.*rest_api_init` | −2 |
| **STR-N09** | REST endpoint missing `permission_callback` (publicly accessible unintentionally) | Regex: `register_rest_route\s*\(` without `permission_callback` key; WPCS/LLM | −5 |
| **STR-N10** | `__return_true` used as `permission_callback` on write (POST/PUT/DELETE) endpoint | Regex: `'methods'.*CREATABLE\|EDITABLE\|DELETABLE.*__return_true\|__return_true.*methods.*POST\|PUT\|DELETE` | −4 |
| **STR-N11** | REST callback uses `wp_send_json()` or `die()` instead of returning data | Regex: `wp_send_json\s*\(\|die\s*\(` inside `register_rest_route` callback | −3 |
| **STR-N12** | `wp_ajax_*` handler does not call `wp_die()` at the end | Regex: `add_action\s*\(\s*'wp_ajax_` callback without `wp_die\s*\(\)` | −1 |
| **STR-N13** | `flush_rewrite_rules()` called inside `init` hook on every request | Regex: `add_action\s*\(\s*'init'.*flush_rewrite_rules\s*\(` | −2 |
| **STR-N14** | Dynamic hook name built via string concatenation (`$a . '_' . $b`) instead of interpolation | Regex: `do_action\s*\(\s*\$\w+\s*\.\s*['"_]` | −1 |

### C. Dimension 9 Scoring Formula

```
raw = sum(positive_weights) - sum(negative_penalties)
max_positive = 23   # sum of all positive weights
D9 = clamp(raw / max_positive * 10, 0, 10)
```

**Critical floor rule:** If STR-N01 (filter returns nothing) or STR-N09 (REST endpoint without permission_callback) is present, D9 **cannot exceed 4**.

---

## D. Overall Score Formula

```python
# Dimension scores (each 0-10)
dimensions = {
    'D1_wpcs':      {'score': D1, 'weight': 0.10},
    'D2_security':  {'score': D2, 'weight': 0.20},
    'D3_sql':       {'score': D3, 'weight': 0.15},
    'D4_perf':      {'score': D4, 'weight': 0.10},
    'D5_wp_api':    {'score': D5, 'weight': 0.10},
    'D6_i18n':      {'score': D6, 'weight': 0.10},
    'D7_a11y':      {'score': D7, 'weight': 0.08},
    'D8_errors':    {'score': D8, 'weight': 0.10},
    'D9_structure': {'score': D9, 'weight': 0.07},
}

# Handle N/A dimensions
applicable = {k: v for k, v in dimensions.items() if v['score'] is not None}
total_weight = sum(v['weight'] for v in applicable.values())

# Normalize weights and compute score
overall = sum(
    v['score'] * (v['weight'] / total_weight)
    for v in applicable.values()
) * 10  # scale to 0-100
```

**Rounding:** Round `overall` to one decimal place.

### Score Interpretation Bands

| Score | Grade | Interpretation |
|---|---|---|
| 90–100 | Excellent | Production-ready; follows all WPCS, security, and WP API best practices |
| 75–89 | Good | Minor issues only; no critical vulnerabilities; suitable with small revisions |
| 60–74 | Acceptable | Some deficiencies; usable but needs review before production deployment |
| 40–59 | Poor | Notable security or quality gaps; significant revision required |
| 20–39 | Bad | Multiple critical issues; not safe for production |
| 0–19 | Failing | Serious security vulnerabilities and/or fundamental API misuse |

---

## E. Automation Mapping Table

### Definitions

| Method | Description |
|---|---|
| **PHPCS-WPCS** | Detected by `phpcs --standard=WordPress` using named sniff |
| **PHPCS-VIP** | Detected by `phpcs --standard=WordPress-VIP-Go` (Automattic VIP standard) |
| **PHPCS-Security** | Detected by `phpcs --standard=Security` (pheromone/phpcs-security-audit) |
| **PHPStan** | Detected by PHPStan with `szepeviktor/phpstan-wordpress` at level ≥ 5 |
| **Regex** | Simple regex pattern applied to source text |
| **Regex+** | Regex with context awareness (e.g., must not be inside string or comment) |
| **AST** | Requires parsing into Abstract Syntax Tree (e.g., PHP-Parser) |
| **LLM** | Requires language model judgment; rule-based automation not sufficient |
| **DOM** | Requires parsing HTML output |
| **File** | Requires inspecting file/directory structure |

### Dimension 1: WPCS Compliance

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| WPCS-P01 | Regex+ | Detect space-only indentation |
| WPCS-P02 | PHPCS-WPCS | `WordPress.NamingConventions.ValidFunctionName`, `ValidVariableName` |
| WPCS-P03 | PHPCS-WPCS | `WordPress.NamingConventions.ValidFunctionName` (class context) |
| WPCS-P04 | Regex | `define\s*\(\s*'[A-Z][A-Z0-9_]+'` |
| WPCS-P05 | PHPCS-WPCS | `WordPress.PHP.YodaConditions` |
| WPCS-P06 | PHPCS-WPCS | `WordPress.WhiteSpace.ControlStructureSpacing` |
| WPCS-P07 | PHPCS-WPCS | `WordPress.WhiteSpace.OperatorSpacing` |
| WPCS-P08 | PHPCS-WPCS | `WordPress.NamingConventions.PrefixAllGlobals` |
| WPCS-P09 | PHPCS-WPCS | `WordPress.Files.FileName` |
| WPCS-P10 | PHPCS-WPCS | `WordPress.PHP.StrictInArray` |
| WPCS-P11 | PHPCS-WPCS | `Generic.ControlStructures.InlineControlStructure` |
| WPCS-P12 | Regex | `else\s+if\b` (negative: absent) |
| WPCS-N01 | PHPCS-WPCS | `Generic.PHP.DisallowShortOpenTag` |
| WPCS-N02 | PHPCS-WPCS | `WordPress.NamingConventions.ValidFunctionName` |
| WPCS-N03 | PHPCS-WPCS | `WordPress.NamingConventions.PrefixAllGlobals` |
| WPCS-N04 | PHPCS-WPCS | `Squiz.Scope.MemberVarScope` |
| WPCS-N05 | PHPCS-WPCS | `Squiz.Scope.MethodScope` |
| WPCS-N06 | Regex | `\?>\s*$` |
| WPCS-N07 | Regex | `\?:` in expression |
| WPCS-N08 | PHPCS-WPCS | `WordPress.PHP.StrictInArray` |
| WPCS-N09 | PHPCS-WPCS | `WordPress.PHP.StrictComparisons` |
| WPCS-N10 | PHPCS-WPCS | `WordPress.PHP.DontExtract` |
| WPCS-N11 | PHPCS-WPCS | `WordPress.PHP.RestrictedPHPFunctions` |
| WPCS-N12 | PHPCS-WPCS | `WordPress.PHP.RestrictedPHPFunctions` |
| WPCS-N13 | Regex | `\bgoto\b` |
| WPCS-N14 | PHPCS-WPCS | `WordPress.PHP.NoSilencedErrors` |
| WPCS-N15 | PHPCS-WPCS | `WordPress.PHP.DevelopmentFunctions` |
| WPCS-N16 | PHPCS-WPCS | `WordPress.DateTime.CurrentTimeTimestamp` |
| WPCS-N17 | PHPCS-WPCS | `WordPress.CodeAnalysis.AssignmentInTernaryCondition` |
| WPCS-N18 | PHPCS-WPCS | `WordPress.CodeAnalysis.EscapedNotTranslated` |
| WPCS-N19 | PHPCS-WPCS | `WordPress.PHP.PregQuoteDelimiter` |
| WPCS-N20 | PHPCS-WPCS | `WordPress.WP.GlobalVariablesOverride` |

### Dimension 2: Security

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| SEC-P01 | PHPCS-WPCS | `WordPress.Security.EscapeOutput` (absence of violations) |
| SEC-P02 | Regex+ | Order-sensitive `wp_unslash` → `sanitize_*` pattern |
| SEC-P03 | PHPCS-WPCS | `WordPress.Security.ValidatedSanitizedInput` (absence of violations) |
| SEC-P04 | PHPCS-WPCS | `WordPress.Security.NonceVerification` (absence of violations) |
| SEC-P05 | Regex + LLM | `current_user_can\s*\(` before data mutation |
| SEC-P06 | PHPCS-WPCS | `WordPress.Security.SafeRedirect` (absence of violations) |
| SEC-P07 | Regex | `wp_check_filetype_and_ext\s*\(` |
| SEC-P08 | Regex | `wp_handle_upload\s*\(` |
| SEC-P09 | Regex | `is_email\s*\(` |
| SEC-P10 | LLM | Safelist + `in_array` guard for ORDER BY |
| SEC-P11 | Regex | `absint\s*\(` |
| SEC-P12 | Regex | `esc_html__\s*\(\|esc_attr__\s*\(` |
| SEC-P13 | LLM | Output context matching |
| SEC-N01 | PHPCS-WPCS | `WordPress.Security.EscapeOutput` |
| SEC-N02 | PHPCS-WPCS | `WordPress.Security.ValidatedSanitizedInput` |
| SEC-N03 | PHPCS-WPCS | `WordPress.Security.NonceVerification` |
| SEC-N04 | LLM | Privilege check presence analysis |
| SEC-N05 | PHPCS-WPCS | `WordPress.Security.SafeRedirect` |
| SEC-N06 | PHPCS-Security | `Security.BadFunctions.PHPInternalFunctions` + Regex |
| SEC-N07 | PHPCS-Security | `Security.BadFunctions.PHPInternalFunctions` |
| SEC-N08 | PHPCS-Security | `Security.BadFunctions.FilesystemFunctions` |
| SEC-N09 | Regex | `$_FILES\[.*\]\['type'\]` in if condition |
| SEC-N10 | LLM | Form template + nonce presence |
| SEC-N11 | Regex | `wp_ajax_nopriv_` without `check_ajax_referer` |
| SEC-N12 | Regex | `current_user_can.*'(administrator\|editor)'` |
| SEC-N13 | LLM | `is_admin()` as security gate |
| SEC-N14 | PHPCS-WPCS | `WordPress.Security.PluginMenuSlug` |
| SEC-N15 | LLM | `href.*esc_html(` context |
| SEC-N16 | LLM | `href.*esc_attr(` context |
| SEC-N17 | LLM | Double-escaping detection |
| SEC-N18 | Regex | `\$_REQUEST\b` |
| SEC-N19 | PHPCS-Security | `Security.BadFunctions.SystemExecFunctions` |
| SEC-N20 | Regex | `preg_replace\s*\(.*\/e` |

### Dimension 3: SQL Safety

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| SQL-P01 | PHPCS-WPCS | `WordPress.DB.PreparedSQL` (absence of violations) |
| SQL-P02 | Regex | `$wpdb->(insert\|update\|delete)\s*\(` |
| SQL-P03 | Regex+ | `esc_like` before `LIKE %s` |
| SQL-P04 | Regex | `new WP_Query\|get_posts\s*\(` |
| SQL-P05 | Regex | `prepare.*%i` |
| SQL-P06 | Regex (negative) | Absence of `".*wp_[a-z_]+` in SQL |
| SQL-P07 | LLM | ORDER BY safelist validation |
| SQL-P08 | LLM + Regex | LIMIT presence in SELECT queries |
| SQL-P09 | Regex | `$wpdb->last_error` |
| SQL-N01 | PHPCS-WPCS | `WordPress.DB.PreparedSQL` |
| SQL-N02 | Regex | Concatenation in SQL string |
| SQL-N03 | PHPCS-WPCS | `WordPress.DB.PreparedSQL` + `WordPress.Security.ValidatedSanitizedInput` |
| SQL-N04 | Regex+ | LIKE without esc_like |
| SQL-N05 | PHPCS-WPCS | `WordPress.DB.PreparedSQLPlaceholders` |
| SQL-N06 | PHPCS-WPCS | `WordPress.DB.PreparedSQLPlaceholders` |
| SQL-N07 | Regex | `->escape\s*\(` |
| SQL-N08 | Regex | `prepare.*esc_sql` |
| SQL-N09 | PHPCS-WPCS | `WordPress.DB.RestrictedFunctions`, `WordPress.DB.RestrictedClasses` |
| SQL-N10 | PHPCS-WPCS | `WordPress.WP.DiscouragedFunctions` |
| SQL-N11 | Regex | `".*wp_[a-z_]+"` in SQL context |
| SQL-N12 | PHPCS-WPCS | `WordPress.WP.PostsPerPage` |
| SQL-N13 | Regex | `'orderby'\s*=>\s*'rand'` |
| SQL-N14 | Regex | `'suppress_filters'\s*=>\s*true` |
| SQL-N15 | Regex | `SELECT\s+\*` in `get_results` |
| SQL-N16 | LLM | Unbounded SELECT without LIMIT |
| SQL-N17 | Regex | `prepare\s*\(\s*\$_(GET\|POST)` |

### Dimension 4: Performance

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| PERF-P01 | Regex+ | `wp_cache_get` + `wp_cache_set` pattern |
| PERF-P02 | Regex | `set_transient` with 3 args |
| PERF-P03 | Regex | `false !== get_transient` |
| PERF-P04 | LLM + Regex | HTTP + transient in same scope |
| PERF-P05 | Regex | `'no_found_rows'\s*=>\s*true` |
| PERF-P06 | Regex | `'fields'\s*=>\s*'ids'` |
| PERF-P07 | Regex | `update_post_meta_cache.*false\|update_post_term_cache.*false` |
| PERF-P08 | Regex | `add_option.*false\|update_option.*false` (autoload) |
| PERF-P09 | PHPCS-WPCS | `WordPress.WP.EnqueuedResourceParameters` |
| PERF-P10 | LLM | Conditional enqueueing |
| PERF-P11 | Regex | `wp_remote_get.*'timeout'` |
| PERF-P12 | Regex | `wp_remote_retrieve_response_code` |
| PERF-N01 | LLM + Regex | DB/meta calls inside loops |
| PERF-N02 | Regex | `$wpdb->` inside `foreach\|while` |
| PERF-N03 | Regex | `set_transient` with 2 args only |
| PERF-N04 | Regex | `wp_cache_flush()` in hook |
| PERF-N05 | LLM | Uncached remote HTTP in hooks |
| PERF-N06 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| PERF-N07 | Regex | `curl_exec\s*\(` |
| PERF-N08 | PHPCS-WPCS | `WordPress.WP.EnqueuedResources` |
| PERF-N09 | Regex | `add_action.*init.*flush_rewrite_rules` |
| PERF-N10 | LLM + Regex | `get_option` inside loop |
| PERF-N11 | LLM | Large data autoloaded without `false` |
| PERF-N12 | Regex | `wp_remote_get` without `is_wp_error` |
| PERF-N13 | Regex | `'orderby'\s*=>\s*'rand'` |

### Dimension 5: WP API Usage

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| WAPI-P01 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-P02 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-P03 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-P04 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-P05 | PHPCS-WPCS | `WordPress.DateTime.RestrictedFunctions` |
| WAPI-P06 | PHPCS-WPCS | `WordPress.WP.EnqueuedResources`, `EnqueuedResourceParameters` |
| WAPI-P07 | Regex | `WP_Filesystem\(\)\|$wp_filesystem->` |
| WAPI-P08 | LLM | Hook-deferred initialization |
| WAPI-P09 | Regex | `add_action.*init.*register_post_type` |
| WAPI-P10 | Regex | `wp_localize_script\|wp_add_inline_script` |
| WAPI-P11 | Regex | `plugins_url\|plugin_dir_url` |
| WAPI-P12 | Regex (negative) | Absence of `query_posts\s*\(` |
| WAPI-P13 | PHPCS-WPCS | `WordPress.WP.DeprecatedFunctions` (absence of violations) |
| WAPI-N01 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-N02 | Regex | `curl_init\|curl_exec` |
| WAPI-N03 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-N04 | PHPCS-WPCS | `WordPress.WP.AlternativeFunctions` |
| WAPI-N05 | PHPCS-WPCS | `WordPress.DateTime.RestrictedFunctions` |
| WAPI-N06 | PHPCS-WPCS | `WordPress.WP.DiscouragedFunctions` |
| WAPI-N07 | PHPCS-WPCS | `WordPress.WP.DeprecatedFunctions` |
| WAPI-N08 | PHPCS-WPCS | `WordPress.WP.DeprecatedClasses` |
| WAPI-N09 | Regex | `require.*phpmailer\|jquery.*cdn` |
| WAPI-N10 | Regex | `\bfopen\s*\(\|file_put_contents\s*\(` |
| WAPI-N11 | Regex | `session_start\s*\(\|\$_SESSION` |
| WAPI-N12 | LLM | Initialization outside hooks |
| WAPI-N13 | LLM | Menu registration without capability in callback |
| WAPI-N14 | PHPCS-WPCS | `WordPress.WP.DeprecatedFunctions` |

### Dimension 6: i18n / l10n

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| I18N-P01 | PHPCS-WPCS + LLM | `WordPress.WP.I18n` + echo/print analysis |
| I18N-P02 | PHPCS-WPCS | `WordPress.WP.I18n` |
| I18N-P03 | Regex | `_n\s*\(\|_nx\s*\(` presence |
| I18N-P04 | Regex | `_x\s*\(\|_ex\s*\(` presence |
| I18N-P05 | Regex | `esc_html__\s*\(\|esc_html_e\s*\(` |
| I18N-P06 | Regex | `esc_attr__\s*\(\|esc_attr_e\s*\(` |
| I18N-P07 | Regex | `%1\$s.*%2\$s` in translation strings |
| I18N-P08 | Regex | `\/\*.*translators` before `__\|_e` with placeholders |
| I18N-P09 | Regex | `number_format_i18n\s*\(` |
| I18N-P10 | Regex | `date_i18n\s*\(` |
| I18N-N01 | PHPCS-WPCS | `WordPress.WP.I18n` |
| I18N-N02 | PHPCS-WPCS | `WordPress.WP.I18n` |
| I18N-N03 | PHPCS-WPCS | `WordPress.WP.I18n` |
| I18N-N04 | Regex | Concatenation around `__\s*\(` |
| I18N-N05 | Regex | Multiple `%s` without positional args |
| I18N-N06 | Regex | `if.*===\s*1.*__(` with else `__(` |
| I18N-N07 | Regex | `%s\|%d` in `__()` without translators comment |
| I18N-N08 | Regex | `echo.*number_format\s*\(` |
| I18N-N09 | Regex | `echo.*\bdate\s*\(\|echo.*\bgmdate\s*\(` |
| I18N-N10 | Regex | HTML tags inside `__()` strings |
| I18N-N11 | Regex | `__\s*\(\s*''` |
| I18N-N12 | LLM | `_e()` in attribute context |
| I18N-N13 | LLM + Regex | `echo\s+['"][A-Z]` without translation wrapper |

### Dimension 7: Accessibility

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| A11Y-P01 | DOM / Regex | `<label\s+for=` + matching `id=` |
| A11Y-P02 | Regex | `.screen-reader-text` + skip link pattern |
| A11Y-P03 | Regex | `aria-label=\|aria-labelledby=` |
| A11Y-P04 | Regex | `role="alert"\|aria-live=` |
| A11Y-P05 | Regex | All `<img` have `alt=` |
| A11Y-P06 | Regex | `<fieldset.*<legend` |
| A11Y-P07 | Regex | `<main\b\|<nav\b\|<aside\b` |
| A11Y-P08 | LLM | `outline:none` with replacement |
| A11Y-P09 | Regex | `autocomplete=` on personal fields |
| A11Y-P10 | Regex | `aria-required.*required\|required.*aria-required` |
| A11Y-P11 | Regex | `aria-describedby=` on inputs |
| A11Y-P12 | Regex | `aria-expanded=\|aria-controls=` |
| A11Y-N01 | DOM / LLM | `<input` without labeling mechanism |
| A11Y-N02 | Regex | `<img\s` without `alt=` |
| A11Y-N03 | Regex | `alt="(image\|photo\|img)"` |
| A11Y-N04 | Regex | `outline:\s*(none\|0)` without replacement |
| A11Y-N05 | LLM | Icon-only interactive element |
| A11Y-N06 | Regex | `onclick=` on `<div\|<span` without role/keyboard |
| A11Y-N07 | Regex | `tabindex="[1-9]` |
| A11Y-N08 | LLM | Placeholder-only input label |
| A11Y-N09 | Regex | `.notice` without `role=` |
| A11Y-N10 | Regex | `.skip-link.*display:\s*none` |
| A11Y-N11 | LLM | Heading hierarchy |
| A11Y-N12 | LLM | Color-only information |
| A11Y-N13 | Regex | Dismiss button `×` without `aria-label` |

### Dimension 8: Error Handling

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| ERR-P01 | Regex | `return new WP_Error\s*\(` |
| ERR-P02 | PHPStan / Regex | `is_wp_error\s*\(` before result use |
| ERR-P03 | Regex | `new WP_Error\s*\(\s*'[a-z_]+'` |
| ERR-P04 | Regex | `new WP_Error.*'status'` |
| ERR-P05 | PHPStan | Type declarations on parameters |
| ERR-P06 | PHPStan | Return type declarations |
| ERR-P07 | Regex | `declare\s*\(\s*strict_types\s*=\s*1` |
| ERR-P08 | Regex | `try\s*\{.*catch` |
| ERR-P09 | LLM | Validation before use |
| ERR-P10 | Regex | `wp_die\s*\(.*'response'` or status code |
| ERR-P11 | Regex | `$wpdb->last_error` |
| ERR-N01 | PHPStan | `szepeviktor/phpstan-wordpress` level 5 |
| ERR-N02 | PHPCS-WPCS | `WordPress.PHP.NoSilencedErrors` |
| ERR-N03 | Regex + LLM | `\bdie\s*\(\|\bexit\s*\(` outside guards |
| ERR-N04 | Regex | `trigger_error\s*\(` |
| ERR-N05 | PHPStan | Missing type hints |
| ERR-N06 | LLM | Silent error discard |
| ERR-N07 | LLM + PHPStan | Undocumented mixed return types |
| ERR-N08 | PHPCS-WPCS | `WordPress.PHP.StrictComparisons` |
| ERR-N09 | Regex | `catch.*\{\s*\}` (empty catch) |
| ERR-N10 | Regex | `wp_send_json_success` without `wp_die` |
| ERR-N11 | LLM | I/O without error check |
| ERR-N12 | Regex | `mysql_error\s*\(` |

### Dimension 9: Code Structure

| Check ID | Automation Method | Tool / Sniff |
|---|---|---|
| STR-P01 | PHPStan / LLM | Filter callback return analysis |
| STR-P02 | Regex | `add_action.*,\s*'[a-z_]+'` (named, not closure) |
| STR-P03 | PHPCS-WPCS | `WordPress.NamingConventions.PrefixAllGlobals`, `ValidHookName` |
| STR-P04 | Regex | Interpolation `"{$var}_hook"` not concatenation |
| STR-P05 | Regex | `register_activation_hook` |
| STR-P06 | Regex | `register_deactivation_hook` |
| STR-P07 | File + Regex | `uninstall.php` + `WP_UNINSTALL_PLUGIN` |
| STR-P08 | Regex | `add_action.*rest_api_init` |
| STR-P09 | Regex | `'permission_callback'` in route args |
| STR-P10 | Regex | `'args'.*'type'.*'sanitize_callback'` |
| STR-P11 | Regex | `extends WP_REST_Controller` |
| STR-P12 | LLM | Single responsibility per class |
| STR-P13 | Regex | `rest_ensure_response\|new WP_REST_Response` |
| STR-N01 | PHPStan / LLM | Filter without return |
| STR-N02 | Regex | `add_action.*function\s*\(` (closure) |
| STR-N03 | PHPCS-WPCS | `WordPress.NamingConventions.PrefixAllGlobals` |
| STR-N04 | LLM + Regex | High priority without comment |
| STR-N05 | LLM | `echo` in filter callback |
| STR-N06 | LLM | Data deletion in deactivation |
| STR-N07 | File + Regex | `uninstall.php` without guard |
| STR-N08 | LLM | `register_rest_route` outside `rest_api_init` |
| STR-N09 | Regex | `register_rest_route` without `permission_callback` |
| STR-N10 | Regex | `__return_true` on write endpoint |
| STR-N11 | Regex | `wp_send_json\|die` in REST callback |
| STR-N12 | Regex | `wp_ajax_` handler without `wp_die` |
| STR-N13 | Regex | `add_action.*init.*flush_rewrite_rules` |
| STR-N14 | Regex | Hook name via concatenation |

---

## F. Ground Truth Scoring Procedure

This procedure defines how to compute a ground truth score for a PHP function or file. Use it in `eval_gen.py` (to score generated code) and `eval_judge.py` (to generate a ground truth for judge model training).

### Step 1: Setup

```bash
# Install PHPCS + WordPress Coding Standards
composer require --dev \
    squizlabs/php_codesniffer \
    wp-coding-standards/wpcs \
    phpcompatibility/phpcompatibility-wp \
    automattic/vipwpcs \
    pheromone/phpcs-security-audit

./vendor/bin/phpcs --config-set installed_paths \
    vendor/wp-coding-standards/wpcs,\
    vendor/automattic/vipwpcs,\
    vendor/phpcompatibility/phpcompatibility-wp,\
    vendor/phpcompatibility/phpcompatibility,\
    vendor/pheromone/phpcs-security-audit

# Install PHPStan
composer require --dev \
    phpstan/phpstan \
    szepeviktor/phpstan-wordpress \
    php-stubs/wordpress-stubs
```

### Step 2: Run Automated Tools

```bash
# 2a. PHPCS — full WordPress ruleset
./vendor/bin/phpcs \
    --standard=WordPress \
    --report=json \
    --report-file=phpcs_wordpress.json \
    "$TARGET_FILE"

# 2b. PHPCS — VIP strict standard (catches caching, slow queries, etc.)
./vendor/bin/phpcs \
    --standard=WordPressVIPMinimum \
    --report=json \
    --report-file=phpcs_vip.json \
    "$TARGET_FILE"

# 2c. PHPCS — Security audit
./vendor/bin/phpcs \
    --standard=Security \
    --report=json \
    --report-file=phpcs_security.json \
    "$TARGET_FILE"

# 2d. PHPStan — level 5 with WordPress stubs
./vendor/bin/phpstan analyse \
    --level=5 \
    --error-format=json \
    "$TARGET_FILE" \
    > phpstan.json 2>&1
```

### Step 3: Parse Tool Outputs

For each tool output, map reported errors/warnings to check IDs using the Automation Mapping Table (Section E). Each triggered check ID is flagged.

```python
def parse_phpcs_output(json_file: str) -> dict[str, int]:
    """Map PHPCS sniff names to flagged check IDs and counts."""
    sniff_to_check = {
        'WordPress.Security.EscapeOutput':          ['SEC-N01', 'SEC-N15', 'SEC-N16'],
        'WordPress.Security.ValidatedSanitizedInput': ['SEC-N02'],
        'WordPress.Security.NonceVerification':     ['SEC-N03'],
        'WordPress.Security.SafeRedirect':          ['SEC-N05'],
        'WordPress.DB.PreparedSQL':                 ['SQL-N01', 'SQL-N03'],
        'WordPress.DB.PreparedSQLPlaceholders':     ['SQL-N05', 'SQL-N06'],
        'WordPress.DB.RestrictedFunctions':         ['SQL-N09'],
        'WordPress.DB.RestrictedClasses':           ['SQL-N09'],
        'WordPress.WP.DiscouragedFunctions':        ['SQL-N10', 'WAPI-N06'],
        'WordPress.WP.DeprecatedFunctions':         ['WAPI-N07', 'WAPI-N14'],
        'WordPress.WP.DeprecatedClasses':           ['WAPI-N08'],
        'WordPress.WP.AlternativeFunctions':        ['WAPI-N01', 'WAPI-N02', 'WAPI-N03', 'WAPI-N04', 'WAPI-N05'],
        'WordPress.WP.EnqueuedResources':           ['PERF-N08', 'WAPI-P06'],
        'WordPress.WP.EnqueuedResourceParameters':  ['PERF-P09'],
        'WordPress.WP.PostsPerPage':                ['SQL-N12'],
        'WordPress.WP.I18n':                        ['I18N-N01', 'I18N-N02', 'I18N-N03'],
        'WordPress.WP.GlobalVariablesOverride':     ['WPCS-N20'],
        'WordPress.NamingConventions.PrefixAllGlobals': ['WPCS-N03', 'STR-N03'],
        'WordPress.NamingConventions.ValidFunctionName': ['WPCS-N02'],
        'WordPress.NamingConventions.ValidHookName': ['STR-N03'],
        'WordPress.PHP.YodaConditions':             ['WPCS-P05'],
        'WordPress.PHP.StrictInArray':              ['WPCS-N08'],
        'WordPress.PHP.StrictComparisons':          ['WPCS-N09', 'ERR-N08'],
        'WordPress.PHP.DontExtract':                ['WPCS-N10'],
        'WordPress.PHP.RestrictedPHPFunctions':     ['WPCS-N11', 'WPCS-N12'],
        'WordPress.PHP.NoSilencedErrors':           ['WPCS-N14', 'ERR-N02'],
        'WordPress.PHP.DevelopmentFunctions':       ['WPCS-N15'],
        'WordPress.DateTime.CurrentTimeTimestamp':  ['WPCS-N16'],
        'WordPress.DateTime.RestrictedFunctions':   ['WAPI-N05'],
        'WordPress.CodeAnalysis.AssignmentInTernaryCondition': ['WPCS-N17'],
        'WordPress.CodeAnalysis.EscapedNotTranslated': ['WPCS-N18'],
        'WordPress.PHP.PregQuoteDelimiter':         ['WPCS-N19'],
        'WordPress.WhiteSpace.ControlStructureSpacing': ['WPCS-P06'],
        'WordPress.WhiteSpace.OperatorSpacing':     ['WPCS-P07'],
        'WordPress.Files.FileName':                 ['WPCS-P09'],
        'Generic.PHP.DisallowShortOpenTag':         ['WPCS-N01'],
        'WordPress.Security.PluginMenuSlug':        ['SEC-N14'],
        'WordPress.DB.SlowDBQuery':                 ['PERF-N01'],
        'Security.BadFunctions.EasyXSS':            ['SEC-N01'],
        'Security.BadFunctions.PHPInternalFunctions': ['SEC-N06', 'SEC-N07'],
        'Security.BadFunctions.FilesystemFunctions': ['SEC-N08'],
        'Security.BadFunctions.SystemExecFunctions': ['SEC-N19'],
    }
    # Returns dict of check_id -> count_of_violations
    ...
```

### Step 4: Apply Regex Checks

Run the supplemental regex patterns (from Section E, Method = "Regex") against the raw PHP source for checks not covered by PHPCS.

```python
import re

REGEX_CHECKS = {
    # Positive checks (presence signals the positive)
    'SQL-P02':   r'\$wpdb->(insert|update|delete)\s*\(',
    'SQL-P03':   r'esc_like\s*\(',
    'SQL-P05':   r'prepare\s*\(.*%i',
    'PERF-P01':  r'wp_cache_get\s*\(',  # also requires wp_cache_set nearby
    'PERF-P02':  r'set_transient\s*\(\s*[^,]+,\s*[^,]+,\s*[^)]+\)',
    'PERF-P05':  r"'no_found_rows'\s*=>\s*true",
    'PERF-P06':  r"'fields'\s*=>\s*'ids'",
    'ERR-P01':   r'return new WP_Error\s*\(',
    'ERR-P02':   r'is_wp_error\s*\(',
    'ERR-P07':   r'declare\s*\(\s*strict_types\s*=\s*1\s*\)',
    'ERR-P08':   r'try\s*\{',
    'STR-P05':   r'register_activation_hook\s*\(',
    'STR-P06':   r'register_deactivation_hook\s*\(',
    'STR-P08':   r"add_action\s*\(\s*'rest_api_init'",
    'STR-P09':   r"'permission_callback'",
    'STR-P11':   r'extends\s+WP_REST_Controller',
    # Negative checks (presence signals the negative)
    'SQL-N04':   r'LIKE\s+%s',   # combined with absence of esc_like
    'SQL-N07':   r'->escape\s*\(',
    'SQL-N10':   r'query_posts\s*\(',
    'SQL-N11':   r'"[^"]*wp_[a-z_]+[^"]*"',
    'SQL-N12':   r"'posts_per_page'\s*=>\s*-1",
    'SQL-N13':   r"'orderby'\s*=>\s*'rand'",
    'SQL-N14':   r"'suppress_filters'\s*=>\s*true",
    'SEC-N18':   r'\$_REQUEST\b',
    'SEC-N20':   r'preg_replace\s*\(.*\/e["\']',
    'WPCS-N06':  r'\?>\s*$',
    'WPCS-N12':  r'\beval\s*\(',
    'WPCS-N13':  r'\bgoto\b',
    'WAPI-N11':  r'session_start\s*\(',
    'STR-N02':   r"add_action\s*\(\s*'[^']+'\s*,\s*function\s*\(",
    'STR-N09':   r'register_rest_route\s*\(',   # presence + absence of permission_callback
    'I18N-N03':  r'__\s*\(\s*"[^"]*\$[a-z_]',
    'I18N-N11':  r"__\s*\(\s*''",
    'A11Y-N02':  r'<img\s(?![^>]*\balt=)',
    'A11Y-N07':  r'tabindex="[1-9][0-9]*"',
    'A11Y-N09':  r'class="notice(?!.*role=)',
    'PERF-N03':  r"set_transient\s*\(\s*['\"\w]+\s*,\s*[^,)]+\s*\)",  # 2-arg form
    'PERF-N07':  r'curl_exec\s*\(',
}
```

### Step 5: LLM-Assisted Checks

For checks marked "LLM" in the Automation Mapping Table, submit the PHP source to a secondary LLM judge with targeted prompts:

```python
LLM_PROMPTS = {
    'SEC-P05': "Does this code call current_user_can() before any privileged write operation, "
               "database modification, or admin action? Answer YES/NO with evidence.",
    'SEC-P13': "Is the output context matched to the correct WordPress escaping function "
               "(esc_html for HTML content, esc_attr for attributes, esc_url for URLs, "
               "esc_js for JavaScript)? List any mismatches.",
    'SEC-N04': "Are there any privileged actions (database writes, option updates, post inserts) "
               "executed WITHOUT a preceding current_user_can() check? List each instance.",
    'SEC-N13': "Is is_admin() used as a security gate to protect data modification (not just "
               "to check context for UI purposes)? Answer YES/NO.",
    'STR-N01': "Do all add_filter() callbacks have a return statement? "
               "List any callbacks that return null or nothing.",
    'STR-N05': "Are there any echo or print statements inside add_filter() callback functions? "
               "List each instance.",
    'STR-N06': "Does any deactivation hook callback delete user data (delete_option, "
               "DROP TABLE, wpdb->delete)? This should only happen in uninstall. Answer YES/NO.",
    'PERF-N01': "Are any database or WordPress API functions (get_post_meta, get_the_terms, "
                "get_user_by, $wpdb->) called inside a foreach or while loop over posts "
                "that could cause N+1 queries? List each instance.",
    'ERR-N01': "Are there any calls to WordPress functions that can return WP_Error "
               "(wp_insert_post, wp_update_post, wp_remote_get, get_posts, etc.) whose "
               "return value is used WITHOUT checking is_wp_error() first? List each.",
    'I18N-N12': "Is _e() or _ex() used inside an HTML attribute context (e.g., "
                "placeholder, title, aria-label)? It should use esc_attr_e() instead.",
    'I18N-N13': "Are there echo or print statements outputting English string literals "
                "that are NOT wrapped in a translation function (__(), _e(), etc.)?",
    'A11Y-N01': "Do all <input>, <select>, and <textarea> elements have an associated "
                "<label for=...>, aria-label=, aria-labelledby=, or title= attribute?",
    'A11Y-N05': "Are there icon-only buttons or links (containing only an icon class "
                "or dashicon) WITHOUT any visible text, aria-label, aria-labelledby, "
                "or .screen-reader-text child element?",
    'WAPI-N12': "Is any WordPress initialization code (register_post_type, add_menu_page, "
                "wp_enqueue_script, register_taxonomy) called at file-include time rather "
                "than deferred inside an add_action() callback?",
}
```

### Step 6: Compute Per-Dimension Scores

```python
from math import floor

def compute_dimension_score(
    positive_hits: dict[str, bool],   # check_id -> triggered
    negative_hits: dict[str, bool],   # check_id -> triggered
    positive_weights: dict[str, int],
    negative_weights: dict[str, int],
    max_positive: int,
) -> float:
    """Compute 0-10 score for a single dimension."""
    raw = sum(positive_weights[k] for k, v in positive_hits.items() if v)
    raw -= sum(negative_weights[k] for k, v in negative_hits.items() if v)
    return max(0.0, min(10.0, (raw / max_positive) * 10))

# Apply dimension-specific floor rules
def apply_floor_rules(scores: dict[str, float], negative_hits: dict[str, bool]) -> dict[str, float]:
    # Security: catastrophic vulnerability cap
    if any(negative_hits.get(k) for k in ['SEC-N01','SEC-N03','SEC-N04','SEC-N06','SEC-N08','SEC-N19','SEC-N20']):
        scores['D2_security'] = min(scores['D2_security'], 3.0)
    # SQL: direct injection cap
    if any(negative_hits.get(k) for k in ['SQL-N01','SQL-N03','SQL-N17']):
        scores['D3_sql'] = min(scores['D3_sql'], 2.0)
    # Code Structure: missing permission_callback or filter without return cap
    if any(negative_hits.get(k) for k in ['STR-N01','STR-N09']):
        scores['D9_structure'] = min(scores['D9_structure'], 4.0)
    return scores
```

### Step 7: Compute Overall Score

```python
WEIGHTS = {
    'D1_wpcs':      0.10,
    'D2_security':  0.20,
    'D3_sql':       0.15,
    'D4_perf':      0.10,
    'D5_wp_api':    0.10,
    'D6_i18n':      0.10,
    'D7_a11y':      0.08,
    'D8_errors':    0.10,
    'D9_structure': 0.07,
}

def compute_overall(dimension_scores: dict[str, float | None]) -> float:
    applicable = {k: v for k, v in dimension_scores.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in applicable)
    weighted_sum = sum(
        applicable[k] * WEIGHTS[k] / total_weight
        for k in applicable
    )
    return round(weighted_sum * 10, 1)  # 0-100 scale
```

### Step 8: Produce Structured Output

```python
@dataclass
class RubricScore:
    file_path: str
    dimension_scores: dict[str, float]    # D1-D9, 0-10 each
    dimension_na: list[str]               # dimensions scored as N/A
    overall: float                         # 0-100
    triggered_checks: dict[str, list[str]] # dimension -> [check_ids triggered]
    tool_evidence: dict[str, list[str]]    # check_id -> [tool output lines]
    grade: str                             # Excellent/Good/Acceptable/Poor/Bad/Failing
    floor_rules_applied: list[str]         # which floor rules triggered
```

### Step 9: Reproducibility and Reviewer Calibration

To ensure two independent reviewers produce scores within ±5 points:

1. **Automated checks first:** Run all PHPCS, PHPStan, and regex checks before any LLM judgment. This produces a deterministic baseline.

2. **LLM prompts are binary:** Each LLM prompt asks for YES/NO + evidence. Do not ask for a score directly.

3. **Tie-breaking rule:** If LLM judge disagrees with regex-based detection, regex wins for checks that have regex coverage. LLM judgment is used only for checks with no regex equivalent.

4. **N/A determination:** Mark a dimension N/A only if both of these are true: (a) no functions from that domain appear in the code, AND (b) the code does not generate any output in that domain. If uncertain, score it — do not mark N/A.

5. **Single-function vs. full-file scope:** When scoring a single function rather than a full file:
   - Skip WPCS-P09 (filename conventions) — mark as N/A
   - Skip STR-P05/P06/P07 (lifecycle hooks) — mark as N/A unless the function IS a lifecycle hook
   - Score only the patterns directly observable in the provided code

6. **Scoring a function that calls another function:** If a function calls `save_data()` but that function is not provided, do not penalize for missing nonce/capability checks inside `save_data()`. Score only what is visible.

---

## Appendix: Quick Reference — Critical Checks by Severity

### Critical (−4 or −5 penalty, or triggers floor rule)

| Check | Dimension | Detection |
|---|---|---|
| `eval()` used | D1 WPCS | PHPCS `WordPress.PHP.RestrictedPHPFunctions` |
| Raw `$_GET`/`$_POST` echoed without escaping | D2 Security | PHPCS `WordPress.Security.EscapeOutput` |
| Form handler without nonce | D2 Security | PHPCS `WordPress.Security.NonceVerification` |
| Privileged action without `current_user_can()` | D2 Security | LLM analysis |
| `unserialize()` on user input | D2 Security | Regex |
| `include`/`require` with user input | D2 Security | Regex |
| `exec()` / `shell_exec()` with user input | D2 Security | PHPCS Security audit |
| Direct SQL interpolation without `prepare()` | D3 SQL | PHPCS `WordPress.DB.PreparedSQL` |
| REST endpoint without `permission_callback` | D9 Structure | Regex |
| Filter callback missing `return` | D9 Structure | PHPStan / LLM |

### High (−3 penalty)

| Check | Dimension |
|---|---|
| No global prefix on functions/classes/hooks | D1 WPCS |
| `query_posts()` used | D3 SQL / D5 WP API |
| N+1 queries in loop | D4 Performance |
| Missing `aria-label` on icon-only button | D7 A11y |
| `is_wp_error()` not checked before use | D8 Errors |
| Custom hook name not prefixed | D9 Structure |
| `echo` inside filter callback | D9 Structure |
| User data deleted in deactivation hook | D9 Structure |

---

*Sources:*
- [WordPress Coding Standards — PHP](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/php/)
- [WordPress Security APIs](https://developer.wordpress.org/apis/security/)
- [WordPress wpdb Reference](https://developer.wordpress.org/reference/classes/wpdb/)
- [WordPress VIP Database Query Docs](https://docs.wpvip.com/databases/optimize-queries/database-queries/)
- [WordPress Internationalization Guidelines](https://developer.wordpress.org/apis/internationalization/internationalization-guidelines/)
- [WordPress Accessibility Standards](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/)
- [WordPress REST API Handbook](https://developer.wordpress.org/rest-api/)
- [WordPress Plugin Handbook — Hooks](https://developer.wordpress.org/plugins/hooks/)
- [WordPress/WordPress-Coding-Standards GitHub](https://github.com/WordPress/WordPress-Coding-Standards)
- [Automattic/VIP-Coding-Standards GitHub](https://github.com/Automattic/VIP-Coding-Standards)
- [szepeviktor/phpstan-wordpress GitHub](https://github.com/szepeviktor/phpstan-wordpress)
- [FloeDesignTechnologies/phpcs-security-audit GitHub](https://github.com/FloeDesignTechnologies/phpcs-security-audit)
- [WCAG 2.2 — W3C](https://www.w3.org/TR/WCAG22/)
- [PHP-FIG PSR-12](https://www.php-fig.org/psr/psr-12/)
- [Make WordPress Core — PHPStan Proposal](https://make.wordpress.org/core/2025/07/11/proposal-phpstan-in-the-wordpress-core-development-workflow/)
- [Plugin Review Team Guidelines](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)
