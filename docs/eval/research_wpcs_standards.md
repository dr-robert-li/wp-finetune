# WordPress Coding Standards (WPCS) — Comprehensive Research

**Sources:**
- [WordPress/WordPress-Coding-Standards GitHub](https://github.com/WordPress/WordPress-Coding-Standards)
- [WordPress Developer Coding Standards Handbook](https://developer.wordpress.org/coding-standards/)
- [WordPress Plugin Review Guidelines](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)
- [PHPCompatibility/PHPCompatibilityWP GitHub](https://github.com/PHPCompatibility/PHPCompatibilityWP)
- [Automattic/VIP-Coding-Standards GitHub](https://github.com/Automattic/VIP-Coding-Standards)

---

## Table of Contents

1. [WPCS Overview & Rulesets](#1-wpcs-overview--rulesets)
2. [Full WPCS Sniff List by Category](#2-full-wpcs-sniff-list-by-category)
   - 2.1 WordPress.Arrays.*
   - 2.2 WordPress.CodeAnalysis.*
   - 2.3 WordPress.DB.*
   - 2.4 WordPress.DateTime.*
   - 2.5 WordPress.Files.*
   - 2.6 WordPress.NamingConventions.*
   - 2.7 WordPress.PHP.*
   - 2.8 WordPress.Security.*
   - 2.9 WordPress.Utils.*
   - 2.10 WordPress.WP.*
   - 2.11 WordPress.WhiteSpace.*
3. [WordPress Official Coding Standards (by language)](#3-wordpress-official-coding-standards-by-language)
   - 3.1 PHP
   - 3.2 HTML
   - 3.3 CSS
   - 3.4 JavaScript
   - 3.5 Accessibility
4. [WordPress Plugin Review Team Requirements](#4-wordpress-plugin-review-team-requirements)
5. [PHPCompatibilityWP](#5-phpcompatibilitywp)
6. [WordPress VIP Coding Standards](#6-wordpress-vip-coding-standards)
   - 6.1 WordPressVIPMinimum.Classes.*
   - 6.2 WordPressVIPMinimum.Constants.*
   - 6.3 WordPressVIPMinimum.Files.*
   - 6.4 WordPressVIPMinimum.Functions.*
   - 6.5 WordPressVIPMinimum.Hooks.*
   - 6.6 WordPressVIPMinimum.JS.*
   - 6.7 WordPressVIPMinimum.Performance.*
   - 6.8 WordPressVIPMinimum.Security.*
   - 6.9 WordPressVIPMinimum.UserExperience.*
   - 6.10 WordPressVIPMinimum.Variables.*
7. [Summary Table: All Sniffs](#7-summary-table-all-sniffs)

---

## 1. WPCS Overview & Rulesets

WordPress Coding Standards (WPCS) is a set of PHP_CodeSniffer rules that enforce WordPress coding conventions. It is maintained at [WordPress/WordPress-Coding-Standards](https://github.com/WordPress/WordPress-Coding-Standards) and installed via Composer.

### Bundled Rulesets

| Ruleset | Description |
|---------|-------------|
| `WordPress-Core` | Core rules that every WordPress project must follow. Corresponds to the official WordPress PHP Coding Standards. |
| `WordPress-Docs` | Documentation standards (inline docblock formatting for functions, classes, hooks). |
| `WordPress-Extra` | Extended best-practice rules not strictly covered in Core. Includes `WordPress-Core` plus additional sniffs. |
| `WordPress` | Alias for `WordPress-Extra` (the full recommended ruleset). |

### Companion Standards (not bundled, but officially recommended)

- **PHPCompatibilityWP** — PHP cross-version compatibility, with WordPress polyfill exclusions.
- **VariableAnalysis** — Undefined/unused variable checks.
- **VIP Coding Standards** — Stricter rules for WordPress.com VIP hosted projects.

---

## 2. Full WPCS Sniff List by Category

All sniffs use the `WordPress` namespace prefix: `WordPress.<Category>.<SniffName>`.

---

### 2.1 WordPress.Arrays.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.Arrays.ArrayDeclarationSpacing` | Enforces WordPress array spacing format: no space before/after parentheses in short single-item arrays; required space in multi-item arrays. |
| `WordPress.Arrays.ArrayIndentation` | Enforces WordPress array indentation for multi-line arrays: each element on its own line, indented one tab. |
| `WordPress.Arrays.ArrayKeySpacingRestrictions` | Checks for proper spacing in array key references: `$foo['bar']` (no spaces), `$foo[ $bar ]` (spaces when variable key). |
| `WordPress.Arrays.MultipleStatementAlignment` | Enforces alignment of the double arrow (`=>`) assignment operator for multi-item, multi-line arrays. |

---

### 2.2 WordPress.CodeAnalysis.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.CodeAnalysis.AssignmentInTernaryCondition` | Detects variable assignments inside ternary conditions — a common code smell where comparison was intended (e.g., `$a = $b` instead of `$a === $b`). |
| `WordPress.CodeAnalysis.EscapedNotTranslated` | Flags calls to escaping functions (e.g., `esc_html()`) that look like they were intended to also be translation calls (e.g., `esc_html( 'some text' )` without `__()` wrapping). |

---

### 2.3 WordPress.DB.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.DB.DirectDatabaseQuery` | Flags direct `$wpdb` database queries (e.g., `$wpdb->query()`, `$wpdb->get_results()`). Encourages using WP API functions instead; requires caching if direct queries are used. |
| `WordPress.DB.PreparedSQL` | Ensures variables are not directly interpolated into SQL statements passed to `$wpdb` methods. Variables must be parameterized through `$wpdb->prepare()`. |
| `WordPress.DB.PreparedSQLPlaceholders` | Validates correct use of `$wpdb->prepare()` placeholders: only `%d`, `%f`, `%F`, `%s`, `%i` are supported; `%%` for literal `%`; simple placeholders must be unquoted in query string; correct number of replacements must be provided. |
| `WordPress.DB.RestrictedClasses` | Prohibits direct use of database classes: `mysqli`, `PDO`, `PDOStatement`. Use `$wpdb` instead. |
| `WordPress.DB.RestrictedFunctions` | Prohibits raw PHP database functions: `mysql_*`, `mysqli_*`, `mysqlnd_*`, `maxdb_*`. Use `$wpdb` object instead. (`mysql_to_rfc3339()` is allowed.) |
| `WordPress.DB.SlowDBQuery` | Flags potentially slow WP_Query/get_posts parameters: `tax_query`, `meta_query`, `meta_key`, `meta_value`. These can cause full table scans on large datasets. |

---

### 2.4 WordPress.DateTime.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.DateTime.CurrentTimeTimestamp` | Disallows using `current_time('timestamp')` or `current_time('U')` to get a Unix timestamp — these return a "WordPress timezone-corrected" value, not a true Unix timestamp. Use `time()` or `current_datetime()->getTimestamp()` instead. |
| `WordPress.DateTime.RestrictedFunctions` | Forbids use of PHP/WP datetime functions that have better WP alternatives: e.g., `date()` (use `wp_date()`), `gmdate()` (use `wp_date()`), `mktime()`, `strtotime()` in timezone-sensitive contexts. |

---

### 2.5 WordPress.Files.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.Files.FileName` | Ensures filenames do not contain underscores (use hyphens) and that class files are prefixed with `class-` (e.g., `class-wp-error.php`). Template tag files in `wp-includes` should append `-template`. |

---

### 2.6 WordPress.NamingConventions.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.NamingConventions.PrefixAllGlobals` | Verifies that everything defined in the global namespace (functions, classes, variables, constants, hooks) is prefixed with a plugin/theme-specific prefix to avoid collisions with other plugins or WordPress itself. |
| `WordPress.NamingConventions.ValidFunctionName` | Enforces WordPress function name and method name format: lowercase letters and underscores for functions (`my_function_name`), no camelCase. |
| `WordPress.NamingConventions.ValidHookName` | Enforces action and filter hook names: lowercase letters with underscores. Words separated by underscores, no camelCase or hyphens. |
| `WordPress.NamingConventions.ValidPostTypeSlug` | Validates custom post type slugs: checks for invalid characters, length restrictions (max 20 chars), and reserved post type names. |
| `WordPress.NamingConventions.ValidVariableName` | Checks variable and member variable naming conventions: lowercase with underscores for regular variables. |

---

### 2.7 WordPress.PHP.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.PHP.DevelopmentFunctions` | Restricts use of development/debugging functions: `error_log()`, `var_dump()`, `var_export()`, `print_r()`, `trigger_error()`, etc. Should not be in production code. |
| `WordPress.PHP.DiscouragedPHPFunctions` | Discourages use of various PHP functions with safer alternatives: `serialize()` / `unserialize()` (use JSON), `base64_encode()` / `base64_decode()` in security contexts, `md5()` for passwords, etc. |
| `WordPress.PHP.DontExtract` | Prohibits use of `extract()`. It injects arbitrary variables into the current scope, making code hard to understand and potentially introducing security vulnerabilities. |
| `WordPress.PHP.IniSet` | Detects use of `ini_set()`. Safe ini directives get a notice; dangerous ones (e.g., `allow_url_fopen`) get an error; everything else gets a warning. |
| `WordPress.PHP.NoSilencedErrors` | Discourages use of the PHP error-silencing operator `@`. A limited allow-list exists for functions where no amount of error checking prevents PHP from throwing errors. |
| `WordPress.PHP.POSIXFunctions` | *(Deprecated in WPCS 3.3.0; use PHPCompatibility instead.)* Previously flagged POSIX regex functions (`ereg`, `eregi`, etc.) in favor of PCRE `preg_*` equivalents. |
| `WordPress.PHP.PregQuoteDelimiter` | Flags calls to `preg_quote()` without the second `$delimiter` parameter — omitting it produces incorrect output when the delimiter is present in the string. |
| `WordPress.PHP.RestrictedPHPFunctions` | Forbids use of dangerous PHP functions that have been fully prohibited: `create_function()` (deprecated/removed in PHP 8.0), `eval()`. |
| `WordPress.PHP.StrictInArray` | Flags calls to `in_array()`, `array_search()`, and `array_keys()` without `true` as the third (strict type comparison) parameter, preventing type-coercion bugs. |
| `WordPress.PHP.TypeCasts` | Verifies correct usage of type-cast keywords: normalized forms required (`(float)` not `(real)`, `(int)` not `(integer)`); use of `(unset)` and `(binary)` casts is discouraged. |
| `WordPress.PHP.YodaConditions` | Enforces Yoda conditional statements for `==`, `!=`, `===`, `!==` comparisons: the static value goes on the left (`if ( true === $the_force )`) to catch accidental assignments. |

---

### 2.8 WordPress.Security.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.Security.EscapeOutput` | **Core security sniff.** Verifies that all outputted strings are escaped with an appropriate escaping function before being echoed/printed. Recognized escaping functions include `esc_html()`, `esc_attr()`, `esc_url()`, `esc_js()`, `wp_kses()`, `wp_kses_post()`, `absint()`, etc. Supports custom escaping function lists via configuration. |
| `WordPress.Security.NonceVerification` | Checks that nonce verification (via `wp_verify_nonce()`, `check_admin_referer()`, or `check_ajax_referer()`) accompanies processing of superglobal input (`$_POST`, `$_GET`, `$_REQUEST`, `$_FILES`, `$_SERVER`). Prevents CSRF attacks. |
| `WordPress.Security.PluginMenuSlug` | Warns about using `__FILE__` as the menu slug when registering admin pages (`add_menu_page()`, `add_submenu_page()`, etc.), which can expose the file system path. |
| `WordPress.Security.SafeRedirect` | Warns when `wp_redirect()` is used and suggests `wp_safe_redirect()` instead, to prevent open redirect vulnerabilities. Also recommends using the `allowed_redirect_hosts` filter and calling `exit()` after redirect. |
| `WordPress.Security.ValidatedSanitizedInput` | **Core security sniff.** Flags any non-validated/non-sanitized input from superglobals (`$_GET`, `$_POST`, `$_REQUEST`, `$_COOKIE`, `$_SERVER`, `$_FILES`). Input must be both validated (checked for existence/type) and sanitized (via `sanitize_text_field()`, `intval()`, `wp_unslash()`, etc.) before use. |

---

### 2.9 WordPress.Utils.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.Utils.I18nTextDomainFixer` | **Fixer/utility sniff (not a standard quality check).** Automatically fixes/replaces the text domain string in internationalization function calls. Useful when migrating or renaming a plugin's text domain. Configured via `old_text_domain` and `new_text_domain` properties. |

---

### 2.10 WordPress.WP.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.WP.AlternativeFunctions` | Discourages PHP functions in favor of WordPress alternatives: `file_get_contents()` → `wp_remote_get()` / WP Filesystem API; `json_encode()` → `wp_json_encode()`; `json_decode()` → caution about use; `rand()` → `wp_rand()`; `strip_tags()` → `wp_strip_all_tags()`. Respects `minimum_wp_version` setting. |
| `WordPress.WP.Capabilities` | Checks that user capabilities are used correctly: user capability strings (e.g., `'edit_posts'`) should be used, not role names (e.g., `'editor'`) or deprecated capabilities. |
| `WordPress.WP.CapitalPDangit` | Verifies the correct spelling of "WordPress" (capital W, capital P) in text strings, comments, OO class names, and namespace names. |
| `WordPress.WP.ClassNameCase` | Verifies that references to WordPress native classes use the proper casing (e.g., `WP_Query` not `Wp_Query` or `wp_query`). |
| `WordPress.WP.CronInterval` | Flags custom WP cron schedules with an interval of less than 15 minutes (900 seconds). Configurable threshold. Prevents performance abuse on VIP and shared hosting. |
| `WordPress.WP.DeprecatedClasses` | Restricts use of deprecated WordPress classes and suggests alternatives. Throws an error for deprecated classes; severity depends on the WP version in which it was deprecated. |
| `WordPress.WP.DeprecatedFunctions` | Restricts use of deprecated WordPress functions (e.g., `get_currentuserinfo()`, `the_category_ID()`) and suggests replacements. Error severity scales with how long ago the function was deprecated relative to the configured `minimum_wp_version`. |
| `WordPress.WP.DeprecatedParameterValues` | Checks for usage of deprecated parameter values in WP functions and provides alternatives. |
| `WordPress.WP.DeprecatedParameters` | Checks for usage of deprecated parameters in WP function calls and suggests alternatives. |
| `WordPress.WP.DiscouragedConstants` | Warns about usage and re-declaration of discouraged WP constants (e.g., `STYLESHEETPATH` instead of `get_stylesheet_directory()`). |
| `WordPress.WP.DiscouragedFunctions` | Discourages use of various WordPress functions with better alternatives: `wp_reset_query()` (use `wp_reset_postdata()`), `query_posts()` (use `WP_Query`), `get_page_by_title()` (deprecated WP 6.2), etc. |
| `WordPress.WP.EnqueuedResourceParameters` | Checks that the 4th (`$ver`) parameter is set for all `wp_enqueue_script()`/`wp_enqueue_style()` calls when a `$src` is provided. Also checks the 5th (`$in_footer`) parameter for scripts. Ensures cache-busting version strings are present. |
| `WordPress.WP.EnqueuedResources` | Makes sure scripts and styles are enqueued via `wp_enqueue_script()`/`wp_enqueue_style()` rather than being directly echoed as `<script>` or `<link>` tags in HTML. |
| `WordPress.WP.GetMetaSingle` | Warns when calls to `get_post_meta()`, `get_user_meta()`, `get_term_meta()`, `get_comment_meta()`, `get_site_meta()`, `get_metadata()`, or `get_metadata_default()` use the `$key` parameter without explicitly passing the `$single` parameter. Omitting `$single` returns an array, which may not be the intended behavior. |
| `WordPress.WP.GlobalVariablesOverride` | Warns about overwriting WordPress native global variables (e.g., `$post`, `$wp_query`, `$wpdb`, `$current_user`). This can break WP core behavior. |
| `WordPress.WP.I18n` | Ensures WordPress internationalization (i18n) functions are used properly: text domain must match the plugin/theme slug; strings must not use variables (translators can't translate dynamic strings); correct function for context/plural; `translators:` comments for `printf`-style placeholders. |
| `WordPress.WP.PostsPerPage` | Flags returning high (configurable, default >100) or unlimited (`-1`) values for `posts_per_page` in query arguments. Large unbounded queries can overload the database. |

---

### 2.11 WordPress.WhiteSpace.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPress.WhiteSpace.CastStructureSpacing` | Ensures type cast statements are preceded by whitespace (e.g., `(int) $foo` not `(int)$foo`). |
| `WordPress.WhiteSpace.ControlStructureSpacing` | Checks that control structures (`if`, `else`, `for`, `foreach`, `while`, `switch`, `try`) have the correct spacing around brackets: space after keyword, space inside parentheses (e.g., `if ( $condition )`). |
| `WordPress.WhiteSpace.ObjectOperatorSpacing` | Ensures there is no whitespace before or after the object operator (`->`): `$foo->bar` not `$foo -> bar`. |
| `WordPress.WhiteSpace.OperatorSpacing` | Verifies operator spacing: spaces required on both sides of arithmetic, assignment, comparison, and logical operators. Extends the Squiz sniff to also cover the spread operator and named argument operator. |

---

## 3. WordPress Official Coding Standards (by language)

### 3.1 PHP Coding Standards

Source: [developer.wordpress.org/coding-standards/wordpress-coding-standards/php/](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/php/)

#### Opening/Closing Tags
- Always use `<?php ?>` — never short tags `<? ?>` or `<?= ?>`.
- Multi-line PHP snippets within HTML: opening/closing tags on their own lines.

#### Quotes
- Use single quotes unless the string requires evaluation.
- Alternate quote styles to avoid escaping.

#### Include/Require
- No parentheses around path; prefer `require_once` over `include_once`.
- Use `ABSPATH` prefix: `require_once ABSPATH . 'filename.php';` (not `__DIR__`).

#### Naming Conventions
- **Functions, variables, actions/filters**: `lowercase_with_underscores`.
- **Classes, interfaces, traits, enums**: `Capitalized_Words_With_Underscores` (acronyms all-caps, e.g., `WP_HTTP`).
- **Constants**: `ALL_UPPERCASE_WITH_UNDERSCORES`.
- **Files**: `lowercase-with-hyphens.php`.
- **Class files**: prefixed `class-` with underscores replaced by hyphens (`class-wp-error.php`).
- No camelCase for functions/variables; no unnecessary abbreviations.

#### Whitespace
- Real tabs (not spaces) for indentation.
- Spaces after commas; spaces on both sides of operators.
- Spaces inside control structure parentheses: `foreach ( $foo as $bar )`.
- Spaces inside function call parentheses: `my_function( $param )`.
- No trailing whitespace; omit closing PHP tag at end of file.

#### Brace Style
- Braces for all blocks, even single-line: `if () { }`.
- `elseif` (not `else if`).
- Alternative syntax (`:` / `endif;`) acceptable in templates.

#### Arrays
- Use long syntax: `array( 1, 2, 3 )` — not `[ 1, 2, 3 ]` in WordPress Core.
- Multi-item arrays: each element on its own line with trailing comma.

#### Yoda Conditions
- Static value on left: `if ( true === $the_force )`.
- Applies to `==`, `!=`, `===`, `!==`.

#### Ternary Operator
- Test for true: `( 'jazz' === $music ) ? 'cool' : 'blah'`.
- No short ternary `?:`.

#### Database
- Use `$wpdb->prepare()` with `%d`, `%f`, `%s`, `%i` placeholders.
- No pre-quoting of values passed to `prepare()`.
- Prefer WP API functions over raw `$wpdb` queries where available.

#### Object-Oriented
- Explicit `public`/`protected`/`private` visibility (no `var`).
- One class/interface/trait/enum per file.
- Visibility and modifier order: `abstract/final` → visibility → `static`.
- Always use parentheses for instantiation: `new Foo()`.

#### Error Handling
- Never use `@` error suppression operator.
- Never use `eval()`, `create_function()`, `extract()`, `goto`.
- Use `preg_*` functions (not POSIX `ereg_*`).

#### Strict Comparisons
- Use `===` rather than `==` where possible.
- Always pass strict `true` to `in_array()`.

### 3.2 HTML Coding Standards

Source: [developer.wordpress.org/coding-standards/wordpress-coding-standards/html/](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/html/)

- **Validation**: All HTML should validate against W3C validator.
- **Self-closing tags**: One space before the slash: `<br />` (not `<br/>`).
- **Case**: All tags and attributes must be lowercase.
- **Attribute values**: Lowercase when machine-interpreted; human-readable values may use proper casing.
- **Quotes**: All attributes must be quoted with double quotes (or single): `type="text"`.
- **Boolean attributes**: Omit value (`disabled`) or use `disabled="disabled"` — never `disabled="true"`.
- **Indentation**: Tabs, not spaces. PHP blocks within HTML indented to match surrounding HTML context.

### 3.3 CSS Coding Standards

Source: [developer.wordpress.org/coding-standards/wordpress-coding-standards/css/](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/css/)

#### Structure
- Tabs for indentation.
- Two blank lines between sections; one blank line between blocks within a section.
- Each selector on its own line.
- Property-value pairs on own lines, ending with semicolons.
- Closing brace flush-left matching the opening selector's indentation.

#### Selectors
- Lowercase, words separated by hyphens.
- Avoid over-qualification (`div.container` → `.container`).
- Attribute selectors use double quotes.

#### Properties
- Colon followed by a space: `color: #fff;`.
- All properties/values lowercase (except font names).
- Colors: hex (`#fff`) or `rgba()` for opacity; shorten values when possible.
- Use shorthand for `background`, `border`, `font`, `list-style`, `margin`, `padding`.

#### Property Ordering
- Group by: Display → Positioning → Box model → Colors/Typography → Other.
- Top/Right/Bottom/Left (TRBL) order for directional properties.

#### Values
- `0` values without units (unless required).
- Unit-less `line-height`.
- Leading zero for decimals: `0.5` not `.5`.
- Font weights as numerics: `400` not `normal`, `700` not `bold`.

#### Vendor Prefixes
- Use Autoprefixer; longest prefix first, unprefixed last.

#### Comments
- Comment liberally; use minified files for production.
- Long comments: manual 80-character line break.
- PHPDoc-style for section headers.

### 3.4 JavaScript Coding Standards

Source: [developer.wordpress.org/coding-standards/wordpress-coding-standards/javascript/](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/javascript/)

- **Indentation**: Tabs, not spaces.
- **Spacing**: Spaces inside `if`, `for`, `while`, `function` parentheses; spaces around operators and after commas; spaces inside array/function call brackets.
- **Braces**: Required for all `if`/`else`/`for`/`while`/`try` blocks; multi-line always.
- **Semicolons**: Always — never rely on ASI.
- **Strings**: Single quotes for string literals.
- **Equality**: Always `===`/`!==`, never `==`/`!=`.
- **Naming**: camelCase for variables/functions; UpperCamelCase for classes; SCREAMING_SNAKE_CASE for constants; acronyms in all-caps (`DOM`, `ID`).
- **Variables**: `const` unless reassigned (`let`); single comma-delimited `var` at function start for legacy code.
- **Globals**: Document at file top with `/* global variableName */` comments.
- **jQuery**: Access via IIFE: `( function( $ ) {} )( jQuery );` — never assume `$` is jQuery.
- **Type Checks**: Use `typeof` checks or Underscore.js helpers; never bare `== null`.
- **Arrays**: Literal `[]` syntax; never `new Array()`.
- **Objects**: Literal `{}` syntax preferred.
- **Linting**: JSHint via `npm run grunt jshint`.
- **Comments**: Before the code they document; capitalize first letter.

### 3.5 Accessibility Standards

Source: [developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/](https://developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/)

All WordPress ecosystem code must conform to **WCAG 2.2 Level A and Level AA**. Level AAA encouraged where applicable. Also references ATAG 2.0 for authoring tool accessibility.

**Core POUR Principles:**
- **Perceivable**: Text alternatives for non-text; captions for media; adaptable presentation; sufficient contrast.
- **Operable**: Full keyboard accessibility; sufficient time; no seizure-inducing content; navigable.
- **Understandable**: Readable text; predictable page behavior; input assistance.
- **Robust**: Compatible with user agents and assistive technologies.

---

## 4. WordPress Plugin Review Team Requirements

Source: [developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)

The Plugin Review Team enforces the following code quality and compliance requirements:

| Requirement | Detail |
|-------------|--------|
| **GPL Compatibility** | All code, data, and images must be GPL-compatible. Strongly recommend "GPLv2 or later". All third-party libraries must be GPL-compatible. |
| **Human-Readable Code** | Code must be mostly human-readable. Prohibits obfuscation (packers, uglify mangle, unreadable naming like `$z12sdf813d`). Source code and build tools must be publicly accessible. |
| **Security** | All code must be as secure as possible. Ultimate responsibility lies with the developer. Plugins with security issues are closed until resolved; may be patched by the WordPress Security Team in extreme cases. |
| **No Executable Code from Third-Party CDNs** | Cannot call external CDNs for non-font JavaScript/CSS — these must be included locally. No serving updates/installs from non-WordPress.org servers. |
| **Use WordPress Default Libraries** | Must use WordPress-bundled versions of jQuery, PHPMailer, SimplePie, etc. Cannot include them separately. |
| **No Trialware** | No restricted/locked functionality, no disabling features after a trial/quota, no sandbox-only API access. |
| **No Illegal/Dishonest Code** | No code that circumvents guidelines, mines cryptocurrency, creates botnets, or manipulates search rankings artificially. |
| **No PHP Sessions** | PHP session functions are discouraged; prefer WP transients or cookies. |
| **Version Numbering** | Increment version numbers for each release. Trunk `readme.txt` must always reflect current version. |
| **Stable Version Required** | A complete, functional plugin is required at submission. Names cannot be reserved. |
| **SVN Commits** | SVN is a release repository only — commit only deployment-ready code. Avoid rapid minor tweaks. |

**Implicit Code Quality Checks (enforced via review):**
- Proper nonce verification on form submissions.
- Proper escaping of all output.
- Proper sanitization of all input.
- Avoid `query_posts()` in favor of `WP_Query`.
- Use `wp_enqueue_script()`/`wp_enqueue_style()` for assets.
- Prefix all global functions, classes, variables, and hooks.
- No `eval()`, no `create_function()`, no `extract()`.

---

## 5. PHPCompatibilityWP

Source: [PHPCompatibility/PHPCompatibilityWP GitHub](https://github.com/PHPCompatibility/PHPCompatibilityWP)

PHPCompatibilityWP is a PHP_CodeSniffer ruleset that wraps the full [PHPCompatibility](https://github.com/PHPCompatibility/PHPCompatibility) standard and **excludes** rules for functions, constants, and interfaces that WordPress itself polyfills — preventing false positives in WordPress projects.

### How It Works

- Extends `PHPCompatibility` (the base ruleset) and also includes `PHPCompatibilityParagonieSodiumCompat` (for WordPress's bundled Sodium/Random polyfills since WP 5.2/4.4).
- You specify the PHP version range to check against: `--runtime-set testVersion 7.2-` (for WP 6.6+ minimum).
- Reports **deprecated/removed** PHP features by default; add `testVersion` for **new feature** detection too.

### PHPCompatibility Base Categories

The base PHPCompatibility standard checks across these categories (applied to all WordPress projects):

| Category | What It Checks |
|----------|---------------|
| `Attributes` | PHP 8.0+ attribute syntax compatibility |
| `Classes` | New, changed, and removed class usage across PHP versions |
| `Constants` | New, deprecated, and removed constants |
| `ControlStructures` | Match expressions, other control structure syntax changes |
| `Extensions` | Removed/deprecated PHP extensions (e.g., `mysql_*` extension) |
| `FunctionDeclarations` | Function signature changes, typed properties, etc. |
| `FunctionNameRestrictions` | Reserved function names across PHP versions |
| `FunctionUse` | New, deprecated, and removed functions |
| `Generators` | Generator syntax and feature compatibility |
| `IniDirectives` | Deprecated/removed `php.ini` directives |
| `InitialValue` | Constant expression changes |
| `Interfaces` | New and removed interfaces |
| `Keywords` | Reserved keywords newly introduced in PHP versions |
| `LanguageConstructs` | Changes to `echo`, `list()`, `match`, etc. |
| `Lists` | Short list `[]` syntax, list() in foreach |
| `MethodUse` | Changes to method behavior across versions |
| `Miscellaneous` | Miscellaneous compatibility checks |
| `Namespaces` | Namespace-related syntax changes |
| `Numbers` | Numeric literal syntax (underscores, hex floats, etc.) |
| `Operators` | New operators (nullsafe `?->`, named arguments `:`, spread, etc.) |
| `ParameterValues` | Function parameter default value changes |
| `Syntax` | PHP syntax changes (arrow functions, match, enums, etc.) |
| `TextStrings` | String-related syntax changes |
| `TypeCasts` | Removed/changed type cast behavior |
| `UseDeclarations` | Changes to `use` statement syntax |
| `Variables` | Variable handling changes (`$$var`, `${}` interpolation removal, etc.) |

### WordPress-Specific Polyfill Exclusions

PHPCompatibilityWP **excludes** false positives for the following WordPress-backfilled functions/constants (a curated list from the ruleset XML):

**Functions backfilled by WordPress (`/wp-includes/compat.php`):**
- `hash_hmac()` (since WP 3.2.0; removed in WP 6.8.0)
- `json_encode()`, `json_decode()` (removed since WP 5.3.0)
- `hash_equals()` (since WP 3.9.2; removed in WP 6.8.0)
- `json_last_error_msg()` (since WP 4.4.0; removed since WP 5.3.0)
- `array_replace_recursive()` (since WP 4.5.3, removed in WP 5.3)
- `is_iterable()` (since WP 4.9.6; removed in WP 6.6.0)
- `is_countable()` (since WP 4.9.6)
- `array_key_first()`, `array_key_last()` (since WP 5.9.0)
- `str_contains()`, `str_starts_with()`, `str_ends_with()` (since WP 5.9.0)
- `array_is_list()` (since WP 6.5.0)
- `array_find()`, `array_find_key()`, `array_any()`, `array_all()` (since WP 6.8.0)
- `array_first()`, `array_last()` (since WP 6.9.0)
- `spl_autoload_register()`, `spl_autoload_unregister()`, `spl_autoload_functions()` (WP 4.6.0–5.2.x)

**Constants backfilled by WordPress:**
- `JSON_PRETTY_PRINT` (since WP 4.1.0; removed since WP 5.3.0)
- `IMAGETYPE_WEBP`, `IMG_WEBP` (since WP 5.8.0; removed WP 6.6.0)
- `IMAGETYPE_AVIF`, `IMG_AVIF` (since WP 6.5.0)
- `IMAGETYPE_HEIF` (since WP 6.9.0)

**Interfaces backfilled:**
- `JsonSerializable` (since WP 4.4.0; removed since WP 5.3.0)

**Sodium/Random Compat (since WP 5.2.0/4.4.0):**
- All `sodium_*`, `random_bytes()`, `random_int()` functions via `PHPCompatibilityParagonieSodiumCompat`.

**Recommended `testVersion` by WordPress minimum:**
| WordPress Version | Minimum PHP | Recommended testVersion |
|---|---|---|
| WP 6.6+ | PHP 7.2.24 | `7.2-` |
| WP 6.3–6.5 | PHP 7.0.0 | `7.0-` |
| WP 5.2–6.2 | PHP 5.6.20 | `5.6-` |

---

## 6. WordPress VIP Coding Standards

Source: [Automattic/VIP-Coding-Standards GitHub](https://github.com/Automattic/VIP-Coding-Standards)

VIP Coding Standards provides **stricter rules** on top of WPCS for projects hosted on WordPress.com VIP and the VIP Go platform. It wraps WPCS and VariableAnalysis with additional sniffs.

### Rulesets
- **`WordPressVIPMinimum`** — Legacy WordPress.com VIP (Classic) platform.
- **`WordPress-VIP-Go`** — Current VIP Go platform (more permissive, preferred for new projects).

All VIP sniffs use the `WordPressVIPMinimum` namespace prefix.

---

### 6.1 WordPressVIPMinimum.Classes.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Classes.DeclarationCompatibility` | Ensures method declarations in subclasses are compatible with parent class method signatures. |
| `WordPressVIPMinimum.Classes.RestrictedExtendClasses` | Restricts extending certain base classes that are inappropriate for VIP plugin use. |

---

### 6.2 WordPressVIPMinimum.Constants.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Constants.ConstantString` | Ensures constant names are passed as strings (not bare constants) when checking whether a constant is defined (`defined( 'CONSTANT_NAME' )`). |
| `WordPressVIPMinimum.Constants.RestrictedConstants` | Restricts usage and re-declaration of certain VIP-platform constants: `A8C_PROXIED_REQUEST` (restricted usage); `JETPACK_DEV_DEBUG`, `WP_CRON_CONTROL_SECRET` (restricted declaration). |

---

### 6.3 WordPressVIPMinimum.Files.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Files.IncludingFile` | Checks for custom variables, functions, and constants — as well as external URLs — used in file inclusion (`include`/`require`). Flags dynamic file includes that could lead to path traversal or code injection. |
| `WordPressVIPMinimum.Files.IncludingNonPHPFile` | Ensures non-PHP files (e.g., `.html`, `.json`, `.svg`) are included via `file_get_contents()` rather than `include`/`require`, preventing embedded PHP code from being automatically executed. |

---

### 6.4 WordPressVIPMinimum.Functions.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Functions.CheckReturnValue` | Enforces checking the return value of a function before passing it to another function. Prevents passing `null`/`false` (e.g., from `get_post()`) directly into further function calls. |
| `WordPressVIPMinimum.Functions.DynamicCalls` | Prohibits dynamically calling certain functions (e.g., `$func()`, `call_user_func( $fn )`). Specific functions should not be called dynamically. |
| `WordPressVIPMinimum.Functions.RestrictedFunctions` | Restricts numerous functions for VIP-specific reasons: |
| | - **Memory**: certain functions prohibited due to memory corruption risk |
| | - **Session functions**: `session_start()`, `session_destroy()`, etc. are prohibited (use WP transients/cookies) |
| | - **Filesystem writes**: `fwrite()`, `file_put_contents()`, `fputs()` etc. are forbidden |
| | - **Internal functions**: `wpcom_vip_*` internal-only functions |
| | - **Non-cached alternatives**: `attachment_url_to_postid()` (use `wpcom_vip_attachment_url_to_postid()`), `url_to_postid()` (use `wpcom_vip_url_to_postid()`), `get_adjacent_post()` (use `wpcom_vip_get_adjacent_post()`), `wp_old_slug_redirect()` (use `wpcom_vip_old_slug_redirect()`) |
| | - **Mobile detection**: `wp_is_mobile()` → use `jetpack_is_mobile()` for better cache compat |
| | - **Role management**: `add_role()` → use `wpcom_vip_add_role()` |
| | - **Uncached queries**: `count_user_posts()` → use `wpcom_vip_count_user_posts()` |
| | - **Intermediate images**: `get_intermediate_image_sizes()` returns empty on VIP |
| | - **Site switching**: `switch_to_blog()` warning (doesn't load plugins/theme of target blog) |
| | - **Stats**: `stats_get_csv()` outside Jetpack context pollutes options table |
| `WordPressVIPMinimum.Functions.StripTags` | Ensures proper tag stripping: `strip_tags()` with an allowlist is flagged as potentially insufficient; `wp_strip_all_tags()` or `wp_kses()` recommended. |

---

### 6.5 WordPressVIPMinimum.Hooks.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Hooks.AlwaysReturnInFilter` | Validates that filter callbacks always return a value. A filter that does not return exits the filter chain and returns `null`, breaking the filtered value. |
| `WordPressVIPMinimum.Hooks.PreGetPosts` | Validates proper usage of `pre_get_posts` action callbacks: the `WP_Query` object must not be modified without first checking `WP_Query::is_main_query()`, to avoid corrupting all queries. |
| `WordPressVIPMinimum.Hooks.RestrictedHooks` | Restricts use of certain hooks that require special care: `upload_mimes` (warn: check for insecure MIME types like SVG, SWF), `http_request_timeout` / `http_request_args` (warn: timeout should not exceed 3s), `do_robotstxt` / `robots_txt` (warn: remember to flush robots.txt cache). |

---

### 6.6 WordPressVIPMinimum.JS.*

(These sniffs check JavaScript files for client-side security issues.)

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.JS.DangerouslySetInnerHTML` | Flags React's `dangerouslySetInnerHTML` attribute — direct use sets unescaped HTML and is an XSS vector. |
| `WordPressVIPMinimum.JS.HTMLExecutingFunctions` | Flags JavaScript functions that execute HTML passed as a string: `html()`, `$()`, `document.write()`, `document.writeln()`, `$.globalEval()`, `eval()`, `after()`, `append()`, `before()`, etc. These can execute injected scripts. |
| `WordPressVIPMinimum.JS.InnerHTML` | Flags direct use of `.innerHTML` property assignment — an XSS vector when the value is not sanitized. |
| `WordPressVIPMinimum.JS.StringConcat` | Flags HTML string concatenation used to build HTML structures. Concatenated HTML often skips proper escaping. |
| `WordPressVIPMinimum.JS.StrippingTags` | Flags incorrect ways of stripping HTML tags in JavaScript (e.g., regex-based stripping) rather than using DOMParser or a sanitization library. |
| `WordPressVIPMinimum.JS.Window` | Flags dangerous `window` properties that should be reviewed: `window.location` assignments (open redirect), `window.name` (XSS vector), etc. |

---

### 6.7 WordPressVIPMinimum.Performance.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Performance.CacheValueOverride` | Detects cases where a cached value is retrieved and then immediately overridden before being used — making the cache retrieval pointless. |
| `WordPressVIPMinimum.Performance.FetchingRemoteData` | Restricts use of `file_get_contents()` for remote URLs — use `wp_remote_get()` for better error handling, timeouts, and VIP platform compatibility. |
| `WordPressVIPMinimum.Performance.LowExpiryCacheTime` | Flags when cache expiration time is set very low (near-zero TTL), which defeats the purpose of caching and hammers the backend. |
| `WordPressVIPMinimum.Performance.NoPaging` | Flags disabling pagination via `'nopaging' => true` or `'posts_per_page' => -1` / `'numberposts' => -1` in query arguments — unlimited queries can return thousands of rows. |
| `WordPressVIPMinimum.Performance.OrderByRand` | Flags `'orderby' => 'rand'` in WP_Query / get_posts — `ORDER BY RAND()` is extremely slow on large tables (full table scan and sort). |
| `WordPressVIPMinimum.Performance.RegexpCompare` | Flags use of `REGEXP` and `NOT REGEXP` in `meta_compare` or `compare` — regex comparisons in MySQL are not indexable and cause full table scans. |
| `WordPressVIPMinimum.Performance.RemoteRequestTimeout` | Flags remote request timeout values greater than 3 seconds — high timeouts block page rendering for the entire duration. |
| `WordPressVIPMinimum.Performance.TaxonomyMetaInOptions` | Restricts implementing taxonomy term meta via the `wp_options` table — use the proper term meta API (`add_term_meta()`/`get_term_meta()`) instead. |
| `WordPressVIPMinimum.Performance.WPQueryParams` | Flags suspicious or performance-impacting `WP_Query`/`get_posts` parameters that could cause unexpectedly large or slow queries. |

---

### 6.8 WordPressVIPMinimum.Security.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Security.EscapingVoidReturnFunctions` | Flags functions that don't return anything (void) being wrapped in escaping function calls — e.g., `esc_html( _e( 'foo' ) )`. This produces no output (since `_e()` returns void) and is a logic error. |
| `WordPressVIPMinimum.Security.ExitAfterRedirect` | Requires that `exit;` (or `die;`) is called immediately after `wp_redirect()` and `wp_safe_redirect()`. Without it, code execution continues after the redirect header is sent. |
| `WordPressVIPMinimum.Security.Mustache` | Detects unescaped output in Mustache templating (`{{{ value }}}` — triple braces bypass escaping) and Handlebars.js templates. |
| `WordPressVIPMinimum.Security.PHPFilterFunctions` | Ensures proper sanitization when using PHP's `filter_var()`, `filter_input()`, and related functions. Flags inadequate filter types for security-sensitive contexts. |
| `WordPressVIPMinimum.Security.ProperEscapingFunction` | Checks that the appropriate escaping function is used for the given context: `esc_url()` for URLs, `esc_attr()` for HTML attributes, `esc_html()` for text content, `esc_js()` for JavaScript contexts. Flags mismatched escaping (e.g., `esc_html()` on a URL). |
| `WordPressVIPMinimum.Security.StaticStrreplace` | Restricts use of `str_replace()` when all three parameters are static string literals — this is usually dead code or an accidental no-op. |
| `WordPressVIPMinimum.Security.Underscorejs` | Detects unescaped output in Underscore.js templates (`<%= value %>` and `<%- value %>` vs `<%-` for escaped). Flags triple-interpolation constructs. |
| `WordPressVIPMinimum.Security.Vuejs` | Detects unescaped output bindings in Vue.js templates: `v-html` directive (injects raw HTML, XSS risk) and `{{{ value }}}` triple-mustache syntax. |

---

### 6.9 WordPressVIPMinimum.UserExperience.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.UserExperience.AdminBarRemoval` | Discourages removal of the WordPress admin bar (`show_admin_bar( false )`, filtering `show_admin_bar`, or direct CSS hiding). The admin bar is a critical UX element for logged-in users. |

---

### 6.10 WordPressVIPMinimum.Variables.*

| Sniff | What It Checks |
|-------|---------------|
| `WordPressVIPMinimum.Variables.RestrictedVariables` | Restricts usage of certain variables in VIP context: e.g., `$wpdb->use_mysqli` (direct DB driver flag), `$_SERVER['PHP_SELF']` variations, specific VIP-platform variables. |
| `WordPressVIPMinimum.Variables.ServerVariables` | Restricts usage of potentially unsafe `$_SERVER` variables: `$_SERVER['PHP_SELF']`, `$_SERVER['HTTP_REFERER']`, `$_SERVER['QUERY_STRING']`, etc. — these are user-controllable and must be sanitized. |

---

## 7. Summary Table: All Sniffs

### WordPress.* Sniffs (WPCS Core — 43 sniffs)

| Full Sniff Name | Severity | Category |
|----------------|----------|----------|
| `WordPress.Arrays.ArrayDeclarationSpacing` | Error/Warning | Formatting |
| `WordPress.Arrays.ArrayIndentation` | Warning | Formatting |
| `WordPress.Arrays.ArrayKeySpacingRestrictions` | Error | Formatting |
| `WordPress.Arrays.MultipleStatementAlignment` | Warning | Formatting |
| `WordPress.CodeAnalysis.AssignmentInTernaryCondition` | Warning | Code Quality |
| `WordPress.CodeAnalysis.EscapedNotTranslated` | Warning | i18n |
| `WordPress.DB.DirectDatabaseQuery` | Warning | Database |
| `WordPress.DB.PreparedSQL` | Error | Security/Database |
| `WordPress.DB.PreparedSQLPlaceholders` | Error | Security/Database |
| `WordPress.DB.RestrictedClasses` | Error | Database |
| `WordPress.DB.RestrictedFunctions` | Error | Database |
| `WordPress.DB.SlowDBQuery` | Warning | Performance |
| `WordPress.DateTime.CurrentTimeTimestamp` | Error | DateTime |
| `WordPress.DateTime.RestrictedFunctions` | Error | DateTime |
| `WordPress.Files.FileName` | Error | Formatting |
| `WordPress.NamingConventions.PrefixAllGlobals` | Error | Naming |
| `WordPress.NamingConventions.ValidFunctionName` | Error | Naming |
| `WordPress.NamingConventions.ValidHookName` | Warning | Naming |
| `WordPress.NamingConventions.ValidPostTypeSlug` | Error | Naming |
| `WordPress.NamingConventions.ValidVariableName` | Error | Naming |
| `WordPress.PHP.DevelopmentFunctions` | Warning | PHP Best Practices |
| `WordPress.PHP.DiscouragedPHPFunctions` | Warning | PHP Best Practices |
| `WordPress.PHP.DontExtract` | Error | PHP Best Practices |
| `WordPress.PHP.IniSet` | Error/Warning | PHP Best Practices |
| `WordPress.PHP.NoSilencedErrors` | Warning | PHP Best Practices |
| `WordPress.PHP.POSIXFunctions` | *(deprecated 3.3.0)* | PHP Best Practices |
| `WordPress.PHP.PregQuoteDelimiter` | Error | PHP Best Practices |
| `WordPress.PHP.RestrictedPHPFunctions` | Error | Security |
| `WordPress.PHP.StrictInArray` | Error | PHP Best Practices |
| `WordPress.PHP.TypeCasts` | Error | Formatting |
| `WordPress.PHP.YodaConditions` | Error | Code Style |
| `WordPress.Security.EscapeOutput` | Error | Security |
| `WordPress.Security.NonceVerification` | Error | Security |
| `WordPress.Security.PluginMenuSlug` | Warning | Security |
| `WordPress.Security.SafeRedirect` | Warning | Security |
| `WordPress.Security.ValidatedSanitizedInput` | Error | Security |
| `WordPress.Utils.I18nTextDomainFixer` | (fixer only) | Utility |
| `WordPress.WP.AlternativeFunctions` | Warning | WP Best Practices |
| `WordPress.WP.Capabilities` | Warning | Security |
| `WordPress.WP.CapitalPDangit` | Warning | Code Style |
| `WordPress.WP.ClassNameCase` | Error | Naming |
| `WordPress.WP.CronInterval` | Warning | Performance |
| `WordPress.WP.DeprecatedClasses` | Error | Deprecation |
| `WordPress.WP.DeprecatedFunctions` | Error | Deprecation |
| `WordPress.WP.DeprecatedParameterValues` | Warning | Deprecation |
| `WordPress.WP.DeprecatedParameters` | Warning | Deprecation |
| `WordPress.WP.DiscouragedConstants` | Warning | WP Best Practices |
| `WordPress.WP.DiscouragedFunctions` | Warning | WP Best Practices |
| `WordPress.WP.EnqueuedResourceParameters` | Error | WP Best Practices |
| `WordPress.WP.EnqueuedResources` | Error | WP Best Practices |
| `WordPress.WP.GetMetaSingle` | Warning | WP Best Practices |
| `WordPress.WP.GlobalVariablesOverride` | Error | WP Best Practices |
| `WordPress.WP.I18n` | Error/Warning | i18n |
| `WordPress.WP.PostsPerPage` | Warning | Performance |
| `WordPress.WhiteSpace.CastStructureSpacing` | Error | Formatting |
| `WordPress.WhiteSpace.ControlStructureSpacing` | Error | Formatting |
| `WordPress.WhiteSpace.ObjectOperatorSpacing` | Error | Formatting |
| `WordPress.WhiteSpace.OperatorSpacing` | Error | Formatting |

### WordPressVIPMinimum.* Sniffs (VIP — 33 sniffs)

| Full Sniff Name | Severity | Category |
|----------------|----------|----------|
| `WordPressVIPMinimum.Classes.DeclarationCompatibility` | Error | OOP |
| `WordPressVIPMinimum.Classes.RestrictedExtendClasses` | Error | OOP |
| `WordPressVIPMinimum.Constants.ConstantString` | Warning | Code Quality |
| `WordPressVIPMinimum.Constants.RestrictedConstants` | Error | Platform |
| `WordPressVIPMinimum.Files.IncludingFile` | Warning | Security |
| `WordPressVIPMinimum.Files.IncludingNonPHPFile` | Warning | Security |
| `WordPressVIPMinimum.Functions.CheckReturnValue` | Warning | Code Quality |
| `WordPressVIPMinimum.Functions.DynamicCalls` | Error | Security |
| `WordPressVIPMinimum.Functions.RestrictedFunctions` | Error | Platform/Security |
| `WordPressVIPMinimum.Functions.StripTags` | Warning | Security |
| `WordPressVIPMinimum.Hooks.AlwaysReturnInFilter` | Error | Code Quality |
| `WordPressVIPMinimum.Hooks.PreGetPosts` | Warning | Code Quality |
| `WordPressVIPMinimum.Hooks.RestrictedHooks` | Warning | Platform/Security |
| `WordPressVIPMinimum.JS.DangerouslySetInnerHTML` | Error | Security (JS) |
| `WordPressVIPMinimum.JS.HTMLExecutingFunctions` | Error | Security (JS) |
| `WordPressVIPMinimum.JS.InnerHTML` | Warning | Security (JS) |
| `WordPressVIPMinimum.JS.StringConcat` | Warning | Security (JS) |
| `WordPressVIPMinimum.JS.StrippingTags` | Warning | Security (JS) |
| `WordPressVIPMinimum.JS.Window` | Warning | Security (JS) |
| `WordPressVIPMinimum.Performance.CacheValueOverride` | Warning | Performance |
| `WordPressVIPMinimum.Performance.FetchingRemoteData` | Warning | Performance |
| `WordPressVIPMinimum.Performance.LowExpiryCacheTime` | Warning | Performance |
| `WordPressVIPMinimum.Performance.NoPaging` | Error | Performance |
| `WordPressVIPMinimum.Performance.OrderByRand` | Warning | Performance |
| `WordPressVIPMinimum.Performance.RegexpCompare` | Warning | Performance |
| `WordPressVIPMinimum.Performance.RemoteRequestTimeout` | Warning | Performance |
| `WordPressVIPMinimum.Performance.TaxonomyMetaInOptions` | Warning | Performance |
| `WordPressVIPMinimum.Performance.WPQueryParams` | Warning | Performance |
| `WordPressVIPMinimum.Security.EscapingVoidReturnFunctions` | Warning | Security |
| `WordPressVIPMinimum.Security.ExitAfterRedirect` | Error | Security |
| `WordPressVIPMinimum.Security.Mustache` | Warning | Security |
| `WordPressVIPMinimum.Security.PHPFilterFunctions` | Warning | Security |
| `WordPressVIPMinimum.Security.ProperEscapingFunction` | Error | Security |
| `WordPressVIPMinimum.Security.StaticStrreplace` | Warning | Code Quality |
| `WordPressVIPMinimum.Security.Underscorejs` | Warning | Security |
| `WordPressVIPMinimum.Security.Vuejs` | Warning | Security |
| `WordPressVIPMinimum.UserExperience.AdminBarRemoval` | Warning | UX |
| `WordPressVIPMinimum.Variables.RestrictedVariables` | Error | Platform |
| `WordPressVIPMinimum.Variables.ServerVariables` | Warning | Security |

---

*Research compiled: March 2026*
*Primary sources: [WPCS GitHub](https://github.com/WordPress/WordPress-Coding-Standards) | [VIP CS GitHub](https://github.com/Automattic/VIP-Coding-Standards) | [PHPCompatibilityWP GitHub](https://github.com/PHPCompatibility/PHPCompatibilityWP) | [WordPress Developer Docs](https://developer.wordpress.org/coding-standards/) | [Plugin Guidelines](https://developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/)*
