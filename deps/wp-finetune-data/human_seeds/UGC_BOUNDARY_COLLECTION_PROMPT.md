# UGC WordPress Boundary Case Collection — Targeted Dimension Strengthening

## Mission

Collect **boundary-case** examples of WordPress PHP code quality issues from public UGC sources, specifically targeting four underrepresented rubric dimensions: **accessibility**, **i18n**, **dependency_integrity**, and **wpcs_compliance**.

A boundary case is code that **appears correct on the surface** — it uses WordPress APIs, follows basic patterns, may even pass a casual review — but has subtle defects that only manifest in specific contexts (different locales, assistive technology, plugin combinations, specific WordPress versions, scale). These are the highest-value training examples because LLM-generated reasoning handles clear-cut failures well but struggles with context-dependent subtlety.

These seeds join an existing set of 120 examples and will be used as few-shot exemplars for training a WordPress code quality judge model, as evaluation ground truth, and as threshold calibration anchors.

## Search strategy — cast wide, filter tight

**Exhaustively search every source listed below.** The goal is to find the best possible examples, not the first acceptable ones. Treat this as a research task: survey the entire landscape, bookmark everything that's remotely relevant, then apply the quality bar ruthlessly at the end. You should visit hundreds of pages and discard most of what you find. The yield rate will be low — that's expected and correct.

Do not stop at the first result per type. Search every listed source for every boundary case type before assembling the final set. Multiple high-quality examples for the same type is fine — include all that pass the quality bar.

## Boundary case types needed per dimension

Each dimension below lists specific boundary case types. Find **at least 1 real UGC example per type**. If a single source covers multiple types, that counts for each. If you find high-quality examples beyond these types, include them.

---

### Accessibility — 5 boundary case types (currently 1 boundary seed)

This is the critical gap. Clear-cut accessibility failures (missing `alt`, missing `<label>`) are already covered. We need the subtle ones:

| # | Type | What to look for | Example search queries |
|---|------|-----------------|----------------------|
| A1 | **Semantically incorrect ARIA** | Widget uses `role="button"` on a `<div>` but doesn't implement `keydown` Enter/Space handlers. Or `aria-label` that contradicts the visible text. Or `aria-hidden="true"` on content that's visually present and interactive. | `site:wordpress.stackexchange.com aria-label wrong button div keyboard`, `"role=\"button\"" wordpress keyboard trap` |
| A2 | **Tab order disruption** | PHP-generated markup with `tabindex` values > 0 that create a logical tab order different from visual order. Common in custom WooCommerce checkout fields, multi-step forms, or admin meta boxes. | `site:wordpress.stackexchange.com tabindex order wordpress`, `tabindex positive "wordpress" accessibility` |
| A3 | **Live region misuse** | AJAX updates using `aria-live` on a container that includes non-update content, causing screen readers to re-announce the entire block. Or missing `aria-live` on WooCommerce cart fragment updates. | `wordpress aria-live ajax update screen reader`, `woocommerce cart fragments aria-live accessibility` |
| A4 | **Color-only state indication** | Form validation marking errors only with color change (`border-color: red`) without icon, text, or `aria-invalid`. Or active/selected tabs indicated only by background color. | `wordpress form validation color only accessibility`, `color contrast "aria-invalid" wordpress plugin` |
| A5 | **Focus management in dynamic content** | Modal/popup plugins rendering without focus trapping or `role="dialog"`. Or AJAX-loaded content that doesn't move focus, leaving keyboard users stranded. Or accordion/tab panels that don't manage `aria-expanded`/`aria-selected` state. | `wordpress modal focus trap accessibility`, `wordpress ajax focus management screen reader`, `"aria-expanded" wordpress tab accordion` |

**Accessibility-specific sources** (search all of these):
- WordPress core Trac: `core.trac.wordpress.org` — keyword `accessibility`, component `Accessibility`
- Gutenberg GitHub: `github.com/WordPress/gutenberg` — label `[Type] Accessibility`, label `Accessibility`
- WordPress Accessibility Team blog: `make.wordpress.org/accessibility/`
- WordPress Accessibility coding standards handbook: `developer.wordpress.org/coding-standards/wordpress-coding-standards/accessibility/`
- WebAIM discussion archive: `webaim.org/discussion/` — search for "wordpress"
- Deque University / axe-core GitHub: `github.com/dequelabs/axe-core` — issues mentioning WordPress patterns
- A11y Project: `a11yproject.com` — posts mentioning WordPress or CMS patterns
- WPCampus accessibility audit: `wpcampus.org/accessibility/` (audited WordPress sites)
- WordPress VIP accessibility documentation: `docs.wpvip.com/` — search "accessibility"
- WordPress Theme Review Team handbook: `make.wordpress.org/themes/handbook/` — accessibility requirements section
- GitHub: search `org:WordPress label:accessibility` across all WordPress org repos
- GitHub: search `"aria-" "wordpress" language:PHP` for ARIA usage in WordPress plugins
- Stack Overflow: `[wordpress] [accessibility] [wai-aria]`
- WordPress Stack Exchange: tag `accessibility`

---

### i18n — 4 boundary case types (currently 4 boundary seeds)

Existing seeds cover missing `__()` wrappers and text domain issues. We need the translator-hostile patterns:

| # | Type | What to look for | Example search queries |
|---|------|-----------------|----------------------|
| I1 | **Non-reorderable sprintf placeholders** | `sprintf( __('Showing %d of %d results for %s'), ...)` where translators can't reorder arguments because positional placeholders (`%1$d`, `%2$d`, `%3$s`) aren't used. Languages with different word order can't translate correctly. | `site:wordpress.stackexchange.com sprintf translatable "%1$" placeholder order`, `wordpress i18n sprintf reorder arguments` |
| I2 | **Broken plural forms** | `_n('1 item', '%d items', $count)` where the singular form hardcodes "1" instead of using `%d`, breaking for Slavic/Arabic languages where the "singular" form applies to numbers ending in 1 (21, 31, 101). Or using `_n()` with only 2 forms when the language needs 3+. | `wordpress _n plural slavic arabic forms`, `site:wordpress.stackexchange.com _n plural wrong "nplurals"` |
| I3 | **HTML inside translatable strings** | `__('<strong>Warning:</strong> cannot be undone')` forcing translators to handle markup in `.po` files. Or `sprintf(__('<a href="%s">click here</a>'), $url)` where the entire link structure is translatable. Should use `sprintf('<strong>%s</strong> %s', __('Warning:'), __('cannot be undone'))`. | `wordpress i18n html inside translation string`, `site:wordpress.stackexchange.com __( "<" html translatable`, `wordpress "gettext" html markup translate` |
| I4 | **Locale-dependent formatting assumptions** | Hardcoded `date('F j, Y')` instead of `wp_date()`, `number_format($price, 2, '.', ',')` instead of `number_format_i18n()`, or `strtolower()`/`strtoupper()` on multibyte strings instead of `mb_strtolower()`. Code that works in English but breaks in Turkish (İ/i), German (ß/SS), or CJK locales. | `wordpress wp_date date_i18n locale format`, `wordpress number_format_i18n locale comma decimal`, `site:wordpress.stackexchange.com strtolower multibyte turkish` |

**i18n-specific sources** (search all of these):
- WordPress core Trac: `core.trac.wordpress.org` — component `I18N`, component `L10N`
- GlotPress / translate.wordpress.org forums and discussions
- Polyglots team blog: `make.wordpress.org/polyglots/` — posts about common translation issues
- WordPress i18n handbook: `developer.wordpress.org/plugins/internationalization/`
- WordPress i18n best practices: `developer.wordpress.org/apis/internationalization/`
- GitHub issues on Polylang: `github.com/polylang/polylang`
- GitHub issues on WPML: search `"wpml" i18n sprintf plural` across GitHub
- GitHub issues on TranslatePress: `github.com/translatepress`
- WordPress Plugin Review Team i18n requirements: `developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/` (i18n section)
- WordPress Theme Review Team i18n requirements: `make.wordpress.org/themes/handbook/`
- PHP i18n best practices docs that reference WordPress: search `"gettext" "wordpress" sprintf plural`
- Stack Overflow: `[wordpress] [internationalization] sprintf`, `[wordpress] [gettext] plural`
- WordPress Stack Exchange: tags `i18n`, `localization`, `translation`
- WP-CLI i18n command docs: `developer.wordpress.org/cli/commands/i18n/` — examples of what the tool catches
- WordPress.org plugin review forum: search `"i18n" OR "text domain" OR "translatable"` in plugin review tickets

---

### dependency_integrity — 3 boundary case types (currently 8 boundary seeds)

Existing seeds cover `function_exists()` checks and script handle errors. We need the deeper interop patterns:

| # | Type | What to look for | Example search queries |
|---|------|-----------------|----------------------|
| D1 | **Silent version incompatibility** | Plugin using a WordPress function/class that was introduced in a specific WP version without checking `global $wp_version`. Or using a WooCommerce method that changed signature between WC versions. Code works on the developer's version but fatals on older installs. | `site:wordpress.stackexchange.com "version_compare" wp_version compatibility`, `wordpress plugin minimum version check fatal`, `"requires at least" wordpress function undefined` |
| D2 | **Autoload pollution / options table bloat** | Plugin storing large serialized arrays in `wp_options` with `autoload=yes`, causing every page load to deserialize megabytes of data. Or using `update_option()` without setting `autoload` to `no` for data that's only needed on specific admin pages. | `site:wordpress.stackexchange.com wp_options autoload bloat`, `wordpress autoload options slow`, `"autoload" "yes" wp_options performance plugin` |
| D3 | **Hook priority conflicts / filter corruption** | Plugin adding a filter at default priority 10 that modifies data a later plugin expects unchanged. Or returning wrong type from a filter (array where string expected). Or `remove_action` that silently fails because the callback is a closure or method on a different instance. | `site:wordpress.stackexchange.com remove_action closure priority`, `wordpress filter wrong return type conflict`, `wordpress hook priority plugin conflict` |

**dependency_integrity-specific sources** (search all of these):
- WordPress core Trac: `core.trac.wordpress.org` — components `Plugins`, `Options/Transients`, `Script Loader`
- WordPress.org support forums: `wordpress.org/support/` — search `"conflict" "plugin" fatal`, `"white screen" plugin update`
- GitHub: search `"is_plugin_active" "function_exists" wordpress compatibility` across public repos
- GitHub: popular plugin repos with "conflict" or "compatibility" labels:
  - `github.com/woocommerce/woocommerce` — label `type: compatibility`
  - `github.com/developer-developer-developer/developer-developer-developer` — label `type: bug` with conflict keywords
  - `github.com/developer-developer-developer/developer-developer-developer` — label `Bug`
  - `github.com/developer-developer-developer/developer-developer-developer` — issues mentioning "conflict" or "compatibility"
  - Org-wide searches: `org:woocommerce`, `org:developer-developer-developer`, `org:developer-developer-developer` — filter by "conflict" or "compatibility" in title
- WordPress Plugin Review Team handbook: `developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/` — dependency requirements
- WordPress VIP code review: `docs.wpvip.com/code-analysis/` — autoload and options bloat guidance
- WordPress.org plugin directory reviews: search for `"autoload" "wp_options"` in review feedback
- WordPress core hooks reference: `developer.wordpress.org/reference/hooks/` — documentation of hook contracts
- Stack Overflow: `[wordpress] plugin conflict "remove_action"`, `[wordpress] "autoload" wp_options slow`
- WordPress Stack Exchange: tags `plugin-development`, `hooks`, `options`
- WPScan vulnerability database: `wpscan.com` — version-specific vulnerabilities showing version check failures
- WordPress developer blog: `developer.wordpress.org/news/` — posts about breaking changes between versions
- WordPress Field Guide (per-release): `make.wordpress.org/core/` — "dev notes" tag showing API changes that break plugins

---

### wpcs_compliance — 3 boundary case types (currently 7 boundary seeds)

Existing seeds cover naming conventions and PHPDoc. We need the standards issues that affect behavior, not just style:

| # | Type | What to look for | Example search queries |
|---|------|-----------------|----------------------|
| W1 | **Non-Yoda conditions in security-sensitive contexts** | `if ($role == 'administrator')` instead of `if ('administrator' === $role)` where accidental assignment (`=` instead of `==`) in a capability check would grant admin access to everyone. The defect is that non-Yoda style is merely a style choice in most contexts but becomes a security vector in permission checks. | `site:wordpress.stackexchange.com yoda condition security`, `wordpress "yoda conditions" assignment vulnerability`, `WPCS yoda condition false negative` |
| W2 | **Incorrect sanitization/escaping function choice** | Using `sanitize_text_field()` on data that will be used in a SQL query (should use `$wpdb->prepare()`). Or `esc_html()` on a URL (should use `esc_url()`). Or `wp_kses_post()` on user input that will be echoed in an attribute context (should use `esc_attr()`). The code appears to be "doing security" but uses the wrong function for the context. | `site:wordpress.stackexchange.com sanitize_text_field sql wrong`, `wordpress esc_html url wrong escaping context`, `"wp_kses_post" attribute context vulnerability` |
| W3 | **Late escaping violations with false safety** | `echo __('Hello', 'my-plugin')` where the developer believes `__()` is "safe" because it's a WordPress function, but the translation file could contain malicious HTML. Should be `echo esc_html__('Hello', 'my-plugin')`. Or caching pre-escaped output in a transient, then echoing the transient without re-escaping (stale escaping). | `wordpress late escaping __() translation xss`, `site:wordpress.stackexchange.com esc_html__ vs esc_html __`, `wordpress transient escaping cached output` |

**wpcs_compliance-specific sources** (search all of these):
- PHPCS WordPress-Coding-Standards GitHub: `github.com/WordPress/WordPress-Coding-Standards` — issues, PRs, discussions (especially "false negative" reports showing code that should fail but doesn't)
- WordPress VIP code review documentation: `docs.wpvip.com/code-analysis/phpcs-report/`
- WordPress VIP code examples: `docs.wpvip.com/` — search "escaping", "sanitization", "late escaping"
- WordPress Plugin Review Team handbook: `developer.wordpress.org/plugins/wordpress-org/detailed-plugin-guidelines/`
- WordPress Plugin Review Team Trac: `plugins.trac.wordpress.org` — closed tickets citing WPCS violations
- WordPress core Trac: `core.trac.wordpress.org` — keyword `coding-standards`
- WordPress coding standards handbook: `developer.wordpress.org/coding-standards/wordpress-coding-standards/`
- WordPress security best practices: `developer.wordpress.org/plugins/security/`
- PHPCompatibilityWP GitHub: `github.com/PHPCompatibility/PHPCompatibilityWP` — issues showing WP-specific compat sniffs
- Patchstack blog: `patchstack.com/articles/` — posts about escaping context mistakes in real plugins
- Wordfence blog: `wordfence.com/blog/` — vulnerability disclosures showing wrong sanitization function choice
- WordPress Tavern: `wptavern.com` — articles about plugin security reviews
- Stack Overflow: `[wordpress] [xss] "esc_html" OR "esc_attr" wrong context`
- WordPress Stack Exchange: tags `security`, `sanitization`, `escaping`
- Tom McFarlin's blog / developer WordPress blogs discussing escaping context gotchas
- WordPress core contributor blogs discussing late escaping policy changes

---

---

## Universal source list — search ALL of these

Beyond the dimension-specific sources above, exhaustively search these general sources for boundary cases across any of the four target dimensions. Work through every category systematically.

### WordPress official ecosystem
- **WordPress core Trac**: `core.trac.wordpress.org` — search by component, keyword, milestone. Check closed tickets with patches for code-before/code-after pairs.
- **WordPress Gutenberg GitHub**: `github.com/WordPress/gutenberg` — 60K+ issues/PRs. Search labels, search code in PRs.
- **WordPress meta Trac**: `meta.trac.wordpress.org` — WordPress.org infrastructure issues
- **WordPress developer blog**: `developer.wordpress.org/news/`
- **WordPress core contributor blogs**: `make.wordpress.org/core/`, `make.wordpress.org/plugins/`, `make.wordpress.org/themes/`
- **WordPress Field Guides**: published per release at `make.wordpress.org/core/` with `dev-notes` tag — documents breaking changes, deprecations, and migration patterns

### Q&A platforms
- **WordPress Stack Exchange**: `wordpress.stackexchange.com` — search by tag combination. High-signal tags: `security+code-review`, `performance+wp-query`, `i18n+plugin-development`, `accessibility+theme-development`, `hooks+plugin-development`
- **Stack Overflow**: `stackoverflow.com` — tag `[wordpress]` combined with dimension-specific tags
- **WordPress.org support forums**: `wordpress.org/support/` — the largest volume source but lowest signal-to-noise. Search within specific plugin support forums for technical threads.
- **WordPress Development Stack Exchange**: check both the main site and meta

### GitHub (code search and issue search)
- **GitHub code search**: `github.com/search?type=code` — search for specific anti-patterns in PHP files:
  - `"aria-hidden=\"true\"" "onclick" language:PHP` (A1: interactive hidden elements)
  - `"tabindex=\"2\"" OR "tabindex=\"3\"" language:PHP path:wp-content` (A2: positive tabindex)
  - `"sprintf" "__(" language:PHP "%d" "%d"` without `%1$` (I1: non-reorderable)
  - `"_n(" "1 " language:PHP path:wp-content` (I2: hardcoded singular)
  - `"update_option" "autoload" language:PHP path:wp-content/plugins` (D2: autoload)
  - `"remove_action" "function(" language:PHP` (D3: removing closures)
- **GitHub issue search**: search across WordPress ecosystem orgs:
  - `org:WordPress`, `org:woocommerce`, `org:developer-developer-developer`, `org:developer-developer-developer`, `org:developer-developer-developer`
  - Filter by label: `accessibility`, `i18n`, `compatibility`, `bug`
- **GitHub PR reviews**: PRs with review comments explaining why code was changed — these are often the highest-quality boundary case annotations

### Security research & vulnerability databases
- **Patchstack**: `patchstack.com/database/` and `patchstack.com/articles/` — WordPress-specific vulnerability database with technical writeups
- **Wordfence Threat Intelligence**: `wordfence.com/threat-intel/` — vulnerability disclosures with code analysis
- **WPScan**: `wpscan.com/vulnerabilities/` — vulnerability database
- **Plugin Vulnerabilities**: `pluginvulnerabilities.com` — independent researchers documenting WordPress plugin flaws with code examples
- **NVD/CVE**: `nvd.nist.gov` — search `cpe:2.3:a:*:*:*:*:*:*:wordpress*` for WordPress plugin CVEs with technical details

### WordPress education & community blogs
- **WordPress VIP**: `docs.wpvip.com` — enterprise-grade code review documentation with annotated examples
- **Developer blogs and publications**: search for "wordpress" combined with dimension keywords on:
  - `smashingmagazine.com`, `css-tricks.com` (now DigitalOcean), `developer-developer-developer`, `developer-developer-developer`
  - Envato Tuts+: `code.tutsplus.com` — WordPress tutorial series with code examples
  - WordPress Tavern: `wptavern.com` — plugin reviews, security incidents, code quality debates
  - SitePoint: `sitepoint.com` — WordPress development articles
  - Developer WordPress blogs on Substack, dev.to, Medium, and personal sites
  - Search Google for: `site:dev.to wordpress accessibility boundary`, `site:medium.com wordpress i18n plural sprintf`

### Reddit
- **r/WordPress**: `reddit.com/r/WordPress` — search "code review", "what's wrong", "accessibility", "i18n"
- **r/ProWordPress**: `reddit.com/r/ProWordPress` — more technical than r/WordPress
- **r/webdev**: `reddit.com/r/webdev` — search "wordpress accessibility", "wordpress i18n"
- **r/PHPhelp**: `reddit.com/r/PHPhelp` — search "wordpress"
- **r/accessibility**: `reddit.com/r/accessibility` — search "wordpress"

### Accessibility-specific communities (beyond WordPress)
- **WebAIM mailing list archive**: `webaim.org/discussion/`
- **A11y Slack**: archived discussions at various aggregator sites
- **W3C WAI**: `w3.org/WAI/` — tutorials and techniques that reference CMS patterns
- **Deque blog**: `deque.com/blog/` — accessibility testing articles mentioning WordPress
- **TPGi blog**: `tpgi.com/blog/` — accessibility consulting firm, CMS-related posts
- **Adrian Roselli's blog**: `adrianroselli.com` — prolific accessibility writer, search for WordPress
- **Scott O'Hara's blog**: `scottohara.me` — ARIA pattern implementations
- **Heydon Pickering's Inclusive Components**: `inclusive-components.design` — component patterns applicable to WP widgets

### i18n-specific communities (beyond WordPress)
- **Unicode CLDR**: `cldr.unicode.org` — locale data that explains why formatting assumptions break
- **PHP Internationalization docs**: `php.net/manual/en/book.intl.php` — function references showing correct vs incorrect usage
- **ICU message format docs**: patterns that apply to WordPress translation functions
- **GNU gettext manual**: `gnu.org/software/gettext/manual/` — the underlying technology WordPress uses for translations

### Conference talks & presentations
- **WordPress.tv**: `wordpress.tv` — search for talks about accessibility, i18n, security, coding standards
- **WordCamp talk recordings**: often include code examples with expert commentary
- **WPCampus**: `wpcampus.org` — higher-ed WordPress community with accessibility focus

### Code quality tools & their documentation
- **PHPCS WordPress-Coding-Standards**: `github.com/WordPress/WordPress-Coding-Standards` — issue tracker, wiki, sniff documentation
- **PHPStan WordPress extensions**: `github.com/szepeviktor/phpstan-wordpress` — static analysis findings
- **Psalm WordPress stubs**: issues showing type-level WordPress API misuse
- **SonarQube rules for PHP**: WordPress-relevant rules with examples

---

## Quality bar

**Include** an example when:
- The source provides real code AND a human explanation that names specific WordPress functions, patterns, or APIs
- The defect is **subtle** — a casual reviewer or basic linter would miss it
- The explanation teaches something a WordPress developer might not know (how screen readers traverse ARIA, how GlotPress handles plural forms, how `autoload` affects every page load)
- The code would plausibly pass a basic code review but fail under specific conditions

**Skip** an example when:
- The defect is obvious (missing `alt` text, missing `__()` wrapper, no `$wpdb->prepare()`)
- The human explanation is vague without citing specific constructs
- The code is trivial (< 3 lines) with an obvious fix
- You would need to fabricate the dimensional analysis — if the source doesn't provide reasoning about *why* it's wrong in the relevant dimension, skip it
- The issue is about configuration, not code

**When in doubt, skip.** A single well-sourced boundary case with genuine expert reasoning is worth more than five shallow examples.

## Output schema

Identical to the existing seed format. Each collected example MUST be structured as a JSON object matching one of two types:

### Type 1: `deep_judge_cot`

```json
{
  "seed_id": "ugc_boundary_{source}_{sequential_number}",
  "seed_type": "deep_judge_cot",
  "source_url": "https://...",
  "source_platform": "wordpress_stackexchange|stackoverflow|github|reddit|wp_org|wp_vip|wp_trac|a11y",
  "source_file": "description of where this code lives",
  "code": "THE ACTUAL PHP/SQL/HTML CODE — preserve exact formatting",
  "human_reasoning": {
    "verdict": "FAIL",
    "dimension_analysis": {
      "accessibility": {
        "score": 1-10,
        "analysis": "HUMAN-WRITTEN explanation from the source. Must cite specific patterns (ARIA roles, WCAG criteria, assistive technology behavior). Must reference specific constructs in the code."
      }
    },
    "overall_score": 0-100,
    "key_observation": "What makes this defect subtle — why would a casual reviewer miss it?"
  },
  "dimensions_addressed": ["accessibility"],
  "defect_subtlety": "boundary",
  "annotation_type": "ugc_expert"
}
```

### Type 2: `critique_then_fix`

```json
{
  "seed_id": "ugc_boundary_{source}_{sequential_number}",
  "seed_type": "critique_then_fix",
  "source_url": "https://...",
  "source_platform": "...",
  "source_file": "description of where this code lives",
  "defective_code": "THE BAD CODE",
  "human_critique": {
    "summary": "1-2 sentence summary",
    "dimensions": {
      "i18n": {
        "severity": "critical|high|medium|low",
        "score": 1-10,
        "reasoning": "HUMAN-WRITTEN explanation from the source"
      }
    },
    "key_observation": "What makes this a boundary case?"
  },
  "corrected_code": "THE FIXED CODE from the source",
  "dimensions_addressed": ["i18n"],
  "defect_subtlety": "boundary",
  "annotation_type": "ugc_expert"
}
```

## The 9 rubric dimensions

| Field Name | What it measures |
|------------|------------------|
| `wpcs_compliance` | Naming, spacing, Yoda conditions, PHPDoc |
| `sql_safety` | `$wpdb->prepare()`, typed placeholders, no concatenation |
| `security` | Nonces, capability checks, output escaping, no `eval()`/`extract()` |
| `performance` | No N+1, no `SELECT *`, caching, no unbounded loops |
| `wp_api_usage` | WP_Query over raw SQL, proper hooks, REST permission callbacks |
| `code_quality` | Single responsibility, error handling, no dead code |
| `dependency_integrity` | Explicit deps, no circular deps, proper autoloading |
| `i18n` | `__()`, `_e()`, text domains, `sprintf()` with translations |
| `accessibility` | Labels, alt text, ARIA, keyboard support, screen reader text |

**Scoring:** 1-2 critical, 3-4 significant, 5-6 below standard, 7 N/A, 8-10 passes.

**Only score dimensions the source material genuinely addresses.** If a source discusses an accessibility issue and also happens to show a missing PHPDoc block, only score accessibility unless the source *also* explicitly discusses the PHPDoc issue. Depth on fewer dimensions beats shallow coverage across many.

## Minimum requirements per dimension

These are quality gates on coverage, not quantity gates on volume:

| Dimension | Boundary types defined | Minimum types covered | Currently have |
|-----------|----------------------|----------------------|----------------|
| accessibility | A1-A5 (5 types) | At least 4 of 5 | 1 (focus-visible) |
| i18n | I1-I4 (4 types) | At least 3 of 4 | partial (mostly text domain issues) |
| dependency_integrity | D1-D3 (3 types) | At least 2 of 3 | partial (function_exists, script handles) |
| wpcs_compliance | W1-W3 (3 types) | At least 2 of 3 | partial (naming, PHPDoc) |

You may find multiple examples per type — include all that meet the quality bar. You may also find boundary cases that don't fit any predefined type — include those too if they're genuinely subtle and well-reasoned.

If a specific type proves impossible to find in UGC (e.g., no public discussion of ARIA live region misuse in WordPress context), note it in the summary as an unfillable gap rather than forcing a low-quality example.

## Output format

Save collected seeds as:
```
~/Desktop/data/wp-finetune-data/human_seeds/ugc_boundary_seeds.json
```

Also produce a summary:
```
~/Desktop/data/wp-finetune-data/human_seeds/ugc_boundary_seeds_summary.json
```

```json
{
  "total_seeds": N,
  "by_dimension": {
    "accessibility": {"total": N, "boundary_types_covered": ["A1", "A3", "A5"]},
    "i18n": {"total": N, "boundary_types_covered": ["I1", "I2", "I4"]},
    "dependency_integrity": {"total": N, "boundary_types_covered": ["D1", "D3"]},
    "wpcs_compliance": {"total": N, "boundary_types_covered": ["W2", "W3"]}
  },
  "unfillable_gaps": ["A3 - no UGC found discussing aria-live misuse in WordPress"],
  "by_platform": {"wp_trac": N, "github": N, ...},
  "by_type": {"deep_judge_cot": N, "critique_then_fix": N}
}
```

## Validation checklist

- [ ] Every seed has `"defect_subtlety": "boundary"` — this prompt is exclusively for boundary cases
- [ ] Every seed has real code, not pseudocode or descriptions
- [ ] Every dimension score has reasoning grounded in what the human source actually wrote
- [ ] No hosting-platform-specific references
- [ ] Every `source_url` is publicly accessible
- [ ] The `key_observation` explains what makes the defect *subtle* — why a casual reviewer would miss it
- [ ] At least 4 of 5 accessibility boundary types (A1-A5) are covered, or gaps are documented as unfillable
- [ ] At least 3 of 4 i18n boundary types (I1-I4) are covered, or gaps are documented
- [ ] At least 2 of 3 dependency_integrity boundary types (D1-D3) are covered, or gaps are documented
- [ ] At least 2 of 3 wpcs_compliance boundary types (W1-W3) are covered, or gaps are documented
