# UGC WordPress Code Quality Seed Collection Prompt

## Mission

Collect real-world examples of **poor WordPress PHP code** from public UGC sources (Stack Overflow, WordPress.org support forums, GitHub issues/PRs, WordPress Stack Exchange, Reddit r/WordPress, r/ProWordPress). Each example must include the **actual code** and a **human-written explanation of what's wrong** — we need the reasoning, not just the code.

Collect as many high-quality examples as you can find. There is no upper limit — every example that meets the quality bar below is valuable. There is also no minimum — 5 excellent boundary-case seeds are worth more than 50 shallow ones. Optimize for signal density, not volume.

These seeds will be used as few-shot exemplars for training a WordPress code quality judge model. They serve triple duty: few-shot exemplars for agent-scale data generation, evaluation ground truth for Spearman correlation, and threshold calibration anchors.

## What to collect

You are looking for forum posts, Q&A threads, GitHub issues, and code review discussions where:

1. **Someone posted WordPress PHP code** (a function, a class method, a SQL query, a hook registration, a template fragment)
2. **Another person (or the community) explained what's wrong with it** — dimension-specific reasoning like "this is vulnerable to SQL injection because you're concatenating user input" or "this N+1 pattern will kill your database at scale"

**Priority targets** (highest value → lowest):

1. **Boundary cases** — Code that looks correct on the surface but has subtle defects that only manifest in specific contexts (at scale, in certain plugin combinations, with specific data shapes). These are worth 3x clear-cut cases.
2. **Multi-dimensional defects** — Code that fails on multiple rubric dimensions simultaneously (e.g., both a security issue AND a performance issue).
3. **WordPress-specific patterns** — Issues that specifically relate to WordPress APIs, hooks, the EAV postmeta schema, WP_Query, REST API, nonce handling, escaping context. Generic PHP issues (null checks, type errors) are lower value.

**All 9 dimensions are in scope.** We already have seeds covering performance, sql_safety, code_quality, wp_api_usage, and security — but a higher-quality UGC example on any of those dimensions is still valuable and should be collected. Better examples can supplement or eventually replace weaker existing ones.

That said, the following dimensions are **underrepresented** in the existing seed set, so high-quality examples addressing them are especially welcome — but only if the source material genuinely addresses them. Do not force examples into dimensions they don't naturally fit.

| Dimension | Current Coverage | Notes |
|-----------|-----------------|-------|
| i18n | Almost none | Hardest to find in UGC — people rarely ask about i18n mistakes |
| accessibility | Minimal | Look for theme/form output reviews |
| dependency_integrity | Minimal | Look for plugin interop issues |
| wpcs_compliance | Light | Look for code review discussions |

### What "higher quality" means for dimensions we already cover:

**performance / sql_safety:** We have many examples of slow WP_Query and meta_query anti-patterns. Higher-quality additions would be: transient misuse, object cache stampedes, autoload bloat in wp_options, unbounded WP_Cron jobs, REST API endpoints without pagination. Novel anti-patterns beat another N+1 example.

**security:** We have nonce and escaping examples. Higher-quality additions would be: CSRF in AJAX handlers, insecure direct object references via REST API, privilege escalation through capability check gaps, file upload validation failures, insecure use of `wp_safe_redirect()` vs `wp_redirect()`.

**code_quality / wp_api_usage:** We have multi-responsibility functions and WP_Query misuse. Higher-quality additions would be: hook priority conflicts, incorrect `add_filter` return values, `wp_die()` misuse in AJAX handlers, `register_rest_route` without `permission_callback`, deprecated function usage with WordPress version context.

### Examples of what to look for in each underrepresented dimension:

**i18n (internationalization):**
- Hardcoded English strings in plugin/theme output without `__()`, `_e()`, `esc_html__()`
- Concatenated translated strings instead of using `sprintf()` with placeholders
- Wrong or missing text domain
- Incorrect use of `_n()` for plurals
- Late escaping violations: `esc_html( __() )` instead of `esc_html__()`

**accessibility:**
- Form inputs without associated `<label>` elements
- Images generated without `alt` attributes in PHP
- Custom interactive widgets without keyboard support or ARIA attributes
- Admin UI not following WordPress admin accessibility patterns
- Missing `screen-reader-text` class usage

**dependency_integrity:**
- Plugins directly `require`-ing vendor files instead of using WordPress autoloading patterns
- Functions calling other custom functions without checking if they exist (`function_exists()`)
- Circular dependency patterns between plugins
- Hard dependency on another plugin without checking if it's active (`is_plugin_active()`)

**wpcs_compliance:**
- Non-Yoda conditions in security-sensitive contexts
- Missing PHPDoc blocks on public API functions
- Wrong naming conventions (camelCase instead of snake_case for functions)
- Missing `@since`, `@param`, `@return` tags

## Where to search

### Primary sources (highest signal-to-noise):

1. **WordPress Stack Exchange** — `wordpress.stackexchange.com`
   - Search: `[security] code review`, `[performance] slow query`, `[plugin-development] anti-pattern`
   - Look for answers with 5+ upvotes that explain what's wrong with the questioner's code
   - Tags: `security`, `performance`, `wp-query`, `wpdb`, `plugin-development`, `hooks`, `rest-api`, `i18n`, `accessibility`

2. **WordPress.org Plugin Review feedback** — `wordpress.org/plugins/*/`
   - Plugin review guidelines violations documented in trac tickets
   - Code review feedback on plugin submissions

3. **GitHub** — Search within WordPress ecosystem repos
   - `github.com` search: `language:PHP "wpdb" "sql injection" OR "unsanitized" OR "not escaped"` 
   - WooCommerce, Elementor, ACF, Yoast issues/PRs with code review comments
   - WordPress core Trac tickets with code review

4. **Stack Overflow** — `stackoverflow.com`
   - Tags: `[wordpress]` combined with `[security]`, `[performance]`, `[sql-injection]`
   - Look for accepted answers that correct the questioner's flawed code

5. **Reddit** — `r/WordPress`, `r/ProWordPress`, `r/PHPhelp`
   - Posts showing code and asking "what's wrong" or "why is this slow"

### Secondary sources:

6. **WordPress Developer Resources** — `developer.wordpress.org`
   - "Common mistakes" documentation, deprecated function notices with migration examples

7. **WordPress VIP Code Review** — `docs.wpvip.com/code-analysis/`
   - Documented anti-patterns with code examples and explanations

8. **PHPCS WordPress Coding Standards** — `github.com/WordPress/WordPress-Coding-Standards`
   - Issue discussions showing real code that violates standards with explanations

## Output schema

Each collected example MUST be structured as a JSON object matching one of two types:

### Type 1: `deep_judge_cot` (preferred — full dimensional analysis)

Use this when the source provides analysis across multiple dimensions.

```json
{
  "seed_id": "ugc_{source}_{sequential_number}",
  "seed_type": "deep_judge_cot",
  "source_url": "https://wordpress.stackexchange.com/questions/XXXXX/...",
  "source_platform": "wordpress_stackexchange|stackoverflow|github|reddit|wp_org|wp_vip",
  "source_file": "description of where this code lives (e.g., 'custom plugin REST endpoint', 'theme functions.php')",
  "code": "THE ACTUAL PHP/SQL CODE — preserve exact formatting, include enough context to judge",
  "human_reasoning": {
    "verdict": "FAIL",
    "dimension_analysis": {
      "performance": {
        "score": 1-10,
        "analysis": "HUMAN-WRITTEN explanation from the source. Must cite specific WordPress patterns by name (e.g., '$wpdb->prepare()', 'wp_verify_nonce()', 'esc_html()'). Must reference specific lines or constructs in the code."
      },
      "security": {
        "score": 1-10,
        "analysis": "..."
      }
    },
    "overall_score": 0-100,
    "key_observation": "One sentence explaining why this example is valuable for training — what makes the defect subtle or interesting"
  },
  "dimensions_addressed": ["performance", "security"],
  "defect_subtlety": "boundary|clear-cut",
  "annotation_type": "ugc_expert"
}
```

### Type 2: `critique_then_fix` (when the source provides both bad code and the fix)

Use this when the source shows the defective code AND a corrected version.

```json
{
  "seed_id": "ugc_{source}_{sequential_number}",
  "seed_type": "critique_then_fix",
  "source_url": "https://...",
  "source_platform": "wordpress_stackexchange|stackoverflow|github|reddit|wp_org|wp_vip",
  "source_file": "description of where this code lives",
  "defective_code": "THE BAD CODE",
  "human_critique": {
    "summary": "1-2 sentence summary of what's wrong",
    "dimensions": {
      "security": {
        "severity": "critical|high|medium|low",
        "score": 1-10,
        "reasoning": "HUMAN-WRITTEN explanation from the source"
      }
    },
    "key_observation": "Why this is a valuable training example"
  },
  "corrected_code": "THE FIXED CODE from the source",
  "dimensions_addressed": ["security"],
  "defect_subtlety": "boundary|clear-cut",
  "annotation_type": "ugc_expert"
}
```

## The 9 rubric dimensions (score 1-10 each)

Every dimension analysis must use the correct field name:

| Field Name | Dimension | What it measures |
|------------|-----------|------------------|
| `wpcs_compliance` | WordPress Coding Standards | Naming, spacing, Yoda conditions, PHPDoc |
| `sql_safety` | SQL Safety | `$wpdb->prepare()`, typed placeholders, no concatenation |
| `security` | Security | Nonces, capability checks, output escaping, no `eval()`/`extract()` |
| `performance` | Performance | No N+1, no `SELECT *`, caching, no unbounded loops |
| `wp_api_usage` | WordPress API Usage | WP_Query over raw SQL, proper hooks, REST permission callbacks |
| `code_quality` | Code Quality | Single responsibility, error handling, no dead code |
| `dependency_integrity` | Dependency Chain | Explicit deps, no circular deps, proper autoloading |
| `i18n` | Internationalization | `__()`, `_e()`, text domains, `sprintf()` with translations |
| `accessibility` | Accessibility | Labels, alt text, ARIA, keyboard support, screen reader text |

**Scoring guide:**
- 1-2: Critical failure, dangerous anti-pattern
- 3-4: Significant issues, would teach bad habits
- 5-6: Below standard, needs improvement
- 7: N/A (dimension doesn't apply to this code)
- 8-10: Meets or exceeds WordPress standards

**Critical rules:**
- Security < 5 = automatic FAIL regardless of other scores
- **Only score dimensions the source material actually addresses.** If the human explanation only discusses security and performance, only include those two dimensions. Do not score dimensions speculatively or to inflate coverage.
- Every scored dimension MUST have an `analysis` or `reasoning` string grounded in what the human actually wrote
- It is perfectly fine for a seed to address only 1-2 dimensions if that's what the source provides — depth on fewer dimensions beats shallow coverage across many

## Quality requirements

### MUST have:
- **Real code** — actual PHP functions, SQL queries, or template fragments (not pseudocode)
- **Human reasoning** — the explanation must come from a human (the answer author, code reviewer, or community member), not be generated by you
- **WordPress specificity** — the defect must relate to WordPress APIs, patterns, or ecosystem (not generic PHP issues)
- **Dimension-specific citations** — reasoning must name specific WordPress functions/patterns (e.g., "`$wpdb->prepare()` is missing", "no `wp_verify_nonce()` check", "`esc_attr()` needed here")

### MUST NOT have:
- **Hosting-specific content** — no references to WP Engine, Kinsta, SiteGround, Pantheon, or any specific hosting platform metrics (BSPs, pods, plan sizes)
- **Fabricated reasoning** — do NOT write the dimensional analysis yourself. Extract it from what the human actually wrote. If the source only addresses 2 dimensions, only include those 2. Do not inflate coverage.
- **Trivial examples** — "missing semicolon" or "undefined variable" are not WordPress code quality issues
- **Code without explanation** — we need the human reasoning, not just the code

### Subtlety classification:
- **`boundary`**: The code appears to work, uses WordPress APIs, but has subtle issues that only manifest in specific contexts (scale, security, edge cases). The defect requires WordPress domain knowledge to identify. **Prefer these.**
- **`clear-cut`**: The defect is obvious to any PHP developer (SQL injection, missing escaping, `eval()` on user input). Include some of these for calibration but prioritize boundary cases.

## Example search queries

```
# WordPress Stack Exchange
site:wordpress.stackexchange.com "wpdb->prepare" "sql injection" code
site:wordpress.stackexchange.com "wp_verify_nonce" missing code review
site:wordpress.stackexchange.com "esc_html" "not escaped" vulnerability
site:wordpress.stackexchange.com "N+1" OR "get_post_meta inside loop" performance
site:wordpress.stackexchange.com "__(" OR "_e(" "text domain" wrong i18n
site:wordpress.stackexchange.com "aria-label" OR "<label" accessibility form

# GitHub
repo:WordPress/WordPress-Coding-Standards label:bug "false negative"
repo:woocommerce/woocommerce label:"type: enhancement" "performance" "query"
"wordpress" "code review" "security" "prepare" language:PHP

# Stack Overflow
[wordpress] [security] "$wpdb->query" -prepare
[wordpress] [performance] "meta_query" slow
[wordpress] [plugin-development] "register_rest_route" "permission_callback"

# Reddit
site:reddit.com/r/WordPress "code review" OR "what's wrong" plugin
site:reddit.com/r/ProWordPress performance query slow
```

## Output format

Save all collected seeds as a single JSON array in:
```
~/Desktop/data/wp-finetune-data/human_seeds/ugc_seeds.json
```

The file should be a valid JSON array of objects, each matching one of the two schemas above.

Also produce a summary file at:
```
~/Desktop/data/wp-finetune-data/human_seeds/ugc_seeds_summary.json
```

With structure:
```json
{
  "total_seeds": N,
  "by_type": {"deep_judge_cot": N, "critique_then_fix": N},
  "by_subtlety": {"boundary": N, "clear-cut": N},
  "by_platform": {"wordpress_stackexchange": N, "stackoverflow": N, ...},
  "dimensions_covered": ["list of dimensions that appear in at least one seed"],
  "dimension_counts": {"performance": N, "security": N, ...}
}
```

## Quality bar — when to include vs skip

**Include** an example when:
- The source provides real code AND a human explanation that names specific WordPress functions, patterns, or APIs
- The explanation teaches something — a reader would learn a WordPress best practice or anti-pattern from it
- The defect is WordPress-specific, not generic PHP (even if generic PHP issues are present alongside WP-specific ones)

**Skip** an example when:
- The human explanation is vague ("this code is bad", "needs improvement") without citing specific constructs
- The code is a trivial snippet (< 3 lines) with an obvious fix that teaches nothing
- The discussion is about configuration, not code (e.g., "change your php.ini settings")
- The only issue is cosmetic (whitespace, bracket style) with no functional or security implication
- You would need to fabricate the dimensional analysis because the source doesn't provide enough reasoning

When in doubt, skip. A smaller set of high-signal seeds is strictly better than a larger set diluted with noise.

## Validation checklist (run before submitting)

- [ ] Every seed has valid JSON structure
- [ ] Every `code` or `defective_code` field contains real PHP/SQL (not pseudocode or descriptions)
- [ ] Every dimension score has an accompanying `analysis` or `reasoning` string **grounded in what the human source actually wrote** — not inferred or fabricated
- [ ] No hosting-platform-specific references in any field
- [ ] Every `source_url` is a real, publicly accessible URL
- [ ] No dimension was scored without the source material addressing it
- [ ] Each seed's `key_observation` explains what makes *this specific example* valuable for training a WordPress code judge
