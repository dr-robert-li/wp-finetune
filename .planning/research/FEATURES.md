# Feature Research

**Domain:** WordPress-specific code generation and judgment language model
**Researched:** 2026-03-26
**Confidence:** HIGH (model's feature spec is defined by the existing pipeline; confirmed against WPCS docs, CodeWP analysis, and LLM-as-judge research)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the model must demonstrate to be considered usable for WordPress development.
Missing any of these = model is worse than a general-purpose code LLM with a system prompt.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| PHPCS-passing PHP output | All serious WP tooling enforces WPCS; generated code that fails PHPCS is immediately rejected by CI pipelines | MEDIUM | Pipeline enforces this as a training quality gate — the trained model must internalize it, not rely on post-processing |
| Correct `$wpdb->prepare()` usage | SQL injection is the top WordPress plugin vulnerability class (Patchstack 2025 report); unprepared queries are an instant trust-breaker | HIGH | Must handle `%s`/`%d`/`%f` placeholders, typed correctly; `%1$s` positional usage also required |
| Nonce generation and verification | CSRF protection via nonces is WordPress convention; missing nonces on state-changing handlers is a common critical vulnerability | MEDIUM | Must know `wp_nonce_field()`, `wp_verify_nonce()`, `check_ajax_referer()`, `check_admin_referer()` |
| Capability checks before privileged operations | `current_user_can()` gating is WPCS mandatory pattern; missing capability checks are exploited constantly (Q3 2025 Patchstack data) | MEDIUM | Must understand capability hierarchy: `manage_options`, `edit_posts`, `delete_users`, etc. |
| Context-appropriate output escaping | `esc_html()`, `esc_attr()`, `esc_url()`, `wp_kses()` — wrong escaping function for context is a functional bug, not just style | MEDIUM | Model must select the right escape function based on output context |
| WP_Query over raw SQL for post queries | Using raw SQL for post queries is a WPCS critical failure; WP_Query is the canonical API | MEDIUM | Includes meta_query, tax_query, date_query construction |
| Hook registration with correct signature | `add_action`/`add_filter` with wrong argument count is a silent runtime bug; priority management affects load order across plugins | LOW | Must match `$accepted_args` to callback signature |
| `register_rest_route()` with `permission_callback` | REST routes missing `permission_callback` is a critical security gap (WPCS critical failure); exposed in 2025 WP security tooling | MEDIUM | Must never omit permission_callback, even for public routes (use `__return_true` explicitly) |
| PHPDoc blocks on public functions | WPCS requires `@param`, `@return`, `@since` on public API functions; required for WP.org plugin submission | LOW | Must know when PHPDoc is required vs optional (private helpers) |
| i18n wrapping for user-facing strings | Translation wrappers (`__()`, `_e()`, `esc_html__()`, `_n()`) are mandatory for WP.org-hosted plugins | LOW | Late-escaping pattern (`esc_html__()`) preferred over chaining |

### Differentiators (Competitive Advantage)

Features that distinguish wp-qwen3-moe from "ChatGPT with a WordPress system prompt."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dual-mode via task tokens (`<wp_gen>` / `<wp_judge>`) | Single model does both generation AND rubric scoring; eliminates two-model workflow for code review pipelines | HIGH | Enabled by MoE routing — specialized expert pathways per task type. Core differentiator from CodeWP (generation-only) |
| 9-dimension structured judgment output | Returns JSON with per-dimension scores (wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility) plus critical_failures list and training_tags | HIGH | Actionable at dimension level, not just pass/fail. Enables targeted remediation. Research shows structured rubric output outperforms holistic scoring for code quality (ICLR 2025 RocketEval) |
| Contrastive defect explanation | Given a mutation pair (bad → good), explains exactly what changed and why — not just "this is wrong" | HIGH | Trained on Phase 2 mutation data with CoT annotations; enables use as an automated code-review explainer |
| Chain-of-thought reasoning for complex patterns | Explains why a pattern is correct (e.g., why a transient is used here over object cache, why this specific WP_Query arg avoids an N+1) | HIGH | Phase 3 CoT examples cover SQL, performance, and architecture patterns. Distinguishes from models that generate correct code without explainability |
| Taxonomy-grounded coverage across WP concept space | Covers all 12 taxonomy categories (sql_patterns, security, hooks_and_filters, data_modeling, rest_api, admin, theme_patterns, performance, plugin_architecture, multisite, cron, i18n, accessibility) with minimum example counts enforced | HIGH | General-purpose models trained on GitHub have uneven WP coverage; multisite and cron patterns are particularly underrepresented in general corpora |
| Multisite awareness | Generates code that correctly handles `switch_to_blog()`, per-site table prefixes, network options vs site options | HIGH | Multisite patterns are nearly absent from general code model training data; common pain point for WP hosting companies and WP VIP developers |
| Security-aware generation with violation detection | Generates code that avoids the five mutation categories (stripped prepare(), removed nonces, stripped escaping, removed capability checks, injected SELECT *) AND can flag those violations in existing code | HIGH | Pipeline trains on controlled mutation data — model learns both the correct pattern and the characteristic shape of violations |
| Style anchoring to real plugin conventions | Generation matches conventions of real, high-quality WordPress plugins rather than abstract best practices | MEDIUM | Achieved via few-shot style anchors from Phase 1 passed code during synthetic generation; output "feels like" WooCommerce/Jetpack, not textbook PHP |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem like good ideas but should be explicitly excluded from v1.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| JavaScript / Gutenberg block generation | Gutenberg/React blocks are a large part of modern WP dev | Entirely different training domain (JS/React/block.json); mixing PHP and JS training without separate expert pathways dilutes both | Explicitly out of scope for v1 per PROJECT.md; address in v2 with a separate `<wp_block>` task token and dedicated JS training data |
| General PHP framework generation (Laravel, Symfony) | Developers want one model for all PHP | WordPress PHP patterns actively conflict with framework patterns (DI containers, PSR autoloading vs `require`, ActiveRecord vs `$wpdb`) — training contamination degrades WP-specific behavior | Model is intentionally WordPress-scoped; use a general PHP model for framework code |
| Real-time PHPCS correction loop at inference | Users want the model to auto-fix generated code | Inference-time PHPCS toolchain creates hard infrastructure dependency; increases latency; creates false confidence when PHPCS passes but logic is wrong | Train the model to generate PHPCS-passing code on the first attempt; the judge mode can flag remaining issues |
| Binary PASS/FAIL judgment without scores | Seems simpler for downstream consumers | Binary output discards dimension-level signal; a function that fails only on i18n needs different remediation than one failing on SQL safety | Always return full 9-dimension scores; consumer can reduce to binary by checking `verdict` field if needed |
| Explanation of WordPress core internals | Developers ask "how does WP_Query work internally?" | Question-answering about WP internals is a documentation retrieval task, not code generation or judgment; optimizing for it dilutes the code-focused training signal | Defer to WP documentation search or a general-purpose model for conceptual questions |
| DPO/RLHF preference optimization | Users want a "better" model post-SFT | RLHF requires a separate reward model and PPO training loop; DPO requires preference pairs beyond the SFT dataset; both are v2 concerns that SFT alone addresses adequately for initial release | Evaluate PHPCS pass rate and judge correlation at v1; if metrics satisfy thresholds (>95% PHPCS, >0.85 correlation), SFT is sufficient |
| Multi-lingual comment generation | International WP developers want native language comments | Multi-lingual requires separate training data curation, translation validation, and language-specific PHPDoc conventions; out of scope per PROJECT.md | English-only for v1; i18n wrapping of user-facing strings (a separate concern) is in scope |

---

## Feature Dependencies

```
[PHPCS-passing output]
    └──requires──> [WPCS naming, spacing, formatting internalized from training data]

[SQL safety]
    └──requires──> [wpdb->prepare() with typed placeholders]
                       └──requires──> [placeholder type selection logic (%s vs %d)]

[Security (nonces + capability checks)]
    └──requires──> [nonce API knowledge]
    └──requires──> [capability hierarchy knowledge]

[wp_judge task token mode]
    └──requires──> [wp_gen task token mode] (judge evaluates generation output)
    └──requires──> [9-dimension rubric internalized from Phase 2 judge training data]
    └──requires──> [structured JSON output format]

[Contrastive defect explanation]
    └──requires──> [wp_judge mode]
    └──requires──> [mutation pair CoT annotations from Phase 2/3 training data]
    └──enhances──> [wp_judge mode] (explains WHY a score is low)

[Chain-of-thought reasoning]
    └──requires──> [Phase 3 CoT training examples]
    └──enhances──> [wp_gen mode] (explains generated code)
    └──enhances──> [wp_judge mode] (explains judgment rationale)

[Taxonomy-grounded coverage]
    └──requires──> [Phase 2 gap analysis + synthetic generation complete]
    └──enables──> [multisite awareness] (multisite is a taxonomy category with min coverage)
    └──enables──> [cron pattern generation]
    └──enables──> [REST API generation with permission_callback]

[Dual-mode task tokens]
    └──requires──> [MoE conversion + tokenizer extension with <wp_gen>/<wp_judge>]
    └──requires──> [50/50 gen/judge training data split]
    └──conflicts──> [single-task fine-tuning] (task tokens meaningless without routing)
```

### Dependency Notes

- **wp_judge requires wp_gen training first:** The judge mode evaluates code quality; without strong generation training, the model has no quality baseline to judge against. The 50/50 dataset split ensures both pathways are trained simultaneously.
- **Structured JSON output requires rubric internalization:** The 9-dimension JSON format is only reliable if the model has seen hundreds of rubric-scored examples during training. Phase 2 judge_dataset.py provides these.
- **Multisite depends on taxonomy gap analysis completing:** Multisite patterns have minimum coverage requirements in taxonomy.yaml (60 examples for `multisite:per_site_tables`). If Phase 2 generation is skipped or incomplete, multisite coverage will be absent.
- **Contrastive explanation conflicts with binary-only judgment:** If the judge mode is trained only on PASS/FAIL labels without mutation metadata and CoT, defect explanation capability is lost. Phase 3 CoT annotations for mutation pairs must not be skipped.

---

## MVP Definition

### Launch With (v1)

These are the minimum capabilities for the model to be useful and trustworthy.

- [ ] PHPCS-passing PHP generation (WPCS compliance) — without this, the model is less useful than PHPCS itself
- [ ] SQL safety (`$wpdb->prepare()` usage) — the highest-severity WordPress vulnerability class; non-negotiable
- [ ] Nonce verification and capability check patterns — table-stakes security for any plugin code
- [ ] Output escaping with context-appropriate function selection — prevents XSS in generated code
- [ ] WP_Query, hook registration, REST API route generation — the three most common WP-specific coding tasks
- [ ] `<wp_judge>` mode returning 9-dimension JSON with scores, verdict, and critical_failures — the core differentiator that doesn't exist in any competitor tool
- [ ] CoT reasoning for SQL and security patterns — makes the model explainable, not just generative

### Add After Validation (v1.x)

- [ ] Contrastive defect explanation with mutation pair annotation — add when judge correlation metric confirms rubric alignment (>0.85 threshold)
- [ ] Multisite-specific pattern generation — add when base security and SQL patterns are validated; multisite is specialized enough that early errors here damage trust less than core security errors
- [ ] Admin UI generation (settings pages, meta boxes, list table columns) — high developer demand but lower security criticality; validate core generation first

### Future Consideration (v2+)

- [ ] DPO/RLHF preference optimization — defer until SFT metrics plateau; requires new training infrastructure
- [ ] JavaScript/Gutenberg block generation via `<wp_block>` task token — requires entirely new training data domain
- [ ] Multi-lingual comment support — requires translation validation pipeline
- [ ] WooCommerce-specific expert pathway (`<wc_gen>`) — WooCommerce has its own conventions (CRUD, hooks, templates); warrants separate task token and training data

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| PHPCS-passing generation | HIGH | MEDIUM (training data quality gate already enforced) | P1 |
| SQL safety (`$wpdb->prepare()`) | HIGH | HIGH (must correctly handle all placeholder types) | P1 |
| Nonce + capability checks | HIGH | MEDIUM | P1 |
| Output escaping (context-correct) | HIGH | MEDIUM | P1 |
| WP_Query / hooks / REST API generation | HIGH | HIGH (broad coverage required) | P1 |
| `<wp_judge>` 9-dimension JSON output | HIGH | HIGH (requires Phase 2 judge dataset complete) | P1 |
| PHPDoc generation | MEDIUM | LOW | P1 |
| i18n wrapping | MEDIUM | LOW | P1 |
| CoT reasoning for complex patterns | HIGH | HIGH (Phase 3 required) | P1 |
| Contrastive defect explanation | HIGH | HIGH (Phase 2 mutation pairs + CoT required) | P2 |
| Multisite patterns | MEDIUM | HIGH (specialized training data, Phase 2 gap fill) | P2 |
| Admin UI generation (meta boxes, settings) | MEDIUM | MEDIUM | P2 |
| Cron / scheduled event patterns | LOW | MEDIUM | P2 |
| Plugin architecture patterns (activation, uninstall) | MEDIUM | LOW | P2 |
| WooCommerce-specific generation | HIGH (for WC devs) | HIGH (separate task token + training data) | P3 |
| Gutenberg block generation | HIGH (modern WP) | VERY HIGH (entirely different domain) | P3 |
| DPO/RLHF refinement | MEDIUM | VERY HIGH (new infra required) | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

| Feature | CodeWP | GitHub Copilot / General LLMs | wp-qwen3-moe approach |
|---------|--------|-------------------------------|----------------------|
| WP-specific code generation | Yes — trained on WP core + popular plugins | Partial — general GitHub training with WP patterns present but inconsistent (Gemini mixes WP/Drupal/Laravel) | Yes — trained exclusively on curated PHPCS-passing WP code |
| WPCS compliance enforcement | Partial — generates WP-style code but no formal WPCS guarantee | No — requires post-hoc PHPCS run | Yes — PHPCS pass rate >95% is evaluation target; baked into training data quality gate |
| Code quality judgment / review | No — generation only | Partial — can critique code but without WP-specific rubric | Yes — `<wp_judge>` mode with 9-dimension structured rubric |
| Structured JSON judgment output | No | No | Yes — JSON with per-dimension scores, critical_failures, training_tags |
| Security-specific defect detection | No formal mechanism | Ad hoc — depends on prompt engineering | Yes — trained on controlled mutation pairs; knows the shape of nonce removal, unprepared queries, etc. |
| Contrastive explanation (bad→good) | No | No | Yes — Phase 2 mutation pairs with CoT annotations |
| Chain-of-thought reasoning | No | Partial (general reasoning, not WP-grounded) | Yes — Phase 3 CoT specifically for WP patterns |
| Multisite awareness | Unknown / limited | Very limited | Yes — taxonomy-enforced minimum coverage |
| Dual-mode single model (gen + judge) | No — separate products/prompts | No — separate calls to different models | Yes — single model, task token routes to MoE expert pathway |
| Open model / self-hostable | No (SaaS) | No (SaaS/enterprise) | Yes — GGUF for Ollama, AWQ for vLLM; runs on DGX Toolbox |

---

## Sources

- [CodeWP AI capabilities analysis](https://deepgram.com/ai-apps/codewp) — MEDIUM confidence (third-party summary)
- [Best WordPress AI tools 2026 (Varun Dubey)](https://vapvarun.com/ai-tools-wordpress-plugin-development/) — MEDIUM confidence (practitioner assessment)
- [State of WordPress Security 2025 — Patchstack](https://patchstack.com/whitepaper/state-of-wordpress-security-in-2025/) — HIGH confidence (primary security research)
- [Q3 2025 Most Exploited WP Vulnerabilities — Patchstack](https://patchstack.com/articles/q3-2025s-most-exploited-wordpress-vulnerabilities-and-how-patchstacks-rapidmitigate-blocked-them/) — HIGH confidence
- [LLM-as-a-Judge complete guide — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) — HIGH confidence
- [Benchmarking Correctness and Security in Multi-Turn Code Generation — OpenReview](https://openreview.net/forum?id=zH9aX65Zyi) — HIGH confidence (peer-reviewed)
- [Fine-tuning LLMs for secure code generation — Springer/ESE](https://link.springer.com/article/10.1007/s10664-026-10803-9) — HIGH confidence (peer-reviewed 2026)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1) — HIGH confidence (official)
- [Unsloth Qwen3 fine-tuning documentation](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — HIGH confidence (official tooling docs)
- [WordPress Namespaces and Coding Standards 2025 — WP Developer Blog](https://developer.wordpress.org/news/2025/09/implementing-namespaces-and-coding-standards-in-wordpress-plugin-development/) — HIGH confidence (official WP docs)
- Project pipeline: `config/judge_system.md` and `config/taxonomy.yaml` — HIGH confidence (primary source)

---

*Feature research for: wp-qwen3-moe WordPress code generation and judgment model*
*Researched: 2026-03-26*
