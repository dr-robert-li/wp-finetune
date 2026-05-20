# Phase 1b Calibration Disagreement Review — COMPLETED

**Source:** `data/phase1b/rejudge_full_20k.jsonl` (20000 rows)  
**Sampling:** 20 disagreements per Claude-bucket, balanced PASS->FAIL / FAIL->PASS  
**Seed:** 42  
**Reviewer:** AI reviewer pass + human final vote  
**Review date:** 2026-05-20  
**Status:** ✅ FINAL — all 80 cases resolved

---

## Reviewer Analysis & Aggregate Verdicts

### Summary Table (FINAL — post human vote)

| Bucket | Sub-group | n | CLAUDE | CAL | NEITHER | Majority |
|--------|-----------|---|--------|-----|---------|----------|
| 0–4.99 | FAIL→PASS (all) | 20 | 0 | 20 | 0 | **CAL** |
| 7–7.99 | PASS→FAIL | 10 | 10 | 0 | 0 | **CLAUDE** |
| 7–7.99 | FAIL→PASS | 10 | 2 | 8 | 0 | **CAL** |
| 8–8.99 | PASS→FAIL | 10 | 10 | 0 | 0 | **CLAUDE** |
| 8–8.99 | FAIL→PASS | 10 | 0 | 10 | 0 | **CAL** |
| 9–10 | PASS→FAIL | 10 | 8 | 0 | 2 | **CLAUDE** |
| 9–10 | FAIL→PASS | 10 | 2 | 8 | 0 | **CAL** |
| **TOTAL** | | **80** | **32** | **46** | **2** | **CAL 59.0%** |

> _Human reviewer resolved all 3 UNCLEAR cases as CLAUDE: #1 (Jetpack XML-RPC raw POST), #2 (LearnPress save_post no internal nonce), #3 (CF7 quiz filter no nonce). This adds 3 to CLAUDE column, 0 to CAL._

---

## Root-Cause Patterns

### Why CAL is correct (FAIL→PASS sub-groups + 0-4.99)

1. **0–4.99 bucket — test code blanket exclusion:** Claude applies `'test code, excluded from training data'` as a hard FAIL regardless of code quality. The rubric/calibration correctly scores PHPUnit test functions on their actual code quality (D1_wpcs, D8_errors, D9_structure). All 20 cases are well-written Gutenberg core tests. The exclusion is a *dataset curation* decision, not a quality verdict — calibration is correct here.

2. **7–7.99 / 8–8.99 / 9–10 FAIL→PASS:** Claude FAILs primarily on `wpcs_compliance` (score 7, threshold 8) due to missing `@since`, `@param`, `@return` PHPDoc on WooCommerce internal classes. WooCommerce uses PSR-style docblocks that legitimately omit `@since`. Rubric D1_wpcs=10 because no *code* WPCS violations exist, only annotation style preferences. CAL is correct — these are production-quality classes.

### Why CLAUDE is correct (PASS→FAIL sub-groups + 3 UNCLEAR resolved)

**SEC-N04 false positive pattern — systematic calibration flaw.** Across all PASS→FAIL cases in 7–7.99, 8–8.99, and 9–10:
- Claude PASSes code with proper `$wpdb->prepare()`, `register_rest_route()` with `permission_callback`, `dbDelta()`, and admin-context migration scripts.
- The rubric fires `SEC-N04` (no nonce/capability check detected) on these, collapsing `D2_security` to **3.0**, which drags `calibrated_overall` below the PASS threshold.
- `SEC-N04` is **context-unaware**: it cannot distinguish admin-only migration scripts, REST routes with permission callbacks, or DB helper classes where auth is enforced by the caller.
- This is a **systematic calibration defect** in the PASS→FAIL direction.

**3 UNCLEAR cases resolved CLAUDE by human reviewer:**
- `WP_Com_Markdown::check_for_early_methods` — raw `$_POST` read is a genuine concern even in XML-RPC context; not a false positive.
- LearnPress `save_post` handler — `SEC-N04` + `SEC-N18` are legitimate; function should verify nonce internally for defensiveness.
- `wpcf7_quiz_validation_filter` — unprotected `$_POST[$name]` read at filter level; Claude's sec=7 is correct.

### NEITHER (2 cases — 9-10 bucket)
- `setVp8Munger` (LiveKit protobuf): Not WordPress code. Both judges wrong.
- `scssphp Compiler.__construct`: Vendored SCSS library, not WP plugin code. Both judges wrong. Pre-filter vendored paths.

---

## Calibration Trustworthiness Verdict (FINAL)

**CAL is CONDITIONALLY TRUSTWORTHY — downstream-ready with one fix.**

| Direction | Trustworthy? | Reason |
|-----------|-------------|--------|
| FAIL→PASS (CAL upgrades) | ✅ Yes | CAL correctly overrides over-strict WPCS/docblock threshold and test-code blanket exclusion |
| PASS→FAIL (CAL downgrades) | ❌ No | SEC-N04 fires false positives on admin-context DB scripts, REST controllers, migration functions — D2_security collapses to 3.0 |

### Recommended Actions Before Downstream Use

1. **Suppress or re-weight SEC-N04** in the calibration formula for:
   - Functions in files matching `update-*.php`, `install*.php`, `*migration*.php`
   - Functions containing `register_rest_route(` with a `permission_callback` key present
   - Classes that extend `WP_REST_Controller`
2. **Re-run calibration** on the ~35% PASS→FAIL disagreement population with patched SEC-N04.
3. **0-4.99 bucket:** Apply test-code exclusion as a *pre-filter by path pattern* (`/phpunit/`, `/tests/`, `-test.php`) before quality scoring, not inside the quality judge.
4. **Vendor pre-filter:** Exclude `vendor/`, `node_modules/`, vendored lib paths from scoring entirely (catches LiveKit, scssphp false entries).

---

## Per-Bucket Case Verdicts (FINAL)

### Bucket 0–4.99 — FAIL→PASS (n=20)
**CLAUDE=0 | CAL=20 | Majority: CAL**  
_All Gutenberg PHPUnit test functions. Claude applies blanket test-exclusion rule. CAL scores on code quality (correctly)._

| Case | Verdict |
|------|--------|
| #1 data_generate_font_size_preset_deprecated_fixtures | **CAL** |
| #2 tear_down (block-visibility) | **CAL** |
| #3 test_should_keep_figcaption_if_it_is_not_empty | **CAL** |
| #4 test_gutenberg_block_core_navigation_overlay_html | **CAL** |
| #5 test_sanitize_for_block_with_style_variations | **CAL** |
| #6 data_update_separator_declarations | **CAL** |
| #7 test_block_core_paragraph_render_appends_css_class | **CAL** |
| #8 test_color_with_skipped_serialization_block_supports | **CAL** |
| #9 test_excludes_unsupported_types | **CAL** |
| #10 test_get_stylesheet_with_block_json_selectors | **CAL** |
| #11 test_should_remove_multiple_declarations | **CAL** |
| #12 test_register_template_invalid_name | **CAL** |
| #13 data_get_core_data | **CAL** |
| #14 test_remove_insecure_properties_removes_unsafe_preset_settings | **CAL** |
| #15 test_gutenberg_get_markup_for_inner_block_site_title | **CAL** |
| #16 test_set_spacing_sizes_when_invalid | **CAL** |
| #17 test_get_stylesheet_custom_root_selector | **CAL** |
| #18 test_block_core_navigation_block_contains_core_navigation_no_navigation | **CAL** |
| #19 tear_down (resolve-patterns) | **CAL** |
| #20 test_render_block_core_file | **CAL** |

---

### Bucket 7–7.99 — PASS→FAIL (n=10)
**CLAUDE=10 | CAL=0 | Majority: CLAUDE**  
_SEC-N04 false positives on rank-math migration scripts + DB helpers. Claude correct._

| Case | Verdict | Root cause |
|------|---------|------------|
| #1 rank_math_1_0_98_fix_as_groups | **CLAUDE** | SEC-N04 FP on admin migration script |
| #2 Query_Builder::insert | **CLAUDE** | SEC-N04 FP on $wpdb->insert wrapper |
| #3 Posts::get_posts | **CLAUDE** | SEC-N04 FP on sitemap query builder |
| #4 iworks_orphan_change_options_autoload_status | **CLAUDE** | SEC-N04 FP + STR-N01 minor |
| #5 rank_math_1_0_98_as_get_group_id | **CLAUDE** | SEC-N04 FP on get_var helper |
| #6 woocommerce_product_attributes_registration | **CLAUDE** | SEC-N04 FP on WXR importer |
| #7 OD_URL_Metrics_Post_Type::delete_all_posts | **CLAUDE** | SEC-N04 FP, has phpcs:disable justification |
| #8 Stats::get_position_data_by_dimension | **CLAUDE** | SEC-N04 FP on analytics query |
| #9 say_what_install | **CLAUDE** | SEC-N04 FP on dbDelta install |
| #10 WC_REST_Payments_Fraud_Outcomes_Controller::register_routes | **CLAUDE** | SEC-N04 FP on REST controller |

---

### Bucket 7–7.99 — FAIL→PASS (n=10)
**CLAUDE=2 | CAL=8 | Majority: CAL**  
_CAL correct 8/10. Cases #14 and #18 resolved CLAUDE by human reviewer (genuine security gaps)._

| Case | Verdict | Note |
|------|---------|------|
| #11 seo-by-rank-math savelinks batch INSERT | **CAL** | |
| #12 AdminMenu::fix_admin_menu | **CAL** | |
| #13 WooCommerce API handler | **CAL** | |
| #14 WP_Com_Markdown::check_for_early_methods | **CLAUDE** | ✏️ Human vote — raw $_POST in XML-RPC |
| #15 woocommerce-payments process_payment | **CAL** | |
| #16 TFPostsWidget::register_controls | **CAL** | |
| #17 sydneytoolbox portfolio render | **CAL** | |
| #18 learnpress save_post | **CLAUDE** | ✏️ Human vote — no internal nonce verify |
| #19 WidgetImporter::import_data | **CAL** | |
| #20 admin-menu fix submenu | **CAL** | |

---

### Bucket 8–8.99 — PASS→FAIL (n=10)
**CLAUDE=10 | CAL=0 | Majority: CLAUDE**  
_SEC-N04 false positives on woocommerce-payments REST controllers and speculation-rules plugin. Claude correct._

| Case | Verdict |
|------|--------|
| #1–#10 (all woocommerce-payments / WC internal REST controllers) | **CLAUDE** ×10 |

---

### Bucket 8–8.99 — FAIL→PASS (n=10)
**CLAUDE=0 | CAL=10 | Majority: CAL**  
_Claude FAILs WooCommerce internal classes for missing @since/@param/@return. WC uses PSR-style docblocks. CAL correct._

| Case | Verdict |
|------|--------|
| #11–#20 (all WooCommerce / WC-adjacent internal classes) | **CAL** ×10 |

---

### Bucket 9–10 — PASS→FAIL (n=10)
**CLAUDE=8 | CAL=0 | NEITHER=2 | Majority: CLAUDE**  
_SEC-N04 false positives #1-#8. NEITHER for #9 (LiveKit protobuf — not WP code) and #10 (scssphp vendor library)._

| Case | Verdict |
|------|--------|
| #1 LP_Install_Sample_Data::__construct | **CLAUDE** |
| #2 masvideos_get_default_persons_per_row | **CLAUDE** |
| #3–#8 (WP plugin production code, SEC-N04 FP) | **CLAUDE** ×6 |
| #9 setVp8Munger (LiveKit protobuf — not WP) | **NEITHER** |
| #10 wp-scss Compiler.__construct (scssphp vendor) | **NEITHER** |

---

### Bucket 9–10 — FAIL→PASS (n=10)
**CLAUDE=2 | CAL=8 | Majority: CAL**

| Case | Verdict | Reason |
|------|---------|--------|
| #11 wpcf7_quiz_validation_filter | **CLAUDE** | ✏️ Human vote — unprotected $_POST read at filter level |
| #12 get_activity_summary (unparameterized SQL) | **CLAUDE** | SQL-N01/N03 correctly fire; no prepare() |
| #13–#20 (WC/plugin production classes) | **CAL** ×8 | Claude WPCS/docblock over-penalization |

---

## Final Tally (FINAL — all 80 resolved)

| | CLAUDE | CAL | NEITHER |
|--|--------|-----|---------|
| Count | 32 | 46 | 2 |
| % (excl NEITHER) | 41.0% | **59.0%** | — |

**Overall: CAL is correct in 59.0% of adjudicated cases.**  
**Trustworthy for FAIL→PASS upgrades. Requires SEC-N04 context-awareness patch before trusting PASS→FAIL downgrades.**
