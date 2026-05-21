# Phase 1b — New FAIL->PASS Flip Review (post SEC-N04 fix) — COMPLETED

**Population:** 1643 functions that flipped cal FAIL->PASS after the SEC-N04 patch.
**Question:** is each newly-PASS function actually training-worthy? (≥18/20 good = ship; ≤14/20 = suppression too broad)
**Branch counts:** admin_path=39, rest=27, llm_revised=1216, other=361

**Review completed:** 2026-05-21
**Reviewer:** Council (GPT-5.5 Thinking / Claude 4.7 Opus Thinking / Gemini 3.1 Pro Thinking) + Robert Li (deciding vote on 3 contested cases)

---

## FINAL TALLY

| Verdict | Count |
|---------|-------|
| ✅ training-worthy | 23 |
| ❌ auth-missing (bad lift) | 2 |
| ❔ unclear | 0 |
| **Total reviewed** | **25** |

**DECISION: ✅ SHIP CALIBRATION** (23/25 ≥ threshold of 18/20)

### Bad-lift cases (2)
1. `jupiterx-core::export_popup_action` — reads `$_GET`, outputs content, calls `die()`, no nonce/cap check visible; SEC-N04 still fires post-patch (**Robert Li deciding vote**)
2. `woocommerce::PostsToOrdersMigrationController::db_query` — raw unprepared query string passed directly to `$wpdb->query()`; bad training exemplar for SQL safety even in migration context (**Robert Li deciding vote**)

### Contested-but-approved case (1)
- `plugnmeet.../Livekit/AgentEvent::setType` — vendored/generated protobuf SDK code; approved training-worthy on grounds it is harmless and has correct WP-style setter pattern (**Robert Li deciding vote**)

---

## Council Consensus Notes

- **admin_path flips** are legitimate: upgrade helpers (`upgrade_380`, `WP_Upgrader::create_lock`) and DB migration steps run in privileged admin-only context — SEC-N04 was a false positive here.
- **rest flips** are legitimate: all REST cases use explicit `permission_callback` delegation — correct WP REST API auth pattern; SEC-N04 was suppressing them incorrectly.
- **llm_revised flips** are the bulk of population (1216/1643 = 74%); sampled cases show LLM self-revision to cleaner patterns, not heuristic suppression artifacts.
- **other flips**: plugin activation hooks and async-action schedulers are privileged lifecycle code; acceptable. `export_popup_action` is the outlier — web-reachable superglobal handler without auth.
- **Raw SQL risk**: `db_query` in WooCommerce migration controller passes unescaped query string to `$wpdb->query()` with a `phpcs:ignore` comment. Contextually correct but a bad training exemplar that could generalize incorrectly.

---

## Full Case Verdicts

### [admin_path] modify_tb_lp_order_items
- **row_id:** `learnpress-bbpress::inc/updates/learnpress-upgrade-4.php::modify_tb_lp_order_items`
- **pre-fix cal:** 34.0 (FAIL) → **post-fix cal:** 51.0 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** DB schema migration helper using LP_Database abstraction layer; exception-handled; admin upgrade context.

---

### [admin_path] upgrade_380
- **row_id:** `wordpress-develop::src/wp-admin/includes/upgrade.php::upgrade_380`
- **pre-fix cal:** 58.1 (FAIL) → **post-fix cal:** 94.8 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** WordPress core upgrade routine gated on DB version constant; perfect canonical exemplar of safe upgrade helper. All scores 10.

---

### [admin_path] WP_Upgrader::create_lock
- **row_id:** `wordpress-develop::src/wp-admin/includes/class-wp-upgrader.php::WP_Upgrader::create_lock`
- **pre-fix cal:** 49.5 (FAIL) → **post-fix cal:** 95.2 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** WordPress core distributed lock implementation using prepared statement with INSERT IGNORE; expires and re-acquires correctly. All scores 10. Strong positive exemplar.

---

### [admin_path] db_query
- **row_id:** `woocommerce::plugins/woocommerce/src/Database/Migrations/CustomOrderTable/PostsToOrdersMigrationController.php::db_query`
- **pre-fix cal:** 45.8 (FAIL) → **post-fix cal:** 92.9 (PASS)
- **Verdict:** ❌ auth-missing (bad lift)
- **Rationale:** Passes raw unprepared query string to `$wpdb->query()` with explicit `phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared`. Contextually correct in a migration controller but teaches the model that `phpcs:ignore` + raw queries are acceptable. Bad SQL safety exemplar regardless of context legitimacy. (**Robert Li deciding vote**)

---

### [admin_path] convert_meta_value_longtext
- **row_id:** `learnpress-buddypress::inc/updates/learnpress-upgrade-4.php::convert_meta_value_longtext`
- **pre-fix cal:** 34.0 (FAIL) → **post-fix cal:** 51.0 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** DB column migration using LP_Database abstraction with error checking; internal upgrade context; no direct web exposure.

---

### [rest] LP_Jwt_Users_V1_Controller::register_routes
- **row_id:** `learnpress-buddypress::inc/jwt/rest-api/version1/class-lp-rest-users-v1-controller.php::LP_Jwt_Users_V1_Controller::register_routes`
- **pre-fix cal:** 26.9 (FAIL) → **post-fix cal:** 70.9 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Comprehensive REST route registration with `permission_callback` on all mutative endpoints; `__return_true` only on password reset (correct — rate-limited by WP). Proper WP REST API patterns.

---

### [rest] Front::register_routes
- **row_id:** `seo-by-rank-math::includes/rest/class-front.php::Front::register_routes`
- **pre-fix cal:** 36.7 (FAIL) → **post-fix cal:** 70.9 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** REST route registration delegates auth to `check_api_key` and capability check; correct permission_callback delegation pattern.

---

### [rest] Controller::register_routes
- **row_id:** `constant-contact-woocommerce::src/Rest/PluginVersion/Controller.php::Controller::register_routes`
- **pre-fix cal:** 32.9 (FAIL) → **post-fix cal:** 95.2 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Minimal clean REST controller with `get_item_permissions_check` callback; all scores 10; strong positive exemplar.

---

### [rest] WP_REST_Plugins_Controller::register_routes
- **row_id:** `wordpress-develop::src/wp-includes/rest-api/endpoints/class-wp-rest-plugins-controller.php::WP_REST_Plugins_Controller::register_routes`
- **pre-fix cal:** 42.3 (FAIL) → **post-fix cal:** 76.5 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** WordPress core REST plugins endpoint; full CRUD with correct `permission_callback` methods per operation; all scores 10. Canonical exemplar.

---

### [rest] WC_Stripe_REST_UPE_Flag_Toggle_Controller::register_routes
- **row_id:** `woocommerce-gateway-stripe::includes/admin/class-wc-stripe-rest-upe-flag-toggle-controller.php::WC_Stripe_REST_UPE_Flag_Toggle_Controller::register_routes`
- **pre-fix cal:** 36.7 (FAIL) → **post-fix cal:** 70.7 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Feature flag toggle endpoint with `check_permission` callback on both read and write; correct WooCommerce REST pattern.

---

### [llm_revised] Manual_Synchronization::assign_next_steps
- **row_id:** `woocommerce-square::includes/Sync/Manual_Synchronization.php::Manual_Synchronization::assign_next_steps`
- **pre-fix cal:** 32.0 (FAIL) → **post-fix cal:** 95.2 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Clean business logic function; sets step queue based on system-of-record config; no auth surface; internally invoked. Good exemplar of conditional step orchestration.

---

### [llm_revised] wpcf7_sendinblue_editor_panels
- **row_id:** `contact-form-7::modules/sendinblue/contact-form-properties.php::wpcf7_sendinblue_editor_panels`
- **pre-fix cal:** 38.0 (FAIL) → **post-fix cal:** 49.5 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Admin panel renderer using `esc_html()`, `wpcf7_format_atts()` escaping throughout; correct i18n usage; admin UI context.

---

### [llm_revised] Epsilon_Setting_Repeater::__construct
- **row_id:** `shapely-companion::inc/libraries/epsilon-framework/customizer/settings/class-epsilon-setting-repeater.php::Epsilon_Setting_Repeater::__construct`
- **pre-fix cal:** 38.0 (FAIL) → **post-fix cal:** 97.5 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Customizer setting constructor with parent delegation; hooks sanitization filter; correct lifecycle pattern.

---

### [llm_revised] AgentEvent::setType
- **row_id:** `plugnmeet::plugnmeet/helpers/libs/plugnmeet-sdk-php/src/gen/Livekit/AgentEvent.php::AgentEvent::setType`
- **pre-fix cal:** 24.2 (FAIL) → **post-fix cal:** 88.9 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Vendored/generated protobuf SDK code; harmless typed setter with enum validation via `GPBUtil::checkEnum`; approved as training-worthy on grounds the pattern is correct. (**Robert Li deciding vote**)

---

### [llm_revised] Translations::combine_official_translation_chunks
- **row_id:** `woocommerce::plugins/woocommerce/src/Internal/Admin/Translations.php::Translations::combine_official_translation_chunks`
- **pre-fix cal:** 54.2 (FAIL) → **post-fix cal:** 94.6 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Uses `$wp_filesystem` abstraction correctly; handles empty/malformed chunks gracefully; array_merge pattern is idiomatic WooCommerce admin.

---

### [llm_revised] AMP_YouTube_Embed_Handler::sanitize_raw_embeds
- **row_id:** `amp::includes/embeds/class-amp-youtube-embed-handler.php::AMP_YouTube_Embed_Handler::sanitize_raw_embeds`
- **pre-fix cal:** 34.0 (FAIL) → **post-fix cal:** 95.2 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** XPath-based DOM sanitizer replacing iframes with AMP components; pure transformation; no auth surface. Clean functional code.

---

### [llm_revised] WC_Product_CSV_Importer_Controller::add_error
- **row_id:** `woocommerce::plugins/woocommerce/includes/admin/importers/class-wc-product-csv-importer-controller.php::WC_Product_CSV_Importer_Controller::add_error`
- **pre-fix cal:** 49.1 (FAIL) → **post-fix cal:** 95.0 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Minimal error accumulator; appends structured array to instance property. Trivially correct; good exemplar of simple clean helper.

---

### [llm_revised] AMP_Validation_Manager::init_validate_request
- **row_id:** `amp::includes/validation/class-amp-validation-manager.php::AMP_Validation_Manager::init_validate_request`
- **pre-fix cal:** 22.9 (FAIL) → **post-fix cal:** 80.1 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Auth-gated validation initializer; sends 401 on unauthorized WP_Error; uses `should_validate_response()` check before any action. Correct pattern.

---

### [llm_revised] LP_User_Item::read_meta
- **row_id:** `learnpress-course-review::inc/user-item/class-lp-user-item.php::LP_User_Item::read_meta`
- **pre-fix cal:** 39.1 (FAIL) → **post-fix cal:** 84.7 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Uses `$wpdb->prepare()` for parameterized query; correct SQL safety pattern. SELECT only; no write exposure.

---

### [llm_revised] LP_REST_Admin_Tools_Controller::register_routes
- **row_id:** `learnpress-bbpress::inc/rest-api/v1/admin/class-lp-admin-rest-tools-controller.php::LP_REST_Admin_Tools_Controller::register_routes`
- **pre-fix cal:** 37.9 (FAIL) → **post-fix cal:** 70.9 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Admin REST controller with `check_permission` callback on every route; correct pattern even for ALLMETHODS endpoints.

---

### [other] Tt4b_Catalog_Class::check_and_start_async_action
- **row_id:** `tiktok-for-business::tiktok-for-business/catalog/Tt4b_Catalog_Class.php::Tt4b_Catalog_Class::check_and_start_async_action`
- **pre-fix cal:** 58.1 (FAIL) → **post-fix cal:** 93.2 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** ActionScheduler deduplication helper; idempotent async enqueue with `as_has_scheduled_action` guard; correct background job pattern. SEC-N04 still fires (post=True) but for non-auth reasons.

---

### [other] Tiktokforbusiness::tt_plugin_activate
- **row_id:** `tiktok-for-business::tiktok-for-business/Tiktokforbusiness.php::Tiktokforbusiness::tt_plugin_activate`
- **pre-fix cal:** 54.3 (FAIL) → **post-fix cal:** 94.8 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Plugin activation hook; sets default options; generates unique external business ID. Privileged lifecycle context; not web-reachable.

---

### [other] set_args
- **row_id:** `polldaddy::polldaddy-xml.php::set_args`
- **pre-fix cal:** 49.1 (FAIL) → **post-fix cal:** 50.5 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Object property hydration helper with scalar/restrict guard; correct pattern. SEC-N04 still fires (post=True) but not auth-related for this function's role.

---

### [other] LP_Admin_Editor_Course::sort_sections
- **row_id:** `learnpress-bbpress::inc/admin/editor/class-lp-admin-editor-course.php::LP_Admin_Editor_Course::sort_sections`
- **pre-fix cal:** 31.6 (FAIL) → **post-fix cal:** 49.5 (PASS)
- **Verdict:** ✅ training-worthy
- **Rationale:** Admin AJAX section reorder; uses `wp_unslash` + `json_decode`; delegates auth upstream to AJAX handler. All scores 10.

---

### [other] export_popup_action
- **row_id:** `jupiterx-core::includes/popups/class.php::export_popup_action`
- **pre-fix cal:** 36.4 (FAIL) → **post-fix cal:** 38.0 (PASS)
- **Verdict:** ❌ auth-missing (bad lift)
- **Rationale:** Reads `$_GET['action']` and `$_GET['template_id']` directly via `htmlspecialchars()` (insufficient sanitization for ID); calls `popup_export_data()` and echoes content without nonce or `current_user_can()` check. Web-reachable action hook with data export effect. SEC-N04 still fires post-patch (post=True). (**Robert Li deciding vote**)
