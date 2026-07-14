# Experiment 5 — Are the Gen Training Targets Below the New Base's Own Raw Output Quality?

**Date:** 2026-07-14. **Scope:** read-only analysis, no GPU, no training, no serving. Tests H2 from
`recipe_provenance_audit.md` §6 ("score the 73 replay-stream targets themselves ... and separately
[compare against] the new base's own raw completions ... If replay targets score materially below the
new base's own raw completions, H2 is confirmed as a contributing cause").

## Inputs

| Set | Source | n |
|---|---|---|
| Training gen targets | `data/reasoning_dataset/openai_train.jsonl`, rows tagged `<wp_gen>` | 73 |
| Raw new-base execution outputs | `output/base21/gen03_fresh_new_base_anchor/wp_bench_results_20260714_082330.jsonl`, `type=execution` | 24 |
| Raw new-base knowledge outputs (code-like sample) | same file, `type=knowledge`, answer matching a code/hook regex | 0 (none qualified — see note) |

Note on file naming: the task brief cited `wp_bench_results_20260714_081354.jsonl` for the raw-base
anchor; the actual raw-base-anchor directory (`gen03_fresh_new_base_anchor/`) contains
`wp_bench_results_20260714_082330.jsonl` (`081354` is the *merged* model's file, in
`gen03_full/`, per `gen_regression_forensics.md`'s own data-sources table). Used the correct raw-base
file (`082330`), confirmed by cross-checking `gen_regression_forensics.md` line 14.

All 73 `<wp_gen>` rows carry `metadata.stream == "replay"` — confirming `recipe_provenance_audit.md`'s
claim that the entire gen-target slice is the old-era plain-code replay stream (§2.2), not the
self-distilled-from-new-base CoT stream.

## Method

1. Mechanical feature extraction (regex-based) over both sets: length, `<?php` opening, hook-wiring
   calls (`add_action`/`add_filter`/`add_shortcode`/`register_*`), docblock presence, sanitize/escape
   calls, nonce checks, tab-indent depth (class-method-fragment signature).
2. Same extraction on the raw base's 24 execution-test `code` outputs (full transcripts persisted for
   this type). Attempted the same for a sample of raw's knowledge-type outputs where code was produced;
   none of raw's 320 knowledge answers matched a code/hook pattern (raw answers are short identifiers,
   e.g. `add_role`, `rest_api_init`, `manage_options` — never full snippets), so that bucket is `n=0`
   by construction, itself a data point (see Discussion).
3. Qualitative side-by-side: 5 prompt-matched pairs (training target vs. raw-base output on the same
   *kind* of task — hook registration, filter registration, self-contained helper logic, and one
   counter-example of a target that *does* wire hooks).
4. PHPCS (WordPress ruleset) on both sets as a secondary, coarser style-conformance check.

## 1. Structural feature table

| Feature | Train targets (n=73) | Raw base exec outputs (n=24) |
|---|---|---|
| Mean length (chars) | 779.2 | 1024.7 |
| `<?php` opening present | 8.2% (6/73) | 50.0% (12/24) |
| Hook wiring present (`add_action`/`add_filter`/`add_shortcode`/`register_*`) | 8.2% (6/73) | 45.8% (11/24) |
| Mean hook calls / snippet | 0.30 | 0.88 |
| Docblock present | 8.2% (6/73) | 29.2% (7/24) |
| Sanitization call present | 5.5% (4/73) | 12.5% (3/24) |
| Escaping call present | 5.5% (4/73) | 8.3% (2/24) |
| Nonce check present | 1.4% (1/73) | 0.0% (0/24) |
| Mean double-tab-indented lines (class-method-body signature) | 16.73 | 10.38 |

The training targets are structurally below the raw base's own execution-test outputs on every
WPCS-relevant axis except one (nonce checks are near-zero in both, and slightly *higher* in the training
set by one row — not a meaningful reversal at n=1). Hook wiring is the largest gap: raw base wires hooks
5.6x more often (45.8% vs 8.2%); `<?php` opening is present 6x more often (50% vs 8.2%). The raw-base
`dtab_lines` figure (10.38) is nontrivial too — the raw base sometimes also emits class-method-shaped
fragments — but at roughly 60% of the training set's rate, and it still wires hooks and opens `<?php`
far more often when it does.

This directly corroborates `gen_regression_forensics.md`'s Q5 table (which measured the *merged* model's
execution outputs at 4/24 hooks, 0/24 `<?php`, vs raw's 11/24 / 12/24): the merged model's degraded
shape is not a random drift, it is a reproduction of the training targets' own shape, measured
independently here directly from the training file rather than inferred from the merged model's
behavior.

## 2. Qualitative side-by-side (5 pairs, matched by task kind)

**Pair 1 — hook registration.** Prompt: *"Handle the `woocommerce_update_order` hook."* (train
target, `<wp_gen>` row #24) vs. raw base's `e-hooks-001` output (add-a-column-to-admin-table task).

- Train target: `function handle_woocommerce_update_order( $order_id, $order ): void { ... }` — a bare
  method, no `add_action('woocommerce_update_order', ...)` anywhere in the target despite the prompt
  naming the hook explicitly.
- Raw base: full `<?php` file, two functions, each immediately followed by its own
  `add_filter('manage_posts_columns', 'add_featured_column_header')` /
  wiring call — self-contained and self-registering.

**Pair 2 — filter registration.** Prompt: *"Filter sold individually quantity for add to cart
requests."* (train target #32) vs. raw base's `e-font-library-disable-001` (disable a block-editor
feature via filter).

- Train target: `function filter_request_data( $request ) { ... apply_filters('woocommerce_add_cart_item_data', ...) ... }`
  — the function *calls* `apply_filters` internally but is never itself hooked via `add_filter`; despite
  the prompt title being "Filter ... for add to cart requests," the target never registers itself as a
  filter callback.
- Raw base: `add_filter( 'block_editor_settings_all', function ( $editor_settings ) { ... } );` — a
  complete, self-registering one-liner.

**Pair 3 — undefined/assumed-external helpers.** Prompt: *"Get the next batch of items to process..."*
(train target #8) vs. raw base's `e-cache-false-hit-001` (a flag-with-cache-miss-fallback task).

- Train target: `get_next_batch_to_process()` branches into `$this->get_next_batch_to_process_hpos($size)`,
  `$this->get_next_batch_to_process_cpt($size)`, `$this->throw_doing_it_wrong(...)` — three method calls
  on `$this` that are never defined in the target, assumed to exist in a surrounding class the training
  row doesn't include.
- Raw base: `wpbp_cached_flag()` inlines the entire `wp_cache_get`/`wp_cache_set` logic in the same
  function body, calling no undefined helpers. This is the exact failure mode
  `gen_regression_forensics.md` documented in the *merged* model's `e-cache-false-hit-001` regression
  (`wpbp_compute_flag()` invented, never defined) — traced here to the training data's own habit of
  referencing not-locally-defined helpers.

**Pair 4 — explicit copy-paste provenance.** Train target #28, prompt: *"Register the Container's shape
dividers controls. TODO: Copied from `section.php`."* vs. raw base's `e-shortcode-001` (shortcode task).

- Train target: the prompt *itself* is a literal `TODO: Copied from` comment lifted from the source
  repository — direct textual evidence that this row is an extracted fragment, not a self-contained
  generation task, sourced from Elementor's codebase where the referenced logic legitimately lives
  elsewhere.
- Raw base: `add_shortcode('greeting', 'greeting_shortcode')` plus the callback, `esc_html()`-escaped,
  fully self-contained in ~6 lines — everything the prompt asked for, nothing assumed to live elsewhere.

**Pair 5 — counter-example (one of the training set's few "good" rows).** Train target #6, prompt:
*"Write a WordPress function that construct[s], registering WordPress action hooks, in the Module
class"* vs. raw base's `e-interactivity-router-replace-001`.

- Train target: a `__construct()` method that *does* call `add_action(...)` five times — one of only
  6/73 (8.2%) targets with any hook wiring. Even here, though, it's registering hooks *for other
  methods*, inside a constructor that itself must be invoked by a class-instantiation pattern not shown
  in the row — still not a standalone, directly-runnable unit.
- Raw base: `wpbp_router_link($url, $text)` is a complete, standalone, directly-callable function with
  no external class/constructor dependency — the shape wp-bench's execution harness actually requires
  (a callable unit it can invoke directly in a sandboxed PHP process, per
  `gen_regression_forensics.md`'s harness description).

Across all 5 pairs, the pattern is consistent and one-directional: the training targets are extracted
class-method fragments assuming an unshown surrounding file, while the raw base's own matched-task
outputs are self-contained, self-registering, directly runnable units — precisely the property
wp-bench's execution tests score.

## 3. PHPCS (WordPress standard) — secondary check, mixed and confounded signal

phpcs 3.x with the `WordPress` ruleset was run over both sets (each snippet wrapped in `<?php` if not
already present; run in isolation, without the surrounding class/file context the fragments assume,
which itself penalizes both sets for "missing file docblock" etc. regardless of target quality).

| | Train targets (n=73) | Raw exec outputs (n=24) |
|---|---|---|
| Mean errors/file | 14.49 | 33.92 |
| Mean warnings/file | 1.55 | 1.46 |
| Files with zero errors | 0.0% | 0.0% |
| Security-prefixed violations (nonce/escape/alt-functions) | 23 total (nonce 8, escape-output 7, other 8) | 0 total |

Raw base's *total* PHPCS error count is higher, but this is dominated by one mechanical rule:
`Generic.WhiteSpace.DisallowSpaceIndent.SpacesUsed` fires 488/814 times (60%) on raw — the raw base
indents with spaces, WPCS requires tabs, and raw's outputs are 2.1x longer on average, so this single
whitespace-character preference alone accounts for most of raw's higher raw error count. The training
targets' top violation, `Generic.WhiteSpace.ScopeIndent.IncorrectExact` (268 hits), is the mirror
artifact of extraction: the fragments are indented at their *original* nested depth (2 tabs, inside a
class/method) but phpcs, seeing them as top-level code, expects 0-1 tabs — an artifact of being lifted
out of a class, not a quality signal per se, though it corroborates the "extracted fragment" reading
independently of the regex features above.

The one PHPCS signal that *does* point the same direction as the structural/qualitative findings:
security-relevant sniffs (`WordPress.Security.NonceVerification.Recommended`,
`WordPress.Security.EscapeOutput.OutputNotEscaped`, and related) fire 23 times across the 73 training
targets and **zero** times across the 24 raw-base execution outputs. This is a small-n, secondary
signal (many training rows plausibly don't need a nonce/escape in their narrow fragment scope) but it
does not contradict the primary finding, and PHPCS-as-run-here is not a clean apples-to-apples
completeness check (it's a style linter, not an execution/wiring checker, which is why it doesn't
directly detect the "missing `add_action`" defect the structural regex and qualitative pairs isolate).

**Verdict on PHPCS specifically**: inconclusive/confounded as a standalone signal — do not use raw
error-count as the tiebreaker; the structural feature table (§1) and qualitative pairs (§2) are the
load-bearing evidence.

## 4. Verdict

**Regression-to-teacher for the gen stream: CONFIRMED.**

The 73 `<wp_gen>` replay-stream training targets are structurally and qualitatively below the raw new
base's own output quality on the WPCS-relevant axes that drive wp-bench's execution scoring:

- Hook wiring: 8.2% of targets vs 45.8% of raw base's matched outputs (5.6x gap).
- `<?php` self-containment: 8.2% vs 50.0% (6.1x gap).
- Docblocks: 8.2% vs 29.2% (3.6x gap).
- 5/5 qualitative pairs show the same one-directional pattern: targets are extracted, unwired,
  sometimes-undefined-helper-dependent class-method fragments; raw base's matched-task outputs are
  self-contained, self-registering, directly runnable units — including one training row (`TODO: Copied
  from section.php`) that is *literally labeled* as a copy-paste extraction by its own prompt text.

This is independent, direct confirmation of `gen_regression_forensics.md`'s Q5 finding (which inferred
the same shape indirectly, from the *merged* model's behavior) and closes H2 from
`recipe_provenance_audit.md` §6: the replay stream is not just narrow in proportion (13% of the mix,
per H1/the task-mix finding) — its content is *also* independently below the new base's own
demonstrated raw capability on the same class of task. H1 (task-mix dilution) and H2 (sub-base-quality
targets) are not mutually exclusive; both are now confirmed contributors, with H1 remaining the larger
lever (86/13 mix ratio dominates the loss signal) and H2 explaining why the 13% that does train exerts
a *downward*, not merely diluting, pull.

**Target-quality bar a rebuilt mix (Experiment 4) must clear:** any replacement/rebuilt gen-training
corpus should require, at minimum, that generated/curated targets for hook-adjacent tasks (>50% of
wp-bench's execution suite by observed prompt intent) (a) open with `<?php` when the task implies a
standalone file, (b) include the relevant `add_action`/`add_filter`/`add_shortcode`/`register_*` call
in the same target when the prompt names or implies a hook, and (c) do not reference `$this->method()`
or free-function calls that are not either defined in the same target or unambiguously a WordPress core
API. Self-distilling new targets from the raw new base's own outputs (as Experiment 4 in
`DIAGNOSTIC_SYNTHESIS.md` proposes, Claude-gated) is directly supported by this experiment: the raw
base already clears this bar 45.8%/50.0%/29.2% of the time unprompted, which is materially higher than
the 8.2% the current replay stream achieves on every one of those same axes.
