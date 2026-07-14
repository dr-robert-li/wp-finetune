# Phase 21 Diagnostic: Why Reasoning-Mix SFT Regressed Codegen on Qwen3.6-35B-A3B

**Date**: 2026-07-14
**Scope**: Read-only forensic analysis. No models touched, no GPU used.
**Headline**: merged gen model (v4, ep3) scores **0.372** overall vs raw new base **0.4897** — a
regression, whereas the identical recipe on the OLD base (Qwen3-30B-A3B) IMPROVED it
(0.4033 raw → 0.4365/0.4484 SFT).

## Data sources

| Arm | File | Overall | Knowledge | Correctness |
|---|---|---|---|---|
| NEW base, merged gen-v4 (ep3) | `output/base21/gen03_full/wp_bench_results_20260714_081354.json` | 0.372 | 0.5625 | 0.2292 |
| NEW base, raw (Qwen3.6-35B-A3B) | `output/base21/gen03_fresh_new_base_anchor/wp_bench_results_20260714_082330.json` | 0.4897 | 0.5594 | 0.4375 |
| OLD base, merged v1.2 (fresh rerun) | `output/bench17/full_gate_rerun/wp_bench_results_20260711_050328.json` | 0.4365 | 0.4906 | 0.3958 |
| OLD base, raw (Qwen3-30B-A3B) | `output/bench17/base_anchor/wp_bench_results_20260712_190952.json` | 0.4033 | 0.4688 | 0.3542 |

Both arms of each pair ran the identical 344-test `wp-core-v1` suite, same harness
(`scripts/run_eval_reasoning.py::_run_wpbench`), same sampling (`temp=0.0`, `max_tokens=2048`,
`concurrency=4`, `seed=1337`), and the same `enable_thinking=False` injection mechanism
(`scripts/_wpbench_pth/usercustomize.py`) applied symmetrically to both serves. `overall` is a
0.3/0.4/0.3 (knowledge/correctness/quality) weighted mean, quality is `null` (unscored) in the
Phase 21 runs, so `overall = (0.3·knowledge + 0.4·correctness) / 0.7` — **the entire regression
is arithmetically explained by the `correctness` (execution) component**, which is only **24
tests**.

---

## Q1 — Paired diff (344 pairs)

`type=knowledge` (n=320): mean Δ = **+0.0031** (net neutral). 10 tests regressed >0.01, 11
improved >0.01, 299 flat — this is ordinary MC noise, not a directional collapse.

`type=execution` (n=24): mean Δ = **−0.2083**. 5 tests regressed >0.01, 0 improved, 19 flat.
Merged has 18/24 zero-scores vs raw's 13/24; merged has 5/24 perfect scores vs raw's 10/24 — a
clean 5-test swing from pass→fail, no partial-credit middle ground (execution scoring is
per-assertion pass/fail, inherently bimodal in both arms).

Top regressions (all Δ = −1.0, all execution or knowledge MC flips):
`k-rest-005`, `e-font-library-disable-001`, `k-queries-024`, `e-cache-false-hit-001`,
`k-dataviews-011`, `k-queries-025`, `k-blockeditor-014`, `k-gotchas-010`, `k-caching-007`,
`e-hooks-001`, `e-shortcode-001`, `k-wpblocks-005`, `k-rest-013`, `k-blockapi-007`,
`e-interactivity-router-replace-001` — 10 knowledge MC flips (offset almost exactly by 11
improvements elsewhere) plus **5 execution flips, which is where the real signal is**:
`e-font-library-disable-001`, `e-cache-false-hit-001`, `e-hooks-001`, `e-shortcode-001`,
`e-interactivity-router-replace-001`.

**Verdict on Q1**: this is not a broad quality collapse across 344 tests. It is a narrow,
concentrated collapse in the 24-test execution/correctness slice, amplified to a large
headline delta because `correctness` carries 0.4/0.7 = 57% of the active weight on only 24
samples (each test ≈ 2.4 overall-points).

---

## Q2 — Transcript forensics on the execution regressions

The harness (`wp-bench/python/wp_bench/core.py::_run_execution_tests`,
`wp_bench/models.py::ModelInterface.generate`) does **not** persist raw completions, thinking
traces, or prompts — only the post-processed `code` (after `strip_code_fences` +, in the
monkeypatch, a `<think>...</think>` regex strip) survives to the results JSON/JSONL. So
transcript forensics had to work from the extracted `code` field plus assertion errors.

Full comparison of all 24 execution pairs (see script output, `code` field):

| Signal | MERGED | RAW |
|---|---|---|
| mean code length | 487 chars | 1025 chars (2.1x) |
| tests with `add_action`/`add_filter`/`add_shortcode` present | **4/24** | **11/24** |
| tests opening with `<?php` | **0/24** | **12/24** |
| mean count of double-tab-indented lines (class-method-body signature) | 12.17 | 0.38 |
| "Call to undefined function/method" runtime errors | 5 | 2 |

On every one of the 5 flipped tests, the pattern is identical and mechanical:

- **`e-font-library-disable-001`**: merged emits a bare `function disable_font_library_ui(...)
  {...}` — no `add_filter(...)` wiring it up. Raw wraps the same logic in
  `add_filter('block_editor_settings_all', function(...) {...});`.
- **`e-hooks-001`**, **`e-shortcode-001`**: same pattern — merged defines the callback(s), never
  registers them (`add_action`/`add_shortcode` missing entirely); raw always registers.
- **`e-cache-false-hit-001`**: merged calls an invented helper `wpbp_compute_flag()` that is
  never defined anywhere in the response (`Call to undefined function`); raw inlines the logic.
- **`e-interactivity-router-replace-001`**: merged calls an invented `wp_is_internal_url()`
  (doesn't exist in WP core); raw computes the same check inline with `wp_parse_url()`.

This is not truncation, not a formatting/markdown issue, and not a thinking-mode deliberation
gap — merged code is **coherent, syntactically valid PHP**, just structurally incomplete: it
looks like a **method body lifted out of a class**, assuming the wiring/registration and any
helper methods exist elsewhere in a surrounding file that isn't there. Confirmed against the
raw training data (see Q5) — this is exactly the shape of the SFT targets.

**Thinking-mode leak (real, but not the driver of the execution regression)**: 10/320 (3.1%)
of merged's *knowledge* answers contain a literal, **unterminated** `<think>` tag (e.g.
`k-wpblocks-003`: `'<think>\nThe user is asking for the value of the BLOCK_ICON_DEFAULT
constant...'`), vs **0/320** for raw. This is exactly the failure mode the monkeypatch's own
docstring warns about ("with thinking on... it writes the answer INSIDE an unterminated think
block") — the defensive `<think>...</think>` regex strip requires a *closing* tag and does
nothing for these. So `enable_thinking=False` is measurably **less reliable post-merge** than
on the raw base, but only accounts for 10 of 320 knowledge tests (which net out to noise in the
headline number) and **none** of the 5 execution flips — none of those 5 show a `<think>`
leak in the code field.

**Output-length paradox**: overall mean output length across all 344 tests is 369.5 chars for
merged vs 72.7 for raw — merged looks *longer* despite terser code, because a subset of its
knowledge answers are bloated with judge-style rubric prose (see Q3) that swamps the shorter
execution code in the average.

---

## Q3 — Parse/harness artifacts vs genuine failures

- **Execution**: 0/24 empty-code responses in either arm, 0/24 `result: null` (grader crash),
  2/24 PHP parse/syntax errors in **both** arms (identical rate — not a regression driver).
  All 18 merged zero-scores are genuine assertion failures on parseable, syntactically valid
  code (missing hooks / undefined-function calls), not harness-parser artifacts. This confirms
  the regression is real, not a scoring-pipeline bug — echoing the note that the old
  "0.2275 k=64 collapse was real."
- **Knowledge**: 0/320 empty answers in either arm. But a **format-confusion artifact is real
  and quantifiable**: 40/320 (12.5%) of merged's knowledge answers are >3 chars (i.e., not a
  clean single-letter answer) — including literal judge-rubric text bleeding into a
  multiple-choice slot (`'WPCS Compliance: score 10/10 — The code adheres to WordPress Coding
  Standards...'`) and bare code snippets (`'function register_post_type()'`) instead of a
  letter — vs **8/320 (2.5%)** for raw, a **5x** increase. All 40 verbose merged answers score
  0 (none happen to start with the correct letter). This IS a genuine model-behavior artifact
  (task-format confusion from the reasoning-mix's judge-dominant training data — see Q5), not a
  grader bug, but because knowledge MC is close to a coin flip either way, it doesn't move the
  knowledge mean much — it just redistributes which questions merged gets wrong.

**Verdict on Q3**: no parser/extraction-pipeline artifacts. The zero-scores are earned. The
one real "artifact" worth naming is the model's own task-format confusion (judge-style prose on
knowledge prompts, unterminated `<think>` leaks), which is a genuine SFT-side regression, not a
scoring bug.

---

## Q4 — Uniform vs bimodal degradation

**Bimodal, and narrowly concentrated.** Knowledge (320 tests, 93% of the suite) is flat (mean Δ
+0.003). Execution (24 tests, 7% of the suite, but 57% of the active scoring weight) is where
100% of the effective damage lives, and even within execution it's not diffuse: 5 specific
tests flip from full pass to full zero; the other 19 execution tests are unchanged (13 already
zero in both arms — genuinely hard tests both models fail — and 6 unchanged passes/partials).
So: the 0.372-vs-0.4897 gap is **not** "the model got generally worse at everything" — it's "5
of 24 execution tests, plus a wash of MC noise on 320 knowledge tests," where those 5 tests
share one mechanical signature (missing hook registration / invented helper functions).

---

## Q5 — Old base (helped) vs new base (hurt): what's actually different

Ran the identical code-signature analysis on the OLD base's execution pairs
(`output/bench17/full_gate_rerun/...json` vs `output/bench17/base_anchor/...json`, same 24
test IDs):

| Signal | OLD SFT (v1.2) | OLD RAW | NEW SFT (merged v4) | NEW RAW |
|---|---|---|---|---|
| hooks present (`add_action`/`filter`/`shortcode`) | 10/24 | 9/24 | **4/24** | 11/24 |
| `<?php` opening tag | 3/24 | 3/24 | 0/24 | 12/24 |
| mean code length | 517 | 491 | 487 | 1025 |
| double-tab-indent lines (class-method signature) | 12.92 | 0.00 | 12.17 | 0.38 |
| mean correctness | 0.396 | 0.354 | 0.229 | 0.4375 |

Two things jump out:

1. **Both** old-SFT and new-SFT models learned the same tab-nested, class-method-body writing
   style (dtab_lines ≈ 13 in both, vs ≈0 for both raw bases) — this confirms the SFT data itself
   (not the base) is the source of that stylistic tell, and it is present in **both** eras.
2. **Only** the new SFT model *lost the hook-registration call specifically*. Old SFT kept
   registering hooks at essentially the same rate as its own raw baseline (10/24 vs 9/24, no
   degradation) — e.g. `e-hooks-001`/`e-shortcode-001`/`e-font-library-disable-001` all score
   1.0 in *both* old arms, with old-SFT still emitting `add_filter(...)`/`add_shortcode(...)`
   right after the function body, just at the same nested tab-indent. New SFT dropped hook
   registration to well below its own base (4/24 vs 11/24) — a **7-test relative loss** on a
   24-test slice, which is the whole story.

**Traced to the training data itself**: `data/reasoning_dataset/openai_train.jsonl` (the file
`gen02_run.json` trained on, `train_path`) is 563 examples: **482 `<wp_judge>` (86%)**, 73
`<wp_gen>` (13%), 8 other. Of the 73 `<wp_gen>` targets, only **6 (8%)** contain any
`add_action`/`add_filter`/`add_shortcode` call — the other 92% are bare `function foo() {...}`
snippets extracted from what look like real plugin/theme class methods (Elementor widget
`content_template()`, a WooCommerce-style `is_free_plan()`, etc.), where in the source
repository the hook wiring lives elsewhere (a parent class, a bootstrap file) — never adjacent
to the method. This is precisely the shape of merged's execution-test failures. This same
`data/reasoning_dataset` lineage is documented as feeding the old-base v2/v3 runs too
(`gen02_run.json`: "3 epochs (matching wp-reasoning-v2/v3 manifests)"), so the training
*data* is not the variable — the base model's *sensitivity* to it is.

**Base-sensitivity reading**: the new base's raw prior for these execution tests is
substantially stronger and more "complete-code" native than the old base's (raw hooks 11/24 vs
9/24, raw correctness 0.4375 vs 0.3542, raw code 2x longer). A 3-epoch, MoE-only LoRA
(rank 32, `train_mlp=True`, router frozen) fine-tune on a code-gen slice that is only 13% of the
mix, 92% of which demonstrates "bare snippet, no wiring," pulled the *stronger* new base's
generation behavior down toward that narrow pattern much more than it pulled the *weaker* old
base (whose raw prior was already closer to bare-snippet quality, so the shift cost it little
to nothing on this axis, while the old base's other knowledge/correctness gains from the
judge-heavy reasoning content dominated and delivered a net +4.5pp).

**Does the evidence support "SFT strips thinking the new base needs"?** Only weakly, and only
as a minor contributor: `enable_thinking=False` was applied identically to both new-base arms;
raw's higher score does not come from longer thinking-mode deliberation (raw was also
thinking-disabled and produced clean output; there's no evidence raw silently ignored the flag
— 0/24 execution and 0/320 knowledge `<think>` leaks in raw). The only thinking-related signal
found is that merge/SFT made the `enable_thinking=False` mechanism *less reliable* post-merge
(10/320 knowledge leaks vs 0), but that touches 0 of the 5 execution flips that drive the
headline number. **Rejected as primary cause.**

**Does the evidence support "SFT data teaches an output style the harness scores lower"?**
**Yes, strongly** — this is the dominant, directly-observed mechanism (Q1/Q2/Q5 above): 92% of
the `<wp_gen>` training targets are bare, non-self-registering class-method snippets, and
merged's execution failures reproduce that exact shape test-for-test.

**Does the evidence support "ep3 overtraining"?** Plausible and testable but **not yet
confirmed** — no ep1/ep2 wp-bench data exists in this artifact set to check against. The
per-epoch sampler checkpoints (`wp-gen-v4-ep1`, `-ep2`, `-ep3`) are all preserved
(`output/base21/gen02_run.json: full.per_epoch_sampler_paths`), so this is a real, cheap,
currently-open question rather than a ruled-out one.

---

## Q6 — Ranked verdict

**1. PRIMARY (confirmed, high confidence): reasoning-mix data composition mismatch, amplified
by base-sensitivity, concentrated on a 24-test slice.**
The `<wp_gen>` slice of the SFT data (13% of the mix, 92% "bare snippet, no
hook-registration") teaches an output style that wp-bench's self-contained execution tests
specifically penalize. The new base is more plastic to this narrow slice than the old base was,
losing hook-registration completeness (4/24 vs its own raw 11/24) while the old base didn't
(10/24 vs its own raw 9/24). Because `correctness` carries 57% of active scoring weight on only
24 tests, a 7-test relative loss on this narrow slice single-handedly explains the entire
0.4897→0.372 headline drop; knowledge (93% of the suite) is flat.
*Cheapest confirming experiment*: re-render the 73 `<wp_gen>` training examples and count how
many require the model to also emit a top-level, self-registering, standalone function (vs. a
class-method fragment) — if wp-bench's execution prompts skew standalone/self-registering while
training targets skew class-fragment, the train/eval prompt-format mismatch is the direct,
already-available proof (no GPU/serving needed, pure data audit of files already on disk).

**2. SECONDARY, confirmed contributor to noise but not the headline number: judge-task format
bleed-through.**
86% of the training mix is `<wp_judge>` (WPCS-rubric scoring prose). Merged answers plain
knowledge MC questions with judge-rubric text or code snippets 5x more often than raw
(40/320 = 12.5% vs 8/320 = 2.5%), but since knowledge MC is near-coin-flip either way, this
nets out close to zero in the headline knowledge mean (it's real behavioral evidence of
task-confusion, just not what moved the score).
*Cheapest confirming experiment*: none needed beyond what's here — already directly observed
in the persisted `answer` field of the existing results JSON; no rerun required.

**3. TERTIARY, real but minor: `enable_thinking=False` is less reliable post-merge.**
10/320 (3.1%) knowledge answers on merged contain an unterminated `<think>` leak (vs 0/320 raw)
that survives the defensive regex strip. Real defect, but touches 0 of the 5 execution flips
and only ~3% of knowledge tests — cannot explain the correctness collapse.
*Cheapest confirming experiment*: bench the merged model with `enable_thinking` explicitly
re-enabled AND explicitly re-disabled via two small (n=24, execution-only) reruns to see if the
leak rate or the hook-registration rate changes — if hook-registration doesn't move, this
confirms it's cosmetic/orthogonal to the main mechanism.

**4. OPEN, plausible but unconfirmed: ep3 overtraining on the narrow gen slice.**
No ep1/ep2 wp-bench data exists yet. Given #1's mechanism (a small, narrow-style slice pulling
behavior toward itself), more epochs of exposure to that 13%-of-mix slice could plausibly
deepen the drift — but this hasn't been measured.
*Cheapest confirming experiment*: merge+bench the already-preserved `wp-gen-v4-ep1` sampler
checkpoint (`tinker://05590ab9-4d38-57fc-bb2f-53a445891caa:train:0/sampler_weights/wp-gen-v4-ep1`)
against the same 24 execution tests. If hook-registration rate at ep1 is close to raw's 11/24
and degrades progressively through ep2→ep3, overtraining is confirmed as an amplifier; if ep1
already shows the drop, it's the data composition (#1) acting from step one, not epoch count.

**Process gap worth flagging**: the training-time terse-gate (`output/base21/gen02_fs_gate.json`,
`gen02_run.json:full.terse_gate`) explicitly scopes its canonical pass/fail metric to `cot+ctf`
streams only and *excludes* the `replay` stream (raw `<wp_gen>` code-gen rows) — the one stream
whose format defect (bare snippets, no registration) is exactly what regressed wp-bench. The
in-driver full-val arm did show elevated terse-rate concentrated in that excluded stream
(20/141 ≈ 14.2%, dismissed as a "known measurement artifact, not format collapse" per that
file's own disposition note) — that dismissal was correct for its own metric ([/REASONING]-tag
presence) but the gate had no way to detect "does the code self-register," so it could not have
caught this regression even in principle.
