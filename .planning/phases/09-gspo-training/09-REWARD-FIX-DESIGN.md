# Reward fixes — design + verification spec (2026-06-26)

Two independent reward defects found during the post-8.1 RL rerun. This is the design,
test plan, and end-to-end verification process for both. Sampler+KL fixes (ff0872e) are
separate and stay.

---

## FIX 1 — Judge reward: add correctness pressure (close the isolation hack)

### Defect (J.7/J.8)
`_fix_score_from_completion(completion)` scores the corrected code IN ISOLATION via
`_extract_verifiable_signals(corrected).overall/100`. Trivial / unrelated / faithfully-
reproduced clean PHP all score ~1.0. There is ZERO gradient distinguishing a fix that
resolves the flagged bug from one that does not. Verified: `<?php echo 'hi';` → 1.0.

### Available data at the scoring site (`collect_rollouts`, rl_rollouts.py:794)
`judge_by_gid[gid]["messages"]` (the originating judge item) is in scope. The ORIGINAL
function under review is the ```php block in the user turn — extract via `extract_php_code`
on the user content. (The pool item has no separate `critique_text`/`php_code` field.)

### Design — graded score with identity + improvement gates
New signature (back-compatible):
`_fix_score_from_completion(completion_text, original_code: str | None = None)`

Tiers (preserve frac_mid > 0; keep 0.0 / 0.25 low/partial extremes):
1. `corrected` empty / no block            → **0.0**
2. non-empty but unparseable PHP           → **0.25**
3. parseable PHP, `original_code is None`  → legacy `rubric.overall/100` (back-compat only)
4. parseable PHP, `original_code` provided → **correctness-pressured**:
   a. **Identity gate** — the corrected block must be a credible edit of the SAME function,
      not a replacement. Require BOTH:
        - same primary function/method name as `original_code`
          (`_primary_php_function_name`), AND
        - anti-gutting floor: `len(corrected.strip()) >= 0.5 * len(original.strip())`
          (a 50-line function "fixed" to `return 1;` fails this).
      Fail → **0.25** (parseable, but not a credible fix of this function — kills the
      trivial-echo, unrelated-clean-code, and gut-the-body hacks).
   b. **Improvement delta** — `rubric_corr = signals(corrected)`,
      `rubric_orig = signals(original)`; `improvement = max(0, corr.overall - orig.overall)/100`.
      - faithful reproduction (no real fix) → improvement ≈ 0
      - genuine fix of the flagged dim     → improvement > 0
   c. **Score** = `0.5 + 0.5 * min(1.0, improvement / IMPROVE_NORM)`, `IMPROVE_NORM = 0.30`
      (a ≥30-pt rubric gain = full credit). Range [0.5, 1.0] for identity-preserved code;
      reproduction → 0.5; real fix → up to 1.0.

Net hack closure: echo/unrelated → identity fail → 0.25; reproduction → delta 0 → 0.5
(not 1.0); gutted body → length floor → 0.25; real fix → up to 1.0.

### Plumbing
`collect_rollouts` extracts `orig = extract_php_code(user_content_of(judge_by_gid[gid]))`
once per judge group and passes it to `_fix_score_from_completion(c.completion, orig)`.

### New helpers (rl_rollouts.py)
- `_primary_php_function_name(code) -> str | None` — first `function NAME(` / method name.
- `_judge_original_code(item) -> str` — extract the original ```php block from the item's
  user message (pre-augmentation; strip the appended fix-instruction).

---

## FIX 2 — Gen reward: credit valid template (Elementor/Underscore) code

### Defect (ticket 09-TICKET-gen-reward-dead-gradient.md)
Elementor `content_template()` completions mix `<?php ?>` with Underscore/JS markup
(`<# #>`, `<%- %>`, `<%= %>`, `{{ }}`, `{{{ }}}`). `extract_php_code` + `php -l` rejects
them → scalar 0 → constant all-zero groups → dropped → zero gen gradient. The 20% codegen
replay is training-inert (RLEV-01 no-regression risk).

### Design — template-aware parseability
New helper `_is_valid_wp_php(code) -> bool` used by the gen non-code guard:
1. If plain `_is_parseable_php(code)` → True (unchanged behavior, fast path).
2. Else if `code` contains template markers (`<#`, `<%`, `{{`) AND a `<?php`:
   - neutralize template directives → replace `<#...#>`, `<%-...%>`, `<%=...%>`,
     `<%...%>`, `{{{...}}}`, `{{...}}` with a benign PHP-safe placeholder (e.g. `''`
     inside expression context / removed in statement context), then `php -l` the result.
   - True iff the neutralized scaffold lints.
3. Else False.

Gen path swaps the `_is_parseable_php(php)` zeroing guard for `_is_valid_wp_php(php)` so
template completions that are well-formed earn their rubric score instead of 0.

`compute_group_rewards` stays UNMODIFIED (D-09-05) — the change is only the boundary guard.

---

## UNIT TESTS (must accompany the fixes)

### Fix 1 (judge correctness pressure) — `tests/test_reward_fix1_judge.py`
- `trivial_echo_scores_low`: `<?php echo 'hi';` vs a real buggy `original` → identity fail → 0.25.
- `unrelated_clean_fn_scores_low`: clean unrelated function, different name → 0.25.
- `gutted_body_scores_low`: same name, body replaced with `return 1;`, original 40+ lines → 0.25.
- `faithful_reproduction_scores_mid`: corrected == original (bug intact) → ~0.5 (identity ok, delta 0).
- `genuine_fix_scores_high`: original has `$_GET` XSS (rubric ~70), corrected escapes it
  (rubric ↑ ≥ +15) → > 0.5, scaling toward 1.0.
- `backcompat_no_original`: `original_code=None` → legacy rubric.overall/100 (existing tests stay green).
- `prose_only` → 0.25 ; `empty` → 0.0 (tiers preserved).
- `frac_mid_preserved`: a mixed batch yields ≥3 distinct score values (SC1 gate parity).

### Fix 2 (gen template) — `tests/test_reward_fix2_gen.py`
- `standalone_php_still_valid`: `<?php function f(){}` → True (no regression).
- `elementor_template_valid`: a real `content_template()` with `<# #>`/`<%- %>` + `<?php` → True.
- `garbage_not_valid`: prose / `<?php func tion(` broken → False.
- `template_with_broken_php_invalid`: template markers but malformed PHP scaffold → False.
- `gen_reward_nonzero_on_template`: end-to-end via the gen guard — a valid template earns
  a non-zero scalar (was 0).

### Regression
Full `tests/test_rl_train.py tests/test_rl_train_integration.py` + any reward_pipeline /
rl_rollouts tests stay green (expect 1 pre-existing unrelated `test_lora_config` failure).

---

## END-TO-END VERIFICATION PROCESS (after unit tests pass, before any RL re-spend)

All $0 (local judges) / cheap Tinker sampling. Reuses this session's harnesses.

### V1 — Reward unit behavior (offline, deterministic, no sampling)
`pytest tests/test_reward_fix1_judge.py tests/test_reward_fix2_gen.py -q` → all green.
Plus a direct hack-probe script asserting: trivial echo / unrelated / gutted ALL < 0.3,
genuine fix > 0.6, against ≥5 hand-built (original, corrected) pairs.

### V2 — Controlled judge-axis eval, hack-resistance (sampling, low variance)
Extend `_check_judge_fixcorr.py` to score with the NEW correctness-pressured reward and to
ALSO emit, per completion, whether identity-gate passed. Run warm-start vs (a fresh short
RL run's) step-50, PLUS an adversarial synthetic policy that emits trivial echoes:
- the echo policy must now score ~0.25 (was ~1.0) — proves the hack is closed live.
- a real-fix-capable policy should out-score reproduction.

### V3 — Gen-axis liveness
Extend the gen check (`_check_step50_vs_warmstart.py` gen path) with `_is_valid_wp_php`:
the Elementor-template completions that scored 0/60 must now earn non-zero reward, and
`frac_groups_all_zero` in a fresh short run must drop below the gen fraction (0.375).

### V4 — Short RL smoke (cheap Tinker, ~10–15 steps, killable)
Relaunch ONE seed with both reward fixes + the committed sampler/KL fixes. Gates:
- plumbing: sampler-id changes each step (already wired), reward rows carry the new
  fix-correctness distribution (≥3 tiers, identity-gate stats logged).
- `frac_groups_all_zero` < 0.375 (gen now contributes) — the J.4 pin is gone.
- judge fix_correctness no longer trivially saturates at 1.0 (identity+delta in effect).
- NO HARD halt, KL/entropy live (regression check on ff0872e).

### V5 — Anti-hack acceptance gate (RLEV-02 alignment)
On a step-50 checkpoint from V4, run V2's controlled eval with the echo-adversary. ACCEPT
only if: echo-adversary ≤ 0.3, reproduction ≤ 0.55, genuine-fix policy > reproduction by a
margin that clears its bootstrap CI. Otherwise the reward still has a hack surface → iterate.

### V6 — Codegen no-regression pre-check (RLEV-01 alignment)
Confirm the gen fix did not corrupt standalone-PHP scoring: re-run V1 gen tests + a wp-bench
spot subset; standalone codegen scores must match the pre-fix baseline (template credit must
be ADDITIVE, not a relaxation that passes broken PHP).

---

## Rollout order
1. Implement Fix 1 + Fix 2 + unit tests (this doc's specs).
2. V1 green → commit.
3. V2/V3 offline-sampling green → commit harness extensions.
4. V4 short smoke → if gates pass, V5/V6 acceptance.
5. Only then authorize a full 500-step re-spend (Dr. Li gate).
