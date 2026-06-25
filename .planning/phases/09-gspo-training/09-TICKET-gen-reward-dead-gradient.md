# Ticket: wp_gen reward path contributes zero gradient (codegen-protection replay non-functional)

**Filed:** 2026-06-26 · **Phase:** 09 (surfaced during post-8.1 RL rerun) · **Severity:** medium (silent; an RLEV-01 no-regression risk, not a crash) · **Status:** OPEN, separate from the J.7/J.8 judge-reward redesign

## Summary

The `wp_gen` branch of the RL reward produces **constant all-zero groups every step**, so
`compute_rollout_advantages`'s `remove_constant_reward_groups` drops them and they contribute
**zero gradient**. The wp_gen stream — intended (v1.2 mix decision) as the ~20% codegen replay
that *protects base coding ability during RL* — has therefore been training-inert the entire run.
The model is shaped only by the judge stream.

## Evidence (this session)

- Live metrics: `frac_groups_all_zero` pinned at **exactly 0.375** for all 50 steps, both seeds and
  the fixed-sampler rerun (J.2/J.4). 0.375 = the gen fraction of each batch — i.e. *every* gen group
  is all-zero, deterministically.
- Controlled sampling (`_check_step50_vs_warmstart.py`, J.5): warm-start AND both step-50
  checkpoints emit **0 / 60 parseable PHP** on gen prompts.
- Raw completions (`_probe_weights_moved.py` PROBE 2, J.5): the gen completions are *real code* —
  Elementor `content_template()` JS/Underscore templates (`<# #>`, `<%- %>`) interleaved with
  `<?php ?>`. Legitimate WordPress/Elementor code, but **not standalone-parseable PHP**, so
  `extract_php_code` → `php -l` fails → `_is_parseable_php` = False → `RewardResult.scalar` zeroed
  (the non-code guard at `rl_rollouts.py` ~line 771).

## Root cause

Two compounding factors:
1. **Prompt-type mismatch.** A chunk of the `wp_gen` pool are Elementor/JS-template tasks whose
   *correct* output is a `content_template()` mixing PHP with Underscore/JS markup — never
   standalone-parseable PHP.
2. **Reward extraction is standalone-PHP-only.** `extract_php_code` + `php -l` treat template-mixed
   code as a failed generation and zero it, with no credit for valid-but-non-standalone WP code.

Because every gen group scores 0 uniformly, the group is *constant* → centered advantages are 0 →
dropped by the constant-reward filter. Not bad gradient, just **no** gradient.

## Impact

- **RLEV-01 no-regression risk (the real concern).** The codegen replay was the mechanism meant to
  keep RL from degrading base code generation (v1.2 mix: reasoning + 30% judge replay + 20% wp_gen
  replay). With it inert, RL has had **no counterweight** protecting codegen — Phase 10 RLEV-01
  (RL vs v1.2 SFT on wp-bench, no-regression) could surface a codegen regression that nobody was
  guarding against.
- Wasted batch capacity: ~3/8 of every batch is dead weight (no learning signal, but still sampled +
  judged — minor compute waste).
- Misleading `reward_mean`: gen zeros drag the logged mean down, muddying the judge-axis signal.

## Pre-existing (not introduced this session)

This predates the post-8.1 rerun and the sampler/KL fixes (`ff0872e`). It is orthogonal to the
J.7/J.8 judge-reward-isolation flaw — different stream, different mechanism. Bundle neither into the
other; attribution stays clean.

## Candidate fixes (decide during the reward reopening, not before)

1. **Credit valid-but-non-standalone WP code.** Detect template-mode completions (Elementor
   `content_template`, mixed `<?php ?>` + `<# #>`/`<%- %>`) and validate them on an appropriate
   gate (e.g. lint the PHP segments only, or a template-aware parse) instead of whole-string `php -l`.
2. **Curate the wp_gen pool** to standalone-PHP-emitting tasks so `php -l` is the right gate, and move
   template tasks to a separate (or dropped) stream.
3. **Re-weight / drop wp_gen** if codegen protection is better served another way (e.g. a KL-to-SFT
   anchor on codegen prompts rather than a reward stream).

## Verification when fixed

- `frac_groups_all_zero` should drop below the gen fraction (gen groups start producing non-constant
  rewards).
- A controlled gen-axis eval (extend `_check_judge_fixcorr.py` pattern to gen prompts) should show
  non-zero, varied gen reward across the pool.
- Phase 10 RLEV-01 wp-bench codegen score must hold vs the v1.2 SFT baseline (the actual gate this
  protects).

## Pointers

- Reward zeroing: `scripts/rl_rollouts.py` `_is_parseable_php` guard + `extract_php_code` (eval/output_parsers).
- Gen reward pipeline: `scripts/reward_pipeline.py` `compute_group_rewards`.
- Constant-group drop: `scripts/rl_rollouts.py` `compute_rollout_advantages` / `remove_constant_reward_groups`.
- Evidence artifacts: `09-LOCAL-RL-STATUS-UPDATES.md` J.2/J.4/J.5; `logs/phase09_rerun/probe_weights.log`.
