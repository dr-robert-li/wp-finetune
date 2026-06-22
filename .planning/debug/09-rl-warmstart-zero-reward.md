# Phase 09 — Warm-started RL: zero-reward / no-gradient (BLOCKING)

**Date:** 2026-06-22 ~18:00 AEST
**Status:** 🔴 RL run KILLED at step 2/500 (task br921nskk). Regen + warm-start were CORRECT; reward pipeline is the blocker.

## What worked
- v4 `save_state` regen: gate PASS (loadable state_path, MoE-only, terse/FS 0.000, loss 12.66→2.22).
- Warm start CONFIRMED in log: `WARM START … train_mlp=True attn=False unembed=False` from
  `tinker://80c93d7c-…/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`.
- Model emits real WordPress PHP + reasoning. NO raw-base refusals (judge_failures new=0 beyond stale baseline 84).

## The bug
Every rollout scores reward **exactly 0.0** (min=max=0.0, n=32) every step → "All rewards are uniform.
There will be no gradient" → no learning. Real run rows in metrics/rl_metrics.jsonl:
`model_id=Qwen/Qwen3-30B-A3B, ts 2026-06-22T07:43/07:49/07:56Z, step 0/1/2 reward 0.0/0.0/0.0`.
(NOTE: metrics file is append-only across runs — it also holds June-20 mock rows [model 'Qwen/Qwen3-30B',
reward 0.9, n=2] and the earlier COLD-START run [A3B, reward ~0.25-0.46, steps 0-60]. Filter by ts/model_id.)

### Root cause (from judge_failures.preguard.jsonl, 3 rollouts)
- [0] finish=stop, [1] finish=stop, [2] finish=length — 2 of 3 stop naturally, so length is NOT the main issue.
- ALL: has_judge_output_tag=False, has_prose_score=False, code_starts_phpish=False, code_has_fence=False.
- raw_text sample = bare PHP method body (`function init(){ add_filter(...) }`) — no `<?php`, no `<wp_gen>`/
  `<wp_judge>` wrapper, no `<judge_output>` JSON.
- Judge-mode is 60% of the batch (rl_rollouts.py JUDGE_RATIO=0.6); judge reward needs a `<judge_output>`
  JSON block (reward_pipeline.py:496). Model isn't emitting it → fix_correctness=0.0. Dominant 0-reward source.
- Gen-mode: rl_rollouts.py:234 prepends `<?php` when missing → gen may be recoverable; verify gen rewards
  separately aren't also 0.

### Hypotheses to test (next session)
1. **Rollout prompt vs SFT template mismatch** — the judge_pool prompt likely doesn't prime the model to emit
   `[REASONING]…[/REASONING]` + `<judge_output>{json}` the way the SFT/eval prompt did. Compare the exact
   rollout ModelInput (rl_rollouts.py prompt render) against the SFT judge training format + eval_judge prompt.
2. **Reward parser too strict / wrong markers** — confirm the gen/judge parsers' required anchors match what
   v4 actually emits (dump 1 full rollout prompt+completion, run it through the parser by hand).
3. **Sampling** — max_new_tokens default 512 (rl_rollouts.py:698); judge reasoning may need more (sample[2]
   hit length at 7771 chars). Check stop tokens too.

### Secondary issue
~7 min/training step (judge vLLM scores 32 completions at ~29 tok/s). 500 steps ≈ 2+ days even once rewards
work. Consider faster judge serving (higher GPU_MEM_UTIL / batching) or fewer total-steps for a first valid run.

## Do NOT
- Do not relaunch RL until a rollout prompt+completion is shown to parse to a NON-zero reward in a standalone
  probe. Otherwise it burns ~7min/step for zero gradient.

---

## PROBE RESULTS (scripts/_probe_rl_reward.py — offline, read-only, 2026-06-22)

Run: `REWARD_SKIP_PHPCS_ASSERT=1 .venv-tinker/bin/python scripts/_probe_rl_reward.py`
(`--completions output/rl_checkpoints/judge_failures.preguard.jsonl` to use captured rollouts.)

Three mechanisms localized — the zero is NOT a single bug but a format mismatch hitting both pathways:

### 1. JUDGE path (60% of batch) — PRIMARY, cleanly confirmed
- Reward = fix_correctness on the ```php fix `extract_php_code` pulls from the completion.
- Warm v4 emits critique **PROSE** (its SFT format: "WPCS Compliance: score 9/10 — …"), NO ```php fence.
- Probe: `prose_critique -> extract not parseable -> fix_correctness=0.000 -> combined=0.000`.
  (`bare_code` and `fenced_fix` both score fix=1.0/combined=0.70 — so the path works; the model's
  *format* is the mismatch.)

### 2. FROZEN reward-judge parse — real mismatch (affects gen judge-component)
- `compute_group_rewards` calls `judge_score_single` -> `parse_judge_response`, which (Strategy 0)
  wants a `<judge_output>{json}</judge_output>` block.
- Probe: `PROSE (v4 SFT format) -> parsed=None`; `JSON <judge_output> -> overall_score=88`.
- So the served wp_judge (merged v4) emitting prose cannot be parsed -> judge score None -> imputed
  (directional bias). The reward parser and the served model's output format disagree.

### 3. GEN path — MO-GRPO within-group normalization centers to ~0
- `composite_pre_gate` is built from `_mo_grpo_norm`'d signals (mean-0, scaled by within-group std).
- Probe (near-identical strong outputs, phpcs 100/100/99.7/99.7): scalars +0.46/+0.88/-0.67/-0.67,
  **mean 0.0**. When v4's group outputs are *uniformly* high quality (variance -> 0), every normalized
  signal -> 0 -> composite -> exactly 0. A strong consistent policy starves its own gradient here.

### Net
Judge rollouts (60%) hit exactly 0.0 via prose-vs-fence (mechanism 1). Gen rollouts center to ~0 via
normalization (mechanism 3), exactly 0 when variance collapses. Batch reward_min=reward_max=0.0 ->
"all rewards uniform" -> no gradient. The warm-start itself is FINE — the reward pipeline assumes the
chatty/fenced/JSON shapes of an *instruct* policy, but the v4 SFT policy emits bare code + prose scores.

## Candidate fixes (decide before relaunch — each needs its own probe pass)
1. JUDGE reward: make the judge rollout PROMPT ask for "critique + corrected code in a ```php fence",
   OR change fix-correctness extraction to accept v4's prose+code shape, OR score the prose dimension
   numbers directly (eval_judge already has a `score N/10` regex). Pick one so judge rollouts parse.
2. FROZEN judge: make `parse_judge_response` accept v4 PROSE scores (reuse the `score\s+\d+\s*/\s*10`
   path), OR serve a judge variant that emits `<judge_output>` JSON. Output format must match parser.
3. GEN gradient: optionally blend a small NON-normalized raw component so a uniformly-strong group still
   carries signal (otherwise gen is gradient-starved by construction — acceptable since gen is solved).
4. GATE before relaunch: `scripts/_probe_rl_reward.py` must show a real rollout shape parsing to a
   NON-zero, NON-uniform group reward before burning another ~7min/step run.

---

## FIXES IMPLEMENTED (2026-06-22) + offline gate PASS

### A — Mechanism 2: prose-score fallback (eval/eval_judge.py)
- Added `_PROSE_LABEL_TO_FIELD` + `_parse_prose_dim_scores()`; wired into `judge_score_single`
  AFTER the JSON/derive path, BEFORE the failure dump. Kept OUT of `parse_judge_response`
  (dual-use / teacher-GT purity — matches the existing `_derive_overall_from_dims` placement).
- Probe: `PROSE -> overall_score=88.9` (was None). Frozen judge can now score v4 prose.

### B — Mechanism 1: judge rollout output-contract + extraction + token cap (scripts/rl_rollouts.py)
- `_JUDGE_FIX_INSTRUCTION` appended to every judge prompt (`_augment_judge_prompt`, idempotent):
  "critique, then corrected code in a ```php fenced block beginning with <?php".
- `_extract_corrected_php()` accepts `<corrected_code>` (SFT delimiter) OR ```php; used in the
  judge fix-correctness path (eval-shared `extract_php_code` left unchanged — no blast radius).
- `JUDGE_MAX_NEW_TOKENS=1536` for judge generation (gen stays 512) via `max_tokens_override`
  plumbed through `_generate_completions`/`_build_sampling_params`. SFT judge target ~900 tok +
  fix ~150 tok < 1536 — no truncation.
- Probe: `critique + ```php -> fix=1.000`; `critique + <corrected_code> -> fix=0.997`; prose-only -> 0.

### Mechanism 3 (gen normalization): left as-is per council (secondary). Probe shows gen non-uniform
on these shapes (min/max spread), so not gradient-dead once judge produces signal.

### OFFLINE GATE (scripts/_probe_rl_reward.py) — ALL PASS
(1) judge shape -> nonzero fix_correctness; (2) prose -> numeric overall; (3) gen non-uniform.
Tests: 175 relevant pass; the 1 failure (test_lora_config) is PRE-EXISTING in rl_train.py
(reproduces with my edits stashed), unrelated.

### REMAINING (LIVE) GATE — cannot be done offline
The offline probe proves the CODE parses/scores correct shapes. It CANNOT prove the warm v4 policy
actually EMITS a fix block under the augmented prompt. Real-rollout gate = a 50-100 step signal run:
require reward_min != reward_max AND reward_mean trending up before committing to the full 500-step
(~2-day) run. If judge rollouts still show fix=0 live, the model isn't honoring the contract -> iterate
on the instruction wording (or accept score-only judge reward, a product decision).
