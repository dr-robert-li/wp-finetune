# Phase 9 — Live GSPO Smoke Findings (2026-06-21)

Branch: `phase9-live-rl-wiring`. Goal: build the missing live-run wiring and run a 1-step
GSPO smoke before any full 500-step run (user-authorized).

## Outcome: live path validated end-to-end EXCEPT the gradient step

The 1-step smoke (`Qwen/Qwen3-30B-A3B`, batch 2, group 2, max_pool 2, isolated outputs
under `output/_smoke/`) reached and ran, **live**:

- ✅ Tinker auth + `ServiceClient` + `TrainingClient` (model `Qwen/Qwen3-30B-A3B`)
- ✅ Prompt pools loaded; `save_weights_and_get_sampling_client()`; `sample()` completions
- ✅ Reward path **live**: PHPCS + `judge_score_single` (vLLM `wp_judge` @ :8000) + capped
  Claude-consistency via `claude_agent` subprocess. Panickssery spot-check fired on REAL
  scores (`fix_correctness=1.0`, `judge_consistency=0.7/0.4`).
- ✅ Advantage assembly (`compute_rollout_advantages`)
- ❌ **`forward_backward_custom`** → `AttributeError: 'dict' object has no attribute 'loss_fn_inputs'`

## Bugs found + fixed in this branch (none of these were live-testable before)

1. **No live entrypoint** — bare `python scripts/rl_train.py` (no `--dry-run`) `SystemExit`ed
   because `main()` couldn't build a judge client. FIXED: extracted `run_training(args)`,
   added `--judge-base-url` (main() builds the `openai.OpenAI` judge client), added
   `--manifest-path`/`--metrics-path` so a smoke can't clobber the canonical manifest.
2. **Wrong base model** — `--model-id` default `Qwen/Qwen3-30B` → Tinker `400 not supported`.
   FIXED: default → `Qwen/Qwen3-30B-A3B` (the supported MoE id). (`tinker_rl_data.BASE_MODEL`
   was already correct.)
3. **Judge truncation** — `judge_score_single` default `max_tokens=512` truncated the v1.2
   *reasoning* judge before its `<judge_output>` JSON → silent `None` → group-mean imputation.
   FIXED: default → 1024 (matches `_judge_create` + the eval path). Verified: clean fn → 53,
   `echo $_GET[...]` → 22.
4. Docs corrected: `run-rl-training` SKILL (Steps 0a/1/2/5 — judge serve, venv, judge-base-url,
   model id) and `09-HUMAN-UAT.md` test #1.
5. New: `scripts/serve_v4_judge_vllm.sh` (serves the v1.2 winner as `wp_judge`).

## The remaining BLOCKER — GSPO gradient datum assembly was never implemented

`run_training_step` → `build_loss_step` → `tc.forward_backward_custom(data, loss_fn)` is
handed a list of **plain dicts** (`{"completion","reward","advantage","group_id"}`) from
`compute_rollout_advantages` (`rl_rollouts._inline_assemble_training_data`). Tinker requires
a list of **`tinker.types.Datum`** objects, each with tokenized inputs and
`loss_fn_inputs` (`target_tokens`, and — for a real GSPO IS ratio — sampling `logprobs`).

Two coupled gaps:
- **No Datum construction.** Completions are never rendered to token ids / `Datum`. The module
  docstring claims delegation to `tinker_cookbook.rl.data_processing`, but **that module does
  not exist** in the installed cookbook (only `rl/{rollouts,types,train,...}.py`). The "inline
  fallback" path is the only path, and it emits dicts.
- **Sampling logprobs discarded.** `rl_rollouts._generate_completions` decodes `sample()` to
  text and drops token ids + sampling logprobs. So `_make_gspo_loss_fn` hits its
  `except (AttributeError, KeyError)` branch and sets `seq_ratio = 1.0` — i.e. even with a
  Datum, GSPO degrades to plain advantage-weighted REINFORCE with **no importance-sampling
  correction** (the whole point of GSPO/RSPO).

This is core RL-update logic, not glue. The dry-run mock and the unit tests both bypass
`forward_backward_custom`, so this never surfaced until a real Tinker call. Fixing it well
requires the cookbook's canonical RL train-loop datum pattern (capture token ids + logprobs
from `sample()`, build `Datum` with `target_tokens` + advantage weights, feed
`forward_backward_custom`) and is correctness-critical — a wrong assembly trains silently-wrong.

## Recommendation

Treat the datum/logprob assembly as a scoped Phase 9 implementation task (re-plan/review,
not an ad-hoc patch). The judge/reward/serving/entrypoint wiring is now proven and reusable.
Phase 10 planning is complete + verified and is unaffected.
