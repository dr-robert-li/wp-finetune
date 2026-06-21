---
phase: 09-gspo-training
plan: 07
type: execute
status: complete
completed: 2026-06-22
requirements: [GRPO-05, GRPO-06, GRPO-07, GRPO-08]
---

# 09-07 SUMMARY — GSPO Datum/logprob assembly gap (corrective)

## ✅ LIVE SMOKE PASSED (Task 5 / DATUM-03, 2026-06-22)
1-step live re-smoke: Tinker `Qwen/Qwen3-30B-A3B`, vLLM `wp_judge` @ :8000, isolated
`output/_smoke/` (`--total-steps 1 --batch-size 2 --group-size 2 --max-pool 2`). EXIT 0.
- (a) ✅ No `loss_fn_inputs` AttributeError, no key-rejection, no traceback.
- (b) ✅ Reached `optim_step`; "Step 1/1: reward_mean=0.3491"; "Training complete"; final
  checkpoint saved (`tinker://…/sampler_weights/final-step-1`); a step-0 `rl_metrics.jsonl`
  row written (e_frac 0.892, halt_reason null, use_gspo true).
- (c) ✅ The GSPO loss body ran on REAL tensors — `sampling_sum` ∈ {-205.9, -134.6, -321.5,
  -280.7} (real, non-zero, large → the action-mask is selecting real action-token logprobs, not
  zeroed obs positions), `used_fallback=False` (except-branch NOT entered). `seq_ratio` was a MIX
  of clamped-1.0 and unclamped (e.g. train -133.4 > sampling -134.6 → exp(+1.2)≈3.3) — both fine:
  per plan Task 5c the necessary-and-sufficient proof is a real non-zero `sampling_sum` with the
  fallback not taken (step-1 ratios near 1.0 are expected precision drift on identical weights).
- (d) ✅ Canonical `output/rl_checkpoints/checkpoint_manifest.json` untouched (mtime 06-20);
  smoke wrote only `output/_smoke/`.
- Reward path live: Panickssery fired on REAL scores (fix_correctness 1.0, judge_consistency 0.0).

### Two corrective fixes the live run surfaced (beyond Tasks 1–4)
1. **SDK API mismatch (Option B).** `forward_backward_custom`'s forward validator accepts only
   `loss_fn_inputs ⊆ {target_tokens, weights}`, but the cookbook datum carries
   `{target_tokens, logprobs, advantages, mask}` (those are for the BACKEND
   `forward_backward(loss_fn="importance_sampling")` + `_remove_mask` path). The plan merged the two
   APIs. Fix (honors D-09-03 GSPO-primary lock; Option A=backend IS is token-level GRPO, a lock-break
   not taken): pass `forward_backward_custom` datums stripped to `{target_tokens}` via
   `_strip_to_target_tokens`; `_make_gspo_loss_fn(full_data)` closes over the FULL datums to read
   mask/advantages/sampling-logprobs (not the stripped `data` arg). GRPO-fallback path also strips
   `mask` (`_strip_mask`). Offline guards: `test_strip_to_target_tokens_satisfies_fbc_validator`
   (subset constraint) + `test_gspo_loss_fn_gradient_flows_to_action_positions` (`loss.backward()` →
   grad non-zero on action, zero on obs).
2. **`optim_step` signature.** `tc.optim_step()` (pre-existing, from 09-05) crashed —
   `TrainingClient.optim_step` requires `AdamParams`. Added `_adam_params(args)` +
   `--learning-rate` (default 1e-5); `run_training_step` now calls `tc.optim_step(_adam_params(args))`.

---
## (Below: the Tasks 1–4 work as originally completed — assembly is correct; only the
## forward_backward call site is being corrected.)

## Objective
Make the GSPO gradient step run on Tinker with a REAL sequence-level importance-sampling
ratio. The live 1-step smoke had reached `tc.forward_backward_custom` and died with
`AttributeError: 'dict' object has no attribute 'loss_fn_inputs'` — the rollout pipeline emitted
plain dicts and discarded sampled logprobs, so even past the crash GSPO fell into its
`except (AttributeError, KeyError): seq_ratio = 1.0` REINFORCE fallback.

## What was built (Tasks 1–4 — offline, all green)
- **Task 1 — `tests/test_rl_datum_assembly.py` (new, 4 tests):** DATUM-01/02 contract guard over
  `trajectory_to_data` / `compute_advantages` / `assemble_training_data` — asserts the four
  `loss_fn_inputs` keys, non-zero sampled logprobs, mask-selected action positions, the
  zero-advantage-member safety, and empty-completion (no-1.0-mask) safety.
- **Task 2 — `scripts/rl_rollouts.py`:** `_Completion` now carries `model_input/tokens/logprobs`;
  `_generate_completions` reads `seq.logprobs` (no longer discarded); `build_trajectory_groups`
  returns `list[TrajectoryGroup]` (one single-turn Trajectory per survivor, T-09-SECDROP applied
  pre-construction, empty groups skipped); `compute_rollout_advantages` delegates to the cookbook
  (`remove_constant_reward_groups → compute_advantages → assemble_training_data`) and returns
  `(list[tinker.Datum], list[float], meta)`. Inline dict-advantage helpers deleted. Origin +
  fix_correctness/consistency stashed in `Transition.logs` for the Panickssery monitor.
- **Task 3 — `scripts/rl_train.py`:** `_make_gspo_loss_fn()` drops the advantages closure and reads
  the action advantage via `mask==1` (NaN-safe on legitimately-zero advantages; numel() guard on
  empty-mask immediate-EOS completions) and the sampled logprobs from `datum.loss_fn_inputs`;
  `build_loss_step` drops the advantages param; `run_training_step` unpacks the 3-tuple and sources
  rewards from `TrajectoryGroup.get_total_rewards()`; `_panickssery_spot_check` reads the pre-Datum
  trajectory groups. CR-04 KL-halt ordering and GRPO-08 autohalt unchanged.
- **Task 4 — tests:** `tests/_rl_fixtures.py` (new) builds real cookbook TrajectoryGroups;
  `test_rl_rollouts.py`, `test_rl_train.py`, `test_rl_train_integration.py` rewritten to the Datum
  contract (no dict shim). Integration fakes gained `.logprobs` and a real `tinker.ModelInput`.
  Wrapper-level regression guard asserts `isinstance(data[0], tinker.Datum)` and
  `"logprobs" in data[0].loss_fn_inputs` — the unit-speed proof the original dict→Datum bug is dead.

## Verification
- `.venv-tinker/bin/python -m pytest tests/test_rl_datum_assembly.py tests/test_rl_rollouts.py
  tests/test_rl_train.py tests/test_rl_train_integration.py tests/test_reward_pipeline.py -q`
  → **86 passed** (includes the 2 direct `gspo_loss_fn` tests below). (Scoped to the rl-touching
  files; `.venv-tinker` is a minimal env — `pytest` was installed into it for this run.)
- Both modules import with AND without tinker (lazy cookbook imports).

## Environment notes
- `tinker` + `tinker_cookbook` + `torch` + `scipy` live in **`.venv-tinker`**, NOT base miniconda3.
  Run all RL code with `.venv-tinker/bin/python`. (The earlier S3219/S3220 "tinker/scipy missing"
  diagnostics were base-conda false starts — resolved.)

## Deviations
- `compute_rollout_advantages` `meta["n_dropped_constant"]` now counts GROUPS (cookbook semantics),
  not completions; the CR-06 filter test was updated accordingly (2 in → 1 constant group dropped).
- **IS-ratio masking correction (plan Task 3 spec was wrong).** The plan/RESEARCH §3 specified
  `train_sum = train_lps.sum()`. But `train_lps` (the SDK's `forward_backward_custom` logprobs) is
  FULL-length — one logprob per target token, obs + action — which is exactly why `loss_fn_inputs`
  carries a `mask`. Summing it unmasked leaks obs-token logprobs into `exp(train_sum - sampling_sum)`
  (sampling_lps zeroes obs positions), corrupting the IS ratio. Fixed to mask BOTH sums to action
  tokens (`mask > 0`), matching the canonical convention in
  `tinker_cookbook/rl/metrics.compute_kl_sample_train` (lines 46–49). Neither the offline assembly
  tests nor Task 5's stated acceptance would have caught this (a masking mismatch still yields a
  non-zero `sampling_sum` with the fallback not taken).
- **Direct loss-fn coverage added.** `gspo_loss_fn`'s body is bypassed by every mock-based test
  (they stop at assembly or mock `forward_backward_custom`). Added two direct tests in
  `test_rl_train.py`: one plants a distinctive `-99.0` logprob on an obs position and asserts it does
  NOT leak into the masked IS ratio (verified discriminating: correct ratio 1.82 vs buggy 1.00); one
  asserts an empty-mask (immediate-EOS) datum contributes 0 without crashing.

## Task 5 acceptance — ADD a masking check
Beyond the plan's a–d: while the temporary `sampling_sum`/`train_sum` logger is in place, confirm the
IS ratio is computed over ACTION tokens only (the per-step `gspo/n_sequences` is sane and `seq_ratio`
is not dominated by obs-token logprobs). The offline `test_gspo_loss_fn_masks_*` already pins this,
but spot-check it on live tensors.

## Task 5 — DONE (passed, see "LIVE SMOKE PASSED" above). Reproduce command:
The 1-step live RL re-smoke (DATUM-03) was run with Tinker credentials (loaded from `.env`) and a
served vLLM `wp_judge` — EXIT 0, all acceptance a–d met. Reproduce:
```bash
bash scripts/serve_v4_judge_vllm.sh         # wait for wp_judge at :8000
export TINKER_API_KEY=...                    # from .env
.venv-tinker/bin/python scripts/rl_train.py \
  --total-steps 1 --batch-size 2 --group-size 2 --max-pool 2 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  --manifest-path output/_smoke/checkpoint_manifest.json \
  --metrics-path output/_smoke/metrics/rl_metrics.jsonl
```
Acceptance: no `loss_fn_inputs` AttributeError; optim_step reached + a `"step": 1` row written;
a REAL non-zero `sampling_sum` with the except-fallback NOT taken (temporarily log sampling_sum/
train_sum per plan Task 5, then remove); canonical `output/rl_checkpoints/` untouched.

SCOPE: this 1-step smoke proves the GSPO gradient MECHANISM (datum/logprob assembly + a real IS
ratio reaching optim_step). It does NOT by itself clear the 4 `09-HUMAN-UAT.md` items or unblock
Phase 10 — those require the FULL multi-step live RL run (reward convergence, KL stability,
Jaccard on real routing, real trained checkpoints). `09-HUMAN-UAT.md` stays `status: partial`.
