---
phase: 09-gspo-training
plan: 07
type: execute
status: complete_pending_live_gate
completed: 2026-06-22
requirements: [GRPO-05, GRPO-06, GRPO-07, GRPO-08]
---

# 09-07 SUMMARY — GSPO Datum/logprob assembly gap (corrective)

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
  → **84 passed**. (Scoped to the rl-touching files; `.venv-tinker` is a minimal env — `pytest`
  was installed into it for this run.)
- Both modules import with AND without tinker (lazy cookbook imports).

## Environment notes
- `tinker` + `tinker_cookbook` + `torch` + `scipy` live in **`.venv-tinker`**, NOT base miniconda3.
  Run all RL code with `.venv-tinker/bin/python`. (The earlier S3219/S3220 "tinker/scipy missing"
  diagnostics were base-conda false starts — resolved.)

## Deviations
- `compute_rollout_advantages` `meta["n_dropped_constant"]` now counts GROUPS (cookbook semantics),
  not completions; the CR-06 filter test was updated accordingly (2 in → 1 constant group dropped).

## Task 5 — REMAINING, human/credential/GPU-gated (NOT run here)
The 1-step live RL re-smoke (DATUM-03) requires Tinker cloud credentials + a served vLLM judge —
it cannot run in this environment. To complete:
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
train_sum per plan Task 5, then remove); canonical `output/rl_checkpoints/` untouched. On success,
this unblocks Phase 10 Wave 1 (Task 3 GATE) and clears the 4 pending `09-HUMAN-UAT.md` items.
