---
status: resolved
trigger: "why hasn't Phase 9 live Tinker RL run landed"
created: 2026-06-22
updated: 2026-06-22
mode: diagnose
---

# Debug: Phase 9 live Tinker RL run not landed

## Symptoms
- `09-HUMAN-UAT.md`: 4 live-run tests all `[pending]` since 2026-06-20.
- `output/rl_checkpoints/metrics/rl_metrics.jsonl` = 8 lines (dry-run only), `dry_run`/`status` null in `checkpoint_manifest.json`.
- Phase 10 Wave 1 GATE (Task 3) blocked — consumes live metrics that don't exist.

## Root Cause (confirmed)
A live 1-step smoke WAS attempted. It cleared the full RL path — Tinker auth, sampling
(`.sample()`), dual gen/judge reward, advantage assembly — then **crashed at the gradient step**:
`tc.forward_backward_custom` was handed plain dicts instead of real `tinker.Datum` objects →
AttributeError before any `optim_step`. Underneath, `_make_gspo_loss_fn` was silently taking the
`except (AttributeError, KeyError): seq_ratio = 1.0` REINFORCE fallback, so the GSPO importance
ratio was never real (ref obs S3233: `seq_ratio != 1.0` is an invalid readiness signal — real
signal is non-zero `sampling_sum` in `datum.loss_fn_inputs["logprobs"]`).

The fix exists as **corrective plan 09-07** ("GSPO Datum/logprob assembly gap"): adopt cookbook
`trajectory_to_data`, capture real `SampledSequence.logprobs` + tokens in `_generate_completions`,
emit `list[Datum]` with baked target_tokens/logprobs/advantages/mask, GSPO reads from
`datum.loss_fn_inputs`. Plan is committed (1ae001e/bb26907) but:
- lives on branch `phase9-live-rl-wiring` (NOT merged to active `phase10-execution`)
- has NO 09-07-SUMMARY on either branch → **NOT EXECUTED**
- branch divergence: phase10-execution 5 ↔ 3 phase9-live-rl-wiring

## Eliminated (not the cause)
- Credentials: TINKER_API_KEY configured (S3218). NOT the blocker.
- Env import errors (S3219 tinker missing, S3220 scipy missing): false-start — caused by running
  base miniconda3 python. `.venv-tinker/bin/python` has tinker + scipy(1.17.1) + openai all
  importing. Env READY. Run MUST use `.venv-tinker`, not base conda.
- Pre-run arg landmine (missing --judge-model/--n-votes, hard judge_client access): RESOLVED 06dcba7
  (now fails fast with actionable guard).

## Fix / Remediation path
1. Bring 09-07 onto active line (merge `phase9-live-rl-wiring` or cherry-pick the plan).
2. Execute 09-07 (TDD plan; `/gsd-execute-phase 09` run 09-07) — closes the Datum/logprob gap so
   the gradient step runs with a real IS ratio. This is the runnable code fix.
3. Launch the credentialed live run via `.venv-tinker/bin/python` (manual, by design):
   `wp-finetune:run-rl-training` / `python scripts/rl_train.py` (no `--dry-run`, judge_client attached).
4. Confirm `sampling_sum` non-zero + `used_fallback=false` in `rl_metrics.jsonl` → unblocks Phase 10.

## Files
- scripts/rl_rollouts.py, scripts/rl_train.py (09-07 targets)
- .planning/phases/09-gspo-training/09-07-PLAN.md (on phase9-live-rl-wiring)
- .planning/phases/09-gspo-training/09-HUMAN-UAT.md, 09-VALIDATION-datum-gap.md
