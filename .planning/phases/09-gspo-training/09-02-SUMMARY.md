---
phase: "09"
plan: "02"
subsystem: rl-training
tags: [gspo, grpo, test-contract, wave-0, tinker, roadmap]
dependency_graph:
  requires: [09-01-SUMMARY.md]
  provides: [tests/test_rl_train.py, tests/conftest.py#mock_tinker_client]
  affects: [.planning/ROADMAP.md, 09-03-PLAN.md, 09-04-PLAN.md, 09-05-PLAN.md]
tech_stack:
  added: [pytest.importorskip pattern, MagicMock Tinker client fixture]
  patterns: [Wave-0 test scaffolding, importorskip RED/SKIP discipline, GSPO-primary assertion]
key_files:
  created: [tests/test_rl_train.py]
  modified: [tests/conftest.py, .planning/ROADMAP.md]
decisions:
  - "mock_tinker_client mocks BOTH forward_backward AND forward_backward_custom (GSPO default path D-09-03)"
  - "test_gspo_rspo_floor asserts both RSPO clamp AND GSPO default path selection in one test"
  - "ROADMAP Goal sentence updated to state GSPO PRIMARY locked (not planning-time decision)"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-20"
  tasks_completed: 3
  tasks_total: 3
  files_created: 1
  files_modified: 2
---

# Phase 9 Plan 02: RL Test Contract (Wave-0 Stubs + ROADMAP Correction) Summary

Wave-0 test contract for GSPO/GRPO RL training: 8 importorskip-guarded stubs encoding GRPO-05/06/07/08 behavioral contracts, `mock_tinker_client` fixture mocking both GRPO fallback (`forward_backward`) and GSPO default path (`forward_backward_custom` per locked D-09-03), and Phase 9 ROADMAP skill-text corrected from stale DGX references to Tinker-native implementation detail.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add `mock_tinker_client` fixture to `tests/conftest.py` | d54c071 | `tests/conftest.py` |
| 2 | Create `tests/test_rl_train.py` with 8 GRPO-05/06/07/08 stubs | a8b7a3c | `tests/test_rl_train.py` |
| 3 | Correct ROADMAP Phase 9 skill text (DGX -> Tinker-native) | 7ba1783 | `.planning/ROADMAP.md` |

## What Was Built

### Task 1: mock_tinker_client fixture

Added `mock_tinker_client` to `tests/conftest.py` (appended; existing 4 fixtures untouched).

Critical design: `tc.forward_backward_custom.return_value = fb_out` is set EXPLICITLY — not left
to auto-mock — because the GSPO dry-run in plan 09-05 calls `forward_backward_custom` and needs
`.metrics` (MoE routing keys) and `.training_logprobs` accessible. Auto-mock would return a bare
MagicMock that cannot expose those attributes predictably.

`fb_out.metrics` contains below-threshold values (`e_frac_with_tokens:mean=0.6` < soft alert 0.7)
so that routing-autohalt tests exercise the halt branch without live Tinker infrastructure.

### Task 2: tests/test_rl_train.py — 8 importorskip stubs

Class `TestRLTrainUnit` with 8 methods, all SKIP while `scripts.rl_train` / `scripts.rl_rollouts`
are absent (plans 09-03/04/05 write those). When modules land, each test becomes RED on wrong or
missing symbol — never vacuously green.

| Method | Req | Module | Contract |
|--------|-----|--------|----------|
| `test_dual_mode_batch` | GRPO-05 | rl_rollouts | batch has both gen and judge items |
| `test_judge_ge_gen_budget` | GRPO-05 | rl_rollouts | n_judge >= n_gen per D-09-04 |
| `test_lora_config` | GRPO-06 | rl_train | train_mlp/attn/unembed=True; NO train_router |
| `test_protected_mask_check` | GRPO-06 | rl_train | Jaccard float in [0, 1] |
| `test_gspo_rspo_floor` | GRPO-07 | rl_train | ratio<1→1.0 (clamp); default=forward_backward_custom |
| `test_grpo_advantages` | GRPO-07 | rl_rollouts | mixed→non-zero; constant→dropped |
| `test_kl_autohalt` | GRPO-08 | rl_train | kl_v1=0.4 > 0.3 triggers halt |
| `test_routing_autohalt` | GRPO-08 | rl_train | e_frac=0.4 < 0.5 triggers halt |

Anti-vacuity verification: `importorskip` count = 8 (in test methods) + 1 (docstring) = 9 total;
`assert ` count = 17. Run confirmed: `8 skipped, 0 passed, 0 failed`. Phase 8 tests: 37 passed.

File-ownership rule: plans 09-03/04/05 write `scripts/rl_train.py` and `scripts/rl_rollouts.py`
ONLY — they must NOT edit `tests/test_rl_train.py`.

### Task 3: ROADMAP Phase 9 skill text correction

Removed all DGX execution references from Phase 9 skill block:
- `dgx.execute("unsloth_studio", ...)` per-epoch loop
- `dgx.validate(["toolbox", "config", "memory:70"])` pre-flight
- `dgx.ensure_ready("unsloth_studio")` container check
- 6-agent telemetry team (`observe-training`)
- `wp-finetune:adaptive-planner` thermal/power config adjustment between epochs
- Per-epoch protected expert regularizer injection

Added Tinker-native implementation detail:
- Execution venue: Tinker cloud exclusively (D-09-01 locked)
- GSPO primary: `tc.forward_backward_custom` with RSPO floor; GRPO fallback via `--grpo-fallback` (D-09-03)
- Per-step autohalt guards: KL 0.1/0.3, routing 0.7/0.5 thresholds
- Protected expert monitoring: `ForwardBackwardOutput.metrics` monitor-only (D-09-02)

Also updated:
- **Goal sentence**: "Whether to also evaluate GRPO... is an implementation decision at planning time" → GSPO PRIMARY locked per D-09-03
- **Success Criteria 2**: "routing regularizer" → router gates FROZEN per D-09-02; monitor-only via metrics

Verification: Phase 9 block has 0 `dgx.execute` references; Phase 10's `dgx.execute("eval_toolbox")` preserved.

## Deviations from Plan

None — plan executed exactly as written.

The `mock_tinker_client` fixture matches the PATTERNS.md spec with one clarifying addition: the
explicit mock of `forward_backward_custom` is noted in docstring/comments with rationale (not just
the bare assignment from the spec). This improves maintainability without changing behavior.

## Known Stubs

All 8 stubs in `tests/test_rl_train.py` are intentional Wave-0 stubs that will be resolved by:
- `09-03-PLAN.md`: provides `scripts.rl_consistency` (scorer)
- `09-04-PLAN.md`: provides `scripts.rl_rollouts` (`sample_interleaved_prompts`, `compute_rollout_advantages`, `combine_judge_reward`)
- `09-05-PLAN.md`: provides `scripts.rl_train` (`rspo_floored_ratio`, `build_loss_step`, `check_halt`, `protected_mask_jaccard`, `build_training_client`)

These stubs are the explicit goal of this plan — downstream plans make them GREEN.

## Self-Check

Files created/modified:
- `tests/conftest.py` — FOUND (edit verified, syntax OK)
- `tests/test_rl_train.py` — FOUND (created, 8 skipped confirmed)
- `.planning/ROADMAP.md` — FOUND (Phase 9 block updated, 0 dgx.execute confirmed)

Commits:
- d54c071 feat(09-02): add mock_tinker_client fixture with GSPO forward_backward_custom mock — FOUND
- a8b7a3c test(09-02): add Wave-0 RL train contract stubs (GRPO-05/06/07/08) — FOUND
- 7ba1783 docs(09-02): correct Phase 9 ROADMAP skill text from DGX to Tinker-native — FOUND

## Self-Check: PASSED
