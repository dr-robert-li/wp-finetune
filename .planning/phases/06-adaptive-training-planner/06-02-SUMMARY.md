---
phase: 06-adaptive-training-planner
plan: 02
subsystem: training
tags: [telemetry, gpusampler, unsloth, failure-classification, power-sampling, thermal-thresholds]

# Dependency graph
requires:
  - phase: 06-adaptive-training-planner/06-01
    provides: telemetry infrastructure (GPUSampler, classify_failure from dgx-toolbox)
provides:
  - train_model.py with power sampling via GPUSampler every 50 steps writing canonical JSONL
  - Unsloth silent override detection via trainer.args inspection, writes _unsloth_actuals.json
  - Failure classification after training via classify_failure, writes _run_classification.json
  - observe-training SKILL.md with corrected 82C/85C thermal thresholds
affects:
  - 06-adaptive-training-planner
  - scripts/train_model.py callers
  - observe-training skill users

# Tech tracking
tech-stack:
  added: [telemetry.sampler.GPUSampler, telemetry.failure_classifier.classify_failure]
  patterns:
    - Lazy GPUSampler init in callback (sentinel False if import fails, prevents retry)
    - trainer.args inspection for Unsloth override detection (safe, no print monkey-patching)
    - try/except around trainer.train() to capture exit_code and training_completed

key-files:
  created: []
  modified:
    - scripts/train_model.py
    - .claude/skills/wp-finetune:observe-training/SKILL.md

key-decisions:
  - "Unsloth override detection via trainer.args (not builtins.print) — reads args after build_trainer returns, never before trainer.train()"
  - "GPUSampler uses lazy init with False sentinel — import failure is permanent, avoids retry on every step"
  - "ADAPTIVE_THERMAL_LOG env var as override for canonical JSONL path — allows adaptive planner to redirect output"
  - "No sudo drop_caches anywhere in train_model.py — cache management delegated to dgx-toolbox UMAMemModel"

patterns-established:
  - "Failure sentinel pattern: self._sampler = False when ImportError, checked via 'is not False' idiom"
  - "trainer.args inspection for Unsloth batch override detection — post-build, pre-train window"

requirements-completed: [TELE-01, TELE-02, TELE-03, TELE-04, BTCH-02, BTCH-03]

# Metrics
duration: 8min
completed: 2026-04-01
---

# Phase 06 Plan 02: Power Telemetry and Unsloth Detection Summary

**GPU power sampling via GPUSampler in MemoryWatchdogCallback, Unsloth silent override detection via trainer.args, failure classification post-training, and 82/85C thermal thresholds in observe-training skill**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-01T05:10:00Z
- **Completed:** 2026-04-01T05:18:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended MemoryWatchdogCallback with GPUSampler lazy-init, appending power records to canonical JSONL every 50 steps (TELE-01/TELE-02)
- Added ADAPTIVE_THERMAL_LOG env var override and default path `telemetry/training/canonical.jsonl` wired into build_trainer
- Implemented Unsloth override detection by reading trainer.args after build_trainer returns — writes _unsloth_actuals.json when batch or grad_accum differs from config (BTCH-02/BTCH-03)
- Wrapped trainer.train() in try/except; classify_failure writes _run_classification.json using final GPUSampler reading (TELE-03)
- Updated observe-training SKILL.md thermal thresholds from 80C/83C to 82C/85C (TELE-04)
- Zero forbidden patterns: no drop_caches, no builtins.print monkey-patching

## Task Commits

1. **Task 1: Extend train_model.py with power sampling, Unsloth detection, failure classification** - `d8c791c` (feat)
2. **Task 2: Update observe-training thresholds to 82/85C** - `72926d2` (feat)

## Files Created/Modified

- `scripts/train_model.py` - MemoryWatchdogCallback extended with GPUSampler; Unsloth detection via trainer.args; trainer.train() wrapped with failure classification
- `.claude/skills/wp-finetune:observe-training/SKILL.md` - Thermal thresholds updated to 82C (WARNING) and 85C (CRITICAL)

## Decisions Made

- trainer.args inspection for Unsloth detection: reads actual args after build_trainer returns — safe, deterministic, no side effects vs print monkey-patching
- GPUSampler False sentinel: once ImportError occurs the sampler is set to False (not None), preventing re-import on every subsequent step
- ADAPTIVE_THERMAL_LOG env var: allows adaptive planner (plan 06-03) to set the JSONL output path dynamically per training run
- No drop_caches: HIGH review concern addressed by omission — dgx-toolbox UMAMemModel handles cache management internally

## Deviations from Plan

None - plan executed exactly as written. The pseudocode in Change 3 had a typo (`_json.write_text` instead of `class_path.write_text`) but the plan's own note explicitly flagged this and instructed the executor to write it correctly.

## Issues Encountered

None. All four targeted changes applied cleanly. The only minor issue was a comment containing the string "builtins.print" that triggered the verification assertion — fixed by rewording the comment.

## User Setup Required

None - no external service configuration required. The `ADAPTIVE_THERMAL_LOG` env var is optional (defaults to `telemetry/training/canonical.jsonl`).

## Next Phase Readiness

- train_model.py now emits canonical JSONL with power_watts and mem_available_gb fields on every 50-step interval
- _unsloth_actuals.json and _run_classification.json written to telemetry/training/ for plan 06-03 (adaptive planner) to consume
- observe-training skill thresholds aligned with power-primary routing logic in plan 06-03
- Plan 06-03 (adaptive planner core) can now safely read telemetry outputs from this plan

---
*Phase: 06-adaptive-training-planner*
*Completed: 2026-04-01*

## Self-Check: PASSED

- scripts/train_model.py: FOUND
- .claude/skills/wp-finetune:observe-training/SKILL.md: FOUND
- .planning/phases/06-adaptive-training-planner/06-02-SUMMARY.md: FOUND
- Commit d8c791c (Task 1): FOUND
- Commit 72926d2 (Task 2): FOUND
