---
phase: 02-dataset-production
plan: 02
subsystem: phase2-pipeline-hardening
tags: [hardening, utils-integration, phpcs, rate-limiting, batch-api, checkpoints]
dependency_graph:
  requires: [02-01]
  provides: [hardened-phase2-scripts]
  affects: [phase2-pipeline-execution]
tech_stack:
  added: []
  patterns: [call_with_backoff, extract_json, load_checkpoint/save_checkpoint, batch_or_direct]
key_files:
  created:
    - tests/test_phase2_mutate.py
    - tests/test_phase2_judge_dataset.py
  modified:
    - scripts/phase2_mutate.py
    - scripts/phase2_generate.py
    - scripts/phase2_judge.py
    - scripts/phase2_judge_dataset.py
decisions:
  - "PHPCS hard-fail guard added at module level in phase2_mutate.py — no silent fallback on FileNotFoundError"
  - "verify_mutation_detectable() now calls sys.exit(1) instead of returning True when PHPCS disappears mid-run"
  - "phase2_generate.py checkpoint key is rejection_examples for the rejection generation section"
  - "batch results saved to disk immediately in phase2_judge_dataset._score_batch (24h expiry protection)"
  - "security auto-FAIL enforced in _apply_security_auto_fail() in phase2_judge.py (score < 5 forces FAIL)"
metrics:
  duration_minutes: 5
  completed_date: "2026-03-26"
  tasks_completed: 2
  files_changed: 6
---

# Phase 02 Plan 02: Harden Phase 2 Pipeline Scripts Summary

**One-liner:** PHPCS hard-fail guard in phase2_mutate.py and full utils.py integration (call_with_backoff, extract_json, checkpoints, batch routing) across all four Phase 2 pipeline scripts.

## What Was Built

All four Phase 2 pipeline scripts hardened with utils.py integration:

1. **phase2_mutate.py** — Added `_require_phpcs()` guard that calls `sys.exit(1)` on `FileNotFoundError` instead of silently returning `True`. Replaced catch-and-return in `verify_mutation_detectable()` with explicit exit.

2. **phase2_generate.py** — Removed `REQUESTS_PER_MINUTE`/`REQUEST_INTERVAL` constants and all `time.sleep()` calls. Added `call_with_backoff` via `generate_one()`, checkpoint/resume with `load_checkpoint`/`save_checkpoint`, `batch_or_direct` routing for large gap deficits, and rejection example generation from `rejection_templates` in `synthetic_prompts.yaml`.

3. **phase2_judge.py** — Removed rate-limit constants and sleep calls. Replaced brittle split-based JSON parsing with `extract_json`. Added `call_with_backoff`, `_apply_security_auto_fail()` enforcement (security score < 5 forces FAIL), checkpoint/resume per file, and batch routing for large file batches.

4. **phase2_judge_dataset.py** — Fixed PIPE-03 concern: removed `REQUEST_INTERVAL` and `time.sleep`. Added `call_with_backoff`, `extract_json`, checkpoint/resume (saves every 100 examples), batch routing, and immediate disk save of batch results after `parse_batch_results()` (24h expiry protection).

## Test Results

- **tests/test_phase2_mutate.py**: 3 tests, all passing
- **tests/test_phase2_judge_dataset.py**: 4 tests, all passing
- **Full test suite**: 39 tests, 0 failures

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing safety] Safe tmp_path initialization in verify_mutation_detectable**
- **Found during:** Task 1
- **Issue:** `tmp_path` variable in `finally` block could cause `NameError` if `NamedTemporaryFile` failed before assignment
- **Fix:** Initialize `tmp_path = None` before the try block and guard `finally` with `if tmp_path is not None`
- **Files modified:** scripts/phase2_mutate.py
- **Commit:** 89d5f2e

None of the other changes deviated from the plan.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | 89d5f2e | feat(02-02): add PHPCS hard-fail guard and harden phase2_generate.py |
| Task 2 | d33d719 | feat(02-02): harden phase2_judge.py and phase2_judge_dataset.py with utils.py |

## Self-Check: PASSED

- scripts/phase2_mutate.py: FOUND
- scripts/phase2_generate.py: FOUND
- scripts/phase2_judge.py: FOUND
- scripts/phase2_judge_dataset.py: FOUND
- tests/test_phase2_mutate.py: FOUND
- tests/test_phase2_judge_dataset.py: FOUND
- Commit 89d5f2e: FOUND
- Commit d33d719: FOUND
- 39 tests passing: CONFIRMED
