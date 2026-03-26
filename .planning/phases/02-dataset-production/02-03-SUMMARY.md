---
phase: 02-dataset-production
plan: "03"
subsystem: pipeline-scripts
tags: [cot, export, ratio-enforcement, metadata, validation, dedup, sample-weight, backoff, checkpoints]
dependency_graph:
  requires: ["02-02"]
  provides: ["hardened-phase3-cot", "export-with-ratio-enforcement", "metadata-json", "test-suite-export"]
  affects: ["03-training"]
tech_stack:
  added: []
  patterns: [exponential-backoff, checkpoint-resume, gen-judge-ratio-enforcement, sha256-dedup, php-lint-validation, sample-weight-training]
key_files:
  created:
    - tests/test_export.py
  modified:
    - scripts/phase3_cot.py
    - scripts/export_dataset.py
decisions:
  - "round() used instead of int() for gen/judge ratio calculation to avoid float precision truncation (20 * 0.60/0.40 = 29.9999... -> int gives 29, round gives 30)"
  - "utils.py checkpoints save every 100 examples in phase3_cot.py (authoritative resume); per-500 progress JSONL files kept for additional recovery"
  - "deduplicate() uses SHA-256 of assistant message content — content hash is most reliable duplicate signal for training examples"
metrics:
  duration: "4 min"
  completed_date: "2026-03-26"
  tasks_completed: 2
  files_modified: 3
---

# Phase 2 Plan 3: Harden CoT Script and Export Pipeline Summary

**One-liner:** Hardened phase3_cot.py with call_with_backoff + utils.py checkpoint resume, and added 40/60 gen/judge ratio enforcement, metadata.json generation, PHP lint validation, SHA-256 dedup, and sample_weight to export_dataset.py with 7-test TDD coverage.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Harden phase3_cot.py with utils.py integration | 5692cab | scripts/phase3_cot.py |
| 2 | Update export_dataset.py + test suite (TDD) | 71baa48 (RED), 5fc31e7 (GREEN) | scripts/export_dataset.py, tests/test_export.py |

## What Was Built

### Task 1: Hardened phase3_cot.py

- Removed `REQUESTS_PER_MINUTE = 40` and `REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE` constants
- Removed all `time.sleep(REQUEST_INTERVAL)` calls (3 total)
- Added `from scripts.utils import call_with_backoff, load_checkpoint, save_checkpoint`
- All 4 `client.messages.create(...)` call sites replaced with `call_with_backoff(client, ...)`
- `main()` now loads checkpoint at start: `checkpoint = load_checkpoint("phase3_cot")`; completed set skips already-processed examples
- `save_checkpoint("phase3_cot", {...})` called every 100 examples and at end of loop
- Per-500 progress JSONL files retained for additional recovery alongside utils.py checkpoints

### Task 2: Updated export_dataset.py + test suite

New constants:
- `GEN_TARGET_RATIO = 0.40`
- `JUDGE_TARGET_RATIO = 0.60`

New functions:
- `enforce_ratio()`: Caps the majority class to achieve 40/60 split using `round()` for float safety
- `add_sample_weight()`: Sets `metadata.sample_weight = 1.5` for `source in ("mutated", "contrastive")` or `overall_score < 7`, else `1.0`
- `deduplicate()`: SHA-256 hash of assistant content; returns `(unique_list, dupe_count)`
- `validate_php_sample()`: Runs `php -l` on up to 50 sampled examples; skips if php unavailable
- `generate_metadata()`: Full stats dict with `gen_ratio_actual`, `judge_ratio_actual`, `taxonomy_coverage`, `train_val_test_counts`, `rejection_examples`, `php_lint_failures`, `duplicates_removed`, `sample_weighted_count`

Updated `main()` pipeline order:
1. `load_dataset()`
2. `deduplicate(dataset)`
3. `enforce_ratio(dataset)`
4. `[add_sample_weight(ex) for ex in dataset]`
5. Shuffle and split (80/10/10)
6. `validate_php_sample(dataset)`
7. `generate_metadata(...)` + `json.dump(metadata.json)`
8. Export all formats (OpenAI, Alpaca, Raw) for each split

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed float precision truncation in enforce_ratio()**
- **Found during:** Task 2 GREEN phase (test_ratio_gen_limited failure)
- **Issue:** `int(20 * (0.60 / 0.40))` = `int(29.999...)` = 29, not 30; test expected 30
- **Fix:** Changed `int(...)` to `round(...)` for both ideal_judge and ideal_gen calculations
- **Files modified:** scripts/export_dataset.py
- **Commit:** 5fc31e7

## Test Results

- 7 new tests in tests/test_export.py, all passing
- Full suite: 46 tests passing (up from 39)

## Self-Check: PASSED

- scripts/phase3_cot.py: FOUND
- scripts/export_dataset.py: FOUND
- tests/test_export.py: FOUND
- Commit 5692cab (phase3_cot.py task): FOUND
- Commit 71baa48 (RED - test suite): FOUND
- Commit 5fc31e7 (GREEN - implementation): FOUND
