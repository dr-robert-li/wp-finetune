---
phase: quick
plan: 260403-utp
subsystem: eval-tests
tags: [tests, eval, rubric-scorer, unit-tests]
dependency_graph:
  requires: [eval/eval_gen.py, eval/eval_judge.py, eval/eval_gate.py, eval/rubric_scorer.py]
  provides: [passing-unit-tests-for-eval-modules]
  affects: [ci, test coverage]
tech_stack:
  added: []
  patterns: [mock-dataclass-instances, helper-builders-for-full-dicts]
key_files:
  created: []
  modified:
    - tests/test_eval_gen.py
    - tests/test_eval_judge.py
    - tests/test_eval_gate.py
decisions:
  - Mock RubricScore directly via dataclass constructor to avoid subprocess calls in unit tests
  - Use _full_results/_full_thresholds helpers in gate tests to prevent KeyError from missing dict keys
metrics:
  duration_min: 8
  completed_date: "2026-04-03T12:17:37Z"
  tasks_completed: 3
  files_modified: 3
---

# Quick Task 260403-utp: Fix Stale Eval Tests to Match Current API Summary

**One-liner:** Rewrote 3 Wave-0 eval test files to match rubric-scorer-based API, removing 4 deleted functions and fixing check_gates return type from list[str] to list[dict].

## What Was Done

Three test files were written spec-first before the eval modules were implemented. When the modules were refactored (phpcs-based scoring replaced by rubric_scorer, eval_gate return type changed), the tests were never updated. All 3 failed at import time or with wrong assertions.

### Task 1: test_eval_gen.py (commit aa7cfc8)

**Old:** Imported `run_phpcs`, `compute_pass_rate`, `classify_security` — all removed from eval_gen during rubric refactor.

**New:**
- `test_extract_php_code`: covers `php fenced block, generic fenced block, raw PHP text
- `test_compute_summary_basic`: mocks 5 RubricScore objects directly, validates all summary keys (total, overall_mean, overall_median, grade_distribution, per_dimension, floor_rules, phpcs_pass_rate, security_pass_rate) and computed values
- `test_compute_summary_empty`: asserts `{"total": 0}` for empty list

**Key technique:** RubricScore dataclass instances constructed directly with test values — avoids phpcs/phpstan subprocess calls.

### Task 2: test_eval_judge.py (commit 00d675f)

**Old:** Imported `invert_phpcs_errors` (removed), had `test_score_inversion` (dead test).

**New:**
- Keep `test_spearman_computation` unchanged (scipy direct, still valid)
- Keep `test_judge_output_parsing` with minor fix: assert malformed response returns `None` (not "None or dict")
- Remove `test_score_inversion`
- Add `test_safe_spearman_edge_cases`: covers <2 pairs, all-identical xs, all-identical ys, valid perfect-correlation pairs

### Task 3: test_eval_gate.py (commit 49ec4b6)

**Old:** Used `check_gates(results, thresholds)` expecting `(bool, list[str])` return — actual return is `(bool, list[dict])`. Thresholds dict missing all new keys. Results dict missing required keys.

**New:**
- `_full_thresholds()` and `_full_results()` helpers build complete dicts from `_FALLBACK_THRESHOLDS` with controllable overrides
- All 5 tests updated to use `gate_rows` (list of dicts) with assertions on `row["gate"]`, `row["passed"]`
- `test_gate_reads_thresholds_from_config`: config YAML includes all threshold keys; results include all required keys; assertions check gate_rows not failures strings

## Verification

```
pytest tests/test_eval_gen.py tests/test_eval_judge.py tests/test_eval_gate.py -v
11 passed in X.Xs
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- tests/test_eval_gen.py: EXISTS
- tests/test_eval_judge.py: EXISTS
- tests/test_eval_gate.py: EXISTS
- commit aa7cfc8: EXISTS (test_eval_gen.py rewrite)
- commit 00d675f: EXISTS (test_eval_judge.py rewrite)
- commit 49ec4b6: EXISTS (test_eval_gate.py rewrite)
- All 11 tests pass: VERIFIED
