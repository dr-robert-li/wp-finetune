---
phase: 08-reward-infrastructure
reviewed: 2026-06-20T02:09:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - scripts/reward_pipeline.py
  - eval/eval_judge.py
  - scripts/build_antihack_set.py
  - tests/conftest.py
  - tests/test_reward_pipeline.py
  - tests/test_reward_pipeline_integration.py
  - tests/test_antihack.py
findings:
  critical: 3
  warning: 7
  info: 2
  total: 12
status: clean
---

# Phase 08: Code Review Report

**Reviewed:** 2026-06-20T02:09:00Z · **Depth:** standard · **Files:** 7 · **Status:** issues_found

## Summary

Three critical defects. Most severe is a confirmed fail-open in the security hard gate when phpcs is absent — five of six deterministic D2_security triggers silently drop, letting SQL-injection and other insecure code earn full reward. Second: `_RECALIB_PATH` relative path causes import-time `FileNotFoundError` from any non-root CWD. Third: `score_and_gate` compares MO-GRPO within-group z-scores cross-group, making the live CI gate semantically meaningless.

---

## Critical Issues

### CR-01: Security Hard Gate Fails Open When phpcs Is Absent
**File:** `eval/rubric_scorer.py:224-226` (called from `scripts/reward_pipeline.py:158`)
Five of six `_REWARD_SEC_TRIGGERS` are `method="phpcs"` (SEC-N01/N03/N06/N08/N19). When phpcs is not installed, `run_phpcs()` returns `_unavailable`; `map_phpcs_to_checks()` skips it; those IDs never enter `triggered_checks`; `_security_fail()` intersection is empty → returns `False`. Insecure code earns full non-zero reward. Only SEC-N20 (regex) still fires, satisfying the non-empty load guard while the gate is degraded. Fail-open on the D-08-05 CRITICAL gate (T-08-SEC).
**Fix:** assert phpcs availability at reward-pipeline startup (fail-closed); `shutil.which("phpcs")` is None → raise, with an explicit `REWARD_SKIP_PHPCS_ASSERT=1` acknowledgement escape hatch.

### CR-02: `_RECALIB_PATH` Relative Path Breaks Import from Non-Root CWD
**File:** `scripts/reward_pipeline.py:42,69`
Module-level `_SCORE_OFFSET = _load_score_offset()` resolves a relative path against cwd → `FileNotFoundError` when imported from any non-root dir (trainer cd, pytest from scripts/, distributed worker).
**Fix:** `_RECALIB_PATH = Path(__file__).resolve().parent.parent / "output/eval_reasoning_v4_winner/judge_recalibration.json"`.

### CR-03: `score_and_gate` Cross-Group CI Comparison Invalid
**File:** `scripts/build_antihack_set.py:579-580`
Live path normalizes perturbed and clean groups separately via MO-GRPO → both scalar means ≈ 0 → CIs overlap regardless of quality → `hi_perturbed < lo_clean` ~never holds → anti-hack set empirically unvalidated in live path.
**Fix:** score perturbed+clean in ONE combined `compute_group_rewards` call so normalization spans both, then split scalars by membership for the CI comparison.

---

## Warnings

### WR-01: `judge_score_single` Crashes on None API Content
**File:** `eval/eval_judge.py:248` — `resp.choices[0].message.content` may be None → `None.strip()` AttributeError aborts the group. Fix: `... .content or ""`.

### WR-02: judge exceptions abort entire group (violates D-08-07)
**File:** `scripts/reward_pipeline.py:474` — wrap `judge_score_single` in try/except → `None` so group-mean imputation runs.

### WR-03: `compute_reward` (single sample) always returns scalar=0.0
**File:** `scripts/reward_pipeline.py:591` — MO-GRPO G=1 degeneracy. Add `warnings.warn` pointing callers to `compute_group_rewards`.

### WR-04: `compute_axis_gate` ZeroDivisionError on empty lists
**File:** `scripts/build_antihack_set.py:421` — guard empty perturbed/clean reward lists with a ValueError.

### WR-05: `test_composite_weights_sum_to_one` tautological
**File:** `tests/test_reward_pipeline.py:522-528` — hardcodes literals; import `_W_PHPCS/_W_VERPO/_W_JUDGE` and assert their sum.

### WR-06: `test_composite_judge_component_weight` asserts nothing
**File:** `tests/test_reward_pipeline.py:530-548` — only checks hasattr; vary judge score and assert proportional composite change.

### WR-07: dead unreachable `else` in `main()`
**File:** `scripts/build_antihack_set.py:750-757` — `not args.score_and_gate` makes elif unconditionally true; restructure branches.

---

## Info
- IN-01: dead `import logging` inside `compute_group_rewards` (`scripts/reward_pipeline.py:459`).
- IN-02: redundant `import json as _json` inside `main()` (`scripts/build_antihack_set.py:761`).

---
_Reviewer: gsd-code-reviewer · depth standard_

## Resolution

All 12 findings fixed. Final pytest: **424 passed** (was 421; +3 new CR-01 phpcs-assert tests).

| Finding | Fix commit | Notes |
|---------|-----------|-------|
| CR-01 | `fix(08): CR-01 CR-02 WR-02 WR-03 IN-01 reward_pipeline fixes` + `fix(08): WR-05 WR-06 + CR-01 unit tests` | phpcs fail-CLOSED assertion in `compute_group_rewards`; escape hatch `REWARD_SKIP_PHPCS_ASSERT=1`; 3 unit tests added |
| CR-02 | `fix(08): CR-01 CR-02 WR-02 WR-03 IN-01 reward_pipeline fixes` | `_RECALIB_PATH` anchored to `Path(__file__).resolve().parent.parent` |
| CR-03 | `fix(08): CR-03 WR-04 WR-07 IN-02 build_antihack_set fixes` | `score_and_gate` now uses ONE combined `compute_group_rewards` call; splits `[:n_perturbed]` / `[n_perturbed:]` after |
| WR-01 | `fix(08): WR-01 judge_score_single None content AttributeError` | `resp.choices[0].message.content or ""` |
| WR-02 | `fix(08): CR-01 CR-02 WR-02 WR-03 IN-01 reward_pipeline fixes` | `judge_score_single` call wrapped in `try/except → None` |
| WR-03 | `fix(08): CR-01 CR-02 WR-02 WR-03 IN-01 reward_pipeline fixes` | `warnings.warn` added in `compute_reward` for G=1 degeneracy |
| WR-04 | `fix(08): CR-03 WR-04 WR-07 IN-02 build_antihack_set fixes` | `ValueError` on empty reward lists in `compute_axis_gate` |
| WR-05 | `fix(08): WR-05 WR-06 + CR-01 unit tests` | Test imports `_W_PHPCS/_W_VERPO/_W_JUDGE` and asserts sum |
| WR-06 | `fix(08): WR-05 WR-06 + CR-01 unit tests` | Test verifies 35/35/30 formula holds and judge_norm varies |
| WR-07 | `fix(08): CR-03 WR-04 WR-07 IN-02 build_antihack_set fixes` | Replaced unreachable `else` with simple `else` branch |
| IN-01 | `fix(08): CR-01 CR-02 WR-02 WR-03 IN-01 reward_pipeline fixes` | Removed dead `import logging` from inside `compute_group_rewards` |
| IN-02 | `fix(08): CR-03 WR-04 WR-07 IN-02 build_antihack_set fixes` | Removed `import json as _json` inside `main()`; uses module-level `json` |
