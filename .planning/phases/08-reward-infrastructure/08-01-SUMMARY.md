---
phase: 08-reward-infrastructure
plan: "01"
subsystem: reward-pipeline
tags: [judge-wrapper, recalibration, test-scaffolding, tdd, wave-0]
dependency_graph:
  requires: [07-02]
  provides:
    - eval.eval_judge.judge_score_single
    - scripts.reward_pipeline._load_score_offset
    - scripts.reward_pipeline._SCORE_OFFSET
    - scripts.reward_pipeline._apply_offset_clip
    - tests/conftest.py (shared fixtures)
    - tests/test_reward_pipeline.py (unit stubs)
    - tests/test_reward_pipeline_integration.py (integration harness stub)
    - tests/test_antihack.py (CI-gate stub)
  affects: [08-02, 08-03, 08-04]
tech_stack:
  added: []
  patterns:
    - TDD (RED in scaffolding, GREEN in implementation commit)
    - Injectable path parameter for testable module-level constants
    - Lazy imports in test bodies to prevent collection failures on absent modules
    - -k keyword embedding in method names (offset_loader, judge_single, mogrpo, etc.)
key_files:
  created:
    - eval/eval_judge.py (function added: judge_score_single)
    - scripts/reward_pipeline.py
    - tests/conftest.py
    - tests/test_reward_pipeline.py
    - tests/test_reward_pipeline_integration.py
    - tests/test_antihack.py
  modified:
    - eval/eval_judge.py
decisions:
  - "Lazy imports inside test bodies: no project-module top-level imports in test files or conftest to prevent collection errors when reward_pipeline.py is absent"
  - "-k keyword embedded in method names (test_judge_single_*, test_offset_loader_*) rather than class names so pytest -k filters select correctly"
  - "3.58 literal appears only in a comment; all code reads score_offset from JSON artifact via injectable _load_score_offset(path)"
  - "TestJudgeWrapper RED tests were committed as part of Task 1 scaffolding (structural pattern); GREEN implemented in Task 2"
metrics:
  duration_seconds: 312
  completed_date: "2026-06-20"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 6
requirements_satisfied: [GRPO-01]
---

# Phase 8 Plan 01: Wave-0 Scaffolding + Judge Wrapper + Offset Loader Summary

**One-liner:** RC-A-guarded `judge_score_single()` + injectable `_load_score_offset()` + full Wave-0 pytest scaffolding (37 stubs, 10 active tests green).

## Tasks Completed

| # | Task | Commit | Key Output |
|---|------|--------|------------|
| 1 | Wave-0 test scaffolding + shared fixtures | `55964a8` | conftest.py, test_reward_pipeline.py, test_reward_pipeline_integration.py, test_antihack.py |
| 2 | Extract judge_score_single() with RC-A guard | `8e32e46` | eval/eval_judge.py (46 lines added) |
| 3 | Injectable recalibration-offset loader | `9be04f3` | scripts/reward_pipeline.py (88 lines) |

## What Was Built

### `eval/eval_judge.py` — `judge_score_single()`
New public function placed after `parse_judge_response` (line 210). Builds a `<wp_judge>` user message, calls `_judge_create(client, model=model, messages=..., max_tokens=max_tokens, temperature=0.0)` — NOT `client.chat.completions.create` directly — preserving the RC-A `enable_thinking=False` guard. Runs `parse_judge_response` on the response content, returns `float(overall_score)` when present and numeric, else `None`.

### `scripts/reward_pipeline.py` — offset loader + clip
- `_RECALIB_PATH`: path constant pointing to `output/eval_reasoning_v4_winner/judge_recalibration.json`
- `_load_score_offset(path=_RECALIB_PATH) -> float`: reads `data["score_offset"]` from JSON; `path` is injectable for tests
- `_SCORE_OFFSET: float`: module-level singleton loaded at import
- `_apply_offset_clip(raw_judge: float) -> float`: applies `_SCORE_OFFSET` then clips to `[0.0, 100.0]` (order: offset → clip per D-08-02)
- `os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)` at module top (deterministic reward compute)
- The literal `3.58` appears ONLY in a `#`-prefixed comment; zero occurrences in non-comment code

### `tests/conftest.py` — shared fixtures
- `recalib_json(tmp_path_factory)`: session-scoped; writes synthetic `judge_recalibration.json` with `score_offset=3.58` to tmp dir
- `php_fixture_dir()`: session-scoped; returns `tests/fixtures/reward_integration_cases` path
- `mock_judge_client()`: function-scoped; MagicMock client returning `{"overall_score": 75}`
- `sample_rollout_group()`: function-scoped; list of 4 minimal PHP strings
- All project-module imports lazy (inside fixture bodies) to prevent collection errors

### `tests/test_reward_pipeline.py` — unit tests
- **Active (7 tests):** `TestJudgeWrapper` (3: judge_single keyword), `TestOffsetApply` (4: offset_loader keyword)
- **Stubbed with pytest.skip:** `TestMOGRPONorm` (mogrpo), `TestVeRPO` (verpo), `TestSecurityGate` (security_gate), `TestCompositeWeights` (composite), `TestBreakdownContract` — for 08-02/03
- All reward_pipeline imports inside test method bodies (not at module level)

### `tests/test_reward_pipeline_integration.py` — integration harness stub
- `FIXTURE_DIR`, `KNOWN_GOOD_DIR`, `KNOWN_BAD_DIR`, `SC2_FILE` constants defined
- `TestRewardPipelineIntegration` class (4 stubs, skip until 08-02)
- `test_sc2_security_fail_scores_zero` stub (skip until 08-03, body preserved in comments)

### `tests/test_antihack.py` — CI-aware gate test stub
- **Active (4 tests):** `test_bootstrap_ci_importable`, `TestAntihackCIGate.test_perturbed_below_clean_passes`, `test_ci_aware_not_bare_point`, `test_all_axes_report_four_ci_bounds`
- **Stubbed:** `TestAntihackCaseCoverage` (3 stubs), `TestAntihackAxisGate` (3 stubs) — for 08-04

## Verification Results

```
pytest tests/test_reward_pipeline.py -k "judge_single or offset_loader" -q
→ 7 passed

pytest tests/ --collect-only -q
→ 399 tests collected (0 errors)
```

## Acceptance Criteria Checklist

- [x] `pytest --collect-only` succeeds for all three new test files (no import/collection errors)
- [x] `tests/conftest.py` defines: recalib_json, php_fixture_dir, mock_judge_client, sample_rollout_group
- [x] `python -c "from scripts.compute_concentration import bootstrap_ci"` exits 0
- [x] Test classes present: TestJudgeWrapper, TestOffsetApply, TestMOGRPONorm, TestVeRPO, TestSecurityGate, TestCompositeWeights, TestBreakdownContract
- [x] `grep -c 'def judge_score_single' eval/eval_judge.py` == 1
- [x] `pytest tests/test_reward_pipeline.py -k judge_single -q` → 3 passed
- [x] `grep` of non-comment lines for `3\.58` in reward_pipeline.py → 0
- [x] `_load_score_offset` accepts injectable `path` parameter
- [x] `_apply_offset_clip` applies offset then clips to [0,100]
- [x] `pytest tests/test_reward_pipeline.py -k offset_loader -q` → 4 passed

## Deviations from Plan

### Auto-fixed / structural adjustments

**1. [Rule 2 - Missing Guard] Lazy imports in test bodies (not plan-prescribed)**
- **Found during:** Task 1 design (pre-implementation advisor review)
- **Issue:** PATTERNS.md shows `from scripts.reward_pipeline import (...)` at module level in test_reward_pipeline.py. At Task 1, reward_pipeline.py doesn't exist; even after Task 3, only 3 symbols exist (8+ needed). Module-level import would break collect-only for entire suite.
- **Fix:** All `scripts.reward_pipeline` and `eval.eval_judge.judge_score_single` imports moved inside test method bodies. conftest.py has no project-module top-level imports.
- **Impact:** Tests still test the same contracts; importability fixed.

**2. [Rule 2 - Missing Guard] -k keyword embedded in method names**
- **Found during:** Task 1 design (advisor review)
- **Issue:** Plan says "use keyword conventions from 08-VALIDATION.md so `-k offset_loader`, `-k judge_single` select correctly." The behavior block names (e.g. `test_offset_read_from_json`) don't contain the required substrings. Pytest `-k offset_loader` would select 0 tests → exit 5 (no tests ran).
- **Fix:** All TestOffsetApply methods prefixed with `test_offset_loader_*`; TestJudgeWrapper methods prefixed with `test_judge_single_*`. Class names unchanged.

**3. [Rule 2 - Missing Fixture] All 4 conftest fixtures defined**
- **Found during:** Task 1 implementation
- **Issue:** PATTERNS.md only shows 2 fixtures (recalib_json, php_fixture_dir). Task 1 action text requires all 4 (also mock_judge_client, sample_rollout_group).
- **Fix:** Added mock_judge_client (MagicMock returning JSON overall_score) and sample_rollout_group (4 PHP strings).

## Known Stubs

| Stub | File | Reason |
|------|------|--------|
| TestMOGRPONorm, TestVeRPO | test_reward_pipeline.py | Implemented in 08-02 (requires _mo_grpo_norm, VeRPO logic) |
| TestSecurityGate, TestCompositeWeights, TestBreakdownContract | test_reward_pipeline.py | Implemented in 08-03 (requires _security_fail, compute_group_rewards) |
| TestRewardPipelineIntegration, test_sc2_security_fail_scores_zero | test_reward_pipeline_integration.py | Implemented in 08-02/03 (requires fixtures + compute_group_rewards) |
| TestAntihackCaseCoverage, TestAntihackAxisGate | test_antihack.py | Implemented in 08-04 (requires adversarial case set) |

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced.
- `judge_score_single` uses `_judge_create` (existing trust boundary, already in threat model as T-08-01)
- `_load_score_offset` reads a local read-only JSON (T-08-02, accepted)
- No new external packages installed

## Self-Check: PASSED

Files created/exist:
- [x] eval/eval_judge.py (modified — `judge_score_single` at line 210)
- [x] scripts/reward_pipeline.py (created)
- [x] tests/conftest.py (created)
- [x] tests/test_reward_pipeline.py (created)
- [x] tests/test_reward_pipeline_integration.py (created)
- [x] tests/test_antihack.py (created)

Commits verified:
- [x] 55964a8 feat(08-01): Wave-0 test scaffolding
- [x] 8e32e46 feat(08-01): judge_score_single()
- [x] 9be04f3 feat(08-01): injectable offset loader
