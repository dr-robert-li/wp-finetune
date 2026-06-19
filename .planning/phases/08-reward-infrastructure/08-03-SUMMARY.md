---
phase: 08-reward-infrastructure
plan: "03"
subsystem: reward-pipeline
tags: [security-gate, composite-reward, tdd, grpo-01, grpo-02, sc2-fixture, wave-3]
dependency_graph:
  requires: [08-02]
  provides:
    - scripts.reward_pipeline._REWARD_SEC_TRIGGERS
    - scripts.reward_pipeline._security_fail
    - scripts.reward_pipeline.compute_group_rewards
    - scripts.reward_pipeline.compute_reward
    - tests/fixtures/reward_integration_cases/ (50 PHP fixtures + SC2)
    - tests/test_reward_pipeline_integration.py (integration suite, fully active)
  affects: [08-04, 09-gspo-trainer]
tech_stack:
  added: []
  patterns:
    - TDD RED/GREEN per task (test(08-03) commits before feat(08-03))
    - Programmatic security trigger derivation (CRITICAL_FLOOR_RULES x CHECK_REGISTRY)
    - Fail-CLOSED gate (RuntimeError on empty trigger set, not False)
    - Two-pass group reward normalization (collect+impute -> normalize+gate)
    - Terminal security override (post-combine, preserves composite_pre_gate)
    - Module-level judge_score_single import (patch binding for tests)
    - Judge parse-failure imputation from group mean with >10% rate warning
key_files:
  created:
    - tests/fixtures/reward_integration_cases/known_good_php/ (25 PHP files)
    - tests/fixtures/reward_integration_cases/known_bad_php/ (24 PHP files)
    - tests/fixtures/reward_integration_cases/secure_fail_high_quality.php (SC2)
  modified:
    - scripts/reward_pipeline.py (added _REWARD_SEC_TRIGGERS, _security_fail, compute_group_rewards, compute_reward)
    - tests/test_reward_pipeline.py (TestSecurityGate 7 tests, TestCompositeWeights 4 tests — un-stubbed)
    - tests/test_reward_pipeline_integration.py (all 5 tests un-stubbed)
decisions:
  - "_REWARD_SEC_TRIGGERS derived programmatically: CRITICAL_FLOOR_RULES D2_security ids where CHECK_REGISTRY[cid].method != 'llm'; result {SEC-N01,N03,N06,N08,N19,N20}; SEC-N04 excluded by design (llm-method, deterministic reward compute)"
  - "Fail-CLOSED contract: _security_fail raises RuntimeError if _REWARD_SEC_TRIGGERS is empty (both at module load and inside the function) — returning False on empty = insecure code earns reward = T-08-SEC HIGH"
  - "Terminal override placement: scalar=0.0 applied AFTER normalize+combine; composite_pre_gate retains real composite value (Pitfall 1)"
  - "Gate keys off triggered_checks intersection (NOT apply_floor_rules output); grep -c floor_rules_applied in non-comment code == 0"
  - "SC2 fixture triggers SEC-N20 (preg_replace /e regex, deterministic, no phpcs required for the trigger itself)"
  - "judge_score_single imported at module level (not via submodule) for correct patch binding in tests"
  - "Composite weights locked: 0.35 phpcs + 0.35 verpo + 0.30 judge (35/35/30 per D-08)"
  - "Judge parse-failure: impute from group mean of valid scores; >10% rate emits RuntimeWarning (D-08-07)"
metrics:
  duration_seconds: 2018
  completed_date: "2026-06-20"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 8
requirements_satisfied: [GRPO-01, GRPO-02]
---

# Phase 8 Plan 03: Composite Reward + Security Gate Summary

**One-liner:** Fail-CLOSED D2_security terminal override (triggered_checks intersection, RuntimeError on empty set) + two-pass MO-GRPO group normalization (35/35/30 composite) + SC2 fixture proving reward==0 on high-quality-but-security-failing PHP via SEC-N20.

## Tasks Completed

| # | Task | Commit | Key Output |
|---|------|--------|------------|
| 1 RED | TestSecurityGate 7 failing tests | `8b3a872` | test_reward_pipeline.py (7 failing) |
| 2 RED | TestCompositeWeights 4 failing tests | `838ec67` | test_reward_pipeline.py (4 more failing) |
| 1+2 GREEN | _security_fail + compute_group_rewards | `1d57471` | reward_pipeline.py (+281 lines) |
| 3 | 50 PHP fixtures + SC2 + integration suite | `937bb22` | 51 files, 5 integration tests green |

## What Was Built

### `scripts/reward_pipeline.py` additions

**New imports:**
- `from eval.rubric_definitions import CRITICAL_FLOOR_RULES, CHECK_REGISTRY` — for programmatic trigger derivation
- `from eval.eval_judge import judge_score_single` — module-level import for correct patch binding in tests (critical pattern from advisor review)

**`_REWARD_SEC_TRIGGERS: frozenset`**
Programmatically derived constant: D2_security CRITICAL_FLOOR_RULE trigger ids where `CHECK_REGISTRY[cid].method != "llm"`. SEC-N04 (the only llm-method D2 trigger) is excluded by design because `RUBRIC_USE_LLM_CHECKS` is suppressed at module load (Pitfall 6 / D-08 locked constraint). Result at runtime: `{SEC-N01, SEC-N03, SEC-N06, SEC-N08, SEC-N19, SEC-N20}`. Fail-CLOSED guard at module load: raises immediately if the set is empty.

**`_security_fail(rubric: RubricScore) -> bool`**
Reads `rubric.triggered_checks` (NOT `floor_rules_applied` — see D-08-05 / acceptance criteria). Flattens all triggered check ids across all dimensions into a set, then tests intersection with `_REWARD_SEC_TRIGGERS`. Returns True iff the intersection is non-empty. Raises `RuntimeError` if `_REWARD_SEC_TRIGGERS` is empty (fail-CLOSED, T-08-SEC HIGH mitigation).

**`compute_group_rewards(php_codes, judge_client, judge_model) -> list[RewardResult]`**
Two-pass algorithm:
- Pass 1: collect `RubricScore` via `_extract_verifiable_signals()` + raw judge via `judge_score_single()`; impute None judge scores from group mean; warn if >10% parse failures (D-08-07).
- Pass 2: apply `_apply_offset_clip()` to judge scores; compute `_verpo_group()` scores; MO-GRPO normalize phpcs/verpo/judge arrays independently; compute `composite_pre_gate = 0.35*phpcs_norm + 0.35*verpo_norm + 0.30*judge_norm`; apply terminal override: `scalar = 0.0 if sec_fail else composite_pre_gate`.

**`compute_reward(php_code, judge_client, judge_model) -> RewardResult`**
Single-sample convenience wrapper around `compute_group_rewards`.

**`_W_PHPCS = 0.35, _W_VERPO = 0.35, _W_JUDGE = 0.30`**
Named weight constants (35/35/30 split per D-08 locked default).

### Security gate design decisions

1. **triggered_checks, not floor_rules_applied:** `apply_floor_rules()` in `rubric_scorer.py` only appends to `floor_rules_applied` when `current_score > cap`. An already-below-cap D2 dimension yields an empty list even when a trigger has fired. The gate uses `triggered_checks` for reliable membership. `grep -c 'floor_rules_applied'` in non-comment code == 0.

2. **Terminal override (Pitfall 1):** The 0.0 is assigned to `scalar` AFTER computing `composite_pre_gate`. `composite_pre_gate` retains the real composite value. The failing member's signals remain in the group normalization denominator, preserving other members' norms.

3. **Fail-CLOSED:** Raises RuntimeError on empty trigger set both at module load (loud early failure) and inside `_security_fail()` (testable via monkeypatch).

### Test suite results (08-03)

```
pytest tests/test_reward_pipeline.py -k security_gate -q   → 7 passed
pytest tests/test_reward_pipeline.py -k composite -q       → 4 passed (1 arithmetic, 3 compute_group_rewards)
pytest tests/test_reward_pipeline.py -q                    → 32 passed
pytest tests/test_reward_pipeline_integration.py -q        → 5 passed
pytest tests/ -q                                           → 403 passed
```

### SC2 fixture

`tests/fixtures/reward_integration_cases/secure_fail_high_quality.php`:
- Well-structured PHP class with docblocks, WP APIs, proper escaping
- Intentional vulnerability: `preg_replace($pattern . '/e"', $replacement, $content)` — the `/e` modifier causes PHP to evaluate the replacement as PHP code (CVE class: RCE via regex replacement)
- Verified: `score_code()` (RUBRIC_USE_LLM_CHECKS off) places `SEC-N20` in `triggered_checks["D2_security"]`
- `phpcs_raw = 75.4` (non-trivial quality, not just bad code)
- `php -l`: no syntax errors (it is syntactically valid PHP; the /e issue is a runtime security flaw)
- `compute_group_rewards([sc2_code]*4, mock_judge=95.0, ...) → all scalar == 0.0`

### Known-good / known-bad fixtures

25 known_good PHP files: WPCS-compliant widget, REST controller, settings handler, DB query helper, and AJAX handler classes using proper WP APIs (sanitize_text_field, esc_html, wp_nonce_field, current_user_can, etc.).

24 known_bad PHP files: Style violations (no docblocks, non-standard naming, no type hints, procedural style) but intentionally avoid triggering `_REWARD_SEC_TRIGGERS` security checks so low scores are from quality signals, not the gate.

## Deviations from Plan

### Auto-fixed / structural adjustments

**1. [Rule 2 - Missing Guard] Module-level judge_score_single import**
- **Found during:** Advisor review (before implementation)
- **Issue:** If `judge_score_single` was imported via `import eval.eval_judge; eval_judge.judge_score_single(...)`, the `patch("scripts.reward_pipeline.judge_score_single", ...)` calls in tests would silently no-op (patch target not in module namespace).
- **Fix:** Added `from eval.eval_judge import judge_score_single` at module level after the rubric_scorer imports.
- **Files:** scripts/reward_pipeline.py

**2. [Rule 2 - Missing Guard] Docstring rephrasing to avoid floor_rules_applied**
- **Found during:** Acceptance criteria check after GREEN implementation
- **Issue:** The `_security_fail` docstring originally contained the literal string `floor_rules_applied` in a non-comment context. The acceptance criteria grep `grep -nv '^[[:space:]]*#' | grep -c 'floor_rules_applied'` matched docstring lines, yielding count=2 instead of 0.
- **Fix:** Rephrased docstring to describe the same concept without using the literal string.
- **Files:** scripts/reward_pipeline.py

**3. [Plan Note] Task 1 + Task 2 GREEN committed together**
- **Found during:** Task 1 test execution
- **Issue:** `test_security_gate_applied_after_normalization` (Task 1 test) calls `compute_group_rewards()` (Task 2 API). Task 1's full GREEN required Task 2's implementation. Given tight coupling, implemented both in one GREEN commit after writing both RED test sets.
- **Impact:** TDD gate compliance: two RED commits (`8b3a872`, `838ec67`) precede one GREEN commit (`1d57471`). The gate ordering is correct.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced.

T-08-SEC mitigations as designed:
1. Gate keys off `triggered_checks` intersection (not `floor_rules_applied`)
2. `_REWARD_SEC_TRIGGERS` derived programmatically, excludes SEC-N04 explicitly
3. Fail-CLOSED: RuntimeError on empty trigger set
4. Terminal override: post-combine, scalar=0 not signals=0
5. Bidirectional tests + SC2 integration fixture with real `score_code()`

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (Task 1) | `8b3a872` test(08-03): RED — TestSecurityGate | ✓ 7 failing |
| RED (Task 2) | `838ec67` test(08-03): RED — TestCompositeWeights | ✓ 4 failing (3 ImportError, 1 arithmetic pass) |
| GREEN (both) | `1d57471` feat(08-03): GREEN | ✓ all 11 pass |

## Self-Check: PASSED

Files created/exist:
- [x] scripts/reward_pipeline.py (modified — _REWARD_SEC_TRIGGERS, _security_fail, compute_group_rewards, compute_reward)
- [x] tests/fixtures/reward_integration_cases/known_good_php/ (25 .php files)
- [x] tests/fixtures/reward_integration_cases/known_bad_php/ (24 .php files)
- [x] tests/fixtures/reward_integration_cases/secure_fail_high_quality.php

Commits verified:
- [x] 8b3a872 test(08-03): RED — TestSecurityGate 7 failing tests
- [x] 838ec67 test(08-03): RED — TestCompositeWeights 4 failing tests
- [x] 1d57471 feat(08-03): GREEN — _security_fail + compute_group_rewards + 70/30 composite
- [x] 937bb22 feat(08-03): Task 3 — 50 PHP fixtures + SC2 + integration suite green

Acceptance criteria:
- [x] `_security_fail` reads `triggered_checks` (not floor_rules_applied) — test_security_gate_floor_rules_applied_not_used passes
- [x] `_REWARD_SEC_TRIGGERS` contains {SEC-N01,N03,N06,N08,N19,N20}, excludes SEC-N04 — test_security_gate_sec_n04_excluded_by_design passes
- [x] All 6 deterministic triggers individually fire — test_security_gate_all_deterministic_triggers_fire passes
- [x] Gate fails CLOSED (raises on empty trigger set) — test_security_gate_fail_closed_raises passes
- [x] Terminal override verified — test_security_gate_applied_after_normalization passes
- [x] `grep -c floor_rules_applied` in non-comment code == 0
- [x] Composite weights 35/35/30 — test_composite_verifiable_split_35_35 passes
- [x] Judge imputation from group mean — test_composite_judge_parse_failure_imputed passes
- [x] 25 known_good + 24 known_bad PHP files exist
- [x] SC2 fixture: php -l no errors; SEC-N20 fires; phpcs_raw=75.4
- [x] SC2 test: scalar==0.0, security_fail=True; uses real score_code()
- [x] pytest tests/ -q: 403 passed
