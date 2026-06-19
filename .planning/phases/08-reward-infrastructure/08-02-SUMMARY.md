---
phase: 08-reward-infrastructure
plan: "02"
subsystem: reward-pipeline
tags: [mo-grpo, verpo, normalization, dataclasses, tdd, wave-2, reward-math]
dependency_graph:
  requires: [08-01]
  provides:
    - scripts.reward_pipeline._mo_grpo_norm
    - scripts.reward_pipeline._EPSILON
    - scripts.reward_pipeline.WP_STANDARDS_CHECK_IDS
    - scripts.reward_pipeline.RewardBreakdown
    - scripts.reward_pipeline.RewardResult
    - scripts.reward_pipeline._verpo_group
    - scripts.reward_pipeline._extract_verifiable_signals
  affects: [08-03]
tech_stack:
  added: []
  patterns:
    - TDD (RED commit -> GREEN commit per task)
    - Population std (ddof=0) for within-group normalization
    - Lazy eval.rubric_scorer import AFTER os.environ.pop (Pitfall 6 guard)
    - Difficulty-weighted partial credit via inverse pass-rate
    - Polarity-aware pass detection (POSITIVE_CHECK_IDS / NEGATIVE_CHECK_IDS)
key_files:
  created: []
  modified:
    - scripts/reward_pipeline.py (248 lines added: _EPSILON, _mo_grpo_norm, WP_STANDARDS_CHECK_IDS, RewardBreakdown, RewardResult, _extract_verifiable_signals, _verpo_group)
    - tests/test_reward_pipeline.py (246 lines added: TestMOGRPONorm 4 tests, TestVeRPO 5 tests + _make_rubric(), TestBreakdownContract 5 tests)
decisions:
  - "VeRPO scoped to WP_STANDARDS_CHECK_IDS = D1_wpcs + D5_wp_api only (D-08-06 locked decision) — SQL/security/other dims covered by 30% judge"
  - "Population std (ddof=0) used in _mo_grpo_norm — consistent with group-level normalization; ddof=0 avoids sample-vs-population mismatch in unit-variance tests"
  - "eval.rubric_scorer import placed AFTER os.environ.pop (line 24) — rubric_scorer reads RUBRIC_USE_LLM_CHECKS at import, suppression must precede it"
  - "NEGATIVE check NOT firing = pass (polarity guard T-08-04) — _verpo_group resolves pass/fail via POSITIVE_CHECK_IDS/NEGATIVE_CHECK_IDS canonical sets, not raw triggered_checks"
  - "_verpo_group also implemented in Task 1 GREEN commit alongside RewardBreakdown — both live in the same wave-2 implementation block; VeRPO tests committed separately (Task 2 commit b002808)"
metrics:
  duration_seconds: 281
  completed_date: "2026-06-20"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
requirements_satisfied: [GRPO-03, GRPO-04]
---

# Phase 8 Plan 02: Reward Math Core (MO-GRPO + VeRPO + Dataclasses) Summary

**One-liner:** `_mo_grpo_norm` with epsilon floor (GRPO-03) + VeRPO difficulty-weighting scoped to D1_wpcs+D5_wp_api only (GRPO-04/D-08-06) + `RewardBreakdown`/`RewardResult` (scalar, breakdown_dict) contract (D-08-04).

## Tasks Completed

| # | Task | Commit | Key Output |
|---|------|--------|------------|
| 1 RED | MO-GRPO + breakdown contract tests | `2f792de` | TestMOGRPONorm (4 tests), TestBreakdownContract (5 tests) — all failing |
| 1 GREEN | MO-GRPO norm + dataclasses impl | `65ce943` | _mo_grpo_norm, _EPSILON, RewardBreakdown, RewardResult, WP_STANDARDS_CHECK_IDS, _extract_verifiable_signals, _verpo_group |
| 2 | VeRPO tests (polarity, scope, difficulty) | `b002808` | TestVeRPO (5 tests) — all passing against Task 1 impl |

## What Was Built

### `scripts/reward_pipeline.py` — additions

**`_EPSILON = 1e-8`**
Epsilon floor constant used in `_mo_grpo_norm` and `_verpo_group` to prevent NaN on zero-variance groups (T-08-03 / Pitfall 4).

**`_mo_grpo_norm(values: np.ndarray) -> np.ndarray`**
Within-group standardization: `(values - mu) / (sigma + _EPSILON)`. Uses `ddof=0` (population std). Zero-variance groups → all-zeros, no NaN. Single-element groups return finite value.

**`WP_STANDARDS_CHECK_IDS: frozenset`**
Check ids from `CHECK_DIMENSION_MAP` filtered to `dim in ("D1_wpcs", "D5_wp_api")` = 59 check ids (WPCS-* + WAPI-*). This is the ONLY scope for VeRPO (D-08-06 locked decision).

**`RewardBreakdown` dataclass**
Full D-08-04 contract with:
- Pre-norm fields: `phpcs_raw`, `verpo_raw`, `judge_raw`, `judge_offset_applied`, `security_fail`
- Post-norm fields: `phpcs_norm`, `verpo_norm`, `judge_norm`
- Composite: `composite_pre_gate`
- VeRPO per-check: `check_pass_rates`, `check_difficulties`
- Group stats: `group_size`, `group_phpcs_mean/std`, `group_judge_mean/std`
- Parse-failure metadata: `judge_parse_failure`, `judge_imputed_from_group` (default False)
- `to_dict()`: serializes to plain dict with all numpy types cast to Python float/int/bool for json.dumps compatibility (RLEV-02)

**`RewardResult` dataclass**
Simple `(scalar: float, breakdown: RewardBreakdown)` container.

**`_extract_verifiable_signals(php_code: str) -> RubricScore`**
Thin wrapper around `score_code()` — RUBRIC_USE_LLM_CHECKS already suppressed at module load, ensuring deterministic output.

**`_verpo_group(rubrics: list[RubricScore]) -> tuple[list[float], dict, dict]`**
VeRPO difficulty-weighted partial credit on WP_STANDARDS_CHECK_IDS only:
- Polarity-aware pass detection (T-08-04): POSITIVE check fired → pass; NEGATIVE check fired → fail; NEGATIVE NOT fired → pass
- `pass_rate_c = group_passes_c / G`; `difficulty_c = 1 - pass_rate_c`
- `verpo_i = sum(difficulty_c * pass_i_c) / (sum(difficulty_c) + _EPSILON)`
- Returns `(per_sample_verpo, check_pass_rates, check_difficulties)`

### `tests/test_reward_pipeline.py` — additions

**TestMOGRPONorm (4 tests, keyword: `mogrpo`)**
- `test_mogrpo_zero_variance_epsilon`: all-same group → no NaN, all-zeros
- `test_mogrpo_mean_centered`: normalized mean < 1e-6
- `test_mogrpo_unit_variance_after_norm`: population std ~1.0 (ddof=0 consistent)
- `test_mogrpo_group_of_one`: single element → finite result

**TestVeRPO (5 tests, keyword: `verpo`)**
- `test_verpo_scope_wp_standards_only`: SQL-N01 excluded, WAPI-* included (D-08-06)
- `test_verpo_rare_check_weights_more`: rarer check (difficulty=0.5) → higher pass score
- `test_verpo_all_pass_score`: all-pass group → finite non-negative scores, no NaN
- `test_verpo_all_fail_score`: all-fail group → finite scores, no NaN
- `test_verpo_positive_negative_polarity`: POSITIVE fired > NEGATIVE fired (T-08-04 guard)

**TestBreakdownContract (5 tests, keyword: `breakdown`)**
- `test_breakdown_has_pre_norm_fields`: phpcs_raw, verpo_raw, judge_raw, judge_offset_applied
- `test_breakdown_has_post_norm_fields`: phpcs_norm, verpo_norm, judge_norm, composite_pre_gate
- `test_breakdown_has_parse_failure_metadata`: judge_parse_failure, judge_imputed_from_group (defaults False)
- `test_breakdown_has_pre_post_norm`: full D-08-04 field set present
- `test_breakdown_serializable`: to_dict() round-trips through json.dumps/loads

## Verification Results

```
pytest tests/test_reward_pipeline.py -k "mogrpo or verpo or breakdown" -q
→ 14 passed

pytest tests/ -q
→ 387 passed (0 failures, 0 errors)
```

## Acceptance Criteria Checklist

- [x] `_mo_grpo_norm(np.ones(5)*42)` returns all-zeros with no NaN (epsilon floor proven)
- [x] RewardBreakdown has pre-norm fields: phpcs_raw, verpo_raw, judge_raw, judge_offset_applied
- [x] RewardBreakdown has post-norm fields: phpcs_norm, verpo_norm, judge_norm
- [x] `RewardBreakdown.to_dict()` output is `json.dumps`-serializable
- [x] `pytest tests/test_reward_pipeline.py -k "mogrpo or breakdown" -q` → 9 passed
- [x] WP_STANDARDS_CHECK_IDS derived from CHECK_DIMENSION_MAP filtered to {D1_wpcs, D5_wp_api} (59 ids; SQL-N01 excluded)
- [x] difficulty_c == 1 - pass_rate_c (rare check → higher weight)
- [x] NEGATIVE check firing scored as fail, POSITIVE as pass (test_verpo_positive_negative_polarity)
- [x] `pytest tests/test_reward_pipeline.py -k verpo -q` → 5 passed
- [x] `scripts/reward_pipeline.py` has `def _mo_grpo_norm` (line 127), min_lines ≥ 120 (actual: 331)

## Deviations from Plan

### Structural adjustments

**1. [Rule 2 - TDD Flow] _verpo_group implemented in Task 1 GREEN commit**
- **Found during:** Task 1 GREEN implementation
- **Issue:** The Task 1 action text (08-02-PLAN.md) mentions RewardBreakdown dataclasses only; _verpo_group is in Task 2's action. However, the rubric scorer imports (CHECK_DIMENSION_MAP, POSITIVE_CHECK_IDS, etc.) were brought in for the WP_STANDARDS_CHECK_IDS computation, and _verpo_group was a natural companion — keeping all the rubric-scorer-dependent code in one GREEN commit avoids partial import state.
- **Fix:** Implemented both `_verpo_group` and `_extract_verifiable_signals` in the Task 1 GREEN commit, then added the Task 2 VeRPO tests as a separate commit (b002808). The TDD commit structure (RED/GREEN per task) is preserved.
- **Impact:** No behavioral change. TestVeRPO tests were committed after _verpo_group was already green — consistent with plan intent.

**2. [Rule 2 - Scope Compliance] TestBreakdownContract moved from 08-03 stub note to 08-02**
- **Found during:** Plan review (Task 1 action explicitly includes TestBreakdownContract)
- **Issue:** 08-01-SUMMARY.md's "Known Stubs" table listed TestBreakdownContract as "for 08-03". The 08-02 plan's Task 1 action text explicitly says "Fill in / un-skip TestMOGRPONorm and TestBreakdownContract."
- **Fix:** Replaced all TestBreakdownContract stubs with real tests in 08-02 (correct scope).
- **Impact:** TestBreakdownContract is now green. TestSecurityGate and TestCompositeWeights remain skipped for 08-03.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced.
- `_extract_verifiable_signals` calls `score_code()` (existing trust boundary, static analysis only, deterministic per RUBRIC_USE_LLM_CHECKS suppression)
- `_verpo_group` is pure computation on RubricScore results
- No new external packages installed

## Known Stubs

| Stub | File | Reason |
|------|------|--------|
| TestSecurityGate | test_reward_pipeline.py | Requires _security_fail + CRITICAL_FLOOR_RULES inspection (08-03) |
| TestCompositeWeights | test_reward_pipeline.py | Requires compute_group_rewards + weight constants (08-03) |
| TestRewardPipelineIntegration | test_reward_pipeline_integration.py | Requires compute_group_rewards (08-03) |

## Self-Check: PASSED

Files modified:
- [x] scripts/reward_pipeline.py (331 lines, modified)
- [x] tests/test_reward_pipeline.py (417+ lines, modified)

Commits verified:
- [x] 2f792de test(08-02): RED - MO-GRPO norm + breakdown contract tests
- [x] 65ce943 feat(08-02): MO-GRPO norm + RewardBreakdown/RewardResult dataclasses
- [x] b002808 test(08-02): VeRPO polarity, scope, difficulty-weight tests (GRPO-04)

Functions present in scripts/reward_pipeline.py:
- [x] def _mo_grpo_norm (line 127)
- [x] _EPSILON = 1e-8 (line 124)
- [x] WP_STANDARDS_CHECK_IDS (line 117)
- [x] class RewardBreakdown (line 155)
- [x] class RewardResult (line 237)
- [x] def _extract_verifiable_signals (line 245)
- [x] def _verpo_group (line 258)
- [x] CHECK_DIMENSION_MAP|POSITIVE_CHECK_IDS pattern (key_links satisfied)
