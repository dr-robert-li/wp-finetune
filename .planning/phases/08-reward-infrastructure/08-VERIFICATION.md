---
phase: 08-reward-infrastructure
verified: 2026-06-20T02:30:00Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run `python -m scripts.build_antihack_set --source-jsonl output/eval_reasoning_v4_winner/eval_gen_results.jsonl --output-dir output/antihack_validation/ --cases-per-axis 15 --score-and-gate` with vLLM judge endpoint live"
    expected: "acceptance_report.json written with report_type=live_scored, all_axes_pass=true, hi_perturbed < lo_clean for all three axes on real model outputs"
    why_human: "Requires vLLM serving the frozen wp_judge checkpoint. The current acceptance_report.json is fixture_backed (synthetic np.random.seed(2024) arrays). The CI gate logic is verified; the 45-case live scoring is the empirical validation of the full D-11 behavioral claim."
---

# Phase 8: Reward Infrastructure Verification Report

**Phase Goal:** A composite reward pipeline is built and validated end-to-end before any RL training begins — PHPCS anchor, security hard gate, VeRPO partial credit, MO-GRPO normalization, and anti-hack eval set all verified independently

**Verified:** 2026-06-20T02:30:00Z
**Status:** human_needed (424 tests pass; one human item — live anti-hack scoring against vLLM judge)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Composite reward = 70% verifiable (35 PHPCS / 35 VeRPO) + 30% frozen wp_judge; scalar per generation; `(scalar, breakdown_dict)` contract | VERIFIED | `_W_PHPCS=0.35, _W_VERPO=0.35, _W_JUDGE=0.30` at lines 417-419; `composite_pre_gate = _W_PHPCS * phpcs_norm + _W_VERPO * verpo_norm + _W_JUDGE * judge_norm` at line 572-576; `RewardResult(scalar, breakdown)` dataclass confirmed |
| 2 | Security scan failure → reward=0 TERMINAL OVERRIDE after normalize+combine, FAIL-CLOSED; SC2 secure-fail-but-high-quality test → reward exactly 0 | VERIFIED | Terminal override at line 582: `final_scalar = 0.0 if sec_fail else composite_pre_gate`; `_security_fail` reads `triggered_checks` (line 190-194); `_REWARD_SEC_TRIGGERS = {SEC-N01, N03, N06, N08, N19, N20}` (6 ids, all phpcs/regex); fail-CLOSED double guard at module load (line 151) and inside `_security_fail` (line 184); `test_sc2_security_fail_scores_zero` PASSED with phpcs live |
| 3 | MO-GRPO within-group variance normalization with epsilon floor; no single-signal dominance | VERIFIED | `_EPSILON = 1e-8` at line 202; `_mo_grpo_norm`: `(values - mu) / (sigma + _EPSILON)` at line 220; three independent signals normalized independently (lines 553-555) |
| 4 | VeRPO difficulty-weighted partial credit scoped to WP-standards subset (D-08-06); difficulty = 1-pass_rate | VERIFIED | `WP_STANDARDS_CHECK_IDS = frozenset(cid for cid, dim in CHECK_DIMENSION_MAP.items() if dim in ("D1_wpcs", "D5_wp_api"))` = 59 check IDs (runtime confirmed); `check_difficulties = {cid: 1.0 - rate ...}` at line 390-392; VeRPO formula at line 406 |
| 5 | Anti-hack set (D-11): 3-axis perturb-real construction + CI-aware gate hi_perturbed < lo_clean via bootstrap_ci; fixture-backed construction + CI gate logic verified | VERIFIED (with live follow-up) | `perturb_verbose_padding`, `perturb_template_critique_collapse`, `perturb_self_preference_swap` all implemented; `compute_axis_gate` uses `bootstrap_ci` with gate = `bool(hi_p < lo_c)` (line 423); `acceptance_report.json` exists with all 4 CI bounds per axis + `gate_pass`; source filter `>= 65.0` (Pitfall 7) at line 101; CR-03 fix: ONE combined `compute_group_rewards` call in `score_and_gate` (line 594-598) |

**Score:** 5/5 truths verified

---

### Additional Specific Items Verified

| Item | Status | Evidence |
|------|--------|----------|
| +3.58 offset loaded from artifact, never hardcoded (CR-02) | VERIFIED | `_RECALIB_PATH = Path(__file__).resolve().parent.parent / "output/eval_reasoning_v4_winner/judge_recalibration.json"` (line 44); only occurrence of `3.58` in file is in a comment (line 40) |
| CR-02 path fix: `Path(__file__).resolve().parent.parent` | VERIFIED | Line 44 confirmed |
| CR-01 phpcs assertion in `compute_group_rewards` | VERIFIED | Lines 466-479: `shutil.which("phpcs") is None → RuntimeError`; escape hatch `REWARD_SKIP_PHPCS_ASSERT=1` present; 3 dedicated tests pass: `TestPhpcsAvailabilityAssertion::test_phpcs_assert_raises_when_phpcs_absent`, `test_phpcs_assert_no_raise_when_phpcs_present`, `test_phpcs_assert_escape_hatch_env` |
| Judge parse-failure group-mean imputation + >10% flag (D-08-07) | VERIFIED | Lines 505-531: `judge_imputed_flags`, `group_judge_mean_raw` imputation; `fail_rate > _JUDGE_IMPUTE_WARN_RATE (0.10)` → `RuntimeWarning`; test `test_composite_judge_parse_failure_imputed` PASSED (warning emitted at 25% fail rate) |
| No external Anthropic API in reward compute | VERIFIED | Zero `anthropic.Anthropic` instantiations in `scripts/reward_pipeline.py` (0 matches) and `scripts/build_antihack_set.py` (1 match = comment in docstring: "NO direct anthropic.Anthropic( calls in the reward compute path."); `test_sc2_security_fail_scores_zero` uses MagicMock judge client |
| SEC-N04 excluded explicitly | VERIFIED | Lines 135-144: documented exclusion — "SEC-N04 (the only llm-method D2_security trigger) is excluded BY DESIGN"; `CHECK_REGISTRY[cid].method != "llm"` filter; runtime confirms `_REWARD_SEC_TRIGGERS = {SEC-N01, N03, N06, N08, N19, N20}` |
| `_security_fail` reads `triggered_checks` NOT `floor_rules_applied` | VERIFIED | Lines 190-194: `all_fired = {cid for ids in rubric.triggered_checks.values() for cid in ids}`; no reference to `floor_rules_applied` anywhere in `reward_pipeline.py` |
| Code-review status: all 12 findings resolved | VERIFIED | `08-REVIEW.md` frontmatter `status: clean`; resolution table lists all 12 findings with fix commits; final pytest 424 passed |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/reward_pipeline.py` | Composite reward pipeline, 120+ lines | VERIFIED | 644 lines; all required functions present and substantive |
| `eval/eval_judge.py` | `judge_score_single` via `_judge_create` | VERIFIED | 981 lines; `judge_score_single` at line 210 routes through `_judge_create` (line 241); RC-A guard active |
| `scripts/build_antihack_set.py` | 3-axis perturbation + CI gate, 80+ lines | VERIFIED | 788 lines; all three axes implemented; `compute_axis_gate` uses `bootstrap_ci` |
| `tests/test_reward_pipeline.py` | Unit tests for reward math | VERIFIED | Includes `TestPhpcsAvailabilityAssertion` (3 tests), `TestCompositeWeights`, `TestMogrpoNorm`, `TestVerpo` |
| `tests/test_reward_pipeline_integration.py` | SC2 fixture integration test | VERIFIED | 5 tests; `test_sc2_security_fail_scores_zero` PASSED with live phpcs |
| `tests/test_antihack.py` | 20 anti-hack tests | VERIFIED | Part of 424 passing |
| `tests/fixtures/reward_integration_cases/secure_fail_high_quality.php` | SC2 fixture (SEC-N20 trigger) | VERIFIED | File exists (2.7K) |
| `tests/fixtures/reward_integration_cases/known_good_php/` | 25 known-good fixtures | VERIFIED | 25 files present |
| `tests/fixtures/reward_integration_cases/known_bad_php/` | 24 known-bad fixtures | VERIFIED | 24 files present |
| `output/antihack_validation/acceptance_report.json` | All 4 CI bounds per axis + gate_pass | VERIFIED | Exists; `report_type=fixture_backed`; `all_axes_pass=true`; all three axes have `lo_perturbed, hi_perturbed, lo_clean, hi_clean, gate_pass` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `compute_group_rewards` | phpcs availability | `shutil.which("phpcs")` at startup | WIRED | Lines 466-479; CR-01 fix confirmed |
| `compute_group_rewards` | `_security_fail` | Call at line 579 | WIRED | `sec_fail = _security_fail(rubric)` |
| `_security_fail` | `triggered_checks` | `rubric.triggered_checks.values()` | WIRED | Line 190-194; NOT `floor_rules_applied` |
| `judge_score_single` | `_judge_create` | Direct call at line 241 | WIRED | RC-A guard active |
| `_load_score_offset` | `_RECALIB_PATH` | `Path(__file__).resolve().parent.parent / ...` | WIRED | CR-02 fix; no relative path |
| `_verpo_group` | `WP_STANDARDS_CHECK_IDS` | Filter at line 375 | WIRED | 59 IDs, D1_wpcs + D5_wp_api only |
| `compute_axis_gate` | `bootstrap_ci` | Lazy import from `scripts.compute_concentration` | WIRED | Lines 419-421 |
| `score_and_gate` | ONE combined `compute_group_rewards` call | `combined_results = compute_group_rewards(perturbed_codes + clean_codes, ...)` | WIRED | Lines 594-598; CR-03 fix confirmed |

---

### Behavioral Spot-Checks

| Behavior | Command / Test | Result | Status |
|----------|----------------|--------|--------|
| SC2 secure-fail → reward=0 | `pytest tests/test_reward_pipeline_integration.py::test_sc2_security_fail_scores_zero` | PASSED | PASS |
| CR-01: phpcs absent → RuntimeError | `pytest -k TestPhpcsAvailabilityAssertion` | 3 PASSED | PASS |
| 424 tests total, 0 failed, 0 skipped | `pytest tests/` | 424 passed, 0 skipped in 74.18s | PASS |
| `_REWARD_SEC_TRIGGERS` runtime value | Python import | `{SEC-N01, N03, N06, N08, N19, N20}` (6 ids) | PASS |
| `WP_STANDARDS_CHECK_IDS` count | Python import | 59 IDs (D1_wpcs + D5_wp_api) | PASS |
| No hardcoded 3.58 in non-comment code | grep `3.58` in `reward_pipeline.py` | 1 match = comment only (line 40) | PASS |
| No `floor_rules_applied` reference | grep in `reward_pipeline.py` | 0 matches | PASS |
| No `anthropic.Anthropic(` instantiation | grep in reward path files | 0 instantiations (1 comment-only match) | PASS |

---

### Requirements Coverage

| Requirement | Plans | Status | Evidence |
|-------------|-------|--------|----------|
| GRPO-01: (scalar, breakdown_dict) contract, composite weights, judge integration | 08-01, 08-02, 08-03, 08-04 | SATISFIED | `RewardResult(scalar, breakdown)` verified; `_W_PHPCS=0.35, _W_VERPO=0.35, _W_JUDGE=0.30` confirmed; `judge_score_single` wired |
| GRPO-02: Security hard gate, fail-closed, SEC-N04 exclusion | 08-03 | SATISFIED | Terminal override post normalize+combine; `_REWARD_SEC_TRIGGERS={6 phpcs/regex ids}`; phpcs assertion (CR-01); `test_sc2_security_fail_scores_zero` PASSED |
| GRPO-03: MO-GRPO within-group normalization, epsilon floor | 08-02 | SATISFIED | `_mo_grpo_norm: (x-mu)/(sigma+1e-8)`; three independent signals normalized |
| GRPO-04: VeRPO difficulty-weighted partial credit, WP-standards scope | 08-02 | SATISFIED | `WP_STANDARDS_CHECK_IDS` = 59 IDs (D1_wpcs + D5_wp_api); `difficulty = 1 - pass_rate` |

All four GRPO requirements SATISFIED.

---

### Anti-Patterns Found

No debt markers (`TODO`, `FIXME`, `TBD`, `XXX`, `HACK`, `PLACEHOLDER`) in `scripts/reward_pipeline.py`, `eval/eval_judge.py`, or `scripts/build_antihack_set.py`. Zero blockers. Zero warnings.

---

### Human Verification Required

#### 1. Live Anti-Hack 45-Case Scoring Run (D-11 follow-up)

**Test:** With the vLLM judge endpoint running (frozen wp_judge checkpoint at EVAL_JUDGE_BASE_URL), execute:
```
python -m scripts.build_antihack_set \
  --source-jsonl output/eval_reasoning_v4_winner/eval_gen_results.jsonl \
  --output-dir output/antihack_validation/ \
  --cases-per-axis 15 \
  --score-and-gate
```

**Expected:** `acceptance_report.json` written with `report_type=live_scored`; `all_axes_pass=true`; for all three axes, `hi_perturbed < lo_clean` (CI gate passes on real model reward scores).

**Why human:** Requires a live vLLM service. The current `acceptance_report.json` is `fixture_backed` — it proves the CI gate logic using synthetic `np.random.seed(2024)` arrays, but does not score actual perturbed PHP candidates through the reward pipeline. The CI gate implementation (`compute_axis_gate` + `score_and_gate` with CR-03 fix) is fully verified by code review and unit tests. This item is the empirical validation of the behavioral claim that the reward pipeline correctly penalizes adversarial perturbations on real model outputs. Documented as a tracked follow-up in `08-04-SUMMARY.md` under "Known Follow-Ups"; deferred pending Phase 9 vLLM infrastructure.

---

### Gaps Summary

No gaps. All five ROADMAP success criteria verified against actual code and test execution. The `human_needed` status is driven solely by the live 45-case anti-hack run, which is an explicitly tracked follow-up requiring Phase 9 vLLM infrastructure — not a defect in Phase 8 deliverables.

---

_Verified: 2026-06-20T02:30:00Z_
_Verifier: Claude (gsd-verifier)_
