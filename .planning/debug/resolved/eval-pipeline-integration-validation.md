---
status: resolved
trigger: "Validate ALL recent eval pipeline fixes with execution testing. Loop fix→test→validate until clean."
created: 2026-04-06T00:00:00Z
updated: 2026-04-06T00:01:00Z
symptoms_prefilled: true
---

## Current Focus

hypothesis: One stale docstring found in merge_adapter.py — fix applied. All other checks pass.
test: Fix applied, re-run all tests
expecting: 174/174 pass, no regressions from docstring fix
next_action: Confirm fix, update CHANGELOG, commit

## Symptoms

expected: All eval pipeline scripts work correctly together with the recent fixes.
actual: Need to validate — fixes were applied individually but not tested as an integrated system.
errors: None known — this is a validation pass.
reproduction: Run each component and verify outputs match expectations.
started: Fixes applied 2026-04-06

## Eliminated

- hypothesis: pre_merge_adapters not called in run_full_triage
  evidence: grep confirmed called at line 1237
  timestamp: 2026-04-06

- hypothesis: _fallback_merge_and_serve still uses docker exec unsloth-headless
  evidence: code at line 609-655 uses HOST merge via merge_adapter.py subprocess, then DGX Toolbox for serving
  timestamp: 2026-04-06

- hypothesis: _get_vllm_container_name / _get_training_container_name missing
  evidence: python3 import test confirmed both exist at lines 162 and 149
  timestamp: 2026-04-06

- hypothesis: merge_adapter.py device_map not cpu
  evidence: line 90 confirmed device_map="cpu"
  timestamp: 2026-04-06

- hypothesis: lora_dropout not zeroed before load
  evidence: lines 98-101 confirmed: PeftConfig loaded, dropout zeroed if non-zero
  timestamp: 2026-04-06

- hypothesis: _clean_stale_results missing or broken
  evidence: execution test passed — 4 stale files removed, profiling file preserved
  timestamp: 2026-04-06

- hypothesis: wp-bench error reporting missing
  evidence: line 871 confirmed: error_detail = result.stderr[:500] if result.stderr else result.stdout[:500]; written to result JSON
  timestamp: 2026-04-06

- hypothesis: _extract_gt_from_assistant missing
  evidence: execution test passed — extracts 5 GT fields, returns None for unparseable
  timestamp: 2026-04-06

- hypothesis: _GT_FIELD_TO_DIM missing
  evidence: verified: maps wpcs_compliance, security_score, performance_score, i18n_score, accessibility_score
  timestamp: 2026-04-06

- hypothesis: gt_source field missing from pair records
  evidence: line 388 confirmed: gt_source written to pair_record dict
  timestamp: 2026-04-06

- hypothesis: eval_gen missing pass_rate_8_inclusive, na_rate, n_applicable_dims_mean
  evidence: execution test passed — all three present; security_pass_rate correctly returns None when no applicable examples
  timestamp: 2026-04-06

- hypothesis: triage_ratios missing gen_quality_scores / judge_calibrations fields
  evidence: execution test passed — TriageResult has both fields; 5pp uses gen_quality_score only; compute_overall_score retained as deprecated alias
  timestamp: 2026-04-06

- hypothesis: SKILL.md has stale 60/40 blended formula, device_map=auto, unsloth-headless
  evidence: grep found no stale references; device_map=cpu correct throughout
  timestamp: 2026-04-06

- hypothesis: test suite regressions
  evidence: 174/174 tests pass in 1.11s
  timestamp: 2026-04-06

## Evidence

- timestamp: 2026-04-06
  checked: scripts/merge_adapter.py docstring line 5
  found: docstring said "device_map=auto" but code on line 90 uses "device_map=cpu"
  implication: misleading but not functional — fix applied

- timestamp: 2026-04-06
  checked: all import tests
  found: all imports clean, no ImportError
  implication: no broken dependencies

- timestamp: 2026-04-06
  checked: full pytest suite
  found: 174/174 passed in 1.11s
  implication: no regressions from any changes

## Resolution

root_cause: One stale docstring in merge_adapter.py: line 5 said "device_map=auto" but code correctly uses "device_map=cpu". All functional fixes were correct.
fix: Updated merge_adapter.py line 5: "device_map=auto" -> "device_map=cpu" in module docstring
verification: 174/174 tests pass after fix
files_changed: [scripts/merge_adapter.py]
