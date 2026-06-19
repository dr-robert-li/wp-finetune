---
status: partial
phase: 08-reward-infrastructure
source: [08-VERIFICATION.md]
started: 2026-06-20T02:40:00Z
updated: 2026-06-20T02:40:00Z
---

## Current Test

[awaiting human testing — requires live vLLM judge endpoint, available in Phase 9 infra]

## Tests

### 1. Live 45-case anti-hack scoring run (D-11)
expected: Running `scripts/build_antihack_set.py --score-and-gate` against the live `wp_judge` vLLM endpoint produces, for each of the 3 hack axes (verbose padding / template-critique collapse / self-preference swap), a CI-aware gate pass `hi_perturbed < lo_clean` over 15 real perturbed cases vs the clean baseline. All 3 axes PASS → anti-hack set empirically validated for use as the RL regression check.
result: [pending — vLLM judge endpoint not running in the build environment; in-scope deliverables (construction script, 3-axis perturbation, combined-group CI gate per CR-03 fix, fixture-backed acceptance report) are complete and tested]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
