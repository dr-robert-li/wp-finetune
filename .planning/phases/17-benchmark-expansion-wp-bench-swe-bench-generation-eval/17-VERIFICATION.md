---
phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
verified: 2026-07-11T00:45:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 17: Benchmark Expansion — wp-bench + SWE-bench Generation Eval — Verification Report

**Phase Goal:** Shipped two-model pair has current, honest benchmark numbers: full wp-bench run on v1.2 gen model (shipping stack) + SWE-bench generation-mode eval; results in MODEL_CARD.md with out-of-domain caveat.
**Verified:** 2026-07-11
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | BENCH-01: full 344-test wp-bench run on shipping stack (vLLM bf16), score/config/seed recorded, delta vs 0.4484 computed | ✓ VERIFIED | `output/bench17/wpbench_full_gate_rerun.json`: 344 tests (320 knowledge + 24 execution), score 0.4365, wp_bench_run_seed 1337, vLLM bf16 same-stack attestation, delta -0.0119 vs 0.4484 (within 5.20pp noise floor), clears 0.4286 acceptance bar |
| 2 | BENCH-02: pre-registration committed strictly before any SWE-bench eval result exists | ✓ VERIFIED | `git merge-base --is-ancestor 65116ed ae488fd` = true; 65116ed (05:35) predates ae488fd (10:15) same day, ~4.5h gap; pre-reg doc explicitly states no eval had been read at commit time |
| 3 | BENCH-02: SWE-bench gen-mode eval at largest honestly-evaluable aarch64 scope, full-scope denominators with disclosed non-resolution categories | ✓ VERIFIED | `output/bench17/swebench_eval_report.json`: Lite300 5+126+80+1+59+29=300 (checksum exact), PHP43 0+20+17+0+6+0=43 (checksum exact); evaluated-subset (131, 20) = resolved+unresolved-in-container in both cases |
| 4 | No fabrication: reported resolved/unresolved dispositions match actual harness output | ✓ VERIFIED | Spot-checked 5/5 declared `resolved_ids` against `logs/run_evaluation/lite300_v1/.../report.json` — all `resolved: true`; spot-checked 1 `patch_apply_failed_id` (django__django-11019) against `run_instance.log` — confirms genuine `Patch Apply Failed` traceback, no report.json (consistent with never reaching test stage) |
| 5 | BENCH-03: MODEL_CARD.md Benchmarks section with both numbers, generation-mode/oracle caveats, out-of-domain caveat, numbers matching receipts exactly | ✓ VERIFIED | `output/packaging/MODEL_CARD.md` "## Benchmarks" (lines 102-134): 0.4365, 1.67% (5/300), 0% (0/43), 3.82% evaluated-subset all byte-match the two receipt JSONs; explicit out-of-domain / "why the number is low" paragraph present |
| 6 | REQUIREMENTS/ROADMAP/STATE consistency: BENCH-01..03 complete, phase 17 marked complete | ✓ VERIFIED | REQUIREMENTS.md lines 387-389 all "Complete"; ROADMAP.md line 107 `[x] Phase 17`, plans 17-01/02/03 all `[x]`; STATE.md `status: phase-17-complete` |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `output/bench17/wpbench_full_gate_rerun.json` | BENCH-01 receipt | ✓ VERIFIED | Present, complete, internally consistent |
| `output/bench17/swebench_scope_preregistration.md` | Locked scope, pre-results | ✓ VERIFIED | Present; commit 65116ed provably precedes eval |
| `output/bench17/swebench_eval_report.json` | BENCH-02 receipt | ✓ VERIFIED | Present; checksums on disclosure categories exact for both datasets |
| `output/packaging/MODEL_CARD.md` (Benchmarks section) | BENCH-03 | ✓ VERIFIED | Present, numbers match receipts, caveats present |
| `logs/run_evaluation/lite300_v1/.../report.json` (per-instance) | Harness ground truth | ✓ VERIFIED | Real per-instance reports exist and back the resolved_ids claims |

### Anti-Patterns Found

None. No TBD/FIXME/XXX/placeholder markers found in the reviewed receipt or MODEL_CARD content.

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| BENCH-01 | Full wp-bench, score+config+seed, compared vs 0.4484 | ✓ SATISFIED | wpbench_full_gate_rerun.json |
| BENCH-02 | SWE-bench gen-mode, largest honest aarch64 scope, pre-registered before results | ✓ SATISFIED | swebench_scope_preregistration.md + git ancestry + swebench_eval_report.json |
| BENCH-03 | MODEL_CARD.md Benchmarks section + out-of-domain caveat | ✓ SATISFIED | MODEL_CARD.md lines 102-134 |

### Human Verification Required

None. All claims were mechanically verifiable via receipt JSON internal consistency, git commit ancestry, and direct cross-reference against raw per-instance harness `report.json`/`run_instance.log` files (not just the consolidated SUMMARY narrative).

### Gaps Summary

None. All six observable truths verified against primary artifacts, not SUMMARY.md prose. Disclosure-category checksums are exact (300 and 43 respectively), the pre-registration-before-results ordering is provable by git ancestry rather than timestamp claim alone, and a 5-of-5 spot-check of "resolved" instance IDs plus a spot-check of one "patch-apply-failed" instance against raw harness logs found no discrepancy with the consolidated report.

---
*Verified: 2026-07-11*
*Verifier: Claude (gsd-verifier)*
