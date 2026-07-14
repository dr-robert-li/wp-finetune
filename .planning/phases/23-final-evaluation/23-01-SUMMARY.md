---
phase: 23-final-evaluation
plan: 01
subsystem: evaluation
tags: [wp-bench, judge-rho, bootstrap-ci, synthesis, milestone-verdict]

requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "gen SFT (ep1/ep3), rebuilt-mix SFT (v4b), judge SFT (s1 served + 3-seed capture) receipts, plus raw-base wp-bench anchor"
  - phase: 21-diagnostic
    provides: "DIAGNOSTIC_SYNTHESIS.md causal analysis (regression-to-teacher, overtraining, engine-numerics ceiling)"
provides:
  - "output/eval4/comparability_audit.json -- receipt-comparability determination + offline raw-base CI backfill"
  - "output/eval4/eval4_final_comparison.json -- machine-readable EVAL4-01 milestone verdict"
  - "output/eval4/VERDICT-EVAL4.md -- human-readable milestone verdict narrative"
  - "scripts/build_eval4_comparison.py -- reusable synthesis script (--emit audit / --emit verdict)"
affects: [24-conditional-gate, 25-conditional-gate, 26-conditional-gate, 27-packaging]

tech-stack:
  added: []
  patterns:
    - "Pure synthesis phase: no GPU/Tinker spend, every figure copied from a named source_receipt (eval3 provenance pattern)"
    - "Offline CI backfill reuses the exact stratified-bootstrap function (_bootstrap_ci_lower) rather than reimplementing it, guaranteeing identical strata/weights/seed across every arm"

key-files:
  created:
    - scripts/build_eval4_comparison.py
    - output/eval4/comparability_audit.json
    - output/eval4/eval4_final_comparison.json
    - output/eval4/VERDICT-EVAL4.md
  modified: []

key-decisions:
  - "gen_role_winner = raw_base: dominates every trained variant on both point estimate and CI-lower; SFT-for-codegen has negative headroom on this stronger base"
  - "needs_confirmatory_gpu_run = false: all four gen candidates share the identical harness fingerprint; the one missing figure (raw-base CI) is backfilled offline via the reused stratified bootstrap; greedy decoding (temp=0, seed=1337) makes a re-serve non-informative"
  - "primary_judge_target_met = false, disposition = valid_recorded_miss: judge rho misses both pre-registered targets (served s1 CI-lower 0.7125 < 0.85; capture ensemble CI-lower 0.7563 < 0.87), recorded as the pre-registered failure disposition, not forced to pass"
  - "relabel_reopen_condition_met = false: the gap-closure diagnostic (capacity/loss-shape/data-cleaning) has not yet been run on THIS base, so the judge relabel campaign re-open condition is not triggered"

patterns-established:
  - "Comparability-audit-before-verdict gate: emit verdict hard-fails if comparability_audit.json is absent or reports any un-reconciled comparability gap"

requirements-completed: [EVAL4-01]

coverage:
  - id: D1
    description: "Receipt-comparability audit across all four gen wp-bench candidates (ep3/ep1/v4b/raw-base) with the raw-base CI backfilled offline via the reused stratified bootstrap"
    requirement: "EVAL4-01"
    verification:
      - kind: other
        ref: "python3 -c \"...assert d['gen_harness_comparable'] is True... assert d['needs_confirmatory_gpu_run'] is False\" against output/eval4/comparability_audit.json"
        status: pass
    human_judgment: false
  - id: D2
    description: "eval4_final_comparison.json assembled with dual-candidate gen A/B (raw base vs ep1), gen-role winner picked, judge A/B, and the mechanical pre-registered verdict applied CI-aware"
    requirement: "EVAL4-01"
    verification:
      - kind: other
        ref: "python3 -c \"...assert d['gen_ab']['gen_role_winner']=='raw_base'... assert v['primary_judge_target_met'] is False...\" against output/eval4/eval4_final_comparison.json, including source_receipt existence checks"
        status: pass
    human_judgment: false
  - id: D3
    description: "VERDICT-EVAL4.md human-readable narrative covering primary verdict, both A/B tables, gen-role-winner rationale, failure disposition, and next-lever pointers, all sourced from eval4_final_comparison.json"
    requirement: "EVAL4-01"
    verification:
      - kind: other
        ref: "python3 -c \"...assert '0.4897' in t and '0.4381' in t... assert 'Artifacts this phase produces' in t\" against output/eval4/VERDICT-EVAL4.md"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-07-15
status: complete
---

# Phase 23 Plan 01: EVAL4-01 Final Verdict Summary

**Milestone verdict recorded: raw base (0.4897) wins the gen role over every trained variant (ep1 0.4381 / v4b 0.4022 / ep3 0.372); judge rho misses both pre-registered targets (served s1 CI-lower 0.7125 < 0.85, capture ensemble CI-lower 0.7563 < 0.87) — a valid, pre-registered recorded miss, not a forced pass.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-07-14T21:54:00Z
- **Completed:** 2026-07-14T22:06:54Z
- **Tasks:** 3 completed
- **Files modified:** 4 (all created)

## Accomplishments

- Verified same-harness/stack/seed comparability across all four gen wp-bench candidates (ep3, ep1, v4b, raw base) via automated field-equality assertion against the pre-registered harness fingerprint.
- Backfilled the one missing figure (raw-base CI: [0.3812, 0.5983]) entirely offline, reusing the exact `_bootstrap_ci_lower` stratified-bootstrap function from `scripts/build_gen03_wpbench.py` — no re-derivation, no new GPU run.
- Assembled `eval4_final_comparison.json`, mirroring the eval3 provenance pattern: every gating figure carries a `source_receipt` path that resolves to a file on disk.
- Applied the pre-registered acceptance criteria (judge rho > 0.85 single OR > 0.87 ensemble; wp-bench >= 0.4286) mechanically and CI-aware — no interpretive "close enough" pass.
- Picked and recorded the gen-role winner (raw base) per the USER DIRECTIVE's dual-candidate structure.
- Authored `VERDICT-EVAL4.md`, the human-readable milestone verdict, reading every figure from the machine-readable JSON.

## Task Commits

Each task was committed atomically:

1. **Task 1: Receipt-comparability audit + offline raw-base CI backfill** - `b290278` (feat)
2. **Task 2: Assemble eval4_final_comparison.json + apply pre-registered criteria mechanically** - `cec5426` (feat)
3. **Task 3: Author VERDICT-EVAL4.md milestone verdict narrative** - `c3f636f` (docs)

_No TDD tasks in this plan (pure synthesis, no code behavior to test-first)._

## Files Created/Modified

- `scripts/build_eval4_comparison.py` - Synthesis script with `--emit audit` and `--emit verdict` entry points; reuses `_bootstrap_ci_lower` from `build_gen03_wpbench.py` for the offline CI backfill.
- `output/eval4/comparability_audit.json` - Field-equality assertion of the wp-bench harness fingerprint across ep3/ep1/v4b; backfilled raw-base CI; confirmatory-GPU-run decision (skip) with three-part justification; judge cross-base + engine-numerics-ceiling flags.
- `output/eval4/eval4_final_comparison.json` - The machine-readable EVAL4-01 milestone verdict: dual-candidate gen A/B, judge A/B, mechanical pre-registered-criteria verdict.
- `output/eval4/VERDICT-EVAL4.md` - Human-readable narrative: primary verdict, both A/B tables, gen-role-winner decision, disposition + next levers, SC2 commit-before-decision note, artifacts list.

## Decisions Made

- **gen_role_winner = raw_base** — raw base dominates every trained variant on both point estimate and CI-lower; robust regardless of whether raw-base CI-lower itself clears the floor, since it is a relative A/B call.
- **needs_confirmatory_gpu_run = false** — all gen candidates share the identical harness fingerprint (asserted, not assumed); the sole missing figure is computed offline via the reused bootstrap; greedy decoding means a re-serve would reproduce identical per-test outcomes.
- **primary_judge_target_met = false, disposition = valid_recorded_miss** — applied mechanically against CI-lower, not point estimate; recorded as the pre-registered failure disposition per plan intent ("no_winner is a result").
- **Judge relabel campaign NOT re-opened** — `relabel_reopen_condition_met = false` because the gap-closure diagnostic (capacity/loss-shape/data-cleaning levers) has not yet been run on the Qwen3.6-35B-A3B base; only one of the two required legs of the re-open condition is satisfied.

## Deviations from Plan

None - plan executed exactly as written. Two source-receipt path constructions in the first draft of Task 2 concatenated two file paths with `" / "` and `"#fragment"` suffixes for candidates whose figure derives from two receipts or a nested JSON block; this broke the verify step's `open(r['source_receipt']).close()` check. Fixed by splitting into a primary `source_receipt` (a real, openable path) plus a secondary descriptive field (`source_receipt_note` / `source_receipt_secondary`) before committing Task 2 — caught by the task's own automated verify command before any commit, so no separate fix-commit was needed.

## Issues Encountered

None beyond the source-receipt path issue documented above, which was resolved within Task 2's normal verify-fix-verify loop prior to committing.

## User Setup Required

None - no external service configuration required. Pure-Python synthesis over existing on-disk receipts.

## Next Phase Readiness

`output/eval4/eval4_final_comparison.json` and `output/eval4/VERDICT-EVAL4.md` are committed to disk and available as inputs to:
- **Phases 24-26 (conditional gates):** the mechanical `pre_registered_verdict` block (primary_judge_target_met=false) is the input those gates evaluate.
- **Phase 27 (packaging):** the `gen_role_winner` (raw base) and judge-role recommendation (v4 SFT judge, served s1 / 3-seed capture ensemble) are the shipping_stack candidates Phase 27 will package — no packaging decision was made in this plan, only the verdict it consumes.

No blockers. The judge relabel re-open condition remains available as a documented (not triggered) next step if the judge targets are pursued further post-milestone.

---
*Phase: 23-final-evaluation*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 5 created files found on disk; all 3 task commit hashes (b290278, cec5426, c3f636f) found in git log.
