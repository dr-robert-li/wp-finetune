---
phase: 13-lora-merge-pruning
plan: 05
subsystem: infra
tags: [moe, pruning, aimer, reap, monotonicity, bounded-worse, conditional-skip, qwen3-30b-a3b]

requires:
  - phase: 13-lora-merge-pruning
    plan: 04
    provides: "output/prune/gated/aimer_25_{gen,judge}.json — the decisive FAIL-both branch input (gen 0.1577 < 0.4284; judge rho 0.1651 < 0.7555, parse 0.4463 < 0.95)"
  - phase: 11-compression-packaging
    provides: "output/sieve/optimal_k.json — k=64/k=32 wp_bench monotonic-collapse precedent used to bound AIMER@50/75 without serving"

provides:
  - "output/prune/expansion_decision.md — branch record: fail-both branch taken, monotonicity argument, physical-feasibility ceiling (k=32 < 40 protected experts in layer 1)"
  - "output/prune/gated/aimer_{50,75}_{gen,judge}.json — 4 bounded-worse-by-monotonicity records (measured=false, pass=false)"
  - "output/prune/gated/reap_{25,50,75}_{gen,judge}.json — 6 documented conditional-skip stubs (skipped=true), PRUNE-02's full 6-cell table now has an explicit disposition per cell"
  - "output/prune/aimer_reap_overlap_25.json — PRUNE-04 documented conditional-skip (no REAP mask to compare)"
affects: [13-06-selection]

tech-stack:
  added: []
  patterns:
    - "Unmeasured/skipped gate cells are always written as explicit records (measured=false or skipped=true + reason), never silently omitted, so downstream selection sees every cell of the comparison table (mirrors output/sieve/optimal_k.json's per_k unmeasured-arm convention)"

key-files:
  created:
    - output/prune/expansion_decision.md
    - output/prune/gated/aimer_50_gen.json
    - output/prune/gated/aimer_50_judge.json
    - output/prune/gated/aimer_75_gen.json
    - output/prune/gated/aimer_75_judge.json
    - output/prune/gated/reap_25_gen.json
    - output/prune/gated/reap_25_judge.json
    - output/prune/gated/reap_50_gen.json
    - output/prune/gated/reap_50_judge.json
    - output/prune/gated/reap_75_gen.json
    - output/prune/gated/reap_75_judge.json
    - output/prune/aimer_reap_overlap_25.json
  modified: []

key-decisions:
  - "Fail-both branch taken per 13-04's decisive AIMER@25 FAIL on both axes: no GB10 wall-clock spent on AIMER@50/75 or REAP calibration; all 10 remaining cells (4 aimer_50/75 + 6 reap) + the overlap report written as documented dispositions, not measurements"
  - "Bounded-worse-by-monotonicity argument grounded in Phase 11's own k-sweep (optimal_k.json): k=64 wp_bench=0.2275 (-22.1pp vs full), k=32 wp_bench=0.0546 (-39.4pp) — both already worse than AIMER@25's own measured 0.1577, so a strict keep-set subset cannot recover the failed axes"
  - "AIMER@75/REAP@75 additionally flagged physically_infeasible: k=32 < 40 (layer 1's protected-expert count), so 75% could never ship regardless of accuracy (PRUNE-06 uniform-per-layer constraint, enforced in scripts/prune_selection.py)"
  - "No reap_scores_gen.npy / reap_scores_judge.npy produced — REAP calibration forward pass was never run (compute_reap_scores remains the unexecuted 13-02 stub); this is a deliberate deviation from the plan's files_modified list, explicitly sanctioned by the plan's own not-running branch text"

requirements-completed: [PRUNE-01, PRUNE-02, PRUNE-03, PRUNE-04]

coverage:
  - id: D1
    description: "Branch decision recorded citing 13-04's exact FAIL numbers on both axes, plus the monotonicity precedent and physical-feasibility ceiling"
    requirement: "PRUNE-01"
    verification:
      - kind: other
        ref: "output/prune/expansion_decision.md exists, cites aimer_25_{gen,judge}.json numbers and optimal_k.json k=64/k=32 values"
        status: pass
    human_judgment: false
  - id: D2
    description: "AIMER@50/75 gen+judge (4 files) recorded as measured=false, pass=false with bounded-worse-by-monotonicity rationale"
    requirement: "PRUNE-01"
    verification:
      - kind: other
        ref: "plan's automated verify one-liner passes: all 4 JSON files load + expansion_decision.md exists"
        status: pass
    human_judgment: false
  - id: D3
    description: "REAP gated on AIMER@25's pass/fail per PRUNE-02's conditional rule; failed, so all 6 REAP cells (25/50/75 x gen/judge) written as documented conditional-skip stubs"
    requirement: "PRUNE-02"
    verification:
      - kind: other
        ref: "plan's automated verify one-liner passes: reap_25_{gen,judge}.json load and are documented-skip records"
        status: pass
    human_judgment: false
  - id: D4
    description: "AIMER-vs-REAP overlap (PRUNE-04) given an explicit disposition (documented conditional-skip, since REAP produced no keep-mask to compare)"
    requirement: "PRUNE-04"
    verification:
      - kind: other
        ref: "plan's automated verify one-liner passes: aimer_reap_overlap_25.json has 'skipped' key"
        status: pass
    human_judgment: false

duration: <10min (no GPU/serving; pure record-writing from 13-04's measured evidence + Phase 11 precedent)
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 05: Conditional Expansion (Fail-Both Branch) Summary

**AIMER@25's decisive FAIL on both axes (13-04) triggered the plan's fail-both branch: zero GPU serving this plan, all 10 remaining gate cells (AIMER@50/75 x gen/judge, REAP@25/50/75 x gen/judge) plus the PRUNE-04 overlap written as documented bounded-worse-by-monotonicity or conditional-skip records, closing the 6-variant comparison table with explicit dispositions instead of silent gaps.**

## Branch taken

Per `13-CONTEXT`/`13-RESEARCH`'s Primary Recommendation, this plan branches on
13-04's AIMER@25 result:
- IF AIMER@25 passed all bars on at least one axis -> expand to 50/75 and run
  REAP.
- ELSE (failed both axes) -> record bounded-worse-by-monotonicity, skip REAP.

13-04 measured AIMER@25 as a decisive FAIL on both axes (gen wp_bench 0.1577
vs bar 0.4284; judge ensemble rho 0.1651 vs bar 0.7555, parse rate 0.4463 vs
bar 0.95). This plan took the **ELSE / fail-both branch**.

## Final 6-variant table (2 methods x 3 ratios)

| Method | Ratio | Gen | Judge | Disposition |
|--------|-------|-----|-------|-------------|
| AIMER | 25% (k=96) | 0.1577 FAIL | rho 0.1651 / parse 0.4463 FAIL | **MEASURED** (13-04) |
| AIMER | 50% (k=64) | null | null | bounded-worse-by-monotonicity |
| AIMER | 75% (k=32) | null | null | bounded-worse-by-monotonicity + physically infeasible |
| REAP | 25% (k=96) | null | null | conditional-skip (AIMER@25 moot) |
| REAP | 50% (k=64) | null | null | conditional-skip |
| REAP | 75% (k=32) | null | null | conditional-skip + physically infeasible |

Every cell carries an explicit, evidence-backed disposition — one measured,
five documented — per the plan's success criterion ("every cell is either
measured or has a documented, evidence-backed disposition").

## Monotonicity argument (why no serving was needed)

Phase 11's own k-sweep (`output/sieve/optimal_k.json`) measured wp_bench at
the exact keep-counts AIMER@50/75 would use: k=64 -> 0.2275 (-22.1pp vs full
0.4484), k=32 -> 0.0546 (-39.4pp). Both are already worse than AIMER@25's own
measured 0.1577. Since AIMER@50/75 keep strict subsets of AIMER@25's
keep-set, they cannot recover an axis that a larger keep-set already failed
by 27-59pp. Full reasoning and citations in `output/prune/expansion_decision.md`.

## Physical-feasibility ceiling

Independent of accuracy, k=32 (75%) is unshippable: layer 1 alone carries 40
protected experts (`output/profiling/reasoning-merged-v4/protected_expert_mask.npy`),
exceeding k=32 under PRUNE-06's uniform-per-layer keep-count constraint
(`scripts/prune_selection.py`'s `max_protected_per_layer` check disqualifies
any k=32 variant regardless of measured accuracy). This is recorded as
`physically_infeasible: true` on both `aimer_75_*.json` and `reap_75_*.json`.

## What this means for 13-06

13-06 selection (`scripts/prune_selection.load_variant_records` +
`select_winner`) will merge all variant records: AIMER@25 is the only
measured variant and fails 3 independent gates (gen bar, judge rho bar,
parse bar); all other variants are unmeasured/skipped (missing required
fields -> fail closed, never a silent pass). Expected verdict: `no_winner` —
the phase ships unpruned, consistent with Phase 11's `optimal_k=full`
sign-off.

## Performance

- **Duration:** <10 min wall-clock — no GPU, no serving; pure record-writing
  from 13-04's already-measured evidence and Phase 11's already-measured
  k-sweep precedent.
- **Completed:** 2026-07-10
- **Tasks:** 3/3
- **Files created:** 12 (1 decision doc, 4 aimer_50/75 records, 6 reap stubs, 1 overlap record)

## Task Commits

1. **Task 1: Branch decision + AIMER 50/75 bounded-worse records** - `fa1d365` (feat)
2. **Task 2: REAP conditional-skip stubs, all 6 cells (PRUNE-02)** - `fa57706` (feat)
3. **Task 3: AIMER-vs-REAP overlap documented conditional-skip (PRUNE-04)** - `cee207c` (feat)

## Files Created

- `output/prune/expansion_decision.md` — branch record: fail-both branch, monotonicity argument (citing optimal_k.json k=64/k=32 numbers), physical-feasibility ceiling, full 6-variant disposition table
- `output/prune/gated/aimer_50_gen.json`, `aimer_50_judge.json` — measured=false, pass=false, bounded-worse-by-monotonicity rationale
- `output/prune/gated/aimer_75_gen.json`, `aimer_75_judge.json` — same, plus `physically_infeasible: true`
- `output/prune/gated/reap_{25,50,75}_{gen,judge}.json` (6 files) — skipped=true, reason cites the conditional rule from 13-CONTEXT/PRUNE-02
- `output/prune/aimer_reap_overlap_25.json` — skipped=true (no REAP keep-mask exists to Jaccard against)

## Deviations from Plan

### Documented, not a bug

**1. [Plan's own not-running branch] No `reap_scores_gen.npy`/`reap_scores_judge.npy` produced**
- The plan's `files_modified` frontmatter lists these two `.npy` files, but the
  plan's Task 2 `<action>` text explicitly defines a not-running branch ("IF
  NOT running... write all six REAP artifacts... as documented conditional-skip
  stubs") that does not call for calibration score arrays. Since AIMER@25
  failed both axes, REAP calibration was never run — `scripts/reap_prune.compute_reap_scores`
  remains the unexecuted NotImplementedError stub left by 13-02. This is the
  plan's own prescribed fail-branch behavior, not a deviation requiring a
  Rule 1-4 fix.

None - otherwise the plan's fail-both branch executed exactly as written.

## Issues Encountered

None. No auth gates, no GPU serving, no build/test infra involved this plan.

## User Setup Required

None.

## Threat Flags

None. No new network endpoints, auth paths, or trust-boundary changes. T-13-06
mitigation exercised as designed: every one of the 10 unmeasured/skipped
cells was written as an explicit record (measured=false or skipped=true +
reason), never silently omitted — `13-06`'s selection input sees the full
6-variant table. T-13-03 (mask/patch tampering) and the REAP-checkpoint
requirement in the threat model were not exercised this plan since no
serving occurred (nothing to tamper with).

## Next Phase Readiness

- `13-06` has everything it needs: `output/prune/gated/{aimer,reap}_{25,50,75}_{gen,judge}.json`
  (14 files across 13-04 + 13-05) merge cleanly through
  `prune_selection.load_variant_records` -> expected verdict `no_winner`
  (phase ships unpruned).
- The 6-variant comparison table (PRUNE-01/02) and the AIMER-vs-REAP overlap
  (PRUNE-04) both have explicit dispositions; no silent gaps remain for
  13-06 to trip over.
- Zero GB10 wall-clock spent on cells with zero decision value — the
  plan's stated success criterion.

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

All 12 created artifacts verified present on disk (`ls`/`json.load` checks
below); all 3 task commits (`fa1d365`, `fa57706`, `cee207c`) found in
`git log --oneline`.
