---
phase: 13-lora-merge-pruning
plan: 06
subsystem: infra
tags: [moe, pruning, selection, no-winner, human-signoff, ship-unpruned, qwen3-30b-a3b]

requires:
  - phase: 13-lora-merge-pruning
    plan: 03
    provides: scripts/prune_selection.py — the eligibility gate + winner-selection rule (reused unmodified)
  - phase: 13-lora-merge-pruning
    plan: 04
    provides: output/prune/gated/aimer_25_{gen,judge,d2}.json — the only measured variant (FAIL both axes)
  - phase: 13-lora-merge-pruning
    plan: 05
    provides: output/prune/gated/{aimer_{50,75},reap_{25,50,75}}_{gen,judge}.json + expansion_decision.md + aimer_reap_overlap_25.json — the 5 documented dispositions closing the 6-variant table
provides:
  - "output/prune/selection.json — no_winner verdict + per-variant reasons + layer_stability_disposition + human_signoff (ship_unpruned)"
  - "output/prune/comparison_table.md — human-facing 6-variant x 2-axis table + PRUNE-04 overlap + layer-stability headroom table + decision line"
affects: [13-07-physical-surgery]

tech-stack:
  added: []
  patterns:
    - "Human sign-off recorded directly in the decision artifact (selection.json human_signoff block), mirroring output/sieve/prune_set_for_phase13.json's convention — approver, date, mechanism, decision string"

key-files:
  created:
    - output/prune/selection.json
    - output/prune/comparison_table.md
  modified: []

key-decisions:
  - "Selection verdict no_winner produced by scripts/prune_selection.py (not hand-declared): AIMER@25 fails 3 independent gates (gen 0.1577<0.4284, judge rho 0.1651<0.7555, parse 0.4463<0.95); all 5 other variants fail closed on missing measured fields (bounded-worse/conditional-skip records carry nulls by design)"
  - "layer_stability headroom obligation dispositioned in selection.json: uniform-K makes per-layer budgets impossible, so enforcement rides the protected mask; per-flagged-layer protected counts recorded (max 36 at layer 35); K>=2x check — K=96 clears (>=72 and >=80), K=64/K=32 fail; candidate_winner_check null since no winner exists, with the note that headroom was never the disqualifying factor for K=96"
  - "HUMAN DECISION (verbatim): 'APPROVED: ship unpruned.' — Dr. Robert Li, 2026-07-10, via AskUserQuestion at the blocking checkpoint. Phase 13 closes without physical surgery; 13-07 does not run"

requirements-completed: [PRUNE-05]

coverage:
  - id: D1
    description: "Single comparison table over all 6 variants x 2 axes produced and presented to the human, including PRUNE-04 overlap and the layer-stability headroom table"
    requirement: "PRUNE-05"
    verification:
      - kind: other
        ref: "output/prune/comparison_table.md — all 12 cells dispositioned (1 measured FAIL, 11 documented); plan's automated verify one-liner passes"
        status: pass
    human_judgment: false
  - id: D2
    description: "Selection rule yields explicit no_winner verdict with per-variant reasons; 75% never selectable (k=32 < 40 physical-feasibility floor)"
    requirement: "PRUNE-05"
    verification:
      - kind: other
        ref: "output/prune/selection.json — verdict no_winner, per_variant reasons for all 6, max_protected_per_layer 40"
        status: pass
    human_judgment: false
  - id: D3
    description: "Blocking human sign-off obtained BEFORE any physical weight removal; decision recorded in the artifact"
    requirement: "PRUNE-05"
    verification:
      - kind: other
        ref: "selection.json human_signoff block — approved true, decision ship_unpruned, Dr. Robert Li 2026-07-10"
        status: pass
    human_judgment: true

duration: ~15min (no GPU; selection script run + table render + sign-off round-trip)
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 06: Selection + Blocking Human Sign-off Summary

**The machine selection rule returned `no_winner` over the full 6-variant table (AIMER@25 measured-FAILs three independent gates; the other five variants are bounded-worse, physically infeasible, or conditional-skip), and the human approved shipping unpruned — Phase 13 closes without physical surgery, 13-07 does not run.**

## Selection verdict

`scripts/prune_selection.py` (13-03's rule, reused unmodified) run over all
`output/prune/gated/*.json` records:

| Variant | K | Eligible | Disqualifying reasons |
|---------|---|----------|----------------------|
| AIMER@25 | 96 | NO | gen 0.1577 < 0.4284; judge rho 0.1651 < 0.7555; parse 0.4463 < 0.95 (all MEASURED, 13-04) |
| AIMER@50 | 64 | NO | unmeasured (bounded-worse-by-monotonicity) — fails closed on missing fields |
| AIMER@75 | 32 | NO | unmeasured + physically infeasible (k=32 < 40 protected in layer 1) |
| REAP@25/50/75 | 96/64/32 | NO | conditional-skip (AIMER@25 didn't pass; REAP never calibrated) — fail closed |

Verdict: **`no_winner`** — a first-class outcome per 13-CONTEXT, consistent
with Phase 11's `optimal_k=full` sign-off. The PRUNE-04 AIMER-vs-REAP overlap
is a documented skip (no REAP mask exists to compare).

## Layer-stability headroom disposition (13-CONTEXT hard constraint)

Recorded in `selection.json.layer_stability_disposition`: flagged layers
{9,13,14,31,35,36,45,46,47} carry 27–36 protected experts each (max 36 at
layer 35). Uniform-K makes a per-layer budget impossible (`num_local_experts`
is a scalar), so the obligation is enforced via the protected mask. Headroom
check K >= 2x max flagged protected count (72; conservative global variant 80):
K=96 clears both, K=64 and K=32 fail. `candidate_winner_check` is null — no
winner exists to assert against — with the recorded note that had AIMER@25
passed its accuracy bars, K=96 would also have cleared headroom.

## Human decision (verbatim)

> **APPROVED: ship unpruned.**

— Dr. Robert Li, 2026-07-10, via AskUserQuestion at the blocking
checkpoint:human-verify (plan 13-06 Task 2). Recorded in
`selection.json.human_signoff` (`decision: "ship_unpruned"`). Consequence:
**13-07 physical surgery does not run**; the model ships at full 128-expert
width per layer.

## Performance

- **Duration:** ~15 min wall-clock — no GPU; selection script + table render + sign-off round-trip
- **Completed:** 2026-07-10
- **Tasks:** 2/2 (Task 1 auto; Task 2 blocking human-verify, decision obtained)
- **Files created:** 2

## Task Commits

1. **Task 1: PRUNE-05 comparison table + selection verdict (no_winner)** - `27aecd8` (feat)
2. **Task 2: human sign-off recorded (APPROVED: ship unpruned)** - `5e530e5` (docs)

## Files Created

- `output/prune/selection.json` — no_winner verdict, per-variant reasons, max_protected_per_layer 40, layer_stability_disposition, human_signoff (ship_unpruned)
- `output/prune/comparison_table.md` — 6-variant x 2-axis table, PRUNE-04 overlap (skipped), layer-stability headroom table, eligibility-gate spec, decision line + what each approval option means for 13-07

## Deviations from Plan

None - plan executed exactly as written. The expected verdict (`no_winner`)
was produced by the selection script, not hand-declared; the blocking
checkpoint was honored (autonomous:false, no self-approval); the human
decision authorized the no-winner outcome.

## Issues Encountered

None. No GPU, no serving, no auth gates.

## User Setup Required

None.

## Threat Flags

None — no new network endpoints or trust-boundary changes. T-13-02 mitigation
exercised: bars are the vLLM-measured floors, missing fields fail closed,
no_winner is an explicit allowed outcome, and the human sign-off is recorded
in the artifact before any surgery. T-13-07 mitigation exercised: the blocking
checkpoint was reached and an explicit approval string obtained; the approval
declines surgery, so 13-07 must not run.

## Next Phase Readiness

- **13-07 does NOT run**: `selection.json` records `verdict: no_winner` +
  `human_signoff.decision: ship_unpruned`. Physical surgery is explicitly
  declined; any future attempt to run 13-07 must check this gate first.
- Phase 13's pruning track is closed: the model ships unpruned with full
  evidence (6-variant table, monotonicity argument, physical-feasibility
  ceiling, human sign-off) — the legitimate close 13-CONTEXT anticipated.

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

Both artifacts + SUMMARY exist on disk; both task commits (27aecd8, 5e530e5)
found in git log.
