---
phase: 11-compression-packaging
plan: 05
subsystem: moe-sieve-decision
tags: [tost, equivalence-gate, optimal-k, prune-set, phase-13-handoff, sieve-close]

requires:
  - phase: 11-01
    provides: "tests/test_tost_gate.py Wave-0 contract; sieve_env_precheck recorded statsmodels_ttost_available=false"
  - phase: 11-04
    provides: "k_sweep_results.json (full/64/32 arms measured under one vLLM harness; k=13 deliberately unrun per FINAL STATE ADDENDUM); sanity_gate_recalibration.json (vLLM-vs-Tinker ~3pp serving gap, recalibrated floors)"
provides:
  - "scripts/tost_gate.py: hand-rolled TOST (two one-sided Welch t-tests), tost_equivalence() bool contract + tost_equivalent() full record + run_gate() 3-sub-gate orchestration"
  - "output/sieve/optimal_k.json: optimal_k='full', no_equivalent_k=true, per-k TOST verdicts + 3 sub-gates, human_signoff stamped (Dr. Robert Li 2026-07-10)"
  - "output/sieve/prune_set_for_phase13.json: the single Phase-13 MERGE-01/PRUNE-01 hand-off (optimal_k=full, 1,480 protected sha-pinned, layer_stability_notes verbatim, hot/cold per layer, regression bars incl. vLLM shipping-rho ~0.81 note)"
  - ".planning/phases/11-compression-packaging/SIEVE-DECISIONS.md: SIEVE-02 N/A rationale, SIEVE-03 30/70 traceability, training-free reinterpretation of SIEVE-01/04/05, s1 fallback line"
affects:
  - "Phase 13 (MERGE-01/PRUNE-01): consumes prune_set_for_phase13.json — NO expert-drop; AIMER weight-level pruning only, conservative given E_eff ~90; protected mask inviolable; layer-stability headroom obligation"
  - "Phase 14 (final eval): vLLM-measured references (wp-bench 0.4484, ensemble rho ~0.81) are the like-for-like regression baselines through the shipping stack, not Tinker-native 0.842/0.827"

tech-stack:
  added: []
  patterns:
    - "TOST equivalence hand-rolled on scipy.stats.t (Welch df) because statsmodels is absent — same review bar as bootstrap_gate.py, zero new dependency"
    - "Missing sweep arms handled gracefully: unmeasured k recorded as measured=false with the monotonicity-bound rationale, never silently skipped"
    - "Sign-off recorded on the artifact itself (optimal_k.json human_signoff key) — repudiation defense per T-11-13"

key-files:
  created:
    - scripts/tost_gate.py
    - output/sieve/optimal_k.json
    - output/sieve/prune_set_for_phase13.json
    - .planning/phases/11-compression-packaging/SIEVE-DECISIONS.md
  modified: []

decisions:
  - "optimal_k = FULL (no_equivalent_k=true), human-approved 2026-07-10: no swept k passes TOST at epsilon=2pp (k=64 -22pp, k=32 -39pp, monotone collapse; k=13 bounded worse). Expert-DROP compression dead; Phase 13 AIMER weight-level pruning is the sole remaining compression path."
  - "TOST reference = the vLLM-measured full arm (wp-bench 0.4484, ensemble rho 0.8075), NOT Tinker-native 0.842/0.827 — like-for-like single-harness arms per sanity_gate_recalibration.json (~3pp systematic serving gap); canonical bars recorded for traceability only."
  - "Judge-rho bar = vLLM full-arm rho minus the s1-s2 two_se seed noise floor (0.8075 - 0.052 = 0.755); s1 single-seed fallback bar (0.8017 - floor = 0.7497) also recorded. Immaterial to the verdict (k=64 rho 0.5415 fails any bar)."
  - "SIEVE-02 declared N/A explicitly (data-routing spec belonged to the superseded training path); SIEVE-03 satisfied by traceability to Phase 4 triage (merged-v4) + Phase 7 matched stimulus (ratio_30_70) — no new ratio decision in Phase 11."

metrics:
  duration: ~35 min active (CPU-only; checkpoint wait excluded)
  completed: 2026-07-10
status: complete
requirements: [SIEVE-02, SIEVE-03, SIEVE-05]
requirements-completed: [SIEVE-02, SIEVE-03, SIEVE-05]
---

# Phase 11 Plan 05: TOST Optimal-k + Phase-13 Prune-Set Summary

**TOST equivalence gate (hand-rolled, epsilon=2pp) confirms no swept k is equivalent to the full model — optimal_k=FULL locked by human sign-off; Phase 13 receives a single prune-set hand-off carrying the no-expert-drop finding, the sha-pinned 1,480-expert protected mask, and the verbatim layer-stability obligations.**

## What was done

**Task 1 — TOST gate + optimal-k declaration.** `scripts/tost_gate.py`:
`tost_equivalence(a, b, epsilon)` (plain-bool, Wave-0 test contract) and
`tost_equivalent(k_scores, full_scores, epsilon)` returning
`{equivalent, p_lower, p_upper, mean_diff, ci}`. Hand-rolled two one-sided
Welch t-tests on `scipy.stats.t` — statsmodels `ttost_ind` confirmed
unavailable by `sieve_env_precheck` (11-01), matching the plan's prescribed
fallback. `run_gate()` consumes the real per-item wp-bench arrays (344
tests/arm from `output/sieve/ksweep/gen_k*/wp_bench_results_*.jsonl`) and
evaluates the three SIEVE-05 sub-gates per k in {13, 32, 64}:

| k | wp-bench | TOST vs full (ε=2pp) | protected retained | judge-rho bar (0.755) | equivalent_k |
|---|---|---|---|---|---|
| full | 0.4484 | (reference) | true | (ref 0.8075) | — |
| 64 | 0.2275 | NO (mean_diff −0.235, p_upper≈1e-12) | true | FAIL (0.5415) | false |
| 32 | 0.0546 | NO (mean_diff −0.416, p_upper≈4e-40) | true | not measured | false |
| 13 | not run | measured=false, bounded worse by monotonicity | — | — | false |

`output/sieve/optimal_k.json`: `optimal_k="full"`, `no_equivalent_k=true`.
Unmeasured arms (k=13 wholly; k=32 judge axis) are recorded explicitly with
the 11-04 FINAL-STATE-ADDENDUM rationale, not silently dropped.
`tests/test_tost_gate.py` 4/4 GREEN; assert-based `__main__` self-check with
synthetic equivalent/non-equivalent arrays included. Commit `bfbbb69`.

**Task 2 — Blocking human-verify checkpoint.** Full decision package
presented (k-sweep table, TOST verdicts, sub-gates, downstream consequences).
**APPROVED by Dr. Robert Li, 2026-07-10, via AskUserQuestion**: lock
`optimal_k="full"`, `no_equivalent_k=true`; zero expert-drop compression
ships; Phase 13 AIMER weight-level pruning is the sole remaining compression
path. Approval stamped into `optimal_k.json` (`human_signoff` key) and
carried into the prune-set.

**Task 3 — Phase-13 prune-set + SIEVE-02/03 record.**
`output/sieve/prune_set_for_phase13.json` (emitted only after sign-off, per
the T-11-13 ordering): `optimal_k="full"`, `no_expert_drop_finding` (E_eff
~88–99/128 ≫ every swept budget — masking removes live capacity),
`hot_cold_per_layer` (all-keep at k=full, drop_candidates empty, per-layer
protected indices duplicated in), `protected_experts` (count **1480**
verified live against the mask at emit time; npy/json sha256 pinned and
matching 11-03/11-04's recorded checksums), `layer_stability_notes` carried
**verbatim** from the Phase-7 mask JSON (low-Jaccard band {9,13,14,31,35,36}
+ late layers {45,46,47} → Phase-13 median-threshold 2,477-expert headroom
obligation), `sieve_profile_mode="shared"`, and the regression bars including
the **vLLM shipping-rho ~0.81 note** (Tinker 0.842/0.827 are sampler-native).
`SIEVE-DECISIONS.md`: SIEVE-02 N/A rationale (data-routing spec applied only
to the superseded training path — no experts retrained, no data to route;
the compression-time control is the k-sweep masking), SIEVE-03 30/70
traceability (Phase 4 triage → `qwen3-30b-wp-30_70-reasoning-merged-v4`,
Phase 7 matched stimulus `data/final_dataset/ratio_30_70`; no new decision),
training-free reinterpretation of SIEVE-01/04/05, and the pre-authorized s1
single-seed fallback line. Commit `14b48c1`.

## Task Commits

1. **Task 1: TOST equivalence gate + optimal-k declaration** — `bfbbb69` (feat)
2. **Task 2: human sign-off checkpoint** — no commit (approval recorded in artifacts)
3. **Task 3: prune-set + SIEVE-DECISIONS.md** — `14b48c1` (docs)

## Files Created/Modified

- `scripts/tost_gate.py` — TOST implementation + 3-sub-gate orchestration + CLI + self-check
- `output/sieve/optimal_k.json` (gitignored, on disk) — verdicts + human_signoff
- `output/sieve/prune_set_for_phase13.json` (gitignored, on disk) — the Phase-13 hand-off
- `.planning/phases/11-compression-packaging/SIEVE-DECISIONS.md` — SIEVE-02/03 + reinterpretation record

## Decisions Made

- optimal_k = FULL, human-approved: expert-count compression has zero headroom on this workload; Phase 13 = AIMER weight-level only, conservative.
- TOST gated against the vLLM full arm (like-for-like), canonical Tinker bars recorded traceability-only.
- Judge-rho bar convention: vLLM reference − s1-s2 two_se noise floor; s1 fallback bar recorded alongside.

## Deviations from Plan

**1. [Rule 3 - Blocking] Per-item wp-bench arrays sourced from the ksweep jsonl files, not k_sweep_results.json**
- **Found during:** Task 1
- **Issue:** TOST needs score *arrays*; `k_sweep_results.json` holds only aggregates. The plan's read-first list pointed at the aggregate file.
- **Fix:** `run_gate()` loads the per-item 344-test jsonl files (`output/sieve/ksweep/gen_k*/wp_bench_results_*.jsonl`, produced by the same 11-04 harness runs — still like-for-like single-harness arms). Per-item scalar = `score` (knowledge) / `correctness` (execution); simple mean differs from the official weighted `overall` by <2pp, immaterial against 22–39pp gaps (noted in the module docstring).
- **Files modified:** scripts/tost_gate.py
- **Commit:** bfbbb69

**2. [Rule 2 - Missing critical] Missing-arm handling made explicit in optimal_k.json**
- **Found during:** Task 1
- **Issue:** k=13 (wholly) and k=32-judge were deliberately never measured (11-04 addendum); a naive gate would crash or silently omit them, corrupting the per-k record the checkpoint reviews (T-11-11).
- **Fix:** unmeasured arms recorded as `measured: false` with the monotonicity-bound rationale inline; missing judge rho at k=32 recorded as bar-failed with an explicit note.
- **Files modified:** scripts/tost_gate.py
- **Commit:** bfbbb69

## Requirements alignment note

SIEVE-04 was declared SATISFIED by the 11-04 FINAL STATE ADDENDUM (orchestrator, 2026-07-10) but
11-04-SUMMARY's frontmatter predates that addendum (`requirements-completed: []`). This plan's
state update marks SIEVE-04 complete in REQUIREMENTS.md alongside this plan's SIEVE-02/03/05, per
the addendum's explicit declaration.

## Known Stubs

None — no stub patterns in the created files; all artifacts carry real measured data.

## Threat Flags

None — no new security-relevant surface. All threat-model mitigations applied: T-11-11 (single-harness arms + full-arm sanity in the checkpoint table), T-11-12 (protected count asserted == 1480 against the sha-pinned mask at emit time), T-11-13 (prune-set emitted strictly after the blocking sign-off), T-11-SC (no package installed; hand-rolled TOST).

## User Setup Required

None.

## Next Phase Readiness

Phase 11 Sieve chain is CLOSED. Phase 13 consumes exactly one artifact:
`output/sieve/prune_set_for_phase13.json`. Phase 12 (sieve comparative eval) should be
re-scoped/skipped at planning time — with optimal_k=full there are no sieve variants to A/B
(the k-sweep evidence in `k_sweep_results.json` already documents the collapse).

## Self-Check: PASSED

- FOUND: scripts/tost_gate.py
- FOUND: output/sieve/optimal_k.json (human_signoff present)
- FOUND: output/sieve/prune_set_for_phase13.json (protected count 1480, stability notes verbatim)
- FOUND: .planning/phases/11-compression-packaging/SIEVE-DECISIONS.md
- FOUND commits: bfbbb69, 14b48c1
- `pytest tests/test_tost_gate.py -x -q` (.venv-tinker): 4 passed
- Protected mask sha256 unchanged: 659af6eb… (.npy) / ade549e0… (.json) — matches 11-03/11-04 records

---
*Phase: 11-compression-packaging*
*Completed: 2026-07-10*
