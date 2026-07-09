---
phase: 13-lora-merge-pruning
plan: 03
subsystem: infra
tags: [moe, pruning, aimer, reap, safetensors, qwen3-30b-a3b, selection-rule]

requires:
  - phase: 11-compression-packaging
    provides: scripts/sieve_expert_mask_inference.py (build_ksweep_mask reused unchanged), scripts/sieve_cross_seed_overlap.py (per-layer Jaccard pattern mirrored)
  - phase: 13-lora-merge-pruning
    plan: 01
    provides: real on-disk tensor layout (per-expert unstacked keys, 13 shards) that prune_apply_physical.py's surgery targets
  - phase: 13-lora-merge-pruning
    plan: 02
    provides: scripts/prune_gated_eval.py's RATIO_TO_K constant (reused, not redefined) and the gated-eval record shape prune_selection.py's loader assumes
provides:
  - scripts/prune_overlap.py — per-layer AIMER-vs-REAP Jaccard overlap analysis (PRUNE-04)
  - scripts/prune_selection.py — winner-selection rule with regression-bar + D2_security + physical-feasibility floors (PRUNE-05)
  - scripts/prune_apply_physical.py — uniform-per-layer physical expert removal + router renorm + config rewrite (PRUNE-06)
affects: [13-04-gate-before-remove-execution, 13-05-reap-calibration-conditional, 13-06-selection-execution, 13-07-physical-surgery-execution]

tech-stack:
  added: []
  patterns:
    - "Physical-feasibility floor (K >= max per-layer protected count) implemented identically in build_uniform_keep_mask (raises ValueError) and prune_selection's evaluate_variant (hard eligibility filter) — same invariant enforced at two points in the pipeline (mask-build time and selection time), matching the plan's must_haves.key_links contract"
    - "Renumber-by-sorted-original-index: kept experts always renumbered 0..K-1 in ascending original-index order, and the router's row-slice uses that exact same order — the only way HF/vLLM's num_local_experts=K contract stays internally consistent between expert weights and router logits"

key-files:
  created:
    - scripts/prune_overlap.py
    - scripts/prune_selection.py
    - scripts/prune_apply_physical.py
    - tests/test_prune_overlap.py
    - tests/test_prune_selection.py
    - tests/test_prune_physical.py
  modified: []

key-decisions:
  - "prune_selection.py's per-variant input schema (gen_wp_bench, judge_ensemble_rho, judge_parse_rate, d2_security_retention, d2_security_baseline, protected_retained) is a NEW contract this plan defines, since 13-04/13-05's real gated-eval records (built in 13-02) don't yet carry D2_security — this module is Wave-0/ahead-of-data by design; load_variant_records() documents the expected output/prune/gated/{method}_{ratio}_{gen,judge,d2}.json file-merge convention for 13-04/13-05/13-06 to follow"
  - "D2_security eligibility check is fail-closed: missing d2_security_retention/baseline fields never silently pass (a missing d2 record makes the variant ineligible with reason missing_field:..., never treated as an implicit pass) — direct implementation of T-13-02"
  - "prune_apply_physical.py re-shards by writing each output tensor under its ORIGINAL shard filename (dropped/renamed tensors, same file grouping) rather than repacking for balanced shard sizes — correct and simple for a Wave-0 module verified on synthetic tiny checkpoints; marked with a ponytail comment noting real 13-07 execution can repack if shard-size balance matters then"
  - "RATIO_TO_K (25->96, 50->64, 75->32) is imported from scripts.prune_gated_eval in prune_selection.py rather than redefined, keeping the ratio-to-K mapping single-sourced; prune_apply_physical.py keeps its own copy since it has no other dependency on prune_gated_eval.py and importing it would pull in vLLM-serve-only code paths"

patterns-established:
  - "Any Phase-13 module needing the ratio->K mapping should import RATIO_TO_K from scripts.prune_gated_eval (established in 13-02) rather than hard-coding 25/50/75 percentages again"

requirements-completed: [PRUNE-04, PRUNE-05, PRUNE-06]

coverage:
  - id: D1
    description: "per_layer_jaccard computes correct Jaccard for identical (1.0), disjoint (0.0), and hand-computed partial-overlap masks across all 48 layers; build_overlap_report rolls up the pre-committed layer_stability_notes band ({9,13,14,31,35,36} + {45,46,47}) separately"
    requirement: "PRUNE-04"
    verification:
      - kind: unit
        ref: "tests/test_prune_overlap.py#test_identical_masks_jaccard_one, test_disjoint_masks_jaccard_zero, test_hand_computed_partial_overlap, test_build_overlap_report_length_and_band_rollup"
        status: pass
      - kind: unit
        ref: "scripts/prune_overlap.py --self-check"
        status: pass
  - id: D2
    description: "select_winner enforces all 6 eligibility checks (gen bar, judge rho bar, judge parse bar, D2_security within 2pp, protected_retained, physical feasibility K>=max-protected-per-layer) and prefers smaller K with D2_security tie-break; empty eligible set returns an explicit no_winner verdict with per-variant reasons"
    requirement: "PRUNE-05"
    verification:
      - kind: unit
        ref: "tests/test_prune_selection.py#test_clean_25_percent_winner, test_75_percent_passes_bars_but_physically_infeasible, test_d2_security_regression_disqualifies, test_all_fail_returns_no_winner, test_prefers_smaller_k_higher_compression, test_ties_broken_by_higher_d2_security"
        status: pass
      - kind: unit
        ref: "scripts/prune_selection.py --self-check"
        status: pass
  - id: D3
    description: "build_uniform_keep_mask produces exactly K True/layer (protected experts always kept, ValueError on infeasible K); apply_physical renumbers kept experts to contiguous 0..K-1, slices the router to K rows in the same order, and rewrites num_local_experts=K on a tiny synthetic checkpoint (no 60GB model load)"
    requirement: "PRUNE-06"
    verification:
      - kind: unit
        ref: "tests/test_prune_physical.py#test_uniform_mask_exactly_k_per_layer, test_infeasible_k_raises, test_apply_physical_shapes_and_renumbering, test_protected_expert_weight_survives_unmodified"
        status: pass
      - kind: unit
        ref: "scripts/prune_apply_physical.py --self-check"
        status: pass

duration: 40min
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 03: Overlap Analysis + Winner-Selection Rule + Physical Surgery Summary

**Built and unit-tested the three offline analysis/surgery modules (PRUNE-04 Jaccard overlap, PRUNE-05 eligibility-gated winner selection, PRUNE-06 uniform-count physical expert removal) that 13-04 through 13-07 will run against real gate results and the real 60GB checkpoints — closing the phase's last Wave-0 test gaps entirely on synthetic fixtures.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-07-10
- **Tasks:** 3/3
- **Files modified:** 6 (3 new scripts, 3 new test files)

## Accomplishments
- `scripts/prune_overlap.py::per_layer_jaccard` mirrors `scripts/sieve_cross_seed_overlap.py`'s Jaccard pattern; `build_overlap_report` produces the model-wide mean/min/max plus a separate roll-up for the pre-committed `layer_stability_notes` band ({9,13,14,31,35,36} + {45,46,47}) and a human-fill interpretation stub for selection-time judgment
- `scripts/prune_selection.py::select_winner` implements the full PRUNE-05 eligibility gate (all 3 regression bars + D2_security-within-2pp + protected_retained + physical feasibility) over a per-variant record list, hard-filtering ratio=75/K=32 whenever any layer's protected count exceeds K (real data: layer 1 alone carries 40 protected experts) — proven on 4 synthetic scenarios (clean winner, feasibility-filtered 75%, D2_security disqualification, all-fail no_winner) plus 2 additional tie-break scenarios
- `scripts/prune_apply_physical.py::build_uniform_keep_mask` + `apply_physical` perform the real physical surgery against the verified on-disk format (per-expert unstacked tensors, sharded index.json, `mlp.gate.weight` router, `num_local_experts` config key) — renumbering survivors to contiguous 0..K-1, slicing the router to matching rows, and rewriting config, verified end-to-end on a tiny synthetic 2-layer/8-expert checkpoint with byte-identical protected-expert weight survival
- All three `--self-check` invocations exit 0; all 14 tests across the 3 new test files pass

## Task Commits

Each task was committed atomically (TDD RED -> GREEN):

1. **Task 1: PRUNE-04 overlap analysis tests** - `ada4257` (test — RED, importorskip skips cleanly per repo convention)
2. **Task 1: PRUNE-04 overlap analysis implementation** - `f5f78e4` (feat — GREEN, 4/4 tests pass, self-check exits 0)
3. **Task 2: PRUNE-05 selection rule tests** - `2eaaba2` (test — RED)
4. **Task 2: PRUNE-05 selection rule implementation** - `0b971cf` (feat — GREEN, 6/6 tests pass, self-check exits 0)
5. **Task 3: PRUNE-06 physical surgery tests** - `7e5161f` (test — RED)
6. **Task 3: PRUNE-06 physical surgery implementation** - `e6dad94` (feat — GREEN, 4/4 tests pass, self-check exits 0)

_TDD gate sequence per task confirmed in git log: test(N) -> feat(N) for all 3 tasks._

## Files Created/Modified
- `scripts/prune_overlap.py` - `per_layer_jaccard`, `build_overlap_report`, CLI (`--mask-a`/`--mask-b`/`--ratio`/`--out`), `--self-check`
- `scripts/prune_selection.py` - `evaluate_variant`, `select_winner`, `max_protected_per_layer`, `load_variant_records` (merges `output/prune/gated/{method}_{ratio}_{gen,judge,d2}.json`), CLI, `--self-check`
- `scripts/prune_apply_physical.py` - `build_uniform_keep_mask`, `apply_physical`, CLI (`--checkpoint`/`--score`/`--protected`/`--ratio`/`--out`), `--self-check`
- `tests/test_prune_overlap.py` - identical/disjoint/partial-overlap Jaccard, band roll-up length/content
- `tests/test_prune_selection.py` - clean winner, feasibility-filtered 75%, D2_security disqualification, all-fail no_winner, smaller-K preference, D2 tie-break
- `tests/test_prune_physical.py` - uniform mask exact-K, infeasible-K raises, post-surgery shape/renumbering assertions, protected-expert byte-identical survival

## Decisions Made
- `prune_selection.py`'s per-variant input schema (`gen_wp_bench`, `judge_ensemble_rho`, `judge_parse_rate`, `d2_security_retention`, `d2_security_baseline`, `protected_retained`) is a new contract this plan defines ahead of 13-04/13-05's real execution, since 13-02's gated-eval records don't yet carry D2_security; `load_variant_records()` documents the `{method}_{ratio}_{gen,judge,d2}.json` merge convention those later plans should follow
- D2_security eligibility is fail-closed: a missing d2 record makes a variant ineligible (`missing_field:...`), never an implicit pass — direct implementation of T-13-02
- `prune_apply_physical.py` re-shards by writing output tensors under their original shard filename rather than repacking for balanced file sizes — correct and simple for this Wave-0 module, marked with a `ponytail:` comment; real 13-07 execution can repack later if needed
- `RATIO_TO_K` is imported from `scripts.prune_gated_eval` in `prune_selection.py` (single-sourced mapping); `prune_apply_physical.py` keeps its own copy to avoid pulling in vLLM-serve-only code paths from `prune_gated_eval.py`

## Deviations from Plan

None - plan executed exactly as written. The plan explicitly pre-registered that `prune_selection.py` consumes records from `output/prune/gated/*.json` (13-04/13-05 not yet run) and that the physical-feasibility floor is "derived this session, not just documented" — both were implemented as instructed, not deviations from it.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Threat Flags

None - no new network endpoints, auth paths, or trust-boundary schema changes. The two threat-model boundaries this plan touches (gated-eval records -> selection; keep-mask + checkpoint -> physical surgery) are exactly the ones the plan's threat register scoped, and both mitigations (T-13-01 protected-count-vs-K assertion, T-13-02 D2_security-within-2pp hard filter, T-13-04 post-surgery shape assertions) are implemented and verified by the unit tests above.

## Next Phase Readiness

- `scripts/prune_overlap.py` is ready for 13-06 to invoke once both AIMER and REAP keep-masks exist at a matched ratio (conditional on REAP running at all, per 13-CONTEXT's AIMER@25%-first gate)
- `scripts/prune_selection.py` is ready for 13-06 to invoke over the real `output/prune/gated/*.json` records 13-04/13-05 will produce — but 13-04/13-05 must additionally emit a `d2_security_retention`/`d2_security_baseline` pair (via a `{method}_{ratio}_d2.json` file or by adding those keys into the existing gen/judge records) for `load_variant_records()` to find; this is a forward dependency on this plan's new schema, not a blocker for 13-04's AIMER@25% gate run itself
- `scripts/prune_apply_physical.py` is ready for 13-07 to invoke against the real 60GB checkpoint once 13-06 selects a winner; verified only on synthetic tiny fixtures per plan's explicit "no full 60GB model load in the test" done criterion
- No blockers for 13-04 (gate-before-remove execution)

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

All 6 created files found on disk (3 scripts, 3 test files); all 6 task commit hashes (`ada4257`, `f5f78e4`, `2eaaba2`, `0b971cf`, `7e5161f`, `e6dad94`) found in git log.
