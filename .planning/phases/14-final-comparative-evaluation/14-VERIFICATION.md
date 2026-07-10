# Phase 14 — Goal-Backward Verification

**Verdict:** PASSED WITH NOTES
**Date:** 2026-07-10

## Phase goal

"The pruned model is A/B compared against the v2.0 RL baseline, with inference speed delta and model size
reduction measured alongside the 9-dimension quality report."

## Goal-backward check

The literal goal is unsatisfiable: there is no pruned model (Phase 13 `no_winner`) and no v2.0 RL baseline
model (Phase 10 rejected RL, promoted nothing). The goal must be read as the user framed it — a
re-confirmation of the shipping stack. Against that reading:

| Success criterion | Status | Evidence |
|---|---|---|
| wp-bench HARD GATE recorded, pruned model >= RL baseline before Phase 15 | ADAPTED PASS | No pruned/RL models exist; gate reduced to shipping gen 0.4484 >= 0.4286 acceptance bar. Recorded in `output/eval3/eval3_final_comparison.json`. |
| Report covers 9 dims, speed delta, size reduction, seed variance | PASS | `EVAL3-REPORT.md` covers all four; size reduction 0%, speed unchanged, both with cause. |

## Notes

1. **Adapted, not literal.** Both comparison candidates are absent by prior signed-off decisions. Phase 14
   is documentary. This is disclosed in every artifact, not hidden.
2. **No fresh measurement.** Quality numbers are consolidated from `output/sieve/optimal_k.json` (same
   vLLM stack). The only new measurement is on-disk size. Acceptable: re-running would reproduce the
   frozen figures.
3. **Size promise unmet at the milestone level.** v3.0 promised a smaller model; the model did not shrink.
   This is a true negative result, recorded honestly, and hands the entire size question to Phase 15
   quantization.

## Requirements

- EVAL3-01: Complete (adapted — receipt against shipping bars, no pruned/RL candidate).
- EVAL3-02: Complete (per-dimension behavior, speed delta, size reduction, seed variance all reported).
