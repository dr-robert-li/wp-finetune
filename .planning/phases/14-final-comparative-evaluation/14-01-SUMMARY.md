# Phase 14-01 — Summary

**Completed:** 2026-07-10
**Requirements closed:** EVAL3-01, EVAL3-02

## What shipped

- `output/eval3/eval3_final_comparison.json` — machine-readable consolidated comparison with provenance
  on every number.
- `.planning/phases/14-final-comparative-evaluation/EVAL3-REPORT.md` — narrative report.

## Result

Re-confirmation PASS. Shipping stack (v1.2 gen + v1.3 3-seed judge ensemble) measured on the vLLM full
arm clears every applicable bar:

- wp-bench 0.4484 >= 0.4286 acceptance bar (HARD GATE for packaging: PASS).
- judge ensemble rho 0.8075 >= 0.7554 floor; single-seed s1 0.8017 >= 0.7497 fallback floor.

Size line is flat: 57 GB/checkpoint bf16, 0% reduction, unchanged inference speed. Expert-drop (Phase 11)
and weight-norm AIMER (Phase 13) both failed their gates, so no parameters were removed. Quantization
(Phase 15) is the only remaining size lever.

## Deviations

- **No fresh GPU eval run.** Deliberate. Both A/B arms (pruned model, RL baseline) do not exist, and the
  shipping-stack numbers are already measured under the identical vLLM stack in
  `output/sieve/optimal_k.json`. Phase 14 consolidates rather than re-derives. Labeled as such in both
  deliverables. This is consistent with Phase 13's documented-disposition close.

## Issues

None.

## Next phase readiness

Phase 15 (packaging + quantization). Gate 1 bf16 baseline can reuse this report's size + quality figures
directly. Gate 2 quantization decision must weigh the known 4-bit Qwen3-MoE router-collapse finding
(STATE / Phase 4.3 RTRN-04) against deployment need.

## Self-check

- `output/eval3/eval3_final_comparison.json` on disk: yes.
- wp-bench gate arithmetic: 0.4484 >= 0.4286 -> PASS.
- Every quality number cites a source artifact: yes.
