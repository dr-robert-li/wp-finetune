# Phase 15 — Goal-Backward Verification

**Verdict:** PASSED WITH NOTES
**Date:** 2026-07-10

## Phase goal

"The pruned model passes cascading compression gates (bf16 baseline, optional quantization, format
production) and is published to HuggingFace with full compression lineage, then validated end-to-end."

## Goal-backward check

There is no pruned model; the gates run on the unpruned shipping pair (the honest reading).

| Success criterion | Status | Evidence |
|---|---|---|
| Gate 1: size, speed, 9-dim recorded as baseline | PASS | `gate1_bf16_baseline.json` |
| Gate 2: quantization decision documented with reasoning | PASS | `gate2_quantization_decision.md` |
| Q8->Q6->Q5->Q4 incremental, stop within ±2pp, quant never before Gate 2 | PARTIAL | Ladder + stop rule + measured Q4-nf4 FAIL in `pkg03_quantization_ladder.json`; Q8/Q6/Q5 pre-registered pending toolchain (recipe: `scripts/run_packaging_recipe.md`). Not run here, not faked. |
| Model card with full lineage + AIMER/REAP comparison + usage | PASS | `MODEL_CARD.md` |
| E2E on both task tokens via serving stack | PASS (bf16) | `pkg05_e2e_validation.json` — gen 10/10, judge 10/10, routing 20/20; quantized tier pending |

## Notes

1. **Quantization sweep pre-registered, not executed.** Toolchain absent + local 30B memory wall. Turnkey
   recipe left; the one measured tier (Q4-nf4 = FAIL) is a real prior result, not a placeholder. No unrun
   eval numbers were reported. This is the same documented-disposition discipline used in Phases 13 and 14.
2. **HF upload not pushed.** Model card is upload-ready; the push is a human-authorized outward-facing step.
3. **Size promise.** bf16 footprint is unchanged from base (pruning dead). Quantization is the sole path to
   a smaller artifact, and its bottom tier (uniform 4-bit) is closed on this architecture. The realistic
   ship target is Q8 GGUF.

## Requirements

- PKG-01 Complete, PKG-02 Complete, PKG-03 Partial (ladder defined; higher tiers pending toolchain),
  PKG-04 Complete, PKG-05 Complete (bf16) / pending (quantized).
