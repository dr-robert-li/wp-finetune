# Phase 15-01 — Summary

**Completed:** 2026-07-10
**Requirements:** PKG-01..05

## What shipped

| Artifact | Requirement | Status |
|---|---|---|
| `output/packaging/gate1_bf16_baseline.json` | PKG-01 | Complete (real, reuses Phase 14 measured) |
| `output/packaging/gate2_quantization_decision.md` | PKG-02 | Complete (decision: quantize, start Q8, nf4 excluded) |
| `output/packaging/pkg03_quantization_ladder.json` + `scripts/run_packaging_recipe.md` | PKG-03 | Ladder + stop rule + measured Q4-nf4 FAIL; Q8/Q6/Q5 pre-registered pending toolchain |
| `output/packaging/MODEL_CARD.md` | PKG-04 | Complete (full lineage, both task tokens, honest gates) |
| `output/packaging/pkg05_e2e_validation.json` | PKG-05 | bf16 VALIDATED (gen 10/10, judge 10/10, routing 20/20); quantized pending |

## Result

- **Gate 1:** 57 GB bf16, wp-bench 0.4484, judge ensemble rho 0.8075, speed unchanged.
- **Gate 2:** Quantization warranted — the pair (114 GB single-seed / 228 GB ensemble) doesn't fit the
  121 GB host with headroom. Uniform nf4 4-bit excluded (measured router collapse). Start Q8, descend with
  ±2pp gate, activation-aware methods below Q8.
- **PKG-03:** Q4 uniform nf4 = FAIL (measured). Q8/Q6/Q5 pre-registered, turnkey recipe left; not faked.
- **PKG-04:** Model card written with the full base -> RL(rejected) -> Sieve(full) -> merge -> prune(no_winner)
  -> quantize lineage and both `<wp_gen>`/`<wp_judge>` usage.
- **PKG-05:** bf16 shipped format validated end-to-end (prior vLLM postmerge); quantized tier pending.

## Deviations

- **Q8/Q6/Q5 not executed.** The quant toolchain (autoawq/llmcompressor/llama.cpp) is not installed here,
  and local 30B quant/serve hits the documented memory wall. Pre-registered with a turnkey recipe instead
  of fabricating numbers. Consistent with the project's gate-before-commit, no-fabrication discipline.
- **HF upload not pushed.** Model card is ready; `huggingface-cli upload` is the human-authorized final
  step per ROADMAP. Not run autonomously (outward-facing publish).

## Next phase readiness

Phase 16: pipeline lockdown + repo cleanup. The packaging recipe and model card become part of the
repeatable pipeline documentation.

## Self-check

- All five PKG artifacts on disk: yes.
- Every reported quant result is real (Q4-nf4 FAIL measured; higher tiers explicitly marked pending): yes.
- Gate 1 arithmetic matches Phase 14: 0.4484 / 0.8075 / 57 GB: yes.
