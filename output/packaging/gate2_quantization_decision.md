# Gate 2 — Quantization Decision (PKG-02)

**Date:** 2026-07-10
**Decision:** Quantization is **WARRANTED**. Start high (Q8), descend with a ±2pp gate, Q4-nf4 pre-excluded.

## Inputs

1. **Gate 1 baseline** (`gate1_bf16_baseline.json`): 57 GB/checkpoint bf16, wp-bench 0.4484, judge ensemble
   rho 0.8075, speed unchanged.
2. **Deployment constraint:** GB10 unified memory is 121 GB. The single-seed pair (gen + one judge seed)
   is 114 GB before KV cache and activations — it fits on paper and starves in practice. The three-seed
   judge ensemble is 228 GB and does not fit. To serve the promoted stack with headroom, the footprint has
   to come down.
3. **Prior measured result (the scar):** Phase 4.3 ran a 4-bit post-hoc gate on this exact architecture.
   bitsandbytes nf4 double-quant collapsed the MoE router and produced degenerate output regardless of the
   adapter. That retired RTRN-04 as invalid on quantized Qwen3-MoE. Artifact tombstone on disk:
   `models/qwen3-30b-wp-30_70-merged-v2-4bit` (config confirms `quant_method: bitsandbytes`,
   `bnb_4bit_quant_type: nf4`, double-quant on).

## Decision and reasoning

**Quantize: yes.** The memory math forces it. There is no serving story for the ensemble at bf16 and only
a fragile one for the single-seed pair.

**But not uniform low-bit.** The failure in Phase 4.3 was specific: uniform nf4 quantizes the router and
gate weights at the same aggressive width as everything else, and the router is exactly the part a sparse
MoE cannot afford to blur. The lesson is method, not just bit-width. The search should:

- **Start at Q8** (INT8 / GGUF Q8_0), which keeps near-full precision and is the safe end of the ladder.
- **Prefer activation-aware or mixed-precision methods** (AWQ, or GGUF K-quants like Q6_K/Q5_K_M that keep
  more bits on attention and router tensors) over uniform round-to-nearest, precisely because the router
  is the sensitive component.
- **Pre-exclude Q4 uniform nf4** as a measured FAIL. If a Q4 tier is attempted at all, it must be an
  activation-aware Q4 (AWQ W4A16) and treated as high-risk given the prior collapse.

## Pre-registered PKG-03 ladder

| Tier | Method (recommended) | Status | Gate |
|---|---|---|---|
| Q8 | GGUF Q8_0 / INT8 | TO RUN | within ±2pp of Gate 1 |
| Q6 | GGUF Q6_K | TO RUN | within ±2pp of Gate 1 |
| Q5 | GGUF Q5_K_M | TO RUN | within ±2pp of Gate 1 |
| Q4 | AWQ W4A16 (NOT uniform nf4) | HIGH RISK | within ±2pp of Gate 1 |
| Q4 (uniform nf4) | bitsandbytes nf4 | **FAIL (measured)** | router collapse, RTRN-04 |

**Stop rule:** descend until a tier drops more than 2pp below the Gate 1 baseline on wp-bench (gen) or the
judge rho floor (0.7554 ensemble / 0.7497 s1); ship the lowest tier still inside the band.

## Execution status (honest)

This environment does not have the quantization toolchain installed (no autoawq, llmcompressor, auto_gptq,
or llama.cpp; only bitsandbytes). The project has always quantized and served through DGX vLLM containers,
not local transformers, because a local 30B load hits the documented unified-memory wall
(`output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`). The Q8/Q6/Q5 measurements are
therefore **pre-registered and pending toolchain provisioning**, with a turnkey recipe in
`scripts/run_packaging_recipe.md`. No quantized eval numbers are reported that were not actually run; the
one tier with a real result (Q4 uniform nf4 = FAIL) is the Phase 4.3 measurement.
