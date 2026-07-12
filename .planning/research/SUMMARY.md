# Project Research Summary — v4.0 Rerun on Qwen/Qwen3.6-35B-A3B

**Milestone:** Rerun the locked PIPELINE.md on Qwen/Qwen3.6-35B-A3B, targeting judge-rho >0.85
single-seed / >0.87 ensemble vs the current base's 0.8075 wall.
**Researched:** 2026-07-12 (re-verifying .planning/V4-RERUN-ROADMAP.md, locked 2026-07-11)
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md (all dated 2026-07-12, same directory)

This file is the deliverable of the research-synthesis step of the v4.0 milestone workflow.

## Key Findings

See STACK.md / FEATURES.md / ARCHITECTURE.md / PITFALLS.md for full detail; digest below.

- Base architecture (256 experts top-8 + 1 shared, 30 DeltaNet-MoE + 10 Gated-Attention-MoE layers,
  model.language_model.* prefix, ~67 GiB bf16 / ~35 GB Q8) is confirmed against live config.json /
  model.safetensors.index.json. Q8 pair (~70 GB) fits the 121 GB GB10 host comfortably.
- Tinker pricing corrected (Prefill $0.36 / Sample $0.54 / Train $1.07) with a ~10-50% price rise landing
  2026-07-17; old base (Qwen3-30B-A3B) was retired from Tinker 2026-06-12, validating the swap timing.
- New required SFT config item: output_router_logits=True to avoid expert collapse (not in the locked doc).
- Always-on-by-default thinking mode (<think>...</think>) is architecturally different from the current
  base and needs an explicit SFT-data-format decision before Stage 2.
- Vendor capability claims (SWE-bench 73.4, LiveCodeBench 80.4) are vendor-only, contested by an informal
  community re-test showing regression and 18% output-format-noncompliance — directly relevant to Stage 3
  judge-output parseability, the exact failure mode that killed 3/4 ratios on the old base.
- Sieve profiler needs only narrow changes: one-line module-traversal fix
  (model.model.layers -> model.model.language_model.layers), n_experts 128->256, and an empirical check
  that the shared expert never appears in router_logits. No fused-vs-unfused weight handling needed.
- Critical infra pitfalls: eos/pad token mismatch (WAI, not upstream-fixed; Stage 1.5 gate already scoped);
  open vLLM CUDA-graph-capture crash on Gated-DeltaNet layers (issue #35945, mitigated by --enforce-eager);
  dual key-prefix convention between merge-time and serve-time for the VL checkpoint (silent partial-load
  risk — must smoke-test merge->serve with real generation, not just merge exit code); llama.cpp support is
  version-sensitive, pin >=b9180 and smoke-test concurrent-sequence load on the CUDA backend; shared-expert
  quant protection must be independently re-verified at packaging time (Phase 28), separate from Sieve
  protection (Phase 25); open chat-template bug emits empty <think></think> blocks in historical turns
  (issue #131) - verify SFT data assembly isn't affected.
- No newer/better open-weight Qwen release found; Qwen3.7 is proprietary/API-only. Base lock stands.

## Implications for Roadmap

All four researchers converged: no new phases, no reordering of the locked Phase 20-29 structure. All
findings are hardenings of already-scheduled phases:

**Phase 20 (base bring-up / Stage 1.5 gate):** eos/pad alignment assert+smoke; DeltaNet smoke test must run
WITH CUDA-graph capture enabled (not eager-only), with --enforce-eager fallback documented; VL merge-path
check extended to a full merge->serve round-trip with real generation; confirm no vision-tower/vision-LoRA
code path is touched; pin vLLM >=0.19.0, --gpu-memory-utilization 0.80; decide on use_kernels=True for
DeltaNet (1.38x prefill speedup vs trust_remote_code tradeoff); one-line profiler module-traversal fix;
verify Tinker's LoRA target-module resolution against language_model.* prefix and log attached modules;
trust_remote_code=True on both model and tokenizer/processor.

**Phase 21/22 (SFT data + gen/judge training):** set output_router_logits=True explicitly; resolve the
thinking-mode/<think> tag SFT-format decision before Stage 2 data is fed in; spot-check rendered examples
for spurious empty <think></think> blocks if using tokenizer.apply_chat_template(); add an explicit
judge-output-format-compliance smoke check on the raw pre-SFT base early in Stage 3 (highest-value new item,
given the 18% community-reported noncompliance rate); assert max tokenized example length stays well under
Tinker's 64K training-context cap.

**Phase 25 (Sieve tooling, before Gate B):** module-path fix, n_experts 128->256 bump across 4 scripts,
empirical shared-expert-exclusion-from-router-logits verification.

**Phase 28 (Stage 5 packaging):** pin llama.cpp >=b9180; smoke-test GGUF block count against the
safetensors index post-conversion; smoke-test the CUDA backend under concurrent-sequence/router load;
independently verify shared-expert quant handling in the packaging tool's own tensor-type metadata (does
not inherit from Phase 25); if descending below Q8, verify mixed-precision handling of DeltaNet
recurrent-state tensors.

**Cost/scheduling:** if v4.0 sign-off or Stage 2/3 spend happens after 2026-07-17, budget the Tinker price
increase (train ~10% up, dominant driver; prefill/sample ~50% up, minor for this workload).

## Research Flags

Needs hands-on verification during execution (not assumption): Phase 20 (CUDA-graph DeltaNet smoke, VL
merge->serve round-trip, use_kernels decision, LoRA target-module log), Phase 21/22 (thinking-mode format
decision, judge-output-format-compliance smoke), Phase 28 (llama.cpp pin + concurrent-sequence smoke,
shared-expert quant metadata).

Standard/well-documented, low research risk: Stage 1 data reuse, eos/pad alignment mechanics, Q8 GGUF
memory sizing (re-verified to within ~2% of projection), eval harness reuse (confirmed no changes needed).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Live-fetched vendor docs, cross-checked 2-3x; exact transformers min version unpinnable, gated via bring-up smoke test instead |
| Features | MEDIUM | Architecture/infra facts HIGH; coding-quality benchmark claims contested (vendor-only vs. unreplicated community re-test) |
| Architecture | HIGH | Primary-source config.json/safetensors index fetched directly; LM-only-subset size and llama.cpp exact commit carried-forward/MEDIUM |
| Pitfalls | MEDIUM-HIGH | Primary-source GitHub issues/discussions for all critical items; Q4-nf4 router-collapse generalization to this base is flagged as unconfirmed extrapolation |

**Gaps not resolved:** no third-party reproduction of the SWE-bench/LiveCodeBench vendor numbers; no
numeric E_eff analysis or REAP benchmark deltas for this exact model; no Qwen3.6-specific (vs
Qwen3.5-analogy) LoRA hyperparameter study; exact transformers minimum version; whether Q4-nf4 router
collapse reproduces on this base's 256-expert routing (re-verify at Gate 2 of Stage 5). None of these block
sign-off — all are execution-time "measure, don't assume" items consistent with the roadmap's existing
discipline.

## Sources

Aggregated from all four research files (accessed/fetched 2026-07-12 unless noted):
tinker-docs.thinkingmachines.ai/tinker/models/; huggingface.co/Qwen/Qwen3.6-35B-A3B (+ -FP8, config.json,
model.safetensors.index.json); huggingface.co/Qwen/Qwen3-30B-A3B/.../model.safetensors.index.json;
huggingface.co/docs/transformers/model_doc/qwen3_5_moe; recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B;
github.com/vllm-project/vllm issues #35945, #28640, #40249, #36275; github.com/ggml-org/llama.cpp issues
#24737, #19903, #19857, #19915, #22135, #22425, #23011; github.com/QwenLM/Qwen3.6 discussions #96, #55,
issue #131; huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/66; huggingface.co/bartowski/, unsloth/,
nvidia/, QuantTrio/ quant repos; forums.developer.nvidia.com DGX Spark/GB10 threads (3 threads);
stevescargall.com, github.com/adadrag/qwen3.5-dgx-spark, github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash;
buildfastwithai.com SWE-bench review; gist.github.com/hungson175 (2026-05-16) independent regression report;
artificialanalysis.ai; huggingface.co/0xSero/, DJLougen/, mudler/ REAP/APEX community checkpoints;
thinkingmachines.ai/blog/lora/; qwen.ai/blog Qwen3.7 coverage + marktechpost.com; local:
.planning/V4-RERUN-ROADMAP.md, .planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md,
PIPELINE.md, scripts/merge_adapter.py, scripts/profile_base_model.py, scripts/extract_protected_mask.py,
scripts/sieve_expert_mask_inference.py.

---
*Synthesized: 2026-07-12. Supersedes prior v1.2-era SUMMARY.md and its 4 source research files.*
