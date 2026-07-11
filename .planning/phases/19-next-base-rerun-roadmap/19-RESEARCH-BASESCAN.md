# Phase 19 pre-research: latest Qwen base candidates (researched 2026-07-11)

## Ranked shortlist
1. **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`, Apr 16 2026, Apache 2.0) — TOP PICK. 35B total / 3B active, native ctx 262K (YaRN 1M). Tinker-supported, Unsloth fine-tune guide (router training disabled by default, matches our frozen-router discipline), vLLM v0.19.0+, llama.cpp, GGUF/AWQ/NVFP4 (NVIDIA official build = Blackwell validation signal). SWE-bench Verified 73.4, LiveCodeBench v6 80.4, Terminal-Bench 2.0 51.5.
2. **Qwen3.5-35B-A3B** (Feb 24 2026) — same architecture family, safer/older twin, more third-party REAP-pruned variants exist. SWE-bench Verified 69.2.
3. **Qwen3.6-27B dense** (Apr 22 2026) — capability wildcard (SWE-bench Pro 53.5 beats 397B MoE), simplest FT story, ~50GB bf16. BUT dense: RL/Sieve/prune gates don't apply as designed; task-token routing would become 2 LoRA adapters. Methodology change, not base swap. Tinker support unconfirmed.

## Excluded
- Qwen3.7-Max/Plus: API-only, no weights.
- Qwen3.5-122B-A10B (~228GB bf16), 397B-A17B (~742GB), Qwen3-Next-80B-A3B (~150GB): exceed 121GB GB10 budget.

## Architecture deltas vs Qwen3-30B-A3B (pipeline impact)
- 3.5/3.6 A3B line = Qwen3-Next hybrid: 40 layers in 10 blocks of (3x [Gated DeltaNet linear-attn -> MoE] + 1x [full Gated Attention -> MoE]); 256 experts top-8 PLUS shared expert (ours: pure top-8-of-128, no shared, uniform attention). MoE-Sieve profiler + protected-mask tooling need adaptation for mixed layer types + always-on shared expert.
- Tokenizer vocab 248,320 padded; task-token extension transfers, BUT known eos/pad ID mismatch between tokenizer and model.config on 3.5/3.6 (QwenLM/Qwen3.6 discussion #96) — align model.config.eos_token_id/pad_token_id before SFT.
- Prunability prospect: community REAP checkpoints (20% experts dropped, "competitive") suggest 256-expert gen has more redundancy than our 128-expert base (which measured E_eff 88-99/128, no_winner). Informal evidence, not TOST-grade.

## Judge precedent
- michaelpious.com: Qwen3.5 judge SFT improved search-relevance accuracy 0.584->0.642, QWK 0.536->0.576.

## Unverified
- Qwen3-30B-A3B SWE-bench 50.3% figure may be Coder variant; Tinker support for 3.6-27B dense; DeltaNet ops on GB10 aarch64 specifically (inferred OK); Terminal-Bench "matches Opus 4.5" single-source.

Sources: HF model cards (Qwen/Qwen3.6-35B-A3B etc.), QwenLM/Qwen3.6 GitHub, Unsloth docs, Tinker models page, NVIDIA blog/NVFP4 builds, discussion #96, MarkTechPost, community REAP GGUFs.
