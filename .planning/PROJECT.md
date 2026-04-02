# wp-qwen3-moe

## What This Is

A WordPress best-practice fine-tuning pipeline that produces training data and trains a Qwen3-30B-A3B-based Mixture-of-Experts model capable of both generating and judging WordPress code. The model uses task tokens (`<wp_gen>`, `<wp_judge>`) to route to specialized expert pathways (~3B active params from 128 experts). Built and served on the DGX Toolbox infrastructure stack.

## Core Value

The fine-tuned model generates WPCS-compliant, security-hardened WordPress code and catches critical defects in existing code — both capabilities in a single network.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Data pipeline scripts (Phase 1-3) — existing
- ✓ PHPCS + Claude 9-dimension quality judgment system — existing
- ✓ Taxonomy-driven gap analysis for synthetic data — existing
- ✓ Automated mutation engine for contrastive pairs — existing
- ✓ Multi-format export (OpenAI, Alpaca, Raw JSONL) with task tokens — existing
- ✓ Configuration system (repos.yaml, taxonomy.yaml, synthetic_prompts.yaml, judge_system.md) — existing
- ✓ Project documentation (PROJECT.md, wp-moe.md, README.md) — existing

### Active

<!-- Current scope. Building toward these. -->

- [x] Curate repos.yaml with high-quality WordPress plugin/theme repositories (236 repos: 1 core + 226 plugins + 9 themes)
- [x] Execute Phase 1: Clone repos, extract functions, judge quality (134,659 judged: 93,904 passed + 40,755 failed)
- [x] Execute Phase 2: Gap analysis, mutations, synthetic generation, judge dataset (143K judge training, 2,720 synthetic passed)
- [x] Execute Phase 3: CoT reasoning, instruction synthesis, final export (29,020 CoT across 4 types, 5 ratio exports from 43K-102K)
- [ ] Download Qwen3-30B-A3B base model (native MoE, no conversion needed)
- [ ] Extend tokenizer with `<wp_gen>` and `<wp_judge>` special tokens
- [ ] Fine-tune via Unsloth Studio (LoRA r=32, multi-task SFT, multi-ratio training)
- [ ] Evaluate model (9-dimension rubric: 241 checks, per-dimension Spearman correlation + wp-bench)
- [ ] Package and deploy (deferred to v2.0 — package after MoE-Sieve + pruning)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- DPO/RLHF refinement — deferred to v2, SFT sufficient for initial release
- JavaScript/Gutenberg block code generation — PHP only for v1
- Multi-lingual comment support — English only for v1
- Mobile app or web UI — model served via DGX Toolbox inference stack (vLLM, Ollama, Open-WebUI)

## Context

**Existing codebase:** Complete data pipeline with 18+ Python scripts across 3 phases, 4 YAML/Markdown config files, 6-file eval suite (241-check 9-dimension rubric scorer), and 8 Claude Code skills. Pipeline has been fully executed against 236 real repositories producing 267K merged training examples.

**Infrastructure:** DGX Spark (Blackwell GB10, 128GB unified memory) via DGX Toolbox provides Unsloth Studio for fine-tuning, vLLM/Ollama for inference, eval-toolbox for benchmarks, and safety harness for guardrails.

**Base model:** Qwen3-30B-A3B selected — native MoE (128 experts, top-8 routing, ~3B active params), proven serving compatibility (vLLM, Ollama, HuggingFace), and fits within DGX Spark 128GB unified memory. No dense-to-MoE conversion needed.

**Execution model:** All LLM-heavy pipeline work (judging, generation, scoring, CoT reasoning) uses **Claude Code agents** instead of the Anthropic API — $0 cost, covered by subscription. Agents are spawned in parallel batches and continuously until data targets are met. See `docs/AGENT_PIPELINE.md` for the full execution model and output format contracts.

**Data pipeline dependencies:** Claude Code agents (subscription), PHP CLI with tokenizer extension, PHP_CodeSniffer with WordPress-Coding-Standards. Anthropic API key in `.env` for fallback/Batch API if needed.

**Target dataset:** 43K-102K examples (exported at 5 ratios: 30/70, 40/60, 50/50, 60/40, 70/30 gen/judge) from 236 WordPress repos (core + top plugins + poor-quality plugins for judge training), synthetic generation, rejection examples, rubric-scored judge training data (143K), and 4-way CoT reasoning (29K).

## Constraints

- **API Cost:** Claude API calls for judging/generation are the primary cost driver — PHPCS pre-filtering reduces volume by ~60%
- **Hardware:** Single DGX Spark — training must fit in 128GB unified memory
- **Base Model:** Qwen3-30B-A3B — native MoE, proven toolchain, not negotiable for v1
- **Quality Floor:** All training examples must pass PHPCS pre-filter AND Claude 9-dimension judge (scores ≥ 8, security auto-FAIL if security < 5)
- **Infrastructure:** DGX Toolbox components for all training, evaluation, and serving

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Qwen3-30B-A3B as base model | Native MoE (no conversion risk), proven vLLM/Ollama/HF serving, ~3B active params, fits DGX Spark | ✓ Good |
| Native MoE over dense-to-MoE conversion | CMoE/ToMoE have no serving stack support (no vLLM, no GGUF); Qwen3-30B-A3B is production-ready | ✓ Good |
| LoRA fine-tuning via Unsloth | Memory-efficient, proven on DGX Spark hardware | — Pending |
| DGX Toolbox for full lifecycle | Demonstrates toolbox usefulness across training→eval→serving | — Pending |
| PHPCS pre-filter before Claude | Reduces API cost ~60% by filtering obvious failures cheaply | ✓ Good |
| Claude Code agents over Batch API | $0 LLM cost (subscription), spawn-until-target pattern for data richness | ✓ Good |
| Multi-ratio export (30/70 through 70/30) | Empirical comparison — train on each, eval decides best ratio | — Pending |
| Judge threshold >= 8 (raised from 7) | Higher quality training data, compensated by larger repo pool | ✓ Good |
| Security auto-FAIL (dim < 5) | Non-negotiable security floor for training data | ✓ Good |
| Rejection examples in training | Model proactively adds security even when prompt omits it | ✓ Good |
| Percentage-based pipeline targets | All targets derived from judged function counts, not hardcoded numbers | ✓ Good |
| 4-way CoT split | gen_pattern + judge_rubric + judge_contrastive + security — each max(500, 10% of base) | ✓ Good |
| Full-coverage judge training | All 134K judged functions converted to judge training format (not sampled) | ✓ Good |
| 9-dimension eval rubric (241 checks) | Replaces PHPCS-only eval with multi-tool ground truth pipeline | ✓ Good |
| Multi-ratio training with isolated checkpoints | Each ratio trains to adapters/{run_name}/, models/{run_name}-merged/ | ✓ Good |
| No one-off scripts in pipeline | Skills + pipeline_orchestrator.py, not throwaway agent scripts | ✓ Good |
| dgx_toolbox.py project-agnostic | All project-specific couplings moved to config/dgx_toolbox.yaml | ✓ Good |

## Current Milestone: v2.0 MoE-Sieve & Expert Pruning

**Goal:** Maximize specialization and inference efficiency by training only WordPress-active experts with task-aware data filtering, then conservatively pruning the coldest experts — producing a smaller, faster model that still handles edge cases.

**Target features:**
- Router profiling pass with task-token affinity tagging (separate routing counts per `<wp_gen>` vs `<wp_judge>`, not just aggregate frequency)
- Hot expert selection (top 25% per layer for LoRA targeting)
- Selective LoRA training with task-aware data filtering (gen-hot experts get golden signal only; judge-hot experts get full spectrum)
- Conservative expert pruning (bottom 10-15% near-zero routing only, preserve edge case coverage)
- Pruning validation gate (eval must confirm no regression before finalizing)
- Retrain best ratio from Phase 4 eval results
- A/B eval against v1.0 full-LoRA on wp-bench
- Packaging and deployment (package final pruned model — GGUF for Ollama, AWQ for vLLM, HuggingFace upload)

**Key constraints:**
- Depends on Phase 4 eval completing first (need winning gen/judge ratio)
- Pruning threshold empirical, not hardcoded — safety over size
- Profiling must tag expert sets by task token affinity, not just overall routing frequency
- Phase 5 (Packaging) deferred from v1.0 into v2.0 as final step — package the production model, not the intermediate

**Research basis:** MoE-Sieve (arxiv 2603.24044) — routing-guided LoRA selection matches full-LoRA within ±1pp, cuts params ~70%, wall-clock ~50%. Qwen3 router in `Qwen3MoeSparseMoeBlock`, PEFT PR #2638 or Unsloth fused MoE LoRA for per-expert targeting.

## Completed Milestones

### v1.1 Adaptive Training Infrastructure (Phase 6, completed 2026-04-01)
Power-primary adaptive planner with batch coupling, telemetry extensions, warmup probes, and failure classification. All 13 requirements verified.

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 — Milestone v2.0 started: MoE-Sieve selective expert training + conservative pruning, packaging deferred to v2.0*
