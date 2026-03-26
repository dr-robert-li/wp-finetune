# wp-qwen3-moe

## What This Is

A WordPress best-practice fine-tuning pipeline that produces training data and trains a Qwen3-8B-based Mixture-of-Experts model capable of both generating and judging WordPress code. The model uses task tokens (`<wp_gen>`, `<wp_judge>`) to route to specialized expert pathways. Built and served on the DGX Toolbox infrastructure stack.

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

- [ ] Curate repos.yaml with high-quality WordPress plugin/theme repositories
- [ ] Execute Phase 1: Clone repos, extract functions, judge quality
- [ ] Execute Phase 2: Gap analysis, mutations, synthetic generation, judge dataset
- [ ] Execute Phase 3: CoT reasoning, instruction synthesis, final export
- [ ] Convert Qwen3-8B dense model to MoE (8 experts, top-2 routing)
- [ ] Extend tokenizer with `<wp_gen>` and `<wp_judge>` special tokens
- [ ] Fine-tune via Unsloth Studio (LoRA r=64, multi-task SFT)
- [ ] Evaluate model (PHPCS pass rate >95%, judge correlation >0.85)
- [ ] Package and deploy (GGUF for Ollama, AWQ for vLLM, HuggingFace upload)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- DPO/RLHF refinement — deferred to v2, SFT sufficient for initial release
- JavaScript/Gutenberg block code generation — PHP only for v1
- Multi-lingual comment support — English only for v1
- Mobile app or web UI — model served via DGX Toolbox inference stack (vLLM, Ollama, Open-WebUI)

## Context

**Existing codebase:** Complete data pipeline with 10 Python scripts across 3 phases, 4 YAML/Markdown config files, and comprehensive documentation. Scripts are written and tested but not yet executed against real data.

**Infrastructure:** DGX Spark (Blackwell GB10, 128GB unified memory) via DGX Toolbox provides Unsloth Studio for fine-tuning, vLLM/Ollama for inference, eval-toolbox for benchmarks, and safety harness for guardrails.

**Base model:** Qwen3-8B selected for strong PHP/code understanding, HuggingFace compatibility, and fit within DGX Spark memory. Dense-to-MoE conversion using LLaMA-MoE methodology.

**Execution model:** All LLM-heavy pipeline work (judging, generation, scoring, CoT reasoning) uses **Claude Code agents** instead of the Anthropic API — $0 cost, covered by subscription. Agents are spawned in parallel batches and continuously until data targets are met. See `docs/AGENT_PIPELINE.md` for the full execution model and output format contracts.

**Data pipeline dependencies:** Claude Code agents (subscription), PHP CLI with tokenizer extension, PHP_CodeSniffer with WordPress-Coding-Standards. Anthropic API key in `.env` for fallback/Batch API if needed.

**Target dataset:** ~20,000+ examples (40/60 `<wp_gen>`/`<wp_judge>` split) from WordPress Core, curated plugins/themes, synthetic generation, rejection examples, and rubric-scored judge training data.

## Constraints

- **API Cost:** Claude API calls for judging/generation are the primary cost driver — PHPCS pre-filtering reduces volume by ~60%
- **Hardware:** Single DGX Spark — training must fit in 128GB unified memory
- **Base Model:** Qwen3-8B — selected, not negotiable for v1
- **Quality Floor:** All training examples must pass PHPCS pre-filter AND Claude 9-dimension judge (scores ≥ 8, security auto-FAIL if security < 5)
- **Infrastructure:** DGX Toolbox components for all training, evaluation, and serving

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Qwen3-8B as base model | Strong code understanding, fits DGX Spark, active community | — Pending |
| Dense-to-MoE conversion | Enables task-token routing to specialized experts | — Pending |
| LoRA fine-tuning via Unsloth | Memory-efficient, proven on DGX Spark hardware | — Pending |
| DGX Toolbox for full lifecycle | Demonstrates toolbox usefulness across training→eval→serving | — Pending |
| PHPCS pre-filter before Claude | Reduces API cost ~60% by filtering obvious failures cheaply | ✓ Good |
| Claude Code agents over Batch API | $0 LLM cost (subscription), spawn-until-target pattern for data richness | ✓ Good |
| 40/60 gen/judge split | Emphasizes critic capability, dual-mode architecture leverage | — Pending |
| Judge threshold >= 8 (raised from 7) | Higher quality training data, compensated by larger repo pool | ✓ Good |
| Security auto-FAIL (dim < 5) | Non-negotiable security floor for training data | ✓ Good |
| Rejection examples in training | Model proactively adds security even when prompt omits it | — Pending |

---
*Last updated: 2026-03-26 after initialization*
