# Changelog

All notable changes to the wp-qwen3-moe project.

## [Unreleased]

### Base Model Switch
- **Switched from Qwen3-8B (dense-to-MoE conversion) to Qwen3-30B-A3B (native MoE)**
- Reason: CMoE and ToMoE have no serving stack support (no vLLM, no GGUF, no Ollama compatibility)
- Qwen3-30B-A3B is production-ready: verified vLLM, Ollama, HuggingFace serving, Unsloth fine-tuning
- ~30B total params, ~3B active per forward pass, 128 experts, top-8 routing
- Fits DGX Spark 128GB unified memory (60GB BF16, 15GB QLoRA)

### Phase 2: Dataset Production (Complete)
- Switched pipeline execution from Anthropic Batch API to Claude Code agents ($0 LLM cost)
- 60 repos cloned, 57 extracted, 22,137 passed judge (69% pass rate)
- 203 synthetic examples generated and judged (98.1% pass rate)
- 4,010 judge training examples (1,500 high + 1,006 low + 1,504 synth)
- 610 CoT reasoning chains (real code + contrastive + synthetic)
- 5,958 final training examples after 40/60 ratio enforcement + dedup
- Created autonomous pipeline skill (`skills/run-data-pipeline.md`) with spawn-until-target pattern
- Created `scripts/pipeline_orchestrator.py` for state tracking and action planning

## [0.2.0] - 2026-03-26

### Phase 1: Pipeline Ready (Complete)
- Created `scripts/utils.py` with 9 shared functions: extract_json (4-strategy fallback), call_with_backoff (exponential + retry-after), checkpoint save/load (atomic rename), Batch API routing (threshold=50)
- Created `scripts/preflight.py` validating PHPCS, PHP CLI, and API key
- Created `scripts/csv_to_repos.py` converting ranked CSV data to repos.yaml
- Generated `config/repos.yaml` with 56 repos (1 core + 49 plugins + 6 themes) with auto-assigned quality_tier from vulnerability data
- 26 passing tests across test_utils.py, test_preflight.py, test_csv_to_repos.py

### Phase 2: Script Hardening (Complete)
- Updated `config/judge_system.md`: threshold raised to >= 8, security auto-FAIL (dim < 5), N/A deflated to 7
- Added rejection templates to `config/synthetic_prompts.yaml` (proactive nonce, capability, escaping)
- Hardened all 8 pipeline scripts with utils.py integration (extract_json, call_with_backoff, checkpoints, Batch API routing)
- Added PHPCS hard-fail guard to phase2_mutate.py
- Updated export_dataset.py with 40/60 gen/judge ratio, deduplication, PHP lint, sample_weight, metadata.json
- Added python-dotenv to all scripts (API key loaded from .env)
- 46 passing tests total

## [0.1.0] - 2026-03-26

### Project Initialization
- Updated base model from LLaMA-MoE to Qwen3-8B throughout all documentation
- Integrated DGX Toolbox references (Unsloth Studio, vLLM, Ollama, eval-toolbox, safety harness)
- Created GSD project structure with 4-phase roadmap and 37 requirements
- Codebase mapping (7 documents), domain research (5 documents)
- Initial pipeline scripts (10 scripts) and configuration files (4 configs)
