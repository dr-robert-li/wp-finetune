# Changelog

All notable changes to the wp-qwen3-moe project.

## [Unreleased]

### Execution Engine Architecture
- Refactored `scripts/dgx_toolbox.py` from path resolver into full execution engine
- Architecture: Skill (intent + recovery) → dgx_toolbox.py (validate + execute) → Docker commands (dynamic)
- New methods: `validate()`, `ensure_ready()`, `execute()`, `run_service()`, `status_report()`
- `CONTAINER_MAP` maps phases to dgx-toolbox containers (unsloth_studio, eval_toolbox, vllm)
- `status_report()` provides structured telemetry for background observer agents
- Removed brittle `run_training_pipeline.sh` — Python engine replaces it
- Skills declare intent + recovery logic, engine handles container lifecycle dynamically
- Idempotency built into `execute()` via `idempotency_check` parameter

### Phase 3: Model Prep and Training (Scripts Complete — Awaiting DGX Execution)
- 75 tests passing, all scripts created and approved
- Eval scripts in `eval/` (eval_gen.py, eval_judge.py, eval_gate.py)
- Training scripts in `scripts/` (train_model.py, merge_adapter.py, download_model.py, prepare_tokenizer.py)
- Configs in `config/` (train_config.yaml, wp-bench.yaml, dgx_toolbox.yaml)
- Unsloth-zoo merge bug FIXED (PR #369 + #559 in container version 2026.3.5)
- wp-bench deferred to Phase 4 (live eval after model is served)
- Phase 4 split into Evaluation (4) + Packaging/Deployment (5) with human review gate

### DGX Toolbox Integration
- Added `config/dgx_toolbox.yaml` — configurable path to dgx-toolbox project (transportable across environments)
- Added `scripts/dgx_toolbox.py` — Python resolver for DGX Toolbox components (`get_toolbox().run("vllm")`)
- Path resolution: env var `DGX_TOOLBOX_PATH` > config file > default `~/dgx-toolbox`
- All Phase 3/4 scripts use the resolver — never hardcoded paths
- Training via Unsloth Studio, eval via eval-toolbox, serving via vLLM/LiteLLM/Ollama

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
