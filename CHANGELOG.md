# Changelog

All notable changes to the wp-qwen3-moe project.

## [Unreleased]

### Phase 2: Dataset Production (In Progress)
- Switched pipeline execution from Anthropic Batch API to Claude Code agents ($0 LLM cost)
- 52 repos cloned, 47 extracted (28,855 functions), 36 repos judged so far
- 23 repos remaining to judge via Claude Code agents

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
