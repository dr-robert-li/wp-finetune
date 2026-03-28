# Changelog

All notable changes to the wp-qwen3-moe project.

## [Unreleased]

### Dataset Production — Complete (94,630 unique examples)
- **Poor-code corpus:** Added 1,000 poorly-rated plugins (<=3 stars) + 186 poorly-rated themes from WordPress.org to balance good/bad code ratio
- **GitHub URL discovery:** 3-phase process (WP.org scraping, `gh search`, validation) found 983 root repo URLs across all 4 datasets
- **4-way CoT split:** Gen pattern CoT, judge rubric CoT, judge contrastive CoT, shared security CoT — each with 10% minimum floor
- **Judge pool:** 3,956 → 30,498 examples (7.7x increase from full-coverage judge training + poor-code corpus)
- **CoT data:** 610 → 29,020 (47x increase across 4 specialised types)
- **5 ratio exports** at 30/70, 40/60, 50/50, 60/40, 70/30 to `data/final_dataset/ratio_{gen}_{judge}/`
- **repos.yaml:** 56 → 213 repos (top plugins/themes + poor-quality corpus)

### Evaluation Suite — 193-Check Canonical Rubric
- `docs/eval/wp_code_quality_rubric.md` — 193 check IDs (83 positive, 110 negative) across 9 weighted dimensions
- `eval/rubric_definitions.py` — all check IDs, weights, detection methods, automation mappings
- `eval/rubric_scorer.py` — 4-tool ground truth scoring engine (PHPCS, PHPStan, regex, LLM)
- Rewrote `eval/eval_gen.py` with full 9-dimension rubric scoring (not just PHPCS pass rate)
- Rewrote `eval/eval_judge.py` with per-dimension Spearman correlation
- Updated `eval/eval_gate.py` with multi-dimension threshold support
- Critical floor rules: Security/SQL/Structure dimensions have automatic score caps for catastrophic flaws
- Research backing: `research_wpcs_standards.md` (58 WPCS + 39 VIP sniffs), `research_wp_security_sql_perf.md` (security/SQL/perf patterns)

### Execution Engine Architecture
- Refactored `scripts/dgx_toolbox.py` from path resolver into project-agnostic execution engine (639 lines)
- Architecture: Skill (intent + recovery) → dgx_toolbox.py (validate + execute) → Docker commands (dynamic from YAML)
- All 8 project-specific couplings moved to `config/dgx_toolbox.yaml` — Python engine is project-agnostic
- New methods: `validate()`, `ensure_ready()`, `execute()`, `run_service()`, `status_report()`
- Idempotency built into `execute()` via `idempotency_check` parameter
- Container lifecycle: start → wait → mount check → dep install → validate — fully automated
- Removed brittle `run_training_pipeline.sh` — Python engine replaces it

### Agentic Telemetry Framework
- 5 stage-specific observe skills: `/observe-data-pipeline` (3 agents), `/observe-training` (6 agents), `/observe-evaluation` (3 agents), `/observe-packaging` (3 agents), `/observe-inference` (5 agents)
- `/review-telemetry` consolidates agent output into `_summary.md`
- Each agent writes append-only markdown to `telemetry/{stage}/{timestamp}/`
- WARNING/CRITICAL thresholds with concrete numbers (GPU temp > 80C, loss divergence, disk > 85%)
- Stop mechanism via `_stop` file
- Agent team assessment checklist for future skill creators

### Phase 3: Model Prep and Training (At Checkpoint)
- 75 tests passing across 13 test files
- Training scripts: `download_model.py`, `prepare_tokenizer.py`, `train_model.py`, `merge_adapter.py`
- Tokenizer extended with `<wp_gen>` (ID 151669) and `<wp_judge>` (ID 151670), mean-initialized embeddings
- Memory pre-check blocks training if < 70GB available (with actionable diagnostics)
- All steps idempotent: download skips if shards exist, tokenizer skips if tokens present, training skips if adapter exists
- Multi-ratio training support: Step 0 selects ratio export, isolated checkpoint dirs per ratio
- BF16 LoRA (not QLoRA) — MoE router weights incompatible with BitsandBytes 4-bit quantization
- Unsloth-zoo merge bug (PR #369) confirmed fixed in DGX Toolbox container version

### Base Model Switch
- **Switched from Qwen3-8B (dense-to-MoE conversion) to Qwen3-30B-A3B (native MoE)**
- CMoE and ToMoE have no serving stack (no vLLM, no GGUF, no Ollama)
- Qwen3-30B-A3B: verified vLLM, Ollama, HuggingFace, Unsloth support
- ~30B total params, ~3B active per forward pass, 128 experts, top-8 routing
- Fits DGX Spark 128GB unified memory (63GB BF16 with headroom)

### Skills (8 total)
- `run-data-pipeline` — autonomous data pipeline with spawn-until-target pattern
- `run-training` — DGX Spark training with dry-run, base model selection, ratio selection
- `observe-{data-pipeline,training,evaluation,packaging,inference}` — stage-specific telemetry
- `review-telemetry` — aggregation and self-introspection

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
