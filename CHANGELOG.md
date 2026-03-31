# Changelog

All notable changes to the wp-qwen3-moe project. Follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Training in progress: 5 sequential LoRA runs on DGX Spark (Run 1 complete, Run 2 resuming)
- Phase 4 (Evaluation) and Phase 5 (Packaging & Deployment) not started

## [0.5.3] - 2026-03-31 — OOM Recovery & Memory-Aware Adaptive Planning

### Added
- **Memory watchdog callback (`MemoryWatchdogCallback`):** Reads `/proc/meminfo` every training step. When `MemAvailable` drops below 2 GB, triggers graceful checkpoint save and clean exit before the OOM killer strikes. Fail-open: if `/proc/meminfo` is unreadable, training continues uninterrupted. Prevents the up-to-200-steps loss observed in Run 2's OOM kills.
- **OOM-aware adaptive planning (Step 8.5):** Detects OOM from telemetry (GPU idle + RAM >95% in final readings). OOM overrides thermal classification — skips scaling and jumps to memory backoff.
- **Memory backoff step (8.5d-mem):** On OOM detection, restores last non-OOM config, steps workers down by 1, force-enables `dataloader_persistent_workers`.
- **`dataloader_persistent_workers` support:** Passthrough added to `SFTConfig` in `train_model.py`. Workers stay alive between epochs instead of respawning, eliminating the sawtooth allocation spikes that caused Run 2's memory creep.

### Changed
- **Adaptive headroom calculation uses peak RAM:** `effective_headroom_gb` uses peak RAM with 5 GB safety margin instead of average-based headroom. On unified memory, the peak spikes (10–20 GB above mean) are what trigger OOM.
- **Batch cap lowered to 8 on unified memory:** Previous cap of 16 was unsafe on DGX Spark's shared memory pool.
- **Worker scaling conservative:** Workers now scale +1 at a time (not doubling), hard-capped at 6 on unified memory, decreased if headroom <10 GB.
- **Thermal history records expanded:** Now include `peak_ram_gb`, `p95_ram_gb`, `safe_headroom_gb`, `effective_headroom_gb`, `likely_oom`, and `dataloader_persistent_workers`.
- **Training config rolled back from adaptive overshoot:** `per_device_train_batch_size` 8→4, `gradient_accumulation_steps` 2→4, `dataloader_num_workers` 8→6 (effective batch size unchanged at 16).

### Fixed
- **Run 2 (40/60) OOM crash:** Adaptive planning after Run 1 doubled batch size and workers based on COOL thermal zone, but didn't account for unified memory pressure from dataloader worker respawns. Two OOM kills at 97% and 99.9% RAM, with driver-level deadlock (DGX Spark known issue — no clean CUDA OOM, system freezes instead). Last valid checkpoint: step 2200/5084. Config rolled back and memory safeguards added.

## [0.5.2] - 2026-03-30 — Canonical Thermal Log & Telemetry Modes

### Added
- **Canonical thermal JSONL log:** All telemetry collectors (observe agents and lightweight monitor) append to `{model}_{date}_{ratio}_thermal.jsonl` — single source of truth for adaptive resource planning
- **Three telemetry modes (Step 0c):** Observe agents (default, full 6-agent team), lightweight monitor (single agent, thermal only), or none (double-confirm warning)
- **Lightweight monitor agent:** Polls nvidia-smi every 10 min, appends to canonical JSONL — sufficient for adaptive planning without the overhead of 6 agents

### Changed
- **Observe/review skills embedded in run-training:** Steps 4/7/8 spawn observe agents inline with concrete `Agent()` blocks and full lifecycle (spawn → execute → `_stop` → review). No separate invocation needed.
- **Step 0c expanded:** From binary on/off to three-way mode selection with `$OBSERVE`/`$MONITOR`/`$TELEMETRY` flags. All step gates updated accordingly.
- **Step 8.5a reads canonical JSONL:** No longer parses markdown — `json.loads()` on each line of the thermal log, source-agnostic
- **Thermal log naming:** `{model_short}_{date}_{ratio}` instead of just `{ratio}` to avoid ambiguity across training sessions
- **Unified memory support:** JSONL schema tracks both `vram_used_mb` (discrete GPU) and `sys_ram_used_mb`/`sys_ram_total_mb` (always available). On unified memory systems (GB10/Grace Hopper) where VRAM reports `[N/A]`, system RAM is used for all memory headroom calculations. Observe agents and lightweight monitor both collect system RAM via `free -m`.
- **Memory thresholds:** WARNING at >90%, CRITICAL at >=98% (warn+log only — does NOT stop training). Insufficient memory is caught pre-training by Step 2 validate and Step 6 dry run.

### Fixed
- **nvidia-smi queries run on host, not container:** Long-running containers lose NVML access (observed after 31hrs on unsloth-headless) while host nvidia-smi stays reliable. Lightweight monitor was using `docker exec nvidia-smi` — fixed to query from host. Explicit `HOST, not docker exec` notes added to both skills.

## [0.5.1] - 2026-03-29 — Adaptive Resource Planning & MLflow

### Added
- **Adaptive resource planning (Step 8.5):** Between sequential training runs, telemetry is parsed to classify GPU thermal zone (COLD/COOL/WARM/HOT/CRITICAL) and auto-adjust batch_size, grad_accum, and dataloader_num_workers for the next run
- **Thermal history (`thermal_history.json`):** Persistent record of each run's config and thermal outcome — survives context resets, enables backoff-to-last-WARM on CRITICAL events
- **CRITICAL backoff:** Instead of blind halving, restores the exact config from the last run that registered WARM (72-77°C peak) — the last known-safe operating point
- **Live thermal guard:** observe-training agent touches `_thermal_pause` at ≥83°C, orchestrator applies CRITICAL rules before next run
- **Telemetry default-on:** Step 0c now defaults to enabled with double-confirmation required to disable, since adaptive resource planning depends on it
- **MLflow integration:** Replaced W&B (cloud) with MLflow (local sqlite at `mlruns.db`) — zero cloud dependencies for training telemetry
- **`formatting_func`** for Unsloth SFTTrainer: converts OpenAI chat format to model chat template

### Changed
- Training config optimized based on telemetry: batch_size 1→4→8, grad_accum 8→4→2, workers 0→4→8 (GPU util improved from ~35% to ~77% avg)
- Container name updated to `unsloth-headless` (no Studio web UI needed for training)
- `extra_special_tokens` format fixed in saved tokenizer (list→dict for transformers 4.56.2 compat)

### Fixed
- Model download: 7 corrupt shards from interrupted download detected and re-downloaded
- W&B auth blocker: removed all cloud-hosted dependencies from scripts, skills, and config

## [0.5.0] - 2026-03-29 — Training Commenced

### Added
- **Training commenced:** 5 sequential runs (30/70, 40/60, 50/50, 60/40, 70/30) on DGX Spark
- Each run produces isolated adapter in `adapters/qwen3-30b-wp-{ratio}/` for A/B/C/D/E eval comparison
- Multi-ratio training workflow: Step 0a model selection, 0b ratio selection, 0c telemetry opt-in, 0d confirmation gate
- Telemetry integration: observe-training (6 agents) during training, review-telemetry between runs
- `wp-moe.md` rewritten to v2.0 reflecting current project state

## [0.4.0] - 2026-03-29 — Dataset Complete (267K merged, 5 ratio exports)

### Added
- **Poor-code corpus:** 1,000 poorly-rated plugins (<=3 stars) + 186 poorly-rated themes from WordPress.org
- **GitHub URL discovery:** 3-phase process (WP.org scraping, `gh search`, validation) — 983 root repo URLs across 4 datasets
- **4-way CoT split:** Gen pattern CoT, judge rubric CoT, judge contrastive CoT, shared security CoT — each with max(500, 10%) floor
- **Percentage-based targets** — all pipeline targets derive from actual data counts, not hardcoded numbers
- **5 ratio exports** at 30/70, 40/60, 50/50, 60/40, 70/30 — from 43K to 102K examples per export

### Changed
- `config/repos.yaml` expanded from 56 → 236 repos (top-quality + poor-quality corpus)
- Pipeline orchestrator rewritten with 4-way CoT actions
- Judge pool: 3,956 → 30,498 examples (7.7x increase)
- CoT data: 610 → 29,020 examples (47x increase across 4 types)
- Total dataset: 5,868 → up to 101,660 depending on ratio

### Fixed
- Double-brace template artifact in synthetic generation (1,909 functions recovered)

## [0.3.2] - 2026-03-28 — Agentic Telemetry Framework

### Added
- 5 stage-specific observe skills: `/wp-finetune:observe-data-pipeline` (3 agents), `/wp-finetune:observe-training` (6 agents), `/wp-finetune:observe-evaluation` (3 agents), `/wp-finetune:observe-packaging` (3 agents), `/wp-finetune:observe-inference` (5 agents)
- `/wp-finetune:review-telemetry` consolidates agent output into `_summary.md`
- Each agent writes append-only markdown to `telemetry/{stage}/{timestamp}/`
- WARNING/CRITICAL thresholds with concrete numbers (GPU temp > 80C, loss divergence, disk > 85%)
- Stop mechanism via `_stop` file
- Agent team assessment checklist for future skill creators

## [0.3.1] - 2026-03-28 — Execution Engine Architecture

### Added
- `scripts/dgx_toolbox.py` refactored into project-agnostic execution engine (639 lines)
- New methods: `validate()`, `ensure_ready()`, `execute()`, `run_service()`, `status_report()`
- Idempotency built into `execute()` via `idempotency_check` parameter
- Container lifecycle: start → wait → mount check → dep install → validate — fully automated

### Changed
- All 8 project-specific couplings moved from Python to `config/dgx_toolbox.yaml`
- Architecture: Skill (intent + recovery) → dgx_toolbox.py (validate + execute) → Docker commands (dynamic from YAML)

### Removed
- Brittle `run_training_pipeline.sh` — Python engine replaces it

## [0.3.0] - 2026-03-28 — Model Prep and Training Scripts

### Added
- Training scripts: `download_model.py`, `prepare_tokenizer.py`, `train_model.py`, `merge_adapter.py`
- Eval scripts: `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`
- Eval rubric: `docs/eval/wp_code_quality_rubric.md` — 241 check IDs (105 positive, 136 negative) across 9 weighted dimensions
- `eval/rubric_definitions.py` — all check IDs, weights, detection methods, automation mappings
- `eval/rubric_scorer.py` — 4-tool ground truth scoring engine (PHPCS, PHPStan, regex, LLM)
- Research backing: `research_wpcs_standards.md`, `research_wp_security_sql_perf.md`
- `config/train_config.yaml` — externalized training hyperparameters
- `config/wp-bench.yaml` — evaluation benchmark config
- Tokenizer extended with `<wp_gen>` (ID 151669) and `<wp_judge>` (ID 151670), mean-initialized embeddings
- Memory pre-check blocks training if < 70GB available (with actionable diagnostics)
- All training steps idempotent: download skips if shards exist, tokenizer skips if tokens present
- 75 tests passing across 13 test files
- Critical floor rules: Security/SQL/Structure dimensions have automatic score caps for catastrophic flaws

### Changed
- **Base model switched from Qwen3-8B (dense-to-MoE conversion) to Qwen3-30B-A3B (native MoE)**
- CMoE and ToMoE rejected: no serving stack (no vLLM, no GGUF, no Ollama)
- BF16 LoRA (not QLoRA) — MoE router weights incompatible with BitsandBytes 4-bit quantization
- Phase 4 split into Evaluation (4) + Packaging/Deployment (5) with human review gate
- wp-bench deferred to Phase 4 (live eval after model is served)

### Fixed
- Unsloth-zoo merge bug (PR #369 + #559) confirmed fixed in DGX Toolbox container version 2026.3.5

## [0.2.0] - 2026-03-26 — Pipeline Ready

### Added
- `scripts/utils.py` with 9 shared functions: extract_json (4-strategy fallback), call_with_backoff (exponential + retry-after), checkpoint save/load (atomic rename), Batch API routing (threshold=50)
- `scripts/preflight.py` validating PHPCS, PHP CLI, and API key
- `scripts/csv_to_repos.py` converting ranked CSV data to repos.yaml
- `config/repos.yaml` with 56 repos (1 core + 49 plugins + 6 themes) with auto-assigned quality_tier from vulnerability data
- `config/judge_system.md`: threshold >= 8, security auto-FAIL (dim < 5), N/A deflated to 7
- Rejection templates in `config/synthetic_prompts.yaml` (proactive nonce, capability, escaping)
- PHPCS hard-fail guard to `phase2_mutate.py`
- `export_dataset.py` with gen/judge ratio enforcement, deduplication, PHP lint, sample_weight, metadata.json
- python-dotenv for API key loading from `.env`
- 46 passing tests total

### Changed
- Hardened all 8 pipeline scripts with utils.py integration

## [0.1.0] - 2026-03-26 — Project Initialization

### Added
- Initial pipeline scripts (10 scripts) and configuration files (4 configs)
- GSD project structure with 4-phase roadmap and 37 requirements
- Codebase mapping (7 documents), domain research (5 documents)
- DGX Toolbox references (Unsloth Studio, vLLM, Ollama, eval-toolbox, safety harness)

### Changed
- Base model updated from LLaMA-MoE to Qwen3-8B throughout all documentation
