# Codebase Structure

**Analysis Date:** 2026-03-31

## Directory Layout

```
wp-finetune/
├── .claude/                                # Claude Code skills and settings
│   ├── skills/                             # Skill definitions (orchestration layer)
│   │   ├── wp-finetune:run-training/       # Training pipeline (Steps 0-9 + adaptive planning)
│   │   ├── wp-finetune:run-data-pipeline/  # Data pipeline (orchestrator-driven)
│   │   ├── wp-finetune:observe-training/   # 6-agent telemetry team for training
│   │   ├── wp-finetune:observe-data-pipeline/ # 3-agent telemetry for data pipeline
│   │   ├── wp-finetune:observe-packaging/  # 3-agent telemetry for merge/export
│   │   ├── wp-finetune:observe-inference/  # Telemetry for inference serving
│   │   ├── wp-finetune:observe-evaluation/ # Telemetry for evaluation runs
│   │   └── wp-finetune:review-telemetry/   # Consolidate telemetry into summaries
│   └── settings.local.json                 # Local Claude Code settings
├── config/                                 # All configuration (YAML + prompt files)
│   ├── dgx_toolbox.yaml                    # DGX Toolbox execution engine config
│   ├── train_config.yaml                   # Base training config (LoRA, hyperparams, eval thresholds)
│   ├── train_config_{ratio}.yaml           # Per-ratio training config overlays (30_70, 40_60, etc.)
│   ├── repos.yaml                          # WordPress repository definitions (66KB)
│   ├── taxonomy.yaml                       # Concept taxonomy with minimum_coverage targets
│   ├── synthetic_prompts.yaml              # Prompt templates for synthetic generation
│   ├── judge_system.md                     # 9-dimension judge rubric criteria
│   └── wp-bench.yaml                       # wp-bench evaluation config
├── scripts/                                # Python pipeline and training scripts
│   ├── __init__.py
│   ├── utils.py                            # Shared utilities (JSON extraction, API retry, checkpoints)
│   ├── dgx_toolbox.py                      # DGX Toolbox execution engine
│   ├── pipeline_orchestrator.py            # Pipeline state machine and action planner
│   ├── preflight.py                        # Pre-flight validation (API key, PHP, PHPCS)
│   ├── csv_to_repos.py                     # Generate repos.yaml from CSV
│   ├── phase1_clone.py                     # Clone WordPress repositories
│   ├── phase1_extract.py                   # Extract PHP functions via tokenizer
│   ├── phase1_judge.py                     # Judge code quality (PHPCS + Claude)
│   ├── phase2_gap_analysis.py              # Analyze taxonomy coverage gaps
│   ├── phase2_generate.py                  # Generate synthetic code filling gaps
│   ├── phase2_mutate.py                    # Generate contrastive mutation pairs
│   ├── phase2_judge.py                     # Judge synthetic code
│   ├── phase2_judge_dataset.py             # Create judge training data (rubric-scored)
│   ├── phase3_cot.py                       # Generate CoT reasoning (4 types)
│   ├── merge_dataset.py                    # Merge all sources into wordpress_finetune.jsonl
│   ├── export_dataset.py                   # Deduplicate, ratio-enforce, split, export formats
│   ├── download_model.py                   # Download base model from HuggingFace Hub
│   ├── prepare_tokenizer.py                # Extend tokenizer with <wp_gen>, <wp_judge> tokens
│   ├── train_model.py                      # Unsloth LoRA SFT training with MemoryWatchdog
│   ├── merge_adapter.py                    # Merge LoRA adapter into base model
│   ├── agent_judge.py                      # Agent-based judging helper
│   └── judge_amp_wp.py                     # WordPress AMP judging
├── eval/                                   # Evaluation suite
│   ├── __init__.py
│   ├── rubric_definitions.py               # 9-dimension rubric constants and check registry
│   ├── rubric_scorer.py                    # Ground truth scorer (PHPCS + PHPStan + regex + LLM)
│   ├── eval_gen.py                         # Generation evaluation (9-dimension scoring)
│   ├── eval_judge.py                       # Judge evaluation (Spearman correlation)
│   └── eval_gate.py                        # Quality gate (threshold checking, exit 0/1)
├── tests/                                  # Test suite
│   ├── __init__.py
│   ├── fixtures/                           # Test data files
│   ├── test_config.py
│   ├── test_csv_to_repos.py
│   ├── test_eval_gate.py
│   ├── test_eval_gen.py
│   ├── test_eval_judge.py
│   ├── test_export.py
│   ├── test_phase2_judge_dataset.py
│   ├── test_phase2_mutate.py
│   ├── test_pipeline_integration.py
│   ├── test_preflight.py
│   ├── test_prepare_tokenizer.py
│   ├── test_train_model.py
│   └── test_utils.py
├── data/                                   # All pipeline data (gitignored, large)
│   ├── checkpoints/                        # Atomic checkpoint files per phase
│   ├── phase1_extraction/                  # Phase 1 output
│   │   ├── repos/                          # Cloned WordPress repos (shallow)
│   │   └── output/
│   │       ├── extracted/                  # Raw extracted functions (JSON per repo)
│   │       ├── passed/                     # Quality-assessed passed functions
│   │       └── failed/                     # Functions that failed assessment
│   ├── phase2_synthetic/                   # Phase 2 output
│   │   ├── gap_report.json                 # Coverage gap analysis
│   │   └── output/
│   │       ├── generated/                  # Claude-generated synthetic examples
│   │       ├── judged/                     # Judged synthetic (passed_*.json, failed_*.json)
│   │       ├── mutated/                    # Contrastive mutation pairs
│   │       └── judge_training/             # Rubric-scored judge training data
│   ├── phase3_cot/                         # Phase 3 output
│   │   └── output/                         # CoT reasoning files (4 types)
│   └── final_dataset/                      # Training-ready dataset (all formats)
│       ├── wordpress_finetune.jsonl         # Merged source (before export)
│       ├── metadata.json                   # Dataset statistics + composition
│       ├── openai_{train,val,test}.jsonl    # OpenAI finetuning format
│       ├── alpaca_{train,val,test}.json     # Alpaca/Llama-MoE format
│       ├── raw_{train,val,test}.jsonl       # Full metadata format
│       └── ratio_{N_M}/                    # Per-ratio exports (e.g., ratio_40_60/)
├── adapters/                               # LoRA adapters and extended tokenizer
│   ├── tokenizer/                          # Extended tokenizer with <wp_gen>, <wp_judge>
│   └── qwen3-wp/                           # Default adapter output (or {run_name}/)
│       ├── adapter_config.json             # LoRA adapter config
│       ├── adapter_model.safetensors       # LoRA weights
│       └── checkpoint-*/                   # Training checkpoints
├── models/                                 # Base and merged models
│   ├── Qwen3-30B-A3B/                      # Downloaded base model
│   └── {run_name}-merged/                  # Merged model (base + adapter)
├── telemetry/                              # Runtime telemetry output
│   └── training/                           # Training telemetry
│       ├── {model}_{date}_{ratio}_thermal.jsonl  # Canonical thermal log (JSONL)
│       ├── thermal_history.json            # Persistent thermal history (adaptive planning)
│       └── {timestamp}/                    # Per-run agent reports (observe mode)
│           ├── gpu-metrics.md
│           ├── thermal-throttling.md
│           ├── training-metrics.md
│           ├── disk-io.md
│           ├── checkpoint-integrity.md
│           ├── container-monitor.md
│           ├── _summary.md                 # Consolidated by review-telemetry
│           └── _stop                       # Sentinel file to stop agents
├── mlruns/                                 # MLflow experiment tracking (local SQLite)
│   └── *.yaml                              # Experiment metadata
├── mlruns.db                               # MLflow SQLite database
├── docs/                                   # Documentation
│   └── eval/                               # Evaluation documentation
├── .planning/                              # GSD planning and codebase analysis
│   └── codebase/                           # Codebase mapping documents
├── .env                                    # Environment variables (ANTHROPIC_API_KEY)
├── .env.example                            # Example environment config
├── .gitignore                              # Git ignore rules
├── PROJECT.md                              # Full project specification (phases A-E)
├── README.md                               # Quick start guide
├── CHANGELOG.md                            # Version changelog
├── JOURNAL.md                              # Training journal (77KB, detailed run logs)
└── wp-moe.md                               # Model architecture specification (MoE design)
```

## Directory Purposes

**`.claude/skills/`**
- Purpose: Skill-based orchestration layer for Claude Code -- each skill defines a multi-step workflow
- Contains: SKILL.md files with step-by-step procedures, agent prompts, error recovery logic
- Key files: `wp-finetune:run-training/SKILL.md` (the main training skill, ~900 lines, Steps 0-9 with adaptive planning), `wp-finetune:run-data-pipeline/SKILL.md` (orchestrator-driven data pipeline)

**`config/`**
- Purpose: All configuration files controlling pipeline behavior, training hyperparameters, and evaluation thresholds
- Contains: YAML configs, prompt markdown
- Key files: `train_config.yaml` (base training config with LoRA/hyperparams/eval), `dgx_toolbox.yaml` (container mappings, ports, pinned deps, validation paths), `repos.yaml` (66KB WordPress repo definitions), `taxonomy.yaml` (concept coverage targets), `judge_system.md` (9-dimension rubric)

**`scripts/`**
- Purpose: All Python pipeline scripts -- data processing, training, and utilities
- Contains: Phase 1-3 pipeline scripts, training scripts, DGX Toolbox execution engine, pipeline orchestrator
- Key files: `dgx_toolbox.py` (execution engine singleton), `pipeline_orchestrator.py` (state machine), `train_model.py` (SFTTrainer with MemoryWatchdog), `utils.py` (shared utilities)

**`eval/`**
- Purpose: Evaluation suite -- rubric scoring, gen/judge evaluation, quality gate
- Contains: Ground truth scorer, rubric definitions, per-dimension evaluation, threshold checking
- Key files: `rubric_scorer.py` (4-tool scoring: PHPCS + PHPStan + regex + LLM), `eval_gate.py` (quality gate, exit 0/1)

**`data/`**
- Purpose: All pipeline data -- extraction output, synthetic generation, CoT reasoning, final datasets
- Contains: Per-phase output directories, checkpoint files, final training data in multiple formats
- Key files: `final_dataset/openai_train.jsonl` (primary training data), `checkpoints/{phase}_checkpoint.json` (resumability)

**`adapters/`**
- Purpose: LoRA adapters and extended tokenizer produced by training
- Contains: Extended tokenizer dir, per-run adapter directories with checkpoints
- Key files: `tokenizer/tokenizer_config.json`, `qwen3-wp/adapter_config.json`, `qwen3-wp/adapter_model.safetensors`

**`models/`**
- Purpose: Downloaded base models and merged (base + adapter) models
- Contains: HuggingFace model downloads, merged model outputs
- Key files: `Qwen3-30B-A3B/config.json`, `{run_name}-merged/config.json`

**`telemetry/`**
- Purpose: Runtime telemetry from background observer agents and adaptive planning history
- Contains: Canonical thermal JSONL logs, per-run agent markdown reports, thermal history JSON
- Key files: `training/thermal_history.json` (persistent cross-run memory for adaptive planning), `training/*_thermal.jsonl` (canonical thermal logs)

## Key File Locations

**Entry Points:**
- `scripts/pipeline_orchestrator.py`: Data pipeline state + action planning (CLI: status, plan, plan-json)
- `scripts/train_model.py`: Training entry point (--resume, --dry-run, --config)
- `scripts/dgx_toolbox.py`: Execution engine (CLI: info, validate, status)
- `scripts/preflight.py`: Pre-flight validation
- `eval/eval_gate.py`: Quality gate (exit 0/1)

**Configuration:**
- `config/train_config.yaml`: Base training config (model, LoRA, training, eval thresholds)
- `config/dgx_toolbox.yaml`: Container mappings, ports, pinned deps, validation paths
- `config/repos.yaml`: WordPress repository definitions with quality_tier and path filters
- `config/taxonomy.yaml`: Concept taxonomy with minimum_coverage targets
- `config/wp-bench.yaml`: wp-bench evaluation config

**Core Logic:**
- `scripts/dgx_toolbox.py`: DGXToolbox class -- resolve, validate, ensure_ready, execute, status_report
- `scripts/pipeline_orchestrator.py`: get_status(), compute_targets(), get_plan()
- `scripts/train_model.py`: load_model_and_tokenizer(), apply_lora(), build_trainer(), MemoryWatchdogCallback
- `scripts/utils.py`: extract_json(), call_with_backoff(), load_checkpoint(), save_checkpoint(), batch helpers
- `eval/rubric_scorer.py`: score_code() -- 4-tool ground truth scoring
- `eval/eval_gate.py`: check_gates() -- threshold comparison

**Testing:**
- `tests/test_*.py`: Unit tests for most scripts and eval modules
- `tests/fixtures/`: Test data files

## Naming Conventions

**Files:**
- Pipeline scripts: `phase{N}_{operation}.py` (e.g., `phase1_extract.py`, `phase2_generate.py`)
- Training scripts: `{verb}_{noun}.py` (e.g., `download_model.py`, `train_model.py`, `merge_adapter.py`)
- Config files: `{purpose}.yaml` or `{purpose}.md` (e.g., `train_config.yaml`, `judge_system.md`)
- Per-ratio configs: `train_config_{gen}_{judge}.yaml` (e.g., `train_config_40_60.yaml`)
- Test files: `test_{module_name}.py` matching the script they test
- Skills: `wp-finetune:{verb}-{noun}/SKILL.md` (e.g., `run-training`, `observe-training`, `review-telemetry`)

**Directories:**
- Phase data: `phase{N}_{operation}/` (e.g., `phase1_extraction`, `phase2_synthetic`, `phase3_cot`)
- Output grouping: `passed/`, `failed/`, `extracted/`, `generated/`, `mutated/`, `judged/`, `judge_training/`
- Per-run isolation: `adapters/{run_name}/`, `models/{run_name}-merged/`
- Telemetry: `telemetry/{stage}/{timestamp}/`

**Variables (Python):**
- Module-level path constants: `SCREAMING_SNAKE_CASE` (e.g., `PROJECT_ROOT`, `REPOS_DIR`, `EXTRACTED_DIR`)
- Functions: `snake_case` (e.g., `extract_repo()`, `load_config()`, `check_memory()`)
- Classes: `PascalCase` (e.g., `DGXToolbox`, `MemoryWatchdogCallback`, `ValidationResult`)
- Config keys: `snake_case` from YAML (e.g., `quality_tier`, `per_device_train_batch_size`)

## Where to Add New Code

**New Data Pipeline Phase:**
- Create `scripts/phase{N}_{operation}.py`
- Follow pattern: define `PROJECT_ROOT`, input/output paths as module constants, use `load_checkpoint()`/`save_checkpoint()` for resumability
- Register in `scripts/pipeline_orchestrator.py` `get_plan()` with appropriate action type ("script" or "agent")
- Add to `run-data-pipeline` skill if agent-based

**New Training Script:**
- Create `scripts/{verb}_{noun}.py`
- Import `from scripts.dgx_toolbox import get_toolbox` (establishes DGX resolver pattern)
- Define `PROJECT_ROOT`, `CONFIG_PATH`, `load_config()` following existing pattern
- Add CLI with `argparse`, support `--config` and `--dry-run` where appropriate
- Register execution in `run-training` skill SKILL.md at appropriate step

**New Evaluation Metric:**
- Add to `eval/rubric_definitions.py` (dimension weights, check registry)
- Update `eval/rubric_scorer.py` scoring logic
- Add threshold to `config/train_config.yaml` `eval` section
- Update `eval/eval_gate.py` `check_gates()` to include new gate

**New Skill:**
- Create `.claude/skills/wp-finetune:{verb}-{noun}/SKILL.md`
- Follow existing skill structure: Trigger, Process (numbered steps), agent prompts with STOP conditions
- For observe skills: define agent team, polling intervals, markdown output format, `_stop` sentinel

**New Telemetry Agent:**
- Add agent definition to the appropriate `observe-*` skill SKILL.md
- Ensure agent writes to the run's telemetry directory (`{TDIR}/`)
- For thermal data: append JSONL to canonical thermal log (`$THERMAL_LOG`)
- Include `_stop` file check in the loop for graceful shutdown

**New Test:**
- Create `tests/test_{module_name}.py`
- Place test fixtures in `tests/fixtures/`
- Follow existing pattern: use `pytest`, mock external dependencies (Docker, API calls, filesystem)

**New Config Parameter:**
- Add to appropriate YAML file (`config/train_config.yaml` for training, `config/dgx_toolbox.yaml` for containers)
- Update the script that reads it with `.get()` and sensible default
- Document in relevant skill SKILL.md

## Special Directories

**`data/phase1_extraction/repos/`**
- Purpose: Shallow-cloned WordPress git repositories
- Generated: Yes (by `phase1_clone.py`)
- Committed: No (gitignored, large)
- Safe to delete: Yes, re-cloned on next run

**`data/checkpoints/`**
- Purpose: Atomic checkpoint files enabling pipeline resumability
- Generated: Yes (by `scripts/utils.py` `save_checkpoint()`)
- Committed: No (runtime state)
- Format: `{phase}_checkpoint.json` with `{completed, failed, batch_job_ids, timestamp}`

**`adapters/`**
- Purpose: LoRA adapters and extended tokenizer (training output)
- Generated: Yes (by `train_model.py`, `prepare_tokenizer.py`)
- Committed: No (large model files)
- Key patterns: `adapters/{run_name}/checkpoint-*/` for training checkpoints, `adapters/tokenizer/` for extended tokenizer

**`models/`**
- Purpose: Downloaded base models and merged outputs
- Generated: Yes (by `download_model.py`, `merge_adapter.py`)
- Committed: No (very large, 60+ GB per model)

**`telemetry/`**
- Purpose: Runtime telemetry from observer agents and adaptive planning
- Generated: Yes (by observe-* skills and lightweight monitor)
- Committed: Partially (thermal_history.json is valuable across sessions)
- Key files: `training/thermal_history.json` (persistent adaptive planning memory)

**`mlruns/` and `mlruns.db`**
- Purpose: MLflow experiment tracking (local SQLite store)
- Generated: Yes (by `train_model.py` via MLflow)
- Committed: `mlruns.db` is committed (1.5 MB), `mlruns/` metadata committed
- Access: `mlflow ui --backend-store-uri sqlite:///mlruns.db`

**`.planning/codebase/`**
- Purpose: Codebase analysis documents produced by `/gsd:map-codebase`
- Generated: Yes (by GSD commands)
- Committed: Yes (documentation for planning/execution)

**`data/final_dataset/ratio_{N_M}/`**
- Purpose: Per-ratio dataset exports for multi-run training comparisons
- Generated: Yes (by `export_dataset.py` with ratio-specific config)
- Committed: No (training data)
- Pattern: `ratio_30_70/`, `ratio_40_60/`, `ratio_50_50/`, etc.

---

*Structure analysis: 2026-03-31*
