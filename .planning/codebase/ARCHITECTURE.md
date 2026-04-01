# Architecture

**Analysis Date:** 2026-03-31

## Pattern Overview

**Overall:** Multi-phase ML pipeline with skill-based orchestration, container-mediated execution, and adaptive resource planning.

**Key Characteristics:**
- Pipeline stages are independent Python scripts invoked via CLI, connected through filesystem-based data handoff (JSON/JSONL files in `data/` subdirectories)
- Claude Code skills (`.claude/skills/`) serve as the orchestration layer, invoking scripts and spawning background telemetry agents
- DGX Toolbox (`scripts/dgx_toolbox.py`) is the execution engine that resolves container paths, validates preconditions, and executes commands inside Docker containers
- All training runs are isolated by dataset ratio, with per-run config overlays and output directories
- Adaptive resource planning reads telemetry JSONL logs between runs and adjusts training config based on thermal/memory history

## Layers

**Skill Layer (Intent + Recovery):**
- Purpose: Define high-level workflows (data pipeline, training, evaluation) with step-by-step procedures and error recovery
- Location: `.claude/skills/wp-finetune:*/SKILL.md`
- Contains: Markdown skill definitions that Claude Code interprets and executes
- Depends on: DGX Toolbox, pipeline scripts, telemetry agents
- Used by: User via Claude Code commands (`/run-training`, `/run-pipeline`, `/observe-training`)
- Key skills: `run-training` (Steps 0-9, adaptive planning), `run-data-pipeline` (orchestrator-driven), `observe-training` (6-agent team), `observe-data-pipeline` (3-agent team), `observe-packaging` (3-agent team), `observe-inference`, `observe-evaluation`, `review-telemetry`

**Execution Engine (DGX Toolbox):**
- Purpose: Resolve container paths, validate state, manage container lifecycle, execute commands inside Docker
- Location: `scripts/dgx_toolbox.py`, `config/dgx_toolbox.yaml`
- Contains: `DGXToolbox` class (singleton via `get_toolbox()`), validation engine with named checks, container lifecycle management, execution with idempotency
- Depends on: Docker CLI, `config/dgx_toolbox.yaml`, external dgx-toolbox project (`~/dgx-toolbox`)
- Used by: All training/eval scripts import `get_toolbox()`, skills call `dgx.validate()`, `dgx.ensure_ready()`, `dgx.execute()`

**Pipeline Orchestrator (State Machine):**
- Purpose: Scan filesystem to determine pipeline state, compute percentage-based targets, produce structured action plans
- Location: `scripts/pipeline_orchestrator.py`
- Contains: State scanning (`get_status()`), dynamic target computation (`compute_targets()` with percentage-based targets + COT_FLOOR minimum), action plan generation (`get_plan()`)
- Depends on: Filesystem state in `data/` directories, `config/repos.yaml`, `config/taxonomy.yaml`
- Used by: `run-data-pipeline` skill reads `plan-json` and dispatches actions as "script" or "agent" types

**Pipeline Scripts (Data Processing):**
- Purpose: Individual data processing stages from extraction through export
- Location: `scripts/*.py`
- Contains: Phase 1 (`phase1_clone.py`, `phase1_extract.py`, `phase1_judge.py`), Phase 2 (`phase2_gap_analysis.py`, `phase2_generate.py`, `phase2_mutate.py`, `phase2_judge.py`, `phase2_judge_dataset.py`), Phase 3 (`phase3_cot.py`), merge (`merge_dataset.py`), export (`export_dataset.py`)
- Depends on: `scripts/utils.py` (shared utilities), Anthropic API (for LLM judging/generation), PHP CLI (for extraction/linting)
- Used by: Pipeline orchestrator dispatches these as "script" actions; skills invoke directly

**Training Scripts (Model Training):**
- Purpose: Model download, tokenizer extension, LoRA SFT training, adapter merge
- Location: `scripts/download_model.py`, `scripts/prepare_tokenizer.py`, `scripts/train_model.py`, `scripts/merge_adapter.py`
- Contains: Unsloth FastLanguageModel loading (BF16, no QLoRA for MoE), LoRA application (r=32, alpha=64), SFTTrainer with MLflow tracking, MemoryWatchdogCallback, checkpoint resume support
- Depends on: DGX Toolbox (container execution), `config/train_config*.yaml`, HuggingFace Hub, Unsloth, TRL, PyTorch
- Used by: `run-training` skill orchestrates these in Steps 4-8 inside Docker containers

**Evaluation Layer:**
- Purpose: Score model outputs against 9-dimension rubric, check quality gates, run wp-bench
- Location: `eval/*.py`
- Contains: Rubric scorer (`rubric_scorer.py` - PHPCS + PHPStan + regex + LLM), rubric definitions (`rubric_definitions.py`), gen eval (`eval_gen.py` - 9-dimension scoring), judge eval (`eval_judge.py` - Spearman correlation), quality gate (`eval_gate.py` - threshold checking)
- Depends on: DGX Toolbox (container execution), PHPCS/PHPStan (code quality tools), OpenAI-compatible API (vLLM endpoint at port 8020), scipy (Spearman correlation)
- Used by: Post-training quality gate, `observe-evaluation` skill

**Telemetry Layer:**
- Purpose: Monitor GPU, thermal, training metrics, disk I/O, checkpoint integrity, container health during long-running operations
- Location: `telemetry/` (runtime output), `.claude/skills/wp-finetune:observe-*/SKILL.md` (definitions)
- Contains: Background agent definitions (6 agents for training, 3 for data pipeline, 3 for packaging), canonical thermal JSONL log, `thermal_history.json`
- Depends on: `nvidia-smi` (HOST, not container -- containers can lose NVML access), `docker` CLI, `/proc/meminfo`, `free`
- Used by: Adaptive resource planning (Step 8.5), `review-telemetry` consolidation

**Configuration Layer:**
- Purpose: Centralize all tunable parameters
- Location: `config/*.yaml`
- Contains: Training hyperparameters (`train_config.yaml`), per-ratio overlays (`train_config_{ratio}.yaml`), LoRA config, eval thresholds, DGX Toolbox mappings (`dgx_toolbox.yaml`), repo definitions (`repos.yaml`), taxonomy (`taxonomy.yaml`), synthetic prompts (`synthetic_prompts.yaml`), wp-bench config (`wp-bench.yaml`)
- Depends on: Nothing (static files, modified by adaptive planning)
- Used by: All scripts and skills

## Data Flow

**Data Pipeline (Extraction to Export):**

1. `scripts/csv_to_repos.py` generates `config/repos.yaml` from source CSV
2. `scripts/phase1_clone.py` clones repos to `data/phase1_extraction/repos/`
3. `scripts/phase1_extract.py` extracts PHP functions via PHP tokenizer to `data/phase1_extraction/output/extracted/*.json`
4. `scripts/phase1_judge.py` (or Claude Code agents in batches of 5 repos) judges functions, splitting into `output/passed/` and `output/failed/`
   - WordPress Core (quality_tier: "core") -> auto-passed, tagged only
   - Others -> PHPCS pre-filter (< 5 errors/100 lines) -> Claude 9-dimension judgment
5. `scripts/phase2_gap_analysis.py` compares tag counts against `config/taxonomy.yaml` minimums, writes `data/phase2_synthetic/gap_report.json`
6. `scripts/phase2_mutate.py` creates contrastive pairs (remove prepare(), strip nonces, strip escaping, remove capability checks, inject SELECT *) in `data/phase2_synthetic/output/mutated/`
7. `scripts/phase2_generate.py` (or agents) fills taxonomy gaps using real code as style anchors in `data/phase2_synthetic/output/generated/`
8. `scripts/phase2_judge.py` / `scripts/phase2_judge_dataset.py` (or agents) assess synthetics into `output/judged/`, generate judge training data in `output/judge_training/`
9. `scripts/phase3_cot.py` (or agents) generates 4 CoT types in `data/phase3_cot/output/`:
   - `cot_gen_pattern` (requirement -> pattern -> implementation -> reasoning)
   - `cot_judge_rubric` (code -> walk 9 dimensions -> scores -> verdict)
   - `cot_judge_contrastive` (bad code -> issues -> fixes -> good version)
   - `cot_security` (security analysis -> nonce/cap/escape -> verdict)
10. `scripts/merge_dataset.py` combines all sources into `data/final_dataset/wordpress_finetune.jsonl`
11. `scripts/export_dataset.py` deduplicates (SHA-256 on assistant content), enforces 40/60 gen/judge ratio, adds sample weights (1.5x for contrastive/low-score), validates PHP lint on sample, splits 80/10/10 train/val/test, writes OpenAI/Alpaca/raw formats + metadata.json

**Training Pipeline (Per-Run, orchestrated by `run-training` skill):**

1. Step 0: User selects base model, dataset ratios, telemetry mode (observe/monitor/none)
2. Step 1: Skill creates run-specific config overlay (`config/train_config_{ratio}.yaml`) with ratio-specific data paths and output dirs
3. Step 2: `dgx.validate(["toolbox", "config", "memory:70"])` checks preconditions
4. Step 3: `dgx.ensure_ready("unsloth_studio")` starts container, mounts project at `/workspace/wp-finetune`, installs pinned deps
5. Step 4: `scripts/download_model.py` downloads base model (idempotent via `idempotency_check`, shared across runs)
6. Step 5: `scripts/prepare_tokenizer.py` extends tokenizer with `<wp_gen>`, `<wp_judge>` tokens (idempotent)
7. Step 6: `scripts/train_model.py --dry-run` validates config, checks memory pre-check (MIN_FREE_MEMORY_GB = 70), serves as warmup probe
8. Step 7: `scripts/train_model.py [--resume] --config {config}` runs SFTTrainer with MemoryWatchdogCallback (threshold: 2 GB)
9. Step 8: `scripts/merge_adapter.py` merges LoRA into base model, verifies special tokens survive roundtrip
10. Step 8.5: Adaptive resource planning (see below)

**Adaptive Resource Planning (Step 8.5):**

```
Collectors (observe agents / monitor)    Canonical log                Downstream consumers
────────────────────────────────        ──────────────               ────────────────────
Lightweight monitor ──┐
                      ├──> {model}_{date}_{ratio}   ──┬──> 8.5a: compute avg/peak metrics
Observe agents ───────┘    _thermal.jsonl              ├──> 8.5b: thermal_history.json (one record per run)
                           (append-only JSONL)         ├──> 8.5c: zone classification (COLD/COOL/WARM/HOT/CRITICAL)
                                                       ├──> 8.5d: CRITICAL backoff (restore last WARM config)
                                                       ├──> 8.5d-mem: OOM recovery (restore last safe config)
                                                       └──> 8.5e: thermal exploitation ladder
```

Thermal zones:
- CRITICAL (>= 83C peak): Pause, backoff to last WARM config, wait for cooldown < 75C
- HOT (78-82C peak): Reduce batch_size by 1, increase grad_accum to maintain eff_batch
- WARM (72-77C peak): Hold current config (target zone)
- COOL (65-71C avg): Apply exploitation ladder
- COLD (< 65C avg): Aggressive exploitation ladder

Exploitation ladder (COOL/COLD, applied in priority order):
1. `prefetch_factor` +1 (cap 4) -- near-zero memory, reduces GPU idle gaps
2. `save_steps` x2 (cap 400) -- zero memory, fewer checkpoint stalls
3. `eval_steps` x2 (cap 200) -- zero memory, fewer eval pauses
4. `batch_size` +1 (last resort, model-scale-aware ceiling, requires warmup probe)

OOM detection: Final telemetry readings show GPU util < 10% while RAM > 95% -> memory backoff (restore last non-OOM config, step down workers by 1, enable persistent_workers)

**Telemetry Flow:**

1. Collectors poll `nvidia-smi` (HOST) / `free` / `docker` at defined intervals (30s for GPU, 60s for training/disk/container, 5m for checkpoints, 10m for lightweight monitor)
2. All collectors append to canonical thermal log: `telemetry/training/{model}_{date}_{ratio}_thermal.jsonl`
   - Schema: `{"ts", "gpu_util", "temp", "vram_used_mb" (null on unified memory), "sys_ram_used_mb", "sys_ram_total_mb", "source"}`
3. Between runs, Step 8.5a reads canonical log, computes metrics
4. Step 8.5b appends summary record to `telemetry/training/thermal_history.json`
5. Step 8.5c-e adjusts next run's config
6. `review-telemetry` skill reads agent markdown reports, produces `_summary.md`

**State Management:**
- Pipeline state is entirely filesystem-based: orchestrator scans `data/` directories to determine current phase
- Checkpoint persistence uses atomic write-and-rename via `scripts/utils.py` (`save_checkpoint()` writes `.tmp` then `rename()`) in `data/checkpoints/`
- Training state managed by HuggingFace Trainer (checkpoint dirs in adapter output)
- Thermal history persists in `telemetry/training/thermal_history.json` as JSON array
- Telemetry stop signal via `_stop` sentinel file -- agents check each cycle and write Final Summary before exiting

## Key Abstractions

**DGXToolbox (Execution Engine):**
- Purpose: Bridge between skill intent and Docker container execution
- Location: `scripts/dgx_toolbox.py`
- Pattern: Singleton (`get_toolbox()`), config-driven container resolution from `config/dgx_toolbox.yaml`, validation engine with named checks (`toolbox`, `training_data`, `config`, `memory:N`, `container:name`, `mounted:name`, `gpu`, `deps:name`), idempotent execution with `idempotency_check` parameter
- Key classes: `DGXToolbox`, `ValidationResult`, `CheckResult`, `ExecResult`

**Pipeline Orchestrator (State Machine):**
- Purpose: Scan filesystem state, compute dynamic targets, produce action plans
- Location: `scripts/pipeline_orchestrator.py`
- Pattern: Pure function over filesystem state -- `get_status()` returns current counts, `compute_targets()` derives percentage-based targets (10% of base counts with COT_FLOOR=500 minimum), `get_plan()` returns structured JSON plan with typed actions ("script" or "agent")

**MemoryWatchdogCallback (Safety):**
- Purpose: Prevent OOM-kill during training by triggering graceful checkpoint save when RAM drops below threshold
- Location: `scripts/train_model.py` (class `MemoryWatchdogCallback`)
- Pattern: HuggingFace `TrainerCallback` reading `/proc/meminfo` every step, sets `control.should_save = True` and `control.should_training_stop = True` when MemAvailable < 2048 MB. Fail-open: returns 999999 if `/proc/meminfo` unreadable.

**Checkpoint Persistence (Resumability):**
- Purpose: Allow pipeline scripts to resume from last successful item after interruption
- Location: `scripts/utils.py` (`load_checkpoint()`, `save_checkpoint()`)
- Pattern: Atomic JSON write (write `.tmp` then `rename()`) storing `{completed: [], failed: [], batch_job_ids: [], timestamp}` per phase in `data/checkpoints/{phase}_checkpoint.json`

**Adaptive Resource Planning (Thermal Exploitation):**
- Purpose: Automatically adjust training hyperparameters between runs based on observed thermal and memory behavior
- Location: `.claude/skills/wp-finetune:run-training/SKILL.md` (Step 8.5)
- Pattern: Read canonical thermal JSONL -> compute aggregates -> classify zone -> apply exploitation ladder or backoff -> write updated config for next run. Persistent memory in `thermal_history.json` enables backoff-to-last-WARM.

**Quality Gate (Evaluation):**
- Purpose: Automated pass/fail decision on trained model
- Location: `eval/eval_gate.py`
- Pattern: Load thresholds from `config/train_config.yaml` `eval` section, load result JSONs, check each metric against threshold, exit 0 (pass) or 1 (fail). Gates: overall mean score, per-dimension gen pass rates, overall Spearman correlation, per-dimension judge correlations, legacy PHPCS/security/spearman thresholds.

**Rubric Scorer (Ground Truth):**
- Purpose: Deterministic code quality scoring for evaluation and judge training
- Location: `eval/rubric_scorer.py`, `eval/rubric_definitions.py`
- Pattern: 4-tool scoring pipeline (PHPCS -> PHPStan -> regex patterns -> LLM checks), 9 weighted dimensions with critical floor rules, grade bands (Excellent/Good/Acceptable/Poor/Bad/Failing)

## Entry Points

**Data Pipeline Entry:**
- Location: `scripts/pipeline_orchestrator.py` (CLI: `status`, `plan`, `plan-json`, `status-json`)
- Triggers: `run-data-pipeline` skill reads `plan-json` output
- Responsibilities: Scan state, compute targets, emit action plan

**Training Entry:**
- Location: `scripts/train_model.py` (via `python -m scripts.train_model [--resume] [--dry-run] [--config PATH]`)
- Triggers: `run-training` skill Step 7, executed inside Docker container via `dgx.execute()`
- Responsibilities: Load config, check memory, load model via Unsloth, apply LoRA, load datasets, train with SFTTrainer

**Evaluation Entry:**
- Location: `eval/eval_gate.py` (via `python -m eval.eval_gate [--results-dir PATH] [--config PATH]`)
- Triggers: Post-training quality gate check
- Responsibilities: Load thresholds from config, load eval results, check all gates, exit 0/1

**DGX Toolbox CLI:**
- Location: `scripts/dgx_toolbox.py` (via `python scripts/dgx_toolbox.py [info|validate|status]`)
- Triggers: Diagnostic use, skill validation steps
- Responsibilities: Show toolbox info, run validation checks, emit structured status report

**Preflight:**
- Location: `scripts/preflight.py` (via `python scripts/preflight.py`)
- Triggers: Before data pipeline scripts
- Responsibilities: Verify ANTHROPIC_API_KEY, php, phpcs, WordPress-Extra standard

## Error Handling

**Strategy:** Fail-fast with diagnostics at validation time; graceful degradation during long-running operations.

**Patterns:**
- **Pre-validation:** DGX Toolbox `validate()` checks toolbox existence, training data, config, memory (>= 70 GB), container state, GPU access, deps before any work begins. Skills call this in Step 2.
- **Container restart loop detection:** `_check_container()` inspects Docker restart count; auto-removes and recreates containers stuck in restart loops (restart_count > 2)
- **Memory watchdog:** `MemoryWatchdogCallback` monitors `/proc/meminfo` every training step; triggers checkpoint save + graceful exit at < 2 GB available
- **OOM detection:** Step 8.5a analyzes final telemetry readings for signs of OOM (GPU idle + RAM > 95%) and triggers memory backoff
- **Thermal guard:** Observe agents touch `_thermal_pause` file when GPU temp >= 83C; skill checks for this after training
- **Checkpoint resumption:** All data pipeline scripts use `load_checkpoint()`/`save_checkpoint()` for per-item progress. Training supports `--resume` to continue from latest checkpoint directory.
- **Idempotency:** `dgx.execute()` accepts `idempotency_check` parameter -- skips execution if output file exists inside container. Training checks for `adapter_config.json` existence before starting.
- **Merge defense-in-depth:** Adapter saved separately by `train_model.py`; merge failure does not lose adapter. Merge verifies special tokens survive roundtrip; failure prints vLLM `--lora-modules` fallback command.
- **API retry:** `scripts/utils.py` `call_with_backoff()` retries Anthropic API calls with exponential backoff (RateLimitError 429, server errors >= 500), respects `retry_after` header.
- **PHP extraction:** 30s timeout per file, returns empty list on timeout/parse error, continues to next file.

## Cross-Cutting Concerns

**Logging:** Console output via `print()` statements with operation labels. MLflow tracks training metrics to local SQLite store (`mlruns.db`). Telemetry agents write structured markdown reports and canonical JSONL thermal logs.

**Validation:** Multi-layer: preflight (external tools), DGX Toolbox validation engine (named checks), dry-run (training config + memory), quality gate (eval thresholds). PHP lint validation on dataset samples via `php -l` in `export_dataset.py`.

**Authentication:** Anthropic API key via `.env` file (loaded by `python-dotenv`). HuggingFace Hub for model download (token via `huggingface-cli login`). vLLM serves locally with `api_key: "none"`. No cloud services required.

**Configuration:** All config in `config/*.yaml`. Training config supports per-ratio overlays (`config/train_config_{ratio}.yaml`). DGX Toolbox config maps component names to container scripts and ports. Eval thresholds live in train_config.yaml `eval` section with fallback defaults.

---

*Architecture analysis: 2026-03-31*
