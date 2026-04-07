# wp-qwen3-moe

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model: Qwen3-30B-A3B](https://img.shields.io/badge/Base_Model-Qwen3--30B--A3B-purple.svg)](https://huggingface.co/Qwen/Qwen3-30B-A3B)
[![Training: BF16 LoRA](https://img.shields.io/badge/Training-BF16_LoRA-green.svg)]()
[![Infrastructure: DGX Spark](https://img.shields.io/badge/Infrastructure-DGX_Spark-76b900.svg)](https://github.com/dr-robert-li/dgx-toolbox)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-orange.svg)](https://claude.com/claude-code)

**Author:** [Dr. Robert Li](https://github.com/dr-robert-li)

An open-weight Mixture-of-Experts model that generates and judges WordPress code according to strict WordPress Coding Standards. A single model, two modes: `<wp_gen>` for code generation, `<wp_judge>` for structured critique with 9-dimension rubric scoring.

No open-source model existed for this. The tools in this space are wrappers around closed-source frontier models (OpenAI, Claude, etc.). This project builds one from scratch — open weights, self-hostable, no vendor lock-in.

## Architecture

| Property | Value |
|----------|-------|
| Base model | Qwen3-30B-A3B (native MoE, 128 experts, top-8 routing) |
| Total params | ~30B |
| Active params | ~3B per forward pass |
| Task routing | First-token: `<wp_gen>` or `<wp_judge>` |
| Training | LoRA SFT via Unsloth on DGX Spark |
| Serving | vLLM, Ollama, GGUF, AWQ |
| Infrastructure | [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) |

See [wp-moe.md](wp-moe.md) for the full model specification.

## Usage

```python
# Generation mode
prompt = "<wp_gen> Create a custom REST API endpoint for retrieving posts by taxonomy with permission checks"

# Judge mode
prompt = "<wp_judge> Rate this function on WPCS compliance, security, and performance:\n```php\n...\n```"
```

The judge returns structured scores across 9 dimensions: WPCS compliance, SQL safety, security, performance, WP API usage, code quality, dependencies, i18n, and accessibility. The model is deliberately opinionated — it pushes back on poor architectural decisions.

## Project Status

| Milestone | Phases | Status |
|-----------|--------|--------|
| v1.0 MVP | 1. Pipeline Ready | Complete |
| | 2. Dataset Production (267K examples, 5 ratio exports) | Complete |
| | 3. Model Prep & Training (60/40 LoRA complete, 43h on DGX Spark) | Complete |
| | 4. Eval Triage — 30/70 wins (gen 0.99+, judge Spearman 0.57) | **Complete** |
| | 5. Packaging & Deployment | Deferred to v3.0 |
| v1.1 Adaptive Training | 6. Adaptive Training Planner (power-primary, memory watchdog) | Complete |
| v1.2 Judge Reasoning | 4.1 Seed Curation + Data Gen → 4.2 Dataset Assembly → 4.3 Reasoning Fine-Tune → 4.4 Eval & Merge | **Next** |
| v2.0 MoE-Sieve | 7. Router Profiling + Ratio Selection (E_eff) → 8. Selective Training → 9. Eval | Planned |
| v3.0 GRPO & Deploy | 10. Rewards → 11. Dual-mode GRPO (gen + judge) → 12. Merge + Prune → 13. Eval → 14. Package | Planned |

**Current:** Phase 4 triage complete — 30/70 is the winning ratio (only adapter producing parseable judge output, Spearman 0.57). Gen is solved across all ratios (97-100% PHPCS). Judge is the bottleneck. v1.2 Phase 4.1 begins next with seed curation + deep judge CoT data generation on the 30/70 adapter. v3.0 Phase 11 GRPO now targets both gen and judge reasoning (judge receives equal or greater budget).

**Building in public.** Read the [Engineering Journal](JOURNAL.md) for real-time decisions, tradeoffs, failures, and lessons learned as the project evolves.

See [PROJECT.md](PROJECT.md) for full phase details and success criteria.

## Data Pipeline

All LLM-heavy pipeline steps run via **Claude Code agents** (covered by subscription) instead of direct API calls. This eliminates per-token cost entirely and enables parallel batch processing with full agent context. See [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md) for the execution model.

Non-LLM steps (cloning, extraction, gap analysis, mutations, export) run as regular Python scripts.

```
Phase 1: Extract & Assess          Phase 2: Synthetic + Judge Data       Phase 3: CoT + Export
─────────────────────────          ───────────────────────────────       ────────────────────
repos.yaml                         gap_report.json                      All passed examples
    │                                  │                                     │
    ▼                                  ▼                                     ▼
Clone repos ──► Extract ──►       Generate synthetic ──►                CoT reasoning
                    │              (style-grounded)    │                     │
                    ▼                   │              ▼                     ▼
              PHPCS pre-filter         ▼          Judge synthetic      Instruction synthesis
                    │             Mutate real code      │                    │
                    ▼              (contrastive)        ▼                    ▼
         Agent judge / static         │          Judge training       Merge all + judge data
         heuristic judge ──►          ▼          data generator            │
              passed/failed       bad→good pairs       │                    ▼
                                                       ▼              Export with task tokens
              WP Core ──►                         <wp_judge> data     ──► train/val/test splits
              auto-passed
```

**Sources:** Top 1000 plugins + top 100 themes (high-quality generation data), 1000 poorly-rated plugins + 186 poorly-rated themes (negative judge data), plus WordPress Core as reference implementation. GitHub URLs discovered via 3-phase process (WP.org scraping, `gh search`, validation).

**Quality gates:** Every non-core example passes PHPCS pre-filtering AND 9-dimension rubric assessment (threshold >= 8, security auto-FAIL below 5). WordPress Core functions are auto-passed as the reference implementation.

### Dataset Composition (Actual)

267K merged examples (134K judged functions + 143K judge training + 29K CoT), exported at 5 gen/judge ratios after dedup:

| Ratio | Gen | Judge | Total | Train |
|-------|-----|-------|-------|-------|
| 30/70 | 13,071 | 30,498 | 43,569 | 34,855 |
| 40/60 | 20,332 | 30,498 | 50,830 | 40,664 |
| 50/50 | 30,498 | 30,498 | 60,996 | 48,796 |
| 60/40 | 45,747 | 30,498 | 76,245 | 60,996 |
| 70/30 | 71,162 | 30,498 | 101,660 | 81,328 |

**4-way CoT split:** Gen pattern CoT, judge rubric CoT, judge contrastive CoT, shared security CoT — each with 10% minimum floor and 500-example minimum.

## Success Criteria

| Metric | Target |
|--------|--------|
| Generator PHPCS pass rate | > 95% |
| Generator security pass rate | > 98% |
| Judge Spearman correlation | > 0.85 |
| Judge classification precision | > 0.90 |
| Active parameters per inference | ~3B |

## Project Structure

```
wp-finetune/
├── config/
│   ├── repos.yaml                  # 236 repos (top + poor-quality plugins/themes)
│   ├── judge_system.md             # 9-dimension judge rubric (threshold >= 8)
│   ├── taxonomy.yaml               # 87 concept tags + coverage minimums
│   ├── synthetic_prompts.yaml      # Generation templates + rejection examples
│   ├── train_config.yaml           # Training hyperparameters (LoRA, scheduler, etc.)
│   ├── wp-bench.yaml               # Evaluation benchmark config
│   └── dgx_toolbox.yaml            # DGX Toolbox execution engine config (project-agnostic)
├── scripts/
│   ├── utils.py                    # Shared utilities (JSON parsing, backoff, checkpoints)
│   ├── dgx_toolbox.py              # Execution engine: validate → resolve → Docker exec
│   ├── pipeline_orchestrator.py    # Pipeline state tracker + action planner
│   ├── download_model.py           # Download base model from HuggingFace
│   ├── prepare_tokenizer.py        # Extend tokenizer with <wp_gen>/<wp_judge>
│   ├── train_model.py              # BF16 LoRA SFT with memory pre-check + OOM watchdog
│   ├── merge_adapter.py            # Merge adapter with verification roundtrip
│   ├── adaptive_planner.py         # Power-primary thermal exploitation ladder
│   ├── profile_base_model.py       # E_eff routing concentration profiler (MoE-Sieve)
│   ├── triage_ratios.py            # GATE-02 elimination logic for eval triage
│   ├── run_eval_triage.py          # Phase 4 orchestrator: profiling + eval + triage
│   ├── phase1_{clone,extract,judge}.py
│   ├── phase2_{gap_analysis,mutate,generate,judge,judge_dataset}.py
│   ├── phase3_cot.py, merge_dataset.py, export_dataset.py
│   └── (+ csv_to_repos, preflight)
├── eval/
│   ├── rubric_definitions.py       # 193 check IDs across 9 weighted dimensions
│   ├── rubric_scorer.py            # 4-tool ground truth scoring engine
│   ├── eval_gen.py                 # Generator eval (9-dimension rubric scoring)
│   ├── eval_judge.py               # Judge eval (per-dimension Spearman correlation)
│   └── eval_gate.py                # Quality gate (pass/fail against thresholds)
├── docs/
│   ├── AGENT_PIPELINE.md           # Agent execution model and output format contracts
│   ├── wp-finetune:run-data-pipeline.md   # Skill: autonomous data pipeline
│   ├── wp-finetune:run-training.md        # Skill: DGX Spark training pipeline
│   ├── wp-finetune:observe-{stage}.md     # Telemetry skills (5 stages, 3-6 agents each)
│   └── wp-finetune:review-telemetry.md    # Telemetry aggregation and summary
├── docs/eval/
│   ├── wp_code_quality_rubric.md   # 241-check canonical rubric (9 dimensions, weighted)
│   ├── research_wpcs_standards.md  # WPCS + VIP sniff reference
│   └── research_wp_security_sql_perf.md  # Security, SQL, performance patterns
├── data/
│   ├── phase1_extraction/          # Cloned repos + extracted/passed/failed functions
│   ├── phase2_synthetic/           # Gap reports + synthetic/mutated/judge training data
│   ├── phase3_cot/                 # CoT reasoning checkpoints
│   ├── final_dataset/              # Train/val/test in OpenAI, Alpaca, Raw JSONL formats
│   └── checkpoints/                # Pipeline execution checkpoints
├── tests/                          # 126 tests (15 test files, incl. 51 E_eff + triage)
├── PROJECT.md                      # Full project specification
├── JOURNAL.md                      # Engineering decisions log
└── wp-moe.md                       # Model architecture specification
```

## Getting Started

### Install Skills

All skills are prefixed `wp-finetune:` for easy discovery in Claude Code's command palette.

```bash
# Skills are already in .claude/skills/ — no install needed if you cloned this repo.
# To install manually from docs/:
mkdir -p .claude/skills/wp-finetune:run-data-pipeline .claude/skills/wp-finetune:run-training
cp docs/wp-finetune:run-data-pipeline.md .claude/skills/wp-finetune:run-data-pipeline/SKILL.md
cp docs/wp-finetune:run-training.md .claude/skills/wp-finetune:run-training/SKILL.md
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (used as fallback only)
```

### Run

In Claude Code, type `/wp-finetune:` to see all available skills, or say:

```
run the pipeline          # Data pipeline: clone, extract, judge, CoT, export
run training              # Training: model selection, ratio selection, DGX execution
run evaluation            # Eval triage: E_eff profiling, quality gates, wp-bench, triage
```

Or check status first:

```bash
python scripts/pipeline_orchestrator.py status   # Current pipeline state
python scripts/pipeline_orchestrator.py plan      # What actions are needed
```

## How It Works

The project has two autonomous skills that handle the full lifecycle from raw repos to a trained model.

### `/wp-finetune:run-data-pipeline` — Data Production

Runs the complete data pipeline end-to-end using Claude Code agents for all LLM work. Single invocation, no prompting required.

```
1. Orchestrator scans output dirs → computes percentage-based targets
2. Clone all repos from repos.yaml (script)
3. Extract PHP functions from cloned repos (script)
4. Judge ALL extracted functions via parallel agents (9-dimension rubric)
5. Gap analysis → synthetic generation → judge synthetics (agents)
6. Judge training data: score all passed (75-100) and failed (10-65) functions
7. 4-way CoT: gen pattern + judge rubric + judge contrastive + security
8. Re-check targets → if not met, loop back to step 2
9. Merge all sources → export at configured ratio → done
```

**Spawn-until-target pattern:** The orchestrator keeps spawning agent waves until all percentage-based targets are met. Targets scale with the dataset — no hardcoded numbers.

**All LLM work via Claude Code agents** (covered by subscription, $0 API cost). Non-LLM steps (cloning, extraction, gap analysis, mutations, export) run as Python scripts.

See [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md) for the full execution model and output format contracts.

### `/wp-finetune:run-training` — Model Training

Runs the training pipeline on DGX Spark via the `dgx_toolbox.py` execution engine. Supports training on multiple dataset ratio exports sequentially with isolated checkpoints.

```
Step 0a: Select base model (Qwen3-30B-A3B, 14B, 8B, or custom HF ID)
Step 0b: Select dataset exports (one or more of the 5 ratio exports)
Step 0c: Telemetry mode (observe agents / lightweight monitor / none)
Step 0d: Review full training plan → confirm before starting
   │
   │  For each selected ratio:
   │
Step 1: Generate per-run config overlay (data paths + output dir)
Step 2: Validate (toolbox, config, memory ≥ 70GB)
Step 3: Ensure Unsloth Studio container ready (start + mount + deps)
Step 4: Download base model (idempotent — shared across runs)
        [observe: 3 data-pipeline agents]
Step 5: Extend tokenizer with <wp_gen>/<wp_judge> (idempotent — shared)
Step 6: Dry run (validate config before committing to hours of training)
Step 7: Train (BF16 LoRA SFT, 6-12 hours, idempotent)
        [observe: 6 training agents + live thermal guard at ≥83°C]
        [monitor: 1 lightweight agent polling every 10min]
        [both: append to canonical {model}_{date}_{ratio}_thermal.jsonl]
Step 8: Merge adapter into base model (with verification roundtrip)
        [observe: 3 packaging agents + review-telemetry → _summary.md]
Step 8.5: Adaptive resource planning (between runs)
        Parse telemetry → classify thermal zone → adjust config for next run
        OOM detection overrides thermal: restore last non-OOM config + step down workers
        Peak RAM headroom (not average) with 5 GB safety margin on unified memory
        CRITICAL: backoff to last WARM config from thermal_history.json
        COOL/COLD: scale up batch_size if headroom allows (capped on unified memory)
Step 9: Report (after all runs: cross-run comparison summary)
```

**Run isolation:** Each ratio trains to `adapters/{model}-wp-{ratio}/` and merges to `models/{model}-wp-{ratio}-merged/`. Previous runs are never overwritten. Re-running the skill skips completed runs via idempotency checks.

**Telemetry modes:** Step 0c offers three modes (default: observe agents). **Observe** spawns the full 6-agent team with rich telemetry. **Monitor** runs a single lightweight agent that only records GPU utilization and temperature. Both write to the same canonical JSONL thermal log that feeds adaptive resource planning. **None** disables all telemetry (double-confirm warning).

**Confirmation gate:** Step 0d presents the full plan (model, LoRA config, hyperparameters, telemetry choice, estimated duration, disk requirements, output paths) and requires explicit confirmation before starting. No silent multi-hour training runs.

### `/wp-finetune:run-evaluation` — Eval Triage Pipeline

Runs the complete evaluation and triage pipeline. Profiles base model routing concentration (E_eff), evaluates all trained adapters through quality gates and wp-bench, then presents a structured triage decision for human approval.

```
Step 0:  Inventory adapters, datasets, DGX readiness
Step 1:  Base-model E_eff profiling (all 5 ratio distributions, ~10 min)
         → Decision Gate 1: E_eff trending down? → Train 60/40 in background
Step 2:  Sequential adapter eval (30/70 → 40/60 → 50/50)
         → Serve via vLLM, run eval_gen + eval_judge + eval_gate + wp-bench per adapter
Step 3:  Automated triage (GATE-02: fail any gate OR >5pp behind = eliminated)
Step 4:  ► HUMAN REVIEW — full comparison table with gates + wp-bench + E_eff
         → Human picks survivors for Phase 7 (MoE-Sieve)
Step 5:  Update STATE.md with triage decisions
```

**Idempotent:** Each step writes a `.complete` marker. Re-running resumes from last incomplete step. Use `--force` to re-run everything.

**Key insight:** Phase 4 is triage, not winner selection. A ratio with slightly lower eval score but sharper routing concentration (lower E_eff) may produce a better production model after MoE-Sieve + pruning. The triage preserves these candidates — Phase 7 makes the final call using BOTH eval quality AND fine-tuned adapter E_eff.

### `/wp-finetune:observe-*` and `/wp-finetune:review-telemetry` — Embedded Telemetry

Observe and review skills are **embedded within `/wp-finetune:run-training`** — they are spawned automatically at the right lifecycle points based on the telemetry mode selected in Step 0c. No need to invoke them separately during training.

They can still be invoked standalone for non-training operations (eval, inference, packaging).

| Skill | Agents | Spawned at | Mode |
|-------|--------|-----------|------|
| `observe-data-pipeline` | 3 | Step 4 (download) | observe only |
| `observe-training` | 6 | Step 7 (training) | observe only |
| `observe-packaging` | 3 | Step 8 (merge) | observe only |
| `review-telemetry` | — | Step 8d + 9b | observe only |
| lightweight monitor | 1 | Step 7 (training) | monitor only |
| `observe-evaluation` | 3 | Step 2 of run-evaluation | embedded |
| `observe-inference` | 5 | Standalone (Phase 5) | — |

```
telemetry/training/
  # Canonical thermal log (one per run — written by both modes)
  qwen3-30b_20260330_30_70_thermal.jsonl   ← {"ts","gpu_util","temp","vram_used_mb","sys_ram_used_mb","sys_ram_total_mb","source"}
  qwen3-30b_20260330_40_60_thermal.jsonl

  # Observe mode only — per-run agent reports
  {timestamp}/
    gpu-metrics.md, thermal-throttling.md, training-metrics.md,
    disk-io.md, checkpoint-integrity.md, container-monitor.md
    _stop, _thermal_pause, _summary.md

  # Shared — adaptive resource planning state
  thermal_history.json       ← Persistent record of all runs (config + thermal zone)
  adaptive_adjustments.md    ← Log of config changes between runs
  cross_run_summary.md       ← Final comparison table across all ratios
```

**Lifecycle per run:** spawn collectors (per mode) → all append to canonical JSONL → execute step → `_stop` → review-telemetry (observe only) → adaptive planning reads JSONL → adjust config → next run.

## DGX Toolbox Integration

This project pairs with [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) for training, evaluation, and serving. The toolbox location is **configurable** — it doesn't need to be at `~/dgx-toolbox`:

```bash
# Option 1: Edit config file
vim config/dgx_toolbox.yaml
# Change: dgx_toolbox_path: /path/to/your/dgx-toolbox

# Option 2: Environment variable (overrides config)
export DGX_TOOLBOX_PATH=/path/to/your/dgx-toolbox

# Verify
python scripts/dgx_toolbox.py
```

All scripts use the execution engine — config-driven, never hardcoded:

```python
from scripts.dgx_toolbox import get_toolbox

dgx = get_toolbox()
dgx.ensure_ready("unsloth_studio")                    # Launch + mount + install deps
dgx.execute("unsloth_studio", "python", "-m", "scripts.train_model")  # Idempotent exec
status = dgx.status_report()                           # Structured telemetry for agents
print(dgx.vllm_endpoint())                            # http://localhost:8020/v1
```

The engine reads all project-specific config from `config/dgx_toolbox.yaml` — container names, validation paths, required imports, status artifacts. Swap the YAML for a different project.

**Components used:**

| Component | Script | Purpose |
|-----------|--------|---------|
| Unsloth Studio | `containers/unsloth-studio.sh` | LoRA fine-tuning on DGX Spark |
| vLLM | `inference/start-vllm.sh` | Model serving for eval + production |
| LiteLLM | `inference/start-litellm.sh` | Unified API proxy (wp-bench uses this) |
| Open-WebUI | `inference/start-open-webui.sh` | Interactive demo |
| eval-toolbox | `eval/eval-toolbox.sh` | lm-eval, MLflow, scipy for eval suite |
| Ollama | `inference/setup-ollama-remote.sh` | GGUF local serving |

## Requirements

- Python 3.10+
- `pyyaml`, `python-dotenv`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards
- [Claude Code](https://claude.com/claude-code) (subscription) — used for all LLM pipeline steps
- [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) — training, eval, and serving (Phase 3+)

## License

Apache 2.0
