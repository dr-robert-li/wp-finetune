# wp-qwen3-moe

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

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Pipeline Ready | Harden scripts, generate repos.yaml from curated data | Complete |
| 2. Dataset Production | Execute agent pipeline, produce training examples | Complete |
| 3. Model Prep & Training | Tokenizer extension, BF16 LoRA SFT on DGX Spark | At checkpoint |
| 4. Evaluation | wp-bench + 193-check rubric eval, quality gates | Not started |
| 5. Packaging & Deployment | AWQ/GGUF quantization, HuggingFace Hub release | Not started |

**Current:** Phase 3 — scripts ready, tokenizer prepared, 5 ratio exports produced (30/70 through 70/30). Training on DGX Spark next.

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

94,630 unique examples after dedup, exported at 5 gen/judge ratios:

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
│   ├── repos.yaml                  # 213 repos (top + poor-quality plugins/themes)
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
│   ├── train_model.py              # BF16 LoRA SFT with memory pre-check
│   ├── merge_adapter.py            # Merge adapter with verification roundtrip
│   ├── phase1_{clone,extract,judge}.py
│   ├── phase2_{gap_analysis,mutate,generate,judge,judge_dataset}.py
│   ├── phase3_cot.py, merge_dataset.py, export_dataset.py
│   └── (+ agent_judge, autopass_core, csv_to_repos, preflight)
├── eval/
│   ├── rubric_definitions.py       # 193 check IDs across 9 weighted dimensions
│   ├── rubric_scorer.py            # 4-tool ground truth scoring engine
│   ├── eval_gen.py                 # Generator eval (PHPCS + security pass rates)
│   ├── eval_judge.py               # Judge eval (per-dimension Spearman correlation)
│   └── eval_gate.py                # Quality gate (pass/fail against thresholds)
├── docs/
│   ├── AGENT_PIPELINE.md           # Agent execution model and output format contracts
│   ├── wp-finetune:run-data-pipeline.md   # Skill: autonomous data pipeline
│   ├── wp-finetune:run-training.md        # Skill: DGX Spark training pipeline
│   ├── wp-finetune:observe-{stage}.md     # Telemetry skills (5 stages, 3-6 agents each)
│   └── wp-finetune:review-telemetry.md    # Telemetry aggregation and summary
├── docs/eval/
│   ├── wp_code_quality_rubric.md   # 193-check canonical rubric (9 dimensions, weighted)
│   ├── research_wpcs_standards.md  # WPCS + VIP sniff reference
│   └── research_wp_security_sql_perf.md  # Security, SQL, performance patterns
├── data/
│   ├── phase1_extraction/          # Cloned repos + extracted/passed/failed functions
│   ├── phase2_synthetic/           # Gap reports + synthetic/mutated/judge training data
│   ├── phase3_cot/                 # CoT reasoning checkpoints
│   ├── final_dataset/              # Train/val/test in OpenAI, Alpaca, Raw JSONL formats
│   └── checkpoints/                # Pipeline execution checkpoints
├── tests/                          # 75 tests (13 test files)
├── PROJECT.md                      # Full project specification
├── JOURNAL.md                      # Engineering decisions log
└── wp-moe.md                       # Model architecture specification
```

## Autonomous Pipeline

The entire data pipeline runs autonomously with a single command in Claude Code — no continuous prompting required.

### Install the Skill

Copy the skills into your Claude Code skills directory:

```bash
# From the project root — install all skills
mkdir -p .claude/skills/wp-finetune:run-data-pipeline .claude/skills/wp-finetune:run-training
cp docs/wp-finetune:run-data-pipeline.md .claude/skills/wp-finetune:run-data-pipeline/SKILL.md
cp docs/wp-finetune:run-training.md .claude/skills/wp-finetune:run-training/SKILL.md
```

Or symlink so updates propagate:

```bash
mkdir -p .claude/skills/wp-finetune:run-data-pipeline .claude/skills/wp-finetune:run-training
ln -sf "$(pwd)/docs/wp-finetune:run-data-pipeline.md" .claude/skills/wp-finetune:run-data-pipeline/SKILL.md
ln -sf "$(pwd)/docs/wp-finetune:run-training.md" .claude/skills/wp-finetune:run-training/SKILL.md
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (used as fallback only)
```

### Run

In Claude Code, say:

```
run the pipeline          # Phase 2: generate training data
run training              # Phase 3: download, tokenizer, train, merge via dgx-toolbox
```

The training skill uses `dgx_toolbox.py` as the execution engine — it validates state, manages containers, installs deps, and executes commands dynamically. No hardcoded Docker commands.

Or check status first:

```bash
python scripts/pipeline_orchestrator.py status   # Current state
python scripts/pipeline_orchestrator.py plan      # What needs doing
```

### How It Works

The skill follows a **spawn-until-target** loop:

```
1. Orchestrator checks state → identifies gaps
2. Claude Code spawns parallel agents for each gap
3. Agents write results (judging, generation, scoring, CoT)
4. Orchestrator re-checks state
5. If targets not met → loop back to step 2
6. When all targets met → merge → export → done
```

All LLM work uses Claude Code agents (covered by subscription, $0 API cost).
Non-LLM steps (cloning, extraction, gap analysis, mutations, export) run as Python scripts.

See [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md) for the full execution model, output format contracts, and scaling guide.

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
| eval-toolbox | `eval/eval-toolbox.sh` | lm-eval, W&B, scipy for eval suite |
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
