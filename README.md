# WordPress MoE Finetuning Data Pipeline

Data pipeline for the WordPress Best-Practice MoE Model (wp-qwen3-moe). Produces training data with `<wp_gen>` and `<wp_judge>` task tokens for a dual-mode Qwen3-8B-based Mixture-of-Experts model, trained and served on the [DGX Toolbox](~/dgx-toolbox) stack.

See [PROJECT.md](PROJECT.md) for the full project specification.
See [wp-moe.md](wp-moe.md) for the model architecture and training strategy.

## Pipeline Overview

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
              Claude judge ──►         │          Judge training       Merge all + judge data
                    │                  ▼          data generator            │
              passed/failed       bad→good pairs       │                    ▼
                                                       ▼              Export with task tokens
                                                  <wp_judge> data     ──► train/val/test splits
```

## Execution Model

This pipeline uses a **hybrid execution approach**:

- **Claude Code agents** handle all LLM-heavy work (judging, generation, scoring, CoT reasoning) — covered by subscription, $0 API cost
- **Python scripts** handle non-LLM work (cloning, extraction, gap analysis, mutations, export)
- **DGX Toolbox** provides training infrastructure (Unsloth Studio, vLLM, eval-toolbox)

Pipeline execution is orchestrated via [GSD](https://github.com/get-shit-done) with parallel agent spawning for throughput.

## Quick Start

```bash
# 0. Install dependencies
pip install anthropic pyyaml python-dotenv
# PHP CLI + PHPCS required for extraction and pre-filtering
composer global require squizlabs/php_codesniffer wp-coding-standards/wpcs

# 1. Configure
cp .env.example .env  # Add your ANTHROPIC_API_KEY
python scripts/preflight.py  # Validate setup

# 2. Generate repos.yaml from curated CSV data
python scripts/csv_to_repos.py

# 3. Pipeline execution via GSD (recommended)
# See .planning/ROADMAP.md for full phase structure
/gsd:execute-phase 2

# Or run scripts manually:
python scripts/phase1_clone.py
python scripts/phase1_extract.py
# LLM judging/generation handled by Claude Code agents
python scripts/phase2_gap_analysis.py
python scripts/phase2_mutate.py
# ... see PROJECT.md for full sequence
```

## Project Structure

```
wp-finetune/
├── .env                            # API keys (not committed)
├── .planning/                      # GSD project management
│   ├── PROJECT.md                  # Project context
│   ├── ROADMAP.md                  # 4-phase roadmap
│   ├── REQUIREMENTS.md             # 37 requirements with traceability
│   ├── STATE.md                    # Current progress
│   ├── config.json                 # Workflow preferences
│   ├── codebase/                   # Codebase analysis (7 docs)
│   ├── research/                   # Domain research (5 docs)
│   └── phases/                     # Per-phase plans and context
├── config/
│   ├── repos.yaml                  # 56 repos (auto-generated from CSV)
│   ├── judge_system.md             # 9-dimension judge rubric (threshold >= 8)
│   ├── taxonomy.yaml               # Concept tags + coverage minimums
│   └── synthetic_prompts.yaml      # Generation templates + rejection examples
├── scripts/
│   ├── utils.py                    # Shared utilities (JSON parsing, backoff, checkpoints, Batch API)
│   ├── preflight.py                # Pre-execution validation
│   ├── csv_to_repos.py             # CSV-to-repos.yaml converter
│   ├── phase1_clone.py             # Clone repos from repos.yaml
│   ├── phase1_extract.py           # Extract PHP functions via tokenizer
│   ├── phase1_judge.py             # PHPCS pre-filter + Claude judge
│   ├── phase2_gap_analysis.py      # Taxonomy coverage gaps
│   ├── phase2_mutate.py            # Automated contrastive mutations
│   ├── phase2_generate.py          # Synthetic generation (gap fill + rejection examples)
│   ├── phase2_judge.py             # Judge synthetic examples
│   ├── phase2_judge_dataset.py     # Rubric-scored judge training data
│   ├── phase3_cot.py               # CoT reasoning chains
│   └── export_dataset.py           # Multi-format export (40/60 gen/judge split)
├── tests/                          # 46 passing tests
└── final_dataset/                  # Output (not committed)
```

## Output Formats

| Format | File | Use |
|--------|------|-----|
| OpenAI | `openai_{split}.jsonl` | OpenAI finetuning API |
| Alpaca | `alpaca_{split}.json` | Qwen3-MoE / Unsloth / Axolotl / TRL |
| Raw | `raw_{split}.jsonl` | Analysis, includes metadata + sample_weight |

All formats include `<wp_gen>` / `<wp_judge>` task tokens for MoE routing.
40/60 gen/judge split emphasizes critic capability.

## Key Design Decisions

- **Base model:** Qwen3-8B with CMoE dense-to-MoE conversion (training-free, 5 min)
- **Judge threshold:** >= 8 on all 9 dimensions (raised from 7)
- **Security auto-FAIL:** Any security dimension < 5 = automatic FAIL
- **Rejection examples:** Model proactively adds security measures even when prompts omit them
- **Infrastructure:** DGX Toolbox (Unsloth Studio, vLLM, Ollama, eval-toolbox, safety harness)

## Requirements

- Python 3.10+
- `anthropic`, `pyyaml`, `python-dotenv`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards
- `.env` file with `ANTHROPIC_API_KEY`
