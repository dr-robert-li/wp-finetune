# wp-qwen3-moe

An open-weight Mixture-of-Experts model that generates and judges WordPress code according to strict WordPress Coding Standards. A single model, two modes: `<wp_gen>` for code generation, `<wp_judge>` for structured critique with 9-dimension rubric scoring.

No open-source model existed for this. The tools in this space are wrappers around closed-source frontier models (OpenAI, Claude, etc.). This project builds one from scratch — open weights, self-hostable, no vendor lock-in.

## Architecture

| Property | Value |
|----------|-------|
| Base model | Qwen3-8B (dense-to-MoE conversion) |
| Total params | ~8B |
| Active params | ~4B per forward pass (top-2 of 8 experts) |
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
| 2. Dataset Production | Execute agent pipeline, produce ~20,000+ training examples | In progress |
| 3. Model Prep & Training | MoE conversion, tokenizer extension, LoRA SFT on DGX Spark | Planned |
| 4. Eval & Deployment | Quality gates, quantization, HuggingFace Hub release | Planned |

**Current:** Phase 2 — 42 repos judged, gap closure in progress (synthetic generation, judge training data, CoT reasoning).

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

**Sources:** Top 1000 WordPress plugins and top 100 themes by active installs, plus WordPress Core as reference implementation.

**Quality gates:** Every non-core example passes PHPCS pre-filtering AND 9-dimension rubric assessment (threshold >= 8, security auto-FAIL below 5). WordPress Core functions are auto-passed as the reference implementation.

### Dataset Composition Target

| Source | ~Count | Task Token |
|--------|--------|------------|
| Real code (passed judge) | 15,000+ | `<wp_gen>` |
| Synthetic gap-fill (passed judge) | 200+ | `<wp_gen>` |
| Judge training — high-score | ~1,500 | `<wp_judge>` |
| Judge training — low-score | ~1,000 | `<wp_judge>` |
| Judge training — synthetic | ~1,500 | `<wp_judge>` |
| CoT reasoning (real + contrastive + synthetic) | ~500 | `<wp_gen>` |
| **Total** | **~20,000+** | **40/60 gen/judge** |

## Success Criteria

| Metric | Target |
|--------|--------|
| Generator PHPCS pass rate | > 95% |
| Generator security pass rate | > 98% |
| Judge Spearman correlation | > 0.85 |
| Judge classification precision | > 0.90 |
| Active parameters per inference | ~4B |

## Project Structure

```
wp-finetune/
├── config/
│   ├── repos.yaml                  # 56 repos (auto-generated from ranked CSVs)
│   ├── judge_system.md             # 9-dimension judge rubric (threshold >= 8)
│   ├── taxonomy.yaml               # 87 concept tags + coverage minimums
│   └── synthetic_prompts.yaml      # Generation templates + rejection examples
├── scripts/
│   ├── utils.py                    # Shared utilities (JSON parsing, backoff, checkpoints)
│   ├── preflight.py                # Pre-execution validation
│   ├── agent_judge.py              # Static heuristic judge (9-dimension rubric scoring)
│   ├── agent_judge_helper.py       # Agent helper: list unjudged repos, split results
│   ├── autopass_core.py            # Auto-pass WordPress Core with taxonomy tagging
│   ├── csv_to_repos.py             # Convert ranked plugin/theme CSVs to repos.yaml
│   ├── phase1_{clone,extract,judge}.py
│   ├── phase2_{gap_analysis,mutate,generate,judge,judge_dataset}.py
│   ├── phase3_cot.py
│   └── export_dataset.py           # Multi-format export (40/60 gen/judge split)
├── docs/
│   └── AGENT_PIPELINE.md           # Agent execution model and output format contracts
├── phase1_extraction/              # Cloned repos + extracted/passed/failed functions
├── phase2_synthetic/               # Gap reports + synthetic/mutated/judge training data
├── phase3_cot/                     # CoT reasoning checkpoints
├── final_dataset/                  # Train/val/test in OpenAI, Alpaca, Raw JSONL formats
├── tests/                          # Unit tests
├── PROJECT.md                      # Full project specification
├── JOURNAL.md                      # Engineering decisions log
└── wp-moe.md                       # Model architecture specification
```

## Requirements

- Python 3.10+
- `pyyaml`, `python-dotenv`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards
- [Claude Code](https://claude.com/claude-code) (subscription) — used for all LLM pipeline steps (judging, generation, CoT)

**Training/serving (Phase 3+):** [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) — Unsloth Studio, vLLM, Ollama, eval-toolbox, safety harness. Runs on DGX Spark (Blackwell GB10, 128GB unified memory).

## License

Apache 2.0
