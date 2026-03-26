# wp-qwen3-moe

An open-weight Mixture-of-Experts model that generates and judges WordPress code according to strict WordPress Coding Standards. A single model, two modes: `<wp_gen>` for code generation, `<wp_judge>` for structured critique with 9-dimension rubric scoring.

No open-source model existed for this. The tools in this space are wrappers around closed-source frontier models. This project builds one from scratch — open weights, self-hostable, no vendor lock-in.

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
| 2. Dataset Production | Execute pipeline, produce ~13,500 training examples | In progress |
| 3. Model Prep & Training | MoE conversion, tokenizer extension, LoRA SFT on DGX Spark | Planned |
| 4. Eval & Deployment | Quality gates, quantization, HuggingFace Hub release | Planned |

**Current:** Phase 2 — gap closure (judging repos, synthetic generation, CoT reasoning).

See [PROJECT.md](PROJECT.md) for full phase details and success criteria.

## Data Pipeline

The training dataset combines real and synthetic WordPress code:

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

**Sources:** Top 1000 WordPress plugins and top 100 themes by active installs, plus WordPress Core as reference implementation.

**Quality gates:** Every example passes PHPCS pre-filtering AND 9-dimension Claude judge assessment (threshold >= 8, security auto-FAIL below 5).

### Dataset Composition Target

| Source | ~Count | Task Token |
|--------|--------|------------|
| WP Core (auto-passed) | 2,000 | `<wp_gen>` |
| Plugin/theme code (assessed) | 3,000 | `<wp_gen>` |
| Synthetic gap-fill | 2,000 | `<wp_gen>` |
| Contrastive pairs (mutations + synthetic) | 2,500 | `<wp_gen>` |
| Judge training (high/low/mutated) | 4,000 | `<wp_judge>` |
| **Total** | **~13,500** | **40/60 gen/judge** |

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
│   ├── phase1_{clone,extract,judge}.py
│   ├── phase2_{gap_analysis,mutate,generate,judge,judge_dataset}.py
│   ├── phase3_cot.py
│   └── export_dataset.py           # Multi-format export (40/60 gen/judge split)
├── phase1_extraction/              # Cloned repos + extracted/passed/failed functions
├── phase2_synthetic/               # Gap reports + synthetic/mutated/judge training data
├── phase3_cot/                     # CoT reasoning checkpoints
├── final_dataset/                  # Train/val/test in OpenAI, Alpaca, Raw JSONL formats
├── tests/                          # 46 tests
├── PROJECT.md                      # Full project specification
├── JOURNAL.md                      # Engineering decisions log
└── wp-moe.md                       # Model architecture specification
```

## Requirements

- Python 3.10+
- `anthropic`, `pyyaml`, `python-dotenv`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards
- `.env` file with `ANTHROPIC_API_KEY`

**Training/serving (Phase 3+):** [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) — Unsloth Studio, vLLM, Ollama, eval-toolbox, safety harness. Runs on DGX Spark (Blackwell GB10, 128GB unified memory).

## License

Apache 2.0
