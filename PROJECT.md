# PROJECT: WordPress Best-Practice MoE Model (wp-qwen3-moe)

## Vision

A single Qwen3-based Mixture-of-Experts model that both **generates** and **judges** WordPress code according to strict WordPress Coding Standards. Task tokens (`<wp_gen>`, `<wp_judge>`) route input to specialized expert pathways within the same network. Built and served on the [DGX Toolbox](~/dgx-toolbox) infrastructure stack.

## Architecture

- **Base:** Qwen3-8B (dense-to-MoE conversion using LLaMA-MoE methodology)
- **Size:** ~8B total params, ~4B active per forward pass (top-2 routing, 8 experts)
- **Modes:** `<wp_gen>` (code generation) and `<wp_judge>` (structured critique with rubric scoring)
- **Compatibility:** HuggingFace `AutoModelForCausalLM`, standard transformers tooling
- **Infrastructure:** DGX Toolbox — Unsloth Studio (fine-tuning), vLLM/Ollama (inference), eval-toolbox (benchmarks), safety harness (guardrails)

See [wp-moe.md](wp-moe.md) for full model specification.

---

## Project Phases

### Phase A: Data Pipeline (this repo)

The data pipeline lives in this directory and produces the training dataset.

#### A1. Repository Curation & Extraction

| Step | Script | Description |
|------|--------|-------------|
| A1.1 | *Manual* | Curate plugin/theme list in `config/repos.yaml` |
| A1.2 | `phase1_clone.py` | Shallow-clone all repositories |
| A1.3 | `phase1_extract.py` | Extract functions via PHP tokenizer (`php_extract_functions.php`) |
| A1.4 | `phase1_judge.py` | **PHPCS pre-filter** rejects high-error-density code cheaply; survivors go to **Claude judge** for 9-dimension assessment (WPCS, SQL safety, security, performance, WP API, code quality, dependencies, i18n, accessibility) |

**Quality tiers:**
- `core` — WordPress Core. Auto-passed as reference implementation, tagged only.
- `assessed` — Everything else. Function-by-function pass/fail. No partial credit.

**Outputs:** `phase1_extraction/output/passed/` and `phase1_extraction/output/failed/`

#### A2. Synthetic Generation & Judge Data

| Step | Script | Description |
|------|--------|-------------|
| A2.1 | `phase2_gap_analysis.py` | Compare tag coverage against `config/taxonomy.yaml` minimums |
| A2.2 | `phase2_mutate.py` | **Automated mutation** of passed real code: remove `prepare()`, strip nonces, strip escaping, remove capability checks, strip sanitization, strip i18n, inject `SELECT *`. Verified detectable by PHPCS. Produces bad->good contrastive pairs. |
| A2.3 | `phase2_generate.py` | Claude generates synthetic code grounded in real Phase 1 style anchors. Fills taxonomy gaps. Contrastive pair templates for Claude-generated bad->good pairs. |
| A2.4 | `phase2_judge.py` | Same judge criteria as Phase 1. Failed synthetics get one revision attempt, then discard. |
| A2.5 | `phase2_judge_dataset.py` | Generates `<wp_judge>` training data: Claude scores passed code (high), failed code (low), and mutated code (controlled defects) on a 0-100 rubric across 6 dimensions. Sanity-checked against expected quality tier. |

**Outputs:**
- `phase2_synthetic/output/judged/` — passed/failed synthetic code
- `phase2_synthetic/output/mutated/` — automated contrastive pairs
- `phase2_synthetic/output/judge_training/` — rubric-scored judge examples

#### A3. Chain-of-Thought & Export

| Step | Script | Description |
|------|--------|-------------|
| A3.1 | `phase3_cot.py` | **Instruction synthesis** for real code (reverse-engineer prompts). **CoT reasoning** for complex examples (SQL, performance, architecture). **Contrastive CoT** for mutation pairs (explain defect + fix). Merges judge training data. |
| A3.2 | `export_dataset.py` | Adds `<wp_gen>`/`<wp_judge>` task tokens. Exports OpenAI JSONL, Alpaca JSON, and raw JSONL with metadata. 80/10/10 train/val/test split. |

**Final outputs in `final_dataset/`:**
- `openai_{train,val,test}.jsonl`
- `alpaca_{train,val,test}.json`
- `raw_{train,val,test}.jsonl`
- `metadata.json`

---

### Phase B: Model Setup

*Not yet implemented. Planned work:*

| Step | Description |
|------|-------------|
| B1 | Convert Qwen3-8B dense model to MoE using LLaMA-MoE methodology (via DGX Toolbox Unsloth Studio) |
| B2 | Extend tokenizer with `<wp_gen>`, `<wp_judge>` special tokens |
| B3 | Resize model embeddings, initialize new token embeddings |
| B4 | Configure MoE routing: 8 experts, top-2, load balancing + z-loss |
| B5 | Set up training infrastructure via DGX Toolbox (Unsloth Studio for LoRA/QLoRA fine-tuning, W&B via eval-toolbox for tracking) |

### Phase C: Training

*Not yet implemented. Planned work:*

| Step | Description |
|------|-------------|
| C1 | **Multi-task SFT via Unsloth Studio:** 50% `<wp_gen>`, 50% `<wp_judge>`, shuffled. LoRA r=64 targeting q_proj, v_proj, gate, experts. Cosine LR schedule, bf16. |
| C2 | **Eval checkpoints** every 500 steps via eval-toolbox (lm-eval + W&B): PHPCS pass rate, security pass rate, judge correlation with ground truth |
| C3 | **DPO refinement** (optional): generate N completions via vLLM batch inference, human-rank via Argilla/Label Studio, DPO to align |

### Phase D: Evaluation & Validation

*Not yet implemented. Planned work:*

| Step | Description |
|------|-------------|
| D1 | **Generator eval:** 500 held-out tasks, PHPCS pass rate (>95%), security pass rate (>98%) |
| D2 | **Judge eval:** 500 held-out (code, scores) pairs, Spearman correlation (>0.85), classification precision (>0.90) |
| D3 | **Integration test:** generate -> judge -> iterate -> validate in wp-playground |
| D4 | **Adversarial test:** attempt to elicit insecure code, verify judge catches it |
| D5 | **Cross-model validation:** compare judge scores with GPT-4 scores on same code (via LiteLLM unified API) |

### Phase E: Packaging & Distribution

*Not yet implemented. Planned work:*

| Step | Description |
|------|-------------|
| E1 | Model card, data card, license (Apache 2.0) |
| E2 | HuggingFace Hub upload with benchmarks |
| E3 | Quantized versions (GGUF for Ollama, AWQ for vLLM) — served via DGX Toolbox inference stack |
| E4 | Example scripts: generate_plugin.py, judge_code.py, end_to_end.py |
| E5 | Containerized serving via DGX Toolbox (vLLM :8020, LiteLLM :4000, Open-WebUI :12000, Triton/TensorRT-LLM for production) |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/repos.yaml` | Repository list with quality tiers, path filters |
| `config/judge_system.md` | Claude judge system instruction (9 dimensions + rubric) |
| `config/taxonomy.yaml` | Concept taxonomy + minimum coverage targets per tag |
| `config/synthetic_prompts.yaml` | Prompt templates for synthetic generation, keyed by gap tag |

## Directory Structure

```
wp-finetune/
├── PROJECT.md                          # This file
├── README.md                           # Quick start guide
├── wp-moe.md                           # Full model specification
├── config/
│   ├── repos.yaml                      # Curated repo list (YOU EDIT THIS)
│   ├── judge_system.md                 # 9-dimension judge criteria
│   ├── taxonomy.yaml                   # Concept tags + coverage minimums
│   └── synthetic_prompts.yaml          # Generation templates by gap tag
├── scripts/
│   ├── phase1_clone.py                 # Clone repos
│   ├── phase1_extract.py              # Extract functions
│   ├── php_extract_functions.php       # PHP tokenizer helper
│   ├── phase1_judge.py                # PHPCS pre-filter + Claude judge
│   ├── phase2_gap_analysis.py         # Coverage gap report
│   ├── phase2_mutate.py               # Automated mutation contrastive pairs
│   ├── phase2_generate.py             # Synthetic generation (gap fill)
│   ├── phase2_judge.py                # Judge synthetic examples
│   ├── phase2_judge_dataset.py        # Generate <wp_judge> training data
│   ├── phase3_cot.py                  # CoT reasoning + merge all data
│   └── export_dataset.py              # Task tokens + multi-format export
├── phase1_extraction/
│   ├── repos/                          # Cloned repositories
│   └── output/
│       ├── extracted/                  # Raw extracted functions (JSON)
│       ├── passed/                     # Quality-assessed passed functions
│       └── failed/                     # Failed functions (kept for analysis)
├── phase2_synthetic/
│   ├── gap_report.json                 # Coverage analysis
│   └── output/
│       ├── generated/                  # Raw synthetic examples
│       ├── judged/                     # Judged synthetic (passed/failed)
│       ├── mutated/                    # Automated contrastive pairs
│       └── judge_training/             # <wp_judge> rubric-scored data
├── phase3_cot/
│   └── output/                         # CoT checkpoints
└── final_dataset/
    ├── metadata.json                   # Dataset statistics
    ├── openai_{train,val,test}.jsonl   # OpenAI finetuning format
    ├── alpaca_{train,val,test}.json    # Qwen3-MoE / Unsloth / Axolotl format
    └── raw_{train,val,test}.jsonl      # Full metadata format
```

## Target Dataset Composition

| Source | Est. Count | Task Token | Purpose |
|--------|-----------|------------|---------|
| WP Core (auto-passed) | ~2,000 | `<wp_gen>` | Reference patterns, conventions |
| Plugins/themes (assessed, passed) | ~3,000 | `<wp_gen>` | Real-world diversity |
| Synthetic gap-fill | ~2,000 | `<wp_gen>` | Long-tail patterns (multisite, batch ops) |
| Synthetic contrastive (Claude) | ~500 | `<wp_gen>` | Bad->good with CoT explanation |
| Automated mutation contrastive | ~2,000 | `<wp_gen>` | Bad->good from real code |
| Judge training (high-quality) | ~1,500 | `<wp_judge>` | Rubric scoring on good code |
| Judge training (low-quality) | ~1,000 | `<wp_judge>` | Rubric scoring on defective code |
| Judge training (mutations) | ~1,500 | `<wp_judge>` | Rubric scoring on controlled defects |
| **Total** | **~13,500** | **~50/50 split** | |

## Quality Gates

Every code example in the final dataset passed at least one of:
1. **WordPress Core origin** (auto-passed as reference implementation)
2. **PHPCS pre-filter** (< 5 errors per 100 lines) **AND** Claude 9-dimension judge (all scores >= 7, no critical failures)
3. **Claude synthetic generation** + same judge criteria (with one revision attempt on failure)

Judge training data is additionally sanity-checked: high-quality source code must score > 50 overall, low-quality must score < 95.

## Success Criteria

| Metric | Target | Measured At |
|--------|--------|-------------|
| Generator PHPCS pass rate | > 95% | Phase D1 |
| Generator security pass rate | > 98% | Phase D1 |
| Judge Spearman correlation | > 0.85 | Phase D2 |
| Judge classification precision | > 0.90 | Phase D2 |
| Model active parameters | ~4B (top-2 of 8 experts) | Phase B |
| Inference latency (DGX Spark) | < 2s via vLLM | Phase E |

## Current Status

- [x] Phase A: Data pipeline — scripts complete, config ready
- [ ] Phase A1.1: Repo curation — **waiting on curated repos.yaml**
- [ ] Phase A1.2-A3.2: Pipeline execution
- [ ] Phase B: Model setup
- [ ] Phase C: Training
- [ ] Phase D: Evaluation
- [ ] Phase E: Packaging

## Dependencies

**Runtime:**
- Python 3.10+
- `anthropic` SDK (ANTHROPIC_API_KEY required)
- `pyyaml`
- PHP CLI with `tokenizer` extension
- PHP_CodeSniffer + WordPress-Coding-Standards

**Training (Phase B-C) — via DGX Toolbox:**
- Unsloth Studio (:8000) — interactive fine-tuning UI with LoRA/QLoRA
- eval-toolbox container — lm-eval benchmarks, W&B tracking
- data-toolbox container — dataset curation, deduplication, quality filtering
- vLLM (:8020) — batch inference for DPO candidate generation
- LiteLLM (:4000) — unified API for cross-model evaluation
- Label Studio (:8081) / Argilla (:6900) — human annotation for DPO preference data
- Safety harness (:5000) — guardrails, red-teaming, PII redaction
- Hardware: DGX Spark (Blackwell GB10, 128GB unified memory)
