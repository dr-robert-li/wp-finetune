# Architecture Research

**Domain:** ML fine-tuning pipeline — data pipeline execution through model training, evaluation, and deployment
**Researched:** 2026-03-26
**Confidence:** HIGH (pipeline architecture from existing codebase), MEDIUM (MoE conversion and DGX Toolbox integration), LOW (DGX eval-toolbox specifics)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE LAYER                          │
│                    (Python scripts, existing)                        │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   Phase 1        │   Phase 2        │   Phase 3                     │
│   Extraction     │   Synthesis      │   CoT + Export                │
│                  │                  │                               │
│  repos.yaml ──>  │  gap_report.json │  phase1/ + phase2/            │
│  clone/extract   │  mutate/generate │  cot reasoning                │
│  PHPCS + judge   │  judge dataset   │  task token tagging           │
│                  │                  │  multi-format export          │
│  phase1_         │  phase2_         │  final_dataset/               │
│  extraction/     │  synthetic/      │  {openai,alpaca,raw}          │
│  passed/ failed/ │  output/         │  _{train,val,test}.*          │
└──────────────────┴──────────────────┴───────────────────────────────┘
                               │
                               ▼  final_dataset/ (13,500 examples)
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL PREPARATION LAYER                         │
│                    (offline, one-time conversion)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Qwen3-8B (dense HF checkpoint)                                     │
│       │                                                             │
│       ▼  dense-to-MoE upcycling (LLaMA-MoE / upcycling method)     │
│  Qwen3-8B-MoE (8 experts, top-2 routing)                           │
│       │                                                             │
│       ▼  tokenizer extension                                        │
│  Extended tokenizer (<wp_gen>, <wp_judge> special tokens)           │
│       │                                                             │
│       ▼  embedding resize (new token embeddings initialized)        │
│  MoE model + extended tokenizer (HF format, ready for training)     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                                │
│                 (Unsloth Studio on DGX Spark)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Unsloth Studio (Docker container on DGX Spark)                     │
│       │                                                             │
│       ├── Input: MoE model + extended tokenizer                     │
│       ├── Input: final_dataset/openai_train.jsonl                   │
│       ├── Config: LoRA r=64, target all linear layers               │
│       │   (q_proj, k_proj, v_proj, o_proj, gate_proj,               │
│       │    up_proj, down_proj — router layers frozen)               │
│       │                                                             │
│       ├── Training: Multi-task SFT over <wp_gen> + <wp_judge>       │
│       │   examples, 128GB unified memory constraint                 │
│       │                                                             │
│       └── Output: LoRA adapter checkpoints + merged model           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                               │
│               (DGX Toolbox eval-toolbox + custom)                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐   ┌───────────────────────────────────┐   │
│  │  Domain Evaluation   │   │   Standard Benchmarks             │   │
│  │                      │   │                                   │   │
│  │  PHPCS pass rate:    │   │  EleutherAI lm-eval-harness or    │   │
│  │  model-generated PHP │   │  bigcode-evaluation-harness       │   │
│  │  run through WPCS    │   │  (HumanEval-PHP, MBPP)            │   │
│  │  target >95%         │   │                                   │   │
│  │                      │   │  Baseline: Qwen3-8B dense         │   │
│  │  Judge correlation:  │   │  Target: no regression from       │   │
│  │  model rubric scores │   │  base model on general code       │   │
│  │  vs Claude reference │   │                                   │   │
│  │  target >0.85 Pearson│   │                                   │   │
│  └──────────────────────┘   └───────────────────────────────────┘   │
│                                                                     │
│  Eval dataset: final_dataset/openai_test.jsonl (10%, ~1,350 ex.)   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼  (if metrics pass)
┌─────────────────────────────────────────────────────────────────────┐
│                     PACKAGING + DEPLOYMENT LAYER                    │
│                      (DGX Toolbox serving stack)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Merged model (FP16/BF16 HF format)                                 │
│       │                                                             │
│       ├──> AWQ 4-bit quantization ──> vLLM on DGX Spark            │
│       │    (primary GPU inference path, ~741 tok/s)                 │
│       │                                                             │
│       ├──> GGUF quantization ──> Ollama on DGX Spark               │
│       │    (local chat/tool use via Open-WebUI)                     │
│       │                                                             │
│       └──> HuggingFace Hub upload (full BF16 + AWQ variant)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| Data Pipeline (Phase 1-3) | Produce 13,500 training examples with task tokens and train/val/test splits | 10 Python scripts, existing and tested |
| MoE Converter | Transform Qwen3-8B dense FFN layers into 8-expert MoE layers with top-2 routing | LLaMA-MoE / upcycling script (to be written) |
| Tokenizer Extender | Add `<wp_gen>` and `<wp_judge>` special tokens and resize model embeddings | HuggingFace `add_special_tokens` + `resize_token_embeddings` (to be written) |
| Unsloth Trainer | Execute LoRA SFT on DGX Spark within memory budget, produce merged checkpoint | Unsloth Studio Docker + Jupyter notebook |
| Domain Evaluator | Measure PHPCS pass rate and judge correlation against project thresholds | Custom Python eval scripts (to be written) |
| Standard Benchmarks | Verify no regression on general PHP code generation | lm-eval-harness or bigcode-eval-harness |
| AWQ Packager | Quantize merged model to 4-bit for vLLM serving | llm-compressor / AutoAWQ |
| GGUF Packager | Quantize merged model for Ollama local use | llama.cpp quantize or Unsloth export |
| vLLM Server | High-throughput GPU inference on DGX Spark | DGX Toolbox vLLM container |
| Ollama Server | Local interactive inference and Open-WebUI integration | DGX Toolbox Ollama container |

## Recommended Project Structure

```
wp-finetune/
├── scripts/                    # Existing data pipeline (Phase 1-3)
│   ├── phase1_*.py
│   ├── phase2_*.py
│   ├── phase3_cot.py
│   ├── export_dataset.py
│   └── php_extract_functions.php
├── config/                     # Existing configuration
│   ├── repos.yaml
│   ├── taxonomy.yaml
│   ├── synthetic_prompts.yaml
│   └── judge_system.md
├── model_prep/                 # NEW — MoE conversion + tokenizer extension
│   ├── convert_to_moe.py       # Dense-to-MoE upcycling script
│   ├── extend_tokenizer.py     # Add <wp_gen>/<wp_judge>, resize embeddings
│   └── verify_moe.py           # Smoke test: routing, token IDs, forward pass
├── training/                   # NEW — Unsloth training config + notebook
│   ├── train_config.yaml       # LoRA hyperparameters, dataset paths
│   └── train.ipynb             # Unsloth Studio notebook for DGX Spark
├── eval/                       # NEW — evaluation scripts
│   ├── eval_phpcs.py           # Generate PHP → run PHPCS → measure pass rate
│   ├── eval_judge_correlation.py  # Compare model rubric vs Claude reference
│   └── eval_benchmarks.sh      # lm-eval-harness wrapper for standard benchmarks
├── packaging/                  # NEW — quantization + upload scripts
│   ├── quantize_awq.py         # AWQ 4-bit quantization for vLLM
│   ├── quantize_gguf.sh        # GGUF export via llama.cpp for Ollama
│   └── upload_hf.py            # Push to HuggingFace Hub with model card
├── phase1_extraction/          # Phase 1 outputs (gitignored)
├── phase2_synthetic/           # Phase 2 outputs (gitignored)
├── phase3_cot/                 # Phase 3 outputs (gitignored)
└── final_dataset/              # Training-ready dataset (gitignored unless small)
    ├── openai_{train,val,test}.jsonl
    ├── alpaca_{train,val,test}.json
    └── raw_{train,val,test}.jsonl
```

### Structure Rationale

- **model_prep/:** Separates the one-time conversion steps from ongoing training; these scripts run once and produce an artifact that training depends on.
- **training/:** Keeps Unsloth-specific config together; the notebook is the interface with DGX Spark's Jupyter environment.
- **eval/:** Domain evaluation is custom and project-specific; kept separate from standard benchmark tooling.
- **packaging/:** Deployment artifacts (AWQ, GGUF, HF upload) are post-training; separating them prevents confusion about what's a prerequisite vs. a deliverable.

## Architectural Patterns

### Pattern 1: Quality-Gated Sequential Pipeline

**What:** Each pipeline stage produces output that is explicitly classified as passed or failed before feeding into the next stage. Nothing advances unless it clears the current stage's quality gate.
**When to use:** Any time training data quality must be enforced; the downstream training cost of bad data exceeds the cost of aggressive filtering.
**Trade-offs:** Reduces yield (fewer examples than naively collected) but prevents quality degradation from propagating; resumable because checkpoints are JSON files per stage.

```
extracted/ ──PHPCS pre-filter──> candidates/ ──Claude 9-dim judge──> passed/ + failed/
                                                                         │
                                                                     Only passed/ feeds next stage
```

### Pattern 2: Upcycling-then-Fine-tune (Two-Stage Model Preparation)

**What:** The dense model is structurally converted to MoE first (FFN layers replicated N times, routers inserted), then fine-tuning is applied to the already-converted architecture. The two stages are cleanly separated: conversion produces a new HF checkpoint, fine-tuning consumes that checkpoint.
**When to use:** When the task requires specialized routing to expert subnetworks (here: generation vs. judgment routing via task tokens). Avoids re-pretraining from scratch.
**Trade-offs:** Conversion adds ~1 hour of compute; initial routing weights are random and must be stabilized by the subsequent SFT pass. Router fine-tuning is disabled by default in Unsloth for MoE stability — this is correct behavior, not a limitation.

```python
# Step 1: Convert
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
moe_model = upcycle_to_moe(model, num_experts=8, top_k=2)
moe_model.save_pretrained("./qwen3-8b-moe-base")

# Step 2: Extend tokenizer BEFORE fine-tuning
tokenizer.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
moe_model.resize_token_embeddings(len(tokenizer))

# Step 3: Fine-tune on the converted model
```

### Pattern 3: Multi-Task Routing via Special Tokens

**What:** Task identity is encoded at the start of each user message via a special token (`<wp_gen>` or `<wp_judge>`). During inference, the same prefix steers which expert pathways activate for that forward pass.
**When to use:** When a single model must serve two distinct task types that benefit from specialized internal representations but share a common vocabulary and base knowledge.
**Trade-offs:** Requires the training data to be consistently token-tagged (handled by `export_dataset.py`); inference callers must prepend the correct task token or they get undefined routing behavior.

```
User: "<wp_gen> Write a secure WP_Query with meta_query..."
        │
        └──> top-2 routing activates generation-specialized experts
             ──> output: PHP code

User: "<wp_judge> Rate this function: [code snippet]"
        │
        └──> top-2 routing activates judgment-specialized experts
             ──> output: rubric scores + reasoning
```

### Pattern 4: Threshold-Gated Deployment

**What:** Packaging and deployment steps are blocked by explicit numeric thresholds from evaluation: PHPCS pass rate >95% AND judge correlation >0.85. If thresholds are not met, the model is not packaged.
**When to use:** When a model has a clear operational contract and shipping a substandard model is worse than shipping nothing.
**Trade-offs:** Requires custom domain evaluation scripts rather than relying solely on standard benchmarks; those scripts must be written alongside the eval phase.

## Data Flow

### End-to-End Pipeline Flow

```
GitHub repos (WordPress Core, plugins, themes)
    │
    ▼  phase1_clone.py
phase1_extraction/repos/  (shallow git clones)
    │
    ▼  phase1_extract.py + php_extract_functions.php
phase1_extraction/output/extracted/  (raw function JSON, one file per repo)
    │
    ▼  phase1_judge.py  [PHPCS gate → Claude gate]
    ├──> phase1_extraction/output/passed/   (training-ready real code)
    └──> phase1_extraction/output/failed/   (analysis only, no training use)

phase1 passed/ ──────────────────────────────────┐
    │                                             │
    ▼  phase2_gap_analysis.py                     │
phase2_synthetic/gap_report.json                  │ (style anchors)
    │                                             │
    ├──> phase2_generate.py ◄────────────────────┘
    │    (synthetic examples grounded in real code style)
    │    ──> phase2_synthetic/output/generated/
    │
    ├──> phase2_mutate.py ◄── phase1 passed/
    │    (bad→good contrastive pairs via deterministic mutations)
    │    ──> phase2_synthetic/output/mutated/
    │
    ▼  phase2_judge.py  [same PHPCS + Claude gates as Phase 1]
    ├──> phase2_synthetic/output/judged/passed/
    └──> phase2_synthetic/output/judged/failed/  (1 revision attempt, then discard)

    ▼  phase2_judge_dataset.py
phase2_synthetic/output/judge_training/  (<wp_judge> rubric-scored examples)

All Phase 1 passed + Phase 2 passed + Phase 2 judge training
    │
    ▼  phase3_cot.py  [instruction synthesis + CoT reasoning]
phase3_cot/output/  (checkpoints with reasoning added)
    │
    ▼  export_dataset.py  [task token tagging + format conversion + split]
final_dataset/
    ├── openai_{train,val,test}.jsonl   (primary training format)
    ├── alpaca_{train,val,test}.json    (alternative format)
    └── raw_{train,val,test}.jsonl      (with full metadata)
```

### Model Preparation Flow

```
HuggingFace Hub: Qwen/Qwen3-8B
    │
    ▼  model_prep/convert_to_moe.py
./qwen3-8b-moe-base/  (8 experts, top-2 routing, router weights random)
    │
    ▼  model_prep/extend_tokenizer.py
./qwen3-8b-moe-tokenized/  (+ <wp_gen>, <wp_judge>; embeddings resized)
    │
    ▼  model_prep/verify_moe.py  (sanity check before investing in training)
    └── smoke test: token IDs exist, forward pass completes, expert routing fires
```

### Training and Artifact Flow

```
./qwen3-8b-moe-tokenized/ + final_dataset/openai_train.jsonl
    │
    ▼  Unsloth Studio (DGX Spark, Docker)
    │   LoRA r=64, all linear layers, router frozen
    │   Multi-task SFT, 128GB unified memory
    │   Validation on final_dataset/openai_val.jsonl
    │
./checkpoints/  (LoRA adapters, saved periodically)
    │
    ▼  merge LoRA into base MoE model
./wp-qwen3-8b-moe-merged/  (full BF16 merged model)
```

### Evaluation Flow

```
./wp-qwen3-8b-moe-merged/
    │
    ├──> eval/eval_phpcs.py
    │    Prompt model → generate PHP → run PHPCS+WPCS
    │    Measure pass rate on eval_test.jsonl generation examples
    │    Target: >95% pass rate
    │
    ├──> eval/eval_judge_correlation.py
    │    Prompt model for rubric scores → compare to Claude reference
    │    Pearson correlation on eval_test.jsonl judgment examples
    │    Target: >0.85 correlation
    │
    └──> eval/eval_benchmarks.sh
         lm-eval-harness: HumanEval-PHP or bigcode-eval-harness PHP tasks
         Baseline: Qwen3-8B dense (pre-conversion score)
         Target: no significant regression
    │
    ▼  (metrics pass)
[DEPLOYMENT GATE CLEARED]
```

### Deployment Flow

```
./wp-qwen3-8b-moe-merged/  (gated by evaluation thresholds)
    │
    ├──> packaging/quantize_awq.py
    │    llm-compressor / AutoAWQ → 4-bit AWQ
    │    ./wp-qwen3-8b-moe-awq/
    │    ──> DGX Toolbox vLLM server (primary production path)
    │
    ├──> packaging/quantize_gguf.sh
    │    llama.cpp convert + quantize (Q4_K_M or Q5_K_M)
    │    ./wp-qwen3-8b-moe.gguf
    │    ──> DGX Toolbox Ollama server (Open-WebUI, local tooling)
    │
    └──> packaging/upload_hf.py
         Push BF16 + AWQ variants to HuggingFace Hub
         Include model card with PHPCS/correlation metrics
```

## Component Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Data Pipeline → Model Prep | File system: `final_dataset/openai_train.jsonl` | Pipeline writes, model prep does not read pipeline scripts |
| Model Prep → Training | File system: `./qwen3-8b-moe-tokenized/` directory | Unsloth Studio loads from local path |
| Training → Evaluation | File system: `./wp-qwen3-8b-moe-merged/` directory | Eval scripts load merged model |
| Evaluation → Packaging | Human gate: metrics must pass thresholds before running packaging | No automated coupling; deliberate manual checkpoint |
| Packaging → DGX Serving | DGX Toolbox container volume mounts | AWQ model dir → vLLM; GGUF file → Ollama |
| Evaluation ↔ PHPCS | CLI subprocess: `phpcs --standard=WordPress` | PHP + WPCS must be installed in eval environment |
| Data Pipeline ↔ Anthropic API | HTTPS: `ANTHROPIC_API_KEY` env var, rate-limited | 40-50 req/min across all phases |

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Anthropic API (Claude) | REST via `anthropic` Python SDK, env var auth | Rate limit 40-50 req/min; primary cost driver in pipeline |
| HuggingFace Hub | `huggingface_hub` Python SDK, HF token | Download Qwen3-8B base; upload final model |
| GitHub (WordPress repos) | `git clone --depth=1` via subprocess | Only needed during Phase 1 cloning |
| PHP CLI + PHPCS + WPCS | Local subprocess calls | Must be installed on the machine running Phase 1 and eval |

### DGX Toolbox Services

| Service | Role | Notes |
|---------|------|-------|
| Unsloth Studio | Fine-tuning (training layer) | Docker container, Jupyter interface |
| vLLM server | Production inference (AWQ model) | DGX Toolbox managed container |
| Ollama server | Local/chat inference (GGUF model) | DGX Toolbox managed container, feeds Open-WebUI |
| eval-toolbox | Standard benchmarks (may be available) | Specifics unconfirmed; fallback is lm-eval-harness directly |

## Build Order (Phase Dependencies)

The build order is strictly determined by data + artifact dependencies:

```
1. Data Pipeline Execution (Phase 1 → Phase 2 → Phase 3)
   No model prep can start until final_dataset/ exists.
   Phase 2 requires Phase 1 passed/ as input.
   Phase 3 requires both Phase 1 and Phase 2 outputs.

2. Model Preparation (MoE conversion → tokenizer extension → verification)
   Can start in parallel with data pipeline if desired (independent inputs).
   In practice, run sequentially to avoid context-switching.
   Must complete before training starts.

3. Training
   Requires: final_dataset/openai_train.jsonl + ./qwen3-8b-moe-tokenized/
   Both must exist before Unsloth training can begin.

4. Evaluation
   Requires: ./wp-qwen3-8b-moe-merged/ + final_dataset/openai_test.jsonl
   Runs after training completes.

5. Packaging + Deployment
   Requires: evaluation metrics pass thresholds
   AWQ and GGUF packaging can run in parallel.
   HF upload is last (after both quantized variants are verified).
```

## Anti-Patterns

### Anti-Pattern 1: Training on the Dense Model Then Converting

**What people do:** Fine-tune Qwen3-8B dense first, then attempt to convert the fine-tuned model to MoE.
**Why it's wrong:** The fine-tuned weights are distributed across the full FFN; splitting them into experts post-hoc produces experts that are all identical copies of the fine-tuned FFN, defeating the purpose of specialization. The routing gate trains on random initialization after fine-tuning has baked in dense representations.
**Do this instead:** Convert dense to MoE first, then fine-tune. The SFT pass trains the routing gate alongside the expert weights, enabling genuine task-token-driven routing to emerge.

### Anti-Pattern 2: Skipping the MoE Verification Step

**What people do:** Convert the model, immediately start training without a smoke test.
**Why it's wrong:** Dense-to-MoE conversion can silently produce a model that loads without error but fails forward passes due to shape mismatches after tokenizer embedding resize. Discovering this after a 6-hour training run wastes compute and debugging time.
**Do this instead:** Run `model_prep/verify_moe.py` — a short script that: (a) checks the two new token IDs exist in the tokenizer, (b) runs a forward pass with a `<wp_gen>`-prefixed prompt, (c) confirms all 8 experts fire at least once over a small batch.

### Anti-Pattern 3: Single Format for All Downstream Uses

**What people do:** Export only in one format (e.g., only OpenAI JSONL) and then manually reformat when a different tool needs a different format.
**Why it's wrong:** Unsloth, vLLM, Ollama, and evaluation harnesses all expect different formats; manual reformatting introduces transcription errors and is not reproducible.
**Do this instead:** The existing `export_dataset.py` already exports three formats simultaneously. Preserve all three in `final_dataset/` and use the format-specific file as the entry point for each downstream system.

### Anti-Pattern 4: Evaluating Only on the Validation Split During Training

**What people do:** Report validation loss from Unsloth training as the quality signal and skip running the domain evaluation scripts.
**Why it's wrong:** Validation loss does not measure PHPCS pass rate or judge correlation — the actual deployment criteria. A model can have low validation loss and still fail PHPCS >40% of the time due to prompt format issues or tokenizer edge cases.
**Do this instead:** Always run `eval_phpcs.py` and `eval_judge_correlation.py` on the held-out test split after training, using the two numeric thresholds (>95%, >0.85) as the deployment gate.

### Anti-Pattern 5: Merging LoRA Before Evaluation

**What people do:** Merge the LoRA adapter into the base model immediately after training, discard the adapter.
**Why it's wrong:** If evaluation fails, the unmerged adapter can be discarded and a new run started from the base MoE model without re-converting. Merging early destroys this rollback capability.
**Do this instead:** Keep the LoRA adapter and the base MoE model as separate artifacts until evaluation passes. Merge only after thresholds are met.

## Scaling Considerations

This is a single-run pipeline on fixed hardware, not a user-serving system. "Scaling" concerns are about data volume and memory budget, not traffic:

| Concern | Current Approach | If Scaling |
|---------|-----------------|------------|
| Claude API cost | PHPCS pre-filter cuts ~60% of calls | Increase pre-filter stringency; batch requests; use haiku for initial filter pass |
| Training memory | LoRA r=64, 128GB unified memory | Reduce LoRA rank; use gradient checkpointing; reduce sequence length |
| Dataset size | ~13,500 examples, fits in RAM | Streaming dataloaders if >100K examples |
| Evaluation latency | Sequential generation + PHPCS subprocess | Batch generation; parallel PHPCS processes |

## Sources

- [LLaMA-MoE: Building Mixture-of-Experts from LLaMA (EMNLP 2024)](https://github.com/pjlab-sys4nlp/llama-moe) — HIGH confidence, official project
- [Llama 3 Meets MoE: Efficient Upcycling (arXiv 2412.09952)](https://arxiv.org/abs/2412.09952) — HIGH confidence, peer-reviewed
- [ToMoE: Converting Dense LLMs to MoE through Dynamic Structural Pruning (arXiv 2501.15316)](https://arxiv.org/abs/2501.15316) — HIGH confidence, peer-reviewed
- [Fine-tuning LLMs with NVIDIA DGX Spark and Unsloth](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth) — HIGH confidence, official Unsloth docs
- [Unsloth Studio introduction](https://unsloth.ai/docs/new/studio) — HIGH confidence, official docs
- [Qwen3 Fine-tuning with Unsloth](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — HIGH confidence, official docs
- [vLLM AWQ quantization](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/awq/) — HIGH confidence, official vLLM docs
- [GGUF in vLLM](https://docs.vllm.ai/en/latest/features/quantization/gguf/) — HIGH confidence, official vLLM docs; note: GGUF in vLLM is experimental; Ollama is preferred path for GGUF
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) — HIGH confidence, official GitHub; supports PHP evaluation tasks
- [Adding Special Tokens to HuggingFace Models](https://medium.com/@coldstart_coder/adding-custom-tokens-to-huggingface-models-1981f114efc1) — MEDIUM confidence, community article

---
*Architecture research for: wp-qwen3-moe end-to-end training pipeline*
*Researched: 2026-03-26*
