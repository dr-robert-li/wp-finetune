# Roadmap: wp-qwen3-moe

## Overview

Four phases take the project from fragile pipeline scripts to a deployed dual-mode WordPress code model. Phase 1 hardens the pipeline and converts existing CSV source data into repos.yaml — no manual URL hunting required. Phase 2 executes that pipeline to produce ~13,500 clean training examples. Phase 3 converts the base model to MoE, writes the evaluation suite, and trains on DGX Spark. Phase 4 runs the quality gate and packages the passing model for deployment.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Pipeline Ready** - Harden all pipeline scripts and convert existing CSV data into repos.yaml before any data is generated
- [ ] **Phase 2: Dataset Production** - Execute all three pipeline phases to produce the final training dataset
- [ ] **Phase 3: Model Prep and Training** - Download Qwen3-30B-A3B, extend tokenizer, write eval suite, and fine-tune on DGX Spark
- [ ] **Phase 4: Evaluation and Deployment** - Run quality gates and package the passing model for serving

## Phase Details

### Phase 1: Pipeline Ready
**Goal**: All pipeline scripts are safe to run at scale and repos.yaml is fully populated with quality-tiered sources, derived from the existing ranked CSVs at `/home/robert_li/Desktop/data/wp-finetune-data/`
**Depends on**: Nothing (first phase)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, REPO-01, REPO-02, REPO-03, REPO-04
**Success Criteria** (what must be TRUE):
  1. Running the pre-flight script with a missing PHPCS install, bad API key, or missing PHP CLI exits with a clear error message before any API calls are made
  2. Killing any long-running script mid-run and restarting it picks up from the last checkpoint rather than restarting from scratch
  3. A conversion script reads `wp_top1000_plugins_final.csv` and `wp_top100_themes_final.csv`, applies quality_tier automatically based on vulnerability data (plugins with unpatched critical CVEs get "assessed" tier with stricter path filters), and writes a valid repos.yaml containing WordPress Core, at least 10 plugins, and at least 5 themes, each with quality_tier, path_filters, and description fields
  4. A test run of phase2_mutate.py with PHPCS unavailable hard-exits instead of silently accepting mutations
  5. All Claude API calls in the pipeline use exponential backoff with jitter and route bulk judging through the Batch API
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Shared utilities and pre-flight (utils.py with extract_json, backoff, checkpoint, Batch API; preflight.py with tool validation)
- [x] 01-02-PLAN.md — CSV-to-repos.yaml conversion (reads ranked CSVs, filters by installs/rating/vulns, auto-assigns quality_tier, emits validated repos.yaml)

### Phase 2: Dataset Production
**Goal**: The full three-phase data pipeline executes against real repositories and produces a clean, split, multi-format training dataset
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11
**Success Criteria** (what must be TRUE):
  1. All repositories in repos.yaml are shallow-cloned and PHP functions are extracted with metadata
  2. Functions pass the PHPCS pre-filter before any Claude API judging occurs, and passed/failed examples are stored in separate files
  3. Gap analysis identifies which taxonomy categories are underrepresented and synthetic generation fills those gaps
  4. final_dataset/ contains at least 10,000 examples in OpenAI JSONL, Alpaca JSON, and raw JSONL formats with an 80/10/10 train/val/test split and task tokens present
  5. The wp_gen and wp_judge example counts follow approximately 40/60 gen/judge split (per user decision)
**Plans**: 7 plans

Plans:
- [x] 02-01-PLAN.md — Config updates (judge threshold >= 8, security auto-FAIL, N/A deflation, rejection templates) + Phase 1 script hardening (clone, extract, judge with utils.py)
- [x] 02-02-PLAN.md — Phase 2 script hardening (mutate PHPCS guard, generate with rejection examples + batch API, judge + judge_dataset with utils.py)
- [x] 02-03-PLAN.md — Phase 3 CoT hardening + export dataset update (40/60 ratio, metadata.json, dedup, PHP lint, sample_weight)
- [ ] 02-04-PLAN.md — [GAP CLOSURE] Judge remaining 23 repos via Claude Code agents (auto-pass wordpress-develop core, judge 22 assessed repos with 5 parallel agents)
- [ ] 02-05-PLAN.md — [GAP CLOSURE] Gap analysis + mutations (Python, no LLM) then synthetic generation via Claude Code agents (~500 rejection examples)
- [ ] 02-06-PLAN.md — [GAP CLOSURE] Judge synthetics + generate judge training data via Claude Code agents (rubric-scored 0-100 examples)
- [ ] 02-07-PLAN.md — [GAP CLOSURE] CoT reasoning via Claude Code agents + export dataset (Python) + human validation checkpoint

### Phase 3: Model Prep and Training
**Goal**: Qwen3-30B-A3B (native MoE) has task tokens added, an evaluation suite is ready before training completes, and a trained LoRA adapter exists on disk (kept separate from base model until eval passes)
**Depends on**: Phase 2 (for training data); model prep scripts can be written during Phase 2
**Requirements**: MODL-01, MODL-02, MODL-03, MODL-04, TRNG-01, TRNG-02, TRNG-03, TRNG-04, TRNG-05, TRNG-06, EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. A smoke test confirms the native MoE model loads, generates coherent text, and recognizes `<wp_gen>` and `<wp_judge>` tokens as single-token IDs
  2. Training completes on DGX Spark without OOM or loss divergence, with W&B tracking showing stable loss and router_aux_loss
  3. scripts/eval_gen.py, scripts/eval_judge.py, and scripts/eval_gate.py are runnable against any served checkpoint before the training run finishes
  4. adapters/qwen3-wp/ exists as a LoRA adapter checkpoint with tokenizer (adapter kept separate until evaluation passes in Phase 4)
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Model download and tokenizer extension (download Qwen3-30B-A3B, add task tokens, mean-init embeddings, smoke test)
- [ ] 03-02-PLAN.md — Evaluation suite (eval_gen.py PHPCS pass rate, eval_judge.py Spearman correlation, eval_gate.py quality gates, wp-bench config)
- [ ] 03-03-PLAN.md — Training configuration and execution (Unsloth LoRA config, DGX Spark run, W&B monitoring, adapter save)

### Phase 4: Evaluation and Deployment
**Goal**: All three quality gates pass and the model is live on vLLM and Ollama with a HuggingFace Hub release
**Depends on**: Phase 3
**Requirements**: DPLT-01, DPLT-02, DPLT-03, DPLT-04, DPLT-05, DPLT-06, DPLT-07
**Success Criteria** (what must be TRUE):
  1. PHPCS pass rate on 500 held-out generation tasks exceeds 95%, judge Spearman correlation on 500 held-out scored pairs exceeds 0.85, and security pass rate exceeds 98%
  2. The model responds to requests at `http://localhost:8020` via vLLM and `http://localhost:11434` via Ollama, and both backends correctly handle `<wp_gen>` and `<wp_judge>` task tokens
  3. Open-WebUI at port 12000 shows the model available for interactive use
  4. The HuggingFace Hub page contains the model card with eval metrics, GGUF and AWQ download links, and copy-pasteable usage examples
**Plans**: TBD

Plans:
- [ ] 04-01: Quality gate execution (run eval suite, confirm all three thresholds pass)
- [ ] 04-02: Packaging (merge LoRA, AWQ quantization for vLLM, GGUF for Ollama)
- [ ] 04-03: Deployment and HuggingFace upload (vLLM serve, Ollama serve, Open-WebUI, HF model card)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Ready | 2/2 | Complete | 2026-03-26 |
| 2. Dataset Production | 3/7 | In progress (gap closure) | - |
| 3. Model Prep and Training | 0/3 | Not started | - |
| 4. Evaluation and Deployment | 0/3 | Not started | - |
