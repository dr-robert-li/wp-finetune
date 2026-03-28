# Requirements: wp-qwen3-moe

**Defined:** 2026-03-26
**Core Value:** A single self-hostable model that generates WPCS-compliant WordPress code and catches critical defects via structured rubric scoring.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Pipeline Hardening

- [x] **PIPE-01**: Pipeline pre-flight script validates PHPCS install, API key, PHP CLI, and WordPress-Coding-Standards before execution
- [x] **PIPE-02**: All long-running scripts support checkpoint/resume to survive interruptions
- [x] **PIPE-03**: API calls use exponential backoff with jitter instead of fixed sleep intervals
- [x] **PIPE-04**: Scripts integrate Anthropic Batch API for high-volume offline processing (50% cost savings)
- [x] **PIPE-05**: Parse failure stubs are detected and rejected instead of silently entering training data

### Repository Curation

Source data already exists at `/home/robert_li/Desktop/data/wp-finetune-data/`: `wp_top1000_plugins_final.csv` (1,000 ranked plugins with github_url, active_installs, rating, vulnerability data, tags), `wp_top100_themes_final.csv` (100 ranked themes with same fields), and `wp_plugins_raw.json`. REPO-01 through REPO-04 are satisfied by writing a conversion script that reads these CSVs and emits repos.yaml — no manual URL hunting or sourcing required.

- [x] **REPO-01**: repos.yaml populated with WordPress Core repository
- [x] **REPO-02**: repos.yaml populated with 10+ high-quality plugins selected from ranked CSV (filtered by active_installs, rating, and vulnerability data)
- [x] **REPO-03**: repos.yaml populated with 5+ high-quality themes selected from ranked CSV
- [x] **REPO-04**: Each repo entry has quality_tier (auto-assigned from vulnerability data: unpatched critical CVEs → "assessed" tier with stricter filters), path_filters, and description

### Data Pipeline Execution

- [x] **DATA-01**: Phase 1 clone completes — all repos in repos.yaml shallow-cloned
- [x] **DATA-02**: Phase 1 extract completes — PHP functions extracted with metadata
- [x] **DATA-03**: Phase 1 judge completes — functions assessed (PHPCS pre-filter + Claude judge), passed/failed separated
- [x] **DATA-04**: Phase 2 gap analysis completes — coverage gaps identified against taxonomy
- [x] **DATA-05**: Phase 2 mutation completes — contrastive bad→good pairs generated from passed code
- [x] **DATA-06**: Phase 2 generate completes — synthetic examples fill taxonomy gaps
- [ ] **DATA-07**: Phase 2 judge completes — synthetic examples assessed, failed get one revision
- [ ] **DATA-08**: Phase 2 judge_dataset completes — rubric-scored judge training data generated
- [x] **DATA-09**: Phase 3 CoT completes — instruction synthesis + reasoning chains generated
- [x] **DATA-10**: Phase 3 export completes — OpenAI, Alpaca, Raw JSONL with task tokens, 80/10/10 split
- [x] **DATA-11**: Final dataset contains ≥10,000 examples with ~50/50 wp_gen/wp_judge split

### Model Preparation

- [x] **MODL-01**: Qwen3-30B-A3B downloaded (native MoE, 128 experts, top-8 routing, no conversion needed)
- [x] **MODL-02**: Tokenizer extended with `<wp_gen>` and `<wp_judge>` special tokens
- [x] **MODL-03**: Model embeddings resized and new token embeddings initialized (mean of existing)
- [x] **MODL-04**: Smoke test passes — model loads, generates coherent text, task tokens are recognized

### Training

- [x] **TRNG-01**: Unsloth LoRA SFT configured on DGX Spark (r=64, bf16, cosine LR)
- [x] **TRNG-02**: LoRA config includes `modules_to_save=["embed_tokens", "lm_head"]` for special tokens
- [x] **TRNG-03**: Training data loaded as 50/50 wp_gen/wp_judge multi-task mix
- [x] **TRNG-04**: MoE load balancing loss monitored throughout training (no routing collapse)
- [x] **TRNG-05**: W&B experiment tracking active via eval-toolbox
- [x] **TRNG-06**: Training completes without OOM or divergence on DGX Spark

### Evaluation

- [ ] **EVAL-01**: Custom eval script measures PHPCS pass rate on 500 held-out generation tasks (target >95%)
- [ ] **EVAL-02**: Custom eval script measures judge Spearman correlation on 500 held-out scored pairs (target >0.85)
- [ ] **EVAL-03**: Security pass rate measured on held-out tasks (target >98%)
- [ ] **EVAL-04**: Eval scripts run via DGX Toolbox eval-toolbox container
- [ ] **EVAL-05**: All three quality gates pass before proceeding to deployment

### Deployment

- [ ] **DPLT-01**: LoRA adapter merged into base model weights
- [ ] **DPLT-02**: AWQ quantization produced for vLLM production serving
- [ ] **DPLT-03**: GGUF quantization produced for Ollama local serving
- [ ] **DPLT-04**: Model served via DGX Toolbox vLLM (:8020) and accessible through LiteLLM (:4000)
- [ ] **DPLT-05**: Model served via DGX Toolbox Ollama (:11434)
- [ ] **DPLT-06**: HuggingFace Hub upload with model card, benchmarks, and usage examples
- [ ] **DPLT-07**: Interactive demo accessible via Open-WebUI (:12000)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Capabilities

- **V2-01**: DPO/RLHF refinement using preference data from Argilla/Label Studio
- **V2-02**: JavaScript/Gutenberg block generation via `<wp_block>` task token
- **V2-03**: Multi-lingual comment support (non-English PHPDoc/i18n)
- **V2-04**: Safety harness integration for production guardrails and red-teaming
- **V2-05**: Triton/TensorRT-LLM optimized inference engine

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time PHPCS correction loop at inference | Adds inference latency, model should internalize standards |
| Binary pass/fail judgment (no rubric) | Structured 9-dimension scoring is the differentiator |
| Mobile app or custom web UI | DGX Toolbox Open-WebUI covers interactive demo needs |
| Multi-model ensemble | Single model is the architectural constraint |
| JavaScript training data | PHP only for v1, different domain requires separate pipeline |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 | Complete |
| PIPE-02 | Phase 1 | Complete |
| PIPE-03 | Phase 1 | Complete |
| PIPE-04 | Phase 1 | Complete |
| PIPE-05 | Phase 1 | Complete |
| REPO-01 | Phase 1 | Complete |
| REPO-02 | Phase 1 | Complete |
| REPO-03 | Phase 1 | Complete |
| REPO-04 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Complete |
| DATA-06 | Phase 2 | Complete |
| DATA-07 | Phase 2 | Pending |
| DATA-08 | Phase 2 | Pending |
| DATA-09 | Phase 2 | Complete |
| DATA-10 | Phase 2 | Complete |
| DATA-11 | Phase 2 | Complete |
| MODL-01 | Phase 3 | Complete |
| MODL-02 | Phase 3 | Complete |
| MODL-03 | Phase 3 | Complete |
| MODL-04 | Phase 3 | Complete |
| TRNG-01 | Phase 3 | Complete |
| TRNG-02 | Phase 3 | Complete |
| TRNG-03 | Phase 3 | Complete |
| TRNG-04 | Phase 3 | Complete |
| TRNG-05 | Phase 3 | Complete |
| TRNG-06 | Phase 3 | Complete |
| EVAL-01 | Phase 3 | Complete (03-02) |
| EVAL-02 | Phase 3 | Complete (03-02) |
| EVAL-03 | Phase 3 | Complete (03-02) |
| EVAL-04 | Phase 3 | Complete (03-02) |
| EVAL-05 | Phase 3 | Complete (03-02) |
| DPLT-01 | Phase 5 | Pending |
| DPLT-02 | Phase 5 | Pending |
| DPLT-03 | Phase 5 | Pending |
| DPLT-04 | Phase 5 | Pending |
| DPLT-05 | Phase 5 | Pending |
| DPLT-06 | Phase 5 | Pending |
| DPLT-07 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-03-26 — REPO-01 through REPO-04 updated to reflect existing CSV source data*
