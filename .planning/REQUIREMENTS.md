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
- [x] **DATA-07**: Phase 2 judge completes — synthetic examples assessed, failed get one revision
- [x] **DATA-08**: Phase 2 judge_dataset completes — rubric-scored judge training data generated
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
- [x] **EVAL-05**: All three quality gates pass before proceeding to deployment

### Deployment (deferred to v2.0 Packaging)

DPLT requirements moved to v2.0 PKG-01 through PKG-05 — package after MoE-Sieve + REAP pruning, not the intermediate full-LoRA model. Serving requirements (vLLM, Ollama, Open-WebUI) covered by PKG-05 E2E validation.

- [ ] ~~**DPLT-01**: LoRA adapter merged into base model weights~~ → subsumed by v2.0 PRUNE-05
- [ ] ~~**DPLT-02**: AWQ quantization produced for vLLM~~ → subsumed by v2.0 PKG-03 (if warranted by Gate 2)
- [ ] ~~**DPLT-03**: GGUF quantization produced for Ollama~~ → subsumed by v2.0 PKG-03 (if warranted by Gate 2)
- [ ] ~~**DPLT-04**: vLLM serving~~ → subsumed by v2.0 PKG-05
- [ ] ~~**DPLT-05**: Ollama serving~~ → subsumed by v2.0 PKG-05
- [ ] ~~**DPLT-06**: HuggingFace upload~~ → subsumed by v2.0 PKG-04
- [ ] ~~**DPLT-07**: Open-WebUI demo~~ → subsumed by v2.0 PKG-05

## v1.1 Requirements — Adaptive Training Infrastructure

Requirements for power-primary adaptive planner. Depends on dgx-toolbox Phase 13 (telemetry/ package).

### Adaptive Planner

- [x] **ADPT-01**: Adaptive planner routes by GPU power zone (THROTTLED/CAPPED/TARGET/MODERATE/UNDERUTILIZED) with temperature as safety brake only at >=82C
- [x] **ADPT-02**: Thermal exploitation ladder fires in v4.0 order: batch (Rung 1) > prefetch (Rung 2) > workers (Rung 3) > save_steps (Rung 4) > eval_steps (Rung 5)
- [x] **ADPT-03**: All thresholds read from config/adaptive_planning.yaml (no hardcoded values in skill logic)

### Batch Coupling

- [x] **BTCH-01**: Every batch_size change auto-adjusts grad_accum to hold effective_batch constant
- [x] **BTCH-02**: Unsloth banner parsing detects silent batch/grad_accum overrides and writes actuals to telemetry/training/_unsloth_actuals.json
- [x] **BTCH-03**: Planner uses Unsloth actual values (not config values) as basis when override detected

### Telemetry

- [x] **TELE-01**: MemoryWatchdogCallback samples GPU power via GPUSampler every 50 steps and writes to canonical JSONL
- [x] **TELE-02**: Canonical JSONL schema includes watts and mem_available_gb fields (per GPUSampler API)
- [x] **TELE-03**: Failure classifier categorizes run outcome as NORMAL/OOM/HANG/THERMAL from telemetry readings
- [x] **TELE-04**: observe-training thresholds updated from 80C/83C to 82C/85C throughout

### Warmup and Anchors

- [x] **PROB-01**: Warmup probe runs 3-5 real training steps (via dgx-toolbox probe.py) when batch increased without anchor
- [x] **PROB-02**: Anchor store persists config+outcome history with config hashing, cooldown tracking, and hard caps
- [x] **PROB-03**: run-training Step 8.5 replaced with adaptive-planner skill invocation

## v2.0 Requirements — MoE-Sieve Selective Training

Requirements for selective expert training and evaluation. Depends on Phase 4 eval completing (need winning gen/judge ratio). Pruning and packaging deferred to v3.0 (must happen after GRPO to prune on final routing distribution).

### Router Profiling

- [ ] **PROF-01**: Router profiling runs gradient-free forward pass hooking `Qwen3MoeSparseMoeBlock` gating output, count-based ranking per layer
- [ ] **PROF-02**: Profiling tags each expert's routing count by task token affinity (`<wp_gen>` vs `<wp_judge>`) separately, not just aggregate frequency
- [ ] **PROF-03**: Profiling uses 10% subsample with Jaccard stability verification against full set (target ≥0.94)
- [ ] **PROF-04**: Outputs routing concentration report: per-layer CV, cumulative coverage curve at each k, layer-depth skew analysis, and effective expert count E_eff = exp(entropy) per layer (mean, max, variance across layers) — E_eff directly predicts pruning headroom
- [ ] **PROF-05**: Profile ALL surviving ratios from Phase 4 triage (not just the winner) — profiling is ~minutes per ratio and routing concentration is a critical decision signal for ratio selection

### Ratio Selection Gate (Phase 7→8)

- [ ] **GATE-01**: Decision matrix combining Phase 4 eval score (normalized 0-1) and Phase 7 routing concentration (mean E_eff, max E_eff, E_eff variance) per surviving ratio — select ratio with lowest E_eff at equivalent quality (within 2pp), preferring compressibility over marginal quality gains
- [x] **GATE-02**: Phase 4 triage uses high bar for elimination (only cut ratios that fail hard gates or are >5pp behind) and low bar for continuation — 1-2pp differences may invert after pruning if routing concentration differs

### Selective Training (MoE-Sieve)

- [ ] **SIEVE-01**: LoRA r=32, α=64, dropout=0.05 applied to hot routed experts + all attention (Q/K/V/O) + router gates + 4 shared experts (always trained); cold routed experts frozen
- [ ] **SIEVE-02**: Gen-hot experts trained on golden signal data only (passed examples, synthetic good); judge-hot experts trained on full spectrum (passed + failed + contrastive)
- [ ] **SIEVE-03**: Retrain uses best gen/judge ratio determined by Phase 4 eval results
- [ ] **SIEVE-04**: K-sweep at minimum 3 budgets (~13, 32, 64 experts per layer from 128 routed) to find accuracy plateau for Qwen3-30B-A3B on WordPress data
- [ ] **SIEVE-05**: Optimal k is smallest budget matching full-LoRA within ±1pp on wp-bench (TOST equivalence test, ε=2pp, 3+ seeds)

### Comparative Evaluation

- [ ] **EVAL2-01**: A/B eval of each k-sweep MoE-Sieve adapter against v1.0 full-LoRA on wp-bench and static eval suite
- [ ] **EVAL2-02**: Report includes per-dimension comparison (all 9 dimensions), overall scores, inference speed delta, and seed variance comparison

## v3.0 Requirements — GRPO & Production Deployment

Requirements for GRPO reinforcement learning on the MoE-Sieve model, followed by LoRA merge, REAP pruning on final routing distribution, and production packaging. GRPO must precede pruning because RL changes which experts matter.

### Reward Infrastructure

- [ ] **GRPO-01**: Composite reward pipeline with 70% verifiable / 30% judge weighting — PHPCS pass rate (high-variance anchor), security scanner (hard gate: score=0 on failure), WordPress standards checks (VeRPO partial credit weighted by check difficulty), frozen wp_judge score (MO-GRPO normalized)
- [ ] **GRPO-02**: Security scanner hard gate — if generation fails security scan, total reward = 0 regardless of all other scores (non-negotiable safety floor)
- [ ] **GRPO-03**: MO-GRPO normalization on all reward signals — each signal normalized by within-group variance to prevent single-signal dominance
- [ ] **GRPO-04**: VeRPO-style partial credit for WordPress standards checks — each check weighted by difficulty (estimated from pass rate across group samples; rarely-passed checks contribute more signal)

### GRPO Training

- [ ] **GRPO-05**: Gen-only GRPO — `<wp_gen>` generation quality improved via RL; `<wp_judge>` capability completely frozen from SFT
- [ ] **GRPO-06**: Hot experts only — GRPO gradients flow to hot routed experts + attention + router gates + shared experts; cold routed experts frozen (structural stability anchor)
- [ ] **GRPO-07**: RSPO router-shift stabilization — compute router-shift ratio between rollout and training phases, apply stop-gradient and floor, multiply into clipped importance ratio before aggregation
- [ ] **GRPO-08**: Router-shift ratio monitored throughout training — log per-step shift metrics; halt training if shift exceeds stability threshold (routing collapse early warning)

### LoRA Merge & Expert Pruning (AIMER vs REAP)

Sub-experiment: Does WordPress domain specialization create enough routing concentration for calibration-based pruning (REAP) to outperform weight-based pruning (AIMER)? Or is PHP/WordPress too close to general code for domain-aware pruning to differentiate?

- [ ] **MERGE-01**: Merge MoE-Sieve + GRPO LoRA adapters into base model weights before pruning — REAP needs activation magnitudes from the unified model, AIMER needs final weight norms
- [ ] **PRUNE-01**: Run AIMER pruning on merged model (weight-based, no calibration, ~1 second) at 25%, 50%, and 75% compression ratios — serves as task-agnostic baseline
- [ ] **PRUNE-02**: Run REAP pruning on same merged model with WordPress calibration data (gen + judge examples), `reap` saliency scoring, at same 25%, 50%, 75% compression ratios — serves as domain-aware comparison
- [ ] **PRUNE-03**: Evaluate both methods via gating mask before weight removal — compare retention across all 9 eval dimensions at each compression ratio (6 variants total: 2 methods × 3 ratios)
- [ ] **PRUNE-04**: Analyze domain specificity signal: compare which experts each method retains/prunes — high overlap suggests WordPress isn't specialized enough for calibration-based advantage; low overlap suggests REAP is capturing domain-specific routing patterns AIMER misses
- [ ] **PRUNE-05**: Select winning method + compression ratio with best dimension-level retention (especially D2_security), prefer higher compression at equivalent quality; if regression on any dimension, reduce compression incrementally until clean
- [ ] **PRUNE-06**: Final model has expert weights physically removed and router softmax re-normalized for removed expert slots; saved as HuggingFace-compatible checkpoint; pruning methodology documented in model card

### Comparative Evaluation

- [ ] **EVAL3-01**: A/B eval of GRPO+pruned model against v2.0 SFT-only (MoE-Sieve without GRPO) on wp-bench and static eval suite
- [ ] **EVAL3-02**: Report includes per-dimension comparison, inference speed delta (expect significant from pruning), model size reduction, and seed variance

### Packaging (cascading compression gates)

- [ ] **PKG-01**: Gate 1 — Eval pruned bf16 model: record size, inference speed, all 9 dimensions as quality baseline for subsequent compression
- [ ] **PKG-02**: Gate 2 — Assess whether quantization is needed based on pruned model size, deployment constraints, and Gate 1 performance margins
- [ ] **PKG-03**: If quantization warranted, test incrementally Q8→Q6→Q5→Q4, eval at each level, stop at lowest quantization holding within ±2pp of Gate 1 baseline
- [ ] **PKG-04**: Model card + adapter uploaded to HuggingFace with full compression lineage (base → MoE-Sieve → GRPO → merge → AIMER/REAP winner → quantization level, eval at each gate) including AIMER vs REAP comparison results
- [ ] **PKG-05**: E2E inference validation on final shipped format (both `<wp_gen>` and `<wp_judge>` prompts via target serving stack)

## v4 Requirements (deferred)

Deferred to future release. Tracked but not in current roadmap.

### Extended Capabilities

- **V4-01**: JavaScript/Gutenberg block generation via `<wp_block>` task token
- **V4-02**: Multi-lingual comment support (non-English PHPDoc/i18n)
- **V4-03**: Safety harness integration for production guardrails and red-teaming
- **V4-04**: Triton/TensorRT-LLM optimized inference engine

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
| DATA-07 | Phase 2 | Complete |
| DATA-08 | Phase 2 | Complete |
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
| DPLT-01 | Phase 5 | Deferred → v2.0 PRUNE-05 |
| DPLT-02 | Phase 5 | Deferred → v2.0 PKG-03 |
| DPLT-03 | Phase 5 | Deferred → v2.0 PKG-03 |
| DPLT-04 | Phase 5 | Deferred → v2.0 PKG-05 |
| DPLT-05 | Phase 5 | Deferred → v2.0 PKG-05 |
| DPLT-06 | Phase 5 | Deferred → v2.0 PKG-04 |
| DPLT-07 | Phase 5 | Deferred → v2.0 PKG-05 |

| ADPT-01 | Phase 6 | Complete |
| ADPT-02 | Phase 6 | Complete |
| ADPT-03 | Phase 6 | Complete |
| BTCH-01 | Phase 6 | Complete |
| BTCH-02 | Phase 6 | Complete |
| BTCH-03 | Phase 6 | Complete |
| TELE-01 | Phase 6 | Complete |
| TELE-02 | Phase 6 | Complete |
| TELE-03 | Phase 6 | Complete |
| TELE-04 | Phase 6 | Complete |
| PROB-01 | Phase 6 | Complete |
| PROB-02 | Phase 6 | Complete |
| PROB-03 | Phase 6 | Complete |

| PROF-01 | Phase 7 | Pending |
| PROF-02 | Phase 7 | Pending |
| PROF-03 | Phase 7 | Pending |
| PROF-04 | Phase 7 | Pending |
| PROF-05 | Phase 7 | Pending |
| GATE-01 | Phase 7 | Pending |
| GATE-02 | Phase 4 | Complete |
| SIEVE-01 | Phase 8 | Pending |
| SIEVE-02 | Phase 8 | Pending |
| SIEVE-03 | Phase 8 | Pending |
| SIEVE-04 | Phase 8 | Pending |
| SIEVE-05 | Phase 8 | Pending |
| EVAL2-01 | Phase 9 | Pending |
| EVAL2-02 | Phase 9 | Pending |
| GRPO-01 | Phase 10 | Pending |
| GRPO-02 | Phase 10 | Pending |
| GRPO-03 | Phase 10 | Pending |
| GRPO-04 | Phase 10 | Pending |
| GRPO-05 | Phase 11 | Pending |
| GRPO-06 | Phase 11 | Pending |
| GRPO-07 | Phase 11 | Pending |
| GRPO-08 | Phase 11 | Pending |
| MERGE-01 | Phase 12 | Pending |
| PRUNE-01 | Phase 12 | Pending |
| PRUNE-02 | Phase 12 | Pending |
| PRUNE-03 | Phase 12 | Pending |
| PRUNE-04 | Phase 12 | Pending |
| PRUNE-05 | Phase 12 | Pending |
| PRUNE-06 | Phase 12 | Pending |
| EVAL3-01 | Phase 13 | Pending |
| EVAL3-02 | Phase 13 | Pending |
| PKG-01 | Phase 14 | Pending |
| PKG-02 | Phase 14 | Pending |
| PKG-03 | Phase 14 | Pending |
| PKG-04 | Phase 14 | Pending |
| PKG-05 | Phase 14 | Pending |

**Coverage:**
- v1 requirements: 37 total (32 complete, 5 eval pending)
- v1.1 requirements: 13 total (13 complete)
- v2.0 requirements: 14 total (0 complete) — PROF(5) + GATE(2) + SIEVE(5) + EVAL2(2)
- v3.0 requirements: 22 total (0 complete) — GRPO(8) + MERGE(1) + PRUNE(6) + EVAL3(2) + PKG(5)
- DPLT requirements: 7 total (deferred → v3.0 PKG/PRUNE)
- Total mapped to phases: 82
- Unmapped: 0

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-04-02 — v2.0 revised (pruning/packaging moved to v3.0), v3.0 GRPO & Production Deployment requirements added*
