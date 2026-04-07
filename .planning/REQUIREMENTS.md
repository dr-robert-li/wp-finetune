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

- [x] **EVAL-01**: Custom eval script measures PHPCS pass rate on 500 held-out generation tasks (target >95%)
- [x] **EVAL-02**: Custom eval script measures judge Spearman correlation on 500 held-out scored pairs (target >0.85)
- [x] **EVAL-03**: Security pass rate measured on held-out tasks (target >98%)
- [x] **EVAL-04**: Eval scripts run via DGX Toolbox eval-toolbox container
- [x] **EVAL-05**: All three quality gates pass before proceeding to deployment
- [ ] **EVAL-06**: Per-example logging — eval_gen.py and eval_judge.py persist input prompt, raw model response, and extracted code alongside scores in per-example JSONL (enables human review, debugging, and GRPO reward signals)
- [ ] **EVAL-07**: eval_gate.py per-dimension gates read correct field names from eval output — currently reads `dimension_pass_rates`/`dimension_correlations` but scripts write `per_dimension` (dead code fix)

### Deployment (deferred to v2.0 Packaging)

DPLT requirements moved to v3.0 PKG-01 through PKG-05 — package after RL + MoE-Sieve + pruning, not the intermediate full-LoRA model. Serving requirements (vLLM, Ollama, Open-WebUI) covered by PKG-05 E2E validation.

- [ ] ~~**DPLT-01**: LoRA adapter merged into base model weights~~ -> subsumed by v3.0 PRUNE-05
- [ ] ~~**DPLT-02**: AWQ quantization produced for vLLM~~ -> subsumed by v3.0 PKG-03 (if warranted by Gate 2)
- [ ] ~~**DPLT-03**: GGUF quantization produced for Ollama~~ -> subsumed by v3.0 PKG-03 (if warranted by Gate 2)
- [ ] ~~**DPLT-04**: vLLM serving~~ -> subsumed by v3.0 PKG-05
- [ ] ~~**DPLT-05**: Ollama serving~~ -> subsumed by v3.0 PKG-05
- [ ] ~~**DPLT-06**: HuggingFace upload~~ -> subsumed by v3.0 PKG-04
- [ ] ~~**DPLT-07**: Open-WebUI demo~~ -> subsumed by v3.0 PKG-05

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

## v1.2 Requirements — Judge Reasoning Fine-Tune

Requirements for deep reasoning fine-tuning of the winning ratio adapter. Depends on Phase 4 eval triage completing (need winning ratio). Must complete before v2.0 RL Alignment (routing profile must reflect reasoning capability).

### Data Generation

- [ ] **DGEN-01**: Pilot validation generates 20-50 deep judge CoT examples and 20-50 critique-then-fix examples, reviewed for quality before bulk generation
- [ ] **DGEN-02**: Deep judge CoT agent generates reasoning-enriched judge examples with dimension-by-dimension analysis, issue identification with line references, fix suggestions, and structured scores
- [ ] **DGEN-03**: Critique-then-fix agent generates examples where defective code (from mutation pool) receives structured critique with severity per dimension, followed by the corrected version
- [ ] **DGEN-04**: Score consistency validation rejects examples where reasoning text contradicts numeric scores before export
- [ ] **DGEN-05**: Reasoning dataset is assembled with training mix: reasoning examples + 30% flat judge replay + 20% wp_gen replay examples

### Reasoning Training

- [ ] **RTRN-01**: Training config uses 5-10x lower learning rate than Phase 3 (max ~2e-5) with warmup, continuing from winning ratio adapter
- [ ] **RTRN-02**: max_seq_length increased to 8192 to accommodate full reasoning chains
- [ ] **RTRN-03**: MoE router weights are confirmed frozen during continued training (no routing shift)
- [ ] **RTRN-04**: Training completes 1-2 epochs on combined reasoning dataset without OOM or loss divergence

### Reasoning Evaluation

- [ ] **REVL-01**: eval_judge.py Spearman correlation on reasoning adapter meets or exceeds winning ratio baseline
- [ ] **REVL-02**: eval_gen.py PHPCS pass rate on reasoning adapter shows no regression (within 2pp of baseline)
- [ ] **REVL-03**: Reasoning quality evaluated by separately spawned Claude evaluator agent (independent context window, receives only generated code + reasoning as opaque inputs, no shared state with model under test) measuring dimension coverage (all 9 dimensions addressed) and score-reasoning consistency
- [ ] **REVL-04**: wp-bench scores on reasoning adapter meet or exceed winning ratio baseline
- [ ] **REVL-05**: Human reviews sample of reasoning outputs to confirm quality before declaring v1.2 complete
- [ ] **REVL-06**: Fix correctness — critique-then-fix corrected code verified through PHPCS + security scanner to confirm the fix actually resolves the identified issue
- [ ] **REVL-07**: Classification accuracy — confusion matrix (TP/TN/FP/FN) computed at score thresholds from eval_judge.py per-example data, measuring whether the model correctly distinguishes good from bad code
- [ ] **REVL-08**: Reasoning length distribution — reasoning chains are neither truncated nor exploding; median, p95, and max token counts recorded and reviewed against expected range

## v2.0 Requirements — RL Alignment

Requirements for RL alignment before MoE-Sieve. Depends on v1.2 completing (need reasoning-enhanced adapter). Pipeline reordered per Issue #1 (D-07): RL runs before MoE-Sieve because routing statistics should reflect reward-aligned behavior, not SFT pre-training usage.

### Router Profiling

- [ ] **PROF-01**: Router profiling runs gradient-free forward pass hooking `Qwen3MoeSparseMoeBlock` gating output, count-based ranking per layer
- [ ] **PROF-02**: Profiling tags each expert's routing count by task token affinity (`<wp_gen>` vs `<wp_judge>`) separately, not just aggregate frequency
- [ ] **PROF-03**: Profiling uses 10% subsample with Jaccard stability verification against full set (target ≥0.94)
- [ ] **PROF-04**: Outputs routing concentration report: per-layer CV, cumulative coverage curve at each k, layer-depth skew analysis, and effective expert count E_eff = exp(entropy) per layer (mean, max, variance across layers) — E_eff directly predicts pruning headroom
- [ ] **PROF-05**: Profile ALL surviving ratios from Phase 4 triage (not just the winner) — profiling is ~minutes per ratio and routing concentration is a critical decision signal for ratio selection

### Ratio Selection Gate (Phase 7→8)

- [ ] **GATE-01**: Decision matrix combining Phase 4 eval score (normalized 0-1) and Phase 7 routing concentration (mean E_eff, max E_eff, E_eff variance) per surviving ratio — select ratio with lowest E_eff at equivalent quality (within 2pp), preferring compressibility over marginal quality gains
- [x] **GATE-02**: Phase 4 triage uses high bar for elimination (only cut ratios that fail hard gates or are >5pp behind) and low bar for continuation — 1-2pp differences may invert after pruning if routing concentration differs

### Reward Infrastructure

- [ ] **GRPO-01**: Composite reward pipeline with 70% verifiable / 30% judge weighting — PHPCS pass rate (high-variance anchor), security scanner (hard gate: score=0 on failure), WordPress standards checks (VeRPO partial credit weighted by check difficulty), frozen wp_judge score (MO-GRPO normalized)
- [ ] **GRPO-02**: Security scanner hard gate — if generation fails security scan, total reward = 0 regardless of all other scores (non-negotiable safety floor)
- [ ] **GRPO-03**: MO-GRPO normalization on all reward signals — each signal normalized by within-group variance to prevent single-signal dominance
- [ ] **GRPO-04**: VeRPO-style partial credit for WordPress standards checks — each check weighted by difficulty (estimated from pass rate across group samples; rarely-passed checks contribute more signal)

### GSPO/GRPO Training

- [ ] **GRPO-05**: Dual-mode RL — both `<wp_gen>` generation quality and `<wp_judge>` reasoning quality improved via RL. Gen rewards: PHPCS + security + VeRPO. Judge rewards: score-reasoning consistency (via separately spawned Claude evaluator agent), fix correctness (PHPCS/security scanner on critique-then-fix corrected code). Judge is the primary bottleneck (Spearman 0.57 vs gen 0.99+ at SFT stage) and receives equal or greater RL budget.
- [ ] **GRPO-06**: Full-MoE RL — GSPO/GRPO gradients flow to all routed experts + attention + router gates + shared experts. Protected expert set from Phase 7 monitored via routing regularizer (KL divergence penalty if protected experts deactivate below baseline frequency). GSPO (sequence-level) preferred per D-08; GRPO with larger group size + Pro-GRPO expand-then-prune as fallback.
- [ ] **GRPO-07**: RSPO router-shift stabilization — compute router-shift ratio between rollout and training phases, apply stop-gradient and floor, multiply into clipped importance ratio before aggregation
- [ ] **GRPO-08**: Router-shift ratio monitored throughout training — log per-step shift metrics; halt training if shift exceeds stability threshold (routing collapse early warning)

### RL Comparative Evaluation

- [ ] **RLEV-01**: RL model (GSPO/GRPO output) evaluated against v1.2 SFT baseline on wp-bench and all 9 eval dimensions — no dimension regression permitted; judge Spearman improvement expected (primary RL target)
- [ ] **RLEV-02**: RL evaluation report includes reward metric convergence curves, router-shift stability log (per-step shift ratios), protected expert retention rate vs Phase 7 baseline, gen/judge quality delta, and anti-hack eval results

## v3.0 Requirements — MoE-Sieve, Pruning & Packaging

Requirements for MoE-Sieve on the RL-trained model, followed by LoRA merge, pruning, evaluation, and production packaging. MoE-Sieve operates post-RL using RL-policy routing logs (not SFT logs) per Issue #1 (D-07). AIMER is the primary pruning method (D-09); REAP is an optional comparison.

**Note:** MoE-Sieve in v3.0 operates on the RL-trained model using RL-policy routing logs (not SFT logs). A fresh profiling pass is required before sieve selection.

### Selective Training (MoE-Sieve) — Post-RL

- [ ] **SIEVE-01**: Fresh routing profiling on RL-trained model; LoRA r=32, alpha=64, dropout=0.05 applied to hot routed experts (per RL-policy routing) + all attention (Q/K/V/O) + router gates + 4 shared experts (always trained); cold routed experts frozen. Protected experts from Phase 7 must be in the retained set.
- [ ] **SIEVE-02**: Gen-hot experts (per RL-policy routing categories) trained on golden signal data only (passed examples, synthetic good); judge-hot experts trained on full spectrum (passed + failed + contrastive)
- [ ] **SIEVE-03**: Retrain uses best gen/judge ratio determined by Phase 4 eval results
- [ ] **SIEVE-04**: K-sweep at minimum 3 budgets (~13, 32, 64 experts per layer from 128 routed) to find accuracy plateau for Qwen3-30B-A3B on WordPress data
- [ ] **SIEVE-05**: Optimal k is smallest budget matching full-LoRA within +/-1pp on wp-bench (TOST equivalence test, epsilon=2pp, 3+ seeds); all protected experts from Phase 7 must be retained at optimal k

### MoE-Sieve Comparative Evaluation

- [ ] **EVAL2-01**: A/B eval of each k-sweep MoE-Sieve adapter against v2.0 RL baseline on wp-bench and static eval suite
- [ ] **EVAL2-02**: Report includes per-dimension comparison (all 9 dimensions), overall scores, inference speed delta, and seed variance comparison

### LoRA Merge & Pruning (AIMER primary, REAP optional)

Sub-experiment: Does WordPress domain specialization create enough routing concentration for calibration-based pruning (REAP) to outperform weight-based pruning (AIMER)? Or is PHP/WordPress too close to general code for domain-aware pruning to differentiate? AIMER is the primary pruning method (D-09) — calibration-free and fast iteration. REAP is an optional comparison.

- [ ] **MERGE-01**: Merge MoE-Sieve + RL LoRA adapters into base model weights before pruning — REAP needs activation magnitudes from the unified model, AIMER needs final weight norms
- [ ] **PRUNE-01**: Run AIMER pruning on merged model (weight-based, no calibration, ~1 second) at 25%, 50%, and 75% compression ratios — serves as primary pruning method (D-09)
- [ ] **PRUNE-02**: Optionally run REAP pruning on same merged model with WordPress calibration data (gen + judge examples), `reap` saliency scoring, at same 25%, 50%, 75% compression ratios — serves as domain-aware comparison
- [ ] **PRUNE-03**: Evaluate both methods via gating mask before weight removal — compare retention across all 9 eval dimensions at each compression ratio (6 variants total: 2 methods × 3 ratios)
- [ ] **PRUNE-04**: Analyze domain specificity signal: compare which experts each method retains/prunes — high overlap suggests WordPress isn't specialized enough for calibration-based advantage; low overlap suggests REAP is capturing domain-specific routing patterns AIMER misses
- [ ] **PRUNE-05**: Select winning method + compression ratio with best dimension-level retention (especially D2_security), prefer higher compression at equivalent quality; if regression on any dimension, reduce compression incrementally until clean
- [ ] **PRUNE-06**: Final model has expert weights physically removed and router softmax re-normalized for removed expert slots; saved as HuggingFace-compatible checkpoint; pruning methodology documented in model card

### Final Comparative Evaluation

- [ ] **EVAL3-01**: A/B eval of pruned model against v2.0 RL baseline on wp-bench and static eval suite
- [ ] **EVAL3-02**: Report includes per-dimension comparison, inference speed delta (expect significant from pruning), model size reduction, and seed variance

### Packaging (cascading compression gates)

- [ ] **PKG-01**: Gate 1 — Eval pruned bf16 model: record size, inference speed, all 9 dimensions as quality baseline for subsequent compression
- [ ] **PKG-02**: Gate 2 — Assess whether quantization is needed based on pruned model size, deployment constraints, and Gate 1 performance margins
- [ ] **PKG-03**: If quantization warranted, test incrementally Q8→Q6→Q5→Q4, eval at each level, stop at lowest quantization holding within ±2pp of Gate 1 baseline
- [ ] **PKG-04**: Model card + adapter uploaded to HuggingFace with full compression lineage (base -> RL -> MoE-Sieve -> merge -> AIMER/REAP winner -> quantization level, eval at each gate) including AIMER vs REAP comparison results
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
| EVAL-06 | Phase 4 | Pending |
| EVAL-07 | Phase 4 | Pending |
| DPLT-01 | Phase 5 | Deferred -> v3.0 PRUNE-05 |
| DPLT-02 | Phase 5 | Deferred -> v3.0 PKG-03 |
| DPLT-03 | Phase 5 | Deferred -> v3.0 PKG-03 |
| DPLT-04 | Phase 5 | Deferred -> v3.0 PKG-05 |
| DPLT-05 | Phase 5 | Deferred -> v3.0 PKG-05 |
| DPLT-06 | Phase 5 | Deferred -> v3.0 PKG-04 |
| DPLT-07 | Phase 5 | Deferred -> v3.0 PKG-05 |

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

| DGEN-01 | Phase 4.1 | Pending |
| DGEN-02 | Phase 4.1 | Pending |
| DGEN-03 | Phase 4.1 | Pending |
| DGEN-04 | Phase 4.2 | Pending |
| DGEN-05 | Phase 4.2 | Pending |
| RTRN-01 | Phase 4.3 | Pending |
| RTRN-02 | Phase 4.3 | Pending |
| RTRN-03 | Phase 4.3 | Pending |
| RTRN-04 | Phase 4.3 | Pending |
| REVL-01 | Phase 4.4 | Pending |
| REVL-02 | Phase 4.4 | Pending |
| REVL-03 | Phase 4.4 | Pending |
| REVL-04 | Phase 4.4 | Pending |
| REVL-05 | Phase 4.4 | Pending |
| REVL-06 | Phase 4.4 | Pending |
| REVL-07 | Phase 4.4 | Pending |
| REVL-08 | Phase 4.4 | Pending |

| PROF-01 | Phase 7 | Pending |
| PROF-02 | Phase 7 | Pending |
| PROF-03 | Phase 7 | Pending |
| PROF-04 | Phase 7 | Pending |
| PROF-05 | Phase 7 | Pending |
| GATE-01 | Phase 7 | Pending |
| GATE-02 | Phase 4 | Complete |
| GRPO-01 | Phase 8 | Pending |
| GRPO-02 | Phase 8 | Pending |
| GRPO-03 | Phase 8 | Pending |
| GRPO-04 | Phase 8 | Pending |
| GRPO-05 | Phase 9 | Pending |
| GRPO-06 | Phase 9 | Pending |
| GRPO-07 | Phase 9 | Pending |
| GRPO-08 | Phase 9 | Pending |
| RLEV-01 | Phase 10 | Pending |
| RLEV-02 | Phase 10 | Pending |
| SIEVE-01 | Phase 11 | Pending |
| SIEVE-02 | Phase 11 | Pending |
| SIEVE-03 | Phase 11 | Pending |
| SIEVE-04 | Phase 11 | Pending |
| SIEVE-05 | Phase 11 | Pending |
| EVAL2-01 | Phase 12 | Pending |
| EVAL2-02 | Phase 12 | Pending |
| MERGE-01 | Phase 13 | Pending |
| PRUNE-01 | Phase 13 | Pending |
| PRUNE-02 | Phase 13 | Pending |
| PRUNE-03 | Phase 13 | Pending |
| PRUNE-04 | Phase 13 | Pending |
| PRUNE-05 | Phase 13 | Pending |
| PRUNE-06 | Phase 13 | Pending |
| EVAL3-01 | Phase 14 | Pending |
| EVAL3-02 | Phase 14 | Pending |
| PKG-01 | Phase 15 | Pending |
| PKG-02 | Phase 15 | Pending |
| PKG-03 | Phase 15 | Pending |
| PKG-04 | Phase 15 | Pending |
| PKG-05 | Phase 15 | Pending |

**Coverage:**
- v1 requirements: 39 total (32 complete, 7 eval pending)
- v1.1 requirements: 13 total (13 complete)
- v1.2 requirements: 17 total (0 complete) — DGEN-01/02/03 -> Phase 4.1; DGEN-04/05 -> Phase 4.2; RTRN-01/02/03/04 -> Phase 4.3; REVL-01/02/03/04/05/06/07/08 -> Phase 4.4
- v2.0 requirements: 17 total (0 complete) — PROF(5) + GATE(2) + GRPO(8) + RLEV(2) [Phases 7-10]
- v3.0 requirements: 21 total (0 complete) — SIEVE(5) + EVAL2(2) + MERGE(1) + PRUNE(6) + EVAL3(2) + PKG(5) [Phases 11-15]
- DPLT requirements: 7 total (deferred -> v3.0 PKG/PRUNE)
- Total mapped to phases: 98 (all requirements mapped, 0 unmapped; +2 from RLEV-01/02)

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-04-08 — Pipeline reordered per Issue #1 (D-07): RL before MoE-Sieve; GRPO-01-04 moved to Phase 8, GRPO-05-08 to Phase 9, SIEVE to Phase 11, EVAL2 to Phase 12, MERGE/PRUNE to Phase 13, EVAL3 to Phase 14, PKG to Phase 15; GRPO-05/06 updated for full-MoE RL + GSPO primary (D-08); RLEV-01/02 added for Phase 10; SIEVE updated for post-RL context; AIMER primary (D-09), REAP optional*
