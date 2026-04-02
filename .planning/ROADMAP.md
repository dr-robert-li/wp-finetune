# Roadmap: wp-qwen3-moe

## Milestones

- 🚧 **v1.0 MVP** - Phases 1-5 (3 of 5 complete, eval + deployment remaining)
- ✅ **v1.1 Adaptive Training Infrastructure** - Phase 6 (complete 2026-04-01)
- 📋 **v2.0 MoE-Sieve Selective Training** - Phases 7-9 (planned)
- 📋 **v3.0 GRPO & Production Deployment** - Phases 10-14 (planned)

## Overview

Six phases take the project from fragile pipeline scripts to a trained dual-mode WordPress code model with adaptive training infrastructure. Phases 1-3 built the data pipeline, prepared the model, and trained it. Phase 4 evaluates quality gates, Phase 5 is deferred, and Phase 6 adds power-primary adaptive training exploiting DGX Spark thermal headroom.

Phases 7-9 (v2.0) implement MoE-Sieve selective expert training — profiling which experts each task token activates, then retraining only those experts with task-aware data routing and a k-sweep across three budgets. Phase 4 must complete before Phase 7 (need winning gen/judge ratio). Phase 9 gates Phase 10.

Phases 10-14 (v3.0) apply GRPO reinforcement learning on the MoE-Sieve model, then merge the LoRA adapter into base weights, prune with REAP on the final routing distribution, evaluate against v2.0, and package for production. LoRA must be merged before REAP runs — activation magnitudes require the unified model.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>v1.0 MVP (Phases 1-3) - Complete</summary>

- [x] **Phase 1: Pipeline Ready** - Harden all pipeline scripts and convert existing CSV data into repos.yaml before any data is generated
- [x] **Phase 2: Dataset Production** - Execute all three pipeline phases to produce the final training dataset (completed 2026-03-29 via /run-data-pipeline skill)
- [x] **Phase 3: Model Prep and Training** - Download Qwen3-30B-A3B, extend tokenizer, write eval suite, and fine-tune on DGX Spark (completed 2026-03-27)

</details>

- [ ] **Phase 4: Evaluation** - Run static eval suite + wp-bench, human review of results
- [ ] **Phase 5: Packaging and Deployment** - Quantize, serve, and publish to HuggingFace (deferred to v3.0 — subsumed by Phase 14)
- [ ] **Phase 6: Adaptive Training Planner** - Power-primary adaptive config engine with batch coupling, telemetry extensions, and warmup probes

<details>
<summary>v2.0 MoE-Sieve Selective Training (Phases 7-9) — Planned</summary>

- [ ] **Phase 7: Router Profiling** - Gradient-free profiling pass tagging expert routing counts by task token affinity, with stability verification and concentration report
- [ ] **Phase 8: Selective Training (MoE-Sieve)** - Retrain with LoRA targeting only hot experts, using task-aware data filtering and k-sweep to find the optimal expert budget
- [ ] **Phase 9: Comparative Evaluation** - A/B compare each k-sweep MoE-Sieve adapter against v1.0 full-LoRA on wp-bench and all 9 eval dimensions

</details>

<details>
<summary>v3.0 GRPO & Production Deployment (Phases 10-14) — Planned</summary>

- [ ] **Phase 10: Reward Infrastructure** - Build composite reward pipeline (70% verifiable / 30% judge) with security hard gate, MO-GRPO normalization, and VeRPO partial credit
- [ ] **Phase 11: GRPO Training** - Gen-only GRPO on hot experts with RSPO router-shift stabilization and collapse monitoring
- [ ] **Phase 12: LoRA Merge & Expert Pruning (AIMER vs REAP)** - Merge adapters, run both AIMER (weight-based) and REAP (calibration-based) at 3 compression ratios, compare to determine if WordPress specialization benefits domain-aware pruning
- [ ] **Phase 13: Comparative Evaluation** - A/B compare GRPO+pruned model against v2.0 SFT-only on wp-bench, all 9 dimensions, speed delta, and model size
- [ ] **Phase 14: Packaging** - Cascading compression gates (bf16 baseline → quantization decision → HuggingFace upload → E2E inference validation)

</details>

## Phase Details

<details>
<summary>v1.0 MVP Phase Details (Phases 1-3)</summary>

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
  3. eval/eval_gen.py, eval/eval_judge.py, and eval/eval_gate.py are runnable against any served checkpoint before the training run finishes
  4. adapters/qwen3-wp/ exists as a LoRA adapter checkpoint with tokenizer (adapter kept separate until evaluation passes in Phase 4)
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Model download, tokenizer extension, config, and test scaffolds (download Qwen3-30B-A3B, add task tokens, mean-init embeddings, smoke test)
- [x] 03-02-PLAN.md — Evaluation suite in eval/ directory (eval_gen.py PHPCS pass rate, eval_judge.py Spearman correlation, eval_gate.py quality gates, wp-bench config)
- [ ] 03-03-PLAN.md — Training script and merge adapter (Unsloth LoRA config, DGX Spark run, W&B monitoring, adapter save, merge with verification)

</details>

### Phase 4: Evaluation (Triage)
**Goal**: Run all 3 ratio adapters (30/70, 40/60, 50/50) through quality gates and wp-bench to triage — eliminate ratios that clearly fail, carry survivors to Phase 7 profiling where routing concentration determines the final winner
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, GATE-02
**Success Criteria** (what must be TRUE):
  1. All 3 ratio adapters evaluated: PHPCS pass rate, judge Spearman correlation, security pass rate
  2. At least one ratio exceeds all hard gates (PHPCS >95%, Spearman >0.85, Security >98%)
  3. wp-bench execution and knowledge tests run for all 3 ratios with scores recorded
  4. Triage decision: ratios that fail hard gates or are >5pp behind the best are eliminated; all others survive to Phase 7 profiling (high bar for elimination, low bar for continuation)
  5. Human has reviewed all eval results and approved triage decisions
**Plans**: TBD

Plans:
- [ ] 04-01: Serve model via vLLM + run static eval suite (PHPCS, Spearman, security gate)
- [ ] 04-02: wp-bench evaluation (clone, install, run WordPress runtime benchmark)
- [ ] 04-03: Human review checkpoint (inspect all eval results, approve or iterate)

### Phase 5: Packaging and Deployment
**Goal**: Model is quantized, served on all DGX Toolbox endpoints, and published to HuggingFace Hub
**Depends on**: Phase 4 (human-approved eval results)
**Requirements**: DPLT-01, DPLT-02, DPLT-03, DPLT-04, DPLT-05, DPLT-06, DPLT-07
**Success Criteria** (what must be TRUE):
  1. LoRA adapter merged into base model (or served via --lora-modules)
  2. AWQ 4-bit quantization produced for vLLM production serving (~8GB)
  3. GGUF quantization produced for Ollama local serving (~9GB)
  4. Model responds at vLLM (:8020), Ollama (:11434), LiteLLM (:4000), Open-WebUI (:12000)
  5. HuggingFace Hub page has model card with eval metrics (including wp-bench scores), quantized download links, and usage examples
**Plans**: TBD

Plans:
- [ ] 05-01: Packaging (merge LoRA adapter, AWQ quantization, GGUF quantization)
- [ ] 05-02: Deployment (vLLM serve, Ollama serve, LiteLLM proxy, Open-WebUI demo)
- [ ] 05-03: HuggingFace Hub upload (model card, benchmarks, download links, usage examples)

---

### v1.1 Adaptive Training Infrastructure

**Milestone Goal:** Replace temperature-zone adaptive planner with power-primary decision engine that correctly exploits the DGX Spark GB10 thermal envelope, plus Unsloth override detection and extended warmup probes.

**Dependency:** dgx-toolbox Phase 13 (telemetry/ package) must be complete before execution.

### Phase 6: Adaptive Training Planner
**Goal**: Training runs automatically adapt batch size, prefetch, workers, and save/eval intervals based on real-time GPU power telemetry, with correct batch/grad_accum coupling and Unsloth override detection
**Depends on**: Phase 5 (v1.0 complete); dgx-toolbox Phase 13 (telemetry/ package)
**Requirements**: ADPT-01, ADPT-02, ADPT-03, BTCH-01, BTCH-02, BTCH-03, TELE-01, TELE-02, TELE-03, TELE-04, PROB-01, PROB-02, PROB-03
**Success Criteria** (what must be TRUE):
  1. Running the adaptive-planner skill with GPU at 50W (UNDERUTILIZED zone) recommends batch increase as Rung 1 action, and at 95W+ (THROTTLED zone) recommends batch decrease to 1 -- with temperature only overriding at >=82C regardless of power zone
  2. After any batch_size change, grad_accum is automatically recalculated so that batch_size * grad_accum equals the original effective_batch value (e.g., batch 4->8 causes grad_accum 4->2)
  3. When Unsloth silently overrides batch_size or grad_accum (visible in its startup banner), the override is detected, written to telemetry/training/_unsloth_actuals.json, and all subsequent planner decisions use the Unsloth actual values instead of config values
  4. MemoryWatchdogCallback writes GPU watts and mem_available_gb to canonical JSONL every 50 training steps (GPUSampler field names), and a failed run is classified as NORMAL/OOM/HANG/THERMAL by the failure classifier
  5. Warmup probe runs 3-5 real training steps (via dgx-toolbox probe.py) when batch is increased without a prior anchor, and the anchor store persists config+outcome history with cooldown tracking
**Plans**: 6 plans

Plans:
- [x] 06-01-PLAN.md — Core adaptive planner Python module + config (routing, coupling, ladder with tests)
- [x] 06-02-PLAN.md — Extend train_model.py (power sampling via GPUSampler, Unsloth detection via trainer.args, failure classification) + observe-training 82/85C
- [x] 06-03-PLAN.md — Adaptive-planner skill wrapper + run-training Step 8.5 replacement + dgx_toolbox.yaml mount
- [x] 06-04-PLAN.md — Cross-file integration verification + human review checkpoint
- [ ] 06-05-PLAN.md — [GAP CLOSURE] Batch downscale for CAPPED/THROTTLED zones (apply_ladder + tests + config)
- [ ] 06-06-PLAN.md — [GAP CLOSURE] PYTHONPATH fix + TELE-02 field name docs correction

---

### v2.0 MoE-Sieve Selective Training

**Milestone Goal:** Train only WordPress-active experts via routing-guided LoRA selection with task-aware data filtering and k-sweep to find the optimal expert budget. Produces MoE-Sieve adapter for GRPO refinement in v3.0. Pruning deferred to v3.0 — GRPO changes routing distribution, must prune on final routing.

**Dependency:** Phase 4 (Evaluation) must complete first — MoE-Sieve needs the winning gen/judge ratio.

### Phase 7: Router Profiling & Ratio Selection
**Goal**: Profile ALL surviving ratios from Phase 4 triage (not just a single winner), producing per-task expert affinity maps and routing concentration reports with E_eff metrics, then select the optimal ratio for single-track MoE-Sieve training based on combined eval quality + routing compressibility
**Depends on**: Phase 4 (surviving ratios from triage); Phase 6 (adaptive training infrastructure)
**Requirements**: PROF-01, PROF-02, PROF-03, PROF-04, PROF-05, GATE-01
**Success Criteria** (what must be TRUE):
  1. A profiling script runs a gradient-free forward pass hooking `Qwen3MoeSparseMoeBlock` gating outputs and produces per-layer routing count tables for EACH surviving ratio — not just one
  2. The routing tables report separate expert activation counts for `<wp_gen>` and `<wp_judge>` prefixed inputs per ratio
  3. Profiling on a 10% subsample achieves Jaccard similarity >=0.94 against the full-set ranking per ratio
  4. Concentration report per ratio includes: per-layer CV, cumulative coverage curves, layer-depth skew, and E_eff = exp(entropy) per layer with mean/max/variance summary — E_eff directly predicts pruning headroom
  5. Decision matrix combining Phase 4 eval score and Phase 7 E_eff selects the ratio with lowest E_eff at equivalent quality (within 2pp) — a single ratio is chosen for all subsequent MoE-Sieve, GRPO, and pruning work
**Plans**: TBD

### Phase 8: Selective Training (MoE-Sieve)
**Goal**: A LoRA-selective retrain applies adapters only to hot experts (per task affinity) plus shared components, with task-aware data routing and a k-sweep across three budgets to identify the smallest expert set matching full-LoRA quality
**Depends on**: Phase 7
**Requirements**: SIEVE-01, SIEVE-02, SIEVE-03, SIEVE-04, SIEVE-05
**Success Criteria** (what must be TRUE):
  1. The training run applies LoRA adapters to hot routed experts, all attention (Q/K/V/O), router gates, and 4 shared experts — and leaves cold routed experts frozen with no gradient flow
  2. Gen-hot experts receive only golden signal data (passed examples, synthetic good) while judge-hot experts receive the full spectrum (passed + failed + contrastive), verifiable by inspecting data routing assignments per expert group
  3. The training uses the best gen/judge ratio identified by Phase 4 eval, not a hardcoded ratio
  4. Three k-sweep runs complete at budgets of approximately 13, 32, and 64 active experts per layer, each producing a separate adapter checkpoint
  5. The optimal k is declared as the smallest budget where wp-bench score falls within ±1pp of full-LoRA, verified by TOST equivalence test (epsilon=2pp) across 3+ seeds
**Plans**: TBD

### Phase 9: Comparative Evaluation
**Goal**: Each k-sweep MoE-Sieve adapter is A/B compared against v1.0 full-LoRA on all 9 eval dimensions, producing the dimension-level report and seed variance analysis that gates v3.0 GRPO work
**Depends on**: Phase 8
**Requirements**: EVAL2-01, EVAL2-02
**Success Criteria** (what must be TRUE):
  1. An A/B eval runs each k-sweep MoE-Sieve adapter (all three k budgets) against v1.0 full-LoRA on wp-bench and the static eval suite, with results recorded per adapter
  2. The report covers all 9 eval dimensions per adapter, overall scores, inference speed delta, and seed variance — sufficient to identify the optimal k and confirm MoE-Sieve quality before proceeding to GRPO
**Plans**: TBD

---

### v3.0 GRPO & Production Deployment

**Milestone Goal:** Apply gen-only GRPO with composite verifiable rewards and RSPO router-shift stabilization, then merge LoRA, REAP prune on final routing distribution (25%/50%/75% compression), evaluate against v2.0, and package for production.

**Dependencies:** Phase 9 (MoE-Sieve eval results) must complete before Phase 10. LoRA merge (Phase 12 MERGE-01) must complete before REAP runs — activation magnitudes require the unified model.

### Phase 10: Reward Infrastructure
**Goal**: A composite reward pipeline is built and validated end-to-end before any GRPO training begins — PHPCS anchor, security hard gate, VeRPO partial credit, and MO-GRPO normalization all verified independently
**Depends on**: Phase 9 (MoE-Sieve eval results confirm readiness for GRPO)
**Requirements**: GRPO-01, GRPO-02, GRPO-03, GRPO-04
**Success Criteria** (what must be TRUE):
  1. The composite reward pipeline produces a scalar reward for any generation: 70% from verifiable signals (PHPCS pass rate, security scan, WordPress standards checks) and 30% from frozen wp_judge score
  2. A generation that fails the security scan receives total reward = 0 regardless of all other signal scores — verified by a test case where a secure-failing but otherwise high-quality generation scores zero
  3. All reward signals pass through MO-GRPO normalization — each signal is normalized by within-group variance before combination, and a single dominant signal cannot inflate total reward
  4. WordPress standards checks use VeRPO partial credit — each check is weighted by difficulty estimated from pass rate across group samples, and rarely-passed checks contribute more signal than common ones
**Plans**: TBD

### Phase 11: GRPO Training
**Goal**: Gen-only GRPO refines the MoE-Sieve model's generation quality on hot experts, with RSPO router-shift stabilization ensuring experts do not drift from their established routing patterns
**Depends on**: Phase 10
**Requirements**: GRPO-05, GRPO-06, GRPO-07, GRPO-08
**Success Criteria** (what must be TRUE):
  1. GRPO training applies gradients only to `<wp_gen>` generation tasks — `<wp_judge>` capability is completely frozen from SFT and receives no gradient updates
  2. GRPO gradients flow only to hot routed experts, attention layers, router gates, and shared experts — cold routed experts receive no updates, preserving structural stability
  3. RSPO router-shift ratio is computed between rollout and training phases, applied as stop-gradient floor multiplied into the clipped importance ratio before aggregation, and logged per step
  4. Training halts automatically if router-shift ratio exceeds the stability threshold — the halt is triggered by per-step monitoring, not a post-hoc check
**Plans**: TBD

### Phase 12: LoRA Merge & Expert Pruning (AIMER vs REAP)
**Goal**: Merge LoRA adapters into base weights, then run both AIMER (task-agnostic, weight-based) and REAP (domain-aware, calibration-based) at three compression ratios to determine whether WordPress domain specialization creates enough routing concentration for calibration-based pruning to outperform generalized weight-based pruning
**Depends on**: Phase 11
**Requirements**: MERGE-01, PRUNE-01, PRUNE-02, PRUNE-03, PRUNE-04, PRUNE-05, PRUNE-06
**Success Criteria** (what must be TRUE):
  1. Both LoRA adapters (MoE-Sieve + GRPO) are merged into base model weights — merged checkpoint produces identical outputs to adapter-on-base configuration
  2. AIMER runs on merged model at 25%, 50%, 75% compression (~1 second per ratio, no calibration needed) producing 3 pruning masks as task-agnostic baseline
  3. REAP runs on same merged model with WordPress calibration data at same 25%, 50%, 75% compression producing 3 domain-aware pruning masks
  4. All 6 variants (2 methods × 3 ratios) evaluated via gating mask across all 9 dimensions before any weight removal — comparison table visible before committing
  5. Domain specificity analysis: expert overlap between AIMER and REAP retention sets quantified per layer — high overlap = WordPress isn't specialized enough for calibration advantage; low overlap = REAP captures domain routing AIMER misses
  6. Winning method + ratio selected by dimension-level retention (especially D2_security), preferring higher compression at equivalent quality; final model physically pruned with router re-normalization
**Plans**: TBD

### Phase 13: Comparative Evaluation
**Goal**: The GRPO+pruned model is A/B compared against v2.0 SFT-only (MoE-Sieve without GRPO), with inference speed delta and model size reduction measured alongside the 9-dimension quality report
**Depends on**: Phase 12
**Requirements**: EVAL3-01, EVAL3-02
**Success Criteria** (what must be TRUE):
  1. An A/B eval runs the GRPO+pruned model against v2.0 SFT-only (best k MoE-Sieve adapter) on wp-bench and the static eval suite, with all results recorded
  2. The report covers all 9 eval dimensions, inference speed delta (expected significant improvement from pruning), model size reduction, and seed variance — sufficient to confirm the full v3.0 pipeline adds value before packaging
**Plans**: TBD

### Phase 14: Packaging
**Goal**: The pruned model passes cascading compression gates (bf16 baseline, optional quantization, format production) and is published to HuggingFace with full compression lineage, then validated end-to-end on the target serving stack
**Depends on**: Phase 13
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05
**Success Criteria** (what must be TRUE):
  1. Gate 1 completes — the pruned bf16 model's size, inference speed, and all 9 eval dimension scores are recorded as the quality baseline for subsequent compression decisions
  2. Gate 2 decision is documented — whether quantization is warranted based on pruned model size, deployment constraints, and Gate 1 performance margins, with reasoning recorded
  3. If quantization is warranted, incremental testing at Q8->Q6->Q5->Q4 stops at the lowest level holding within ±2pp of the Gate 1 baseline; quantization is the final step and is never applied before Gate 2 confirms it is needed
  4. The HuggingFace model card documents the full compression lineage (base -> MoE-Sieve -> GRPO -> merge -> REAP -> quantization level) with eval scores at each gate and usage examples for both task tokens
  5. E2E inference validation confirms both `<wp_gen>` and `<wp_judge>` prompts produce correct outputs via the final shipped format on the target serving stack (vLLM or Ollama)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14
Note: Phase 5 (Packaging/Deployment v1.0) is deferred — v3.0 Phase 14 replaces it as the production packaging step.
Note: Phase 9 gates Phase 10 — MoE-Sieve eval results must confirm readiness before GRPO begins.
Note: Phase 12 MERGE-01 must complete before REAP runs — activation magnitudes require the unified model.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Pipeline Ready | v1.0 | 2/2 | Complete | 2026-03-26 |
| 2. Dataset Production | v1.0 | 6/7 | Complete | 2026-03-29 |
| 3. Model Prep and Training | v1.0 | 3/3 | Complete | 2026-03-27 |
| 4. Evaluation | v1.0 | 0/3 | Not started | - |
| 5. Packaging and Deployment | v1.0 | 0/3 | Deferred to v3.0 | - |
| 6. Adaptive Training Planner | v1.1 | 6/6 | Complete | 2026-04-01 |
| 7. Router Profiling | v2.0 | 0/? | Not started | - |
| 8. Selective Training (MoE-Sieve) | v2.0 | 0/? | Not started | - |
| 9. Comparative Evaluation | v2.0 | 0/? | Not started | - |
| 10. Reward Infrastructure | v3.0 | 0/? | Not started | - |
| 11. GRPO Training | v3.0 | 0/? | Not started | - |
| 12. LoRA Merge & REAP Pruning | v3.0 | 0/? | Not started | - |
| 13. Comparative Evaluation | v3.0 | 0/? | Not started | - |
| 14. Packaging | v3.0 | 0/? | Not started | - |
