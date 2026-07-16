# Roadmap: wp-qwen3-moe

## Milestones

- ✅ **v1.0 MVP** - Phases 1-5 (Phases 1-4 complete; Phase 5 deferred/closed → v3.0 Phase 15)
- ✅ **v1.1 Adaptive Training Infrastructure** - Phase 6 (complete 2026-04-01)
- ✅ **v1.2 Judge Reasoning Fine-Tune** - Phases 4.1-4.4 (complete; relabel SFT → v1.3 judge promoted 2026-07-04)
- ❌ **v2.0 RL Alignment** - Phases 7-10 (infra COMPLETE; RL REJECTED at Phase 10 gate, closed 2026-07-05)
- ✅ **v3.0 MoE-Sieve, Pruning & Packaging** - Phases 11-16 (complete 2026-07-11 — Sieve full, prune no_winner, Q8 GGUF lossless ship tier, pipeline locked in PIPELINE.md)
- ✅ **v3.1 Benchmark, Publish & Next Base** - Phases 17-19 (complete 2026-07-12 — wp-bench + SWE-bench gen eval, HF pair published PUBLIC, V4 rerun roadmap locked on Qwen3.6-35B-A3B)
- 🚧 **v4.0 Pipeline Rerun on Qwen3.6-35B-A3B** - Phases 20-27 (ACTIVE — started 2026-07-12; base Qwen/Qwen3.6-35B-A3B LOCKED; targets judge rho >0.85 single-seed / >0.87 ensemble vs the 0.8075 wall, wp-bench floor 0.4286)

## Overview

Six phases take the project from fragile pipeline scripts to a trained dual-mode WordPress code model with adaptive training infrastructure. Phases 1-3 built the data pipeline, prepared the model, and trained it. Phase 4 evaluates quality gates, Phase 5 is deferred, and Phase 6 adds power-primary adaptive training exploiting DGX Spark thermal headroom.

Phases 4.1-4.4 (v1.2) add deep judge reasoning capability to the winning ratio adapter — generating reasoning-enriched judge data and critique-then-fix pairs, continued fine-tuning at lower LR, and re-evaluating before proceeding. Phase 4 triage (identifying the winning adapter) is a hard prerequisite. The v1.2 reasoning adapter must be complete before Phase 7 because routing profiles must reflect the final reasoning capability.

Phases 7-10 (v2.0) implement RL alignment per Issue #1's recommended order: first profile routing and identify the protected expert set (Phase 7), then build reward infrastructure with anti-hack eval (Phase 8), then run GSPO on the FULL MoE (Phase 9), and finally evaluate RL output against the v1.2 SFT baseline (Phase 10). RL runs before MoE-Sieve because "routing statistics should reflect reward-aligned behavior, not SFT pre-training usage" (Issue #1). GSPO (sequence-level) is the primary RL objective for MoE stability (D-08). Whether GRPO is also evaluated as a fallback is an optional decision deferred to Phase 9 planning time. Phase 10 gates Phase 11.

Phases 11-15 (v3.0) apply MoE-Sieve on the RL-trained model using RL-policy routing logs (Phase 11), evaluate the sieved model (Phase 12), merge LoRA and prune with AIMER (primary, D-09) or REAP (optional comparison) on the final routing distribution (Phase 13), evaluate against v2.0 (Phase 14), and package for production (Phase 15). MoE-Sieve operates post-RL so that sieve selection reflects reward-aligned routing, not SFT routing. LoRA must be merged before pruning runs — activation magnitudes require the unified model.

**AMENDMENT 2026-07-03 (v3.0 base = v1.2 SFT):** RL was REJECTED at the Phase 10 gate and the
rejection held through two Phase-08.2 gated smokes (2026-07-01 calib-dead run; 2026-07-03 seedA2
honest hybrid@0.8 run — killed on G1, teacher-Spearman never left the noise band while the reward
stayed un-Goodharted). **v3.0 therefore proceeds on the v1.2 SFT model.** Everywhere Phases 11-15
say "RL-trained model" / "RL-policy routing logs" / "v2.0 RL baseline", read "v1.2 SFT model" /
"v1.2 SFT-policy routing logs" / "v1.2 SFT baseline". Phase 11's fresh profiling pass profiles the
v1.2 SFT policy (the Phase 7 profiles remain the protected-expert reference). A future RL reopening
requires: teacher-ceiling headroom ≥ ~0.1 (measured), a reward-v2 with materially stronger per-step
signal (defect-grounded / MO-GRPO), offline oracle gate, then ONE gated smoke — see
logs/phase09_rerun/SMOKE_READS_TALLY.md and .continue-here.md.

**AMENDMENT 2026-07-08 (two-model pair + judge ceiling closed):** The shipped v3.0 artifact is a
**two-model pair**, not a single model: **v1.3 3-seed median ensemble judge** (rho 0.842; single-seed
s1 0.827 fallback) for the wp_judge role + **v1.2 SFT generation model** (codegen bar 0.4616) for the
wp_gen role. The relabel SFT that produced v1.3 (0.748 → 0.827) is the judge; v1.2 stays the generator
because no single-model mix recovers both axes (judge-rho vs codegen orthogonal). A gap-closure
investigation (2026-07-08) confirmed judge rho 0.827 is a **local optimum** — capacity (rank64+attn
overfit 0.662), loss-reshaping (uniform CE is the peak), and data-cleaning (gap distributed) all fail
to beat it; the ceiling-moving lever is a stronger base model (future qwen3.6/3.7 work). Evidence:
`output/relabel/gap_closure_summary.json`. **Packaging implication for Phase 11:** the ensemble judge
is 3 LoRA seeds (3× judge inference) — Phase 11 must decide whether MoE-Sieve/AIMER target the ensemble
or the leaner single-seed s1 (0.827). Phases 11-15 "v2.0 RL baseline" comparison targets read "v1.2 SFT
(gen) / v1.3 ensemble (judge) baseline".

**LLM Execution Pattern (MANDATORY across all phases):** ALL LLM-driven work — data generation, reasoning generation, judge evaluation, eval-by-LLM scoring — MUST use **Claude Code agents** (parallel `Agent(run_in_background=true)` spawning following the `wp-finetune:run-data-pipeline` SKILL.md pattern), NOT the Anthropic API directly. The direct API path (`call_with_backoff`, `anthropic.Anthropic()`) is acceptable only for small pilots (<50 examples). At scale (>100 examples), the direct API fails due to rate limits, timeout cascades, and error recovery brittleness; Claude Code agents use the subscription quota and parallelize reliably. This is a hard rule established from prior phase experience (Phase 1/2 generated 134K judged + 29K CoT examples successfully via agents).

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

- [x] **Phase 4: Evaluation** - Run static eval suite + wp-bench, human review of results (COMPLETE — triage identified the winning 30/70 ratio adapter; extended by v1.2 Phases 4.1-4.4)
- [x] **Phase 5: Packaging and Deployment** - Quantize, serve, and publish to HuggingFace (CLOSED — DEFERRED to v3.0/Phase 15; not implemented in v1.0, intentionally out of scope)
- [x] **Phase 6: Adaptive Training Planner** - Power-primary adaptive config engine with batch coupling, telemetry extensions, and warmup probes (COMPLETE 2026-04-01, v1.1)

<details>
<summary>v1.2 Judge Reasoning Fine-Tune (Phases 4.1-4.4) — INSERTED — depends on Phase 4 triage completing</summary>

- [x] **Phase 4.1: Reasoning Data Generation** - Curate human-annotated seeds, then pilot-validate and run parallel deep judge CoT and critique-then-fix data generation streams (COMPLETE — reasoning dataset produced)
- [x] **Phase 4.2: Dataset Assembly** - Score consistency validation, training mix assembly, and export of the reasoning dataset (COMPLETE — openai_train/val exported)
- [x] **Phase 4.3: Reasoning Fine-Tune** - Continued SFT on winning ratio adapter at lower LR with frozen router and 8192-token sequences (COMPLETE — Tinker MoE-only grid; r32-rp30 v4-winner selected)
- [x] **Phase 4.4: Reasoning Eval & Merge** - Verify reasoning adapter meets all quality gates; human review; merge adapter (COMPLETE 2026-06-14 — v4-winner promoted to canonical under D-V4-10 waiver; 10+10 validation 10/10·10/10)

</details>

<details>
<summary>v2.0 RL Alignment (Phases 7-10) — Planned</summary>

- [x] **Phase 7: Router Profiling & Protected Expert Set** - Gradient-free profiling pass tagging expert routing counts by task token affinity, identify dual-purpose experts that must not be pruned (D-10), with stability verification and concentration report (COMPLETE 2026-06-19 — all automated gates green under D-09 CI-aware; 1,480-expert protected mask exported; human sign-off APPROVED, council-unanimous on both judgment items)
- [x] **Phase 8: Reward Infrastructure** - Build composite reward pipeline (70% verifiable / 30% judge) with security hard gate, MO-GRPO normalization, VeRPO partial credit, and anti-hack eval set (D-11) (COMPLETE 2026-06-20 — verified + HUMAN-UAT; consumed by Phase 9)
- [x] **Phase 8.1: Reward Redesign** - INSERTED 2026-06-24. Replace the effectively-binary 0.7 fix_correctness verifiable term with graded partial credit (fraction of PHPCS/security/syntax sub-checks) + rebalance/per-group diversity shaping so GSPO groups carry non-zero advantage, and add diagnostic logging (component means, frac_groups_all_zero, entropy). Triggered by Phase 9's flat-reward finding (binary reward → uniform groups → advantage collapse → vanishing gradient; RLEV-01 verdict FLAT). Gates a targeted Phase 9 rerun. (completed 2026-06-24)
- [x] **Phase 8.2: Reward Validity Gate** - INSERTED 2026-07-01. Close the gap 08.1 left: 08.1 fixed reward SHAPE (gradient existed) but never checked reward VALIDITY (does the reward track the validated target?). Build an offline reward-validity oracle (a candidate reward's per-checkpoint trajectory must rank-correlate with the validated teacher-Spearman target before any GPU), redesign the judge-axis reward around the only form that passes it (per-group pairwise rank-agreement vs teacher GT), add an in-run wp-bench codegen trip-wire, and gate a 50/250-step smoke on the VALIDATED metric. Triggered by Phase 10 RLEV verdict: seedA RL Goodharted — fix_correctness proxy rose +0.028 but teacher-Spearman didn't move and wp-bench regressed −0.049 (oracle: fix_correctness↔target corr −0.24 INVALID; pairwise_rank_agreement +0.70 VALID). Gates any future RL rerun. (planning) (completed 2026-07-01)
- [x] **Phase 9: GSPO Training** - Dual-mode RL (gen + judge reasoning) on FULL MoE with router-shift stabilization and collapse monitoring; GSPO (sequence-level) is the primary objective for MoE stability (D-08); GRPO is an optional fallback decided at Phase 9 planning time; protected experts from Phase 7 monitored — Complete 2026-06-20 (live Tinker run tracked in 09-HUMAN-UAT.md)
- [x] **Phase 10: RL Comparative Evaluation** - Compare RL model against v1.2 SFT baseline on wp-bench and all 9 eval dimensions; gates v3.0 (CLOSED 2026-07-05 — RL REJECTED: Goodharted proxy, no teacher-Spearman gain, wp-bench regression; v1.2 SFT stays canonical)

</details>

<details>
<summary>v3.0 MoE-Sieve, Pruning & Packaging (Phases 11-15) — Planned</summary>

- [x] **Phase 11: Post-RL MoE-Sieve** *(AMENDED 2026-07-03: RL rejected — operates on v1.2 SFT)* - Training-free Sieve: routing profile + k-sweep expert masking on v1.2 SFT (COMPLETE 2026-07-09 — optimal_k=full, no mask wins)
- [x] **Phase 12: MoE-Sieve Comparative Evaluation** - A/B compare k-sweep MoE-Sieve variants (SKIPPED 2026-07-10 — optimal_k=full leaves no variants to A/B)
- [x] **Phase 13: LoRA Merge & Pruning (AIMER primary, REAP optional)** - Merge adapters, AIMER/REAP pruning at 3 compression ratios (COMPLETE 2026-07-10 — no_winner: pruning gives nothing on this base; merged pair retained unpruned)
- [x] **Phase 14: Final Comparative Evaluation** - A/B eval, speed delta, model size (COMPLETE 2026-07-10 — re-confirmation PASS on unpruned pair; size flat)
- [x] **Phase 15: Packaging** - Cascading compression gates (COMPLETE 2026-07-11 — Gates 1-2 done, Q8 GGUF LOSSLESS ship tier 30.2 GiB −47%, model card + bf16 E2E validated; Q6/Q5 descent deferred, no deployment need; HF upload moved to Phase 18)
- [x] **Phase 16: Pipeline Lockdown & Repo Cleanup** - Freeze v3.0 method into PIPELINE.md, deprecate one-off scaffolding, clean folder layout (COMPLETE 2026-07-10)

</details>

<details>
<summary>v3.1 Benchmark, Publish & Next Base (Phases 17-19) — COMPLETE 2026-07-12</summary>

- [x] **Phase 17: Benchmark Expansion — wp-bench + SWE-bench Generation Eval** - Full wp-bench run on the shipped gen model plus a SWE-bench generation-mode eval to position the model against a public coding benchmark; document scores honestly in the model card
- [x] **Phase 18: Production Sweep & HuggingFace Publication** - Full repo sweep (docs current, stale artifacts to deprecated/, streamlined layout), then package the two-model pair (v1.2 gen + v1.3 judge ensemble, Q8 GGUF ship tier) and publish to HuggingFace (COMPLETE 2026-07-12 — both HF repos PUBLIC, verification PASSED 7/7)
- [x] **Phase 19: Next-Base Rerun Roadmap** - Plan the full pipeline rerun on the latest Qwen-family base (research current best same-class MoE), producing a costed roadmap for the next milestone (COMPLETE 2026-07-11 — base LOCKED Qwen3.6-35B-A3B, V4-RERUN-ROADMAP.md)

</details>

<details open>
<summary>v4.0 Pipeline Rerun on Qwen3.6-35B-A3B (Phases 20-27) — ACTIVE</summary>

- [x] **Phase 20: Base Bring-Up** - Download/load smoke, eos/pad token-ID alignment (Stage 1.5), DeltaNet-aarch64 serving smoke, VL merge-path round-trip check (completed 2026-07-13)
- [x] **Phase 21: SFT Training — Generation & Judge Models** - Stage 2/3 SFT on reused data (reasoning-mix gen, 3-seed relabel judge); pre-registered rho/wp-bench targets measured (completed 2026-07-14)
- [x] **Phase 22: Sieve Tooling Adaptation** - Adapt MoE-Sieve profiler + protected-mask pipeline for 256-expert/shared-expert/mixed-strata routing, ready before Gate B (completed 2026-07-15)
- [x] **Phase 23: Final Evaluation** - A/B eval vs v3.0 shipping figures under pre-registered criteria — the milestone's primary verdict (completed 2026-07-14)
- [ ] **Phase 24: Conditional Gate A — RL Re-Test** - RL re-test only with a materially different reward family; no_winner is a valid result
- [ ] **Phase 25: Conditional Gate B — MoE-Sieve Re-Test** - k-sweep re-test on adapted tooling (TOST, CI-aware); no_winner is a valid result
- [ ] **Phase 26: Conditional Gate C — Merge + Prune Re-Test** - AIMER/REAP gate-before-remove re-test; no_winner is a valid result
- [ ] **Phase 27: Packaging & Publication Refresh** - Q8 GGUF pair conversion + cascading compression gates + HF card lineage refresh

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

**Plans**: 3 plans

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
  5. The wp_gen and wp_judge example counts follow approximately 40/60 gen/judge split (per user decision) — **SUPERSEDED 2026-06-26:** static 40/60 target replaced by the ratio_30_70..70_30 export sweep; 30/70 chosen as Phase 4 triage winner (line 41). Accepted via 02-VERIFICATION override.

**Plans**: 7 plans

Plans:

- [x] 02-01-PLAN.md — Config updates (judge threshold >= 8, security auto-FAIL, N/A deflation, rejection templates) + Phase 1 script hardening (clone, extract, judge with utils.py)
- [x] 02-02-PLAN.md — Phase 2 script hardening (mutate PHPCS guard, generate with rejection examples + batch API, judge + judge_dataset with utils.py)
- [x] 02-03-PLAN.md — Phase 3 CoT hardening + export dataset update (40/60 ratio, metadata.json, dedup, PHP lint, sample_weight)
- [x] 02-04-PLAN.md — [GAP CLOSURE] Judge remaining 23 repos via Claude Code agents (auto-pass wordpress-develop core, judge 22 assessed repos with 5 parallel agents)
- [x] 02-05-PLAN.md — [GAP CLOSURE] Gap analysis + mutations (Python, no LLM) then synthetic generation via Claude Code agents (~500 rejection examples)
- [x] 02-06-PLAN.md — [GAP CLOSURE] Judge synthetics + generate judge training data via Claude Code agents (rubric-scored 0-100 examples)
- [x] 02-07-PLAN.md — [GAP CLOSURE] CoT reasoning via Claude Code agents + export dataset (Python) + human validation checkpoint

<!-- 02-04..07 executed 2026-03-29 via /run-data-pipeline; checkboxes ticked + re-verified 2026-06-26 (02-VERIFICATION.md status: passed). -->

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

### Phase 4: Base-Model Profiling & Evaluation (Triage)

**Goal**: First, profile the base model with all 5 ratio data distributions (~minutes) to determine whether 60/40 and 70/30 warrant training. Then eval existing adapters (30/70, 40/60, 50/50) through quality gates and wp-bench in parallel with any new training. Triage eliminates clearly failing ratios; survivors carried to Phase 7.
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, GATE-02
**Success Criteria** (what must be TRUE):

  1. Base-model profiling runs gradient-free forward passes with all 5 ratio data distributions, producing E_eff per layer for each — determines whether 60/40 and 70/30 training is warranted (E_eff trending down = train, flat/up = skip)
  2. If E_eff signal warrants, 60/40 (and optionally 70/30) training started in background while eval runs on existing 3 adapters
  3. All available ratio adapters evaluated: PHPCS pass rate, judge Spearman correlation, security pass rate
  4. At least one ratio exceeds all hard gates (PHPCS >95%, Spearman >0.85, Security >98%)
  5. wp-bench execution and knowledge tests run for all evaluated ratios with scores recorded
  6. Triage decision: ratios that fail hard gates or are >5pp behind the best are eliminated; all others survive to Phase 7 (high bar for elimination, low bar for continuation)
  7. Human has reviewed all eval results and E_eff profiling data, approved triage decisions
  8. eval_gen.py and eval_judge.py persist input prompt, raw model response, and extracted code in per-example JSONL — not just aggregate scores
  9. eval_gate.py per-dimension gates use correct field names matching eval script output (field name mismatch fix verified by unit test)

**Plans**: 3 plans

Plans:

- [x] 04-01-PLAN.md — Base-model E_eff profiling script + triage decision script (RoutingCollector hooks, E_eff computation, GATE-02 elimination logic with unit tests)
- [x] 04-02-PLAN.md — Eval orchestrator + DGX execution (clone wp-bench, create run_eval_triage.py, execute profiling + sequential adapter eval + wp-bench + triage)
- [x] 04-03-PLAN.md — Human review checkpoint (inspect profiling E_eff + eval results + wp-bench scores, approve triage survivors for Phase 7)

---

### v1.2 Judge Reasoning Fine-Tune — INSERTED

**Milestone Goal:** Fine-tune the winning ratio adapter on deep reasoning data so the judge articulates dimension-by-dimension analysis, score justification, and corrected versions — not just numeric rubric output.

**Dependency:** Phase 4 triage must complete first — all v1.2 phases start from the winning ratio adapter identified by Phase 4.

**Note for v2.0:** Even with the MoE router frozen during v1.2 training, routing profiles from the v1.0 adapter are invalidated by continued fine-tuning. Phase 7 must run a fresh profiling pass on the v1.2 reasoning adapter, not the v1.0 adapter.

### Phase 4.1: Reasoning Data Generation — INSERTED

**Goal**: Curate human-annotated seed examples, then use them as few-shot exemplars for Claude Code agents generating two parallel streams of reasoning training data — deep judge CoT examples (dimension-by-dimension analysis with WP-specific line citations) and critique-then-fix triples (structured critique with severity tags followed by corrected code)
**Depends on**: Phase 4 (winning ratio identified via triage decision)
**Requirements**: DGEN-01, DGEN-02, DGEN-03
**Success Criteria** (what must be TRUE):

  1. 50-100 human-annotated seed examples curated — focused on boundary cases (subtle defects, context-dependent issues) with dimension-specific contrastive reasoning. Seeds drawn from existing mutation pairs (phase2_mutate.py). These seeds serve triple duty: few-shot exemplars for agent generation, validated test set for Phase 4.4 eval, and threshold calibration anchors
  2. A pilot batch of 20-50 deep judge CoT examples and 20-50 critique-then-fix examples is generated using human seeds as few-shot and manually reviewed before bulk generation starts — pilot confirms WP-specific pattern citations (e.g., `$wpdb->prepare()`, `wp_verify_nonce()`, `esc_html()`) appear by name in reasoning chains and that dimension coverage spans all 9 rubric dimensions
  3. Bulk deep judge CoT agent generates reasoning-enriched examples where each response contains dimension-by-dimension analysis with line references, issue identification, fix suggestions, and structured scores — sourced from `data/phase1_extraction/output/{passed,failed}/`
  4. Bulk critique-then-fix agent generates examples from the existing mutation pool (`data/phase2_synthetic/output/mutated/`) where each triple contains the defective code, a structured critique with severity per dimension (critical/high/medium/low), and the corrected version in a clearly delimited `<corrected_code>` block
  5. Both generation streams reach their target example counts without >2% parse failure rate (measured by multi-strategy JSON extraction with hard rejection)

**Skill**: `wp-finetune:run-reasoning-data-gen` (NEW — optional, may be created if Phase 4.2 needs to re-run generation; 4.1-01/02/03 plans serve as pattern reference and define the agent spawning pattern)

  - **LLM execution: Claude Code agents ONLY** — parallel `Agent(run_in_background=true)` per batch following `wp-finetune:run-data-pipeline` pattern. NO Anthropic API. Pilot scripts (4.1-01/02) used direct API which works for 40 examples; bulk (4.1-03) uses agent spawning per the global LLM rule
  - DGX usage: NONE — pure CPU work, no GPU containers needed
  - Pattern: spawn N parallel agents → each reads input batch JSON + seeds + rubric → generates output batch JSON via in-context reasoning → merge step applies quality gates
  - Fix-test-validate loop: pilot → human review → bulk via agents → quality audit → if accepted count below target, spawn additional agent batches with incremented index
  - Quality gates run as a deterministic merge step (not in agents): citation accuracy verification, PHP lint via `subprocess.run(["php", "-l", ...])`, critique-fix alignment check, all 9 dimensions present

**Plans**: 3 plans

Plans:

- [x] 04.1-01-PLAN.md — Seed import + deep judge CoT generation script (seed data import, few-shot agent generation with 9-dimension quality gate)
- [x] 04.1-02-PLAN.md — Critique-then-fix generation script + pilot execution of both streams with human review gate
- [ ] 04.1-03-PLAN.md — Bulk generation of both streams via Claude Code agents (NOT API) after pilot approval

### Phase 4.2: Reasoning Dataset Assembly — INSERTED

**Goal**: Both generation streams are merged into a quality-validated training dataset with score consistency enforcement, canonical output template compliance, and the correct training mix (60% CoT + 25% CtF + 15% replay, D-05) — ready for continued fine-tuning
**Depends on**: Phase 4.1 (both generation streams complete)
**Requirements**: DGEN-04, DGEN-05
**Success Criteria** (what must be TRUE):

  1. Score consistency validation rejects any example where the written reasoning contradicts the numeric scores (e.g., reasoning describes a critical SQL injection vulnerability but the security dimension score is ≥7) — rejection rate and example count logged to metadata.json
  2. All retained reasoning examples conform to the canonical output template: dimension-by-dimension analysis prose followed by `[/REASONING]` separator followed by a JSON scores block inside `<judge_output>` tags — no example deviates from this structure
  3. The assembled training mix contains reasoning examples (CoT + CtF) plus replay at 60/25/15 ratio (CoT/CtF/replay per D-05) — actual counts and percentages recorded in metadata.json
  4. `data/reasoning_dataset/openai_train.jsonl` and `openai_val.jsonl` are exported with an 80/20 split (larger val slice than main dataset due to smaller total size)

**Skill**: `wp-finetune:run-reasoning-assembly` (created 2026-04-23)

  - **LLM execution**: Claude Code agents ONLY for any LLM-judged consistency checks. NO Anthropic API. Score consistency validation MAY be deterministic (regex/threshold rules: "if reasoning text contains 'critical' and security score >=7 → reject") OR may spawn Claude Code agents for nuanced cases (NOT API). Decision deferred to phase planning.
  - DGX usage: NONE — pure Python (json manipulation, regex, file I/O)
  - Pattern: read 4.1 batch outputs → run deterministic consistency rules → for ambiguous cases, spawn Claude Code agents to judge consistency → assemble training mix via Python → export multi-format
  - Fix-test-validate loop: dry-run consistency rules on 10 known-good + 10 known-bad pilot examples → tune thresholds → run on full dataset → human review of rejected examples → adjust if false-positive rate >5% → re-run
  - Quality audit: rejection rate per rule type, training mix percentage verification, output schema validation

**Plans**: 1/1 plans complete

### Phase 4.3: Reasoning Fine-Tune — INSERTED

**Goal**: The winning ratio adapter is continued-fine-tuned on the assembled reasoning dataset at a 5-10x lower learning rate than Phase 3, with MoE router weights confirmed frozen, producing a reasoning adapter that does not suffer format collapse, generation regression, or loss divergence
**Goal (RE-OPENED 2026-06-11 — corrective Tinker MoE-only retrain; supersedes the DGX-framed goal above)**: Produce a merge-ready grid-winner reasoning LoRA adapter whose MoE judge skill is less codegen-destructive when merged into the stock base — closing the REVL-04 wp-bench codegen gap (>= baseline 0.4537) WITHOUT losing judge skill (POINT Spearman >= v3 floor 0.263) and without regressing format stability (RTRN-05). Delivered via a rank {8,16,32} x replay {15,30,50%} MoE-only grid with a pre-registered selection rule and per-candidate eval economy; no candidate clearing the HARD gate => documented escalation, not auto-ship. NOTE: RTRN-01/02/03 above are local-DGX-framed and SUPERSEDED by the Tinker regime (LR 4.99e-4, cloud LoRA) per CONTEXT.md ROADMAP-drift; RTRN-05 still binds; RTRN-04 4-bit post-hoc gate is retired (invalid on quantized Qwen3-MoE).
**Depends on**: Phase 4.2 (reasoning dataset assembled and validated)
**Requirements**: RTRN-01, RTRN-02, RTRN-03, RTRN-04, RTRN-05
**Success Criteria** (what must be TRUE):

  1. `train_config_reasoning.yaml` specifies a learning rate at most 2e-5 (5-10x lower than Phase 3's 2e-4) with warmup, and the training run starts from the winning ratio adapter checkpoint — gradient norms in the first 100 steps stay below 3 (not the 5-10 seen in early Phase 3)
  2. `max_seq_length` is set to 8192 in the training config, and the training run processes examples longer than 4096 tokens without truncation errors or OOM
  3. MoE router layer weights are confirmed frozen in the Unsloth PEFT config before training begins — training log shows router parameters excluded from the optimizer parameter count
  4. Training completes 1-2 epochs on the combined reasoning dataset without OOM or loss divergence, and parse failure rate on checkpoint eval outputs stays below 5% throughout (abort condition if exceeded)

**Skill**: Reuse `wp-finetune:run-training` (reasoning-specific config)

  - **LLM execution**: No LLM API used — pure DGX training execution. The model itself is being trained on the reasoning dataset from Phase 4.2; no external LLM calls during the training loop.
  - DGX pre-flight: `dgx.validate(["toolbox", "config", "memory:70"])` + `dgx.ensure_ready("unsloth_studio")` — same pattern as Phase 3 training
  - Config: `train_config_reasoning.yaml` with LR <=2e-5, `max_seq_length: 8192`, `base_adapter: adapters/qwen3-30b-wp-{winning}/`
  - Router freeze verification: before training starts, confirm router params excluded from optimizer via `--dry-run` output inspection
  - Embeds `observe-training` telemetry agents inline (6-agent team) for gradient norm, loss, and router_aux_loss monitoring
  - Calls `wp-finetune:adaptive-planner` at Step 8.5 for thermal/power-based batch adjustment (8192-token sequences need careful memory management)
  - Fix-test-validate loop: dry-run first → if OOM on 8192 sequences, `adaptive-planner` reduces batch → re-dry-run → proceed when clean; if loss divergence during training, halt and present gradient norm history to user
  - Checkpoint eval loop: at each checkpoint, run `eval_judge.py` on 50 samples → if parse failure rate >5%, abort training early (RTRN-04 abort condition)
  - Idempotency: `idempotency_check="adapters/qwen3-30b-wp-{winning}-reasoning/adapter_config.json"`
  - Invokes `wp-finetune:review-telemetry` after training completes

**Plans**: 4 plans (corrective Tinker MoE-only retrain, re-planned 2026-06-11). Prior DGX/Unsloth plans 04.3-01/02 are OBVIATED by the Tinker pivot + RC-B attribution and archived under `_archive-unsloth-2026-06-11/` (kept for audit; do NOT execute).

Plans:

- [x] 04.2-01-PLAN.md

- [x] 04.3-01-PLAN.md — Enabling code (Wave 1): add `--train-attn`/`--train-unembed` MoE-only flags to `tinker_reasoning_sft.py` (D-N1); add `is_moe_only` detection + gated attention/unembed merge stages to `merge_tinker_v3.py`; add the `is_moe_only=True` merge-convention test
- [x] 04.3-02-PLAN.md — Replay variants (Wave 1): NEW `build_replay_mix.py` (pure wp_gen, leakage-guarded, D-N4); verify phase1 pool >=423; build the 30%/50% replay train variants (negatives preserved)
- [x] 04.3-03-PLAN.md — Grid driver (Wave 2): NEW `run_grid_eval.py` — cheap pre-merge judge filter -> only-if-pass merge -> REVL-04 wp-bench; bars POINT Spearman >= 0.263 (D-N7) + wp-bench >= 0.4537 HARD (D-N8); pre-registered selection rule; escalation exit 2 (no auto-ship)
- [x] 04.3-04-PLAN.md — Grid execution + selection (Wave 3): train the 9 MoE-only rank{8,16,32}xreplay{15,30,50%} candidates; assemble `grid_manifest.json`; run the grid; apply the pre-registered rule; confirm the winner (2nd 344-test wp-bench + post-merge judge); export merge-ready v4 adapter for Phase 04.4 — OR documented escalation

(Archived, OBVIATED — local DGX/Unsloth iteration, kept for audit only:)

- [x] 04.3-01-PLAN.md (archived) — Reasoning continued-FT of the merged 30_70 base (Option B)
- [x] 04.3-02-PLAN.md (archived) — RTRN-05 format-stability diagnostic bisect (ckpt-50 vs ckpt-72)

### Phase 4.4: Reasoning Eval & Adapter Merge — INSERTED

**Goal**: The reasoning adapter passes all existing quality gates (Spearman, PHPCS pass rate, wp-bench) with no regression versus the winning ratio baseline, human reviews a sample of reasoning outputs to confirm quality, and the adapter is merged into base weights
**Depends on**: Phase 4.3 (reasoning fine-tune complete)
**Requirements**: REVL-01, REVL-02, REVL-03, REVL-04, REVL-05, REVL-06, REVL-07, REVL-08
**Success Criteria** (what must be TRUE):

  1. `eval_judge.py` Spearman correlation on the reasoning adapter meets or exceeds the winning ratio baseline — absolute score distributions per dimension are compared and any dimension with mean shift >0.5 points vs baseline is flagged
  2. `eval_gen.py` PHPCS pass rate on the reasoning adapter is within 2pp of the winning ratio baseline — generation regression is not masked by improved judge metrics
  3. Reasoning quality evaluated by separately spawned Claude evaluator agent (independent context, opaque inputs only): dimension coverage rate, score-reasoning consistency rate, and coherence assessment on representative sample — recorded alongside Nemotron-free automated checks (regex dimension coverage, issue specificity)
  4. **[wp-bench HARD GATE]** wp-bench score on the reasoning adapter meets or exceeds the winning ratio baseline — this gate was deferred from Phase 4 triage (wp-bench was skipped there) and MUST execute here before adapter merge. Requires a different eval harness than Phase 4: serve the reasoning adapter as a merged model (not LoRA-on-base) and point wp-bench config at the merged checkpoint. Adapter merge is blocked until this gate passes.
  5. Human reviews a sample of reasoning outputs (deep judge CoT and critique-then-fix) and explicitly approves quality before the adapter merge runs — `models/qwen3-30b-wp-{winning}-reasoning-merged/` is written only after human sign-off
  6. Fix correctness: critique-then-fix corrected code passes PHPCS + security scanner, confirming fixes actually resolve identified issues — pass rate recorded
  7. Classification accuracy: confusion matrix (TP/TN/FP/FN) at score thresholds derived from eval_judge.py per-example data — precision, recall, F1 recorded per dimension
  8. Reasoning length distribution: median, p95, max token counts recorded and reviewed against expected range (flag if p95 > 6000 tokens or median < 500)

**Skill**: Reuse `wp-finetune:run-evaluation` (reasoning-specific eval + merge)

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. REVL-03 reasoning quality scoring (dimension coverage, score-reasoning consistency, coherence) spawns independent Claude Code evaluator agents with opaque inputs. REVL-06 fix correctness uses deterministic PHPCS + security scanner (no LLM). REVL-05 human review is manual (no LLM).
  - DGX execution: serve reasoning adapter as merged model via `dgx.execute("vllm", ...)` — NOT LoRA-on-base (wp-bench requires merged checkpoint)
  - Embeds `observe-evaluation` telemetry agents inline during eval runs
  - Sequential eval loop: eval_gen.py → eval_judge.py → eval_gate.py → wp-bench — each gate checked before proceeding to next
  - Fix-test-validate loop: if eval_gen PHPCS regresses >2pp → flag generation regression, present per-example failures for diagnosis; if Spearman drops → present dimension-level comparison for targeted investigation; if wp-bench fails → serve model differently (check tokenizer, check merge correctness) → re-eval
  - Claude evaluator agent: spawned independently (separate context, opaque inputs) for REVL-03 reasoning quality — dimension coverage + score-reasoning consistency measured
  - Fix correctness loop: run PHPCS + security scanner on critique-then-fix corrected code from eval samples → if pass rate below threshold, flag specific failure patterns
  - Reasoning length check: compute median/p95/max token counts → flag if outside expected range (p95 >6000 or median <500)
  - Human review checkpoint: present full eval comparison table (reasoning adapter vs winning ratio baseline) + reasoning output samples before gating merge
  - Adapter merge: after human approval, `dgx.execute("unsloth_studio", "python", "-m", "scripts.merge_adapter", ...)` with idempotency check on `models/qwen3-30b-wp-{winning}-reasoning-merged/`
  - Post-merge validation: load merged model, run 10 inference samples for both `<wp_gen>` and `<wp_judge>`, verify coherent output and correct task token routing
  - Invokes `wp-finetune:review-telemetry` for consolidated eval summary

**Plans**: ACTIVE = v4-winner post-merge re-gate (ITERATION 2, fresh 01-04 — see the ACTIVE block below). Prior tracks are DEAD/archived: v3 track (01-03) + merge-fix remediation (06-09) both failed REVL-04 → 04.3 retrained MoE-only → wp-reasoning-v4-winner; their plans+summaries moved to archive-stale-v4-nolmhead/. Earlier sets in archive-stale-v2-prereval/ + archive-stale-v3-lmhead/ (see CONTEXT ITERATION 2 + DISCUSSION-LOG)

Plans (v3 track — superseded by remediation):

- [x] 04.4-01-PLAN.md — [W1] merge_tinker_v3.py (Tinker per-expert MoE convention) + Wave-0 tests + CPU merge to v3 staging + 3 anchor gates (tensor/fp32-control/forward)
- [x] 04.4-02-PLAN.md — [W2] v3 vLLM serve + 3-layer merge-fidelity gate (L2 24-prompt invalid-PHP sentinel + L3 Spearman≥0.95 BLOCKING; L1 corroboration) → REVL-01/02 carry decision
- [✗] 04.4-03-PLAN.md — [W3] REVL-04 wp-bench HARD gate fresh on merged-served v3 — **FAILED** (reasoning 0.3716 < baseline 0.4537; 19% parse fails). Root cause: lm_head LoRA delta on extended-vocab base. Failure record kept.
- [~] 04.4-04/05-PLAN.md — ARCHIVED to archive-stale-v3-lmhead/ (v3-pinned, never executed; superseded by remediation track below)

Plans (merge-fix remediation track — exclude lm_head, attempt-1; gate order REVL-01A→REVL-04→REVL-05):

- [x] 04.4-06-PLAN.md — [W6] re-merge with manual lm_head stage DROPPED (--exclude-lm-head, q_proj kept) → new v4 candidate models/_staging/...-merged-v4-nolmhead + output/merge_v4_nolmhead/merge_report.json; 3 anchors re-certify
- [x] 04.4-07-PLAN.md — [W7] REVL-01A parse-failure census on merged-served v4 (≤5% progression gate, fresh) + judge Spearman + REVL-02 fresh PHPCS + REVL-03 thin + REVL-07/08 SOFT + REVL-06 N/A → 04.4-GATE-LEDGER-V4.md; emits parse_gate_pass
- [x] 04.4-08-PLAN.md — [W8] REVL-04 wp-bench HARD gate (autonomous, fail-fast): precondition early-exit on parse_gate_pass≤5% before the ~2.7h run; pass = reasoning≥baseline (~0.4537); fail-path note (attempt-2 q_proj per D-IT-05, then D-IT-02) — not a plan
- [ ] 04.4-09-PLAN.md — [W9] REVL-05 thin v4 spot-check (HUMAN_APPROVED_V4_POSTMERGE) + triple-gated idempotent promote v4→canonical + post-merge 10+10 validation → closes 4.4, unblocks Phase 7

Plans (v4-winner post-merge re-gate — ITERATION 2, ACTIVE 2026-06-13; merge-fix tracks above ARCHIVED to archive-stale-v4-nolmhead/ — both DEAD, failed REVL-04; 04.3 retrained MoE-only → wp-reasoning-v4-winner r32-rp30. Fresh plan set numbered 01-04):

- [x] 04.4-01-PLAN.md — [W1] Clean re-merge the MoE-only v4-winner into models/_staging/...-merged-v4 (is_moe_only path, D-V4-07) + re-certify 3 anchors + byte-identity check vs the grid 0.4603 staging → reuse_revl04 boolean (D-V4-02)
- [x] 04.4-02-PLAN.md — [W2] Single GPU serve of clean staging → capture judge-val + 24 sentinel + reasoning + gen + eval_gen(REVL-02) from the MERGED endpoint (D-V4-03); conditional REVL-04 re-bench only if reuse_revl04==false (D-V4-02/05)
- [x] 04.4-03-PLAN.md — [W3] Offline full 8-gate cascade from captures (D-V4-01): HARD REVL-01A≥0.263 (D-V4-04)/REVL-02 PHPCS within-2pp (re-measured)/REVL-04≥0.4537/sentinel 0/24/confusion Pareto + SOFT REVL-03/06/07/08 + per-dim guard flagging the knowledge dip (D-V4-06) → GATE-LEDGER-V4-WINNER + automated_pass
- [x] 04.4-04-PLAN.md — [W4] REVL-05 human review (gated on automated_pass, human-last D-V4-01) → triple-gated idempotent promote to canonical v4-suffixed dir (D-V4-08) + post-merge 10+10 validation → closes 4.4, unblocks Phase 7

---

### Phase 5: Packaging and Deployment (DEFERRED → v3.0 Phase 15)

**Goal**: Model is quantized, served on all DGX Toolbox endpoints, and published to HuggingFace Hub
**Depends on**: Deferred — all DPLT requirements subsumed by v3.0 PKG/PRUNE phases (Phase 13-15)
**Requirements**: DPLT-01, DPLT-02, DPLT-03, DPLT-04, DPLT-05, DPLT-06, DPLT-07
**Success Criteria** (what must be TRUE):

  1. LoRA adapter merged into base model (or served via --lora-modules)
  2. AWQ 4-bit quantization produced for vLLM production serving (~8GB)
  3. GGUF quantization produced for Ollama local serving (~9GB)
  4. Model responds at vLLM (:8020), Ollama (:11434), LiteLLM (:4000), Open-WebUI (:12000)
  5. HuggingFace Hub page has model card with eval metrics (including wp-bench scores), quantized download links, and usage examples

**Plans**: 1 plan

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
**Depends on**: Phase 3 (training script exists); dgx-toolbox Phase 13 (telemetry/ package). Phase 5 was the original dependency but is now deferred to v3.0 — Phase 6 is independent of Phase 5.
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

### v2.0 RL Alignment

**Milestone Goal:** Profile routing to identify the protected expert set, build reward infrastructure with anti-hack eval, run GSPO on the FULL MoE (not sieve-constrained), and evaluate RL output against v1.2 SFT baseline. RL runs before MoE-Sieve per Issue #1: routing statistics should reflect reward-aligned behavior, not SFT pre-training usage. GSPO (sequence-level) is the primary RL objective for MoE stability (D-08). Whether to also evaluate GRPO as an alternative is an optional decision deferred to Phase 9 planning time.

**Dependency:** Phase 4.4 (v1.2 complete — reasoning adapter merged) must complete before Phase 7. Phase 7 profiles the v1.2 reasoning adapter, not the v1.0 adapter. Phase 10 gates Phase 11 (MoE-Sieve).

### Phase 7: Router Profiling & Protected Expert Set

**Goal**: Profile surviving ratio ADAPTERS (not base model — that was Phase 4 step 1) to capture how fine-tuning shifted routing, producing per-task expert affinity maps with E_eff metrics. Identify dual-purpose experts (active for both gen and judge) that must not be pruned in any subsequent phase (D-10). Combined with Phase 4 eval scores to select the optimal ratio for single-track RL training.
**Depends on**: Phase 4.4 (v1.2 reasoning adapter complete); Phase 6 (adaptive training infrastructure)
**Requirements**: PROF-01, PROF-02, PROF-03, PROF-04, PROF-05, GATE-01
**Success Criteria** (what must be TRUE):

  1. Profiling runs on each surviving ratio's fine-tuned adapter (not base model) hooking `Qwen3MoeSparseMoeBlock` gating outputs — captures how LoRA fine-tuning shifted routing relative to base-model profiling from Phase 4
  2. Routing tables report separate expert activation counts for `<wp_gen>` and `<wp_judge>` per ratio
  3. Profiling on 10% subsample achieves Jaccard similarity >=0.94 against full-set ranking per ratio
  4. Concentration report per ratio: per-layer CV, cumulative coverage curves, layer-depth skew, E_eff per layer with mean/max/variance — compared against Phase 4 base-model E_eff to quantify fine-tuning routing shift
  5. Decision matrix combining Phase 4 eval score and Phase 7 adapter E_eff selects the ratio with lowest E_eff at equivalent quality (within 2pp) — single ratio chosen for all subsequent work
  6. Protected expert set identified: experts with significant activation for BOTH gen and judge tasks are flagged as dual-purpose and must be retained through all subsequent phases (MoE-Sieve, pruning). Protected set exported as a per-layer mask for downstream consumption

**Skill**: `wp-finetune:run-profiling` (NEW — create during phase planning)

  - **LLM execution**: No LLM API used — gradient-free profiling runs forward passes of the ratio adapters on DGX, hooking `Qwen3MoeSparseMoeBlock` gating outputs. No external LLM calls; the model under profiling is the only model invoked (and only for local forward-pass routing capture, not generation judging).
  - Extends `run-evaluation` pattern: `dgx.execute("eval_toolbox", ...)` for GPU-bound profiling
  - Embeds `observe-evaluation` telemetry agents inline during profiling runs
  - Idempotency: `idempotency_check="output/profiling/{ratio}/routing_report.json"`
  - Execution test loop: after each ratio profile, validate Jaccard >=0.94 against full-set; if fail → re-profile with larger subsample and re-test
  - Human review checkpoint: present E_eff comparison table + protected expert set before ratio selection
  - **NOTE (CONTEXT D-01):** SC5 multi-ratio decision matrix DROPPED — Phase-4 triage gave NO_SURVIVORS except 30/70; single survivor already merged/promoted. PROF-05 + GATE-01 are N/A-with-rationale. Stimulus = matched training data (`data/final_dataset/ratio_30_70/openai_train.jsonl`) per amended D-05.

**Plans**: 2 plans

Plans:

- [x] 07-01-PLAN.md — Profiling code + tests + run-profiling skill (merged-model profiler, Jaccard PROF-03, concentration PROF-04, protected mask D-03/D-04, bootstrap CI D-09) — autonomous, GPU-free
- [x] 07-02-PLAN.md — DGX profiling run + post-processing + PROF-05/GATE-01 N/A rationale + human sign-off checkpoint (COMPLETE 2026-06-19 — gates green, council-reviewed APPROVED, Phase 7 closed)

### Phase 8: Reward Infrastructure

**Goal**: A composite reward pipeline is built and validated end-to-end before any RL training begins — PHPCS anchor, security hard gate, VeRPO partial credit, MO-GRPO normalization, and anti-hack eval set all verified independently
**Depends on**: Phase 7 (ratio selected, protected expert set identified)
**Requirements**: GRPO-01, GRPO-02, GRPO-03, GRPO-04
**Success Criteria** (what must be TRUE):

  1. The composite reward pipeline produces a scalar reward for any generation: 70% from verifiable signals (PHPCS pass rate, security scan, WordPress standards checks) and 30% from frozen wp_judge score
  2. A generation that fails the security scan receives total reward = 0 regardless of all other signal scores — verified by a test case where a secure-failing but otherwise high-quality generation scores zero
  3. All reward signals pass through MO-GRPO normalization — each signal is normalized by within-group variance before combination, and a single dominant signal cannot inflate total reward
  4. WordPress standards checks use VeRPO partial credit — each check is weighted by difficulty estimated from pass rate across group samples, and rarely-passed checks contribute more signal than common ones
  5. Anti-hack eval set constructed and validated (D-11) — penalizes verbosity reward hacking, template critique collapse, and self-preference bias; eval set used as a regression check during RL training

**Skill**: No new skill — reward pipeline is a Python module (`scripts/reward_pipeline.py`) with pytest test suite

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. The reward pipeline itself uses deterministic signals (PHPCS, security scanner, VeRPO, MO-GRPO normalization) plus the frozen local `wp_judge` model for the 30% judge component — no external LLM during reward computation. Anti-hack eval set construction (D-11 adversarial examples: verbose padding, template critiques, self-preferencing) uses Claude Code agents to score candidate adversarial cases during set construction.
  - Fix-test-validate loop: each reward component (PHPCS, security, VeRPO, MO-GRPO norm) is built, unit-tested, and validated independently before integration
  - Integration test: end-to-end reward computation on 50 held-out gen+judge examples with known-good and known-bad cases
  - Anti-hack eval set validated: run reward pipeline on adversarial examples (verbose padding, template critiques, self-preferencing) — all must score below threshold

**Plans**: 4/4 plans complete

Plans:

- [x] 08-01-PLAN.md — Foundation: Wave-0 test scaffolding + judge_score_single() RC-A wrapper + injectable recalibration-offset loader (GRPO-01)
- [x] 08-02-PLAN.md — Reward math core: dataclasses + MO-GRPO within-group normalization + VeRPO difficulty weighting on WP-standards subset (GRPO-03, GRPO-04)
- [x] 08-03-PLAN.md — Composite 70/30 assembly + security TERMINAL hard gate (fail-closed) + judge-imputation + 50-case integration incl. SC2 (GRPO-01, GRPO-02)
- [x] 08-04-PLAN.md — Anti-hack set: 3-axis perturb-real + background-agent scoring + CI-aware gate (hi_perturbed < lo_clean) + acceptance report (D-11)

### Phase 8.1: Reward Redesign — INSERTED

**Goal**: The composite reward produces a graded, per-group-discriminating signal so a GSPO rerun gets a non-zero advantage and an actual learning gradient — replacing the effectively-binary 0.7 `fix_correctness` term that collapsed Phase 9's optimization.
**Depends on**: Phase 8 (reward pipeline built + verified), Phase 9 first run (flat-reward evidence: training reward flat ~0.27 over 250 steps; RLEV-01 teacher-Spearman FLAT, no checkpoint beyond noise vs warmstart).
**Trigger / root cause**: `fix_correctness` (0.7 weight) is effectively binary (Panickssery divergent-rollout frac<0.1=0.53, frac>0.9=0.47, **frac_mid=0.00** — a step function). 4 samples of a "fixable" prompt all ~1, "unfixable" all ~0 → uniform GSPO groups → normalized advantage ~0 → vanishing gradient. Graded consistency (0.3) too small to drive learning. Temperature=1.0/group_size=4 → exploration fine (low-temp hypothesis refuted).
**Success Criteria** (what must be TRUE):

  1. `fix_correctness` returns graded partial credit (fraction of weighted PHPCS/security/syntax sub-checks passed), not pass/fail — verified by a probe showing frac_mid > 0 on the divergent-rollout set.
  2. On a held-out rollout-group probe, within-group reward variance is non-zero for a meaningful fraction of groups (groups no longer collapse to uniform) — i.e. `frac_groups_all_zero`/`frac_groups_uniform` measurably below the pre-redesign baseline.
  3. Diagnostic logging added per `09-RL-LOGGING-REQS.md`: per-component reward means, `frac_groups_all_zero`, and group-reward entropy emitted every step to the metrics JSONL.
  4. Reward weights rebalanced and/or per-group diversity/advantage-floor shaping applied so the saturated term no longer sets a prompt-mix-determined flat mean (validated against the new per-group logging).
  5. Offline reward-pipeline tests still pass (security hard gate, MO-GRPO normalization, anti-hack set thresholds) — the redesign does not regress Phase 8's verified guarantees.

**Skill**: No new skill — edits to `scripts/reward_pipeline.py` / `scripts/rl_judge_dispatch.py` + pytest; consumes `09-RL-LOGGING-REQS.md` (logging spec) and `09-LOCAL-RL-HANDOFF.md` §5/§7 + status doc Section H (evidence).
**Gates**: a targeted Phase 9 RERUN (fresh warm-start from v4, NO resume) only after the new per-group logging confirms a restored gradient on a short signal-check run.
**Plans**: 4/4 plans complete

- [x] 08.1-01-PLAN.md — Wave 1: MEASURE-FIRST (D-81-01). Extend `_probe_rl_reward.py` to histogram `rubric.overall` on the parseable subset + parse-fail rate + per-group stats (frac_groups_all_zero) for BOTH gen + judge paths (D-81-04); emit `08.1-MEASUREMENT.md` selecting the lever from the measured distribution (D-81-02). [SC1, SC2]
- [x] 08.1-02-PLAN.md — Wave 1: Logging core (D-81-03 P1-3). Pre-drop per-group stats into `compute_rollout_advantages` meta (before `remove_constant_reward_groups`) + extend `_log_step` with component means / frac_groups_all_zero / entropy. [SC3]
- [x] 08.1-03-PLAN.md — Wave 2: Reward-shape fix on the saturated path (selected lever, D-81-02/04) + full logging spec tiers 4-6 (histograms, window means, `should_flag_for_review`). Security gate untouched. [SC1, SC2, SC4, SC3]
- [x] 08.1-04-PLAN.md — Wave 3: OFFLINE signal-check gate (frac_groups_all_zero < baseline, frac_mid > 0 on 100+ completions, BEFORE any GPU) + Phase 8 regression suite (no regress). GPU 50-step signal-check stays Phase 9. [SC1, SC2, SC4, SC5] (completed 2026-06-24)

### Phase 8.2: Reward Validity Gate — INSERTED

**Goal**: The judge-axis RL reward provably TRACKS the validated downstream target (judge teacher-Spearman) — established by an offline reward-validity oracle BEFORE any GPU spend — so optimizing the reward actually moves the target instead of Goodharting. 08.1 guaranteed the reward had a *gradient*; 08.2 guarantees the gradient points at the *right thing*, plus a codegen-regression guardrail.
**Depends on**: Phase 8.1 (reward-shape graded + per-group logging), Phase 9 (GSPO run mechanics), Phase 10 RLEV verdict (seedA evidence: proxy moved, target flat, codegen regressed).
**Trigger / root cause**: Phase 10 RLEV proved seedA Goodharted. The offline oracle (`scripts/_reward_validity_oracle.py`, built 2026-07-01) quantifies it: across 11 checkpoints, the OPTIMIZED proxy `fix_correctness` has Spearman −0.24 (CI [−0.87,+0.42], includes 0) vs the teacher-Spearman target — optimizing it could not move the target. `pairwise_rank_agreement` vs teacher GT scores +0.70 (CI lower +0.15>0) — a dense, per-completion-decomposable VALID reward. 08.1 never tested validity, only shape.
**Success Criteria** (what must be TRUE):

  1. An offline reward-validity oracle exists + is the standing gate: a candidate reward's per-checkpoint trajectory must rank-correlate with teacher-Spearman (bootstrap CI lower > 0) before it may go to GPU. [DONE: `scripts/_reward_validity_oracle.py` + `output/reward_validity/ORACLE_FINDING.md`]
  2. The judge-axis reward includes a per-group pairwise rank-agreement-vs-teacher-GT calibration term (validity-gated form), wired in `scripts/reward_pipeline.py` + `scripts/rl_rollouts.py`, replacing/augmenting `_fix_score_from_completion`; offline reward tests still pass (security gate, MO-GRPO, anti-hack — no Phase-8 regression).
  3. An in-run codegen trip-wire guards generation: a periodic wp-bench probe (reuse `scripts/_rlev01_wpbench_ckpt.py`) early-stops the run if codegen drops below the v1.2 SFT bar (0.4616).
  4. An offline reward weight/form sweep on `data/rl_probe/judge_probe_corpus.jsonl`, scored by the oracle, selects the highest reward↔target-correlation form with no codegen penalty.
  5. A 50/250-step 2-seed RL smoke is PLANNED and gated on the VALIDATED metric (within-run paired teacher-Spearman trend + codegen trip-wire + echo-adversary ≤0.30), kill-at-50 if the validated metric isn't moving — NOT executed in this phase (stops before GPU/Tinker spend; execution is a gated Phase 9 rerun).

**Skill**: No new skill — `scripts/_reward_validity_oracle.py` (built), `scripts/_rlev01_wpbench_ckpt.py` (built, reused), edits to `scripts/reward_pipeline.py` / `scripts/rl_rollouts.py` + pytest. Consumes Phase 10 `RLEV_FINAL_REPORT.md` + `ORACLE_FINDING.md`.
**Gates**: any future Phase 9 RL rerun — no rerun until SC1–SC4 hold and SC5's smoke passes the validated-metric gate.
**Plans**: 5 plans (planned 2026-07-01; 4 waves; RVAL-01..05 map to SC1..SC5; all offline/CPU — HARD boundary: no GPU/Tinker spend, SC5 smoke is spec-only)

- [x] 08.2-01-PLAN.md — [W1] Formalize oracle→standing gate (run_validity_gate + regression test + GATE-RULE.md) + the TRAIN-teacher-GT wiring precondition (content-hash sidecar; val held out) [RVAL-01]
- [x] 08.2-02-PLAN.md — [W1] In-run codegen trip-wire (reuse _rlev01_wpbench_ckpt) wired to rl_train checkpoint cadence; halt below 0.4616 via existing seam; no-GPU dry-run test [RVAL-03]
- [x] 08.2-03-PLAN.md — [W2] Per-group pairwise-rank-agreement-vs-TRAIN-teacher calibration term (parameterized form+weight) in reward_pipeline+rl_rollouts; registered in oracle FORMS → VALID; no Phase-8 regression [RVAL-02]
- [x] 08.2-04-PLAN.md — [W3] Offline (form,weight) sweep on the GT-attached probe corpus; DUAL-LENS select (oracle CI-lower>0 + 08.1 gradient density + echo≤0.30); ranked table + selected config [RVAL-04]
- [x] 08.2-05-PLAN.md — [W4] Gated 50/250-step 2-seed smoke SPEC + guarded launcher (validated-metric gates + kill-at-50); execution OUT-OF-SCOPE — dry-print only, no GPU spend [RVAL-05]

### Phase 9: GSPO Training

**Goal**: Dual-mode RL refines both generation quality and judge reasoning quality on the FULL MoE (not sieve-constrained), with router-shift stabilization. GSPO (sequence-level importance sampling via `tc.forward_backward_custom`) is the PRIMARY RL objective per locked D-09-03; GRPO (token-level, `tc.forward_backward`) is a documented fallback only if GSPO proves unstable (select via `--grpo-fallback` / `--no-gspo`). Judge is the primary bottleneck (Spearman 0.57 vs gen 0.99+ at SFT stage) and receives equal or greater RL budget. Gen rewards use PHPCS + security + VeRPO. Judge rewards use score-reasoning consistency (separately spawned Claude evaluator agent) and fix correctness (PHPCS/security scanner on critique-then-fix corrected code). Protected experts from Phase 7 monitored per step via native `ForwardBackwardOutput.metrics` MoE routing keys (monitor-only per D-09-02; no active regularizer injection).
**Depends on**: Phase 8
**Requirements**: GRPO-05, GRPO-06, GRPO-07, GRPO-08
**Success Criteria** (what must be TRUE):

  1. RL training applies gradients to both `<wp_gen>` and `<wp_judge>` task pathways — gen uses verifiable code quality rewards, judge uses reasoning consistency + fix correctness rewards
  2. RL gradients flow to all routed experts and shared experts — MoE-only RL (attn + unembed FROZEN per D-09-08, superseding D-09-02's `train_attn/unembed=True`; D-IT showed attn deltas net-harmful to codegen and judge skill is MoE-borne), not hot-only (sieve comes after RL). Router gates are FROZEN (D-09-02: no `train_router` in LoraConfig). Protected expert set from Phase 7 monitored via per-step `e_frac_with_tokens:mean` and `e_max_violation` from `ForwardBackwardOutput.metrics` (monitor-only; no active regularizer injection)
  3. Router-shift ratio is computed between rollout and training phases, applied as stop-gradient floor multiplied into the clipped importance ratio before aggregation, and logged per step
  4. Training halts automatically if router-shift ratio exceeds the stability threshold — the halt is triggered by per-step monitoring, not a post-hoc check

**Skill**: `wp-finetune:run-rl-training` (NEW — create during phase planning)

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. GRPO-05 judge rewards use "score-reasoning consistency (separately spawned Claude evaluator agent)" during rollout scoring — these MUST be Claude Code agents. Gen rewards use deterministic PHPCS + security + VeRPO (no LLM). The training loop runs on **Tinker cloud** (D-09-01 locked; GB10 cannot host Qwen3-30B-A3B bf16 for RL); external judge-consistency scoring dispatches to Claude Code agents in parallel batches between rollout and gradient steps.
  - **Execution venue**: Tinker cloud exclusively — warm-started from the v1.2 SFT (v4-winner) `save_state` via `create_training_client_from_state(...)`, which inherits the checkpoint's MoE-only flags (`train_mlp=True`, `train_attn=False`, `train_unembed=False` per D-09-08; NO `train_router` per D-09-02). Cold-start `create_lora_training_client(...)` is invalid for this run — a raw-base LoRA fails RLEV-01 by construction (see 09-RL-INIT-RECONCILIATION.md)
  - **GSPO primary**: per-step loop calls `tc.forward_backward_custom(data, gspo_loss_fn, loss_type_input="logprobs")` with RSPO stop-gradient floor (`seq_ratio.clamp(min=1.0)`) — GRPO fallback via `tc.forward_backward(data, loss_fn="importance_sampling")` selected only with `--grpo-fallback` (D-09-03)
  - **Per-step autohalt guards**: KL `kl_sample_train_v1` soft alert ≥ 0.1 / hard halt ≥ 0.3; MoE routing `e_frac_with_tokens:mean` soft alert < 0.7 / hard halt < 0.5 — halt signals raised synchronously from `check_halt()` before next gradient step
  - **Protected expert monitoring**: per-step Jaccard score logged from `ForwardBackwardOutput.metrics` (`e_frac_with_tokens:mean`, `e_max_violation:mean/max`) against Phase 7 mask at `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` — monitor-only, no regularizer injection (D-09-02)
  - Fix-test-validate loop: dry-run first (`--dry-run`), then real training; autohalt guards detect instability and surface halt reason to user before rollback
  - Anti-hack regression: run anti-hack eval set after training completes; if regression detected, flag for human review before proceeding
  - Invokes `wp-finetune:review-telemetry` after training completes for consolidated summary

**Plans**: 6 plans

  - [x] 09-01-PLAN.md — RL prompt corpus assembly (audited, val-clean) + Tinker prompt-only data adapter (GRPO-05)
  - [x] 09-02-PLAN.md — RL test contract (8 named stubs) + mock_tinker_client fixture + ROADMAP DGX→Tinker skill-text correction (GRPO-05/06/07/08)
  - [x] 09-03-PLAN.md — Claude score-reasoning consistency scorer: async batch dispatch + content-hash cache + 120s timeout/impute (GRPO-05)
  - [x] 09-04-PLAN.md — Interleaved rollouts + dual rewards (Phase 8 pipeline unmodified) + capped judge combination + cookbook advantages (GRPO-05)
  - [x] 09-05-PLAN.md — Tinker RL loop: frozen-router LoRA, GRPO/GSPO switchable loss + RSPO floor, per-step KL/MoE auto-halt, monitor-only Jaccard, persistent checkpoints (GRPO-06/07/08)
  - [x] 09-06-PLAN.md — New Tinker-native wp-finetune:run-rl-training skill (zero DGX; deviations documented; anti-hack regression gate) (GRPO-05/06/07/08)

### Phase 10: RL Comparative Evaluation

**Goal**: The RL model is compared against the v1.2 SFT baseline on all quality dimensions, confirming RL improved judge reasoning (the primary target) without regressing generation quality — gates v3.0 MoE-Sieve
**Depends on**: Phase 9
**Requirements**: RLEV-01, RLEV-02
**Success Criteria** (what must be TRUE):

  1. **[wp-bench HARD GATE]** The RL model is evaluated against the v1.2 SFT baseline on wp-bench and all 9 eval dimensions — no dimension regression permitted; judge Spearman improvement expected (primary RL target). wp-bench is a hard gate — RL model must meet or exceed v1.2 SFT baseline wp-bench score before Phase 11 begins.
  2. RL evaluation report includes reward metric convergence curves, router-shift stability log (per-step shift ratios), protected expert retention rate vs Phase 7 baseline, gen/judge quality delta, and anti-hack eval results — sufficient to confirm RL added value before proceeding to MoE-Sieve

**Skill**: Reuse `wp-finetune:run-evaluation` (extend with RL-specific metrics)

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. Any LLM-judged eval dimensions (reasoning quality, score-reasoning consistency, anti-hack adversarial scoring for RLEV-02 anti-hack eval results) spawn independent Claude Code evaluator agents. Deterministic evals (PHPCS, security, wp-bench) run on DGX without LLM calls.
  - Extends existing eval skill with: router-shift stability report, protected expert retention comparison, anti-hack eval pass rates
  - DGX execution: `dgx.execute("eval_toolbox", ...)` for serving RL model + running eval suite
  - Embeds `observe-evaluation` telemetry agents inline during eval runs
  - Fix-test-validate loop: if any eval dimension regresses, present specific failure to user with suggested fix (re-train with adjusted regularizer, adjust reward weights) before declaring gate pass/fail
  - Human review checkpoint: present full comparison table (v1.2 SFT vs RL) before gating v3.0

**Plans**: 1 plan

  - [ ] 10-01-PLAN.md — RL-vs-v1.2 comparative eval: CI-aware bootstrap_gate + RLEV-02 five-part conjunctive gate (W0 build/test) → merge+serve+eval both checkpoints + live v1.2 anti-hack baseline (W1) → winner select + human v3.0 gate (W2) (RLEV-01, RLEV-02)

---

### v3.0 MoE-Sieve, Pruning & Packaging

**Milestone Goal:** Apply MoE-Sieve on the RL-trained model using RL-policy routing logs, then merge LoRA, prune with AIMER (primary, D-09) or REAP (optional comparison) on the final routing distribution, evaluate, and package for production. MoE-Sieve operates post-RL so that sieve selection reflects reward-aligned routing.

**Dependencies:** Phase 10 (RL eval results) must complete before Phase 11. LoRA merge (Phase 13 MERGE-01) must complete before pruning runs — activation magnitudes require the unified model.

**Note:** MoE-Sieve in v3.0 operates on the RL-trained model using RL-policy routing logs (not SFT logs). A fresh profiling pass is required before sieve selection — the Phase 7 SFT-era profiles are used only for protected expert identification and pre-RL baseline comparison.

### Phase 11: Post-RL MoE-Sieve

**Goal**: Re-profile routing using RL-policy logs, then apply MoE-Sieve selective training on the RL-trained model with conservative threshold, validating that protected experts from Phase 7 are retained. Optional recovery SFT pass if sieve causes regression.
**Depends on**: Phase 10 (RL eval confirms readiness for sieve)
**Requirements**: SIEVE-01, SIEVE-02, SIEVE-03, SIEVE-04, SIEVE-05
**Success Criteria** (what must be TRUE):

  1. Fresh routing profiling on RL-trained model produces updated hot/cold expert classification using RL-policy routing logs (not SFT-era logs) — the training run applies LoRA adapters to hot routed experts, all attention (Q/K/V/O), router gates, and 4 shared experts; cold routed experts frozen with no gradient flow
  2. Gen-hot experts (per RL-policy routing) receive only golden signal data (passed examples, synthetic good) while judge-hot experts receive the full spectrum (passed + failed + contrastive), verifiable by inspecting data routing assignments per expert group
  3. The training uses the best gen/judge ratio identified by Phase 4 eval, not a hardcoded ratio
  4. Three k-sweep runs complete at budgets of approximately 13, 32, and 64 active experts per layer, each producing a separate adapter checkpoint
  5. The optimal k is declared as the smallest budget where wp-bench score falls within +/-1pp of full-LoRA, verified by TOST equivalence test (epsilon=2pp) across 3+ seeds; all protected experts from Phase 7 must be in the retained set at the optimal k

**Skill**: `wp-finetune:run-sieve-training` (NEW — create during phase planning)

  - **LLM execution**: No LLM API used — MoE-Sieve selective training runs on DGX with the RL-trained model. Re-profiling uses gradient-free forward passes (no external LLM). Training applies LoRA adapters to hot routed experts with no external LLM calls. Any inline wp-bench scoring during k-sweeps is deterministic (no LLM judge).
  - Extends `run-training` pattern: per-k-budget loop with `dgx.execute("unsloth_studio", ...)` for DGX Spark execution
  - Step 0: Re-profile routing using `wp-finetune:run-profiling` on the RL-trained model (RL-policy routing logs, not SFT)
  - DGX validation: `dgx.validate(["toolbox", "config", "memory:70"])` + `dgx.ensure_ready("unsloth_studio")`
  - Embeds `observe-training` telemetry agents inline per k-sweep run
  - Calls `wp-finetune:adaptive-planner` between k-sweep runs for thermal/power config adjustment
  - K-sweep loop: for each k in [13, 32, 64], train sieve adapter → run wp-bench inline → compare against full-LoRA baseline
  - Fix-test-validate loop per k: if training OOMs → `adaptive-planner` adjusts batch → retry; if wp-bench regresses → log and continue to next k
  - Protected expert retention check: after each k-sweep, verify all Phase 7 protected experts are in retained set → if not, adjust k threshold and re-run
  - Idempotency: `idempotency_check="adapters/qwen3-30b-wp-sieve-k{k}/adapter_config.json"` per k-budget
  - Invokes `wp-finetune:review-telemetry` after all k-sweeps complete
  - Human review checkpoint: present k-sweep comparison table before declaring optimal k

**Note (planned 2026-07-08, TRAINING-FREE scope per CONTEXT lock):** The literal retraining spec above
is superseded — Phase 11 is a training-free routing-analysis + inference-time expert-masking k-sweep on
the shipped two-model pair (v1.2 gen + v1.3 3-seed judge ensemble). No LoRA retraining. SIEVE-01/04/05
reinterpreted as profile/mask/TOST deliverables; SIEVE-02 N/A, SIEVE-03 traceability. See 11-CONTEXT.md.

**Plans**: 5/5 plans complete

- [x] 11-01-PLAN.md — Wave-0 scaffolding: env pre-check + SIEVE-01/04/05 test contracts (wave 0)
- [x] 11-02-PLAN.md — Export + merge s0/s2 judge seeds into 13-shard checkpoints (wave 1)
- [x] 11-03-PLAN.md — Profile 3 judge seeds + cross-seed overlap + protected-mask subset verify (wave 2)
- [x] 11-04-PLAN.md — Inference-time expert-masking k-sweep (13/32/64) → wp-bench + judge rho (wave 3)
- [x] 11-05-PLAN.md — TOST optimal-k + human sign-off + Phase-13 prune-set + SIEVE-02/03 docs (wave 4)

### Phase 12: MoE-Sieve Comparative Evaluation

**Goal**: Each k-sweep MoE-Sieve adapter is A/B compared against v2.0 RL baseline on all 9 eval dimensions, producing the dimension-level report and seed variance analysis that gates v3.0 pruning
**Depends on**: Phase 11
**Requirements**: EVAL2-01, EVAL2-02
**Success Criteria** (what must be TRUE):

  1. **[wp-bench HARD GATE]** An A/B eval runs each k-sweep MoE-Sieve adapter (all three k budgets) against v2.0 RL baseline on wp-bench and the static eval suite, with results recorded per adapter. wp-bench is a hard gate — each k-sweep adapter must be evaluated on wp-bench; any adapter that regresses below the v2.0 RL baseline wp-bench score is eliminated. Note: this phase requires a different eval harness than Phase 4 (adapters served as merged models; wp-bench config must target the merged checkpoint for each k-sweep variant).
  2. The report covers all 9 eval dimensions per adapter, overall scores, inference speed delta, and seed variance — sufficient to identify the optimal k and confirm MoE-Sieve quality before proceeding to pruning

**Skill**: Reuse `wp-finetune:run-evaluation` (extend with sieve-specific A/B comparison)

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. Any LLM-judged eval dimensions (reasoning quality, score-reasoning consistency) across the k-sweep adapters spawn independent Claude Code evaluator agents. Deterministic evals (wp-bench, PHPCS, security, TOST equivalence) run on DGX without LLM calls.
  - Per-k-adapter eval loop: for each k-sweep adapter, serve as merged model via `dgx.execute("vllm", ...)` → run full eval suite + wp-bench → record results
  - Embeds `observe-evaluation` telemetry agents inline during eval
  - Fix-test-validate loop: if eval harness fails (model serving error, wp-bench timeout) → fix serving config → re-run eval for that adapter
  - TOST equivalence test automated: `eval_gate.py --tost --epsilon 2pp --seeds 3` across all k variants
  - Human review checkpoint: present full A/B comparison table before gating pruning phase

**Plans**: 1 plan

### Phase 13: LoRA Merge & Pruning (AIMER primary, REAP optional)

**Goal**: Merge LoRA adapters into base weights, then run AIMER (task-agnostic, weight-based, primary per D-09) and optionally REAP (domain-aware, calibration-based) at three compression ratios to determine whether WordPress domain specialization creates enough routing concentration for calibration-based pruning to outperform generalized weight-based pruning
**Depends on**: Phase 12
**Requirements**: MERGE-01, PRUNE-01, PRUNE-02, PRUNE-03, PRUNE-04, PRUNE-05, PRUNE-06
**Success Criteria** (what must be TRUE):

  1. All LoRA adapters (MoE-Sieve + RL) are merged into base model weights — merged checkpoint produces identical outputs to adapter-on-base configuration
  2. AIMER runs on merged model at 25%, 50%, 75% compression (~1 second per ratio, no calibration needed) producing 3 pruning masks as task-agnostic baseline (primary method per D-09)
  3. REAP optionally runs on same merged model with WordPress calibration data at same 25%, 50%, 75% compression producing 3 domain-aware pruning masks (comparison experiment)
  4. All variants evaluated via gating mask across all 9 dimensions before any weight removal — comparison table visible before committing
  5. Domain specificity analysis: expert overlap between AIMER and REAP retention sets quantified per layer — high overlap = WordPress isn't specialized enough for calibration advantage; low overlap = REAP captures domain routing AIMER misses
  6. Winning method + ratio selected by dimension-level retention (especially D2_security), preferring higher compression at equivalent quality; final model physically pruned with router re-normalization

**Skill**: `wp-finetune:run-pruning` (NEW — create during phase planning)

  - **LLM execution**: No LLM API used — AIMER is weight-based (task-agnostic, no calibration) and REAP is calibration-based using WordPress calibration data (activation statistics, not LLM judging). Both methods run on the merged model via DGX forward passes; no external LLM calls. Eval across all 9 dimensions in this phase uses gating-mask evaluation (not LLM-judged scoring).
  - Step 1: Merge LoRA via `dgx.execute("unsloth_studio", "python", "-m", "scripts.merge_adapter", ...)` with idempotency check on merged checkpoint
  - Step 2: Merge validation — load merged model, run 10 inference samples, compare outputs against adapter-on-base (exact match)
  - Step 3: AIMER loop: for each ratio in [25%, 50%, 75%], run AIMER → eval via gating mask across all 9 dims → record results
  - Step 4 (optional): REAP loop: same ratios with WordPress calibration data → eval → record
  - Step 5: Domain specificity analysis: compute per-layer expert overlap between AIMER and REAP retention sets
  - Fix-test-validate loop: if any pruning ratio causes >2pp regression on security dimension → try intermediate ratio → re-eval until clean
  - Embeds `observe-packaging` telemetry agents inline during merge and pruning steps
  - Human review checkpoint: present full comparison table (6 variants: 2 methods x 3 ratios) before committing to physical pruning
  - Step 6: Physical pruning + router re-normalization → verify pruned model loads and generates coherent output

**Plans**: 7/7 plans complete

- [x] 13-01-PLAN.md — MERGE-01 traceability + AIMER weight-norm scorer (wave 1)
- [x] 13-02-PLAN.md — gate-before-remove eval driver + REAP scorer module (wave 1)
- [x] 13-03-PLAN.md — overlap + selection + physical-surgery modules & Wave-0 tests (wave 1)
- [x] 13-04-PLAN.md — AIMER@25% gated eval, gen + judge ensemble (wave 2, the decision gate)
- [x] 13-05-PLAN.md — conditional AIMER 50/75% + REAP + AIMER-vs-REAP overlap (wave 3)
- [x] 13-06-PLAN.md — comparison table + selection + blocking human sign-off (wave 4)
- [x] 13-07-PLAN.md — physical surgery for winner OR documented ship-unpruned close (wave 5)

### Phase 14: Final Comparative Evaluation

**Goal**: The pruned model is A/B compared against the v2.0 RL baseline, with inference speed delta and model size reduction measured alongside the 9-dimension quality report
**Depends on**: Phase 13
**Requirements**: EVAL3-01, EVAL3-02
**Success Criteria** (what must be TRUE):

  1. **[wp-bench HARD GATE]** An A/B eval runs the pruned model against v2.0 RL baseline on wp-bench and the static eval suite, with all results recorded. wp-bench is a hard gate before packaging — the pruned model must meet or exceed the v2.0 RL baseline wp-bench score before Phase 15 begins. Note: this phase requires a different eval harness than Phase 4 (pruned model is a full merged model with no adapter; wp-bench config must target the pruned checkpoint directly).
  2. The report covers all 9 eval dimensions, inference speed delta (expected significant improvement from pruning), model size reduction, and seed variance — sufficient to confirm the full v3.0 pipeline adds value before packaging

**Skill**: Reuse `wp-finetune:run-evaluation` (extend with pruned-model serving + speed benchmarks)

  - **LLM execution**: Claude Code agents ONLY (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline` SKILL.md pattern) — NO Anthropic API direct calls. Any LLM-judged eval dimensions (reasoning quality, score-reasoning consistency) for the pruned model comparison spawn independent Claude Code evaluator agents. Deterministic evals (wp-bench, PHPCS, security, latency/throughput benchmarks) run on DGX without LLM calls.
  - Serve pruned model via `dgx.execute("vllm", ...)` — no LoRA adapter, direct model loading
  - Embeds `observe-inference` telemetry agents inline for latency/throughput measurement during eval
  - Speed benchmark: TTFT and tokens/sec measured across 100 prompts for both `<wp_gen>` and `<wp_judge>` task types
  - A/B comparison automated: pruned model vs v2.0 RL baseline across all 9 dimensions + speed delta + model size
  - Fix-test-validate loop: if pruned model fails to serve (missing weights, router mismatch) → diagnose → fix pruning step → re-serve → re-eval
  - Invokes `wp-finetune:review-telemetry` for consolidated inference performance summary
  - Human review checkpoint: present full comparison report before gating packaging

**Plans**: 1 plan

### Phase 15: Packaging

**Goal**: The pruned model passes cascading compression gates (bf16 baseline, optional quantization, format production) and is published to HuggingFace with full compression lineage, then validated end-to-end on the target serving stack
**Depends on**: Phase 14
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05
**Success Criteria** (what must be TRUE):

  1. Gate 1 completes — the pruned bf16 model's size, inference speed, and all 9 eval dimension scores are recorded as the quality baseline for subsequent compression decisions
  2. Gate 2 decision is documented — whether quantization is warranted based on pruned model size, deployment constraints, and Gate 1 performance margins, with reasoning recorded
  3. If quantization is warranted, incremental testing at Q8->Q6->Q5->Q4 stops at the lowest level holding within +/-2pp of the Gate 1 baseline; quantization is the final step and is never applied before Gate 2 confirms it is needed
  4. The HuggingFace model card documents the full compression lineage (base -> RL -> MoE-Sieve -> merge -> AIMER/REAP winner -> quantization level) with eval scores at each gate, AIMER vs REAP comparison results, and usage examples for both task tokens
  5. E2E inference validation confirms both `<wp_gen>` and `<wp_judge>` prompts produce correct outputs via the final shipped format on the target serving stack (vLLM or Ollama)

**Skill**: `wp-finetune:run-packaging` (NEW — create during phase planning)

  - **LLM execution**: No LLM API used — quantization (Q8/Q6/Q5/Q4), model serving validation, and HuggingFace upload are deterministic operations. E2E inference validation runs `<wp_gen>` and `<wp_judge>` prompts directly against the final shipped model on the target serving stack (vLLM or Ollama) and checks for coherent output — no external LLM judge.
  - Extends `observe-packaging` telemetry pattern: file-integrity agents track quantization output sizes and special token presence
  - Gate 1: `dgx.execute("eval_toolbox", ...)` for bf16 baseline measurement with idempotency check
  - Gate 2: Human decision checkpoint — present Gate 1 results, recommend quantization decision
  - Quantization loop (if warranted): for each level in [Q8, Q6, Q5, Q4] → quantize via `dgx.execute("vllm", ...)` → eval against Gate 1 baseline → if within ±2pp, record as candidate → if regression >2pp, stop and use previous level
  - Fix-test-validate loop: if quantized model fails special token check (AWQ/GGUF token embedding) → fix quantization config → re-quantize → re-test
  - E2E validation: serve final model on target stack → run 20 `<wp_gen>` + 20 `<wp_judge>` prompts → verify coherent output with correct task token routing
  - Embeds `observe-inference` telemetry agents inline during E2E validation for production-representative latency numbers
  - HuggingFace upload: model card generation with full lineage, eval scores at each gate, AIMER vs REAP results
  - Human review checkpoint: final sign-off before `huggingface-cli upload`

**Plans**: 1 plan

### Phase 16: Pipeline Lockdown & Repo Cleanup

**Goal**: Freeze the v3.0 method into a single repeatable pipeline document that an outside user can run on
a fresh same-architecture base (e.g. Qwen3.6-30B-class), preserving the gates that returned nothing (RL
rejected, Sieve full, prune no_winner) as conditional stages rather than deleting them. Deprecate one-off
experiment scaffolding into `deprecated/`, strip dead references from active code, and restructure the
project folder so it is clean and parseable by outside followers.
**Depends on**: Phase 15
**Requirements**: PIPE-01 (repeatable PIPELINE.md), PIPE-02 (deprecate + strip references), PIPE-03
(folder restructure + root cleanup)
**Success Criteria** (what must be TRUE):

  1. `PIPELINE.md` documents every stage end-to-end (data -> SFT gen + judge -> merge -> eval/wp-bench ->
     router-profiling -> sieve k-sweep gate -> merge+AIMER prune gate -> packaging quant gates) with the
     runnable entrypoint for each, the pass/fail gate, and the known Qwen3-30B-A3B result. Gates that
     returned no winner are kept as conditional re-test stages for the next base, not removed.

  2. Dead one-off experiment drivers are moved to `deprecated/` with a README explaining each; no active
     script or skill still references a moved file (verified by grep).

  3. The project root and folder layout read cleanly: stray logs/artifacts swept, semantic grouping, so an
     outside user can clone and follow along.

**Skill**: none new — documentation + refactor phase.

**Plans**: 1 plan

### v3.1 Benchmark, Publish & Next Base

### Phase 17: Benchmark Expansion — wp-bench + SWE-bench Generation Eval

**Goal**: The shipped two-model pair has current, honest benchmark numbers: a full wp-bench run on the v1.2 generation model (shipping stack), and a SWE-bench generation-mode eval that positions the model against a public coding benchmark for the model card
**Depends on**: Phase 15 (shipped pair locked), Phase 16 (pipeline entrypoints documented)
**Requirements**: BENCH-01 (full wp-bench on shipped stack), BENCH-02 (SWE-bench generation-mode eval, feasible scope documented), BENCH-03 (results folded into MODEL_CARD.md)
**Success Criteria** (what must be TRUE):

  1. A full (unlimited) wp-bench run completes on the v1.2 gen model via the shipping stack, score recorded with config + seed, and compared against the 0.4484 Gate-1 receipt
  2. A SWE-bench generation-mode eval (patch generation, non-agentic) runs at the largest scope the DGX Spark (aarch64) toolchain can honestly evaluate — scope, harness constraints, and any subset choice recorded BEFORE results are read
  3. MODEL_CARD.md gains a Benchmarks section with both results, including the honest caveat that the model is WordPress-specialized and SWE-bench is out-of-domain

**Plans**: 3/3 plans executed — Phase 17 COMPLETE 2026-07-11

Plans:

- [x] 17-01-PLAN.md — BENCH-01: full 344-test wp-bench run on v1.2 gen model via vLLM bf16, receipt + delta vs 0.4484 (Wave 1)
- [x] 17-02-PLAN.md — BENCH-02 part 1: arm64 Docker + throughput feasibility probes, arm64 make_test_spec wrapper, scope pre-registration committed before results (Wave 1, checkpoint)
- [x] 17-03-PLAN.md — BENCH-02 part 2 + BENCH-03: SWE-bench generation + arm64 eval at locked scope, then MODEL_CARD Benchmarks section + docs, commit/push as dr-robert-li (Wave 2)

### Phase 18: Production Sweep & HuggingFace Publication

**Goal**: The repo reads production-clean (docs current, stale artifacts deprecated, streamlined layout), and the two-model pair (v1.2 gen + v1.3 3-seed ensemble judge, Q8 GGUF ship tier) is packaged together and published to HuggingFace with the full-lineage model card
**Depends on**: Phase 17 (benchmark numbers for the card)
**Requirements**: PUB-01 (repo sweep + doc currency), PUB-02 (pair packaging), PUB-03 (HF upload + post-upload validation)
**Success Criteria** (what must be TRUE):

  1. README/PROJECT/PIPELINE/STATE agree with each other and with the shipped artifacts; stale files are in deprecated/ with README notes; root is clean
  2. A single HF repo (or paired repos) carries both models with the MODEL_CARD.md lineage, quantization ladder results, and usage examples for both task tokens
  3. Post-upload validation: files download, GGUF loads, and a smoke gen/judge prompt round-trips from the published artifact

**Plans**: 2/2 plans complete

- [x] 18-01-PLAN.md — PUB-01: production sweep — README/PROJECT/PIPELINE currency vs Phase 17 numbers + MODEL_CARD, archive stale artifacts to deprecated/ via mandatory double-grep, clean root (Wave 1)
- [x] 18-02-PLAN.md — PUB-02+03: package the two-model pair, publish two PUBLIC HF repos under iamchum (gen bf16 + judge Q8 GGUF) with cross-linked cards, post-upload download/GGUF-load/gen+judge round-trip validation (Wave 2)

### Phase 19: Next-Base Rerun Roadmap

**Goal**: A costed, evidence-linked roadmap exists for rerunning the full locked pipeline (PIPELINE.md) on the latest same-class Qwen-family MoE base, carrying forward every v3.0 lesson (truncation-aware evals, warm-up gating, no-winner gates as conditional stages)
**Depends on**: Phase 18
**Requirements**: NEXT-01 (base-model research + selection), NEXT-02 (roadmap doc with per-stage cost/time estimates and re-test gates)
**Success Criteria** (what must be TRUE):

  1. The candidate base is selected from current Qwen releases with documented rationale (architecture match, size class, routing concentration prospects, license)
  2. A roadmap document maps every PIPELINE.md stage to the new base with expected deltas, the conditional re-test gates (RL, Sieve, prune) explicitly carried forward, and rough compute/cost estimates

**Plans**: 1/1 plans executed — Phase 19 COMPLETE 2026-07-11

Plans:

- [x] 19-01-PLAN.md — NEXT-01: verify + lock the next base (Qwen3.6-35B-A3B front-runner) with source-cited five-axis rationale; NEXT-02: write `.planning/V4-RERUN-ROADMAP.md` (every PIPELINE stage mapped, three no-winner gates carried forward as re-test stages, architecture-delta work items, cost estimates, carry-forward lessons, pre-registered judge-rho success criteria); closeout + push (Wave 1)

**Outputs:** base locked in `.planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md`
(`Qwen/Qwen3.6-35B-A3B`, live-verified); full stage map in `.planning/V4-RERUN-ROADMAP.md`. v4.0 execution
is a FUTURE milestone requiring explicit human sign-off — no downloads/training happened in this phase.

### v4.0 Pipeline Rerun on Qwen3.6-35B-A3B

**Milestone Goal:** Rerun the locked PIPELINE.md on Qwen/Qwen3.6-35B-A3B (base LOCKED, Phase 19) to break the
judge-rho 0.8075 SFT wall. Pre-registered success: judge rho >0.85 single-seed OR >0.87 3-seed ensemble;
wp-bench >=0.4286 floor (CI-aware — bootstrap lower bound must clear the bar; all six v3.0 carry-forward
lessons apply: truncation-aware evals, warm-up gating, --parallel splitting, CI-aware gates, benchmark
pre-registration, double-grep archive rule). Master plan: `.planning/V4-RERUN-ROADMAP.md`.

**Dependency:** Phase 19 (base locked, roadmap costed) + explicit human sign-off to open v4.0 GPU/Tinker
spend. No downloads/training happen until that sign-off is given.

**Structure note:** This milestone compresses V4-RERUN-ROADMAP.md's proposed 10-phase (20-29) sketch to 8
phases (20-27) by merging Stage 2 (gen SFT) and Stage 3 (judge SFT) into one phase — they share Phase 20 as
their only dependency and run parallel-safe — and merging Packaging with Publication Refresh (strictly
sequential, small scope). The three conditional gates (RL/Sieve/Prune) and the Sieve-tooling work item are
kept as separate phases — each has an independent "no_winner is a result" disposition and distinct,
non-overlapping dependencies (Sieve tooling is explicitly independent of SFT and must not be gated behind
RL) that a merge would blur.

### Phase 20: Base Bring-Up

**Goal**: Qwen3.6-35B-A3B is downloaded, loads correctly, and every architecture-specific serving/training
risk (token alignment, DeltaNet kernel, VL merge path) is smoke-tested and resolved before any SFT run starts
**Depends on**: Phase 19 sign-off (nothing within v4.0 — first phase)
**Requirements**: BASE-01, BASE-02, BASE-03, BASE-04
**Success Criteria** (what must be TRUE):

  1. Qwen3.6-35B-A3B downloads and loads on the GB10 host with `trust_remote_code` on model+tokenizer,
     transformers 5.x import succeeds, and `Qwen3_5MoeForConditionalGeneration` resolves correctly

  2. eos/pad token-ID alignment gate passes — `model.config.eos_token_id`/`pad_token_id` assert-match the
     tokenizer's special tokens, confirmed by a stop-token smoke generation; Stage 2/3 SFT is blocked from
     starting until this gate is green

  3. DeltaNet aarch64 serving smoke passes with vLLM CUDA-graph capture enabled (vLLM >=0.19.0); if capture
     fails, `--enforce-eager` fallback is documented and the `use_kernels` decision is recorded

  4. A full VL merge-path round-trip succeeds: Tinker LoRA export -> merge onto `model.language_model.*`
     keys -> vLLM serve (`--language-model-only`) -> real generation returns coherent output, with the dual
     key-prefix silent-partial-load risk explicitly checked

**Plans**: 4/4 plans complete
**Wave 1**

- [x] 20-01-PLAN.md — BASE-01: v4 config sibling + download/load smoke (class resolution, forward, receipt)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 20-02-PLAN.md — BASE-02: eos/pad alignment gate (fix + natural-stop smoke + Stage 1.5 blocking receipt)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 20-03-PLAN.md — BASE-03: bf16 recipe + v4 serve script + DeltaNet vLLM smoke (CUDA-graph capture, use_kernels=False, vLLM ver log)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 20-04-PLAN.md — BASE-04: prefix-aware merge + serve round-trip (base-vs-merged diff, Tinker target-module log)

### Phase 21: SFT Training — Generation & Judge Models

**Goal**: Both the generation and judge pathways are fine-tuned on the new base using the existing
reasoning-mix and relabel data (no regeneration), and each clears its pre-registered acceptance bar — or the
miss is recorded as a valid, measured outcome
**Depends on**: Phase 20
**Requirements**: GEN-01, GEN-02, GEN-03, JUDGE-01, JUDGE-02, JUDGE-03
**Success Criteria** (what must be TRUE):

  1. The thinking-mode/`<think>` SFT data-format decision is recorded and a rendered-example spot-check
     confirms no spurious empty `<think></think>` blocks and max tokenized length stays under Tinker's 64K
     training-context cap

  2. Generation model SFT completes (MoE-only LoRA r32, LR <=2e-5, frozen router, `output_router_logits=True`)
     on the reused reasoning-mix data and clears the wp-bench floor 0.4286 (CI lower bound, or a
     freshly-measured noise-adjusted floor)

  3. A judge-output-format-compliance smoke on the raw pre-SFT base is run and recorded before bulk judge
     training — guards against the 18%-noncompliance failure mode that killed 3/4 ratios on the old base

  4. 3-seed relabel-SFT (seeds {1,0,2}) completes reusing the v1.3 human-relabeled data (`data/relabel_v1/`)
  5. Judge rho vs held-out relabeled val is measured (vLLM-served, 8192-token cap,
     `scripts/relabel/eval_relabel.py`) against the pre-registered targets (>0.85 single-seed OR >0.87
     3-seed ensemble); if unmet, the miss is recorded as a valid outcome per the milestone's
     failure-disposition rule, not forced

**Cost note**: First real Tinker spend in v4.0 (~$2/run gen, ~$6 for the 3-seed judge ensemble, anchored to
v1.3's $1.83 actual). **Tinker's price rise lands 2026-07-17** — if this phase's runs start on/after that
date, budget the ~10% training-cost increase (prefill/sample ~50% up but minor for this workload).

**Plans**: 6/6 plans complete

Plans:

**Wave 1** *(gates all real Tinker spend)*

- [x] 21-01-PLAN.md — GEN-01 + MoE-merge probe: fork v4 data-adapter/driver siblings, record renderer/LR/format decision (auto-LR ~4.99e-4 supersedes ≤2e-5 spec), prove the `train_mlp=True` fused-expert merge path (discharges 20-VERIFICATION CR-01 + merge-smoke carry-forwards)

**Wave 2** *(all Tinker spend front-loaded before 2026-07-17; JUDGE-01 sole GB10 user)*

- [x] 21-02-PLAN.md — GEN-02: full generation-model SFT on the reused reasoning-mix (MoE-only LoRA r32, frozen router, auto-LR) + terse gate (remote Tinker)
- [x] 21-03-PLAN.md — JUDGE-02: 3-seed relabel-SFT {1,0,2} reusing v1.3 labels (remote Tinker)
- [x] 21-04-PLAN.md — JUDGE-01: raw-base judge-format-compliance smoke vs 18% anchor (local GB10)

**Wave 3** *(local GB10 serial)*

- [x] 21-05-PLAN.md — GEN-03: merge gen adapter → vLLM serve → wp-bench CI-aware vs 0.4286 floor

**Wave 4** *(local GB10 serial; bumped from W3 for single-residency serialization vs 21-05)*

- [x] 21-06-PLAN.md — JUDGE-03: 3-seed 8192-cap Tinker-capture rho + ensemble → merge promoted seed → vLLM-served 8192-cap rho, CI-aware vs pre-registered >0.85 single / >0.87 ensemble

### Phase 22: Sieve/Protected-Mask Tooling Adaptation

**Goal**: The MoE-Sieve profiler and protected-mask pipeline are adapted for the new base's 256-expert,
shared-expert, mixed-DeltaNet/Attention-strata architecture, so Conditional Gate B can run against audited
tooling instead of tooling built for the old 128-expert uniform-attention base
**Depends on**: Phase 20 (parallel-safe with Phase 21 — does not block or wait on SFT training)
**Requirements**: GATE4-02
**Success Criteria** (what must be TRUE):

  1. Profiler module-traversal path is corrected to `model.model.language_model.layers` and `n_experts` is
     bumped from 128 to 256 across all 4 affected scripts

  2. DeltaNet-MoE layers and Gated-Attention-MoE layers are treated as separate strata in the per-layer
     E_eff computation and k-sweep masking logic, not one uniform stack

  3. An empirical check confirms the shared expert never appears in `router_logits` and is excluded from
     the sweepable/prunable expert set

  4. The adapted tooling is verified ready before Conditional Gate B (Phase 25) consumes it — this phase
     closes independently of the RL gate's outcome

**Plans**: 2/2 plans complete

Plans:

- [x] 22-01-PLAN.md — [W1, GPU-free] Architecture-parameterize the Sieve profiler + protected-mask + k-sweep stack: new scripts/sieve_arch.py (config/data-derived dims, layer_types strata, hook-count-verified traversal, jsonl dim inference), wire into the 6 consumer scripts + adapt the vLLM router-mask patch class, all under mock pytest (v3 [48,128] + v4 [40,256] fixtures)
- [x] 22-02-PLAN.md — [W2, GB10] Bounded single-load tooling smoke on models/Qwen3.6-35B-A3B-judge-v4-s1-merged → output/sieve-v4/tooling_smoke.json receipt proving 40 hooks (SC1), router_logits last-dim 256 with shared expert absent (SC3), strata 30 DeltaNet/10 Attention (SC2), tooling ready for Gate B (SC4)

### Phase 23: Final Evaluation

**Goal**: The new gen+judge pair's actual performance against v3.0's shipping figures is measured and
committed — this is the single result the whole v4.0 rerun exists to produce
**Depends on**: Phase 21
**Requirements**: EVAL4-01
**Success Criteria** (what must be TRUE):

  1. An A/B eval of the new pair against v3.0 shipping figures runs on wp-bench and judge rho using the
     identical harness (`scripts/relabel/eval_relabel.py`, vLLM-served, 8192-token cap)

  2. Results are committed to disk (`output/eval4/...`) before any packaging or conditional-gate-continuation
     decision is made

  3. The pre-registered acceptance criteria (judge rho >0.85 single-seed OR >0.87 ensemble; wp-bench
     >=0.4286) are applied mechanically against the measured numbers, and the outcome — met or not met — is
     recorded as the milestone's primary verdict

**Plans**: 1/1 plans complete

Plans:

- [x] 23-01-PLAN.md — Final-eval A/B verdict: receipt-comparability audit + offline raw-base CI backfill, then eval4_final_comparison.json (dual-gen A/B raw-base vs best-trained + judge A/B) with the pre-registered criteria applied mechanically, + VERDICT-EVAL4.md milestone verdict

### Phase 24: Conditional Gate A — RL Re-Test

**Goal**: Determine whether RL is worth re-attempting on the new, higher-rho judge — and only with a
materially different reward family than the one that failed on the old base. **`no_winner` is a valid,
recorded result for this phase — not a failure to force.**
**Depends on**: Phase 23
**Requirements**: GATE4-01
**Success Criteria** (what must be TRUE):

  1. A materially different reward family (execution-grounded, preference-based, or multi-turn — not more
     steps of the old calibration reward) is selected and documented before any run starts

  2. The pre-registered kill criterion (validated teacher-Spearman improvement over warm-start noise, plus
     the codegen Goodhart trip-wire) is carried forward unchanged and wired into the run

  3. A smoke-scale probe runs and is evaluated against the kill criterion before any full-budget commitment
  4. The gate closes with one of two equally valid dispositions — RL adopted (kill criterion cleared) or RL
     rejected/`no_winner` (kill criterion not cleared) — either outcome unblocks Phase 25

**Plans**: TBD

### Phase 25: Conditional Gate B — MoE-Sieve Re-Test

**Goal**: Determine whether the new base's 256-expert routing has enough redundancy for MoE-Sieve
expert-drop to beat full-LoRA at equivalent quality, using tooling already adapted for this architecture.
**`no_winner` (optimal_k=full) is a valid, recorded result for this phase.**
**Depends on**: Phase 24 (RL-before-Sieve precedent — sieve profiling reflects the RL-or-no-RL final
policy), Phase 22 (adapted tooling)
**Requirements**: GATE4-03
**Success Criteria** (what must be TRUE):

  1. Routing is re-profiled on Phase 24's final policy using the Phase 22-adapted tooling (not the
     unaudited 128-expert-era tooling)

  2. A k-sweep runs at multiple expert budgets with the same TOST equivalence gate (epsilon=2pp) and
     CI-aware (bootstrap lower bound) disposition used on the old base

  3. The gate closes with one of two equally valid dispositions — an optimal k below full-LoRA is adopted,
     or `no_winner` (optimal_k=full) is recorded — either outcome unblocks Phase 26

**Note (SC1 supersession):** Phase 24 (RL) is SKIPPED this milestone (no new reward family), so "Phase 24's
final policy" == the v4 judge SFT s1 checkpoint (`models/Qwen3.6-35B-A3B-judge-v4-s1-merged`) — documented in
25-01 and its pre-registration receipt, not silently ignored.

**Plans**: 2/2 plans executed

- [x] 25-01-PLAN.md — [W1] Profile the v4 judge s1 over a bounded judge stimulus (AutoModelForImageTextToText loader guard) → fresh protected mask [40,256] + per-stratum E_eff report + stage-1 k-sweep grid pre-registration
- [x] 25-02-PLAN.md — [W2] Judge-only k-sweep driver + TOST verdict scorer; detached sweep (full arm = live vLLM patch confirm + same-stack TOST reference, then descending k @8192) → optimal_k_v4.json (optimal_k or no_winner), human sign-off

### Phase 26: Conditional Gate C — Merge + Prune Re-Test

**Goal**: Determine whether AIMER/REAP weight-level pruning finally pays off on a base whose Gate B result
(if positive) indicates real expert redundancy. **`no_winner` (ship unpruned) is a valid, recorded result
for this phase.**
**Depends on**: Phase 25
**Requirements**: GATE4-04
**Success Criteria** (what must be TRUE):

  1. LoRA adapters are merged into base weights before any pruning scoring runs
  2. AIMER (primary) and optionally REAP (domain-aware comparison) are scored at the same compression
     ratios used on the old base, with per-dimension retention — especially D2_security — evaluated via
     gate-before-remove (no physical weight removal before the eval passes)

  3. The gate closes with one of two equally valid dispositions — a winning method+ratio ships pruned, or
     `no_winner` ships the merged-unpruned model — either outcome unblocks Phase 27

**Plans**: TBD

### Phase 27: Packaging & Publication Refresh

**Goal**: The final shipping artifact (whatever Phases 23-26 produced) is quantized to a memory-feasible
format, validated end-to-end, and published to HuggingFace with an honest, full-lineage model card
**Depends on**: Phase 23 (bf16 pair minimum — packaging can proceed even if all conditional gates land on
`no_winner`); Phase 25/26 if either produced a compression winner
**Requirements**: PKG4-01, PKG4-02, PUB4-01
**Success Criteria** (what must be TRUE):

  1. Q8 GGUF pair conversion completes (llama.cpp >=b9180) with GGUF block-count verified against the
     safetensors index, concurrent-sequence CUDA-backend smoke passing, and shared-expert quant-type
     metadata independently verified (does not inherit from Phase 22's Sieve-side protection)

  2. Cascading compression gates re-run — Gate 1 bf16 baseline, Gate 2 (pre-determined "warranted" since
     the bf16 pair at 134 GiB exceeds the 121 GB GB10 host), ladder Q8->Q6->Q5 within +/-2pp, no uniform
     4-bit nf4

  3. HuggingFace model cards are updated with the full v4.0 lineage and benchmark deltas vs v3.0, and a
     post-upload round-trip (download, GGUF load, gen/judge smoke) validates the published artifact — same
     discipline as v3.1's PUB-03

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order, SKIPPING Phase 5 (deferred): 1 -> 2 -> 3 -> 4 -> 4.1 -> 4.2 -> 4.3 -> 4.4 -> [5 SKIPPED] -> 6 (already complete) -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15
Note: Phase 4.1-4.4 (v1.2) insert between Phase 4 and Phase 6 — Phase 4 triage is a hard prerequisite for Phase 4.1. Phase 5 is deferred and skipped.
Note: Phase 5 (Packaging/Deployment v1.0) is deferred — v3.0 Phase 15 replaces it as the production packaging step. Phase 5 is never executed standalone.
Note: Phase 6 was completed independently of Phase 5 (depends on Phase 3 + dgx-toolbox).
Note: Phase 7 profiles the v1.2 reasoning adapter (from Phase 4.4), not the v1.0 adapter — v1.2 must complete before Phase 7 begins.
Note: RL (Phases 8-9) runs BEFORE MoE-Sieve (Phase 11) per Issue #1 — routing statistics should reflect reward-aligned behavior.
Note: Phase 10 gates Phase 11 — RL eval results must confirm readiness before MoE-Sieve begins.
Note: Phase 13 MERGE-01 must complete before pruning runs — activation magnitudes require the unified model.
Note: v4.0 (Phases 20-27) starts after Phase 19 sign-off. Phase 22 (Sieve tooling) depends only on Phase 20 and runs parallel-safe with Phase 21 (SFT) — it does not block SFT start. Phases 24/25/26 (Conditional Gates A/B/C) are sequential re-tests where `no_winner` closes the phase validly; Phase 27 (Packaging) proceeds on the Phase 23 bf16 pair regardless of gate outcomes.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Pipeline Ready | v1.0 | 2/2 | Complete | 2026-03-26 |
| 2. Dataset Production | v1.0 | 7/7 | Complete | 2026-03-29 |
| 3. Model Prep and Training | v1.0 | 3/3 | Complete | 2026-03-27 |
| 4. Evaluation | v1.0 | 3/3 | Complete   | 2026-06-07 |
| 4.1. Reasoning Data Generation | v1.2 | 3/3 | Complete | 2026-04-23 |
| 4.2. Reasoning Dataset Assembly | v1.2 | 1/1 | Complete   | 2026-04-25 |
| 4.3. Reasoning Fine-Tune | v1.2 | 4/4 | Complete   | 2026-06-11 |
| 4.4. Reasoning Eval & Merge | v1.2 | 5/4 | Complete   | 2026-06-13 |
| 5. Packaging and Deployment | v1.0 | 0/3 | Deferred to v3.0 | - |
| 6. Adaptive Training Planner | v1.1 | 6/6 | Complete | 2026-04-01 |
| 7. Router Profiling & Protected Expert Set | v2.0 | 2/2 | Complete (approved 2026-06-19) | 2026-06-19 |
| 8. Reward Infrastructure | v2.0 | 4/4 | Complete   | 2026-06-19 |
| 9. GSPO Training | v2.0 | 6/6 | Complete   | 2026-06-20 |
| 10. RL Comparative Evaluation | v2.0 | - | CLOSED — RL rejected (2026-07-05) | 2026-07-05 |
| 11. Compression & Packaging (two-model pair) | v3.0 | 5/5 | Complete   | 2026-07-09 |
| 12. MoE-Sieve Comparative Evaluation | v3.0 | - | SKIPPED (optimal_k=full — no variants to A/B) | 2026-07-10 |
| 13. LoRA Merge & Pruning | v3.0 | 7/7 | Complete   | 2026-07-10 |
| 14. Final Comparative Evaluation | v3.0 | 1/1 | Complete (re-confirmation; no pruned/RL variant) | 2026-07-10 |
| 15. Packaging | v3.0 | 1/1 | Complete (Gates 1-2 + card + bf16 E2E; Q8 GGUF LOSSLESS ship tier; Q6/Q5 deferred) | 2026-07-11 |
| 16. Pipeline Lockdown & Repo Cleanup | v3.0 | 1/1 | Complete (PIPELINE.md + 95 files deprecated + cleanup) | 2026-07-10 |
| 17. Benchmark Expansion (wp-bench + SWE-bench gen) | v3.1 | 2/3 | In Progress|  |
| 18. Production Sweep & HuggingFace Publication | v3.1 | 2/2 | Complete   | 2026-07-12 |
| 19. Next-Base Rerun Roadmap | v3.1 | 1/1 | Complete (base locked: Qwen3.6-35B-A3B; V4-RERUN-ROADMAP.md written) | 2026-07-11 |
| 20. Base Bring-Up | v4.0 | 4/4 | Complete    | 2026-07-13 |
| 21. SFT Training — Generation & Judge Models | v4.0 | 6/6 | Complete    | 2026-07-14 |
| 22. Sieve/Protected-Mask Tooling Adaptation | v4.0 | 2/2 | Complete    | 2026-07-15 |
| 23. Final Evaluation | v4.0 | 1/1 | Complete    | 2026-07-14 |
| 24. Conditional Gate A — RL Re-Test | v4.0 | 0/TBD | Not started | - |
| 25. Conditional Gate B — MoE-Sieve Re-Test | v4.0 | 2/2 | In Progress|  |
| 26. Conditional Gate C — Merge + Prune Re-Test | v4.0 | 0/TBD | Not started | - |
| 27. Packaging & Publication Refresh | v4.0 | 0/TBD | Not started | - |
