---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: MVP
status: executing
stopped_at: Phase 7 context gathered
last_updated: "2026-06-14T05:31:47.231Z"
progress:
  total_phases: 10
  completed_phases: 8
  total_plans: 35
  completed_plans: 34
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-05)

**Core value:** A single self-hostable model that generates WPCS-compliant WordPress code and catches critical defects via structured 9-dimension rubric scoring
**Current focus:** Phase 07 — router-profiling-protected-expert-set

## Current Position

Phase: 07 (router-profiling-protected-expert-set) — EXECUTING
Plan: 2 of 2

**Outcome:** v1.2 reasoning-merged-v4 promoted to canonical `models/qwen3-30b-wp-30_70-reasoning-merged-v4`
(13 shards), serves correctly. Post-merge 10+10 validation: **wp_gen 10/10, wp_judge 10/10, routing 20/20**.

**How it closed (D-V4-10 waiver):** The automated 8-gate cascade returned `automated_pass=false` on 3
blockers — REVL-01A Spearman 0.240<0.263, confusion false_FAIL 0.452>0.403, REVL-02 PHPCS 0.9412 (1/17).
The mechanism diagnosis (`04.4-D-V4-JUDGE-MECHANISM-DIAGNOSIS.md`) proved all 3 are statistically weak:
bootstrap CIs span their bars, paired Δρ(merged−grid) includes 0, REVL-01B teacher-corr ROSE — the model
is statistically indistinguishable from the grid winner that PASSED 4.3. The only real effect is a
SIGNIFICANT but rank-preserving −3.58pt calibration offset (bf16-merge artifact), corrected downstream.
REVL-04 codegen PASSED (0.4603, reproduced grid exactly). Human chose: diagnose → recalibrate (cleared 0
gates, offset rank-invariant) → **WAIVER** (D-V4-10) → REVL-05 sign-off (regression scan clean:
invalid-PHP-pass 0/24, terse 0.8%) → triple-gated promote.

**Forward obligations (todos created):**

  - **Phase 8 MUST inherit the judge recalibration** `output/eval_reasoning_v4_winner/judge_recalibration.json`
    (score_offset=+3.58, D-V4-09) as a HARD input to the 30% wp_judge reward — gate/reward consistency.

  - **Phase 7/8 gate definitions** should adopt a CI-aware noise-band disposition (require bootstrap
    lower bound to clear the bar, measured identically on baseline + candidate) — D-V4-10 hardening.

Next: Phase 04.4 complete. The v1.2 reasoning model unblocks **Phase 7 (Router Profiling)** — the
downstream v2.0 dependency. NOTE: `phase.complete` routed `next_phase=06` (Adaptive Training Planner),
but Phase 6 is a v1.1 phase already marked complete 2026-04-01 (its detail checkbox is stale) — confirm
the intended next phase (likely Phase 7) before planning. Artifacts: `04.4-04-SUMMARY.md`,
`04.4-D-V4-10-WAIVER.md`, `postmerge_validation_v4.json`.

---
### (Historical, superseded) Pre-04.4 RC-A/RC-B diagnosis — RC-A was the harness ghost, now fixed

**RC-B is the SOLE remaining blocker.** D-IT-02 diagnosis (debug session
`reasoning-merge-gen-regression`) split the "merge regression" into two independent causes:

  - **RC-A (CONFIRMED + FIXED 2026-06-10):** the judge parse-failure/Spearman regression was an
    EVAL HARNESS BUG, not the merge. eval_judge omitted `enable_thinking=False` → merged Qwen3
    emitted UNCLOSED `<think>` → unparseable judge JSON. Fix (commit b88faa3): `_judge_create()`
    helper passes the kwarg at both judge call sites with loud fallback. CONFIRMED by re-running
    REVL-01A on EXISTING v3 staging through the patched harness: parse 0.190 -> **0.0248** (<=0.05),
    Spearman -> **0.2446** (~= E3 Tinker-runtime 0.2626, baseline 0.2678). The parse gate that
    arrested plans 07/08 was a harness ghost. Evidence:
    `output/eval_reasoning_v3/revl01a_v3_rcA_confirm.json`.

  - **RC-B (CONFIRMED real, FULLY ATTRIBUTED 2026-06-10):** wp-bench codegen drop 0.4537 -> 0.3716
    under CORRECT thinking-off inference. Genuine reasoning↔codegen interference. Three cheap probes
    (NO full ablation): (1) per-expert delta-norms UNIFORM across 128 experts -> MoE-subset salvage
    DEAD; (2) single-component wp-bench probe (anchored, WPBENCH_LIMIT=30) -> CODEGEN damage is mostly
    MoE (attn-only execution-correctness 0.375 = baseline EXACTLY; MoE-only 0.3125 ~ v3_full 0.292);
    (3) single-component JUDGE census -> **judge skill lives ENTIRELY in MoE** (attn-only = 100%
    parse fail / no judge; MoE-only Spearman 0.3124, BETTER than v3_full 0.2446). NET PICTURE:
    **MoE deltas carry BOTH judge skill AND codegen damage (entangled in one component); attention
    deltas are net-HARMFUL (add codegen damage + slightly hurt judge Spearman, contribute no judge
    skill).** MoE-only is strictly better than v3_full on BOTH axes but still < baseline codegen.
    Artifacts: `output/eval_reasoning_probe_dit02/dit02_attribution_result.json` +
    `dit02_judge_location_result.json` + `dit02_expert_delta_norms.json`.
**HUMAN DECISION 2026-06-10: attribution probe -> RETRAIN.** CORRECTED direction (judge==MoE, so
"cut MoE" would kill judge; "lean attention-heavier" is WRONG — attention is judge-useless):
re-open Phase 4.3 and retrain to make the MoE judge-training LESS codegen-destructive without
losing judge skill — candidate levers: lower MoE LoRA rank/LR (smaller perturbation), MORE wp_gen
codegen replay (protect base coding), consider DROPPING the attention target entirely (net-harmful),
find the MoE rank/replay sweet spot. Re-merge + re-gate REVL-04 (reliable post RC-A fix). The
MoE-only merge (Spearman 0.3124, codegen 0.4071) is a cheap interim improvement over v3_full but
still fails REVL-04 (< baseline ~0.4537) — retrain is required to close the codegen gap.
OPEN sub-question to settle in/around the retrain: confirm judge skill survives reduced-MoE (the
probe measured codegen only; if judge skill is MoE-borne, balance rank accordingly). The lm_head /
attempt-2(q_proj) merge-variant track is moot — it chased the RC-A harness ghost.
Evidence: debug `.planning/debug/reasoning-merge-gen-regression.md` + `revl01a_v3_rcA_confirm.json`

+ `dit02_attribution_result.json` + `dit02_expert_delta_norms.json`.

---
### (Historical, pre-04.4-merge) v3 corrective-training readiness — SUPERSEDED by the REVL-04 result above

**REVL-05 human re-gate against `wp-reasoning-v3`, then Phase 7.** P0-P5 DONE. The
judge-quality corrective branch is complete and promoted **`wp-reasoning-v3`** (Tinker run
`3497a27e...:train:0`) — it fixes BOTH halves of REVL-05. Approach (see
`04.3-P5-CORRECTIVE-RESULTS.md` + `VERDICT-POLICY.md`): 30 invalid-PHP/fabricated-API
`should_fail` training negatives (distinct from the 24 held-out sentinel) + a verdict POLICY
(PASS iff overall>=70 & no auto-FAIL class; deterministic post-hoc threshold, NO teacher
relabel) + a two-sided confusion gate (so a FAIL-everything model can't false-green the
one-sided sentinel). Scorecard v2(P4)->v3: FS terse 5.8%/5.6% -> **0%/1.1%** PASS; REVL-01A
0.316 -> **0.263** (>=0.171 baseline) PASS (slight dip — score-correlation traded for verdict
quality); invalid-PHP sentinel POLICY false-pass 1/24 -> **0/24** PASS (the @100 fabricated
blind spot FIXED); confusion false-FAIL-on-PASS 40%->**24%** + recall-on-FAIL 64%->**78%**
(Pareto win, NOT over-strict). ALL FOUR gates pass. Export:
`models/tinker_export/wp-reasoning-v3/checkpoint.tar` (HF PEFT LoRA adapter); manifest
`output/tinker/wp-reasoning-v3-manifest.json`. (P4 `wp-reasoning-v2` superseded; its results
in `04.3-P4-TINKER-RESULTS.md`.) Driver: `scripts/tinker_reasoning_sft.py --train-path`;
data: `scripts/build_reasoning_negatives.py` + `build_augmented_train.py`; gates:
`check_invalid_php_sentinel.py` + `check_verdict_confusion.py` + `tinker_fs_gate.py` +
`eval/eval_judge.py --responses-jsonl`.
Status: Ready to execute
Note: Local artifacts `models/qwen3-30b-wp-30_70-merged-v2` + `...-reasoning-merged` + `adapters/.../checkpoint-72` are READ-ONLY references/fallback only (NOT promoted). The GB10 memory wall is documented in `output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`. `04.3-REOPEN-PLAN.md` remains a 0-task brief — do not execute.

Progress: [██████████] 97%

## Performance Metrics

**Velocity:**

- Total plans completed: 9
- Average duration: 9 min
- Total execution time: 0.62 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-pipeline-ready | 2 | 27 min | 13 min |
| 02-dataset-production | 2 | ~10 min | ~5 min |
| 03-model-prep-and-training | 2 | 34 min | 17 min |
| 04.4 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: 03-02 (22 min), 03-01 (12 min), 01-02 (2 min), 01-01 (25 min)
- Trend: steady

*Updated after each plan completion*
| Phase 02-dataset-production P01 | 25 | 2 tasks | 7 files |
| Phase 02-dataset-production P03 | 4 | 2 tasks | 3 files |
| Phase 03-model-prep-and-training P01 | 12 | 2 tasks | 5 files |
| Phase 03-model-prep-and-training P02 | 22 | 2 tasks | 8 files |
| Phase 03-model-prep-and-training P03 | 10 | 2 tasks | 2 files |
| Phase 02-dataset-production P04 | 3 | 2 tasks | 98 files |
| Phase 02-dataset-production P05 | 8 | 2 tasks | 28 files |
| Phase 02-dataset-production P06 | 5 | 2 tasks | 37 files |
| Phase 06-adaptive-training-planner P01 | 18 | 2 tasks | 3 files |
| Phase 06-adaptive-training-planner P02 | 8 | 2 tasks | 2 files |
| Phase 06-adaptive-training-planner P03 | 8 | 2 tasks | 3 files |
| Phase 06-adaptive-training-planner P04 | 5 | 2 tasks | 0 files |
| Phase 04-evaluation P02 | 6 | 1 tasks | 1 files |
| Phase 04.1 P02 | 142 | 2 tasks | 4 files |
| Phase 04.4 P06 | 8 | 2 tasks | 4 files |
| Phase 04.4-reasoning-eval-adapter-merge-inserted P07 | 0 | 2 tasks | 3 files |
| Phase 04.4-reasoning-eval-adapter-merge-inserted P08 | 12 | 1 tasks | 2 files |
| Phase 04.3-reasoning-fine-tune-inserted P03 | 4min | 3 tasks | 2 files |
| Phase 04.3 P01 | 18min | 3 tasks | 3 files |
| Phase 04.3-reasoning-fine-tune-inserted P02 | 15 | 3 tasks | 5 files |
| Phase 04.4-reasoning-eval-adapter-merge-inserted P01 | 15 | 2 tasks | 2 files |
| Phase 04.4-reasoning-eval-adapter-merge-inserted P02 | bookkeeping-only | 2 tasks | 2 files |
| Phase 04.4-reasoning-eval-adapter-merge-inserted P03 | 15m | 3 tasks | 11 files |
| Phase 07-router-profiling-protected-expert-set P01 | 20 | 3 tasks | 10 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.2 Roadmap 2026-04-05]: Phases 4.1-4.4 inserted as decimal phases between Phase 4 and Phase 5; covers all 14 v1.2 requirements (DGEN-01-05, RTRN-01-04, REVL-01-05)
- [v1.2 Roadmap 2026-04-05]: Phase 4.1 requires pilot validation (20-50 examples per stream) before bulk generation — pilot confirms WP-specific citation quality
- [v1.2 Roadmap 2026-04-05]: Phase 4.2 score consistency validation is a hard gate — examples where reasoning contradicts numeric scores are rejected before training mix assembly
- [v1.2 Roadmap 2026-04-05]: Training mix: reasoning examples + 30% flat judge replay + 20% wp_gen replay — prevents format collapse and generation regression
- [v1.2 Roadmap 2026-04-05]: Phase 4.3 LR must be ≤2e-5 (5-10x lower than Phase 3's 2e-4); router weights confirmed frozen before training
- [v1.2 Roadmap 2026-04-05]: Phase 4.4 human sign-off required before adapter merge — adapter written to models/ only after human approval
- [v1.2 Roadmap 2026-04-05]: Phase 7 dependency updated to Phase 4.4 (v1.2 reasoning adapter) — v1.2 must complete before MoE-Sieve profiling; fresh routing profile required even though router was frozen during v1.2
- [v2.0 Reorder 2026-04-08]: Pipeline reordered per Issue #1 (D-07): RL (Phases 8-9) runs BEFORE MoE-Sieve (Phase 11) — routing statistics should reflect reward-aligned behavior
- [v2.0 Reorder 2026-04-08]: v2.0 now Phases 7-10 (Router Profiling, Reward Infra, GSPO Training, RL Eval); GSPO primary per D-08
- [v2.0 Reorder 2026-04-08]: Phase 10 (RL Eval) gates Phase 11 (Post-RL MoE-Sieve) — RL eval results must confirm readiness
- [v3.0 Reorder 2026-04-08]: v3.0 now Phases 11-15 (Post-RL MoE-Sieve, Sieve Eval, Merge+Pruning, Final Eval, Packaging)
- [v3.0 Reorder 2026-04-08]: Phase 13 MERGE-01 (LoRA merge) must precede pruning — activation magnitudes require unified model
- [v3.0 Packaging]: Quantization is the final step in Phase 15, gated by cascading eval (Gate 1 bf16 baseline, Gate 2 quantization decision)
- [v3.0 Pruning]: AIMER/REAP tests 25%, 50%, 75% compression ratios; WordPress domain narrowness may support aggressive pruning to ~8-12B total params
- [Phase 4 Triage 2026-04-06]: Human override — 30_70 accepted as winning ratio despite Spearman gate failure (0.5698 < 0.85); only ratio with non-zero Spearman, perfect PHPCS+security; Spearman threshold waived for this triage run, remains a hard gate in Phase 4.4 with human-annotated test set
- [Phase 4 Triage 2026-04-06]: wp-bench gate deferred to Phase 4.4 — was skipped in triage run; Phase 4.4 must run full wp-bench eval before adapter merge
- [v2.0 Roadmap]: Phase 7 execution blocked on Phase 4.4 completing (need reasoning-enhanced adapter; fresh routing profile required for both RL and eventual MoE-Sieve)
- [Phase 06-adaptive-training-planner]: Human review checkpoint approved 2026-04-01 — all Phase 6 scripts verified before DGX execution
- [Phase 06-adaptive-training-planner]: Canonical JSONL schema updated to GPUSampler fields (watts, temperature_c, gpu_util_pct, mem_available_gb)
- [Phase 06-adaptive-training-planner]: adaptive-planner skill is a thin wrapper: all decision logic stays in scripts/adaptive_planner.py
- [Phase 04-evaluation]: Task 2 execution deferred: CUDA unavailable in current Python env (cpu-only torch); GPU execution requires correct CUDA-enabled environment activation on DGX Spark
- [Phase 04-evaluation]: run_eval_and_wpbench_for_ratio keeps vLLM alive between eval_gen, eval_judge, and wp-bench for same ratio to avoid restart overhead
- [Phase 04.1-02]: Model name claude-sonnet-4-6-20250514 does not exist; correct name is claude-sonnet-4-5 for this API key
- [Phase ?]: 04.4-06
- [Phase ?]: D-IT-04 applied: lm_head LoRA stage dropped; D-IT-05 attempt-1 scope q_proj kept
- [Phase ?]: v4 flag defaults OFF preserving v3 reproducibility; anchors parameterized with --report/--staging back-compat defaults
- [Phase 04.4-07 2026-06-10]: D-IT-04/05 attempt-1 hypothesis falsified — lm_head exclusion + q_proj retention did NOT recover merged-served parse rate (0.2479 > 0.05; marginally worse than v3 0.1901); Spearman also regressed (0.1534 < baseline 0.2678); REVL-02 PHPCS passed (1.0); plan 08 will early-exit on parse_gate_pass=false
- [Phase 04.4-reasoning-eval-adapter-merge-inserted]: 04.4-08: REVL-04 exit 7 is designed correct behavior — parse_gate_pass=False (0.2479) fired D-IT-09 fail-fast; v4 attempt-1 disqualified at parse gate; fail-path = attempt-2 (exclude q_proj, D-IT-05) or D-IT-02 diagnosis
- [Phase 04.3-03]: decide() extracted as pure module-level function (no subprocess/file-IO) so a selection-key or exit-2 bug surfaces in minutes via synthetic test, not after ~24h of live grid runs (T-04.3-08)
- [Phase 04.3-03]: Judge bar mode default is 'point' (POINT Spearman >= 0.263); ci_lower mode available as noise-guard diagnostic but is NOT the acceptance threshold (D-N7 resolved 2026-06-11)
- [Phase 04.3-03]: Wpbench baseline 0.4537 is the full 344-test HARD gate (D-N8); the dit02 30-test probe value 0.4857 is NOT used as the acceptance threshold
- [Phase 04.3]: Preserved --exclude-lm-head merge_type on non-MoE-only path in merge_tinker_v3.py to avoid silent regression: PATTERNS only set merge_type in is_moe_only branch; added else branch for tinker_per_expert_moe_plus_peft_attention_NO_lm_head (Rule 1 auto-fix)
- [Phase 04.3]: Used patch.object on force-resolved lazy transformers class objects (_AMFCLM = transformers.AutoModelForCausalLM at import time) rather than patch('transformers.AutoModelForCausalLM'): _LazyModule always re-derives class from internal mapping bypassing __dict__ writes; patching from_pretrained on the resolved class is the only reliable intercept
- [Phase ?]: Phase 04.3-02: Body-keyed leakage guard for wp_gen replay; pool is 66K+ not short
- [Phase ?]: Added --adapter redirect to _04.4_anchors_v3.py so v4-winner tar overrides v3 default
- [Phase ?]: reuse_revl04=false: grid staging shards absent, Plan 03 must re-bench REVL-04 on clean canonical staging
- [Phase ?]: baseline pinned at 0.4537 policy constant (D-V4-02/D-N8); fresh-bench baseline 0.4299 recorded as drift check only in revl04_disposition.json
- [Phase ?]: automated_pass=false: 3 HARD gate failures (REVL-01A rho 0.240<0.263, REVL-02 PHPCS 0.9412<0.98, confusion Pareto false-FAIL 0.468>0.403) block Wave 4 promotion

### Pending Todos

- Phase 4 triage COMPLETE (human override accepted 2026-04-06) — begin Phase 4.1 (pilot 20-50 examples per stream before bulk generation) using 30_70 adapter
- Verify mutation pool size at `data/phase2_synthetic/output/mutated/` before setting Phase 4.1 critique-then-fix targets
- Resolve Unsloth PEFT stacking question before Phase 4.3: Option A (nested LoRA on adapter) vs Option B (LoRA on merged model) — blocking question for training setup

### Blockers/Concerns

- [Phase 4.4 — BLOCKED 2026-06-08, D-05 ITERATE]: **REVL-04 wp-bench HARD gate FAILED on the v3 reasoning-merge.** Fresh full 344-test run on merged-served v3 vs baseline merged-v2: **reasoning 0.3716 < baseline 0.4537, pass=false** (`output/eval_reasoning_v3/04.4_wp_bench_results.json`). Faithful — no `<think>` leakage, no max_tokens truncation, both serves thinking-OFF (the defensible "did the merge preserve base coding ability" comparison). Root-cause of 3 harness iterations: Qwen3 template enables thinking by default → v3 wrote code inside an unterminated `<think>` → fixed by `chat_template_kwargs enable_thinking=false` (commits fbda6d3, 3267f3b; matches plan-02 fidelity invocation). The `target_modules=all-linear` merge degraded generation layers (execution corr 0.292 vs 0.417); JUDGE path still transferred (plan 02 L3≥0.95). **User chose ITERATE** (not abandon). Plans 04.4-04/05 BLOCKED. **Next: planning workflow to define the iteration** (re-open Phase 4.3 re-train OR a targeted-merge that spares gen layers), then re-run REVL-04 on the new candidate. v3 staging archived, NOT promoted. See `04.4-03-SUMMARY.md`.
- [Phase 4.2]: COMPLETE — gate passed, 418-example dataset shipped to data/reasoning_dataset/
- [Phase 4.3]: COMPLETE — training loss 1.22→0.86, ckpt-72 shipped. RTRN-04 post-hoc gate INVALID at 4-bit on Qwen3-MoE (router-quant collapse → degenerate output regardless of adapter). Training-loss curve is the success signal.
- [Phase 4.4 BLOCKER — UPDATED 2026-05-29]: Eval architecture investigation produced 4 findings: (a) **v1 30_70 baseline merge ACCEPTED** as `models/qwen3-30b-wp-30_70-merged-v2/` via CPU-only raw-HF+PEFT script `_p0_unsloth_merge_v3.py` — adapter contains zero `gate_up_proj` LoRA tensors (training never wrote them); v3 correctly fuses everything that exists (down_proj per-expert + attn + embed/lm_head). (b) **RESEARCH "Pitfall 5" narrative requires revision** — v1 partial baseline was not caused by PEFT dropping target_parameters at merge time; the gate_up_proj LoRA was never trained. (c) **P0 v2 (GPU Unsloth) OOMed** on GB10 unified memory: `max_memory={0:'80GiB','cpu':'30GiB'}` is an accelerate hint, not a hard cap; Unsloth pinned ~110 GiB GPU+CPU pages on a 121 GiB total system. CPU-only v3 path avoids NVRM entirely. (d) **ckpt-72 reasoning adapter uses Unsloth's fused-experts shared-rank LoRA layout** (`mlp.experts.base_layer.lora_A/B` + `mlp.experts.lora_A/B`); raw PEFT `merge_and_unload()` would silently corrupt per-expert deltas because PEFT's strided B-indexing convention (`B[:, e::E]`) DIFFERS from Unsloth's training-time contiguous-block convention (`B[:, e*R:(e+1)*R]`). Council-approved merge path: **unsloth-static fused-MoE candidate** — hybrid (attention-only PEFT adapter merge + custom Unsloth-convention per-expert MoE delta application + gate/up chunk split for Llama-style fused output dim). Promotion gates: tensor-level + multi-layer forward-pass equivalence anchors against Unsloth's `_extract_lora_from_wrapper`.
- [Phase 4.4 — RESOLVED/REJECTED 2026-06-02]: All eval-architecture blockers above were resolved (merge done + 5-gate certified; eval-harness prose compat shipped via eval/output_parsers.py + dim_map.json). Gates RAN: REVL-01/02/04 PASS, REVL-03 MARGINAL, REVL-07 PASS-SOFT, REVL-08 FLAG-SOFT, REVL-06 N/A. **REVL-05 human review REJECTED** the reasoning-merged model (35% terse-JSON judge collapse + invalid-PHP-pass). ckpt-72 NOT promoted. Disposition pending: Phase 4.3 format-stability re-train (recommended) vs ship v1 merged-v2 as v1.2-final. See 04.4-GATE-LEDGER.md + 04.4-D05-DIAGNOSIS.md.
- [Phase 7]: BLOCKED — needs a v1.2 reasoning adapter promoted; Phase 4.4 rejected ckpt-72, so Phase 7 waits on either the 4.3 re-train OR an explicit decision to ship v1 merged-v2 as v1.2-final (baseline carries no reasoning enhancement).
- [Phase 6]: dgx-toolbox Phase 13 (telemetry/ package) must be complete before Phase 6 can execute
- [Phase 8]: Phase 7 (router profiling + protected expert set) must complete before Phase 8 (reward infrastructure) begins
- [Phase 10]: Phase 9 (GSPO training) must complete before Phase 10 (RL eval) — RL eval gates v3.0 MoE-Sieve
- [Phase 11]: Phase 10 (RL eval) must confirm readiness before Phase 11 (post-RL MoE-Sieve) begins — fresh RL-policy routing profiling required
- [Phase 13]: LoRA merge (MERGE-01) must complete before pruning — strictly sequential within the phase; AIMER primary (D-09)
- [Phase 15]: Quantization (PKG-03) is gated by Gate 2 decision — verify AWQ support for Qwen3-30B-A3B in vLLM (likely native)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260328-h1d | Make dgx_toolbox.py project-agnostic by moving all hardcoded couplings to dgx_toolbox.yaml | 2026-03-28 | e1cec35 | [260328-h1d-make-dgx-toolbox-py-project-agnostic-by-](./quick/260328-h1d-make-dgx-toolbox-py-project-agnostic-by-/) |
| 260329-cy1 | Sync STATE.md, ROADMAP.md, PROJECT.md with current project state | 2026-03-29 | 726ec5a | [260329-cy1-update-state-md-roadmap-md-project-md-to](./quick/260329-cy1-update-state-md-roadmap-md-project-to/) |
| 260329-g30 | Prefix all 8 skills with wp-finetune: for Claude Code discoverability | 2026-03-29 | ba82fab | [260329-g30-rename-all-skills-with-wp-finetune-prefi](./quick/260329-g30-rename-all-skills-with-wp-finetune-prefi/) |
| 260329-g6l | Update README and CHANGELOG to match current project state | 2026-03-29 | 33f5311 | [260329-g6l-update-readme-and-changelog-to-reflect-c](./quick/260329-g6l-update-readme-and-changelog-to-reflect-c/) |
| 260329-g9p | README How It Works section with both pipeline and training skills | 2026-03-29 | 163497e | [260329-g9p-readme-how-it-works-section-with-both-pi](./quick/260329-g9p-readme-how-it-works-section-with-both-pi/) |
| 260329-gmf | README add observe and review-telemetry skill descriptions | 2026-03-29 | d31e013 | [260329-gmf-readme-add-observe-and-review-telemetry-](./quick/260329-gmf-readme-add-observe-and-review-telemetry-/) |
| 260329-gre | Add telemetry opt-in Step 0c to run-training skill | 2026-03-29 | 09beab1 | [260329-gre-add-telemetry-opt-in-step-0c-to-run-trai](./quick/260329-gre-add-telemetry-opt-in-step-0c-to-run-trai/) |
| 260329-gvs | Update README CHANGELOG with telemetry opt-in and training steps | 2026-03-29 | 9a99da8 | [260329-gvs-update-readme-changelog-with-telemetry-o](./quick/260329-gvs-update-readme-changelog-with-telemetry-o/) |
| 260403-rut | Remove Unsloth and dgx_toolbox from merge_adapter.py; use AutoModelForCausalLM | 2026-04-03 | e8ae427 | [260403-rut-fix-container-dependency-hell-add-standa](./quick/260403-rut-fix-container-dependency-hell-add-standa/) |
| 260403-utp | Fix stale eval tests to match current API (rubric refactor) | 2026-04-03 | 49ec4b6 | [260403-utp-fix-stale-eval-tests-to-match-current-ev](./quick/260403-utp-fix-stale-eval-tests-to-match-current-ev/) |
| 260403-vvg | Fix stale unsloth refs in dgx_toolbox; fix CONFIG_PATH; add missing dataloader fields to 30_70/40_60 configs | 2026-04-03 | f340b22 | [260403-vvg-fix-stale-unsloth-refs-and-config-incons](./quick/260403-vvg-fix-stale-unsloth-refs-and-config-incons/) |

## Session Continuity

Last session: 2026-06-14T05:31:47.222Z
Stopped at: Phase 7 context gathered

Prior session: 2026-06-02T21:31:00.000Z
Stopped at: Phase 4.4 CLOSED **REJECTED** at REVL-05 (human). All automated gates run; merge NOT promoted; D-05 disposition pending (recommend Phase 4.3 format-stability re-train). See `04.4-GATE-LEDGER.md` + `04.4-D05-DIAGNOSIS.md`. Resume = decide D-05.

### Session 2026-06-02 — Phase 4.4 gate execution → REVL-05 REJECTED

- **Session-start recovery:** prior context contained FABRICATED tool output (nonexistent `/home/robert_li/models`, a phantom `VALIDATION.md`, invented scores). Re-verified everything with clean tools. Truth: 8 REVL gates (REQUIREMENTS.md:129-136), merge already done + 5-gate-certified at `models/qwen3-30b-wp-30_70-reasoning-merged` (merge_report.json). `04.4-02-PLAN.md` is legitimate, NOT hallucinated. Deleted my wrong ground-truth doc.
- **REVL-04** ✅ PASS — committed wp-bench 8-blocker fix chain (commit a1cc63a): 0.4616 ≥ 0.4286. Repro helpers `_wpbench_shim/npx` + `_wpbench_pth/usercustomize.py`.
- **REVL-06** 🚫 N/A (Option C) — judge-only model emits 0 `<corrected_code>` across 478 rows; fix-correctness already gated by REVL-04. Slot retired, not vacuously passed. Forces W5-01 (REVL-05) to CoT-only.
- **REVL-03** ⚠️ MARGINAL — built capture+emitter+aggregator (commits ae0cbe7/a77a3ee). Advisor caught a taxonomy bug: pinned to model's REAL 8-dim rubric (was naive D1..D9 incl. unreachable error_handling), consistent w/ REVL-01 dim_map.json. Final 0.814 (2048-capture, parseable 0.992), but bootstrap 95% CI [0.751, 0.871] STRADDLES 0.80 + between-run spread 0.038 — not a clean pass. Deferred to REVL-05.
- **REVL-07** ✅ PASS (SOFT) + **REVL-08** ⚠️ FLAG (SOFT) — commit 6b80177. REVL-07 F1-opt thr 50.0 (F1 0.924). REVL-08 median 456<500 (terse) — confirmed REAL bimodal behavior, NOT a capture-cap artifact (re-captured @2048; max 1121≪2048).
- **Bimodality finding:** model output is ~63% full-prose+[/REASONING] / ~35% terse direct-JSON (no reasoning). Same prose/JSON split as W0-03 smoke. The terse mode drags REVL-03 coverage to its floor.
- **REVL-05** ❌ **REJECTED** (commit 1cd809f) — `output/v1.2_human_review_completed.md` sentinel HUMAN_REJECTED: persistent terse-JSON mode + boilerplate drift + annotated CRITICAL (model passed syntactically-invalid PHP, `$this->` in a standalone function = runtime fatal). Corroborates the REVL-03 marginal as genuinely disqualifying. Built `build_human_review.py` + sentinel gate (commit 62d0d42).
- **D-05 diagnosis** (commit f3cd4e7, `04.4-D05-DIAGNOSIS.md`): terse-JSON 35%, NO trigger cluster (uniform across format / code-length / difficulty). Scoring mostly harsh-not-lax (4.1% canonical false-pass; underscores GT 36% vs overscores 5%). Read = training-config / FORMAT-STABILITY failure, not data/approach. Recommend targeted Phase 4.3 re-train; v1 merged-v2 stays certified fallback. ckpt-72 NOT promoted.
- **Phase 4.3 re-train first task** (when opened): compare terse-JSON rate at **checkpoint-50 vs checkpoint-72** (both exist under `adapters/qwen3-30b-wp-30_70-reasoning/`) on a held-out slice → answers late-collapse (LR/epoch) vs format-token root cause cheaply before any GPU re-train.
- Commits this session (9): a1cc63a, ae0cbe7, a77a3ee, 6b80177, 0a6d59e, fa4ef70, 62d0d42, 1cd809f, f3cd4e7.

### Session 2026-05-30 (prior) — W1-W6 cascade eval-harness compat

Last session: 2026-05-30T06:30:00.000Z
Stopped at: W1-W6 cascade BLOCKED on eval-harness prose compat (2 layers). Findings + council direction in EVAL-HARNESS-COMPAT.md. Next: resolve Blocker-2 dim-map → build eval/output_parsers.py → preflight → orchestrator → cascade. Merges + smoke all done/certified.

### Session 2026-05-30 W1-W6 cascade attempt → eval-harness compat blocker

- Attempted W1-W6 eval cascade kickoff (user: full-set + wp-bench). W1 merge already done (reasoning-merged certified). W2-02 orchestrator + serve_reasoning + REVL-03/06/07/08 NOT built (planned). eval_gen/eval_judge exist (Phase-4); run_eval_triage.py (53K) reusable machinery; wp-bench dir+config present.
- **BLOCKER 1 (council ACKED)**: eval_judge.parse_judge_response JSON-only; reasoning model emits prose → REVL-01 Spearman uncomputable both sides (model + GT). eval_gen <think> contamination risk. Council binding: Option B (output_format json|prose|auto flag, shared eval/output_parsers.py), two-GT (rubric_scorer=canonical/REVL-01A HARD; teacher-target=diagnostic/REVL-01B SOFT), per-row provenance, no silent fallback, parser-coverage preflight HARD gate. GRPO reward MUST ground in rubric_scorer (not assistant targets).
- **BLOCKER 2 (NEW, needs council resolution)**: dimension-taxonomy mismatch. Reasoning prose 9-dim rubric {wpcs, security, sql, perf, wp_api, i18n, a11y, code_quality, dependency_integrity} ≠ eval_judge {D1-D7 + D8_error_handling, D9_code_structure}. 7/9 clean; 2 diverge (Code Quality, Dependency Integrity vs D8, D9). Affects parser dim-map + REVL-01 per-dim Spearman. Recommended: overall-Spearman HARD + 7-dim-clean per-dim SOFT. See EVAL-HARNESS-COMPAT.md.
- Council resolved: Option 3 (overall-Spearman HARD + 6-dim-clean per-dim SOFT). dim_map empirically = 6 clean (I18n absent from prose, not 7).
- **BUILT + committed 0b14735**: eval/dim_map.json (checked-in dim reconciliation) + eval/output_parsers.py (strip_think, parse_judge_scores json|prose|auto, extract_php_code) + 12 tests. VALIDATED on real data: reasoning-merged model output 5/5 parse (4 prose+1 json), teacher GT 111/121, gen code clean. 31 phase4_4 tests pass.
- **BLOCKER 3 (NEW, surfaced, awaiting council)**: prose output has NO overall_score; REVL-01A HARD = Spearman(model_overall vs rubric_overall) needs prose model_overall DERIVED. Rec: weighted-mean via DIMENSION_WEIGHTS (symmetric w/ rubric). See EVAL-HARNESS-COMPAT.md Blocker 3.
- NEXT (after B3 ack): wire eval_judge (output_format param, parse_judge_scores for model output, two-GT rubric-canonical, derive prose overall, restrict per-dim to 6 clean, provenance fields) + eval_gen (extract_php_code) + parser-coverage preflight HARD gate + W2-02 orchestrator + cascade.

### Session 2026-05-29/30 PR1+PR2 + W0-03 smoke PASS

- **PR1 (4f2ea11)**: triage_ratios HUMAN_OVERRIDE sentinel guard (refuse-clobber + --force-override + sanity assert), restored override file, 7 tests.
- **PR2 (a4a8a7a/b25306b/dc5374a/PR2b/2f723df/c246a20)**: dual-stage W0-03 smoke gate.
  - Council redesign: original adapter-vs-old-baseline via Unsloth in-process is STALE (merge forensics changed inputs). New: merged-reasoning vs merged-v2, both bf16 vLLM-servable, no 4-bit. Threat model shifted (4-bit collapse → vLLM serving divergence + reasoning-effect-present).
  - Stage 1 CPU degenerate pre-flight (3 prompts, 128 tok, 180s guard); Stage 2 vLLM certifying (10 prompts, full check stack).
  - smoke_common classifiers (19 unit tests): is_degenerate (loop/length/4.3-fingerprint), judge_coherent (BIMODAL prose-OR-json), baseline_similarity (no-op canary), inter_prompt_distinctness (mode-collapse), strip_think.
  - PR2a committed manifest data/phase4_4/smoke_prompts.json (5 judge + 5 gen, CtF idx 85); PR2b committed baseline outputs (vLLM merged-v2).
  - 3 harness bugs found on first run (model output was GOOD): chat-template missing (Stage1 false-empty); judge bimodal CoT→prose/CtF→JSON (prose-only false-failed JSON); <think></think> contaminating php_lint. All fixed + 5 added tests.
  - **CERTIFIED VERDICT (c246a20)**: smoke_pass=True exit=0 distinctness=0.879. judge 5/5 (prose 9/9 dims + 1 CtF json), gen 5/5 php_lint, baseline-sim 0.02-0.42 (<0.85 canary → reasoning diverges). Artifact: merge-artifacts/w0_03_smoke_PASS_verdict.json.
  - Data finding flagged: reasoning judge output is dimensional PROSE (CoT) or JSON (CtF), NOT <REASONING>-tagged. parse_judge_response(JSON-only) would have false-failed all CoT — coherence redesigned prose-aware + json-aware.

Resume file: None
Next: apply PR1+PR2 pre-exec blockers (HUMAN_OVERRIDE sentinel + sanity assertions + smoke-gate hardening), THEN W0-03 smoke gate against models/qwen3-30b-wp-30_70-reasoning-merged/ vs models/qwen3-30b-wp-30_70-merged-v2/ baseline, THEN REVL-01..08 eval gates

### Session 2026-05-29 reasoning MERGE COMPLETE + PROMOTED

- **Reasoning adapter ckpt-72 merged + promoted** → `models/qwen3-30b-wp-30_70-reasoning-merged/` (13 shards). Merge type: unsloth-static fused-MoE per-expert + PEFT attention. Script `scripts/_p0_merge_unsloth_static_moe.py`.
- **Council "broadcast" math FALSIFIED before launch.** Pre-flight probe `scripts/_p0_extraction_probe.py`: broadcast `(B@A)*scale` → all experts gave cos_sim 0.08 vs per-expert (orthogonal, 12× over-magnitude). Correct = per-expert contiguous block `delta_e = B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] * scale`, byte-exact (max_diff<1e-6) to Unsloth `_extract_lora_from_wrapper` (moe_utils.py:421-426).
- **transformers 5.3.0 stores experts as FUSED stacked 3D params** (`gate_up_proj (128,1536,2048)`, `down_proj (128,2048,768)`) — no nn.ModuleList, no gate/up split (gate_up stays fused). First merge attempt crashed `Qwen3MoeExperts not subscriptable`; fixed to operate on `param.data[e]`.
- **5-gate promotion ALL PASS** (recorded in `models/qwen3-30b-wp-30_70-reasoning-merged/merge_report.json`):
  1. static-classified (Hypothesis A/C, no router LoRA)
  2. tensor-anchor byte-exact to Unsloth source (384 checks, max_diff 0.0) — `scripts/_p0_anchor_tensor_full.py`
  3. forward-anchor bf16-calibrated 9/9 router-invariant — `scripts/_p0_anchor_forward_rankpath.py`
  4. fp32-control rms sub-bf16-floor — `scripts/_p0_anchor_fp32_control.py`
  5. merge_report nonzero touched counts (gate_up 6144, down 6144, per-expert-differ 0.000129)
- **Forward-anchor threshold recalibration (council A+B).** Initial fp32-grade thresholds (cos≥0.99999, rel_l2≤1e-3) FAILED AS EXPECTED — bf16-stored 30B weights can't meet fp32 equivalence. fp32 weight-control is PRIMARY certifier (cand == bf16(true fp32 merge), rms 2-3e-5 < bf16_floor 9e-5). Forward anchor demoted to bf16-calibrated corroboration (cos≥0.99990, rel_l2≤1e-2, mean≤2e-3, router-invariant hard-required) → PASS 9/9. Observed cos 0.99996.
- New scripts: `_p0_merge_unsloth_static_moe.py`, `_p0_extraction_probe.py`, `_p0_anchor_tensor_full.py`, `_p0_anchor_forward_rankpath.py`, `_p0_anchor_fp32_control.py`. Logs: `logs/phase4.4/unsloth_static_merge.log`, `forward_anchor.log`.

### Session 2026-05-29 P0 forensics

### Session 2026-05-29 P0 forensics

- **v1 30_70 baseline ACCEPTED as v3 CPU-only merge.** P0 v2 (Unsloth GPU) OOMed at 15:28:19 local during PEFT adapter load — root cause: `max_memory` accelerate hint not enforced, GPU+CPU pinned pages exceeded 121 GiB unified ceiling. Path A (CPU-only raw-HF+PEFT) implemented in `scripts/_p0_unsloth_merge_v3.py` + launcher `_run_p0_remerge_v3.py`. Result: `models/qwen3-30b-wp-30_70-merged-v2/` (13 shards, 56.9 GiB), RETURNCODE 0 in ~3 min.
- **v3 differential check uncovered structural finding.** v1 adapter contains 0 `gate_up_proj` LoRA tensors (12674 keys total: 12288 per-expert down_proj + 384 attention + 2 modules_to_save). Council ruling: v3 IS correct baseline; no further v1 merge work.
- **ckpt-72 reasoning adapter ≠ v1 structure.** 576 keys, 12 patterns × 48 layers, all under Unsloth fused-experts shared-rank format: `mlp.experts.base_layer.lora_A (4096, 2048)` + `lora_B (1536, 4096)` (fused gate_up_proj), `mlp.experts.lora_A (4096, 768)` + `lora_B (2048, 4096)` (fused down_proj). 4096 = 32 × 128.
- **Block-diagonal disambiguation test** (`scripts/_p0_ckpt72_block_diag_test_v2.py`): cross/diag norm ratio ≈ 0.99 under both Unsloth contiguous-block and PEFT strided conventions. Initially interpreted as "dense cross-mix → Path I invalid" but Unsloth source probe revealed cross-terms are random-init artifacts; PEFT and Unsloth runtimes both use diagonal-block-only contractions. Block-diagonal test was misframed as a lossless-extraction proxy.
- **Unsloth source probe (`unsloth_zoo/temporary_patches/moe_utils.py`)** confirmed Hypothesis A/C (static linear weight-level delta, no router dependence). Training-time convention: `delta[e] = weight_B[:, e*R:(e+1)*R] @ weight_A[e*R:(e+1)*R, :] * scaling`. **PEFT 0.18.1's `get_delta_weight` uses INCOMPATIBLE strided convention** (`B[:, e::E]`) — would silently corrupt per-expert deltas on Unsloth-trained adapters.
- **Gate/up split convention confirmed** for Qwen3-MoE: `gate, up = x.chunk(2, dim=-1)` (gate first half, up second half of fused 1536-dim output).
- **Council direction (binding)**: rename "Path I" → "unsloth-static fused-MoE candidate". Hybrid merge: attention-only PEFT temp adapter + custom Unsloth-convention MoE per-expert delta + gate/up chunk split. Two-stage anchor: tensor-level pre-flight + multi-layer (0/23/47) forward-pass equivalence. Candidate dir `models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate/`. Promotion only after all 5 gates pass (static-classified, tensor-match, forward-match, sanity checks, merge_report.json shows non-zero touched counts).
- **04.4-RESEARCH.md Pitfall 5 narrative requires revision** before W5-01/W5-02 fire (prevents reviewer bias toward incorrect baseline-degradation conclusion).
- New files: `scripts/_p0_unsloth_merge_v2.py` (v2, deprecated — OOMed), `_p0_unsloth_merge_v3.py` (v3, accepted), `_run_p0_remerge_v3.py`, `_p0_v3_diff_check.py`, `_p0_ckpt72_block_diag_test.py`, `_p0_ckpt72_block_diag_test_v2.py`. Logs: `logs/phase4.4/p0_remerge*.log`.

### Phase 4.2 progress (2026-05-21) — superseded by 4.4 context but kept for audit

### Phase 4.2 progress (2026-05-21)

- ✅ Task 0 — consistency validation. Rewrote validator mega-prompt → batched ThreadPoolExecutor (10-ex batches, workers=3, haiku/claude_agent per D-01). Fixed parse bug (status colon + null reason). Result: 365/375 consistent, 10 rejected (2.7%, genuine contradictions).
- ✅ Task 1 — assembly. Fixed 2 KeyErrors (replay examples lack metadata.source_file → defensive fallback) + tagged replay stream/format. Mix policy decision: keep ALL 365 reasoning + 15% replay (CoT supply-capped at 190, can't hit 60%). Output: data/reasoning_dataset/ — openai_train.jsonl (350) + openai_val.jsonl (79) = 429; mix 44.3 CoT/40.8 CtF/14.9 replay; stratified split consistent across train/val; metadata.json written. Canonical template verified (CoT+CtF have [/REASONING]+<judge_output>; replay raw).
- ✅ Task 2 — SKILL.md structural checks pass (Trigger, 4 steps, agent constraint, Key Rules).
- ✅ Task 3 — checkpoint:human-verify GATE PASSED (2026-05-21). Multi-model external review (GPT-5.5/Opus 4.7/Gemini 3.1) flagged vendor contamination + truncated-PASS; Claude Code agent inspected 29 flagged rows against actual data. Verdict: 2 qualitative claims TRUE (10 protobuf rows scored as WP; idx251 truncated+PASS), scaled claims FALSE (CoT JSON 181/181 valid; reasoning↔verdict 10/10 coherent; review's evidence was phase1b calibration artifact, wrong dataset).
- ✅ FILTER FIX: added reproducible `filter_reason()` to assemble_reasoning_dataset.py (protobuf/GPBUtil/SDK-namespace vendor regex + brace-imbalance>2 truncation). Validator intermediate (consistency_valid.jsonl) was lost → applied filter in-place to reviewed output (re-validation would re-roll the LLM gate). Removed exactly 11 (10 vendor: 8 CoT + 2 replay; 1 truncated CoT). Backups: data/reasoning_dataset/*.pre_vendorfilter.* (untracked, local safety net).
- ✅ FINAL dataset: 429→418 (341 train / 77 val), mix 43.3 CoT / 41.9 CtF / 14.8 replay. Verified: 0 vendor, 0 brace-imbalance, 0 "protobuf"; CoT 181/181 valid+markers; CtF 175/175 markers.
- ⚖️ D-05 deviation ACCEPTED by human: locked mix 60/25/15 → actual 43/42/15 (CoT supply caps ~181, can't reach 60%). Kept 3 correctly-FAILed truncation rows (138/147/287) as reject-broken→FAIL signal.
- ⚠️ Unit tests stale: scripts/test_validate_reasoning_consistency.py + test_assemble_reasoning_dataset.py import removed `process_examples` API + mock-patch removed funcs. Pipeline functionally verified instead. Tests need separate update pass (cf. quick-task 260403-utp pattern).
- Uncommitted: scripts/validate_reasoning_consistency.py, scripts/assemble_reasoning_dataset.py, data/reasoning_dataset/* (new).

### Active Background Work (2026-05-14 → 2026-05-19)

**Stream: Judge Re-Calibration (Phase 1a/1b — informal, predates next roadmap phase)**

- Phase 1a COMPLETE — XGBoost dual-head classifier + regressor; schema-tolerant derive_gt(); Pearson ≥ 0.75 gate (council vote swap from Spearman); v2 calibration trained on 580-row dataset
- Phase 1b pilot COMPLETE — 800 functions stratified, calibrated_overall clip [0,100] applied, 57.3% agreement w/ Claude verdict (divergence at 7-7.99 boundary noted)
- **Phase 1b 20K rejudge COMPLETE 2026-05-19T23:53** — 20000/20000, 0 failures, 4.7 days elapsed
  - Output: `data/phase1b/rejudge_full_20k.jsonl` (17.6M, 14 keys/row)
  - Calibrated_overall: mean 70.3, range [0, 98.4]
  - Verdict dist: 14137 PASS / 5862 FAIL / 1 None
  - Bucket strat preserved: 9178×(8-8.99 + 9-10), 866×(7-7.99), 778×(0-4.99); 5-6.99 pool gap as known
  - vLLM config: bf16 Qwen3.6-35B-A3B, prefix-caching enabled, max_model_len=32768, max_num_batched_tokens=32768, gpu-mem-util=0.8
  - Bottleneck: decode-bound at 11.4 tok/s per stream → ~0.045 functions/s aggregate across 4 workers

## Session Activity

| Date | Activity |
|------|----------|
| 2026-04-23 | Phase 4.1 complete — 196 CoT + 179 CtF bulk examples accepted. Data quality fixes: rejection examples restored, metadata corrected. |
| 2026-04-23 | Session resumed — ready for Phase 4.2 planning |
| 2026-05-14 | Phase 1a calibration complete (v2 dual-head XGBoost, Pearson gate); Phase 1b pilot 800 done; 20K launched workers=6 after OOM recovery. |
| 2026-05-15 | Phase 1b 20K vLLM restart w/ prefix-caching + 32k batch tokens + 32k model_len; rejudge workers=4 --resume; decode-bound at 11.4 tok/s confirmed. |
| 2026-05-19 | Phase 1b 20K rejudge COMPLETE — 20000 rows, 14137 PASS / 5862 FAIL, calibrated mean 70.3. |
| 2026-05-20 | Disagreement review done (CAL 46/80 = 59%, conditionally trustworthy). SEC-N04 false-positive pattern flagged. Patched: SEC-N04 prompt + severity 4->2, context-aware suppression (admin paths + REST routes + WP_REST_Controller), test/vendor pre-filter. Launched 2689-row PASS->FAIL subset rerun (PID 1378489, ETA ~18.7hr). Smoke 20/20 confirmed 60% flip FAIL->PASS. |
| 2026-05-21 | SEC-N04 subset rerun COMPLETE (2689 rows, 1642=61.1% flipped FAIL->PASS). Spliced into rejudge_full_20k.jsonl (backup: rejudge_full_20k.pre_secn04.jsonl). **Agreement 65.2% -> 73.4%.** New verdict dist: 15780 PASS / 4219 FAIL / 1 None. Bucket 8-8.99: 70.0%, 9-10: 83.1%. |
| 2026-05-21 | Advisor review: flip mechanism validated, outcome not yet. Flip-branch analysis: admin_path=39 (all legit migration/upgrade files), rest=27, llm_revised=1216 (74% — LLM self-revised w/ new prompt), other=361. Dumped 25-case new-flip spot-check (output/phase1b_newflip_review.md) — GATING human eyeball. Built consumption-ready data/phase1b/rejudge_20k_downstream.jsonl (18894 rows after dropping 1105 test/vendor + 1 None; 14987 PASS/3907 FAIL; **75.3% agreement**). 772 of 778 in 0-4.99 bucket were test code. |

### Calibration Readiness — GATE PASSED ✅ (2026-05-21)

**Status:** Ready to execute

- ✅ SEC-N04 false-positive fix applied + validated (agreement 65.2%->75.3% on consumption file)
- ✅ Test/vendor pre-filter applied (1105 dropped)
- ✅ **GATE PASSED: new-flip review 23/25 training-worthy (92% ≥ 90% threshold). output/phase1b_newflip_review_completed.md.**
- Residual error mode (2/25 = 8%): (1) db_query WooCommerce migration — admin-path suppression hides unrelated raw-SQL issue; (2) export_popup_action jupiterx — severity 4->2 let genuine auth-missing case squeak to 38.0 (marginal). Both edge cases.

**Carried caveats:**

- XGBoost not retrained on post-suppression feature distribution (empirically consistent w/ smoke + review, accept for v1)
- 1 None row dropped from consumption file
- 3181 cl=FAIL->cal=PASS unchanged (reviewer validated CAL-correct: docblock over-strictness)

**Future refinement (non-blocking):** make SEC-N04 severity drop (4->2) conditional on suppression-context present, to keep genuine no-context auth-missing cases at FAIL. Would require another rerun.

**Committed in 1e53e31** (2026-05-21, "SEC-N04 + 20K calibration shipped"): eval/rubric_definitions.py, eval/rubric_scorer.py, scripts/phase1b_*.py, scripts/monitor_phase1b_*.sh, data/phase1b/* (rejudge_full_20k, rejudge_20k_downstream, rerun_secn04_fix, secn04_subset_row_ids). Phase 1b stream is COMMITTED — earlier "uncommitted" note was stale. NOTE: output/triage_decision.md has an uncommitted clobber (auto-regen flipped 30_70 human-override → NO_SURVIVORS) — do NOT commit; needs HUMAN_OVERRIDE sentinel (see 04.3-CONTEXT follow-ups).
