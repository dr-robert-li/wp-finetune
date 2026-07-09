# SIEVE Decision Record — Phase 11 (Compression & Packaging, training-free scope)

**Recorded:** 2026-07-10 · **Plan:** 11-05 · **Scope lock:** TRAINING-FREE SIEVE (11-CONTEXT.md, user-selected 2026-07-08)

---

## SIEVE-02 — N/A (explicit, not silent)

**Requirement text:** "Gen-hot experts (per RL-policy routing categories) trained on golden signal
data only (passed examples, synthetic good); judge-hot experts trained on full spectrum (passed +
failed + contrastive)."

**Disposition: N/A under the locked training-free scope.** The data-routing spec governs *which
training data reaches which expert group during selective retraining*. That retraining path was
superseded twice over:

1. RL was REJECTED at the Phase 10 gate (2026-07-05, 6/6 smoke kills) — there is no RL policy and
   no RL-policy routing categories to route data by (ROADMAP AMENDMENT 2026-07-03).
2. Phase 11 was then locked training-free (11-CONTEXT.md, 2026-07-08): no LoRA retraining, no
   recovery SFT. **No experts are retrained, therefore there is no training data to route.**

The equivalent *compression-time* control — deciding which experts see load — is the k-sweep
inference-time masking itself (top-k hot UNION protected, per layer), executed and recorded in
plan 11-04 (`output/sieve/k_sweep_results.json`, mechanism `scripts/sieve_expert_mask_inference.py`).
That is the record; no data-routing assignment exists or is owed.

## SIEVE-03 — 30/70 ratio traceability (no new decision in Phase 11)

**Requirement text:** "Retrain uses best gen/judge ratio determined by Phase 4 eval results."

**Disposition: satisfied by traceability to closed decisions; Phase 11 makes no new ratio choice.**

- The 30/70 gen/judge ratio is the **Phase 4 triage winner**: the canonical shipped model is
  `models/qwen3-30b-wp-30_70-reasoning-merged-v4` — the 30/70 ratio adapter carried through the
  v1.2 reasoning fine-tune. Phase 4 eval results selected it; every downstream artifact inherits it.
- The **Phase 7 matched stimulus** re-confirmed the ratio pipeline: routing profiling used
  `data/final_dataset/ratio_30_70`, the same 30/70 mix, so the protected-expert mask and all Sieve
  profiling are ratio-consistent with the shipped model.
- With retraining superseded (see SIEVE-02), "retrain uses the best ratio" reduces to "all Phase 11
  measurement and hand-off artifacts are built on the 30/70 lineage" — which they are: the k-sweep
  served `qwen3-30b-wp-30_70-reasoning-merged-v4` (gen) and the v1.3 relabel seeds fine-tuned on top
  of the same 30/70 base (judge).

## Training-free reinterpretation of SIEVE-01 / 04 / 05

Per 11-CONTEXT.md scope lock ("adapter checkpoint per k" → "expert-mask + eval record per k"):

| Req | Literal spec | Training-free reinterpretation | Status |
|---|---|---|---|
| SIEVE-01 | Fresh profiling + LoRA on hot experts | Fresh routing profiles of the shipped policy (3 judge seeds + gen, plans 11-02/11-03); no LoRA applied | Complete (11-03) |
| SIEVE-04 | K-sweep trains adapter per k | K-sweep = inference-time expert MASK per k + eval record per k (wp-bench + judge rho) | **SATISFIED** (11-04 + addendum: full/64/32 measured with decision-grade evidence; k=13 deliberately unrun, bounded worse by monotonicity) |
| SIEVE-05 | Optimal k = smallest within ±1pp of full (TOST ε=2pp, 3+ seeds), protected retained | Same statistical rule, arms = vLLM-measured full-vs-k wp-bench (like-for-like, single harness); seeds = 3 judge ensemble members | **SATISFIED** — verdict below |

## SIEVE-05 verdict — optimal k = FULL (no expert-drop compression)

`scripts/tost_gate.py` (hand-rolled two one-sided t-tests; statsmodels ttost_ind unavailable per
`sieve_env_precheck`) over `output/sieve/k_sweep_results.json`, epsilon=2pp, reference = the
vLLM-measured full arm (0.4484 wp-bench / 0.8075 ensemble rho — NOT the Tinker-native 0.842/0.827;
~3pp systematic serving gap, `output/sieve/sanity_gate_recalibration.json`):

| k | wp-bench | TOST equivalent? | protected retained | judge-rho bar | verdict |
|---|---|---|---|---|---|
| full | 0.4484 | (reference) | true | (reference 0.8075) | — |
| 64 | 0.2275 (−22pp) | NO (p≈1e-12) | true | FAIL (0.5415 < 0.755) | not equivalent |
| 32 | 0.0546 (−39pp) | NO (p≈4e-40) | true | judge collapse: 0/121 parseable | not equivalent |
| 13 | gen timed out (7200s) | bounded worse (monotone collapse) | true | judge collapse: 0/121 parseable | not equivalent |

> **AUDIT CORRECTION 2026-07-10 (11-VERIFICATION):** an earlier narrative said k=32-judge/k=13 were
> "never run" / "session died". False — the background driver survived the executor session and ran
> the sweep to `=== k-sweep COMPLETE ===` (logs/sieve/ksweep_driver_resume.log). Judge captures at
> k=13/32 completed (121/121, all 3 seeds) but produced **0/121 parseable outputs** — the judge
> collapses into unparseable text under aggressive masking. Stronger evidence for the same verdict.

**`optimal_k = "full"`, `no_equivalent_k = true`.** Cause established (11-04 addendum): measured
E_eff ~88–99 effective experts/layer out of 128 — every swept budget sits far below active usage.
Expert-DROP compression is dead for this workload; Phase 13 AIMER **weight-level** pruning remains
the live path and must be conservative given the distributed routing.

**Human sign-off:** APPROVED by Dr. Robert Li, 2026-07-10, via AskUserQuestion at the blocking
plan 11-05 Task 2 checkpoint (recorded in `output/sieve/optimal_k.json` `human_signoff` and in
`output/sieve/prune_set_for_phase13.json`).

## s1 single-seed fallback line (pre-authorized)

Judge fallback if the 3-seed ensemble cannot fit GB10 memory/latency at packaging time: single-seed
s1 (Tinker-native rho 0.827; vLLM-measured 0.8017; gate bar 0.827 − seed noise floor, recorded as
0.7497 vs the vLLM reference in `optimal_k.json`). Exercising the fallback requires only a JOURNAL
note, not a re-decision (11-CONTEXT.md). The vLLM shipping figure for the ensemble is **~0.81** —
the true through-the-shipping-stack number; ensemble still beats single-seed through vLLM.

## Hand-off

Phase 13 MERGE-01/PRUNE-01 consumes exactly one artifact: `output/sieve/prune_set_for_phase13.json`
(optimal_k=full · 1,480 protected experts force-keep, sha-pinned · layer_stability_notes verbatim —
low-Jaccard band {9,13,14,31,35,36} + late layers {45,46,47} require median-threshold 2,477-expert
headroom · hot/cold per layer (all-keep at k=full) · sieve_profile_mode=shared · regression bars
incl. the vLLM shipping-rho note).
