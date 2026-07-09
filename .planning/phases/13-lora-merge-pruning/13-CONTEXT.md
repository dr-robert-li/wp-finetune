# Phase 13 — LoRA Merge & Pruning (AIMER primary, REAP optional) — CONTEXT

**Scaffolded:** 2026-07-10 · **Status:** ready to plan
**Requirements:** MERGE-01, PRUNE-01..06 (check REQUIREMENTS.md lines ~194-206)

## Inherited verdicts (LOCKED — do not relitigate)

1. **optimal_k = full (Phase 11, human sign-off 2026-07-10).** NO expert-count compression headroom
   at k≤64: wp-bench −22pp at k=64, judge collapses to 0/121 parseable at k≤32. Cause: E_eff ~88-99
   active experts/layer of 128. `output/sieve/prune_set_for_phase13.json` is the binding handoff.
2. **Ship pair:** v1.2 gen (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`, wp-bench 0.4484 vLLM) +
   v1.3 3-seed judge ensemble (merged s0/s1/s2 under `models/_staging/`, ens rho 0.8075 vLLM).
   Single-seed s1 fallback pre-authorized.
3. **vLLM serving gap:** Tinker-native numbers are sampler-specific; ALL Phase 13+ gates reference the
   vLLM-measured baselines (0.4484 / 0.8075) per `output/sieve/sanity_gate_recalibration.json`.
4. **Protected mask inviolable:** 1,480 experts ([48,128] bool, sha-pinned). NO pruning method may
   remove a protected expert. `layer_stability_notes`: low-Jaccard band {9,13,14,31,35,36} + late
   layers {45,46,47} carry a pre-committed **median-threshold (2,477-expert) headroom** obligation —
   pruning on those layers must be more conservative.
5. **MERGE-01 is largely moot:** there are no unmerged adapters left — RL was rejected (no RL LoRA)
   and the Sieve was training-free (no Sieve LoRA). Gen and all 3 judge seeds are ALREADY merged
   full checkpoints. MERGE-01 closes with an N/A-style traceability record, not new work.

## The live question Phase 13 answers

**Does WEIGHT-level expert selection (AIMER norms; optionally REAP saliency) find a prunable expert
subset that ROUTING-coldness could not?** The k-sweep pruned by routing counts. AIMER ranks by weight
norms — a different signal. Key untested region: **25% compression keeps 96 experts/layer > E_eff ~90**
— genuinely open. 50% (keep 64) replicates the k=64 collapse territory; 75% is bounded-dead. Expect
the answer to be "≤25% or nothing"; the pre-registered PRUNE-05 rule (reduce compression until clean)
handles it.

## HARD CONSTRAINTS

1. Protected mask excluded from every candidate prune set (both methods, every ratio). Verify subset
   property programmatically per ratio; mask files byte-unchanged (sha check).
2. **PRUNE-03 gate-before-remove:** evaluate every method×ratio via GATING MASK first (the Phase 11
   `scripts/sieve_expert_mask_inference.py` machinery is exactly this — reuse it). Physical weight
   removal (PRUNE-06) happens ONLY for the winning variant after PRUNE-05 selection.
3. Regression bars (vLLM, like-for-like): gen wp-bench ≥ 0.4484 − 2pp; judge ensemble rho ≥ 0.8075 −
   0.052 (two-SE floor per `optimal_k.json` convention). Judge parse-rate must stay ≥ 95% (121-item
   val) — the k-sweep showed parse collapse is the judge's failure mode.
4. GB10 memory wall: one ~60GB model resident at a time; sequential serve/swap. In-process bf16 load
   peaks ~100GiB (2× staging transient).
5. No training of any kind. Pruning + router renormalization only.
6. Judge ensemble = 3 seeds sharing one routing profile (cross-seed Jaccard 0.933, `sieve_profile_mode
   = shared`). A single shared prune-set must be validated on ALL 3 seeds (ensemble rho gate), not
   just s1.

## Open questions for research/planning

- AIMER implementation: is there a maintained reference (paper/repo) or does D-09's "weight-based,
  no calibration, ~1 second" reduce to computing per-expert weight norms locally? (Likely the latter —
  norms over each expert's w1/w2/w3 tensors; check wp-moe.md + any D-09 notes.)
- REAP optionality (PRUNE-02/04): given no-expert-drop and the ≤25% ceiling, is the REAP comparison
  worth its calibration-run cost? Recommend: run REAP only if AIMER@25% passes gates (otherwise the
  domain-specificity sub-experiment is moot — nothing prunable to compare).
- Whether gen and judge get DIFFERENT prune sets (gen E_eff ~61-88 lower than judge ~73-99 — gen may
  tolerate more pruning) or one shared set for operational simplicity.
- PRUNE-06 physical removal mechanics for Qwen3-30B-A3B on vLLM: variable experts-per-layer support,
  router re-normalization (Phase 11's apply_mask renorm is the reference), HF checkpoint shape.

## Key inputs

- `output/sieve/prune_set_for_phase13.json` (binding handoff)
- `output/sieve/k_sweep_results.json`, `optimal_k.json` (collapse evidence + bars)
- `scripts/sieve_expert_mask_inference.py` + tests (gate-before-remove machinery, reusable)
- `output/profiling/reasoning-merged-v4/` + `output/sieve/judge-s{0,1,2}/` (routing reports)
- `wp-moe.md` (method reference), REQUIREMENTS.md PRUNE-01..06 exact text
