---
phase: 26-conditional-gate-c-merge-prune-re-test
plan: 01
subsystem: moe-prune
tags: [aimer, prune, gate-before-remove, moe-sieve, v4-judge, tost, d2-security]
status: complete
requires:
  - output/sieve-v4/protected_expert_mask.npy
  - output/sieve-v4/ksweep/kfull/s1/judge_responses.jsonl
  - models/Qwen3.6-35B-A3B-judge-v4-{s0,s1,s2}-merged
provides:
  - scripts/aimer_prune_v4.py
  - scripts/prune_gate_v4.py
  - output/prune-v4/aimer_scores_judge_v4.npy
  - output/prune-v4/masks/aimer_k224.npy
  - output/prune-v4/protected_manifest_v4.json
  - output/prune-v4/gated/aimer_224_judge.json
  - output/prune-v4/gated/aimer_224_d2.json
affects: [phase-26-02]
tech_stack:
  added: []
  patterns: [stacked-tensor-axis0-reduction, serve-not-load, same-stack-tost, gate-before-remove]
key_files:
  created:
    - scripts/aimer_prune_v4.py
    - scripts/prune_gate_v4.py
    - output/prune-v4/aimer_scores_judge_v4.npy
    - output/prune-v4/masks/aimer_k224.npy
    - output/prune-v4/protected_manifest_v4.json
    - output/prune-v4/gated/aimer_224_judge.json
    - output/prune-v4/gated/aimer_224_d2.json
  modified: []
decisions:
  - "k=224 keep-mask built with build_uniform_keep_mask (exactly 224/layer), NOT build_ksweep_mask — surgery requires a uniform count for a single num_experts=224"
  - "AIMER scored as the mean across s0/s1/s2 merged seeds (v3 shared-profile convention, A1)"
  - "Ship criterion = pre-registered routing-(B) non-inferiority (ci_lower >= -2pp), distinct from the stricter two-sided TOST equivalence"
metrics:
  duration_min: 95
  completed: 2026-07-17
  tasks: 3
  files: 7
---

# Phase 26 Plan 01: Stacked-Tensor AIMER + Gate-Before-Remove Eval Summary

Adapted the v3 prune stack to v4's stacked-tensor MoE layout, scored AIMER at the single authorized compression point k=224, and ran the judge-only gate-before-remove eval on the same patched vLLM — recording an honest s1 TOST vs the same-stack Gate B full arm plus D2_security retention, with **no weight physically removed**.

## What was built

- **`scripts/aimer_prune_v4.py`** — AIMER `P/sqrt(N*Q)` (unchanged, scale-invariant, bounded) re-targeted at v4's stacked `experts.gate_up_proj [256,1024,2048]` + `experts.down_proj [256,2048,512]` via an axis-0 reduction. Prefix (`model.language_model`) is derived from the index weight_map and asserted; the `mtp.layers.*` block is excluded; a missing/renamed expert key raises (Pitfall 1). Main() also runs SC1 merge-of-record verification, builds the k=224 uniform mask, and pins the protected-mask sha256.
- **`scripts/prune_gate_v4.py`** — judge-only gate-before-remove driver: serve/capture/score re-wiring of `sieve_ksweep_v4_run.py` pointed at ONE mask. Same-stack TOST vs the Gate B full arm (0.7935); D2_security via reused `_d2_security_mean`; protected-mask sha256 re-verified vs the v4 manifest before any serve. Imports no v3 fixed floors (goalpost-move guard).

## Measured gate result (k=224, AIMER)

| Metric | Value | Bar | Verdict |
|--------|-------|-----|---------|
| s1 rho | **0.8184** | full-arm 0.7935 | +0.0280 point-better |
| TOST two-sided | ci [−0.0190, +0.0768], `equivalent:false` | ⊂ (−2pp,+2pp) | fails (UPPER bound — arm is better) |
| Non-inferiority | ci_lower −0.0190 ≥ −0.020 | ci_lower ≥ −2pp | **non-inferior** (slack +0.0010, thin) |
| D2_security | retention 6.326 ≥ baseline 6.115 | within 0.02pp | **pass_d2_security:true** |
| protected_retained | true | — | pass |
| parse_fail | 1/121 | — | clean (no collapse) |
| **pass_ship** | **true** | routing-(B) non-inferiority | **ship** |

AIMER@224 did **not** reproduce v3's parse-collapse failure mode (parse_fail 1/121, rho 0.8184) — a genuinely different outcome from v3's AIMER@25 (rho 0.165, parse 0.446). The mask patch resolved `qwen3_next.Qwen3NextSparseMoeBlock` and applied 224/256 per layer (T-26-03 fail-loud satisfied).

## Deviations from Plan

### Auto-fixed / corrected

**1. [Rule 1 — Correctness] k=224 mask built with `build_uniform_keep_mask`, not `build_ksweep_mask`**
- **Found during:** Task 1.
- **Issue:** The plan names `build_ksweep_mask` (top-k UNION protected → ≥k, non-uniform per layer). The done-criterion "exactly 224/layer" AND 26-02's physical surgery (a single `text_config.num_experts=224` requires the same kept-count in every layer) both need a **uniform** mask. `build_ksweep_mask` cannot guarantee that (a protected expert outside the top-224 pushes a layer above 224).
- **Fix:** Used `build_uniform_keep_mask` (already in `prune_apply_physical.py`; also never drops a protected expert; feasible because max_protected/layer=98 ≤ 224). Result: exactly 224/layer, protected retained — satisfies the plan's verify (min==224) and the surgery contract.
- **Files:** `scripts/aimer_prune_v4.py`. **Commit:** eda07ea.

**2. [Ship-criterion clarification, coordinator-relayed] Added code-computed `pass_ship` (non-inferiority)**
- The plan gated `pass` on two-sided TOST equivalence. The pre-registered routing-(B) ship bar (25-02 sign-off, reaffirmed via orchestrator relay) is **non-inferiority** (ci_lower ≥ −2pp) AND D2 retained AND protected retained. Added a distinct, code-computed `pass_ship` while keeping the measured two-sided `pass:false` / `equivalent:false` intact. Not a goalpost move — non-inferiority was always the pre-registered criterion; two-sided TOST is the stricter secondary metric.
- **Files:** `scripts/prune_gate_v4.py`. **Commit:** (gate-run commit).

## Verification

- `aimer_prune_v4 --self-check`: scale-invariant, deterministic, bounded, mtp-excluded, missing-key raises — PASS.
- `prune_gate_v4 --self-check`: pass on equivalent+D2-retained, pass_ship on non-inferior point-better, fail on collapse/D2-regression; `_d2_security_mean` exercised — PASS.
- SC1 merge-of-record: no adapter files, arch `Qwen3_5MoeForConditionalGeneration`, num_experts=256, num_hidden_layers=40 — confirmed.
- k=224 mask: min==max==224/layer, protected retained — confirmed.

## Self-Check: PASSED
- Created files exist: aimer_prune_v4.py, prune_gate_v4.py, aimer_scores_judge_v4.npy, masks/aimer_k224.npy, protected_manifest_v4.json, gated/aimer_224_judge.json, gated/aimer_224_d2.json — all present.
- Commits exist: eda07ea, 5dc87af, + gate-run commit — all in git log.
