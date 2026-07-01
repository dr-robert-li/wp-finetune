# Gated Smoke — seedA Step-50 Read Tally (Phase 08.2 Option B execution)

**Date:** 2026-07-01 · **Verdict: KILL at step 50** (Gate 1 fail — validated metric flat)

## Config
- Warm-start: regenerated v4 save_state `tinker://d59dea4e-…:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`
  (MoE-only, rank 32, 0% terse; Step-1 regen accepted).
- Reward: `hybrid` calib, `calib_weight=0.8` (top oracle-valid sweep entry; runbook candidate).
- Run: `rl_train.py --lora-seed 12345 --total-steps 250 --checkpoint-every 50 --codegen-probe-every 50
  --codegen-bar 0.4616 --judge-max-new-tokens 4096`, isolated outputs `output/rl_checkpoints/smoke_seedA/`.
- Judge: `wp_judge` vLLM @ :8000 (v4-winner merge). seedB NOT run (killed at seedA step-50 gate).

## Step-0 gate (Step 5) — PASS
WARM START confirmed (`train_mlp=True attn=False unembed=False`); judge_failures=0; fix_correctness_mean=0.3625≠0;
e_frac 0.963; kl 0.009. reward_mean 0.146 LOW but hybrid-form artifact (calib term ~0/neg pre-train), not a warm-start failure.

## Step-50 read (n=41 aligned; runbook §5 specifies n=40 per policy → meets the spec floor)
**n caveat:** the step-50 capture was repeatedly reaped under tinker contention; n=41 is where it stopped, not a
chosen sample size. It meets the runbook's n≈40 floor and is sufficient for the kill decision (see below), but the
bootstrap CI at n=41 is low-power — "includes zero" is corroborating, not the load-bearing reason for the kill.

**Baseline equivalence:** ρ_initial was measured on the ep3 **sampler_weights**
(`…/sampler_weights/wp-reasoning-v4-r32-rp30-savestate-ep3`) while the RL warm-started from the ep3 **final-state**
(`…/weights/…-savestate-final-state`) — the SAME end-of-epoch-3 model, different serialization (sampler vs loadable
state). So ρ_initial 0.6243 is the correct step-0 policy-judge baseline for this run.

Baseline ρ_initial = **0.6243** (per-ckpt) / 0.6212 (common set).

| Gate | Measure | Value | Bar | Result |
|---|---|---|---|---|
| **G1 teacher-Spearman** | step-50 (common n=41) | 0.6364 | — | — |
| | Δ vs ρ_initial (common) | **+0.0152** | > +0.02 | ❌ inside noise band |
| | bootstrap CI [lo,hi] n_boot=2000 | **[−0.0598, +0.0995]** | lo > 0 | ❌ includes zero |
| | `improved_beyond_noise` | **false** | true | ❌ |
| **G2 codegen trip-wire** | in-run probe | **NOT ENFORCED** | ≥0.4616 | ⚠️ silently skipped (see below) |
| **G3 echo-adversary** | — | not separately read | ≤0.30 | (moot — G1 already fails) |

**Corroborating Goodhart signature:** forbidden proxy `fix_correctness_mean` rose +0.025 (0.363→0.387) — MORE than
teacher-Spearman moved (+0.015); `REVIEW FLAG step 50: ENTROPY_COLLAPSE policy narrowing`; consistency 0.06→0.0.
This is the exact Phase-10 pattern (proxy up, validated metric flat) — caught at step 50, not 500.

## Gate-2 defect found (record for fix)
`rl_train.py:1067` calls `run_codegen_probe(model_dir=getattr(args,"codegen_probe_model_dir","."))`. No such CLI flag exists
/ was passed → model_dir=`"."` → probe returns None → `check_codegen_tripwire(None)` returns None (silent skip,
`rl_codegen_tripwire.py:64`). So `halt_reason: null` = "codegen NOT checked", NOT "passed". The in-run codegen trip-wire
was inert this run. To arm it, a `--codegen-probe-model-dir` (auto-merged step checkpoint) must be wired, or run the
heavy manual probe (`_rlev01_probe_ckpt.sh`, RUN id edited to this run) per checkpoint. (Moot here — G1 already killed.)

## Disposition
The ONE gated smoke (Option B) **did not clear the kill-at-50 bar**: hybrid@0.8 failed to move the validated
teacher-Spearman above warm-start noise (Δ+0.015 < +0.02) while the forbidden `fix_correctness` proxy rose MORE
(+0.025) — Goodhart-consistent. Note this is "failed to demonstrate it works in 50 steps + kill-at-50 triggered,"
NOT "hybrid@0.8 proven dead at scale" — a flat/ambiguous step-50 read is a KILL by construction (the runbook's whole
point), and the load-bearing evidence is the proxy/validated *divergence*, which is independent of capture-n. A
refined hybrid (calibration + a real codegen/anti-hack term) is NOT foreclosed by this result.

This REINFORCES Phase-08.2's conclusion (no offline-safe reward found) and the Phase-10 reject-RL verdict.
**Recommendation stands: hold RL, ship v1.2 SFT for v3.0.** Cost paid: v4 save_state regen (270 SFT steps) +
~52 RL steps + ρ_initial/step-50 captures. Kill-at-50 discipline worked as designed.
