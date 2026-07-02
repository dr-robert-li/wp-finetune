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

---

## ADDENDUM 2026-07-02 — verdict RE-LABELED (root cause found: calib never fired)

Code review found the 07-01 smoke's GT hash-join was DEAD: the reward path hashed RAW
original code (`rl_rollouts.py:1092`) while `build_reward_gt_sidecar.py` hashed
whitespace-NORMALIZED code → join 0/482 → `calib_r=NaN` for every completion →
`augment_judge_scalar` fell back to the raw judge scalar. With the consistency scorer
also dead (ANTHROPIC keys unset → 0.0 at 0.45 weight), the run's effective reward was
**pure fix_correctness** — the exact reward Phase 10 proved Goodharts. Proof: raw-hash
join = 0/482; normalized-hash join = 342/482.

**Re-label:** this tally's kill verdict is "pure-fc re-confirmed Goodhart (third time),"
NOT "hybrid@0.8 failed to move the metric." hybrid@0.8 was never in the loss and remains
UNTESTED by this run. The observed signature (fc +0.025, teacher-Spearman flat,
ENTROPY_COLLAPSE) is exactly what pure-fc training predicts.

Fixes + loud-fail wiring shipped (commit "fix(08.2 reward): GT hash-join was DEAD…"):
canonical `normalized_code_hash` both sides, GT-coverage pool filter (482→342),
per-step calib telemetry, step-0 CALIB_JOIN_DEAD halt, codegen trip-wire
misconfiguration halt (+ `--codegen-probe-model-dir` flag), explicit consistency
weight-0 when unkeyed. Rerun: `output/rl_checkpoints/smoke_seedA2/` (2026-07-02) —
the FIRST actual test of hybrid@0.8, same kill-at-50 discipline, ρ_initial 0.6243.

---

## seedA2 READ (2026-07-03) — the FIRST honest hybrid@0.8 test: **KILL (G1 fail)**

**Run:** `output/rl_checkpoints/smoke_seedA2/` (2026-07-02 08:40 → 2026-07-03 ~07:54,
died silently at step 161; kill verdict already earned at the step-50 read).
Config: seedA-equivalent + hash-join fix + loud-fail wiring; `--codegen-probe-every 0`
(explicitly disarmed, manual Gate-2 planned); consistency weight 0 (loud, by design).

**Step-0 gate: ALL GREEN, calib IN THE LOSS for the first time** — calib_fired_frac 0.90,
calib_mean 0.732±0.197, reward_mean 0.447 (vs 0.146 calib-dead on 07-01), WARM START,
GT-coverage filter 482→342.

**Process breach (recorded):** the step-50 sentinel checked `checkpoint['step']==50` but the
manifest keys checkpoints by `name` — sentinel never fired, the run trained past the gate to
step 161 before the read happened (~110 steps of unadjudicated spend). The banked step-100/150
checkpoints turned part of that spend into a trend read.

**Gate-1 (mechanical, n=86 common aligned, full capture n=121/117 parseable):**
| ckpt | rho | Δ vs 0.6243 | bootstrap CI | beyond_noise |
|---|---|---|---|---|
| step-50 | 0.6066 | −0.018 | [−0.130,+0.082] | false ❌ |
| step-150 | 0.6434 | +0.019 | [−0.059,+0.099] | false ❌ |

**KILL at the 50-read** (rho DOWN −0.018). The 150 trend read is informational: direction
reversed (+0.037 from 50→150) but still under the +0.02 bar with CI spanning zero.

**Interpretation — the load-bearing difference from 07-01:** this run's reward was HONEST:
fix_correctness flat (0.28–0.34, window-means 0.286/0.298/0.302/0.284), entropy flat
(~0.41, no collapse), calib fired 0.75–1.0 every step. No Goodhart signature. So the verdict
is a REAL negative on hybrid@0.8-as-configured: the oracle-valid calibration signal, honestly
wired, is too weak/slow to move validated teacher-Spearman beyond noise in 50 (or 150) steps.
Parse rate drifted 117→109 (mild format degradation). G2 codegen NOT read (moot for the kill;
required before any future continuation). G3 echo not read (moot).

**Disposition:** kill-at-50 discipline upheld (read late due to sentinel bug, verdict applied
on read). hybrid@0.8 has now had its real test: FAILED TO DEMONSTRATE within the gated
budget. Continuing to ~500 steps on the +0.037 slope extrapolation is speculation the runbook
forbids ("no 500-step runs on faith"). Recommendation unchanged and now doubly supported:
**hold RL, ship v1.2 SFT for v3.0.** Any future RL attempt needs a reward with materially
stronger per-step signal (e.g. grounded defect-detection terms / MO-GRPO separation per
09-REWARD redesign notes), not more steps of this one.

All processes dead; judge stopped (GPU freed). Captures: `output/rl_eval/step-{50,150}-seedA2/`,
07-01 captures archived to `output/rl_eval_seedA1_0701/`.
