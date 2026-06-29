# seedA live-RL controlled-eval reads (_check_judge_fixcorr, n=40/policy, temp 0.2)
# PASS = fixed meaningfully > warm AND stale ~= warm. Binding flat-gate at step 250.

| read | step | warm | fixed | Δ(fix-warm) | stale | Δ(stale-warm) | echo | verdict |
|------|------|------|-------|-------------|-------|---------------|------|---------|
| pre  |  15  | 0.3315 | 0.2750 | -0.057 | 0.3718 | +0.040 | 0.25 | NULL (pre-run V4 smoke) |
| #1   |  50  | 0.3252 | 0.3512 | +0.026 | 0.3750 | +0.050 | 0.25 | NULL → continue |
| #2   | 100  | 0.3252 | 0.3575 | +0.032 | 0.3812 | +0.056 | 0.25 | CONFOUNDED (stale>warm); within-run fixed: s50 0.351 -> s100 0.358 = +0.006 FLAT |
| #3   | 150  | 0.3190 | 0.3937 | +0.075 | 0.3812 | +0.062 | 0.25 | CONFOUNDED-flag but fixed>stale now; within-run fixed: s50 0.351 -> s100 0.358 -> s150 0.394 CLIMBING |
| #4   | 200  | 0.3500 | 0.3689 | +0.019 | 0.3628 | +0.013 | 0.25 | NULL; within-run fixed: 0.351/0.358/0.394/0.369 = noisy ~0.37, s150 was a spike |
| #5   | 250  | 0.3281 | 0.4127 | +0.085 | 0.3941 | +0.066 | 0.25 | n=80 BINDING; fixed-250 TOPS warm+stale; within-run n=80: fixed-50 0.385 -> fixed-250 0.413 = +0.027 LEARNING (not flat) |
| #6   | 300  | 0.3501 | 0.4098 | +0.060 | 0.4144 | +0.064 | 0.25 | n=80 TREND (post-binding); CONFOUNDED-label moot (stale=known J.4 artifact); within-run n=80: fixed-50 0.385 -> 250 0.413 -> 300 0.410 = SUSTAINED +0.025 over fixed-50, not collapsing |
| #7   | 350  | 0.3297 | 0.4127 | +0.083 | 0.4035 | +0.074 | 0.25 | n=80 TREND; fixed-350 TOPS stale; within-run n=80: 0.385 -> 250 0.413 -> 300 0.410 -> 350 0.413 = SUSTAINED +0.028, stable |
| #8   | 400  | 0.3314 | 0.4098 | +0.078 | 0.4039 | +0.072 | 0.25 | n=80 TREND; fixed TOPS stale; within-run n=80: 0.385 -> 250 0.413 -> 300 0.410 -> 350 0.413 -> 400 0.410 = SUSTAINED +0.025, dead stable (reward_mean step-400=0.451 up) |
| #9   | 450  | 0.3314 | 0.4034 | +0.072 | 0.4033 | +0.072 | 0.25 | n=80 TREND; fixed=stale (tied); within-run n=80: ...400 0.410 -> 450 0.403 = plateau ~0.41 (+0.018 over fixed-50) |
| #10  | 500  | 0.3252 | 0.4131 | +0.088 | 0.4193 | +0.094 | 0.25 | n=80 FINAL; within-run n=80: 0.385 -> 250 0.413 -> 300 0.410 -> 350 0.413 -> 400 0.410 -> 450 0.403 -> 500 0.413 = ENDPOINT +0.028 over fixed-50, plateau held. Training COMPLETE (step-500 + final-step-500 ckpts saved, clean exit). |

## RESUME (context exhausted 2026-06-28)
- seedA RL run LIVE: PID in output/rl_checkpoints/metrics/rl_run.seedA.pid; ~step 255+, healthy; watcher to step 300 running. Metrics: rl_metrics.seedA.jsonl; manifest has ckpts 50/100/150/200/250.
- BINDING (step 250) PENDING a confirmatory read at seed 99999: logs/phase09_rerun/read5c_step250_seed99999.log + read5d_step50_seed99999.log. Compute fixed-250 minus fixed-50.
  - seed 12345 (n=80): fixed-50=0.3854, fixed-250=0.4127 -> +0.027 (weak-positive; fixed-250 tops warm+stale; echo 0.25 PASS).
  - If seed-99999 late-early replicates ~+0.02-0.03 -> real weak win. If ~0 -> artifact -> STOP.
- DECISION FORK (advisor-reframed): (1) push seedA to 500 (slow grind, KL pinned ~0.009, proj ~+0.03 more); OR (2) STOP at 250 + restart higher LR (instrument's standing rec: LR 1e-05 too conservative). Do NOT launch seedB to confirm a marginal signal.
- Env: `set -a; . ./.env; set +a; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN; PY=.venv-tinker/bin/python`. Gate plan: 09-POST-V6-HANDOVER.md §9.

## BINDING GATE RESOLVED (step 250) -> PUSH TO 500
- Confirmatory read seed 99999 (n=80): fixed-50=0.3553, fixed-250=0.4105 -> late-early=+0.055.
- seed 12345 (n=80): late-early=+0.027. BOTH seeds positive (mean ~+0.04); fixed-250 tops warm(~+0.08)+stale on both; echo 0.25 PASS.
- VERDICT: learning is REAL (not a seed artifact). Gate = NOT FLAT -> CONTINUE seedA to 500. No seedB.
- ACTION NEEDED: none to continue (seedA already training past 250). Next context: keep monitoring (reads every 50 at 300/350/400/450/500), then run final RLEV-01/02 gate (wp-bench + anti-hack) on step-500 vs v1.2 SFT. Phase 10 Task-3 gate is now satisfiable: real checkpoints exist in manifest.seedA.json.
