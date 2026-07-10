#!/usr/bin/env bash
# launch_smoke_seedA2.sh — Phase 08.2 gated smoke RERUN (2026-07-02)
#
# WHY A RERUN: the 2026-07-01 seedA smoke never tested hybrid@0.8 — the GT
# hash-join was dead (raw vs normalized hashing, 0/482) so the calib term never
# fired and the run trained on pure fix_correctness. Fixed + loud-fail wired
# (commit "fix(08.2 reward): GT hash-join was DEAD..."). This rerun is the FIRST
# actual test of the calibration reward.
#
# Deltas vs the seedA launch (SMOKE_READS_TALLY.md config):
#   --codegen-probe-every 0   EXPLICITLY DISARMED — the in-run probe has no
#                             merged-model dir to serve (auto-merge not wired);
#                             with the new loud-fail semantics an armed-but-
#                             misconfigured probe HALTS. Gate-2 is read manually
#                             on the step-50 sampler (_rlev01_probe_ckpt.sh)
#                             IFF Gate-1 passes.
#   isolated outputs          output/rl_checkpoints/smoke_seedA2/
#   consistency               ANTHROPIC keys unset -> weight forced to 0 (loud,
#                             logged) instead of 0.45 x dead-zero.
#
# Step-0 gate additions (loud-fail wiring): run HALTS itself with
# CALIB_JOIN_DEAD if calib_fired_frac < 0.5 at step 0. Expect the log line
# "GT-coverage filter ... judge pool 482 -> 342".
#
# Kill-at-50 (unchanged, 08.2-SMOKE-RUNBOOK.md §3-4): teacher-Spearman
# rho_50 > rho_initial(0.6243) + 0.02 with bootstrap CI-lower > 0, echo <= 0.30,
# manual wp-bench >= 0.4616. Any fail -> kill, no 250 push.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "${1:-}" != "--i-understand-this-spends-gpu" ]; then
    echo "DRY-PRINT. Pass --i-understand-this-spends-gpu to launch."
    DRY=1
else
    DRY=0
fi

set -a; . ./.env; set +a
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN || true
export PYTHONPATH=.

INIT_FROM=$(python3 -c "import json;print(json.load(open('output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json'))['state_path'])")
OUT=output/rl_checkpoints/smoke_seedA2
mkdir -p "$OUT/metrics"

CMD=(.venv-tinker/bin/python scripts/rl_train.py
    --init-from "$INIT_FROM"
    --lora-seed 12345
    --total-steps 250
    --checkpoint-every 50
    --codegen-probe-every 0
    --calib-form hybrid
    --calib-weight 0.8
    --judge-base-url http://localhost:8000/v1
    --judge-max-new-tokens 4096
    --manifest-path "$OUT/checkpoint_manifest.json"
    --metrics-path "$OUT/metrics/rl_metrics.jsonl"
)

echo "INIT_FROM: $INIT_FROM"
echo "CMD: ${CMD[*]}"
if [ "$DRY" = "1" ]; then exit 0; fi

curl -s http://localhost:8000/v1/models | grep -q wp_judge || {
    echo "ERROR: judge :8000 not serving wp_judge" >&2; exit 1; }

nohup setsid "${CMD[@]}" > "$OUT/full_run.log" 2>&1 &
echo "PID: $!"
echo "Log: $OUT/full_run.log"
