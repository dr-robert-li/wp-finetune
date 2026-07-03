#!/usr/bin/env bash
# Phase C gated RL smoke — calib-ONLY reward from v1.3 warm-start (B2 CONDITIONAL-GO).
#
# B2 conditions implemented (output/relabel/B2_REVIEW.md, 2026-07-04):
#   - reward = pure calibration (--calib-form calibration --calib-weight 1.0);
#     the defect stream anti-correlated offline and is NOT used.
#   - GT = relabel sidecar v2 (REWARD_SIDECAR_PATH; loader accepts train_* provenance).
#   - warm start = v1.3 (s1-ss final-state) — never fresh-LoRA into RL (JOURNAL L127).
#   - drift trip-wire: rl_metrics.jsonl calib telemetry (fired/mean/std/n per step)
#     watched by the runner; halt if calib_mean drifts > 0.15 from step-0 or
#     calib_std collapses < 0.02 (score-distribution drift / discrimination collapse).
#   - G1 at reads (50/100/150): paired-delta-rho vs warmstart, CI-lower > 0
#     (B3: point bars under 0.05 are sub-2SE noise at n=121).
#   - G2: wp-bench >= A4 v1.3 bar (output/rl_eval/wpbench_v1.3/wpbench_result.json).
#   - codegen trip-wire ARMED via --codegen-probe-model-dir (the seedA2 gap).
#
# Usage: launch_smoke_v13_calib.sh <seed> [--i-understand-this-spends-gpu]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED="${1:?need seed (e.g. 101 / 202)}"
if [ "${2:-}" != "--i-understand-this-spends-gpu" ]; then
    echo "DRY-PRINT. Pass --i-understand-this-spends-gpu to launch."
    DRY=1
else
    DRY=0
fi

set -a; . ./.env; set +a
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN || true
export PYTHONPATH=.
export REWARD_SIDECAR_PATH=data/relabel_v1/judge_gt_sidecar_v2.jsonl

INIT_FROM=$(python3 -c "import json;print(json.load(open('output/tinker/wp-reasoning-relabel-s1-ss-manifest.json'))['state_path'])")
MERGED=models/_staging/qwen3-30b-wp-v1.3-merged
OUT="output/rl_checkpoints/smoke_v13_calib_seed${SEED}"
mkdir -p "$OUT/metrics"

[ -f "$MERGED/config.json" ] || { echo "ERROR: merged v1.3 missing ($MERGED) — run _rlev01_probe_any.sh first (A3)"; exit 1; }
[ -f output/rl_eval/wpbench_v1.3/wpbench_result.json ] || { echo "ERROR: A4 bar missing — wp-bench v1.3 first"; exit 1; }

CMD=(.venv-tinker/bin/python scripts/rl_train.py
    --init-from "$INIT_FROM"
    --lora-seed "$SEED"
    --total-steps 200
    --checkpoint-every 50
    --codegen-probe-model-dir "$MERGED"
    --calib-form calibration
    --calib-weight 1.0
    --judge-base-url http://localhost:8000/v1
    --judge-max-new-tokens 4096
    --manifest-path "$OUT/checkpoint_manifest.json"
    --metrics-path "$OUT/metrics/rl_metrics.jsonl"
)

echo "INIT_FROM: $INIT_FROM"
echo "SIDECAR: $REWARD_SIDECAR_PATH"
echo "CMD: ${CMD[*]}"
if [ "$DRY" = "1" ]; then exit 0; fi

curl -s http://localhost:8000/v1/models | grep -q wp_judge || {
    echo "ERROR: judge :8000 not serving wp_judge" >&2; exit 1; }

nohup setsid "${CMD[@]}" > "$OUT/full_run.log" 2>&1 &
echo "PID: $!"
echo "Log: $OUT/full_run.log"
