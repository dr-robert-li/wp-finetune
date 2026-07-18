#!/usr/bin/env bash
# Post-8.1 RL rerun launcher — one seed per invocation.
#   bash scripts/_launch_post81_rerun.sh <seedNum> <suffix>   e.g. 42 seedA / 7 seedB
# Env hygiene: source .env for TINKER_API_KEY, then UNSET Anthropic keys so the
# $0 local-vLLM consistency/judge path can never leak into paid API (billing rule).
set -uo pipefail

SEED="${1:?usage: <seedNum> <suffix>}"
SUF="${2:?usage: <seedNum> <suffix>}"

cd "$(dirname "$0")/.."
MDIR=output/rl_checkpoints/metrics
mkdir -p "$MDIR" logs/phase09_rerun

set -a; . ./.env; set +a
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

export WP_JUDGE_DEBUG_DUMP="$MDIR/judge_failures.$SUF.jsonl"

LOG="logs/phase09_rerun/full_run.$SUF.log"
PIDF="$MDIR/rl_run.$SUF.pid"

nohup .venv-tinker/bin/python scripts/rl_train.py \
  --init-from "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state" \
  --model-id Qwen/Qwen3-30B-A3B \
  --lora-rank 32 \
  --lora-seed "$SEED" \
  --total-steps 500 \
  --batch-size 8 \
  --checkpoint-every 50 \
  --jaccard-every 20 \
  --kl-soft 0.1 --kl-hard 0.3 \
  --efrac-soft 0.7 --efrac-hard 0.5 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  --judge-max-new-tokens 4096 \
  --consistency-base-url http://localhost:8001/v1 --consistency-model wp_consistency \
  --metrics-path "$MDIR/rl_metrics.$SUF.jsonl" \
  --manifest-path "$MDIR/manifest.$SUF.json" \
  > "$LOG" 2>&1 &

echo $! > "$PIDF"
echo "launched $SUF (seed $SEED) pid=$(cat "$PIDF") log=$LOG"
