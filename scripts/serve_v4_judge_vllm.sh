#!/usr/bin/env bash
# vLLM server for the v1.2 SFT winner used as the RL reward JUDGE (Phase 9 live run).
#
# Serves models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4 (the merge_v4_winner
# canonical, out_dir from output/merge_v4_winner/merge_report.json) under the served
# model name "wp_judge" — the name scripts/rl_train.py --judge-model defaults to and
# eval/eval_judge.judge_score_single passes to /v1/chat/completions.
#
# The judge call (eval_judge._judge_create) sends chat_template_kwargs enable_thinking=false
# per-request (RC-A guard), so no server-side thinking flag is needed.
#
# Idempotent: removes any previous "wp-v4-judge-vllm" container before relaunch.
#
# Usage:   bash scripts/serve_v4_judge_vllm.sh            # default port 8000
#          PORT=8000 bash scripts/serve_v4_judge_vllm.sh
# Ready:   curl -s http://localhost:8000/v1/models
# Stop:    docker stop wp-v4-judge-vllm

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-v4-judge-vllm}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_NAME="${SERVED_NAME:-wp_judge}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: v4 judge model dir not found: $MODEL_DIR" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model: $MODEL_DIR (v4 winner = judge)"
echo "  served-model-name: $SERVED_NAME"
echo "  image: $IMAGE"

docker run --rm -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/workspace/model:ro" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  "$IMAGE" \
  vllm serve /workspace/model \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --enable-prefix-caching \
    --served-model-name "$SERVED_NAME" \
    -tp 1 -pp 1

echo ""
echo "Follow readiness:  docker logs -f $NAME"
echo "Ready when:        curl -s http://localhost:${PORT}/v1/models"
echo "Stop:              docker stop $NAME"
