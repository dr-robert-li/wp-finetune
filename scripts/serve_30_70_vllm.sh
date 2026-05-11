#!/usr/bin/env bash
# Phase 0.3 vLLM server for Qwen3-30B-A3B base + 30/70 LoRA adapter.
#
# Bypasses sparkrun because sparkrun cannot serve local model directories
# (DGX_TOOLBOX_ISSUES.md#8 + #9). Pulls the same prebuilt image sparkrun
# would use and `docker run` it ourselves.
#
# Idempotent: stops any previous "wp-30_70-vllm" container before relaunch.
#
# Usage:
#   bash scripts/serve_30_70_vllm.sh              # default port 8001
#   PORT=9001 bash scripts/serve_30_70_vllm.sh
#
# Stop with: docker stop wp-30_70-vllm

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-30_70-vllm}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/Qwen3-30B-A3B}"
ADAPTER_DIR="${ADAPTER_DIR:-$REPO_ROOT/adapters/qwen3-30b-wp-30_70}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  exit 1
fi
if [ ! -d "$ADAPTER_DIR" ]; then
  echo "ERROR: adapter dir not found: $ADAPTER_DIR" >&2
  exit 1
fi

# Stop any previous instance
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model:   $MODEL_DIR"
echo "  adapter: $ADAPTER_DIR"
echo "  image:   $IMAGE"

docker run --rm -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/workspace/model:ro" \
  -v "${ADAPTER_DIR}:/workspace/adapter:ro" \
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
    --enable-lora \
    --lora-modules "wp-30_70=/workspace/adapter" \
    --max-lora-rank "$MAX_LORA_RANK" \
    -tp 1 -pp 1

echo ""
echo "Container started. Follow weight-load + readiness:"
echo "  docker logs -f $NAME"
echo ""
echo "When you see 'Application startup complete' + 'Uvicorn running on http://0.0.0.0:${PORT}':"
echo "  curl -s http://localhost:${PORT}/v1/models"
echo ""
echo "Stop with:"
echo "  docker stop $NAME"
