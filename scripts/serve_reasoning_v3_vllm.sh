#!/usr/bin/env bash
# Phase 4.4 vLLM server for the wp-reasoning-v3 merged-staging model (plan 04.4-02).
#
# Serves models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3 — the anchor-certified
# Tinker v3 merge (plan 01), NOT the OLD ckpt-72 canonical. v3 is directly servable:
# its adapter had modules_to_save=null (unlike the v1 30_70 adapter, which trained the
# <wp_gen>/<wp_judge> embeddings via modules_to_save and could not be a runtime LoRA).
# v3 was trained on the STOCK tokenizer (task markers are plain text), so the staging dir's
# stock tokenizer is the correct one to serve — do NOT override --tokenizer.
#
# Thinking: v3 was trained with the Tinker `qwen3_disable_thinking` renderer. Qwen3 vLLM
# enables thinking by default; the fidelity gate sends chat_template_kwargs enable_thinking=false
# per-request AND applies strip_think_blocks before scoring (robust double-guard).
#
# Idempotent: stops any previous "wp-reasoning-v3-vllm" container before relaunch.
#
# Usage:   bash scripts/serve_reasoning_v3_vllm.sh           # default port 8021
#          PORT=8021 bash scripts/serve_reasoning_v3_vllm.sh
# Stop:    docker stop wp-reasoning-v3-vllm

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-reasoning-v3-vllm}"
PORT="${PORT:-8021}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: v3 staging model dir not found: $MODEL_DIR" >&2
  echo "Run plan 04.4-01 (scripts/_04.4_run_merge_v3.py) first." >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model: $MODEL_DIR (v3 merged staging)"
echo "  served-model-name: wp-reasoning-v3"
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
    --served-model-name wp-reasoning-v3 \
    -tp 1 -pp 1

echo ""
echo "Follow readiness:  docker logs -f $NAME"
echo "Ready when:        curl -s http://localhost:${PORT}/v1/models"
echo "Stop:              docker stop $NAME"
