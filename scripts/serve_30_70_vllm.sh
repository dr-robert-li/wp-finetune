#!/usr/bin/env bash
# Phase 0.3 vLLM server for the 30/70 fine-tune.
#
# Bypasses sparkrun (DGX_TOOLBOX_ISSUES.md #8 + #9) by docker-running the
# same prebuilt image sparkrun would use.
#
# Serves the PRE-MERGED 30/70 checkpoint (models/qwen3-30b-wp-30_70-merged/).
# Why merged instead of base + --enable-lora:
#   vLLM rejects PEFT adapters that have non-None modules_to_save with
#   "vLLM only supports modules_to_save being None". Our adapter trained
#   the <wp_gen> / <wp_judge> task token embeddings via modules_to_save=
#   [embed_tokens, lm_head], so it cannot be served as a runtime LoRA.
#   The merged checkpoint has these embeddings baked into the base weights.
#
# Caveat: the merge step ran with raw PEFT (not unsloth's FastLanguageModel),
# so the MoE expert LoRA (target_parameters=[...]) was NOT merged — same
# silent-fail mode as the raw-PEFT loader (DGX_TOOLBOX_ISSUES.md #7).
# Result: served model has trained embed_tokens + attention LoRA + lm_head,
# but BASE expert MLPs. Partial 30/70. Good enough for Phase 0.3 signal.
# True-30/70 quality needs Path B (FastLanguageModel-based eval).
#
# Idempotent: stops any previous "wp-30_70-vllm" container before relaunch.
#
# Usage:
#   bash scripts/serve_30_70_vllm.sh              # default port 8001
#   PORT=9001 bash scripts/serve_30_70_vllm.sh
#   MODEL_DIR=/path/to/other/merged bash scripts/serve_30_70_vllm.sh
#
# Stop with: docker stop wp-30_70-vllm

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-30_70-vllm}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/qwen3-30b-wp-30_70-merged}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "Expected merged checkpoint at: $MODEL_DIR" >&2
  echo "Override with MODEL_DIR=<path> ..." >&2
  exit 1
fi

# Stop any previous instance
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model: $MODEL_DIR (merged)"
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
    --served-model-name wp-30_70 \
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
