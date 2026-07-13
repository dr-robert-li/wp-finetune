#!/usr/bin/env bash
# Phase 20 (v4.0 base bring-up) vLLM server for Qwen3.6-35B-A3B (bf16, local
# download at models/Qwen3.6-35B-A3B by default).
#
# Modeled on scripts/serve_30_70_vllm.sh (same docker-run-direct pattern,
# bypassing sparkrun; same no --rm / docker-logs-retrievable-on-crash
# behavior), but for the NEW v4 base: no hardcoded --served-model-name, no
# expert-mask docker-args block (that is a v3.0/Phase 25 concern, not this
# phase's).
#
# New (BASE-03) env toggles, both default UNSET so the first serve attempt
# runs WITH CUDA-graph capture enabled (Pitfall 2 — an eager-only smoke is a
# false pass for vLLM #35945):
#   LANGUAGE_MODEL_ONLY=1   append --language-model-only (serve the VL
#                           checkpoint text-only)
#   ENFORCE_EAGER=1         append --enforce-eager (documented fallback if
#                           CUDA-graph capture crashes)
#
# Idempotent: stops any previous same-named container before relaunch.
#
# Usage:
#   bash scripts/serve_base20_vllm.sh
#   LANGUAGE_MODEL_ONLY=1 bash scripts/serve_base20_vllm.sh
#   LANGUAGE_MODEL_ONLY=1 ENFORCE_EAGER=1 bash scripts/serve_base20_vllm.sh
#   MODEL_DIR=/path/to/other/checkpoint bash scripts/serve_base20_vllm.sh
#
# Stop with: docker rm -f base20-vllm (or $CONTAINER_NAME)

set -euo pipefail

NAME="${CONTAINER_NAME:-base20-vllm}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
# Optional --served-model-name override. The wp-bench harness
# (run_eval_reasoning._run_wpbench) hardcodes model name wp-30_70 (the name
# serve_30_70_vllm.sh always sets). Default UNSET preserves this script's
# existing served identity (/workspace/model) for all prior callers.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/Qwen3.6-35B-A3B}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

EXTRA_ARGS=()
if [ -n "$LANGUAGE_MODEL_ONLY" ]; then
  EXTRA_ARGS+=(--language-model-only)
  echo "  language-model-only: serving VL checkpoint text-only"
fi
if [ -n "$ENFORCE_EAGER" ]; then
  EXTRA_ARGS+=(--enforce-eager)
  echo "  enforce-eager: CUDA-graph capture DISABLED (documented fallback path)"
fi
if [ -n "$SERVED_MODEL_NAME" ]; then
  EXTRA_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
  echo "  served-model-name: $SERVED_MODEL_NAME"
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: model dir not found: $MODEL_DIR" >&2
  echo "Expected local checkpoint at: $MODEL_DIR" >&2
  echo "Override with MODEL_DIR=<path> ..." >&2
  exit 1
fi

# Stop any previous instance
if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model: $MODEL_DIR (bf16, local)"
echo "  image: $IMAGE"
echo "  gpu-memory-utilization: $GPU_MEM_UTIL"

# NOTE: no --rm — a crashed boot must leave the container behind so
# `docker logs` is retrievable. Cleanup is already handled: this script
# rm -f's any previous $NAME above, and every caller's stop_vllm() does
# `docker rm -f` when done.
docker run -d \
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
    "${EXTRA_ARGS[@]}" \
    -tp 1 -pp 1

echo ""
echo "Container started. Follow weight-load + readiness:"
echo "  docker logs -f $NAME"
echo ""
echo "When you see 'Application startup complete' + 'Uvicorn running on http://0.0.0.0:${PORT}':"
echo "  curl -s http://localhost:${PORT}/v1/models"
echo ""
echo "Stop with:"
echo "  docker rm -f $NAME"
