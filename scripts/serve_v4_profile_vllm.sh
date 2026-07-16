#!/usr/bin/env bash
# vLLM server for Phase 25 ROUTING PROFILE of the v4 judge (Qwen3.6-35B-A3B,
# 256 experts). Serves the merged judge so vLLM's own memory manager loads the
# weights (fits the 121 GiB GB10 pool) instead of an in-process from_pretrained
# (which OOMs -- .planning/debug/v4-judge-load-oom-recurrence.md). The
# _sieve_profile_vllm_patch hook (bind-mounted via PYTHONPATH) accumulates
# per-layer/per-expert top-k selection counts during serving and dumps them to
# the mounted output path. Drive it with scripts/drive_v4_routing_profile.py.
#
# THREE non-negotiable serve flags for a CORRECT profile:
#   --enforce-eager        hook is Python; CUDA-graph replay would skip it.
#   --language-model-only  the judge is a VL checkpoint; serve text-only.
#   NO --enable-prefix-caching : a cached prefix skips recomputation, so its
#                          tokens never route again -> undercounted experts.
#
# Usage:
#   SIEVE_PROFILE_OUT=/abs/host/dir PROFILE_OUT_NAME=routing_counts_full.npy \
#     bash scripts/serve_v4_profile_vllm.sh
# Ready:  curl -s http://localhost:${PORT}/v1/models
# Stop:   docker stop wp-v4-profile-vllm   (triggers the patch's final flush)

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-v4-profile-vllm}"
PORT="${PORT:-8010}"
# Weights ~67 GiB resident, KV tiny (profiling is prefill-only, max_tokens=1),
# so we can hand vLLM most of the pool. Override if other services are resident.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"   # matches profile_v4_judge.py max_seq_len
# Decoupled from MAX_MODEL_LEN: prefill-only profiling batches many prompts per
# engine step, so a larger batched-token budget is the throughput lever for a
# big stimulus (e.g. the 34,855-example ratio_30_70 set). Default = MAX_MODEL_LEN
# (1 long prompt/step, serialized) unless raised.
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-$MAX_MODEL_LEN}"
SERVED_NAME="${SERVED_NAME:-judge-v4-s1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/Qwen3.6-35B-A3B-judge-v4-s1-merged}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

# Host dir that receives the counts .npy dump (bind-mounted rw to /sieve_out).
PROFILE_OUT_DIR="${SIEVE_PROFILE_OUT:-$REPO_ROOT/output/sieve-v4/profile}"
PROFILE_OUT_NAME="${PROFILE_OUT_NAME:-routing_counts.npy}"
PROFILE_TOPK="${SIEVE_PROFILE_TOPK:-8}"   # MUST match profile_v4_judge top_k_jaccard

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: v4 judge model dir not found: $MODEL_DIR" >&2
  exit 1
fi
mkdir -p "$PROFILE_OUT_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT (ROUTING PROFILE)"
echo "  model:  $MODEL_DIR"
echo "  patch:  $REPO_ROOT/scripts/_sieve_profile_vllm_patch -> /sieve_profile"
echo "  counts: $PROFILE_OUT_DIR/$PROFILE_OUT_NAME  (top_k=$PROFILE_TOPK)"
echo "  flags:  --enforce-eager --language-model-only  (prefix-caching OFF)"

# NOTE: no --rm — a crashed boot must leave the container for `docker logs`.
docker run -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/workspace/model:ro" \
  -v "${REPO_ROOT}/scripts/_sieve_profile_vllm_patch:/sieve_profile:ro" \
  -v "${PROFILE_OUT_DIR}:/sieve_out" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYTHONPATH=/sieve_profile \
  -e SIEVE_PROFILE_OUT="/sieve_out/${PROFILE_OUT_NAME}" \
  -e SIEVE_PROFILE_TOPK="$PROFILE_TOPK" \
  "$IMAGE" \
  vllm serve /workspace/model \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --enforce-eager \
    --language-model-only \
    --served-model-name "$SERVED_NAME" \
    -tp 1 -pp 1

echo ""
echo "Follow readiness:  docker logs -f $NAME   (look for '[sieve-profile] patched ...')"
echo "Ready when:        curl -s http://localhost:${PORT}/v1/models"
echo "Then drive:        python scripts/drive_v4_routing_profile.py --port ${PORT} ..."
echo "Stop (flush):      docker stop $NAME"
