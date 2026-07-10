#!/usr/bin/env bash
# vLLM server for the RL score-reasoning CONSISTENCY judge (Phase 9 Option 1, $0).
#
# Serves nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (text-only A3B MoE, NVFP4)
# under the served name "wp_consistency" on :8001, so it coexists with the v4
# fix-scoring judge (wp_judge on :8000). scripts/rl_judge_dispatch.py points the
# consistency scorer here via --consistency-base-url http://localhost:8001/v1,
# replacing the paid `claude -p` path (D-09-05 / deprecated/planning-handoffs/09-HANDOFF.md Option 1).
#
# MODEL NOTE: the deprecated/planning-handoffs/09-HANDOFF.md named nvidia/Nemotron-3-Nano-Omni-...-NVFP4, but that
# is a VISION (Omni) multimodal model — wrong for a pure text 0-1 consistency score
# and heavier on the shared GB10 memory pool. This script uses the TEXT-ONLY sibling
# (NemotronHForCausalLM, no vision tower, 19.4GB). Both are vLLM-servable; the
# text variant was chosen 2026-06-22 (user-confirmed). See 09-LOCAL-RL-STATUS-UPDATES.md.
#
# REASONING PARSER: the recipe lists --reasoning-parser nemotron_v3, but the
# consistency call sends chat_template_kwargs enable_thinking=False (no <think> to
# parse) and rl_judge_dispatch._parse_consistency_score strips any leaked think
# tags. The parser is therefore omitted to avoid a startup failure if nemotron_v3
# is not registered in this vLLM build.
#
# MEMORY: GB10 is one shared (unified) memory pool. --gpu-memory-utilization is
# each process's fraction of TOTAL memory; two servers must sum < ~0.9. wp_judge
# (:8000) runs at 0.55, so this server defaults to 0.30 (0.55 + 0.30 = 0.85).
# NVFP4 weights are ~16GB, so 0.30 (~38GB on a 128GB pool) leaves ample KV room.
#
# Idempotent: removes any previous container of the same name before relaunch.
#
# Usage:   bash scripts/serve_consistency_vllm.sh           # default port 8001
#          PORT=8001 GPU_MEM_UTIL=0.30 bash scripts/serve_consistency_vllm.sh
# Ready:   curl -s http://localhost:8001/v1/models | grep -q wp_consistency
# Stop:    docker stop wp-consistency-vllm

set -euo pipefail

NAME="${CONTAINER_NAME:-wp-consistency-vllm}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
# 12288, not the recipe's 32768: realistic consistency prompt is rubric (~400t) + the
# judge's PHP completion (<=1536t) + critique (~900t) + 256t output ~= 3.2k tokens. A
# smoke-test WORST-CASE (pathologically long php+critique) measured 6825 tok at 8192,
# leaving only 1367 headroom — "close" per the truncation directive, so bumped to 12288
# (~1.8x margin over that extreme). No memory cost: KV pool is sized by GPU_MEM_UTIL,
# not max-model-len, which only caps per-sequence length.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SERVED_NAME="${SERVED_NAME:-wp_consistency}"
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

if [ ! -d "$HF_CACHE" ]; then
  echo "ERROR: HF cache not found: $HF_CACHE (download the model first)" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Stopping previous container: $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
fi

echo "Launching $NAME on :$PORT"
echo "  model: $MODEL_ID (text-only NVFP4 consistency judge)"
echo "  served-model-name: $SERVED_NAME"
echo "  gpu-memory-utilization: $GPU_MEM_UTIL  (coexists with wp_judge :8000)"
echo "  moe-backend: $MOE_BACKEND  image: $IMAGE"

docker run --rm -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  vllm serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --moe-backend "$MOE_BACKEND" \
    --enable-prefix-caching \
    --trust-remote-code

echo ""
echo "Follow readiness:  docker logs -f $NAME"
echo "Ready when:        curl -s http://localhost:${PORT}/v1/models | grep -q ${SERVED_NAME}"
echo "Stop:              docker stop $NAME"
