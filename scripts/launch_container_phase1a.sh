#!/usr/bin/env bash
# Launch (or reuse) a long-lived nvcr.io/nvidia/pytorch:25.11-py3 container for
# Phase 1a + downstream calibration work. Bypasses the dgx-toolbox unsloth-studio
# launcher because that path hits issues #4 (resolved), #10 (torchcodec aarch64),
# and #11 (system-Python dispatch) — and we don't need the Studio UI anyway, just
# a shell with PHP scoring tools + Python ML deps.
#
# Idempotent:
#   - if container is already running  -> reuse it
#   - if container exists but stopped  -> start it
#   - otherwise                         -> docker run a fresh one
#
# After launch, runs scripts/setup_container_phase1a.sh inside to install
# PHP + composer + WPCS + PHPStan + xgboost + sklearn + pyyaml. Idempotent too.
#
# Usage (from host, in wp-finetune project root):
#   bash scripts/launch_container_phase1a.sh
#   docker exec -it unsloth-studio bash
#   # inside the container:
#   cd /workspace/project
#   RUBRIC_USE_LLM_CHECKS=1 python -m scripts.extract_pass_anchors --emit-features ...
set -euo pipefail

CONTAINER_NAME="unsloth-studio"
IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ensure_dir() {
    [ -d "$1" ] || mkdir -p "$1"
}

ensure_dir "$HOME/.cache/huggingface"
ensure_dir "$HOME/.cache/pip"
ensure_dir "$HOME/unsloth-data"

# Resolve the host `claude` CLI binary directory for bind-mount. The wrapper at
# ~/.local/bin/claude is a symlink to ~/.local/share/claude/versions/<ver>, which
# is the actual ELF binary. Mount the whole versions dir RO so the symlink is
# resolvable inside the container.
HOST_CLAUDE_BIN="$(command -v claude || true)"
if [ -n "$HOST_CLAUDE_BIN" ]; then
    HOST_CLAUDE_VERSIONS_DIR="$(readlink -f "$HOST_CLAUDE_BIN" | xargs dirname)"
else
    HOST_CLAUDE_VERSIONS_DIR=""
fi

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    cat <<EOF >&2
WARNING: CLAUDE_CODE_OAUTH_TOKEN env var not set on host.
  LLM checks inside the container will be unavailable.
  Generate a 1-year subscription-backed token via:
      claude setup-token
  Then re-run this script with the token exported:
      export CLAUDE_CODE_OAUTH_TOKEN=<paste-token-here>
      bash scripts/launch_container_phase1a.sh
  Proceeding without the token (deterministic-only rubric scoring).
EOF
fi

# docker inspect prints a stray newline to stdout on miss, so the `|| echo missing`
# fallback ends up with a leading-newline value. Probe explicitly via docker ps -a.
state=$(docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.State}}' | head -1)
[ -z "$state" ] && state="missing"

case "$state" in
    running)
        echo "Container '$CONTAINER_NAME' is already running. Reusing."
        ;;
    exited|created)
        echo "Container '$CONTAINER_NAME' is $state. Starting."
        docker start "$CONTAINER_NAME" >/dev/null
        ;;
    missing)
        echo "Container '$CONTAINER_NAME' not found. Creating fresh."
        DOCKER_ARGS=(
            -d
            --name "$CONTAINER_NAME"
            --gpus all
            --ipc=host
            -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
            -v "$HOME/.cache/pip:/root/.cache/pip"
            -v "$HOME/unsloth-data:/workspace/work"
            -v "$PROJECT_ROOT:/workspace/project"
            -e "CLAUDE_CONFIG_DIR=/tmp/claude-state"
        )
        if [ -n "$HOST_CLAUDE_VERSIONS_DIR" ]; then
            DOCKER_ARGS+=(-v "$HOST_CLAUDE_VERSIONS_DIR:/opt/claude:ro")
        fi
        if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
            DOCKER_ARGS+=(-e "CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN")
        fi
        DOCKER_ARGS+=(--restart unless-stopped "$IMAGE" sleep infinity)
        docker run "${DOCKER_ARGS[@]}" >/dev/null
        ;;
    *)
        echo "Container '$CONTAINER_NAME' is in unexpected state: $state" >&2
        echo "Remove with: docker rm -f $CONTAINER_NAME  and re-run." >&2
        exit 1
        ;;
esac

# Wait until the container reports healthy (sleep infinity = effectively ready immediately).
for _ in $(seq 1 30); do
    if [ "$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')" = "running" ]; then
        break
    fi
    sleep 0.5
done

echo
echo "Running setup_container_phase1a.sh inside container ..."
docker exec "$CONTAINER_NAME" bash /workspace/project/scripts/setup_container_phase1a.sh

echo
echo "Container ready. Open a shell with:"
echo "  docker exec -it $CONTAINER_NAME bash"
