#!/usr/bin/env bash
# Phase 04.3-04 Task 1 — detached grid-training launcher.
# Sources .env for TINKER_API_KEY (tinker_reasoning_sft.py reads it from env), then
# runs the resumable 9-candidate train+export orchestrator, logging to logs/grid_train.log.
set -euo pipefail
cd "$(dirname "$0")/.."

set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

mkdir -p logs output/tinker
echo "[grid-train.sh] starting $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/grid_train.log
exec .venv-tinker/bin/python scripts/_run_grid_train.py >> logs/grid_train.log 2>&1
