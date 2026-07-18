#!/usr/bin/env bash
# Phase 04.3-04 Task 2/3 — grid eval driver wrapper.
# Runs run_grid_eval.py under the PROJECT venv (miniconda: scipy/transformers/vLLM-serve)
# while sourcing .env so the .venv-tinker capture + fs_gate subprocesses inherit
# TINKER_API_KEY. All run_grid_eval.py flags pass through ("$@").
set -euo pipefail
cd "$(dirname "$0")/.."

set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

# wp-bench CLI lives in miniconda/bin; run_eval_reasoning's wp-bench subprocess inherits
# this PATH, so miniconda/bin MUST be present or `wp-bench` 404s as "CLI not on PATH"
# (obs 2866/2869). Launching python by absolute path does not add it — export explicitly.
export PATH="/home/robert_li/miniconda3/bin:$PATH"
PROJECT_PY=/home/robert_li/miniconda3/bin/python
exec "$PROJECT_PY" scripts/run_grid_eval.py "$@"
