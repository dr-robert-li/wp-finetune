#!/usr/bin/env bash
# Retry wrapper for scripts/tinker_export_checkpoint.py (Plan 11-02 Task 1).
# The Tinker archive-creation job runs server-side; a client request can exceed the
# SDK's internal retry budget (~10-20 min) while the server is still packing, but the
# job keeps progressing regardless of client disconnect. A fresh request after a short
# backoff picks up the (by-then-ready) signed URL fast. Precedent: logs/relabel_sft/
# export_v13_retry.log ("try N failed rc=1; sleeping 120s (server may still be packing)").
# ponytail: bash retry loop, not a code change to the SDK/export script — ceiling is
# MAX_TRIES total attempts; raise MAX_TRIES if the server is unusually slow to pack.
set -uo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
TINKER_PATH="${1:?need tinker path}"
OUT_DIR="${2:?need out-dir}"
MAX_TRIES=10
SLEEP_SEC=120

set -a; . ./.env; set +a
export PYTHONPATH=.
PY=.venv-tinker/bin/python

for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== export try ${i}/${MAX_TRIES} ==="
  "$PY" scripts/tinker_export_checkpoint.py --tinker-path "$TINKER_PATH" --out-dir "$OUT_DIR"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "${OUT_DIR}/checkpoint.tar" ] && [ -f "${OUT_DIR}/export_manifest.json" ]; then
    echo "=== export SUCCEEDED on try ${i} ==="
    exit 0
  fi
  echo "try ${i} failed rc=${rc}; sleeping ${SLEEP_SEC}s (server may still be packing)"
  sleep "$SLEEP_SEC"
done
echo "=== export FAILED after ${MAX_TRIES} tries ==="
exit 1
