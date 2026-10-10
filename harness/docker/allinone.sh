#!/bin/sh
# All-in-one: download + llama-server in the background, wait for /health, then the harness API.
set -e
/bin/sh /llama-entrypoint.sh &
LLAMA=$!
until curl -fs http://127.0.0.1:8080/health >/dev/null 2>&1; do
  kill -0 "$LLAMA" 2>/dev/null || { echo "llama-server exited" >&2; exit 1; }
  sleep 5
done
exec wpjudge serve --host 0.0.0.0 --port 8000
