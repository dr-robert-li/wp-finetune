#!/usr/bin/env bash
# PKG-03 GGUF eval runner: serve a GGUF via llama-server, capture judge responses on the
# val set, score Spearman rho vs relabel val labels. Same harness as the bf16 vLLM measurement.
# Usage: _pkg_gguf_eval_run.sh <gguf_path> <alias> <out_dir> <port>
set -uo pipefail
GGUF="$1"; ALIAS="$2"; OUT="$3"; PORT="${4:-8091}"
LS=~/llama.cpp/build/bin/llama-server
ROOT=/home/robert_li/Desktop/projects/wp-finetune
cd "$ROOT"; mkdir -p "$OUT"

[ -f "$GGUF" ] || { echo "MISSING gguf $GGUF"; exit 2; }
echo "[eval] serving $GGUF as $ALIAS on :$PORT"
"$LS" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -ngl 999 -c 4096 --jinja -a "$ALIAS" \
  --parallel 8 > "$OUT/serve.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

# wait for health (up to 5 min for the 30B load)
for i in $(seq 1 150); do
  curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { echo "[eval] server ready ($i*2s)"; break; }
  kill -0 $SPID 2>/dev/null || { echo "[eval] server DIED"; tail -20 "$OUT/serve.log"; exit 3; }
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' || { echo "[eval] server never healthy"; tail -20 "$OUT/serve.log"; exit 3; }

echo "[eval] capturing judge responses on openai_val.jsonl"
python3 -m scripts.sieve_capture_judge_http \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$ALIAS" \
  --dataset data/reasoning_dataset/openai_val.jsonl \
  --out "$OUT/judge_responses.jsonl" 2>&1 | tail -5

echo "[eval] scoring rho"
python3 scripts/relabel/eval_relabel.py "$OUT/judge_responses.jsonl" 2>&1 | tee "$OUT/rho.txt"
echo "[eval] DONE $ALIAS"
