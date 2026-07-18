#!/usr/bin/env bash
# PKG-03 GGUF eval runner: serve a GGUF via llama-server, capture judge responses on the
# val set, score Spearman rho vs relabel val labels. Same harness as the bf16 vLLM measurement.
# Usage: _pkg_gguf_eval_run.sh <gguf_path> <alias> <out_dir> <port>
set -uo pipefail
GGUF="$1"; ALIAS="$2"; OUT="$3"; PORT="${4:-8091}"; MAXTOK="${5:-2048}"
LS=~/llama.cpp/build/bin/llama-server
ROOT=/home/robert_li/Desktop/projects/wp-finetune
cd "$ROOT"; mkdir -p "$OUT"

[ -f "$GGUF" ] || { echo "MISSING gguf $GGUF"; exit 2; }
echo "[eval] serving $GGUF as $ALIAS on :$PORT"
# n_ctx is split across --parallel slots; per-slot must fit prompt(<=~2k) + MAXTOK gen.
PAR=4
CTX=$(( PAR * (MAXTOK + 3072) ))
"$LS" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -ngl 999 -c "$CTX" --jinja -a "$ALIAS" \
  --parallel "$PAR" > "$OUT/serve.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

# Readiness = a REAL generation succeeds (not just /health, which returns ok while the
# 30B is still loading -> early flooded requests 503 "Loading model" -> empty responses).
READY=0
for i in $(seq 1 180); do
  kill -0 $SPID 2>/dev/null || { echo "[eval] server DIED"; tail -20 "$OUT/serve.log"; exit 3; }
  out=$(curl -sf -m 30 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null)
  echo "$out" | grep -q '"content"' && { echo "[eval] model warm ($i*2s)"; READY=1; break; }
  sleep 2
done
[ "$READY" = 1 ] || { echo "[eval] model never warmed"; tail -20 "$OUT/serve.log"; exit 3; }

echo "[eval] capturing judge responses on openai_val.jsonl"
python3 -m scripts.sieve_capture_judge_http \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$ALIAS" \
  --dataset data/reasoning_dataset/openai_val.jsonl \
  --max-tokens "$MAXTOK" \
  --out "$OUT/judge_responses.jsonl" 2>&1 | tail -5

echo "[eval] scoring rho"
python3 scripts/relabel/eval_relabel.py "$OUT/judge_responses.jsonl" 2>&1 | tee "$OUT/rho.txt"
echo "[eval] DONE $ALIAS"
