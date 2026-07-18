#!/usr/bin/env bash
# Phase 23-03 extension, llama.cpp fallback: serve the raw base GGUF +
# converted GGUF LoRA adapter via `llama-server --lora`, capture the 121
# wp_judge val prompts, score rho. Same capture/score harness as
# scripts/_pkg_gguf_eval_run.sh (v3's shipped-stack pattern), just with
# --lora added and no baked-in merge.
# Usage: eval4_ext_unmerged_llamacpp_run.sh <base_gguf> <lora_gguf> <alias> <out_dir> <port> [max_tokens]
set -uo pipefail
BASE_GGUF="$1"; LORA_GGUF="$2"; ALIAS="$3"; OUT="$4"; PORT="${5:-8091}"; MAXTOK="${6:-8192}"
LS=~/llama.cpp/build/bin/llama-server
ROOT=/home/robert_li/Desktop/projects/wp-finetune
cd "$ROOT"; mkdir -p "$OUT"

[ -f "$BASE_GGUF" ] || { echo "MISSING base gguf $BASE_GGUF"; exit 2; }
[ -f "$LORA_GGUF" ] || { echo "MISSING lora gguf $LORA_GGUF"; exit 2; }
echo "[eval] serving $BASE_GGUF + --lora $LORA_GGUF as $ALIAS on :$PORT"
PAR=4
CTX=$(( PAR * (MAXTOK + 3072) ))
"$LS" -m "$BASE_GGUF" --lora "$LORA_GGUF" --host 127.0.0.1 --port "$PORT" -ngl 999 -c "$CTX" --jinja -a "$ALIAS" \
  --parallel "$PAR" > "$OUT/serve.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

READY=0
for i in $(seq 1 180); do
  kill -0 $SPID 2>/dev/null || { echo "[eval] server DIED"; tail -40 "$OUT/serve.log"; exit 3; }
  out=$(curl -sf -m 30 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null)
  echo "$out" | grep -q '"content"' && { echo "[eval] model warm ($i*2s)"; READY=1; break; }
  sleep 2
done
[ "$READY" = 1 ] || { echo "[eval] model never warmed"; tail -40 "$OUT/serve.log"; exit 3; }

# Diff-gate BEFORE spending the full capture: toggle the loaded LoRA's scale to 0 via
# POST /lora-adapters (llama-server applies --lora at scale=1 by default), generate,
# then restore scale=1 and generate again on the SAME loaded weights/process -- an
# in-process A/B that isolates the adapter's effect without a second model load.
LORA_ID=$(curl -sf "http://127.0.0.1:$PORT/lora-adapters" | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])")
DIFF_PROMPT='<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction wpcs_get_option( $key ) {\n\treturn get_option( $key, false );\n}\n```'
gen_diff() {
  # --jinja splits <think>...</think> into reasoning_content separately from content;
  # 200 tokens of a reasoning-heavy judge is often still mid-<think> (content=""), so
  # combine both fields for a meaningful diff (and use more tokens for headroom).
  curl -sf -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"$DIFF_PROMPT\"}],\"max_tokens\":500,\"temperature\":0}" \
    | python3 -c "
import json, sys
m = json.load(sys.stdin)['choices'][0]['message']
print((m.get('reasoning_content') or '') + (m.get('content') or ''))
"
}
curl -sf -X POST "http://127.0.0.1:$PORT/lora-adapters" -H 'Content-Type: application/json' -d "[{\"id\":$LORA_ID,\"scale\":0}]" >/dev/null
OFF_TEXT=$(gen_diff)
curl -sf -X POST "http://127.0.0.1:$PORT/lora-adapters" -H 'Content-Type: application/json' -d "[{\"id\":$LORA_ID,\"scale\":1}]" >/dev/null
ON_TEXT=$(gen_diff)
{
  echo "lora_id=$LORA_ID"
  echo "--- lora OFF (scale=0) ---"
  echo "$OFF_TEXT"
  echo "--- lora ON (scale=1) ---"
  echo "$ON_TEXT"
} > "$OUT/diff_gate.txt"
if [ "$OFF_TEXT" = "$ON_TEXT" ] || [ -z "$ON_TEXT" ]; then
  echo "[eval] DIFF GATE FAILED: lora on/off outputs identical or empty -- see $OUT/diff_gate.txt"
  echo "diff_gate_status=blocked" > "$OUT/diff_gate_status.txt"
  exit 4
fi
echo "[eval] diff gate PASSED (lora on/off outputs differ) -- see $OUT/diff_gate.txt"
echo "diff_gate_status=ok" > "$OUT/diff_gate_status.txt"

echo "[eval] capturing judge responses on openai_val.jsonl"
python3 -m scripts.sieve_capture_judge_http \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$ALIAS" \
  --dataset data/reasoning_dataset/openai_val.jsonl \
  --max-tokens "$MAXTOK" \
  --out "$OUT/judge_responses.jsonl" 2>&1 | tail -5

echo "[eval] scoring rho"
python3 scripts/relabel/eval_relabel.py "$OUT/judge_responses.jsonl" 2>&1 | tee "$OUT/rho.txt"
echo "[eval] DONE $ALIAS"
