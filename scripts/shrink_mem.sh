#!/usr/bin/env bash
# Shrink study: measure the memory one variant needs. Same settings as shrink_eval.sh; load the model,
# run one judge prompt (the first wp_judge val item, 2048-token cap), record the peak.
#   gpu_peak_mib  peak per-process GPU memory from nvidia-smi (weights + context cache + work buffers)
#   ram_peak_mib  peak resident system RAM of the server process (VmHWM)
# Writes output/shrink/<name>/memory.json.
# Usage: shrink_mem.sh <variant_name> <gguf_path> [extra llama-server args...]
set -uo pipefail
NAME="$1"; GGUF="$2"; shift 2; EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
LS="${LLAMA_BIN:-$HOME/work/llama.cpp/build/bin}/llama-server"
OUT="output/shrink/$NAME"; PORT=8098; CTX=$((2048 + 3072))
mkdir -p "$OUT"
[ -f "$GGUF" ] || { echo "MISSING $GGUF"; exit 2; }

"$LS" -m "$GGUF" --host 127.0.0.1 --port $PORT -ngl 999 -c $CTX --jinja -a "$NAME" \
  --parallel 1 --seed 1234 --no-mmap --no-cache-prompt --cache-ram 0 "${EXTRA[@]}" > "$OUT/mem_serve.log" 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
PEAK=0
sample() { local m; m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
  | awk -F', ' -v p=$SPID '$1==p{print $2}'); [[ "$m" =~ ^[0-9]+$ ]] && [ "$m" -gt "$PEAK" ] && PEAK=$m; }
READY=0
for i in $(seq 1 300); do
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -5 "$OUT/mem_serve.log"; exit 3; }
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" | grep -q ok && { READY=1; break; }; sleep 2
done
[ "$READY" = 1 ] || { echo "server never ready"; exit 3; }
PROMPT=$(python3 -c "
import json
for l in open('data/reasoning_dataset/openai_val.jsonl'):
    m = json.loads(l)['messages']
    if m[0]['content'].startswith('<wp_judge>'):
        print(json.dumps({'model': '$NAME', 'messages': [m[0]], 'max_tokens': 2048, 'temperature': 0,
                          'chat_template_kwargs': {'enable_thinking': False}})); break")
[ -n "$PROMPT" ] || { echo "no prompt"; exit 4; }
curl -sf -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$PROMPT" \
  > "$OUT/mem_response.json" &
CPID=$!
while kill -0 $CPID 2>/dev/null; do sample; sleep 1; done
wait $CPID || { echo "request failed"; exit 4; }
sample
[[ "$PEAK" =~ ^[0-9]+$ && "$PEAK" -gt 0 ]] || { echo "no GPU memory reading"; exit 5; }
RAM=$(awk '/VmHWM/{print int($2/1024)}' /proc/$SPID/status)
python3 - "$GGUF" "$PEAK" "$RAM" "${EXTRA[*]:-}" > "$OUT/memory.json" <<'PY2'
import json, os, sys
g, peak, ram, extra = sys.argv[1:5]
print(json.dumps({"gguf": g, "file_gib": round(os.path.getsize(g) / 2**30, 2), "gpu_peak_mib": int(peak),
                  "ram_peak_mib": int(ram), "server_args": extra}))
PY2
cat "$OUT/memory.json"
