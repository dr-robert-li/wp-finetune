#!/usr/bin/env bash
# Shrink study: serve one GGUF variant, capture judge responses on the 121 wp_judge val prompts,
# record size + memory + speed, score against the baseline capture.
# Deterministic: --parallel 1 (no cross-request batching), greedy decode (temperature 0), fixed seed.
# Usage: shrink_eval.sh <variant_name> <gguf_path> [extra llama-server args...]
set -uo pipefail
NAME="$1"; GGUF="$2"; shift 2; EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
LS="${LLAMA_BIN:-$HOME/work/llama.cpp/build/bin}/llama-server"
OUT="output/shrink/$NAME"; BASE_CAP="output/shrink/baseline_q6k/judge_responses.jsonl"
PORT=8097; MAXTOK=2048; CTX=$((MAXTOK + 3072))
mkdir -p "$OUT"
[ -f "$GGUF" ] || { echo "MISSING $GGUF"; exit 2; }

"$LS" -m "$GGUF" --host 127.0.0.1 --port $PORT -ngl 999 -c $CTX --jinja -a "$NAME" \
  --parallel 1 --seed 1234 --no-mmap "${EXTRA[@]}" > "$OUT/serve.log" 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 300); do
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$OUT/serve.log"; exit 3; }
  curl -sf -m 30 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null \
    | grep -q '"content"' && break
  sleep 2
done

T0=$(date +%s)
python3 -m scripts.sieve_capture_judge_http --base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
  --dataset data/reasoning_dataset/openai_val.jsonl --max-tokens $MAXTOK \
  --out "$OUT/judge_responses.jsonl" 2>&1 | tail -2
WALL=$(( $(date +%s) - T0 ))
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

# Size and memory facts come from the file and llama.cpp's own buffer report (deterministic).
python3 - "$GGUF" "$OUT" "$WALL" "${EXTRA[*]:-}" <<'PY'
import json, os, re, sys
gguf, out, wall, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
log = open(f"{out}/serve.log").read()
def mib(pat):
    return round(sum(float(x) for x in re.findall(pat + r"\s*=\s*([\d.]+) MiB", log)), 1)
tg = [float(x) for x in re.findall(r"([\d.]+) tokens per second\)\s*\n\s*eval time", log)]
ev = [float(x) for x in re.findall(r"eval time =.*?([\d.]+) tokens per second", log)]
json.dump({"gguf": gguf, "file_gib": round(os.path.getsize(gguf) / 2**30, 2),
           "gpu_weights_mib": mib(r"CUDA0 model buffer size"),
           "cpu_weights_mib": mib(r"CPU(?:_Mapped)? model buffer size"),
           "kv_mib": mib(r"CUDA0 KV buffer size") + mib(r"CPU KV buffer size"),
           "compute_mib": mib(r"CUDA0 compute buffer size"),
           "gen_tok_s_median": sorted(ev)[len(ev) // 2] if ev else None,
           "wall_s": wall, "server_args": extra}, open(f"{out}/resources.json", "w"), indent=2)
PY
[ -f "$BASE_CAP" ] && [ "$NAME" != baseline_q6k ] && \
  python3 scripts/shrink_score.py "$OUT/judge_responses.jsonl" "$BASE_CAP" "$OUT/score.json"
echo "DONE $NAME"
