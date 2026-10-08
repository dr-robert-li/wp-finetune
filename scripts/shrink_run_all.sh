#!/usr/bin/env bash
# Shrink study driver: build each size-reduction variant from the shipped Q6_K and evaluate it.
# Idempotent: a variant with output/shrink/<name>/resources.json is skipped.
# Usage: shrink_run_all.sh [variant ...]   (no args = all single options)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
BIN="${LLAMA_BIN:-$HOME/work/llama.cpp/build/bin}"; G=models/_gguf
SRC=$G/wp-judge-v4-pruned-k224.Q6_K.gguf; IMAT=$G/shrink.imatrix
EXPS='ffn_(gate|up|down|gate_up)_exps'

quant() {  # quant <out> <type> [extra llama-quantize args...]  (source is already Q6_K -> requantize)
  local out=$1 type=$2; shift 2
  [ -f "$out" ] || "$BIN/llama-quantize" --allow-requantize "$@" "${IN:-$SRC}" "$out" "$type" > "$out.log" 2>&1
}
imatrix() {  # calibration text = training-split judge examples (never the val set)
  [ -f "$IMAT" ] && return
  python3 - > $G/shrink_calib.txt <<'PY'
import json
for l in open("data/reasoning_dataset/openai_train.jsonl"):
    m = json.loads(l)["messages"]
    if m[0]["content"].startswith("<wp_judge>"):
        print(m[0]["content"]); print(m[-1]["content"]); print()
PY
  "$BIN/llama-imatrix" -m "$SRC" -f $G/shrink_calib.txt -o "$IMAT" -ngl 999 -c 2048 --chunks 200 > "$IMAT.log" 2>&1
}
run() {  # run <name> <gguf> [server args...]
  [ -f "output/shrink/$1/resources.json" ] && { echo "skip $1"; return; }
  bash scripts/shrink_eval.sh "$@"
}

v_baseline_q6k()      { run baseline_q6k "$SRC"; }
v_baseline_repeat()   { run baseline_repeat "$SRC"; }      # determinism check: must match baseline exactly
v_q5km()              { quant $G/shrink-q5km.gguf Q5_K_M; run q5km $G/shrink-q5km.gguf; }
v_q4km()              { quant $G/shrink-q4km.gguf Q4_K_M; run q4km $G/shrink-q4km.gguf; }
v_q3km()              { quant $G/shrink-q3km.gguf Q3_K_M; run q3km $G/shrink-q3km.gguf; }
v_iq4xs_im()          { imatrix; quant $G/shrink-iq4xs-im.gguf IQ4_XS --imatrix "$IMAT"; run iq4xs_im $G/shrink-iq4xs-im.gguf; }
v_iq3xxs_im()         { imatrix; quant $G/shrink-iq3xxs-im.gguf IQ3_XXS --imatrix "$IMAT"; run iq3xxs_im $G/shrink-iq3xxs-im.gguf; }
v_mix_e4km()          { quant $G/shrink-mix-e4km.gguf Q6_K --tensor-type "$EXPS=q4_K"; run mix_e4km $G/shrink-mix-e4km.gguf; }
v_mix_eiq4xs()        { imatrix; quant $G/shrink-mix-eiq4xs.gguf Q6_K --imatrix "$IMAT" --tensor-type "$EXPS=iq4_xs"; run mix_eiq4xs $G/shrink-mix-eiq4xs.gguf; }
v_mix_eiq3xxs()       { imatrix; quant $G/shrink-mix-eiq3xxs.gguf Q6_K --imatrix "$IMAT" --tensor-type "$EXPS=iq3_xxs"; run mix_eiq3xxs $G/shrink-mix-eiq3xxs.gguf; }
v_rt_cpumoe()         { run rt_cpumoe "$SRC" --n-cpu-moe 99; }                        # runtime: experts in system RAM
v_rt_kvq8()           { run rt_kvq8 "$SRC" -fa on -ctk q8_0 -ctv q8_0; }              # runtime: 8-bit context cache
v_prune_k192()        { [ -f $G/shrink-k192.Q6_K.gguf ] || python3 scripts/shrink_prune_gguf.py --in "$SRC" --out $G/shrink-k192.Q6_K.gguf --keep 192; run prune_k192 $G/shrink-k192.Q6_K.gguf; }
v_prune_k160()        { [ -f $G/shrink-k160.Q6_K.gguf ] || python3 scripts/shrink_prune_gguf.py --in "$SRC" --out $G/shrink-k160.Q6_K.gguf --keep 160; run prune_k160 $G/shrink-k160.Q6_K.gguf; }

ALL=(baseline_q6k baseline_repeat q5km q4km iq4xs_im q3km iq3xxs_im mix_e4km mix_eiq4xs mix_eiq3xxs rt_kvq8 rt_cpumoe prune_k192 prune_k160)
for v in "${@:-${ALL[@]}}"; do echo "=== $v $(date +%T)"; "v_$v"; done
