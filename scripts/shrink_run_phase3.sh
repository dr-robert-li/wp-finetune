#!/usr/bin/env bash
# Shrink study phase 3: combinations of surviving options, then a memory probe for every candidate.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; G=models/_gguf; SHIP=$G/wp-judge-v4-pruned-k224.Q6_K.gguf
KV=(-fa on -ctk q8_0 -ctv q8_0)
bash scripts/shrink_run_all.sh c_iq4xs_kvq8 c_eiq4xs_kvq8 c_eiq3xxs_kvq8
mem() { [ -f "output/shrink/$1/memory.json" ] || bash scripts/shrink_mem.sh "$@"; }
mem baseline_q6k "$SHIP"
mem rt_kvq8 "$SHIP" "${KV[@]}"
mem rt_cpumoe "$SHIP" --n-cpu-moe 99
mem q5km_f16 $G/shrink-q5km_f16.gguf
mem q4km_f16 $G/shrink-q4km_f16.gguf
mem iq4xs_im_f16 $G/shrink-iq4xs-im_f16.gguf
mem mix_eiq4xs_f16 $G/shrink-mix-eiq4xs_f16.gguf
mem mix_eiq3xxs_f16 $G/shrink-mix-eiq3xxs_f16.gguf
mem c_iq4xs_kvq8 $G/shrink-iq4xs-im_f16.gguf "${KV[@]}"
mem c_eiq4xs_kvq8 $G/shrink-mix-eiq4xs_f16.gguf "${KV[@]}"
mem c_eiq3xxs_kvq8 $G/shrink-mix-eiq3xxs_f16.gguf "${KV[@]}"
mem c_eiq3xxs_cpumoe $G/shrink-mix-eiq3xxs_f16.gguf --n-cpu-moe 99
echo "PHASE3 COMPLETE"
