#!/usr/bin/env bash
# Shrink study phase 2: once the Q6_K-sourced q4km run finishes, stop the phase-1 driver and run the
# file variants from the rebuilt f16 (true release path), then the runtime and expert-cut options.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# Phase-1 driver and its running q4km eval are addressed by PID (text-matching pkill can hit shell
# wrappers). A stopped driver cannot start the next variant; the q4km eval finishes and scores.
DRIVER=${DRIVER:?phase-1 driver PID}; EVAL=${EVAL:?q4km shrink_eval PID}
kill -STOP "$DRIVER"
# a finished child of the stopped driver stays a zombie (not reaped) -> treat state Z as done
while s=$(ps -o stat= -p "$EVAL" 2>/dev/null) && [[ "$s" != Z* ]]; do sleep 30; done
kill -KILL "$DRIVER"
[ -f output/shrink/q4km/score.json ] || echo "WARN q4km score missing"
SRC_GGUF=models/_gguf/wp-judge-v4-pruned-k224.f16.gguf TAG=_f16 bash scripts/shrink_run_all.sh \
  q5km q4km iq4xs_im q3km iq3xxs_im mix_e4km mix_eiq4xs mix_eiq3xxs
bash scripts/shrink_run_all.sh rt_kvq8 rt_cpumoe prune_k192 prune_k160
echo "PHASE2 COMPLETE"
