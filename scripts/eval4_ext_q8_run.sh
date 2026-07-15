#!/usr/bin/env bash
# Phase 23-02 extension: v4 judge, 3-seed Q8_0 GGUF ensemble @8192, on the
# shipped llama.cpp stack. Reuses scripts/_pkg_gguf_eval_run.sh UNCHANGED
# (v3's exact serve+capture+score harness) -- one seed at a time (sole GB10
# residency), then median-ensembles with the unchanged
# scripts/relabel/eval_relabel_ensemble.py.
set -uo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
R=scripts/_pkg_gguf_eval_run.sh
BASE=output/eval4/ext_q8
MAXTOK=8192
mkdir -p "$BASE"

for s in s0 s1 s2; do
  echo "=== v4 Q8 $s @8192 ==="
  bash "$R" "models/_gguf/wp-v4-judge-$s.Q8_0.gguf" "wp-v4-judge-q8-$s" "$BASE/q8_$s" 8093 $MAXTOK
done

echo "=== ENSEMBLE SCORING ==="
python3 scripts/relabel/eval_relabel_ensemble.py "$BASE/q8_ensemble.json" \
  "$BASE/q8_s0/judge_responses.jsonl" "$BASE/q8_s1/judge_responses.jsonl" "$BASE/q8_s2/judge_responses.jsonl" | tee "$BASE/q8_ensemble.txt"
echo "=== EXT Q8 RUN DONE ==="
