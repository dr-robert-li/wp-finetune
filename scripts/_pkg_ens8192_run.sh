#!/usr/bin/env bash
# PKG-03 full-ensemble eval at max_tokens=8192: 3 Q8 seeds + 3 bf16 seeds, then median-ensemble score.
set -uo pipefail
cd /home/robert_li/Desktop/projects/wp-finetune
R=scripts/_pkg_gguf_eval_run.sh
BASE=output/packaging/ens8192
MAXTOK=8192
mkdir -p "$BASE"

for s in s0 s1 s2; do
  echo "=== Q8 $s @8192 ==="
  bash "$R" models/_gguf/wp-v1.3-judge-$s.Q8_0.gguf wp-v1.3-judge-q8-$s "$BASE/q8_$s" 8092 $MAXTOK
done
for s in s0 s1 s2; do
  echo "=== bf16 $s @8192 ==="
  bash "$R" models/_gguf/wp-v1.3-judge-$s.bf16.gguf wp-v1.3-judge-bf16-$s "$BASE/bf16_$s" 8092 $MAXTOK
done

echo "=== ENSEMBLE SCORING ==="
python3 scripts/relabel/eval_relabel_ensemble.py "$BASE/q8_ensemble.json" \
  "$BASE/q8_s0/judge_responses.jsonl" "$BASE/q8_s1/judge_responses.jsonl" "$BASE/q8_s2/judge_responses.jsonl" | tee "$BASE/q8_ensemble.txt"
python3 scripts/relabel/eval_relabel_ensemble.py "$BASE/bf16_ensemble.json" \
  "$BASE/bf16_s0/judge_responses.jsonl" "$BASE/bf16_s1/judge_responses.jsonl" "$BASE/bf16_s2/judge_responses.jsonl" | tee "$BASE/bf16_ensemble.txt"
echo "=== ENS8192 ALL DONE ==="
