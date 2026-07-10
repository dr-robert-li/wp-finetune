#!/usr/bin/env bash
# 13-04 chained AIMER@25% gate driver: gen arm -> (gen must have measured) -> judge arm.
# One backgrounded process; wait on its PID. Logs: logs/prune/13-04_full_gate.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; . ./.env; set +a

PY=.venv-tinker/bin/python

echo "=== [$(date -u +%FT%TZ)] GEN ARM: aimer@25 gen gate ==="
$PY -m scripts.prune_gated_eval --method aimer --ratio 25 --axis gen \
    --score-npy output/prune/aimer_scores_gen.npy
gen_rc=$?
echo "=== [$(date -u +%FT%TZ)] gen arm exit=$gen_rc ==="
if [ "$gen_rc" -ne 0 ]; then
  echo "GEN ARM FAILED (infra) -- not starting judge arms"
  exit "$gen_rc"
fi

echo "=== [$(date -u +%FT%TZ)] JUDGE ARM: aimer@25 judge gate (3 seeds sequential) ==="
$PY -m scripts.prune_gated_eval --method aimer --ratio 25 --axis judge \
    --score-npy output/prune/aimer_scores_judge.npy
judge_rc=$?
echo "=== [$(date -u +%FT%TZ)] judge arm exit=$judge_rc ==="
exit "$judge_rc"
