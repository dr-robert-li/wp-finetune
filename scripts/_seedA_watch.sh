#!/usr/bin/env bash
# seedA RL monitor — exits (re-invoking Claude) when target step lands OR anything breaks.
# Usage: _seedA_watch.sh <target_step>
set -u
cd /home/robert_li/Desktop/projects/wp-finetune
TARGET="${1:?need target step}"
PIDFILE=output/rl_checkpoints/metrics/rl_run.seedA.pid
METRICS=output/rl_checkpoints/metrics/rl_metrics.seedA.jsonl
JFAIL=output/rl_checkpoints/metrics/judge_failures.seedA.jsonl
PY=.venv-tinker/bin/python

while true; do
  PID=$(cat "$PIDFILE" 2>/dev/null)
  ALIVE=no; kill -0 "$PID" 2>/dev/null && ALIVE=yes
  # parse last metric line
  read STEP HALT KL EFRAC RWD < <("$PY" - "$METRICS" <<'EOF'
import json,sys
try:
    last=open(sys.argv[1]).readlines()[-1]
    d=json.loads(last)
    print(d.get("step"), d.get("halt_reason"), round(d.get("kl_sample_train_v1") or 0,4),
          round(d.get("e_frac_with_tokens_mean") or 0,3), round(d.get("reward_mean") or 0,3))
except Exception as e:
    print("ERR ERR ERR ERR ERR")
EOF
)
  JF=$(wc -l < "$JFAIL" 2>/dev/null || echo 0)
  # judge servers
  J8000=down; curl -s -m 4 http://localhost:8000/v1/models >/dev/null 2>&1 && J8000=up
  J8001=down; curl -s -m 4 http://localhost:8001/v1/models >/dev/null 2>&1 && J8001=up

  REASON=""
  [ "$ALIVE" = no ] && REASON="PID_DEAD"
  [ "$HALT" != "None" ] && [ "$HALT" != "null" ] && [ -n "$HALT" ] && REASON="HALT=$HALT"
  case "$STEP" in (''|*[!0-9]*) : ;; (*) [ "$STEP" -ge "$TARGET" ] && REASON="${REASON:-REACHED_$TARGET}";; esac
  # KL/efrac breach (numeric compare via awk)
  awk "BEGIN{exit !($KL>0.1)}" 2>/dev/null && REASON="${REASON:-KL_BREACH=$KL}"
  awk "BEGIN{exit !($EFRAC<0.7)}" 2>/dev/null && REASON="${REASON:-EFRAC_BREACH=$EFRAC}"
  [ "$J8000" = down ] || [ "$J8001" = down ] && REASON="${REASON:-JUDGE_DOWN j8000=$J8000 j8001=$J8001}"

  if [ -n "$REASON" ]; then
    echo "WATCH_EXIT reason=$REASON step=$STEP alive=$ALIVE halt=$HALT kl=$KL efrac=$EFRAC reward=$RWD jfail=$JF j8000=$J8000 j8001=$J8001"
    exit 0
  fi
  sleep 120
done
