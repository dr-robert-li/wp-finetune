#!/usr/bin/env bash
# 04.3-03 memory de-risk: run the STREAMING bf16 load probe under a free-RAM watchdog and
# log the FULL available-RAM curve (every poll), so we can see plateau-vs-climb. Kills the
# load in the container if available RAM drops below TRIP_MIB (host OOM-cascade floor ~16 GiB).
set -u

MODEL="${1:-models/qwen3-30b-wp-30_70-merged-v2}"
TRIP_MIB="${2:-18432}"   # 18 GiB: backstop floor (the probe answers the curve; trip protects host)
POLL_SEC=2
LOG=logs/stream_load_probe.log
WLOG=logs/stream_load_watchdog.log
CURVE=logs/stream_load_curve.tsv
mkdir -p logs
: > "$LOG"; : > "$WLOG"; : > "$CURVE"
echo -e "ts\tavail_mib\tprogress" > "$CURVE"

ts() { date -u +%H:%M:%S; }
wlog() { echo "[$(ts)] $*" | tee -a "$WLOG"; }
avail_mib() { free -m | awk '/^Mem:/{print $7}'; }

wlog "START model=$MODEL trip=${TRIP_MIB}MiB poll=${POLL_SEC}s initial_avail=$(avail_mib)MiB"

docker exec -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -w /workspace/wp-finetune unsloth-headless \
  python -u scripts/_stream_load_probe.py "$MODEL" \
  >> "$LOG" 2>&1 &
LOAD_PID=$!
wlog "stream-probe launched host_pid=$LOAD_PID"

MIN_AVAIL=$(avail_mib)
TRIPPED=0
while kill -0 "$LOAD_PID" 2>/dev/null; do
  A=$(avail_mib)
  [ "$A" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$A
  PROG=$(grep -oE "Loading checkpoint shards: *[0-9]+%|LOAD COMPLETE" "$LOG" 2>/dev/null | tail -1)
  echo -e "$(ts)\t${A}\t${PROG}" >> "$CURVE"
  if [ "$A" -lt "$TRIP_MIB" ]; then
    wlog "!!! WATCHDOG TRIP avail=${A}MiB < ${TRIP_MIB}MiB — KILLING LOAD"
    docker exec unsloth-headless pkill -9 -f _stream_load_probe 2>/dev/null
    kill -9 "$LOAD_PID" 2>/dev/null
    TRIPPED=1
    break
  fi
  sleep "$POLL_SEC"
done

if [ "$TRIPPED" -eq 0 ]; then
  wait "$LOAD_PID"; RC=$?
  wlog "stream-probe exited rc=$RC"
else
  RC=137; sleep 3; wlog "post-kill avail=$(avail_mib)MiB"
fi
wlog "DONE tripped=$TRIPPED rc=$RC min_avail=${MIN_AVAIL}MiB (peak_used~$((121000 - MIN_AVAIL))MiB)"
echo "RESULT tripped=$TRIPPED rc=$RC min_avail_mib=$MIN_AVAIL" >> "$WLOG"
