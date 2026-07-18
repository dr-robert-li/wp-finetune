#!/usr/bin/env bash
# Generic free-RAM watchdog for GB10 unified-memory loads. Runs the given command (its args)
# backgrounded inside the unsloth-headless container, polls host free RAM @1s, and pkills the
# load at TRIP_MIB available (16 GiB ~= kernel OOM-cascade / host-reboot floor — trip above it).
#   usage: _mem_watchdog.sh <logtag> <python-module-and-args...>
# Example: _mem_watchdog.sh tf4bit scripts._tf_4bit_peak_probe --save models/...-4bit
set -u

TAG="${1:?need a logtag}"; shift
TRIP_MIB="${TRIP_MIB:-20480}"
POLL_SEC=1
LOG="logs/${TAG}.log"
WLOG="logs/${TAG}_watchdog.log"
mkdir -p logs; : > "$LOG"; : > "$WLOG"

ts() { date -u +%H:%M:%S; }
wlog() { echo "[$(ts)] $*" | tee -a "$WLOG"; }
avail_mib() { free -m | awk '/^Mem:/{print $7}'; }

wlog "START tag=$TAG trip=${TRIP_MIB}MiB cmd: python -u -m $* | initial_avail=$(avail_mib)MiB"

docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace/wp-finetune unsloth-headless \
  python -u -m "$@" >> "$LOG" 2>&1 &
LOAD_PID=$!
wlog "load launched host_pid=$LOAD_PID"

MIN_AVAIL=$(avail_mib); TRIPPED=0
while kill -0 "$LOAD_PID" 2>/dev/null; do
  A=$(avail_mib)
  [ "$A" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$A
  if [ "$A" -lt "$TRIP_MIB" ]; then
    wlog "!!! WATCHDOG TRIP avail=${A}MiB < ${TRIP_MIB}MiB — KILLING LOAD"
    docker exec unsloth-headless pkill -9 -f "$1" 2>/dev/null
    kill -9 "$LOAD_PID" 2>/dev/null
    TRIPPED=1; break
  fi
  sleep "$POLL_SEC"
done

if [ "$TRIPPED" -eq 0 ]; then wait "$LOAD_PID"; RC=$?; wlog "load exited rc=$RC"
else RC=137; sleep 3; wlog "post-kill avail=$(avail_mib)MiB"; fi

wlog "DONE tripped=$TRIPPED rc=$RC min_avail=${MIN_AVAIL}MiB"
echo "RESULT tripped=$TRIPPED rc=$RC min_avail_mib=$MIN_AVAIL" >> "$WLOG"
