#!/usr/bin/env bash
# 04.3-03 Task 1: binding-dryrun gate under free-RAM watchdog.
# Launches the 30B bf16 load (--no-4bit, ~57 GiB flat; the bf16 pivot — 4-bit
# on-the-fly is RULED OUT, ~108 GiB double-hold) inside unsloth-headless, polls
# host free RAM, pkills the load in the container if available RAM drops below
# TRIP_MIB. The ~57 GiB bf16 peak is itself SAFE; the watchdog is a backstop.
# Catastrophe floor is ~16 GiB (kernel OOM-cascade reboots host); trip well above.
set -u

TRIP_MIB=18432          # 18 GiB safe backstop. (The 2026-06-06 112-GiB-allowance test at 9920 is
                        # reverted: it tripped at 82% weight-load / ~114.7 GiB used WITHOUT completing
                        # — the bf16 in-process load+adapter needs >=124.6 GiB total, infeasible on
                        # this machine at ANY floor. See MEMORY-INVESTIGATION-bf16.md. Do not re-lower.)
POLL_SEC=1
LOG=logs/binding_dryrun_ckpt72.log
WLOG=logs/binding_dryrun_watchdog.log
mkdir -p logs
: > "$LOG"
: > "$WLOG"

ts() { date -u +%H:%M:%S; }
wlog() { echo "[$(ts)] $*" | tee -a "$WLOG"; }

avail_mib() { free -m | awk '/^Mem:/{print $7}'; }

wlog "START trip=${TRIP_MIB}MiB poll=${POLL_SEC}s initial_avail=$(avail_mib)MiB"

# Launch the load inside the container, detached, output -> $LOG.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set via -e (must be in env BEFORE the
# process imports torch/unsloth) to tame load-then-free fragmentation on unified memory:
# the ~100GiB peak with two-wave drain is the allocator grabbing new segments instead of
# reusing freed bf16-shard blocks. expandable_segments returns/reuses them -> lower peak.
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace/wp-finetune unsloth-headless \
  python -u -m scripts.checkpoint_parse_check \
  --checkpoint-dir adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72 \
  --base models/qwen3-30b-wp-30_70-merged-v2 \
  --binding-dryrun --no-4bit --max-memory-gib 0 \
  >> "$LOG" 2>&1 &
LOAD_PID=$!
wlog "load launched host_pid=$LOAD_PID"

MIN_AVAIL=$(avail_mib)
TRIPPED=0

while kill -0 "$LOAD_PID" 2>/dev/null; do
  A=$(avail_mib)
  [ "$A" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$A
  if [ "$A" -lt "$TRIP_MIB" ]; then
    wlog "!!! WATCHDOG TRIP avail=${A}MiB < ${TRIP_MIB}MiB — KILLING LOAD"
    docker exec unsloth-headless pkill -9 -f checkpoint_parse_check 2>/dev/null
    kill -9 "$LOAD_PID" 2>/dev/null
    TRIPPED=1
    break
  fi
  sleep "$POLL_SEC"
done

# Reap and capture exit status if it finished on its own
if [ "$TRIPPED" -eq 0 ]; then
  wait "$LOAD_PID"
  RC=$?
  wlog "load exited rc=$RC"
else
  RC=137
  # drain a couple more polls to confirm recovery
  sleep 3
  wlog "post-kill avail=$(avail_mib)MiB"
fi

wlog "DONE tripped=$TRIPPED rc=$RC min_avail=${MIN_AVAIL}MiB (peak_used=$((121000 - MIN_AVAIL))MiB approx)"
echo "RESULT tripped=$TRIPPED rc=$RC min_avail_mib=$MIN_AVAIL" >> "$WLOG"
