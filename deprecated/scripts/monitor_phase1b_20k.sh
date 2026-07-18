#!/usr/bin/env bash
# Phase 1b 20K rejudge monitor. Polls every 30 min; emits structured status line.
# Self-exits when the rejudge process is gone AND output file is complete (>=20000 rows)
# OR when the process is gone with partial output (failure exit).
#
# Watches:
#   - process matching "phase1b_stratified_rejudge"
#   - data/phase1b/rejudge_full_20k.jsonl   (target output)
#   - /tmp/rejudge_20k.log                  (stdout/stderr log)

set -u

OUT="data/phase1b/rejudge_full_20k.jsonl"
LOG="/tmp/rejudge_20k.log"
SCRIPT_NAME="phase1b_stratified_rejudge"
TARGET=20000
INTERVAL=1800   # 30 min
START_EPOCH=$(date +%s)
last_count=0
saw_running=0

emit() {
    echo "$@"
    # Mirror to project log for durability across monitor restarts
    mkdir -p logs
    echo "$@" >> logs/phase1b_20k_monitor.log
}

while true; do
    ts=$(date '+%Y-%m-%dT%H:%M:%S%z')
    pid=$(pgrep -f "$SCRIPT_NAME" | head -1)
    count=0
    if [ -f "$OUT" ]; then
        count=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    fi
    last_log=$(tail -1 "$LOG" 2>/dev/null | tr -d '\r\n' | head -c 200)
    elapsed=$(( $(date +%s) - START_EPOCH ))
    delta=$((count - last_count))
    last_count=$count

    if [ -n "$pid" ]; then
        saw_running=1
        emit "[$ts] RUNNING pid=$pid rows=$count/$TARGET delta_30m=+$delta monitor_elapsed=${elapsed}s last=\"$last_log\""
    else
        # Process not running
        if [ "$count" -ge "$TARGET" ]; then
            emit "[$ts] DONE rows=$count monitor_elapsed=${elapsed}s last=\"$last_log\""
            exit 0
        fi
        if [ "$saw_running" = "1" ]; then
            # Was running, now gone, output incomplete → failure
            emit "[$ts] EXITED_PARTIAL rows=$count/$TARGET monitor_elapsed=${elapsed}s last=\"$last_log\""
            exit 1
        fi
        # Never seen running yet → still waiting for user to launch
        emit "[$ts] WAITING_FOR_LAUNCH rows=$count monitor_elapsed=${elapsed}s"
    fi

    sleep $INTERVAL
done
