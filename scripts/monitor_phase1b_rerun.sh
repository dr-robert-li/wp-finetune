#!/usr/bin/env bash
# Phase 1b SEC-N04 subset rerun monitor. Polls every 30 min.
# Self-exits when process gone AND output complete (>=2689 rows) OR EXITED_PARTIAL.

set -u

OUT="data/phase1b/rerun_secn04_fix.jsonl"
LOG="/tmp/rerun_subset.log"
SCRIPT_NAME="phase1b_rerun_subset"
TARGET=2689
INTERVAL=1800
START_EPOCH=$(date +%s)
last_count=0
saw_running=0

emit() {
    echo "$@"
    mkdir -p logs
    echo "$@" >> logs/phase1b_rerun_monitor.log
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
        if [ "$count" -ge "$TARGET" ]; then
            emit "[$ts] DONE rows=$count monitor_elapsed=${elapsed}s last=\"$last_log\""
            exit 0
        fi
        if [ "$saw_running" = "1" ]; then
            emit "[$ts] EXITED_PARTIAL rows=$count/$TARGET monitor_elapsed=${elapsed}s last=\"$last_log\""
            exit 1
        fi
        emit "[$ts] WAITING_FOR_LAUNCH rows=$count monitor_elapsed=${elapsed}s"
    fi

    sleep $INTERVAL
done
