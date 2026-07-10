#!/usr/bin/env bash
# OOM guard for the DGX Spark (GB10) — UNIFIED memory, NO memory protection.
# An OOM event hangs the machine unrecoverably, so this watchdog acts BEFORE the
# hard floor: if MemAvailable drops below THRESH_MB it PAUSES the RL run (kills the
# trainer) and STOPS the vLLM containers, then logs to the status doc and exits.
#
# Trips when MemAvailable drops BELOW 2GB (2048MB). Polls fast (1s) so it reacts
# quickly at the floor, since there is no margin above the trip point.
#
# Run as a persistent background watchdog for the whole multi-day run.
#   bash scripts/_oom_guard.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THRESH_MB="${OOM_THRESH_MB:-2048}"
POLL_S="${OOM_POLL_S:-1}"
DOC="$REPO/.planning/phases/09-gspo-training/09-LOCAL-RL-STATUS-UPDATES.md"
PIDFILE="$REPO/output/rl_checkpoints/rl_run.pid"
CONTAINERS=("wp-consistency-vllm" "wp-v4-judge-vllm")

while true; do
  avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  if [ -z "$avail" ]; then sleep "$POLL_S"; continue; fi
  if [ "$avail" -lt "$THRESH_MB" ]; then
    ts=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    # 1) pause the RL run first (it is the active allocator)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; fi
    pkill -f "scripts/rl_train.py" 2>/dev/null || true
    # 2) stop the vLLM servers to release the unified-memory reservation
    docker stop "${CONTAINERS[@]}" 2>/dev/null || true
    avail2=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
    {
      echo ""
      echo "### 🛑 OOM-GUARD TRIPPED · $ts"
      echo "- MemAvailable hit ${avail}MB (< ${THRESH_MB}MB threshold). DGX Spark has no OOM"
      echo "  protection, so the run was PAUSED to avoid an unrecoverable hang."
      echo "- Actions: killed rl_train.py + stopped containers ${CONTAINERS[*]}."
      echo "- MemAvailable after stop: ${avail2}MB."
      echo "- RESUME requires manual restart (re-serve judges, relaunch run). Investigate the"
      echo "  allocation spike before relaunch (lower gpu-memory-utilization / max-num-seqs)."
    } >> "$DOC"
    echo "OOM_GUARD_TRIPPED avail=${avail}MB after_stop=${avail2}MB — run paused, containers stopped"
    exit 1
  fi
  sleep "$POLL_S"
done
