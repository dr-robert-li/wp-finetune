#!/usr/bin/env bash
# Phase 4.3 reasoning-training monitor.
# 30-min cadence. GB10 unified-memory aware (host free, not nvidia-smi memory).
# Sources training metrics from trainer_state.json in the newest checkpoint
# (docker logs cannot see commands launched via docker exec).
#
# Usage:  _phase4.3_monitor.sh <TDIR> [interval_sec] [max_hours]
# Stop:   touch "<TDIR>/_stop"  (or training finishes -> adapter_config.json appears).

set -u

TDIR="${1:?usage: $0 <TDIR> [interval_sec] [max_hours]}"
INTERVAL="${2:-1800}"
MAX_HOURS="${3:-12}"
MAX_CHECKS=$(( MAX_HOURS * 3600 / INTERVAL ))
[ "$MAX_CHECKS" -lt 1 ] && MAX_CHECKS=1

mkdir -p "$TDIR"
JSONL="${TDIR}/monitor.jsonl"
LATEST_MD="${TDIR}/monitor_latest.md"
LOG="${TDIR}/monitor.log"

CONTAINER="unsloth-headless"
ADAPTER_DIR="adapters/qwen3-30b-wp-30_70-reasoning"
ADAPTER_DONE="${ADAPTER_DIR}/adapter_config.json"

ts0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$ts0] monitor start tdir=$TDIR interval=${INTERVAL}s max=${MAX_CHECKS} adapter=$ADAPTER_DONE" >> "$LOG"

latest_step() {
  # Newest checkpoint, then parse last log_history entry.
  local newest
  newest=$(ls -dt "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null | head -1 || true)
  [ -z "$newest" ] && { echo ""; return; }
  local ts="${newest}/trainer_state.json"
  [ -f "$ts" ] || { echo ""; return; }
  python3 - "$ts" <<'PY' 2>/dev/null || echo ""
import json,sys
try:
  s=json.load(open(sys.argv[1]))
  h=s.get("log_history",[])
  if h:
    e=h[-1]
    print(f'{e.get("step","")}|{e.get("loss","")}|{e.get("grad_norm","")}|{e.get("epoch","")}|{e.get("learning_rate","")}')
except Exception:
  pass
PY
}

for (( i=1; i<=MAX_CHECKS; i++ )); do
  [ -f "${TDIR}/_stop" ] && { echo "[$(date -u +%FT%TZ)] _stop sentinel — exit" >> "$LOG"; break; }
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0, 0, 0, 0")
  gpu_util=$(echo "$gpu_raw" | awk -F', ' '{print int($1)}')
  gpu_mem_util=$(echo "$gpu_raw" | awk -F', ' '{print int($2)}')
  gpu_mem_used_raw=$(echo "$gpu_raw" | awk -F', ' '{print $3}')
  gpu_mem_total_raw=$(echo "$gpu_raw" | awk -F', ' '{print $4}')
  case "$gpu_mem_used_raw" in *N/A*) gpu_mem_used=-1 ;; *) gpu_mem_used=${gpu_mem_used_raw// /} ;; esac
  case "$gpu_mem_total_raw" in *N/A*) gpu_mem_total=-1 ;; *) gpu_mem_total=${gpu_mem_total_raw// /} ;; esac
  temp=$(echo "$gpu_raw" | awk -F', ' '{print int($5)}')
  watts=$(echo "$gpu_raw" | awk -F', ' '{printf "%.1f", $6}')

  read host_total host_used host_avail < <(free -g | awk '/^Mem:/{print $2, $3, $7}')

  cstats=$(docker stats --no-stream --format '{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}' "$CONTAINER" 2>/dev/null || echo "|||")
  c_cpu="${cstats%%|*}"; rest="${cstats#*|}"
  c_memusage="${rest%%|*}"; c_memperc="${rest#*|}"

  ckpt_count=$(ls -d "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null | wc -l | tr -d ' ')
  newest_ckpt=$(ls -dt "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null | head -1 || true)
  newest_ckpt_name=$(basename "$newest_ckpt" 2>/dev/null || echo "")

  ts_line=$(latest_step)
  step="${ts_line%%|*}"; rest="${ts_line#*|}"
  loss="${rest%%|*}"; rest="${rest#*|}"
  grad_norm="${rest%%|*}"; rest="${rest#*|}"
  epoch="${rest%%|*}"; rest="${rest#*|}"
  lr="${rest%%|*}"

  done="false"
  [ -f "$ADAPTER_DONE" ] && done="true"

  printf '{"ts":"%s","check":%d,"gpu_util_pct":%d,"gpu_mem_util_pct":%d,"gpu_mem_used_mib":%s,"gpu_mem_total_mib":%s,"temp_c":%d,"watts":%s,"host_total_gb":%s,"host_used_gb":%s,"host_avail_gb":%s,"container_cpu":"%s","container_mem":"%s","container_mem_pct":"%s","checkpoints":%d,"newest_ckpt":"%s","step":"%s","loss":"%s","grad_norm":"%s","epoch":"%s","lr":"%s","adapter_done":%s}\n' \
    "$ts" "$i" "$gpu_util" "$gpu_mem_util" "$gpu_mem_used" "$gpu_mem_total" "$temp" "$watts" \
    "$host_total" "$host_used" "$host_avail" "$c_cpu" "$c_memusage" "$c_memperc" \
    "$ckpt_count" "$newest_ckpt_name" "$step" "$loss" "$grad_norm" "$epoch" "$lr" "$done" >> "$JSONL"

  cat > "$LATEST_MD" <<EOF
# Phase 4.3 monitor — check ${i}/${MAX_CHECKS}

- ts (UTC): ${ts}
- interval: ${INTERVAL}s

## GPU (host nvidia-smi)
- util: ${gpu_util}%   mem_util: ${gpu_mem_util}%
- mem: ${gpu_mem_used} / ${gpu_mem_total} MiB  (-1 = N/A on unified memory)
- temp: ${temp}C   power: ${watts} W

## Host (DGX unified memory — this IS GPU memory on GB10)
- total: ${host_total} GB   used: ${host_used} GB   avail: ${host_avail} GB

## Container (${CONTAINER})
- cpu: ${c_cpu}   mem: ${c_memusage}   mem_pct: ${c_memperc}

## Training
- checkpoints: ${ckpt_count}   newest: ${newest_ckpt_name:-none}
- step: ${step:-?}   loss: ${loss:-?}   grad_norm: ${grad_norm:-?}   epoch: ${epoch:-?}   lr: ${lr:-?}
- adapter_config.json: ${done}
EOF

  echo "[${ts}] check ${i}/${MAX_CHECKS} gpu=${gpu_util}% temp=${temp}C watts=${watts}W host_avail=${host_avail}GB ckpts=${ckpt_count} step=${step:-?} loss=${loss:-?}" >> "$LOG"

  if [ "$done" = "true" ]; then
    echo "[${ts}] adapter_config.json present — training done. monitor exiting." >> "$LOG"
    break
  fi
  [ "$i" -lt "$MAX_CHECKS" ] && sleep "$INTERVAL"
done

echo "[$(date -u +%FT%TZ)] monitor exit after ${i} check(s)" >> "$LOG"
