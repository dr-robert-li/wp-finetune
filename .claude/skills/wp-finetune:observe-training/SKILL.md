# Skill: wp-finetune:observe-training

Spawn background monitoring for DGX Spark training. Writes telemetry to `telemetry/training/{timestamp}/`.

**Agent team assessment:** Training is GPU-heavy (6-12hr), runs in Docker, produces checkpoints, and has training metrics. Full agent team uses 6 Sonnet agents: gpu-metrics, thermal-throttling, training-metrics, disk-io, checkpoint-integrity, container-monitor.

> When creating new skills that involve training, assess whether this agent team needs modification:
> - Uses GPU? Yes -> gpu-metrics, thermal-throttling included
> - Runs > 30 min? Yes -> system resources covered by disk-io + container-monitor
> - Writes large files? Yes -> disk-io included
> - Runs in Docker? Yes -> container-monitor included
> - Has checkpoints? Yes -> checkpoint-integrity included
> - Has progress metric? Yes -> training-metrics included

> **Container name:** Agents reference `unsloth-headless` which is resolved from `config/dgx_toolbox.yaml -> containers.unsloth_studio.container_name`. If the container name changes in config, update the agent prompts below.

## Trigger

User says: "observe training", "monitor training", "/observe-training"

## Process

### 1. Create or Resume Run Directory

**Self-recovery:** Check for an existing active monitor before creating a new directory.

```bash
# Check for running lightweight monitor from a previous session
ACTIVE_PID=""
ACTIVE_DIR=""
for pidfile in telemetry/training/*/monitor.pid; do
    [ -f "$pidfile" ] || continue
    pid=$(cat "$pidfile")
    dir=$(dirname "$pidfile")
    if kill -0 "$pid" 2>/dev/null && [ ! -f "$dir/_stop" ]; then
        ACTIVE_PID="$pid"
        ACTIVE_DIR="$dir"
        break
    fi
done
```

**If active monitor found:** Use AskUserQuestion:
- header: "Monitor found"
- question: "Active monitor detected at ${ACTIVE_DIR} (PID ${ACTIVE_PID}). What do you want to do?"
- options:
  - "Keep it running" — Monitor is healthy, just report its location
  - "Restart in same directory" — Kill and restart (appends to existing JSONL)
  - "Start fresh" — New timestamped directory

**If no active monitor but an incomplete run exists** (TDIR with monitor.jsonl but no `_stop` and dead PID — monitor crashed):
- Offer to **restart** in the same directory (appends to existing JSONL, no data loss)

**If no previous monitor:** Create fresh:
```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/training/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all steps below.

### 2. Choose Monitoring Mode

Use AskUserQuestion:
- header: "Monitor mode"
- question: "Which monitoring level for this training run?"
- options:
  - "Lightweight (Recommended)" -- Single bash script, ~5MB, writes JSONL. Safe on DGX Spark with tight memory
  - "Full agent team" -- 6 Sonnet agents writing detailed markdown reports. Adds ~2.4GB memory overhead. Use only with >25GB headroom

### 3a. Lightweight Monitor (default)

Generate and write the following script to `{TDIR}/monitor.sh`, then start it:

```bash
#!/usr/bin/env bash
set -euo pipefail
TDIR="$1"
JSONL="${TDIR}/monitor.jsonl"
MAX_CHECKS=${2:-504}  # 504 * 60s = 8.4 hours default

for (( i=1; i<=MAX_CHECKS; i++ )); do
    [[ -f "${TDIR}/_stop" ]] && echo "Stop signal. Exiting after $((i-1)) checks." && exit 0

    gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0")
    gpu_util=$(echo "$gpu_raw" | awk -F', ' '{print int($1)}')
    temp=$(echo "$gpu_raw" | awk -F', ' '{print int($2)}')
    watts=$(echo "$gpu_raw" | awk -F', ' '{printf "%.1f", $3}')
    mem_line=$(free -m | grep Mem)
    available=$(echo "$mem_line" | awk '{print $7}')
    mem_available_gb=$(awk "BEGIN {printf \"%.1f\", $available / 1024}")

    # Training-specific: check latest loss from docker logs
    loss=$(docker logs --tail 5 unsloth-headless 2>&1 | grep -oP "'loss': \K[0-9.]+" | tail -1 || echo "")
    step=$(docker logs --tail 5 unsloth-headless 2>&1 | grep -oP "'step': \K[0-9]+" | tail -1 || echo "")

    # Check for checkpoints
    ckpt_count=$(find adapters/qwen3-30b-wp-*/checkpoint-* -maxdepth 0 -type d 2>/dev/null | wc -l || echo "0")

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"ts\": \"$ts\", \"watts\": $watts, \"temperature_c\": $temp, \"gpu_util_pct\": $gpu_util, \"mem_available_gb\": $mem_available_gb, \"loss\": \"$loss\", \"step\": \"$step\", \"checkpoints\": $ckpt_count, \"source\": \"monitor\"}" >> "$JSONL"
    echo "$ts Check $i/$MAX_CHECKS | temp=${temp}C watts=${watts}W gpu=${gpu_util}% ram_avail=${mem_available_gb}GB loss=${loss} step=${step} ckpts=${ckpt_count}"

    (( temp >= 85 )) && touch "${TDIR}/_thermal_pause" && echo "$ts CRITICAL: temp=${temp}C >= 85"
    (( i < MAX_CHECKS )) && sleep 60
done
```

Start the monitor:

```bash
chmod +x {TDIR}/monitor.sh
nohup bash {TDIR}/monitor.sh "{TDIR}" > {TDIR}/monitor.log 2>&1 &
echo $! > {TDIR}/monitor.pid
```

Report: "Training lightweight monitor active. PID: {pid}. Output: {TDIR}/monitor.jsonl. Touch {TDIR}/_stop to end. Run /review-telemetry to check status."

### 3b. Full Agent Team (optional, sonnet only)

Spawn agents with EXPLICIT `model="sonnet"` parameter. This is critical -- haiku agents do not persist loops.

#### GPU Metrics Observer

```
Agent(
  model="sonnet",
  description="Telemetry: GPU metrics",
  prompt="You are a GPU metrics observer. Write observations to {TDIR}/gpu-metrics.md.

  LOOP (every 30 seconds):
  1. Run: nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,clocks.current.sm --format=csv,noheader,nounits
  2. Run: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  3. Run: free -m | grep Mem (system RAM -- critical for unified memory systems like GB10 where VRAM reports [N/A])
  4. Parse memory: use nvidia-smi memory.used if available, otherwise compute from system RAM (total - available)
  5. Append to {TDIR}/gpu-metrics.md:
     ### {HH:MM:SS}
     - VRAM: {used}/{total} MiB ({pct}%) [or N/A on unified memory]
     - System RAM: {used}/{total} MiB ({pct}%)
     - GPU Util: {util}%
     - Mem Util: {mem_util}%
     - SM Clock: {clock} MHz
     - Processes: {list}
  6. Flag WARNING if memory > 90%, CRITICAL (warn+log, do NOT stop training) if memory >= 98%
     Memory is caught pre-training by validate (Step 2) and dry run (Step 6). During training, just observe.
  7. Flag WARNING if utilization.gpu < 50% for 3+ consecutive readings
  8. Check if {TDIR}/_stop exists -- if so, write ## Final Summary (peak memory, avg util, total readings) and exit
  9. Sleep 30 seconds, repeat

  NOTE: All nvidia-smi and free commands run on the HOST, not via docker exec.
  Long-running containers can lose NVML access while the host nvidia-smi stays reliable.

  NOTE: On unified memory systems (NVIDIA GB10/Grace Hopper), nvidia-smi memory reports [N/A].
  System RAM IS the GPU memory -- use free/proc/meminfo instead. On discrete GPU systems,
  both VRAM and system RAM are meaningful and should both be recorded.

  STOP CONDITIONS: _stop file exists OR no GPU compute processes for 15+ minutes",
  run_in_background=true
)
```

#### Thermal/Throttling Observer

```
Agent(
  model="sonnet",
  description="Telemetry: thermal/throttling",
  prompt="You are a GPU thermal observer. Write observations to {TDIR}/thermal-throttling.md.

  LOOP (every 30 seconds):
  1. Run: nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
  2. Run: nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -i 'throttle\|clock\|slow' || true
  3. Run: nvidia-smi dmon -s pucvmet -c 1 2>/dev/null || true
  4. Append to {TDIR}/thermal-throttling.md:
     ### {HH:MM:SS}
     - Temperature: {temp}C
     - Power: {draw}W / {limit}W ({pct}%)
     - Throttle reasons: {reasons or 'None'}
  5. Flag WARNING if temp > 82C
  6. Flag CRITICAL if temp >= 85C or any throttle reason is Active
     -> Touch {TDIR}/_thermal_pause to signal adaptive resource planning
     -> Write thermal event details to {TDIR}/_thermal_pause (temp, timestamp, power)
  7. Flag WARNING if power.draw > 90% of power.limit
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (peak temp, throttle event count, avg power) and exit
  9. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no GPU activity for 15+ minutes",
  run_in_background=true
)
```

#### Training Metrics Observer

```
Agent(
  model="sonnet",
  description="Telemetry: training metrics",
  prompt="You are a training metrics observer. Write observations to {TDIR}/training-metrics.md.

  LOOP (every 60 seconds):
  1. Run: docker logs --tail 30 unsloth-headless 2>&1 | grep -i 'loss\|step\|epoch\|lr\|grad'
  2. Check for trainer state: docker exec unsloth-headless find /workspace/wp-finetune/adapters -name trainer_state.json -type f 2>/dev/null | sort -t- -k2 -n | tail -1 | xargs cat 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d[\"log_history\"][-3:], indent=2))' 2>/dev/null || true
  3. Check MLflow logs: docker exec unsloth-headless ls -la /workspace/wp-finetune/mlruns.db 2>/dev/null || true
  4. Parse and append to {TDIR}/training-metrics.md:
     ### {HH:MM:SS} -- Step {N}/{total}
     - Loss: {loss} (delta: {change from last})
     - Learning rate: {lr}
     - Gradient norm: {norm}
     - Router aux loss: {aux or N/A}
     - Epoch: {epoch}
     - Throughput: {samples/sec or tokens/sec if available}
  5. Flag WARNING if loss increases for 3+ consecutive readings
  6. Flag CRITICAL if loss > 10 or grad_norm > 100 (divergence/explosion)
  7. Check {TDIR}/_stop -- if so, write ## Final Summary (final loss, min loss, total steps, loss trend) and exit
  8. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no new training log output for 15+ minutes",
  run_in_background=true
)
```

#### Disk I/O Observer

```
Agent(
  model="sonnet",
  description="Telemetry: disk I/O",
  prompt="You are a disk I/O observer. Write observations to {TDIR}/disk-io.md.

  LOOP (every 60 seconds):
  1. Run: iostat -x 1 1 2>/dev/null || cat /proc/diskstats | head -10
  2. Run: sar -u 1 1 2>/dev/null || vmstat 1 2 | tail -1
  3. Run: df -h /home/robert_li/Desktop/projects/wp-finetune
  4. Run: du -sh adapters/ models/*-merged/ 2>/dev/null || true
  5. Run: docker stats --no-stream --format '{{.Name}}: Block I/O {{.BlockIO}}' unsloth-headless 2>/dev/null
  6. Append to {TDIR}/disk-io.md:
     ### {HH:MM:SS}
     - iowait: {pct}%
     - Disk util: {pct}%
     - Disk free: {free} / {total}
     - Adapter dir size: {size}
     - Container block I/O: {read} / {write}
  7. Flag WARNING if iowait > 20% or disk usage > 85%
  8. Flag CRITICAL if disk usage > 95%
  9. Check {TDIR}/_stop -- if so, write ## Final Summary (peak iowait, total disk consumed, IO trend) and exit
  10. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no disk activity changes for 15+ minutes",
  run_in_background=true
)
```

#### Checkpoint Integrity Observer

```
Agent(
  model="sonnet",
  description="Telemetry: checkpoint integrity",
  prompt="You are a checkpoint integrity observer. Write observations to {TDIR}/checkpoint-integrity.md.

  LOOP (every 5 minutes):
  1. List checkpoints: ls -ltd adapters/qwen3-30b-wp-*/checkpoint-*/ 2>/dev/null || docker exec unsloth-headless ls -ltd /workspace/adapters/qwen3-30b-wp-*/checkpoint-*/ 2>/dev/null
  2. For latest checkpoint, verify:
     - adapter_config.json exists and is valid JSON (python3 -c 'import json; json.load(open(\"...\"))')
     - adapter_model.safetensors exists and size > 0
     - optimizer.pt or optimizer.safetensors present
  3. Check tokenizer: ls adapters/tokenizer/ 2>/dev/null | grep -c 'tokenizer\|special_tokens'
  4. Check for merged model: ls models/*-merged/*.safetensors 2>/dev/null | wc -l
  5. Append to {TDIR}/checkpoint-integrity.md:
     ### {HH:MM:SS}
     - Checkpoints found: {N}
     - Latest: checkpoint-{step} ({size})
     - Config valid: {yes/no}
     - Safetensors valid: {yes/no} ({size})
     - Tokenizer files: {count}
     - Merged model: {not started / in progress / complete with N shards}
  6. Flag WARNING if no new checkpoint in 30+ minutes during active training
  7. Flag CRITICAL if latest checkpoint has 0-byte safetensors
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (total checkpoints, final adapter state, merge status) and exit
  9. Sleep 300 seconds, repeat

  STOP CONDITIONS: _stop file exists OR merged model verified complete",
  run_in_background=true
)
```

#### Container Monitor

```
Agent(
  model="sonnet",
  description="Telemetry: container monitor",
  prompt="You are a container health observer. Write observations to {TDIR}/container-monitor.md.

  LOOP (every 60 seconds):
  1. Run: docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null
  2. Run: docker stats --no-stream --format '{{.Name}}: CPU {{.CPUPerc}} MEM {{.MemUsage}} ({{.MemPerc}})' unsloth-headless 2>/dev/null
  3. Run: docker exec unsloth-headless ps aux --sort=-%mem 2>/dev/null | head -10
  4. Run: free -m (system RAM -- especially important on unified memory)
  5. Run: dmesg 2>/dev/null | tail -5 | grep -i 'oom\|kill\|error' || echo 'No OOM events'
  6. Append to {TDIR}/container-monitor.md:
     ### {HH:MM:SS}
     - Unsloth Headless: {status} | CPU: {pct} | MEM: {usage}
     - System RAM: {used}/{total} MiB ({pct}%) | Available: {available} MiB
     - Top processes: {list}
     - Other containers: {list or 'none'}
     - OOM events: {count or 'none'}
  7. Flag WARNING if unsloth-headless is not running
  8. Flag WARNING if system RAM available < 10GB
  9. Flag CRITICAL if OOM killer detected in dmesg
  10. Check {TDIR}/_stop -- if so, write ## Final Summary (uptime, peak container memory, OOM count) and exit
  11. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR container stopped for 10+ minutes",
  run_in_background=true
)
```

Report: "Training telemetry active with 6 Sonnet observers. Output: {TDIR}/. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

## Stopping Observers

To stop all observers (both lightweight and agent team):
```bash
touch {TDIR}/_stop
```
Each agent checks for this file on every cycle and will write a Final Summary before exiting. The lightweight monitor also checks each cycle and exits cleanly.

## Self-Recovery

The lightweight bash monitor survives Claude session restarts (it's a `nohup` process). On next invocation:

1. **Monitor still running:** Step 1 detects the PID via `monitor.pid` — user can keep it, restart, or start fresh
2. **Monitor crashed** (PID dead, no `_stop` file): Offer to restart in the same TDIR — appends to existing JSONL, no data loss
3. **Monitor completed** (`_stop` file exists or MAX_CHECKS reached): Start a new TDIR

For **agent team mode**: Sonnet agents die with the Claude session. On re-invocation, no PID is found — offer to restart agents in the same TDIR (existing `.md` reports are preserved, new readings append below) or start fresh.
