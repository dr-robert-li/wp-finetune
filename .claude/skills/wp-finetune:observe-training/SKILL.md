# Skill: wp-finetune:observe-training

Spawn background observer agents to monitor DGX Spark training. Writes telemetry to `telemetry/training/{timestamp}/`.

**Agent team assessment:** Training is GPU-heavy (6-12hr), runs in Docker, produces checkpoints, and has training metrics. This requires the full 6-agent team: gpu-metrics, thermal-throttling, training-metrics, disk-io, checkpoint-integrity, container-monitor.

> When creating new skills that involve training, assess whether this agent team needs modification:
> - Uses GPU? Yes -> gpu-metrics, thermal-throttling included
> - Runs > 30 min? Yes -> system resources covered by disk-io + container-monitor
> - Writes large files? Yes -> disk-io included
> - Runs in Docker? Yes -> container-monitor included
> - Has checkpoints? Yes -> checkpoint-integrity included
> - Has progress metric? Yes -> training-metrics included

## Trigger

User says: "observe training", "monitor training", "/observe-training"

## Process

### 1. Create Run Directory

```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/training/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all agent prompts below.

### 2. Spawn GPU Metrics Observer

```
Agent(
  description="Telemetry: GPU metrics",
  prompt="You are a GPU metrics observer. Write observations to {TDIR}/gpu-metrics.md.

  LOOP (every 30 seconds):
  1. Run: nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,clocks.current.sm --format=csv,noheader,nounits
  2. Run: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  3. Append to {TDIR}/gpu-metrics.md:
     ### {HH:MM:SS}
     - Memory: {used}/{total} MiB ({pct}%)
     - GPU Util: {util}%
     - Mem Util: {mem_util}%
     - SM Clock: {clock} MHz
     - Processes: {list}
  4. Flag CRITICAL if memory.used > 120000 MiB
  5. Flag WARNING if utilization.gpu < 50% for 3+ consecutive readings
  6. Check if {TDIR}/_stop exists -- if so, write ## Final Summary (peak memory, avg util, total readings) and exit
  7. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no GPU compute processes for 15+ minutes",
  run_in_background=true
)
```

### 3. Spawn Thermal/Throttling Observer

```
Agent(
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
  5. Flag WARNING if temp > 80C
  6. Flag CRITICAL if temp > 85C or any throttle reason is Active
  7. Flag WARNING if power.draw > 90% of power.limit
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (peak temp, throttle event count, avg power) and exit
  9. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no GPU activity for 15+ minutes",
  run_in_background=true
)
```

### 4. Spawn Training Metrics Observer

```
Agent(
  description="Telemetry: training metrics",
  prompt="You are a training metrics observer. Write observations to {TDIR}/training-metrics.md.

  LOOP (every 60 seconds):
  1. Run: docker logs --tail 30 unsloth-headless 2>&1 | grep -i 'loss\|step\|epoch\|lr\|grad'
  2. Check for trainer state: docker exec unsloth-headless cat /workspace/outputs/trainer_state.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d[\"log_history\"][-3:], indent=2))' 2>/dev/null || true
  3. Check W&B: docker exec unsloth-headless ls -la /workspace/wandb/latest-run/ 2>/dev/null || true
  4. Check TensorBoard: docker exec unsloth-headless find /workspace -name 'events.out.tfevents*' -newer /tmp/.last_tb_check 2>/dev/null || true
  5. Parse and append to {TDIR}/training-metrics.md:
     ### {HH:MM:SS} -- Step {N}/{total}
     - Loss: {loss} (delta: {change from last})
     - Learning rate: {lr}
     - Gradient norm: {norm}
     - Router aux loss: {aux or N/A}
     - Epoch: {epoch}
     - Throughput: {samples/sec or tokens/sec if available}
  6. Flag WARNING if loss increases for 3+ consecutive readings
  7. Flag CRITICAL if loss > 10 or grad_norm > 100 (divergence/explosion)
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (final loss, min loss, total steps, loss trend) and exit
  9. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no new training log output for 15+ minutes",
  run_in_background=true
)
```

### 5. Spawn Disk I/O Observer

```
Agent(
  description="Telemetry: disk I/O",
  prompt="You are a disk I/O observer. Write observations to {TDIR}/disk-io.md.

  LOOP (every 60 seconds):
  1. Run: iostat -x 1 1 2>/dev/null || cat /proc/diskstats | head -10
  2. Run: sar -u 1 1 2>/dev/null || vmstat 1 2 | tail -1
  3. Run: df -h /home/robert_li/Desktop/projects/wp-finetune
  4. Run: du -sh adapters/ merged_model/ 2>/dev/null || true
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

### 6. Spawn Checkpoint Integrity Observer

```
Agent(
  description="Telemetry: checkpoint integrity",
  prompt="You are a checkpoint integrity observer. Write observations to {TDIR}/checkpoint-integrity.md.

  LOOP (every 5 minutes):
  1. List checkpoints: ls -ltd adapters/qwen3-wp/checkpoint-*/ 2>/dev/null || docker exec unsloth-headless ls -ltd /workspace/adapters/qwen3-wp/checkpoint-*/ 2>/dev/null
  2. For latest checkpoint, verify:
     - adapter_config.json exists and is valid JSON (python3 -c 'import json; json.load(open(\"...\"))')
     - adapter_model.safetensors exists and size > 0
     - optimizer.pt or optimizer.safetensors present
  3. Check tokenizer: ls adapters/tokenizer/ 2>/dev/null | grep -c 'tokenizer\|special_tokens'
  4. Check for merged model: ls merged_model/*.safetensors 2>/dev/null | wc -l
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

### 7. Spawn Container Monitor

```
Agent(
  description="Telemetry: container monitor",
  prompt="You are a container health observer. Write observations to {TDIR}/container-monitor.md.

  LOOP (every 60 seconds):
  1. Run: docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null
  2. Run: docker stats --no-stream --format '{{.Name}}: CPU {{.CPUPerc}} MEM {{.MemUsage}} ({{.MemPerc}})' unsloth-headless 2>/dev/null
  3. Run: docker exec unsloth-headless ps aux --sort=-%mem 2>/dev/null | head -10
  4. Run: dmesg 2>/dev/null | tail -5 | grep -i 'oom\|kill\|error' || echo 'No OOM events'
  5. Append to {TDIR}/container-monitor.md:
     ### {HH:MM:SS}
     - Unsloth Headless: {status} | CPU: {pct} | MEM: {usage}
     - Top processes: {list}
     - Other containers: {list or 'none'}
     - OOM events: {count or 'none'}
  6. Flag WARNING if unsloth-headless is not running
  7. Flag CRITICAL if OOM killer detected in dmesg
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (uptime, peak container memory, OOM count) and exit
  9. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR container stopped for 10+ minutes",
  run_in_background=true
)
```

### 8. Report

Tell the user: "Training telemetry active with 6 observers. Output: {TDIR}/. Say 'stop observing' or touch {TDIR}/_stop to end. Run /review-telemetry to check status."

## Stopping Observers

To stop all observers:
```bash
touch {TDIR}/_stop
```
Each agent checks for this file on every cycle and will write a Final Summary before exiting.
