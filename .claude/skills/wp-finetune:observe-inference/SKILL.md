# Skill: wp-finetune:observe-inference

Spawn background observer agents to monitor model serving (vLLM, Ollama, LiteLLM). Writes telemetry to `telemetry/inference/{timestamp}/`.

**Agent team assessment:** Inference serving is network-facing, latency-sensitive, and long-running. Requires 5-agent team: request-latency, throughput, gpu-utilization, memory, error-rates.

> When creating new skills that involve model serving, assess whether this agent team needs modification:
> - Uses GPU? Yes -> gpu-utilization included
> - Runs > 30 min? Yes (serving is continuous) -> all agents run indefinitely
> - Writes large files? No -> disk-io not needed
> - Runs in Docker? Yes -> covered by memory agent (docker stats)
> - Has checkpoints? No
> - Serves network? Yes -> request-latency, throughput, error-rates all included
> - Memory-sensitive? Yes -> dedicated memory agent

## Trigger

User says: "observe inference", "monitor serving", "monitor inference", "/observe-inference"

## Process

### 1. Create or Resume Run Directory

**Self-recovery:** Check for an existing active monitor before creating a new directory.

```bash
# Check for running lightweight monitor from a previous session
ACTIVE_PID=""
ACTIVE_DIR=""
for pidfile in telemetry/inference/*/monitor.pid; do
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
  - "Keep it running" -- Monitor is healthy, just report its location
  - "Restart in same directory" -- Kill and restart (appends to existing JSONL)
  - "Start fresh" -- New timestamped directory

**If no active monitor but an incomplete run exists** (TDIR with monitor.jsonl but no `_stop` and dead PID -- monitor crashed):
- Offer to **restart** in the same directory (appends to existing JSONL, no data loss)

**If no previous monitor:** Create fresh:
```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/inference/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all steps below.

### 2. Choose Monitoring Mode

Use AskUserQuestion:
- header: "Monitor mode"
- question: "Which monitoring level for this inference run?"
- options:
  - "Lightweight (Recommended)" -- Single bash script, ~5MB, writes JSONL. Safe on DGX Spark with tight memory
  - "Full agent team" -- 5 Sonnet agents writing detailed markdown reports. Adds ~2.0GB memory overhead. Use only with >25GB headroom

### 3a. Lightweight Monitor (default)

Generate and write the following script to `{TDIR}/monitor.sh`, then start it:

```bash
#!/usr/bin/env bash
set -euo pipefail
TDIR="$1"
JSONL="${TDIR}/monitor.jsonl"
MAX_CHECKS=${2:-504}  # 504 * 30s = 4.2 hours

for (( i=1; i<=MAX_CHECKS; i++ )); do
    [[ -f "${TDIR}/_stop" ]] && echo "Stop signal. Exiting after $((i-1)) checks." && exit 0

    gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0")
    gpu_util=$(echo "$gpu_raw" | awk -F', ' '{print int($1)}')
    temp=$(echo "$gpu_raw" | awk -F', ' '{print int($2)}')
    watts=$(echo "$gpu_raw" | awk -F', ' '{printf "%.1f", $3}')
    mem_line=$(free -m | grep Mem)
    available=$(echo "$mem_line" | awk '{print $7}')
    mem_available_gb=$(awk "BEGIN {printf \"%.1f\", $available / 1024}")

    # Inference-specific: check vLLM health + latency probe
    vllm_status=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8020/health 2>/dev/null || echo "000")
    vllm_latency=$(curl -s -o /dev/null -w '%{time_total}' http://localhost:8020/health 2>/dev/null || echo "0")
    # Dynamic model discovery for TTFT probe
    model_name=$(curl -s http://localhost:8020/v1/models 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
    if [ -n "$model_name" ]; then
        ttft=$(curl -s -o /dev/null -w '%{time_total}' http://localhost:8020/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$model_name\",\"prompt\":\"<wp_gen> \",\"max_tokens\":1}" 2>/dev/null || echo "0")
    else
        ttft="0"
    fi
    # Error count from docker logs
    error_count=$(docker logs --tail 50 vllm 2>&1 | grep -ci 'error\|exception\|traceback' || echo "0")

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"ts\": \"$ts\", \"watts\": $watts, \"temperature_c\": $temp, \"gpu_util_pct\": $gpu_util, \"mem_available_gb\": $mem_available_gb, \"vllm_status\": $vllm_status, \"vllm_latency_s\": $vllm_latency, \"ttft_s\": $ttft, \"model\": \"$model_name\", \"error_count\": $error_count, \"source\": \"monitor\"}" >> "$JSONL"
    echo "$ts Check $i/$MAX_CHECKS | temp=${temp}C gpu=${gpu_util}% ram=${mem_available_gb}GB | vllm=$vllm_status latency=${vllm_latency}s ttft=${ttft}s errors=$error_count"

    (( i < MAX_CHECKS )) && sleep 30
done
```

Start the monitor:

```bash
chmod +x {TDIR}/monitor.sh
nohup bash {TDIR}/monitor.sh "{TDIR}" > {TDIR}/monitor.log 2>&1 &
echo $! > {TDIR}/monitor.pid
```

Report: "Inference lightweight monitor active. PID: {pid}. Output: {TDIR}/monitor.jsonl. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

### 3b. Full Agent Team (optional, sonnet only)

Spawn agents with EXPLICIT `model="sonnet"` parameter. This is critical -- haiku agents do not persist loops.

#### Request Latency Observer

```
Agent(
  model="sonnet",
  description="Telemetry: request latency",
  prompt="You are a request latency observer. Write observations to {TDIR}/request-latency.md.

  LOOP (every 30 seconds):
  1. Probe vLLM health: curl -s -o /dev/null -w '%{time_total}' http://localhost:8020/health 2>/dev/null
  2. Probe LiteLLM health: curl -s -o /dev/null -w '%{time_total}' http://localhost:4000/health 2>/dev/null
  3. Probe Ollama: curl -s -o /dev/null -w '%{time_total}' http://localhost:11434/api/tags 2>/dev/null
  4. If vLLM is up, discover the model name and send a minimal completion to measure TTFT:
     MODEL=$(curl -s http://localhost:8020/v1/models | python3 -c \"import json,sys; print(json.load(sys.stdin)['data'][0]['id'])\" 2>/dev/null || echo \"\")
     if [ -n \"$MODEL\" ]; then
       time curl -s http://localhost:8020/v1/completions -d \"{\\\"model\\\":\\\"$MODEL\\\",\\\"prompt\\\":\\\"<wp_gen> \\\",\\\"max_tokens\\\":1}\" 2>/dev/null
     else
       echo \"Model discovery failed -- skipping TTFT probe\"
     fi
  5. Append to {TDIR}/request-latency.md:
     ### {HH:MM:SS}
     - vLLM health: {latency}ms ({status})
     - LiteLLM health: {latency}ms ({status})
     - Ollama health: {latency}ms ({status})
     - TTFT (time to first token): {latency}ms
  6. Flag WARNING if any health check > 500ms
  7. Flag CRITICAL if any endpoint is down
  8. Check {TDIR}/_stop -- if so, write ## Final Summary (avg/p50/p95/p99 latency) and exit
  9. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR all endpoints down for 10+ minutes",
  run_in_background=true
)
```

#### Throughput Observer

```
Agent(
  model="sonnet",
  description="Telemetry: throughput",
  prompt="You are a throughput observer. Write observations to {TDIR}/throughput.md.

  LOOP (every 30 seconds):
  1. Check vLLM Prometheus metrics: curl -s http://localhost:8020/metrics 2>/dev/null | grep -E 'vllm:num_requests|vllm:generation_tokens|vllm:prompt_tokens|vllm:avg_generation_throughput'
  2. If Prometheus not available, parse vLLM logs: docker logs --tail 10 vllm 2>&1 | grep -i 'tok/s\|throughput'
  3. Append to {TDIR}/throughput.md:
     ### {HH:MM:SS}
     - Active requests: {N}
     - Generation throughput: {tok/s}
     - Prompt throughput: {tok/s}
     - Total requests served: {N}
  4. Flag WARNING if throughput drops > 50% from baseline
  5. Check {TDIR}/_stop -- if so, write ## Final Summary (avg throughput, peak throughput, total requests) and exit
  6. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR server stopped",
  run_in_background=true
)
```

#### GPU Utilization Observer

```
Agent(
  model="sonnet",
  description="Telemetry: GPU utilization (inference)",
  prompt="You are a GPU utilization observer during inference. Write observations to {TDIR}/gpu-utilization.md.

  LOOP (every 30 seconds):
  1. Run: nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
  2. Run: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  3. Append to {TDIR}/gpu-utilization.md:
     ### {HH:MM:SS}
     - Memory: {used} MiB | Util: {pct}% | Temp: {temp}C | Power: {watts}W
     - Processes: {list}
  4. Flag WARNING if temp > 80C or utilization sustained > 95% (may need scaling)
  5. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  6. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no GPU processes for 10+ minutes",
  run_in_background=true
)
```

#### Memory Observer

```
Agent(
  model="sonnet",
  description="Telemetry: memory (inference)",
  prompt="You are a memory observer during inference. Write observations to {TDIR}/memory.md.

  LOOP (every 30 seconds):
  1. Run: free -h
  2. Run: docker stats --no-stream --format '{{.Name}}: MEM {{.MemUsage}} ({{.MemPerc}})' 2>/dev/null | grep -E 'vllm|ollama|litellm'  (ollama/litellm are optional -- may not be running)
  3. Run: nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  4. Append to {TDIR}/memory.md:
     ### {HH:MM:SS}
     - System RAM: {used}/{total}
     - GPU Memory: {used}/{total} MiB
     - vLLM container: {mem usage}
     - Ollama container: {mem usage}
     - Swap: {used}
  5. Flag WARNING if system RAM > 85% or GPU memory > 90%
  6. Flag CRITICAL if swap > 10GB (memory pressure)
  7. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  8. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR servers stopped",
  run_in_background=true
)
```

#### Error Rate Observer

```
Agent(
  model="sonnet",
  description="Telemetry: error rates",
  prompt="You are an error rate observer. Write observations to {TDIR}/error-rates.md.

  LOOP (every 60 seconds):
  1. Check vLLM metrics for errors: curl -s http://localhost:8020/metrics 2>/dev/null | grep -E 'http_requests.*5[0-9][0-9]\|error'
  2. Check vLLM logs for errors: docker logs --tail 20 vllm 2>&1 | grep -i 'error\|exception\|traceback\|OOM' | tail -5
  3. Check LiteLLM logs (optional -- only if litellm container is running): docker logs --tail 20 litellm 2>&1 | grep -i 'error\|exception\|429\|5[0-9][0-9]' | tail -5
  4. Append to {TDIR}/error-rates.md:
     ### {HH:MM:SS}
     - HTTP 5xx count: {N}
     - OOM events: {N}
     - Error log lines: {count}
     - Recent errors: {last error or 'none'}
  5. Flag WARNING if any 5xx errors detected
  6. Flag CRITICAL if OOM or repeated errors (3+ in one cycle)
  7. Check {TDIR}/_stop -- if so, write ## Final Summary (total errors, error rate, categories) and exit
  8. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR servers stopped",
  run_in_background=true
)
```

Report: "Inference telemetry active with 5 Sonnet observers. Output: {TDIR}/. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

## Stopping Observers

To stop all observers (both lightweight and agent team):
```bash
touch {TDIR}/_stop
```
Each agent checks for this file on every cycle and will write a Final Summary before exiting. The lightweight monitor also checks each cycle and exits cleanly.

Run /review-telemetry to consolidate results.

## Self-Recovery

The lightweight bash monitor survives Claude session restarts (it's a `nohup` process). On next invocation:

1. **Monitor still running:** Step 1 detects the PID via `monitor.pid` -- user can keep it, restart, or start fresh
2. **Monitor crashed** (PID dead, no `_stop` file): Offer to restart in the same TDIR -- appends to existing JSONL, no data loss
3. **Monitor completed** (`_stop` file exists or MAX_CHECKS reached): Start a new TDIR

For **agent team mode**: Sonnet agents die with the Claude session. On re-invocation, no PID is found -- offer to restart agents in the same TDIR (existing `.md` reports are preserved, new readings append below) or start fresh.
