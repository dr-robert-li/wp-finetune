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

### 1. Create Run Directory

```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/inference/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

### 2. Spawn Request Latency Observer

```
Agent(
  description="Telemetry: request latency",
  prompt="You are a request latency observer. Write observations to {TDIR}/request-latency.md.

  LOOP (every 30 seconds):
  1. Probe vLLM health: curl -s -o /dev/null -w '%{time_total}' http://localhost:8020/health 2>/dev/null
  2. Probe LiteLLM health: curl -s -o /dev/null -w '%{time_total}' http://localhost:4000/health 2>/dev/null
  3. Probe Ollama: curl -s -o /dev/null -w '%{time_total}' http://localhost:11434/api/tags 2>/dev/null
  4. If vLLM is up, send a minimal completion and measure TTFT:
     time curl -s http://localhost:8020/v1/completions -d '{\"model\":\"wp-qwen3-moe\",\"prompt\":\"<wp_gen> \",\"max_tokens\":1}' 2>/dev/null
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

### 3. Spawn Throughput Observer

```
Agent(
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

### 4. Spawn GPU Utilization Observer

```
Agent(
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

### 5. Spawn Memory Observer

```
Agent(
  description="Telemetry: memory (inference)",
  prompt="You are a memory observer during inference. Write observations to {TDIR}/memory.md.

  LOOP (every 30 seconds):
  1. Run: free -h
  2. Run: docker stats --no-stream --format '{{.Name}}: MEM {{.MemUsage}} ({{.MemPerc}})' 2>/dev/null | grep -E 'vllm|ollama|litellm'
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

### 6. Spawn Error Rate Observer

```
Agent(
  description="Telemetry: error rates",
  prompt="You are an error rate observer. Write observations to {TDIR}/error-rates.md.

  LOOP (every 60 seconds):
  1. Check vLLM metrics for errors: curl -s http://localhost:8020/metrics 2>/dev/null | grep -E 'http_requests.*5[0-9][0-9]\|error'
  2. Check vLLM logs for errors: docker logs --tail 20 vllm 2>&1 | grep -i 'error\|exception\|traceback\|OOM' | tail -5
  3. Check LiteLLM logs: docker logs --tail 20 litellm 2>&1 | grep -i 'error\|exception\|429\|5[0-9][0-9]' | tail -5
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

### 7. Report

Tell the user: "Inference telemetry active with 5 observers. Output: {TDIR}/. Say 'stop observing' or touch {TDIR}/_stop to end."

## Stopping Observers

```bash
touch {TDIR}/_stop
```
