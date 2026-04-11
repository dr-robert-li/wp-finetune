# Skill: wp-finetune:observe-evaluation

Spawn background monitoring for model evaluation. Writes telemetry to `telemetry/evaluation/{timestamp}/`.

**Agent team assessment:** Evaluation runs GPU inference (wp-bench, PHPCS eval), produces result files, and has a quality gate. Full agent team uses 3 Sonnet agents: eval-progress, gpu-metrics, result-tracking.

> When creating new skills that involve evaluation, assess whether this agent team needs modification:
> - Uses GPU? Yes -> gpu-metrics included
> - Runs > 30 min? Possibly -> covered by gpu-metrics cycle
> - Writes large files? No -> disk-io not needed (eval outputs are small)
> - Runs in Docker? Possibly (eval-toolbox container) -> add container-monitor if so
> - Has checkpoints? No
> - Has progress metric? Yes -> eval-progress tracks script completion
> - Serves network? No (inference is local for eval)

## Trigger

User says: "observe evaluation", "monitor eval", "/observe-evaluation"

## Process

### 1. Create or Resume Run Directory

**Self-recovery:** Check for an existing active monitor before creating a new directory.

```bash
# Check for running lightweight monitor from a previous session
ACTIVE_PID=""
ACTIVE_DIR=""
for pidfile in telemetry/evaluation/*/monitor.pid; do
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
TDIR="telemetry/evaluation/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all steps below.

### 2. Choose Monitoring Mode

Use AskUserQuestion:
- header: "Monitor mode"
- question: "Which monitoring level for this evaluation run?"
- options:
  - "Lightweight (Recommended)" -- Single bash script, ~5MB, writes JSONL. Safe on DGX Spark with tight memory
  - "Full agent team" -- 3 Sonnet agents writing detailed markdown reports. Adds ~1.2GB memory overhead. Use only with >25GB headroom

### 3a. Lightweight Monitor (default)

Generate and write the following script to `{TDIR}/monitor.sh`, then start it:

```bash
#!/usr/bin/env bash
set -euo pipefail
TDIR="$1"
JSONL="${TDIR}/monitor.jsonl"
MAX_CHECKS=${2:-252}  # 252 * 60s = 4.2 hours

for (( i=1; i<=MAX_CHECKS; i++ )); do
    [[ -f "${TDIR}/_stop" ]] && echo "Stop signal. Exiting after $((i-1)) checks." && exit 0

    gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0")
    gpu_util=$(echo "$gpu_raw" | awk -F', ' '{print int($1)}')
    temp=$(echo "$gpu_raw" | awk -F', ' '{print int($2)}')
    watts=$(echo "$gpu_raw" | awk -F', ' '{printf "%.1f", $3}')
    mem_line=$(free -m | grep Mem)
    available=$(echo "$mem_line" | awk '{print $7}')
    mem_available_gb=$(awk "BEGIN {printf \"%.1f\", $available / 1024}")

    # Eval-specific: check which ratios have results
    eval_gen_done=$(ls output/eval_triage/*/eval_gen_results.json 2>/dev/null | wc -l || echo "0")
    eval_judge_done=$(ls output/eval_triage/*/eval_judge_results.json 2>/dev/null | wc -l || echo "0")
    wpbench_done=$(ls output/eval_triage/*/wp_bench_results.json 2>/dev/null | wc -l || echo "0")
    triage_done=$([[ -f output/triage_decision.md ]] && echo "true" || echo "false")
    # Check vLLM
    vllm_status=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8020/health 2>/dev/null || echo "000")

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"ts\": \"$ts\", \"watts\": $watts, \"temperature_c\": $temp, \"gpu_util_pct\": $gpu_util, \"mem_available_gb\": $mem_available_gb, \"eval_gen_done\": $eval_gen_done, \"eval_judge_done\": $eval_judge_done, \"wpbench_done\": $wpbench_done, \"triage_done\": $triage_done, \"vllm_status\": $vllm_status, \"source\": \"monitor\"}" >> "$JSONL"
    echo "$ts Check $i/$MAX_CHECKS | temp=${temp}C gpu=${gpu_util}% ram=${mem_available_gb}GB | gen=$eval_gen_done judge=$eval_judge_done wpbench=$wpbench_done triage=$triage_done vllm=$vllm_status"

    [[ $i -lt $MAX_CHECKS ]] && sleep 60 || true
done
```

Start the monitor:

```bash
chmod +x {TDIR}/monitor.sh
nohup bash {TDIR}/monitor.sh "{TDIR}" > {TDIR}/monitor.log 2>&1 &
echo $! > {TDIR}/monitor.pid
```

Report: "Evaluation lightweight monitor active. PID: {pid}. Output: {TDIR}/monitor.jsonl. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

### 3b. Full Agent Team (optional, sonnet only)

Spawn agents with EXPLICIT `model="sonnet"` parameter. This is critical -- haiku agents do not persist loops.

#### Eval Progress Observer

```
Agent(
  model="sonnet",
  description="Telemetry: eval progress",
  prompt="You are an evaluation progress observer. Write observations to {TDIR}/eval-progress.md.

  LOOP (every 2 minutes):
  1. Check running eval processes: ps aux | grep -E 'run_eval_triage|vllm|wp-bench' | grep -v grep
  2. Check eval output files: ls -la output/eval_triage/ 2>/dev/null
  3. Check wp-bench results: ls -la output/eval_triage/*/wp_bench_results.json 2>/dev/null
  4. Check git log for eval commits: git log --oneline -5 | grep -i eval
  5. Append to {TDIR}/eval-progress.md:
     ### {HH:MM:SS}
     - Running: {process list or 'idle'}
     - eval_gen: {not started / running / complete}
     - eval_judge: {not started / running / complete}
     - wp-bench: {not started / running / complete}
     - eval_gate: {not started / running / complete}
  6. Flag WARNING if any eval script errors in output
  7. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  8. Sleep 120 seconds, repeat

  STOP CONDITIONS: _stop file exists OR eval_gate complete",
  run_in_background=true
)
```

#### GPU Metrics Observer

```
Agent(
  model="sonnet",
  description="Telemetry: GPU metrics (eval)",
  prompt="You are a GPU metrics observer during evaluation. Write observations to {TDIR}/gpu-metrics.md.

  LOOP (every 30 seconds):
  1. Run: nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
  2. Run: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  3. Append to {TDIR}/gpu-metrics.md:
     ### {HH:MM:SS}
     - Memory: {used} MiB | GPU Util: {pct}% | Temp: {temp}C | Power: {watts}W
     - Processes: {list}
  4. Flag WARNING if temp > 80C
  5. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  6. Sleep 30 seconds, repeat

  STOP CONDITIONS: _stop file exists OR no GPU processes for 10+ minutes",
  run_in_background=true
)
```

#### Result Tracking Observer

```
Agent(
  model="sonnet",
  description="Telemetry: result tracking",
  prompt="You are an eval result tracking observer. Write observations to {TDIR}/result-tracking.md.

  LOOP (every 2 minutes):
  1. Read eval_gen results if they exist: cat output/eval_triage/*/eval_gen_results.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"PHPCS pass: {d.get(\"phpcs_pass_rate\", \"N/A\")}, Security: {d.get(\"security_pass_rate\", \"N/A\")}\")' 2>/dev/null
  2. Read eval_judge results: cat output/eval_triage/*/eval_judge_results.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"Spearman: {d.get(\"spearman_corr\", \"N/A\")}\")' 2>/dev/null
  3. Read eval_gate results: cat output/eval_triage/*/eval_gate_results.json 2>/dev/null
  4. Append to {TDIR}/result-tracking.md:
     ### {HH:MM:SS}
     - Generator PHPCS pass rate: {value or pending}
     - Generator security pass rate: {value or pending}
     - Judge Spearman correlation: {value or pending}
     - Quality gate: {PASS / FAIL / pending}
  5. Flag WARNING if any metric below target (PHPCS < 95%, Spearman < 0.85, security < 98%)
  6. Flag CRITICAL if quality gate = FAIL
  7. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  8. Sleep 120 seconds, repeat

  STOP CONDITIONS: _stop file exists OR quality gate result written",
  run_in_background=true
)
```

Report: "Evaluation telemetry active with 3 Sonnet observers. Output: {TDIR}/. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

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
