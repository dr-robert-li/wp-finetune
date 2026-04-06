# Skill: wp-finetune:observe-data-pipeline

Spawn background observer agents to monitor data pipeline execution. Writes telemetry to `telemetry/data-pipeline/{timestamp}/`.

**Agent team assessment:** Data pipeline is CPU-bound (no GPU), long-running file operations, no Docker dependency. Requires 3-agent team: pipeline-progress, system-resources, disk-io.

> When creating new skills that involve data processing, assess whether this agent team needs modification:
> - Uses GPU? No -> gpu-metrics not needed
> - Runs > 30 min? Yes -> system-resources included
> - Writes large files? Yes -> disk-io included
> - Runs in Docker? No -> container-monitor not needed
> - Has checkpoints? Yes -> covered by pipeline-progress (orchestrator tracks state)
> - Has progress metric? Yes -> pipeline-progress uses orchestrator status

## Trigger

User says: "observe data pipeline", "monitor pipeline", "/observe-data-pipeline"

## Process

### 1. Create or Resume Run Directory

**Self-recovery:** Check for an existing active monitor before creating a new directory.

```bash
# Check for running lightweight monitor from a previous session
ACTIVE_PID=""
ACTIVE_DIR=""
for pidfile in telemetry/data-pipeline/*/monitor.pid; do
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
TDIR="telemetry/data-pipeline/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all steps below.

### 2. Choose Monitoring Mode

Use AskUserQuestion:
- header: "Monitor mode"
- question: "Which monitoring level for this data pipeline run?"
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
MAX_CHECKS=${2:-360}  # 360 * 120s = 12 hours

for (( i=1; i<=MAX_CHECKS; i++ )); do
    [[ -f "${TDIR}/_stop" ]] && echo "Stop signal. Exiting after $((i-1)) checks." && exit 0

    # System resources (no GPU for data pipeline)
    mem_line=$(free -m | grep Mem)
    available=$(echo "$mem_line" | awk '{print $7}')
    mem_available_gb=$(awk "BEGIN {printf \"%.1f\", $available / 1024}")
    disk_free=$(df -BG /home/robert_li/Desktop/projects/wp-finetune 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')

    # Pipeline-specific: file counts per stage
    passed=$(ls data/phase1_extraction/output/passed/*.json 2>/dev/null | wc -l || echo "0")
    synth_passed=$(ls data/phase2_synthetic/output/judged/passed_*.json 2>/dev/null | wc -l || echo "0")
    synth_failed=$(ls data/phase2_synthetic/output/judged/failed_*.json 2>/dev/null | wc -l || echo "0")
    judge_training=$(ls data/phase2_synthetic/output/judge_training/*.json 2>/dev/null | wc -l || echo "0")
    cot=$(ls data/phase3_cot/output/*.json 2>/dev/null | wc -l || echo "0")
    exports=$(ls data/final_dataset/*.jsonl 2>/dev/null | wc -l || echo "0")
    # Checkpoint state
    ckpt_count=$(ls data/checkpoints/*.json 2>/dev/null | wc -l || echo "0")

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    # Canonical fields use watts=0, temperature_c=0, gpu_util_pct=0 for CPU-only pipeline
    echo "{\"ts\": \"$ts\", \"watts\": 0, \"temperature_c\": 0, \"gpu_util_pct\": 0, \"mem_available_gb\": $mem_available_gb, \"disk_free_gb\": $disk_free, \"passed\": $passed, \"synth_passed\": $synth_passed, \"synth_failed\": $synth_failed, \"judge_training\": $judge_training, \"cot\": $cot, \"exports\": $exports, \"checkpoints\": $ckpt_count, \"source\": \"monitor\"}" >> "$JSONL"
    echo "$ts Check $i/$MAX_CHECKS | ram=${mem_available_gb}GB disk=${disk_free}GB | passed=$passed synth=$synth_passed/$synth_failed judge=$judge_training cot=$cot exports=$exports ckpts=$ckpt_count"

    (( i < MAX_CHECKS )) && sleep 120
done
```

Start the monitor:

```bash
chmod +x {TDIR}/monitor.sh
nohup bash {TDIR}/monitor.sh "{TDIR}" > {TDIR}/monitor.log 2>&1 &
echo $! > {TDIR}/monitor.pid
```

Report: "Data pipeline lightweight monitor active. PID: {pid}. Output: {TDIR}/monitor.jsonl. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

### 3b. Full Agent Team (optional, sonnet only)

Spawn agents with EXPLICIT `model="sonnet"` parameter. This is critical -- haiku agents do not persist loops.

#### Pipeline Progress Observer

```
Agent(
  model="sonnet",
  description="Telemetry: pipeline progress",
  prompt="You are a pipeline progress observer. Write observations to {TDIR}/pipeline-progress.md.

  LOOP (every 2 minutes):
  1. Run: python scripts/pipeline_orchestrator.py status
  2. Check checkpoint state: ls data/checkpoints/*.json 2>/dev/null (shows resume markers for clone, extract, judge)
  3. Count files: ls data/phase1_extraction/output/passed/*.json 2>/dev/null | wc -l
  4. Count passed+failed: ls data/phase2_synthetic/output/judged/passed_*.json 2>/dev/null | wc -l (passed); ls data/phase2_synthetic/output/judged/failed_*.json 2>/dev/null | wc -l (failed)
  5. Count files: ls data/phase2_synthetic/output/judge_training/*.json 2>/dev/null | wc -l
  6. Count files: ls data/phase3_cot/output/*.json 2>/dev/null | wc -l
  7. Check exports: ls data/final_dataset/*.jsonl 2>/dev/null | wc -l
  8. Append to {TDIR}/pipeline-progress.md:
     ### {HH:MM:SS}
     - Phase: {current phase from status}
     - Passed repos: {N} (+{delta})
     - Checkpoint state: {clone/extract/judge resume markers present or absent}
     - Synthetic judged: {passed}/{failed} (+{delta})
     - Judge training: {N} (+{delta})
     - CoT examples: {N} (+{delta})
     - Targets met: {yes/no}
  9. Flag WARNING if no file count changes for 10+ minutes during active pipeline
  10. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  11. Sleep 120 seconds, repeat

  STOP CONDITIONS: _stop file exists OR all targets met (orchestrator reports complete)",
  run_in_background=true
)
```

#### System Resources Observer

```
Agent(
  model="sonnet",
  description="Telemetry: system resources",
  prompt="You are a system resources observer. Write observations to {TDIR}/system-resources.md.

  LOOP (every 60 seconds):
  1. Run: free -h
  2. Run: uptime
  3. Run: df -h /home/robert_li/Desktop/projects/wp-finetune
  4. Run: ps aux --sort=-%mem | head -8
  5. Append to {TDIR}/system-resources.md:
     ### {HH:MM:SS}
     - RAM: {used}/{total} ({pct}%)
     - Swap: {used}/{total}
     - Load avg: {1m} {5m} {15m}
     - Disk: {used}/{total} ({pct}%)
     - Top process: {name} ({mem})
  6. Flag WARNING if RAM > 80% or disk > 90% or load > 2x CPU count
  7. Check {TDIR}/_stop -- if so, write ## Final Summary (peak RAM, peak load, min free disk) and exit
  8. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR pipeline complete",
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
  3. Run: du -sh data/phase1_extraction/output/ data/phase2_synthetic/output/ data/phase3_cot/output/ data/final_dataset/ 2>/dev/null
  4. Append to {TDIR}/disk-io.md:
     ### {HH:MM:SS}
     - iowait: {pct}%
     - Disk util: {pct}%
     - Data sizes: phase1={size} phase2={size} phase3={size} final={size}
  5. Flag WARNING if iowait > 20% or disk > 85%
  6. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  7. Sleep 60 seconds, repeat

  STOP CONDITIONS: _stop file exists OR pipeline complete",
  run_in_background=true
)
```

Report: "Data pipeline telemetry active with 3 Sonnet observers. Output: {TDIR}/. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

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
