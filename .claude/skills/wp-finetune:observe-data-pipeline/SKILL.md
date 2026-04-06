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

### 1. Create Run Directory

```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/data-pipeline/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

### 2. Spawn Pipeline Progress Observer

```
Agent(
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

### 3. Spawn System Resources Observer

```
Agent(
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

### 4. Spawn Disk I/O Observer

```
Agent(
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

### 5. Report

Tell the user: "Data pipeline telemetry active with 3 observers. Output: {TDIR}/. Say 'stop observing' or touch {TDIR}/_stop to end."

## Stopping Observers

```bash
touch {TDIR}/_stop
```
