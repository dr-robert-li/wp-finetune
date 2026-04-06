# Skill: wp-finetune:observe-packaging

Spawn background observer agents to monitor model packaging (quantization, GGUF export). Writes telemetry to `telemetry/packaging/{timestamp}/`.

**Agent team assessment:** Packaging is file-heavy and integrity-critical. AWQ quantization uses GPU; GGUF conversion is CPU-heavy. Requires 3-agent team: quantization-progress, file-integrity, size-tracking.

> When creating new skills that involve model packaging, assess whether this agent team needs modification:
> - Uses GPU? Yes (AWQ) -> gpu metrics embedded in quantization-progress
> - Runs > 30 min? Yes -> covered by progress observer cycle
> - Writes large files? Yes -> size-tracking included
> - Runs in Docker? Possibly -> add container-monitor if containerized
> - Has checkpoints? No (quantization is atomic per format)
> - Has progress metric? Yes -> quantization-progress tracks format completion
> - Produces final artifacts? Yes -> file-integrity is critical
>
> **Note:** Packaging (v3.0 Phase 14) has not been implemented yet. The `quantized/` directory structure referenced below is aspirational -- actual output paths will be defined when the packaging scripts are written. All `quantized/` checks use `2>/dev/null` and will gracefully report nothing until packaging runs.

## Trigger

User says: "observe packaging", "monitor packaging", "monitor quantization", "/observe-packaging"

## Process

### 1. Create or Resume Run Directory

**Self-recovery:** Check for an existing active monitor before creating a new directory.

```bash
# Check for running lightweight monitor from a previous session
ACTIVE_PID=""
ACTIVE_DIR=""
for pidfile in telemetry/packaging/*/monitor.pid; do
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
TDIR="telemetry/packaging/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

Save TDIR for use in all steps below.

### 2. Choose Monitoring Mode

Use AskUserQuestion:
- header: "Monitor mode"
- question: "Which monitoring level for this packaging run?"
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
MAX_CHECKS=${2:-180}  # 180 * 120s = 6 hours

for (( i=1; i<=MAX_CHECKS; i++ )); do
    [[ -f "${TDIR}/_stop" ]] && echo "Stop signal. Exiting after $((i-1)) checks." && exit 0

    gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0")
    gpu_util=$(echo "$gpu_raw" | awk -F', ' '{print int($1)}')
    temp=$(echo "$gpu_raw" | awk -F', ' '{print int($2)}')
    watts=$(echo "$gpu_raw" | awk -F', ' '{printf "%.1f", $3}')
    mem_line=$(free -m | grep Mem)
    available=$(echo "$mem_line" | awk '{print $7}')
    mem_available_gb=$(awk "BEGIN {printf \"%.1f\", $available / 1024}")

    # Packaging-specific: disk usage and model sizes
    disk_free=$(df -BG /home/robert_li/Desktop/projects/wp-finetune 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    merged_size=$(du -sm models/*-merged/ 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum/1024}' || echo "0")
    quantized_awq=$(du -sm quantized/awq/ 2>/dev/null | awk '{printf "%.1f", $1/1024}' || echo "0")
    quantized_q4=$(du -sm quantized/gguf-q4/ 2>/dev/null | awk '{printf "%.1f", $1/1024}' || echo "0")
    quantized_q8=$(du -sm quantized/gguf-q8/ 2>/dev/null | awk '{printf "%.1f", $1/1024}' || echo "0")
    # Active quantization processes
    quant_procs=$(ps aux 2>/dev/null | grep -cE 'autoawq|llama-quantize|convert.*gguf|awq' | grep -v grep || echo "0")

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"ts\": \"$ts\", \"watts\": $watts, \"temperature_c\": $temp, \"gpu_util_pct\": $gpu_util, \"mem_available_gb\": $mem_available_gb, \"disk_free_gb\": $disk_free, \"merged_size_gb\": $merged_size, \"awq_size_gb\": $quantized_awq, \"q4_size_gb\": $quantized_q4, \"q8_size_gb\": $quantized_q8, \"quant_procs\": $quant_procs, \"source\": \"monitor\"}" >> "$JSONL"
    echo "$ts Check $i/$MAX_CHECKS | temp=${temp}C gpu=${gpu_util}% ram=${mem_available_gb}GB disk=${disk_free}GB | merged=${merged_size}GB awq=${quantized_awq}GB q4=${quantized_q4}GB q8=${quantized_q8}GB procs=$quant_procs"

    (( temp >= 85 )) && touch "${TDIR}/_thermal_pause" && echo "$ts CRITICAL: temp=${temp}C >= 85"
    [[ $i -lt $MAX_CHECKS ]] && sleep 120 || true
done
```

Start the monitor:

```bash
chmod +x {TDIR}/monitor.sh
nohup bash {TDIR}/monitor.sh "{TDIR}" > {TDIR}/monitor.log 2>&1 &
echo $! > {TDIR}/monitor.pid
```

Report: "Packaging lightweight monitor active. PID: {pid}. Output: {TDIR}/monitor.jsonl. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

### 3b. Full Agent Team (optional, sonnet only)

Spawn agents with EXPLICIT `model="sonnet"` parameter. This is critical -- haiku agents do not persist loops.

#### Quantization Progress Observer

```
Agent(
  model="sonnet",
  description="Telemetry: quantization progress",
  prompt="You are a quantization progress observer. Write observations to {TDIR}/quantization-progress.md.

  LOOP (every 2 minutes):
  1. Check running processes: ps aux | grep -E 'autoawq\|llama-quantize\|convert.*gguf\|awq' | grep -v grep
  2. Run: nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  3. Check output files: ls -lh quantized/ 2>/dev/null || ls -lh models/*awq* models/*gguf* 2>/dev/null
  4. Append to {TDIR}/quantization-progress.md:
     ### {HH:MM:SS}
     - Running: {process or 'idle'}
     - GPU: {mem} MiB | {util}% | {temp}C
     - AWQ: {not started / running / complete}
     - GGUF Q4_K_M: {not started / running / complete}
     - GGUF Q8_0: {not started / running / complete}
  5. Flag WARNING if GPU temp > 80C during quantization
  6. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  7. Sleep 120 seconds, repeat

  STOP CONDITIONS: _stop file exists OR all formats complete",
  run_in_background=true
)
```

#### File Integrity Observer

```
Agent(
  model="sonnet",
  description="Telemetry: file integrity",
  prompt="You are a file integrity observer. Write observations to {TDIR}/file-integrity.md.

  LOOP (every 5 minutes):
  1. For each quantized model directory, check:
     - All expected files present (config.json, tokenizer files, model weights)
     - No 0-byte files: find quantized/ -size 0 2>/dev/null
     - JSON files are valid: python3 -c 'import json; json.load(open(\"config.json\"))' for each
  2. For GGUF files, check: file quantized/*.gguf 2>/dev/null (should report 'GGUF')
  3. Verify special tokens in quantized tokenizer: grep 'wp_gen\|wp_judge' quantized/*/tokenizer_config.json 2>/dev/null
  4. Append to {TDIR}/file-integrity.md:
     ### {HH:MM:SS}
     - AWQ: {file count} files, config valid: {yes/no}, tokens present: {yes/no}
     - GGUF: {file count} files, format valid: {yes/no}
     - Zero-byte files: {count}
  5. Flag CRITICAL if any output has 0-byte files or missing config
  6. Flag CRITICAL if special tokens missing from quantized tokenizer
  7. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  8. Sleep 300 seconds, repeat

  STOP CONDITIONS: _stop file exists OR all formats verified",
  run_in_background=true
)
```

#### Size Tracking Observer

```
Agent(
  model="sonnet",
  description="Telemetry: size tracking",
  prompt="You are a model size tracking observer. Write observations to {TDIR}/size-tracking.md.

  LOOP (every 2 minutes):
  1. Run: du -sh models/*-merged/ 2>/dev/null (BF16 baseline)
  2. Run: du -sh quantized/awq/ quantized/gguf-q4/ quantized/gguf-q8/ 2>/dev/null
  3. Run: df -h /home/robert_li/Desktop/projects/wp-finetune
  4. Calculate compression ratios vs BF16 baseline
  5. Append to {TDIR}/size-tracking.md:
     ### {HH:MM:SS}
     - BF16 baseline: {size}
     - AWQ 4-bit: {size} ({ratio}x compression)
     - GGUF Q4_K_M: {size} ({ratio}x compression)
     - GGUF Q8_0: {size} ({ratio}x compression)
     - Disk free: {free}
  6. Flag WARNING if disk free < 50GB
  7. Check {TDIR}/_stop -- if so, write ## Final Summary and exit
  8. Sleep 120 seconds, repeat

  STOP CONDITIONS: _stop file exists OR all formats complete",
  run_in_background=true
)
```

Report: "Packaging telemetry active with 3 Sonnet observers. Output: {TDIR}/. Touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

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
