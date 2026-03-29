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

## Trigger

User says: "observe packaging", "monitor packaging", "monitor quantization", "/observe-packaging"

## Process

### 1. Create Run Directory

```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/packaging/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

### 2. Spawn Quantization Progress Observer

```
Agent(
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

### 3. Spawn File Integrity Observer

```
Agent(
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

### 4. Spawn Size Tracking Observer

```
Agent(
  description="Telemetry: size tracking",
  prompt="You are a model size tracking observer. Write observations to {TDIR}/size-tracking.md.

  LOOP (every 2 minutes):
  1. Run: du -sh merged_model/ 2>/dev/null (BF16 baseline)
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

### 5. Report

Tell the user: "Packaging telemetry active with 3 observers. Output: {TDIR}/. Say 'stop observing' or touch {TDIR}/_stop to end."

## Stopping Observers

```bash
touch {TDIR}/_stop
```
