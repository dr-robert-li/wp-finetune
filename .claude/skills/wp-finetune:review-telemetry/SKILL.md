# Skill: wp-finetune:review-telemetry

Read all telemetry files for a stage and produce a consolidated summary with alerts and key metrics.

**This skill does not spawn background agents.** It reads existing telemetry files produced by observe-* skills and generates a summary.

## Trigger

User says: "review telemetry", "telemetry summary", "how is training going", "check telemetry", "/review-telemetry"

## Process

### 1. Find Available Telemetry Runs

```bash
# Timestamped run directories (from observe agents or lightweight monitors)
ls -dt telemetry/*/* 2>/dev/null | head -10
```

If no telemetry directories exist, report "No telemetry found. Run /observe-training, /observe-evaluation, /observe-inference, /observe-data-pipeline, or /observe-packaging first."

If multiple runs exist, show the list and ask the user which to review. Default to the most recent.

### 2. Read All Telemetry Data

For the selected telemetry run directory:

```bash
TDIR="telemetry/{stage}/{timestamp}"
ls -la "${TDIR}/"
```

Handle **both** data formats produced by the two-tier monitoring system:

#### Agent Reports (`.md` files)

For each `.md` file in the directory (excluding `_summary.md`):
1. Read the full file
2. Extract all lines containing WARNING or CRITICAL
3. Extract the `## Final Summary` section if present
4. Extract key numeric values (memory, temperature, loss, latency, etc.)

#### Monitor JSONL (`monitor.jsonl`)

If `monitor.jsonl` exists in the directory:
1. Read the file
2. Parse each line as JSON
3. **Canonical fields** (present in all monitor types): compute min/avg/max for:
   - `watts` -- power draw
   - `temperature_c` -- GPU temperature
   - `gpu_util_pct` -- GPU utilization percentage
   - `mem_available_gb` -- available system memory
4. **Type-specific fields** based on which observe type produced the data:
   - **Training:** `loss`, `step`, `checkpoints`
   - **Evaluation:** `eval_gen_done`, `eval_judge_done`, `wpbench_done`, `triage_done`, `vllm_status`
   - **Inference:** `vllm_status`, `vllm_latency_s`, `ttft_s`, `model`, `error_count`
   - **Data Pipeline:** `disk_free_gb`, `passed`, `synth_passed`, `synth_failed`, `judge_training`, `cot`, `exports`, `checkpoints`
   - **Packaging:** `disk_free_gb`, `merged_size_gb`, `awq_size_gb`, `q4_size_gb`, `q8_size_gb`, `quant_procs`
5. Detect the type from the directory path (`telemetry/{type}/...`) and extract the relevant fields

#### Monitor Logs (`monitor.log`)

If `monitor.log` exists:
1. Read the file for human-readable output
2. Extract any CRITICAL lines
3. Count total check cycles completed
4. Note if monitor exited cleanly (stop signal) or is still running

### 3. Produce Consolidated Summary

Write `{TDIR}/_summary.md` with this structure:

```markdown
# Telemetry Summary: {stage} -- {timestamp}

## Status: {OK | WARNINGS | CRITICAL}

## Data Source
{Lightweight monitor (N checks over Xh) | Agent team (N reports) | Both}

## Alerts
{Chronological list of all WARNING and CRITICAL flags with timestamps and source}

## Key Metrics

### Canonical GPU/System Metrics (from JSONL)
| Metric | Min | Avg | Max |
|--------|-----|-----|-----|
| Power (watts) | ... | ... | ... |
| Temperature (C) | ... | ... | ... |
| GPU Util (%) | ... | ... | ... |
| RAM Available (GB) | ... | ... | ... |

### {Stage-specific metrics}
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ... | ... | ... | OK/WARN/CRIT |

## Timeline
{Significant events in chronological order across all data sources}

## Agent Reports
{One-paragraph summary per agent .md file, if any exist}

## Recommendations
{Based on alerts and metrics, suggest actions}
```

### 4. Display Summary

Print the full summary to the conversation. Highlight any CRITICAL alerts prominently.

### 5. Stage-Specific Metrics to Extract

**Training:**
- Peak GPU memory, avg GPU utilization
- Final training loss, min loss, total steps
- Peak temperature, throttle events
- Checkpoint count, final adapter size
- Container uptime, OOM events
- From JSONL: min/avg/max for watts, temperature_c, gpu_util_pct, mem_available_gb; latest loss, step, checkpoints

**Data Pipeline:**
- Current file counts vs targets
- Pass rate, synthetic count, CoT count
- Peak RAM, peak disk usage
- From JSONL: mem_available_gb trend, disk_free_gb trend, file count progression

**Evaluation:**
- PHPCS pass rate, security pass rate
- Spearman correlation (overall + per-dimension)
- Quality gate result (PASS/FAIL)
- GPU utilization during eval
- From JSONL: vllm_status history, eval stage completion timeline

**Packaging:**
- Compression ratios (AWQ, GGUF Q4, GGUF Q8)
- File integrity status per format
- Special token verification
- From JSONL: merged_size_gb, awq/q4/q8 size progression, disk_free_gb trend

**Inference:**
- Avg/p95/p99 latency
- Throughput (tok/s)
- Error rate, 5xx count
- Peak GPU memory during serving
- From JSONL: vllm_latency_s min/avg/max, ttft_s min/avg/max, error_count total
