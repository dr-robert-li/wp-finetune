# Skill: review-telemetry

Read all telemetry files for a stage and produce a consolidated summary with alerts and key metrics.

**This skill does not spawn background agents.** It reads existing telemetry files produced by observe-* skills and generates a summary.

## Trigger

User says: "review telemetry", "telemetry summary", "how is training going", "check telemetry", "/review-telemetry"

## Process

### 1. Find Available Telemetry Runs

```bash
ls -dt telemetry/*/20* 2>/dev/null | head -10
```

If multiple runs exist, show the list and ask the user which to review. Default to the most recent.

### 2. Read All Agent Reports

For the selected telemetry run directory:

```bash
TDIR="telemetry/{stage}/{timestamp}"
ls -la "${TDIR}/"
```

For each `.md` file in the directory (excluding `_summary.md`):
1. Read the full file
2. Extract all lines containing WARNING or CRITICAL
3. Extract the `## Final Summary` section if present
4. Extract key numeric values (memory, temperature, loss, latency, etc.)

### 3. Produce Consolidated Summary

Write `{TDIR}/_summary.md` with this structure:

```markdown
# Telemetry Summary: {stage} -- {timestamp}

## Status: {OK | WARNINGS | CRITICAL}

## Alerts
{Chronological list of all WARNING and CRITICAL flags with timestamps and source agent}

## Key Metrics

### {Stage-specific metrics}
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ... | ... | ... | OK/WARN/CRIT |

## Timeline
{Significant events in chronological order across all agents}

## Agent Reports
{One-paragraph summary per agent file}

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

**Data Pipeline:**
- Current file counts vs targets
- Pass rate, synthetic count, CoT count
- Peak RAM, peak disk usage

**Evaluation:**
- PHPCS pass rate, security pass rate
- Spearman correlation, classification precision
- Quality gate result (PASS/FAIL)
- GPU utilization during eval

**Packaging:**
- Compression ratios (AWQ, GGUF Q4, GGUF Q8)
- File integrity status per format
- Special token verification

**Inference:**
- Avg/p95/p99 latency
- Throughput (tok/s)
- Error rate, 5xx count
- Peak GPU memory during serving
