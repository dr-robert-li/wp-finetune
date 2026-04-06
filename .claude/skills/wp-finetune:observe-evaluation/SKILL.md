# Skill: wp-finetune:observe-evaluation

Spawn background observer agents to monitor model evaluation. Writes telemetry to `telemetry/evaluation/{timestamp}/`.

**Agent team assessment:** Evaluation runs GPU inference (wp-bench, PHPCS eval), produces result files, and has a quality gate. Requires 3-agent team: eval-progress, gpu-metrics, result-tracking.

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

### 1. Create Run Directory

```bash
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
TDIR="telemetry/evaluation/${TIMESTAMP}"
mkdir -p "${TDIR}"
echo "Telemetry directory: ${TDIR}"
```

### 2. Spawn Eval Progress Observer

```
Agent(
  description="Telemetry: eval progress",
  prompt="You are an evaluation progress observer. Write observations to {TDIR}/eval-progress.md.

  LOOP (every 2 minutes):
  1. Check running eval processes: ps aux | grep -E 'run_eval_triage|vllm|wp-bench' | grep -v grep
  2. Check eval output files: ls -la output/eval_triage/ 2>/dev/null
  3. Check wp-bench results: ls -la output/eval_triage/ratio_*/wp_bench_results.json 2>/dev/null
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

### 3. Spawn GPU Metrics Observer

```
Agent(
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

### 4. Spawn Result Tracking Observer

```
Agent(
  description="Telemetry: result tracking",
  prompt="You are an eval result tracking observer. Write observations to {TDIR}/result-tracking.md.

  LOOP (every 2 minutes):
  1. Read eval_gen results if they exist: cat output/eval_triage/ratio_*/eval_gen_results.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"PHPCS pass: {d.get(\"phpcs_pass_rate\", \"N/A\")}, Security: {d.get(\"security_pass_rate\", \"N/A\")}\")' 2>/dev/null
  2. Read eval_judge results: cat output/eval_triage/ratio_*/eval_judge_results.json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"Spearman: {d.get(\"spearman_corr\", \"N/A\")}\")' 2>/dev/null
  3. Read eval_gate results: cat output/eval_triage/ratio_*/eval_gate_results.json 2>/dev/null
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

### 5. Report

Tell the user: "Evaluation telemetry active with 3 observers. Output: {TDIR}/. Say 'stop observing' or touch {TDIR}/_stop to end. Run /review-telemetry to consolidate results."

## Stopping Observers

```bash
touch {TDIR}/_stop
```
