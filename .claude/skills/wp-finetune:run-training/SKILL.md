# Skill: wp-finetune:run-training

Run the training pipeline via the DGX Toolbox execution engine. Supports training on one or more dataset ratio exports sequentially, with isolated checkpoints per run.

## Architecture

```
Skill (this file — intent + recovery logic)
  → dgx_toolbox.py (resolve paths, validate state, manage containers, execute)
    → Docker commands (generated dynamically from config, not hardcoded)
      → Python scripts (inside container, idempotent)
        → Output (adapters/{run_name}/, models/{run_name}-merged/)
```

## Telemetry

> Telemetry is **embedded** — this skill spawns observe-data-pipeline (3 agents), observe-training (6 agents), and observe-packaging (3 agents) inline at the appropriate steps. It also invokes review-telemetry after each run and after all runs complete. No need to invoke observe skills separately.

## Trigger

User says: "run training", "train the model", "start DGX training", "/run-training"

## Process

### Step 0a: Select base model

Check for available/downloaded models and let the user choose exactly one:

```bash
# Check what's already downloaded locally
ls models/*/config.json 2>/dev/null
```

Present a selection using AskUserQuestion:
```
Select base model (1 only):

| # | Model | Status | Params | Active | Notes |
|---|-------|--------|--------|--------|-------|
| 1 | Qwen/Qwen3-30B-A3B | [Downloaded/Not downloaded] | 30B | ~3B (MoE) | Default — native MoE, proven serving |
| 2 | Qwen/Qwen3-14B | [Downloaded/Not downloaded] | 14B | 14B (dense) | Faster iteration, smaller |
| 3 | Qwen/Qwen3-8B | [Downloaded/Not downloaded] | 8B | 8B (dense) | Quick experiments |
| 4 | Custom | — | — | — | Enter HuggingFace model ID |
```

If "Custom": ask for the HuggingFace model ID (e.g., `meta-llama/Llama-3-8B`).

Store as `$BASE_MODEL` (HF name) and `$MODEL_LOCAL_DIR` (local path like `models/Qwen3-30B-A3B`).

**Only one base model per training session.** All selected ratios train against the same base.

### Step 0b: Select dataset exports

List available ratio exports and let the user choose one or more:

```bash
ls data/final_dataset/ratio_*/metadata.json
```

Read each metadata.json and present a table:
```
Available dataset exports:

| # | Ratio | Gen    | Judge  | Total  | Train  |
|---|-------|--------|--------|--------|--------|
| 1 | 30/70 | ...    | ...    | ...    | ...    |
| 2 | 40/60 | ...    | ...    | ...    | ...    |
| ... |

Select exports to train (comma-separated, e.g. "2,3,4" or "all"):
```

Use AskUserQuestion for selection. Store as `$SELECTED_RATIOS` list.

### Step 0c: Telemetry monitoring

Choose which telemetry collectors to run. Both feed the same **canonical thermal log** — a single JSONL file per run that drives adaptive resource planning (Step 8.5).

Use AskUserQuestion:
- header: "Telemetry"
- question: "Select telemetry mode. Both options write to the same canonical thermal log used by adaptive resource planning."
- multiSelect: false
- options:
  - "Observe agents (Recommended)" → set `$TELEMETRY_MODE = "observe"`
    - description: "Full 6-agent team: GPU metrics, thermal/throttling, training loss, disk I/O, checkpoint integrity, container health. Richer data but heavier — 6 background agents running continuously."
  - "Lightweight monitor" → set `$TELEMETRY_MODE = "monitor"`
    - description: "Single background agent polling nvidia-smi every 10 minutes. Only GPU utilization and temperature are recorded — no loss tracking, checkpoint integrity, or container health. Sufficient for adaptive resource planning."
  - "None" → set `$TELEMETRY_MODE = "none"`

**If the user selects "None"**, show a warning via AskUserQuestion:
- header: "Warning"
- question: "No telemetry means no adaptive resource planning. Training config will NOT auto-adjust between runs — GPU may be underutilized or overheat without detection. Are you sure?"
- options:
  - "Use observe agents" → set `$TELEMETRY_MODE = "observe"`
  - "Use lightweight monitor" → set `$TELEMETRY_MODE = "monitor"`
  - "Disable anyway" → set `$TELEMETRY_MODE = "none"`

Store as `$TELEMETRY_MODE` ("observe" | "monitor" | "none").

Derive convenience flags:
- `$TELEMETRY = ($TELEMETRY_MODE != "none")` — controls adaptive planning (Step 8.5)
- `$OBSERVE = ($TELEMETRY_MODE == "observe")` — controls full agent spawning (Steps 4/7/8)
- `$MONITOR = ($TELEMETRY_MODE == "monitor")` — controls lightweight monitor spawning

#### Canonical thermal log

All telemetry collectors (observe agents and lightweight monitor) append to the same canonical file:

```
telemetry/training/{model_short}_{date}_{ratio}_thermal.jsonl
```

- `{model_short}` = base model short name (e.g., `qwen3-30b`)
- `{date}` = date training commenced (e.g., `20260330`)
- `{ratio}` = dataset ratio (e.g., `30_70`)
- Example: `telemetry/training/qwen3-30b_20260330_30_70_thermal.jsonl`

**JSONL schema** (one line per reading):

```jsonl
{"ts": "2026-03-30T06:57:47Z", "gpu_util": 82, "temp": 65, "vram_used_mb": null, "sys_ram_used_mb": 109568, "sys_ram_total_mb": 122368, "source": "monitor"}
{"ts": "2026-03-30T06:58:17Z", "gpu_util": 85, "temp": 66, "vram_used_mb": 92493, "sys_ram_used_mb": 109568, "sys_ram_total_mb": 122368, "source": "observe-agent"}
```

**Memory fields:**
- `vram_used_mb` — nvidia-smi `memory.used`. Reports `null` on unified memory systems (GB10/Grace Hopper).
- `sys_ram_used_mb` — from `free -m` (total - available). Always available.
- `sys_ram_total_mb` — from `free -m`. Always available.
- On unified memory: `sys_ram_used_mb` IS the GPU memory — VRAM and system RAM share the same pool.
- On discrete GPU: both fields are meaningful and tracked independently.

Store the canonical log path as `$THERMAL_LOG` for this run:
```python
THERMAL_LOG = f"telemetry/training/{model_short}_{date}_{ratio}_thermal.jsonl"
```

This file is the **single source of truth** for adaptive resource planning. Step 8.5a reads only this file — it never parses agent markdown or monitor output directly.

```
Collectors (Step 0c)              Canonical log                Downstream consumers
────────────────────             ──────────────               ────────────────────
Lightweight monitor ──┐
                      ├──► {model}_{date}_{ratio}    ──┬──► Step 8.5a: compute avg/peak
Observe agents ───────┘    _thermal.jsonl              │
                           (append-only JSONL)         ├──► Step 8.5b: thermal_history.json
                                                       │    (one summary record per run)
                                                       │
                                                       ├──► Step 8.5c: zone classification
                                                       │
                                                       └──► Step 8.5d: CRITICAL backoff
                                                            (lookup last WARM in history)
```

#### What each mode spawns

**Observe agents** (`$TELEMETRY_MODE = "observe"`):

| Training phase | Spawned | Purpose |
|---------------|---------|---------|
| Step 4: Download | observe-data-pipeline (3 agents) | Network I/O, disk, progress |
| Step 7: Train | observe-training (6 agents) | GPU, thermal (writes to `$THERMAL_LOG`), loss, disk, checkpoints, container |
| Step 8: Merge | observe-packaging (3 agents) | Merge progress, file integrity, size |
| Step 8d | review-telemetry | Per-run `_summary.md` |
| Step 9b | review-telemetry | Cross-run comparison |

**Lightweight monitor** (`$TELEMETRY_MODE = "monitor"`):

| Training phase | Spawned | Purpose |
|---------------|---------|---------|
| Step 7: Train | 1 background agent | Polls nvidia-smi every 10 min, appends to `$THERMAL_LOG` |

No observe agents, no review-telemetry, no per-run summaries. Only the canonical thermal log is produced — sufficient for adaptive resource planning.

**Lifecycle (observe mode):**
1. Spawn observe agents in background before the long-running step
2. Agents append to their markdown files AND to `$THERMAL_LOG` (thermal/GPU agents)
3. Execute the step (download/train/merge)
4. Touch `_stop` file — agents write Final Summary and exit
5. Invoke review-telemetry to consolidate into `_summary.md`
6. Proceed to next step

**Lifecycle (monitor mode):**
1. Spawn single monitor agent before training
2. Monitor appends to `$THERMAL_LOG` every 10 minutes
3. Execute training
4. Touch `_stop` — monitor exits
5. Proceed to next step (no review-telemetry)

**Between runs (both modes):** Step 8.5 reads `$THERMAL_LOG`, computes aggregates, updates `thermal_history.json`, adjusts config.

**After all runs (observe mode only):** Final review-telemetry produces `cross_run_summary.md`.

### Step 0d: Confirm training plan

Present a full summary of what will happen and ask for explicit confirmation. Training runs are long (6-12 hours per ratio) and expensive — mistakes here are costly.

```
╔══════════════════════════════════════════════════════════════╗
║  TRAINING PLAN — PLEASE REVIEW                               ║
╚══════════════════════════════════════════════════════════════╝

  Base model:    {BASE_MODEL}
  Local path:    {MODEL_LOCAL_DIR}
  Status:        [Downloaded ✓ / Not yet downloaded]

  LoRA config:   r={r}, alpha={alpha}, dropout={dropout}
  Target modules: {target_modules}
  Modules saved: {modules_to_save}

  Training:      {num_epochs} epochs, batch={batch_size}, grad_accum={grad_accum}
  Learning rate: {lr} ({scheduler})
  Precision:     bf16

  Telemetry:     {TELEMETRY ? "✓ Enabled (observe-training + review)" : "✗ Disabled"}

  Runs planned:  {len(SELECTED_RATIOS)}
  Est. duration: ~{len(SELECTED_RATIOS) * 6}-{len(SELECTED_RATIOS) * 12} hours total

  ┌─────┬─────────────────────────────┬──────────┬────────────────────────────────┐
  │ Run │ Dataset                     │ Train ex │ Output                         │
  ├─────┼─────────────────────────────┼──────────┼────────────────────────────────┤
  │  1  │ ratio_50_50 (gen=30K j=30K) │ 48,796   │ adapters/{run_name_1}/         │
  │  2  │ ratio_60_40 (gen=46K j=30K) │ 60,996   │ adapters/{run_name_2}/         │
  │ ... │                             │          │                                │
  └─────┴─────────────────────────────┴──────────┴────────────────────────────────┘

  Disk required: ~{estimate}GB per run (adapter + checkpoints)
  Memory required: ~70GB (Qwen3-30B-A3B bf16 + LoRA optimizer states)

──────────────────────────────────────────────────────────────
→ Type "confirmed" to start training, or describe changes
──────────────────────────────────────────────────────────────
```

Read the base config from `config/train_config.yaml` to populate LoRA and training hyperparameters in the summary. Check disk space with `df -h .` and model download status with `ls models/*/config.json`.

Use AskUserQuestion:
- header: "Training Plan Confirmation"
- question: "Review the plan above. Start training?"
- options:
  - "Confirmed — start training" → proceed to Step 1
  - "Change model" → go back to Step 0a
  - "Change ratios" → go back to Step 0b
  - "Change telemetry" → go back to Step 0c
  - "Change hyperparameters" → tell user to edit config/train_config.yaml, then re-run
  - "Abort" → exit skill

**Do NOT proceed to Step 1 until the user explicitly confirms.**

**For each selected ratio**, execute Steps 1-8 below with run-specific paths:
- `run_name` = `{model_short}-wp-{ratio}` (e.g., `qwen3-30b-wp-50_50`)
- `data_dir` = `data/final_dataset/ratio_{ratio}/`
- `adapter_dir` = `adapters/{run_name}/`
- `merged_dir` = `models/{run_name}-merged/`

### Step 1: Configure run

Before training, create a run-specific config overlay:

```python
import yaml

base_config = yaml.safe_load(open("config/train_config.yaml"))

# Override model to user's selection
base_config["model"]["name"] = BASE_MODEL           # e.g. "Qwen/Qwen3-30B-A3B"
base_config["model"]["local_dir"] = MODEL_LOCAL_DIR  # e.g. "./models/Qwen3-30B-A3B"

# Override data paths for this ratio
base_config["data"]["train_file"] = f"data/final_dataset/ratio_{ratio}/openai_train.jsonl"
base_config["data"]["val_file"] = f"data/final_dataset/ratio_{ratio}/openai_val.jsonl"
base_config["data"]["test_file"] = f"data/final_dataset/ratio_{ratio}/openai_test.jsonl"

# Override output dir for run isolation
base_config["training"]["output_dir"] = f"./adapters/{run_name}"

# Write run config
run_config_path = f"config/train_config_{ratio}.yaml"
yaml.dump(base_config, open(run_config_path, "w"))
```

### Step 2: Validate

```python
from scripts.dgx_toolbox import get_toolbox
dgx = get_toolbox()

result = dgx.validate(["toolbox", "config", "memory:70"])
print(result.report())
if not result.ok:
    for f in result.failures:
        print(f"  FIX: {f.name} — {f.message}")
    # STOP
```

Also verify the ratio-specific training data exists:
```python
from pathlib import Path
train_file = Path(f"data/final_dataset/ratio_{ratio}/openai_train.jsonl")
if not train_file.exists():
    print(f"ERROR: Training data not found: {train_file}")
    # STOP
```

### Step 3: Ensure container ready

```python
ready = dgx.ensure_ready("unsloth_studio")
print(ready.report())
if not ready.ok:
    for f in ready.failures:
        print(f"  ISSUE: {f.name} — {f.message}")
    # STOP
```

### Step 4: Download model (idempotent — shared across runs)

**If `$TELEMETRY` and model not yet downloaded**, spawn observe-data-pipeline before download:

```
# 4a: Spawn telemetry (only if download will actually run)
if $OBSERVE and not Path("models/Qwen3-30B-A3B/config.json").exists():
    DOWNLOAD_TDIR = f"telemetry/data-pipeline/{timestamp}"
    mkdir -p $DOWNLOAD_TDIR

    # Invoke wp-finetune:observe-data-pipeline inline — spawn its 3 agents:
    Agent(description="Telemetry: pipeline progress", prompt="...write to {DOWNLOAD_TDIR}/pipeline-progress.md...", run_in_background=true)
    Agent(description="Telemetry: system resources", prompt="...write to {DOWNLOAD_TDIR}/system-resources.md...", run_in_background=true)
    Agent(description="Telemetry: disk I/O", prompt="...write to {DOWNLOAD_TDIR}/disk-io.md...", run_in_background=true)
```

```python
# 4b: Execute download
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.download_model",
    idempotency_check="models/Qwen3-30B-A3B/config.json",
)
print(result.summary())
```

```
# 4c: Stop telemetry
if $OBSERVE and DOWNLOAD_TDIR:
    touch $DOWNLOAD_TDIR/_stop   # agents write Final Summary and exit
```

### Step 5: Extend tokenizer (idempotent — shared across runs)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.prepare_tokenizer",
    idempotency_check="adapters/tokenizer/tokenizer_config.json",
)
print(result.summary())
```

### Step 6: Dry run (per-run config)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.train_model",
    "--dry-run",
    "--config", f"config/train_config_{ratio}.yaml",
    capture=True,
)
print(result.stdout)
if not result.ok:
    print(f"Dry run failed: {result.stderr[-500:]}")
    # STOP
```

**Present dry run output to user.** If it shows errors, fix them. If valid, proceed.

### Step 7: Train (long-running, per-run isolated)

**7a: Set up canonical thermal log and spawn collectors.**

```python
# Canonical thermal log path for this run
THERMAL_LOG = f"telemetry/training/{model_short}_{date}_{ratio}_thermal.jsonl"
Path(THERMAL_LOG).parent.mkdir(parents=True, exist_ok=True)
```

**If `$OBSERVE`** — spawn full 6-agent observe-training team. The thermal and GPU agents append to both their markdown files AND `$THERMAL_LOG`:

```
TRAIN_TDIR = f"telemetry/training/{timestamp}"
mkdir -p $TRAIN_TDIR

# GPU Metrics Observer (every 30s)
Agent(
  description="Telemetry: GPU metrics",
  prompt="You are a GPU metrics observer. Write to {TRAIN_TDIR}/gpu-metrics.md.
  ALSO append JSONL to {THERMAL_LOG} on each reading:
    {\"ts\": \"...\", \"gpu_util\": N, \"temp\": N, \"vram_used_mb\": N_or_null, \"sys_ram_used_mb\": N, \"sys_ram_total_mb\": N, \"source\": \"observe-agent\"}
  LOOP (30s):
    1. nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    2. free -m | grep Mem → parse used and total system RAM
    3. If nvidia-smi memory reports [N/A]: set vram_used_mb=null (unified memory — sys_ram IS GPU memory)
    4. Append to gpu-metrics.md AND {THERMAL_LOG}
    5. WARNING if memory > 90%. CRITICAL (warn+log only, do NOT stop training) if memory >= 98%.
       Memory issues are caught pre-training by Step 2 (validate memory >= 70GB) and Step 6 (dry run).
       During training, memory CRITICAL is observational — the OS/driver will OOM-kill if truly exhausted.
    6. WARNING if util < 50% 3x.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Thermal/Throttling Observer (every 30s) — includes live thermal guard
Agent(
  description="Telemetry: thermal/throttling",
  prompt="You are a GPU thermal observer. Write to {TRAIN_TDIR}/thermal-throttling.md.
  LOOP (30s): nvidia-smi temp/power/throttle → append. WARNING > 80C. CRITICAL >= 83C → touch {TRAIN_TDIR}/_thermal_pause.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Training Metrics Observer (every 60s)
Agent(
  description="Telemetry: training metrics",
  prompt="You are a training metrics observer. Write to {TRAIN_TDIR}/training-metrics.md.
  LOOP (60s): docker logs → grep loss/step/epoch. Check trainer_state.json, MLflow logs.
  WARNING if loss increases 3x. CRITICAL if loss > 10 or grad_norm > 100.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Disk I/O Observer (every 60s)
Agent(
  description="Telemetry: disk I/O",
  prompt="You are a disk I/O observer. Write to {TRAIN_TDIR}/disk-io.md.
  LOOP (60s): iostat, df, du adapters/. WARNING if iowait > 20% or disk > 85%.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Checkpoint Integrity Observer (every 5 min)
Agent(
  description="Telemetry: checkpoint integrity",
  prompt="You are a checkpoint integrity observer. Write to {TRAIN_TDIR}/checkpoint-integrity.md.
  LOOP (5m): ls checkpoints, verify adapter_config.json valid, safetensors > 0 bytes, tokenizer present.
  WARNING if no new checkpoint in 30m. CRITICAL if 0-byte safetensors.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Container Monitor (every 60s)
Agent(
  description="Telemetry: container monitor",
  prompt="You are a container health observer. Write to {TRAIN_TDIR}/container-monitor.md.
  LOOP (60s): docker ps, docker stats, top processes, dmesg OOM check.
  WARNING if container not running. CRITICAL if OOM killer detected.
  STOP: {TRAIN_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)
```

**If `$MONITOR`** — spawn single lightweight agent that only writes to `$THERMAL_LOG`:

```
Agent(
  description="Telemetry: lightweight thermal monitor",
  prompt="You are a lightweight GPU monitor. Append JSONL to {THERMAL_LOG} every 10 minutes.

  IMPORTANT: All queries run on the HOST, not inside the container. docker exec nvidia-smi
  can lose NVML access on long-running containers while the host nvidia-smi stays reliable.

  LOOP (every 600 seconds):
  1. Run: nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits  (HOST, not docker exec)
  2. Run: free -m | grep Mem → parse total, used, available system RAM (HOST)
  3. Parse gpu_util, temp from nvidia-smi. Set vram_used_mb from memory.used (null if [N/A]).
  4. Compute sys_ram_used_mb = total - available, sys_ram_total_mb = total
  5. Append one JSONL line to {THERMAL_LOG}:
     {\"ts\": \"YYYY-MM-DDTHH:MM:SSZ\", \"gpu_util\": N, \"temp\": N, \"vram_used_mb\": N_or_null, \"sys_ram_used_mb\": N, \"sys_ram_total_mb\": N, \"source\": \"monitor\"}
  6. If temp >= 83: also touch telemetry/training/_thermal_pause
  7. Check if telemetry/training/_stop exists → exit
  8. Sleep 600 seconds, repeat

  STOP: telemetry/training/_stop file exists OR after 84 checks (14 hours).",
  run_in_background=true
)
```

**7b: Detect existing checkpoints and execute training.**

Check for existing checkpoints from a prior interrupted run. If found, pass `--resume` so training picks up from the latest checkpoint instead of restarting from scratch.

```python
from pathlib import Path
import re

adapter_dir = Path(f"adapters/{run_name}")
checkpoints = sorted(adapter_dir.glob("checkpoint-*"), key=lambda p: int(re.search(r"\d+", p.name).group())) if adapter_dir.exists() else []

train_cmd = ["python", "-m", "scripts.train_model", "--config", f"config/train_config_{ratio}.yaml"]

if checkpoints:
    latest_ckpt = checkpoints[-1]
    print(f"Found {len(checkpoints)} existing checkpoint(s), latest: {latest_ckpt.name}")
    print(f"Resuming training from {latest_ckpt}")
    train_cmd.extend(["--resume", str(latest_ckpt)])

result = dgx.execute(
    "unsloth_studio",
    *train_cmd,
    idempotency_check=f"adapters/{run_name}/adapter_config.json",
    timeout=None,  # No timeout — training takes 6-12 hours
)
print(result.summary())
if not result.ok:
    print(f"Training failed for {run_name}. Check MLflow logs: mlflow ui --backend-store-uri mlruns/")
    print("To resume: run this skill again (idempotency will skip completed runs)")
```

**7c: Stop training telemetry and check for thermal events.**

```
if $OBSERVE:
    touch $TRAIN_TDIR/_stop   # all 6 agents write Final Summary and exit
if $MONITOR:
    touch telemetry/training/_stop   # lightweight monitor exits

# Check for live thermal guard trigger (both modes write this)
if Path("telemetry/training/_thermal_pause").exists():
    print("⚠ THERMAL EVENT detected during training — adaptive planning will apply CRITICAL backoff")
    Path("telemetry/training/_thermal_pause").unlink()  # reset for next run
```

### Step 8: Merge adapter (per-run isolated)

**8a: Spawn observe-packaging (3 agents) before merge.**

If `$OBSERVE`:

```
MERGE_TDIR = f"telemetry/packaging/{timestamp}"
mkdir -p $MERGE_TDIR

# Quantization/Merge Progress Observer (every 2 min)
Agent(
  description="Telemetry: merge progress",
  prompt="You are a merge progress observer. Write to {MERGE_TDIR}/quantization-progress.md.
  LOOP (2m): ps aux for merge process, nvidia-smi, check output files in models/{run_name}-merged/.
  STOP: {MERGE_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# File Integrity Observer (every 5 min)
Agent(
  description="Telemetry: file integrity",
  prompt="You are a file integrity observer. Write to {MERGE_TDIR}/file-integrity.md.
  LOOP (5m): verify config.json valid, safetensors > 0 bytes, special tokens in tokenizer.
  CRITICAL if 0-byte files or missing config. CRITICAL if wp_gen/wp_judge missing.
  STOP: {MERGE_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)

# Size Tracking Observer (every 2 min)
Agent(
  description="Telemetry: size tracking",
  prompt="You are a size tracking observer. Write to {MERGE_TDIR}/size-tracking.md.
  LOOP (2m): du -sh merged dir, df -h disk free. WARNING if disk free < 50GB.
  STOP: {MERGE_TDIR}/_stop exists → write Final Summary → exit.",
  run_in_background=true
)
```

**8b: Execute merge.**

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.merge_adapter",
    "--adapter-dir", f"adapters/{run_name}",
    "--output-dir", f"models/{run_name}-merged",
    idempotency_check=f"models/{run_name}-merged/config.json",
)
print(result.summary())
if not result.ok:
    print(f"Merge failed. Adapter is safe at adapters/{run_name}/")
    print(f"Fallback: serve with vLLM --lora-modules adapters/{run_name}")
```

**8c: Stop packaging telemetry.**

```
if $OBSERVE:
    touch $MERGE_TDIR/_stop   # 3 agents write Final Summary and exit
```

**8d: Invoke review-telemetry to consolidate this run (observe mode only).**

If `$OBSERVE`, invoke `wp-finetune:review-telemetry` inline — read all agent reports from this run's telemetry dirs and produce `_summary.md`:

```
# Review training telemetry
Read all .md files in $TRAIN_TDIR (gpu-metrics, thermal-throttling, training-metrics, disk-io, checkpoint-integrity, container-monitor)
Extract: all WARNING/CRITICAL lines, Final Summary sections, key metrics (peak temp, final loss, peak memory, avg util)
Write: $TRAIN_TDIR/_summary.md with consolidated status, alerts, metrics, timeline, recommendations

# Review packaging telemetry (if merge ran)
Read all .md files in $MERGE_TDIR
Write: $MERGE_TDIR/_summary.md

# Print summary to conversation
print(f"Run {run_name} complete. Telemetry summary:")
print(f"  Training: {TRAIN_TDIR}/_summary.md")
print(f"  Packaging: {MERGE_TDIR}/_summary.md")
print(f"  Canonical thermal log: {THERMAL_LOG}")
```

If `$MONITOR` (no agent reports to review), just print the canonical log path:
```
print(f"Run {run_name} complete. Thermal log: {THERMAL_LOG}")
```

### Step 8.5: Adaptive resource planning (between runs)

**After each run's merge completes and before the next ratio starts**, reassess GPU headroom using telemetry from the just-completed run and adjust training config for the next run.

**Requires `$TELEMETRY = true`.** If telemetry is disabled, skip this step entirely (config stays static across all runs).

#### 8.5a: Collect metrics from canonical thermal log

Read the canonical JSONL thermal log for the completed run. This file is the single source of truth — it contains readings from whichever collector was active (observe agents, lightweight monitor, or both).

```python
import json
from pathlib import Path

# THERMAL_LOG was set in Step 7a
readings = []
for line in Path(THERMAL_LOG).read_text().splitlines():
    if line.strip():
        readings.append(json.loads(line))

if not readings:
    print(f"WARNING: No thermal data in {THERMAL_LOG} — skipping adaptive planning")
    # Skip to next ratio

gpu_utils = [r["gpu_util"] for r in readings]
gpu_temps = [r["temp"] for r in readings]
vram_readings = [r["vram_used_mb"] for r in readings if r.get("vram_used_mb")]
sys_ram_readings = [r["sys_ram_used_mb"] for r in readings if r.get("sys_ram_used_mb")]
sys_ram_totals = [r["sys_ram_total_mb"] for r in readings if r.get("sys_ram_total_mb")]
```

Compute:
- `avg_gpu_util = sum(gpu_utils) / len(gpu_utils)`
- `peak_gpu_util = max(gpu_utils)`
- `avg_temp = sum(gpu_temps) / len(gpu_temps)`
- `peak_temp = max(gpu_temps)`
- `num_readings = len(readings)`
- `sources = set(r.get("source", "unknown") for r in readings)`

Memory (supports both discrete GPU and unified memory systems):
- `vram_used_gb = max(vram_readings) / 1024 if vram_readings else None`
- `sys_ram_used_gb = max(sys_ram_readings) / 1024 if sys_ram_readings else 0`
- `sys_ram_total_gb = max(sys_ram_totals) / 1024 if sys_ram_totals else 0`
- `is_unified_memory = (vram_used_gb is None)` — True on GB10/Grace Hopper
- If unified memory: `mem_used_gb = sys_ram_used_gb`, `mem_total_gb = sys_ram_total_gb`
- If discrete GPU: `mem_used_gb = vram_used_gb`, `mem_total_gb` from nvidia-smi
- `mem_headroom_gb = mem_total_gb - mem_used_gb`

**Peak RAM with safety margin** (unified memory only — dataloader workers cause transient spikes between samples):
- `peak_ram_gb = max(sys_ram_readings) / 1024 if sys_ram_readings else 0`
- `p95_ram_gb = sorted(sys_ram_readings)[int(len(sys_ram_readings) * 0.95)] / 1024 if sys_ram_readings else 0`
- `safe_headroom_gb = mem_total_gb - peak_ram_gb`
- On unified memory, **`safe_headroom_gb` is used for all scaling decisions** (not `mem_headroom_gb` which uses averages)
- The 10-min sample interval misses sub-minute spikes from worker buffer refills, so apply a **5 GB safety margin** on unified memory: `effective_headroom_gb = safe_headroom_gb - 5`

**OOM detection** — check if the run died before completing:
```python
# Check if training completed or was OOM-killed
# Signs of OOM: GPU util drops to <10% in final readings while RAM is >95%
final_readings = readings[-5:] if len(readings) >= 5 else readings
final_gpu_utils = [r["gpu_util"] for r in final_readings]
final_ram_pcts = [r["sys_ram_used_mb"] / r["sys_ram_total_mb"] * 100 for r in final_readings if r.get("sys_ram_total_mb")]
likely_oom = (
    any(u < 10 for u in final_gpu_utils) and
    any(p > 95 for p in final_ram_pcts)
)
```
If `likely_oom` is True, treat as a **memory CRITICAL** event — apply memory backoff (8.5d-mem) regardless of thermal zone.

#### 8.5b: Update thermal history

Maintain a persistent thermal history file that records each run's config and thermal outcome. This is the memory that enables backoff-to-last-WARM.

```python
import json, yaml
from pathlib import Path

history_file = Path("telemetry/training/thermal_history.json")
history = json.loads(history_file.read_text()) if history_file.exists() else []

config = yaml.safe_load(open(f"config/train_config_{ratio}.yaml"))

# Classify thermal zone
if peak_temp >= 83:
    zone = "CRITICAL"
elif peak_temp >= 78:
    zone = "HOT"
elif peak_temp >= 72:
    zone = "WARM"
elif avg_temp >= 65:
    zone = "COOL"
else:
    zone = "COLD"

# Append this run's record
history.append({
    "ratio": ratio,
    "zone": zone,
    "peak_temp": peak_temp,
    "avg_temp": avg_temp,
    "avg_gpu_util": avg_gpu_util,
    "peak_gpu_util": peak_gpu_util,
    "vram_used_gb": vram_used_gb,
    "sys_ram_used_gb": sys_ram_used_gb,
    "sys_ram_total_gb": sys_ram_total_gb,
    "peak_ram_gb": peak_ram_gb,
    "p95_ram_gb": p95_ram_gb,
    "safe_headroom_gb": safe_headroom_gb,
    "effective_headroom_gb": effective_headroom_gb if is_unified_memory else mem_headroom_gb,
    "mem_headroom_gb": mem_headroom_gb,
    "is_unified_memory": is_unified_memory,
    "likely_oom": likely_oom,
    "batch_size": config["training"]["per_device_train_batch_size"],
    "grad_accum": config["training"]["gradient_accumulation_steps"],
    "dataloader_num_workers": config["training"]["dataloader_num_workers"],
    "dataloader_persistent_workers": config["training"].get("dataloader_persistent_workers", False),
    "eff_batch": config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"],
})

history_file.write_text(json.dumps(history, indent=2))
```

#### 8.5c: Apply thermal and memory safety rules

**If `likely_oom` is True**, skip thermal scaling entirely and jump to **8.5d-mem (memory backoff)** — an OOM overrides all thermal decisions.

**Thermal zones** (GPU temperature in °C) — only applied when `likely_oom` is False:

| Zone | Temp Range | Action |
|------|-----------|--------|
| CRITICAL | ≥ 83°C peak | **PAUSE.** Backoff to last WARM config (see 8.5d). Alert user. Wait for temp < 75°C before next run. |
| HOT | 78-82°C peak | Reduce `batch_size` by 1 (min 1). Increase `grad_accum` to maintain eff_batch. Log warning. |
| WARM | 72-77°C peak | Hold current config. This is the target zone — no changes needed. |
| COOL | 65-71°C avg | Headroom available. Proceed to utilization scaling (8.5e). |
| COLD | < 65°C avg | Significant headroom. Aggressive scaling in 8.5e. |

#### 8.5d: CRITICAL backoff — restore last WARM config

When a CRITICAL thermal event occurs, do NOT simply halve the batch size — instead, **restore the exact config from the last run that registered WARM**. This is the last known-safe operating point.

```python
if zone == "CRITICAL":
    # Find last WARM entry in thermal history
    warm_entries = [h for h in history if h["zone"] == "WARM"]

    if warm_entries:
        last_warm = warm_entries[-1]
        new_batch = last_warm["batch_size"]
        new_accum = last_warm["grad_accum"]
        new_workers = last_warm["dataloader_num_workers"]
        reason = f"CRITICAL backoff → restored config from {last_warm['ratio']} (last WARM: peak {last_warm['peak_temp']}°C)"
    else:
        # No WARM history exists — fall back to conservative defaults
        new_batch = max(config["training"]["per_device_train_batch_size"] // 2, 1)
        new_accum = max(eff_batch // new_batch, 1)
        new_workers = 4
        reason = f"CRITICAL backoff → halved batch (no WARM history available)"

    # Alert user
    # Use AskUserQuestion:
    #   header: "THERMAL ALERT"
    #   question: "GPU hit {peak_temp}°C during {ratio}. Backing off to last WARM config
    #              (batch={new_batch}, accum={new_accum} from {last_warm['ratio']}).
    #              Continue?"
    #   options: "Continue with safe config" / "Abort remaining runs"

    # Wait for cooldown before next run
    # Poll GPU temp every 30s until < 75°C
```

**If HOT:** Step down incrementally (not a full backoff):
```python
elif zone == "HOT":
    new_batch = max(batch_size - 1, 1)
    new_accum = max(eff_batch // new_batch, 1)
    new_workers = workers  # keep unchanged
    reason = f"HOT zone ({peak_temp}°C) — reduced batch by 1"
```

#### 8.5d-mem: Memory backoff (OOM recovery)

**Triggered when `likely_oom` is True.** This takes priority over all thermal scaling.

Find the last run in thermal history that did NOT OOM. Restore its config, then step down workers by 1 as additional safety margin:

```python
if likely_oom:
    # Find last non-OOM entry in thermal history
    safe_entries = [h for h in history[:-1] if not h.get("likely_oom", False)]

    if safe_entries:
        last_safe = safe_entries[-1]
        new_batch = last_safe["batch_size"]
        new_accum = last_safe["grad_accum"]
        new_workers = max(last_safe["dataloader_num_workers"] - 1, 2)  # step down 1 from last safe, floor 2
        new_persistent_workers = True  # always enable after OOM — eliminates respawn spikes
        reason = (
            f"OOM RECOVERY → restored config from {last_safe['ratio']} "
            f"(batch={new_batch}, workers={last_safe['dataloader_num_workers']}→{new_workers}) "
            f"+ persistent_workers=true"
        )
    else:
        # No safe history — fall back to conservative defaults
        new_batch = 2
        new_accum = max(eff_batch // new_batch, 1)
        new_workers = 2
        new_persistent_workers = True
        reason = "OOM RECOVERY → conservative defaults (no safe history available)"

    # Alert user
    # Use AskUserQuestion:
    #   header: "MEMORY ALERT"
    #   question: "Run {ratio} appears to have been OOM-killed (GPU idle + RAM at {peak_ram_gb:.0f}/{mem_total_gb:.0f} GB).
    #              Restoring last safe config: batch={new_batch}, accum={new_accum}, workers={new_workers}, persistent_workers=true.
    #              Continue?"
    #   options: "Continue with safe config" / "Abort remaining runs"

    # Skip 8.5e entirely — go straight to 8.5f
```

#### 8.5e: Thermal exploitation ladder (only if thermal zone is COOL or COLD, and `likely_oom` is False)

When thermal and memory headroom exist, exploit them using a **prioritized ladder** that applies zero-memory changes first, then low-memory changes, and only touches batch size as a last resort. Each rung is applied independently — multiple rungs can fire in one planning step.

```python
headroom = effective_headroom_gb if is_unified_memory else mem_headroom_gb
reason_parts = []

# Start with current config
new_batch = batch_size
new_accum = grad_accum
new_prefetch = config["training"].get("dataloader_prefetch_factor", 2)
new_save_steps = config["training"]["save_steps"]
new_eval_steps = config["training"]["eval_steps"]
```

**Rung 1: `prefetch_factor`** (near-zero memory cost, ~200 MB per worker per increment)

Increases how many batches each worker pre-loads into the queue. Directly addresses GPU idle gaps between batch consumption and next batch arrival. Cost: ~200 MB × num_workers per +1 increment.

```python
if avg_gpu_util < 80% and new_prefetch < 4:
    new_prefetch = min(new_prefetch + 1, 4)  # cap at 4 — diminishing returns beyond this
    reason_parts.append(f"prefetch_factor {config['training'].get('dataloader_prefetch_factor', 2)}→{new_prefetch} (reduce GPU idle gaps)")
```

**Rung 2: `save_steps`** (zero memory cost, reduces checkpoint write stalls)

Each checkpoint save serializes ~3.3 GB adapter weights to disk, stalling the GPU for 30-60s (visible as 6-7% util dips in telemetry). The memory watchdog provides a safety net between checkpoints.

```python
if avg_gpu_util < 80% and new_save_steps < 400:
    new_save_steps = min(new_save_steps * 2, 400)  # cap at 400 — watchdog covers the gap
    reason_parts.append(f"save_steps {config['training']['save_steps']}→{new_save_steps} (fewer checkpoint stalls)")
```

**Rung 3: `eval_steps`** (zero memory cost, reduces eval pauses)

Eval runs the val set (~5K examples) in inference mode, pausing training. Less frequent eval means more training steps per hour. Loss is still logged every `logging_steps` (10) for monitoring.

```python
if avg_gpu_util < 80% and new_eval_steps < 200:
    new_eval_steps = min(new_eval_steps * 2, 200)  # cap at 200
    reason_parts.append(f"eval_steps {config['training']['eval_steps']}→{new_eval_steps} (fewer eval pauses)")
```

**Rung 4: Batch size +1** (last resort — model-scale-aware, requires warmup probe)

Only attempted when rungs 1-3 are maxed out AND the model scale permits it. On unified memory (DGX Spark), a failed allocation can cause a driver-level deadlock (system freeze, no clean CUDA OOM), so batch scaling requires a **warmup probe** in Step 6 of the next run.

**Model-scale-aware policy:**

The practical batch ceiling depends on model size (from DGX Spark UGC):

| Model Scale | Params | Batch Ceiling | Min Headroom | Notes |
|---|---|---|---|---|
| Small | ≤1B | 64 | 15% of total | I/O bound — batch freely |
| Medium | 1B-13B | 16 | 20% of total | Balanced — room to explore |
| Large | 13B-30B | 8 | 25% of total | Model dominates |
| XL | 30B+ | 4 | 30% of total | At the cliff — batch increase rarely safe |

The **85% memory ceiling rule** (stop before 85% of total memory) is the universal safety gate, but larger models need even more margin because their activation memory scales non-linearly with batch size.

```python
import re

# Detect model scale from model name or param count
model_name = config["model"]["name"]  # e.g., "Qwen/Qwen3-30B-A3B"
# Extract param count from name (e.g., "30B" → 30)
param_match = re.search(r"(\d+)[Bb]", model_name)
param_billions = int(param_match.group(1)) if param_match else 0

# Scale-aware ceilings
if param_billions >= 30:
    batch_ceiling = 4
    min_headroom_pct = 0.30  # 30% of total memory must remain free
    scale_label = "XL (30B+)"
elif param_billions >= 13:
    batch_ceiling = 8
    min_headroom_pct = 0.25
    scale_label = "Large (13-30B)"
elif param_billions >= 1:
    batch_ceiling = 16
    min_headroom_pct = 0.20
    scale_label = "Medium (1-13B)"
else:
    batch_ceiling = 64
    min_headroom_pct = 0.15
    scale_label = "Small (≤1B)"

min_headroom_gb = mem_total_gb * min_headroom_pct
mem_usage_pct = peak_ram_gb / mem_total_gb

rungs_1_to_3_maxed = (new_prefetch >= 4 and new_save_steps >= 400 and new_eval_steps >= 200)
below_ceiling = batch_size < batch_ceiling
has_headroom = headroom > min_headroom_gb
below_85pct = mem_usage_pct < 0.85
gpu_underutilized = avg_gpu_util < 65%

if rungs_1_to_3_maxed and below_ceiling and has_headroom and below_85pct and gpu_underutilized:
    proposed_batch = min(batch_size + 1, batch_ceiling)
    if proposed_batch != batch_size:
        new_batch = proposed_batch
        new_accum = max(eff_batch // new_batch, 1)
        reason_parts.append(
            f"batch_size {batch_size}→{new_batch} (scale={scale_label}, ceiling={batch_ceiling}, "
            f"mem={mem_usage_pct:.0%}, headroom={headroom:.0f}/{min_headroom_gb:.0f} GB min, "
            f"util={avg_gpu_util:.0f}%); REQUIRES warmup probe in Step 6"
        )
elif rungs_1_to_3_maxed and not below_ceiling:
    reason_parts.append(
        f"batch_size held at {batch_size} (AT CEILING for {scale_label} — "
        f"UGC reports batch {batch_ceiling} is practical max for {param_billions}B model on Spark)"
    )
elif rungs_1_to_3_maxed and not has_headroom:
    reason_parts.append(
        f"batch_size held at {batch_size} (headroom {headroom:.0f} GB < {min_headroom_gb:.0f} GB min "
        f"for {scale_label})"
    )
    # Flag for Step 6: run 1 real training step at new batch size, check memory survived
    # If probe OOMs, revert to previous batch size and proceed
```

If no rungs fired (already well-utilized or no headroom):
```python
if not reason_parts:
    reason_parts.append(f"hold config (util={avg_gpu_util:.0f}%, headroom={headroom:.0f} GB — no changes needed)")
```

**Worker scaling** — adjust `dataloader_num_workers` conservatively:
- Only increase workers if `avg_gpu_util < 70%` AND `headroom > 15` — low GPU util with tight memory means workers are NOT the bottleneck
- Increase by 1 at a time (not doubling)
- **Hard cap**: `min(cpu_count // 2, 6)` on unified memory, `min(cpu_count // 2, 16)` on discrete GPU
- If `headroom < 5` for **2 consecutive runs**: **decrease** workers by 1 (min 2) — sustained memory pressure confirmed. A single run below 5 GB could be a transient spike; requiring 2 consecutive runs avoids unnecessary worker reduction that hurts GPU utilization.
- The watchdog (2 GB threshold) handles acute within-run pressure — the planner handles structural cross-run trends.

```python
if is_unified_memory:
    max_workers = min(os.cpu_count() // 2, 6)  # hard cap for unified memory
else:
    max_workers = min(os.cpu_count() // 2, 16)

# Check for sustained memory pressure: headroom < 5 GB in last 2 consecutive runs
recent_runs = history[-2:] if len(history) >= 2 else history
sustained_pressure = (
    len(recent_runs) >= 2 and
    all(h.get("effective_headroom_gb", 999) < 5 for h in recent_runs)
)

if headroom < 5 and sustained_pressure:
    new_workers = max(workers - 1, 2)
    reason_parts.append(f"workers {workers}→{new_workers} (sustained pressure: headroom <5 GB for 2 consecutive runs)")
elif avg_gpu_util < 70% and headroom > 15 and workers < max_workers:
    new_workers = workers + 1
    reason_parts.append(f"workers {workers}→{new_workers} (GPU starving, headroom OK)")
else:
    new_workers = workers

reason = "; ".join(reason_parts)
```

**Persistent workers** — always preserve the current setting. If `dataloader_persistent_workers` was enabled (especially after an OOM recovery), never disable it automatically:
```python
new_persistent_workers = config["training"].get("dataloader_persistent_workers", False)
# Once enabled, persistent_workers stays on — it stabilizes memory and improves GPU util
```

#### 8.5f: Apply and log adjustment

```python
import yaml

base_config = yaml.safe_load(open("config/train_config.yaml"))
old_batch = base_config["training"]["per_device_train_batch_size"]
old_accum = base_config["training"]["gradient_accumulation_steps"]
old_workers = base_config["training"]["dataloader_num_workers"]
old_persistent = base_config["training"].get("dataloader_persistent_workers", False)
old_prefetch = base_config["training"].get("dataloader_prefetch_factor", 2)
old_save = base_config["training"]["save_steps"]
old_eval = base_config["training"]["eval_steps"]

base_config["training"]["per_device_train_batch_size"] = new_batch
base_config["training"]["gradient_accumulation_steps"] = new_accum
base_config["training"]["dataloader_num_workers"] = new_workers
base_config["training"]["dataloader_persistent_workers"] = new_persistent_workers
base_config["training"]["dataloader_prefetch_factor"] = new_prefetch
base_config["training"]["save_steps"] = new_save_steps
base_config["training"]["eval_steps"] = new_eval_steps

yaml.dump(base_config, open("config/train_config.yaml", "w"))

# Log the adjustment
adjustment_log = f"""
### Adaptive adjustment after {ratio}
- Thermal zone: {zone} (peak={peak_temp}°C, avg={avg_temp}°C)
- GPU util: avg={avg_gpu_util}%, peak={peak_gpu_util}%
- Memory: peak={peak_ram_gb:.0f}/{mem_total_gb:.0f} GB, effective_headroom={effective_headroom_gb:.0f} GB [{'unified' if is_unified_memory else 'discrete VRAM'}]
- OOM detected: {likely_oom}
- Thermal ladder applied:
  - prefetch_factor: {old_prefetch} → {new_prefetch} (rung 1)
  - save_steps: {old_save} → {new_save_steps} (rung 2)
  - eval_steps: {old_eval} → {new_eval_steps} (rung 3)
  - batch_size: {old_batch} → {new_batch} (rung 4 — last resort)
- grad_accum: {old_accum} → {new_accum}
- eff_batch: {old_batch * old_accum} → {new_batch * new_accum}
- workers: {old_workers} → {new_workers}
- persistent_workers: {old_persistent} → {new_persistent_workers}
- thermal_history: {len(history)} runs recorded, {len([h for h in history if h['zone']=='WARM'])} WARM, {len([h for h in history if h.get('likely_oom')])} OOM
- reason: {reason}
"""
with open("telemetry/training/adaptive_adjustments.md", "a") as f:
    f.write(adjustment_log)
print(adjustment_log)
```

**If batch_size was increased (rung 4):** Flag the next run for a warmup probe in Step 6. The dry run (`--dry-run`) validates config but doesn't process a real batch. The warmup probe runs 1 actual training step and checks that memory survived:

```python
if new_batch > old_batch:
    # Write a flag file that Step 6 checks
    Path("telemetry/training/_warmup_probe_required").write_text(
        f"batch_size increased {old_batch}→{new_batch} by adaptive planner\n"
        f"Run 1 real training step and verify MemAvailable > {OOM_WATCHDOG_THRESHOLD_MB} MB\n"
        f"If probe fails: revert to batch_size={old_batch}, grad_accum={old_accum}\n"
    )
    print(f"  ⚠ Warmup probe flagged for next run (batch {old_batch}→{new_batch})")
```

**Step 6 warmup probe** (added behavior when `_warmup_probe_required` exists):

After the normal dry run succeeds, if the warmup probe flag exists:
1. Run `python -m scripts.train_model --config <run_config> --max-steps 1` (1 real step)
2. Read `/proc/meminfo` → check `MemAvailable > 2048 MB`
3. If OK: delete the flag, proceed to Step 7
4. If OOM or `MemAvailable < 2048 MB`: revert batch_size in config, delete the flag, re-run dry run, proceed with safe config
5. Log probe result to `adaptive_adjustments.md`

**Then regenerate the next ratio's config overlay** using the updated base config before proceeding to Step 1 of the next ratio.

#### 8.5g: Live thermal guard during training (Step 7)

During long-running training (Step 7), the telemetry observer agents should also implement a **live thermal guard**:

- If any check sees GPU temp ≥ 83°C: immediately touch `telemetry/training/_thermal_pause`
- The orchestrator checks for `_thermal_pause` periodically (or after training completes)
- If `_thermal_pause` exists when training ends (regardless of exit code):
  1. Record the thermal event in `thermal_history.json` and `adaptive_adjustments.md`
  2. Apply CRITICAL backoff rules (8.5d) — restore last WARM config
  3. Ask user before continuing

This ensures we never cook the GPU across multi-day sequential runs.

#### Thermal history file format

`telemetry/training/thermal_history.json` — append-only array of run records:

```json
[
  {
    "ratio": "30_70",
    "zone": "COOL",
    "peak_temp": 70,
    "avg_temp": 65,
    "avg_gpu_util": 77,
    "peak_gpu_util": 95,
    "vram_used_gb": null,
    "sys_ram_used_gb": 92.5,
    "sys_ram_total_gb": 119.7,
    "peak_ram_gb": 104.0,
    "p95_ram_gb": 101.5,
    "safe_headroom_gb": 15.7,
    "effective_headroom_gb": 10.7,
    "mem_headroom_gb": 27.2,
    "is_unified_memory": true,
    "likely_oom": false,
    "batch_size": 4,
    "grad_accum": 4,
    "dataloader_num_workers": 4,
    "dataloader_persistent_workers": false,
    "eff_batch": 16
  },
  {
    "ratio": "40_60",
    "zone": "COOL",
    "peak_temp": 79,
    "avg_temp": 68,
    "avg_gpu_util": 80,
    "peak_gpu_util": 96,
    "vram_used_gb": null,
    "sys_ram_used_gb": 110,
    "sys_ram_total_gb": 119.7,
    "peak_ram_gb": 119.3,
    "p95_ram_gb": 117.8,
    "safe_headroom_gb": 0.4,
    "effective_headroom_gb": -4.6,
    "mem_headroom_gb": 9.7,
    "is_unified_memory": true,
    "likely_oom": true,
    "batch_size": 8,
    "grad_accum": 2,
    "dataloader_num_workers": 8,
    "dataloader_persistent_workers": false,
    "eff_batch": 16
  }
]
```

This file persists across skill invocations — if the user re-runs `/run-training` after a context reset, the thermal history from prior runs is preserved and the adaptive logic picks up where it left off.

### Step 9: Report (after all runs complete)

**9a: Print run status table.**

```python
status = dgx.status_report()
print(f"\nTraining runs complete:")
for ratio in selected_ratios:
    run_name = f"qwen3-30b-wp-{ratio}"
    adapter_exists = Path(f"adapters/{run_name}/adapter_config.json").exists()
    merged_exists = Path(f"models/{run_name}-merged/config.json").exists()
    print(f"  {run_name}: adapter={'✓' if adapter_exists else '✗'}  merged={'✓' if merged_exists else '✗'}")
```

**9b: Invoke final cross-run review-telemetry.**

If `$TELEMETRY` (either mode), produce a cross-run comparison from `thermal_history.json`:

```
# Read thermal_history.json for config progression (written by Step 8.5b for every run)
Read telemetry/training/thermal_history.json
Read telemetry/training/adaptive_adjustments.md

# If $OBSERVE: also read per-run _summary.md files for richer data
if $OBSERVE:
    for each ratio in selected_ratios:
        Read telemetry/training/{ratio_timestamp}/_summary.md
        Read telemetry/packaging/{ratio_timestamp}/_summary.md
        Extract: final_loss, training_duration, alerts

# Write cross-run comparison summary
Write telemetry/training/cross_run_summary.md:

    # Cross-Run Training Summary
    ## Config Progression (adaptive resource planning)
    | Run | Ratio | batch | accum | workers | Zone | Peak Temp | Avg Util | Duration |
    |-----|-------|-------|-------|---------|------|-----------|----------|----------|
    | 1   | 30/70 | 4     | 4     | 4       | COOL | 70°C      | 73%      | 33h      |
    | 2   | 40/60 | 8     | 2     | 8       | WARM | 74°C      | 85%      | 18h      |
    | ... |

    ## Canonical Thermal Logs
    {List all JSONL files: telemetry/training/{model}_{date}_{ratio}_thermal.jsonl}

    ## Alerts Across All Runs
    {Consolidated WARNING/CRITICAL events from _summary.md files (observe) or thermal logs (monitor)}

    ## Recommendations
    {Which ratio had best loss? Any thermal concerns? Suggested eval order.}

print(f"\nTelemetry reports: telemetry/training/")
print(f"Cross-run summary: telemetry/training/cross_run_summary.md")
print(f"Thermal history: telemetry/training/thermal_history.json")
print(f"Adaptive adjustments: telemetry/training/adaptive_adjustments.md")
```

**9c: Next steps.**

```
print(f"\nNext: /wp-finetune:run-evaluation to compare model quality across ratios")
```

## Checkpoint Storage

Each training run produces isolated artifacts:

```
adapters/
  qwen3-wp-30_70/           # LoRA adapter for 30/70 ratio
    adapter_config.json
    adapter_model.safetensors
    checkpoint-200/          # Intermediate checkpoints
    checkpoint-400/
    ...
  qwen3-wp-50_50/           # LoRA adapter for 50/50 ratio
    adapter_config.json
    adapter_model.safetensors
    ...
  tokenizer/                 # Shared tokenizer (not per-run)

models/
  Qwen3-30B-A3B/             # Shared base model (not per-run)
  qwen3-wp-30_70-merged/     # Merged model for 30/70 ratio
  qwen3-wp-50_50-merged/     # Merged model for 50/50 ratio
  ...

config/
  train_config.yaml           # Base config
  train_config_30_70.yaml     # Per-run config overlay
  train_config_50_50.yaml     # Per-run config overlay
  ...
```

**Why isolated:** Each ratio produces a different model. Keeping them separate lets you:
- Run eval on each to find the best ratio
- Serve any model via vLLM (`--model models/qwen3-wp-50_50-merged/`)
- Roll back to any ratio without retraining
- Compare MLflow runs side-by-side (`mlflow ui --backend-store-uri mlruns/`)

## Recovery Logic

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Container not running | `validate(["container:unsloth_studio"])` fails | `ensure_ready()` starts it |
| Project not mounted | `validate(["mounted:unsloth_studio"])` fails | `ensure_ready()` restarts with EXTRA_MOUNTS |
| Deps missing | `validate(["deps:unsloth_studio"])` fails | `ensure_ready()` installs them |
| OOM | `validate(["memory:70"])` fails | Report top consumers, suggest `docker stop` |
| Download interrupted | `execute()` returns non-zero | Re-run (resume support built in) |
| Training interrupted | `execute()` returns non-zero | Re-run skill — auto-detects checkpoints and passes `--resume` |
| Merge fails | `execute()` returns non-zero | Adapter is safe, suggest `--lora-modules` fallback |
| GPU not accessible | `validate(["gpu:unsloth_studio"])` fails | Check container has `--gpus all` |
| Previous run exists | Idempotency check finds adapter | Skip to next ratio |

**Key principle:** Every step is idempotent. Re-running the skill picks up where it left off — completed runs are skipped, interrupted runs resume from the latest checkpoint.

## Telemetry Integration

Background telemetry agents (spawned by `/observe-training`) can poll:

```python
dgx = get_toolbox()
status = dgx.status_report()
# status["containers"] — running container states
# status["memory"] — system memory
# status["artifacts"] — pipeline progress (what exists on disk)
# status["execution_log"] — recent command results
# status["endpoints"] — vLLM/LiteLLM URLs for inference monitoring
```

## Key Constraints

- `load_in_4bit=False` — QLoRA off-limits for MoE
- `output_router_logits=True` — MoE load balancing monitoring
- `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings
- Base config from `config/train_config.yaml`, per-run overlay in `config/train_config_{ratio}.yaml`
- All paths from `config/dgx_toolbox.yaml`
- Adapter saved separately before merge (defense-in-depth)
- Pinned versions: transformers==4.56.2, trl==0.24.0, datasets==4.3.0, bitsandbytes==0.48.0
- Base model + tokenizer shared across runs (downloaded once)
