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

> **Recommended:** Say `/observe-training` before starting to spawn background telemetry agents.
> They consume `dgx.status_report()` for GPU health, training metrics, and checkpoint monitoring.

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

Telemetry is **enabled by default** — it is required for adaptive resource planning (Step 8.5) which adjusts batch size, gradient accumulation, and workers between runs based on GPU thermal and utilization data.

Use AskUserQuestion:
- header: "Telemetry"
- question: "Telemetry is enabled by default (required for adaptive resource planning). Disable?"
- options:
  - "Keep enabled (Recommended)" → set `$TELEMETRY = true`
  - "Disable telemetry" → set `$TELEMETRY = false`

Store as `$TELEMETRY` (true/false).

**If the user selects "Disable telemetry"**, show a warning via AskUserQuestion:
- header: "Warning"
- question: "Disabling telemetry also disables adaptive resource planning. Without it, training config will NOT auto-adjust between runs — GPU may be underutilized or overheat without detection. Are you sure?"
- options:
  - "Keep telemetry enabled" → set `$TELEMETRY = true`
  - "Disable anyway — I'll monitor manually" → set `$TELEMETRY = false`

**When `$TELEMETRY = true`**, the orchestrator spawns the appropriate observe skill agents at each training phase:

| Training phase | Observe skill spawned | Why |
|---------------|----------------------|-----|
| Step 4: Download model | `wp-finetune:observe-data-pipeline` | Network I/O, disk usage during multi-GB download |
| Step 7: Train | `wp-finetune:observe-training` | GPU metrics, thermal, loss curves, checkpoint integrity (6 agents) |
| Step 8: Merge adapter | `wp-finetune:observe-packaging` | Merge progress, file integrity, disk usage |

**Between runs:** After each ratio's Step 8 completes, spawn `wp-finetune:review-telemetry` to consolidate that run's telemetry into `_summary.md` before starting the next ratio.

**After all runs:** Final `wp-finetune:review-telemetry` produces a cross-run comparison summary.

**Lifecycle per observe agent spawn:**
1. Spawn observe agents in background before the long-running step
2. Run the step (download/train/merge)
3. Touch `_stop` file to signal agents to write final summaries and exit
4. Spawn review-telemetry to consolidate
5. Proceed to next step

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

**If `$TELEMETRY` and model not yet downloaded:** Spawn `wp-finetune:observe-data-pipeline` agents in background before download (monitors network I/O, disk usage). Touch `_stop` after download completes.

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.download_model",
    idempotency_check="models/Qwen3-30B-A3B/config.json",
)
print(result.summary())
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

**If `$TELEMETRY`:** Spawn `wp-finetune:observe-training` agents in background before training starts. These 6 agents monitor GPU metrics, thermal throttling, training loss, disk I/O, checkpoint integrity, and container health. They write to `telemetry/training/{timestamp}/` and stop when `_stop` is touched after training completes.

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.train_model",
    "--config", f"config/train_config_{ratio}.yaml",
    idempotency_check=f"adapters/{run_name}/adapter_config.json",
    timeout=None,  # No timeout — training takes 6-12 hours
)
print(result.summary())
if not result.ok:
    print(f"Training failed for {run_name}. Check MLflow logs: mlflow ui --backend-store-uri mlruns/")
    print("To resume: run this skill again (idempotency will skip completed runs)")
```

### Step 8: Merge adapter (per-run isolated)

**If `$TELEMETRY`:** Spawn `wp-finetune:observe-packaging` agents in background before merge. Touch `_stop` after merge completes. Then spawn `wp-finetune:review-telemetry` to consolidate this run's telemetry into `_summary.md` before proceeding to the next ratio.

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

### Step 8.5: Adaptive resource planning (between runs)

**After each run's merge completes and before the next ratio starts**, reassess GPU headroom using telemetry from the just-completed run and adjust training config for the next run.

**Requires `$TELEMETRY = true`.** If telemetry is disabled, skip this step entirely (config stays static across all runs).

#### 8.5a: Collect metrics from completed run

Parse the telemetry monitor file for the completed ratio:

```python
import re
from pathlib import Path

monitor_file = Path(f"telemetry/training/ratio_{ratio}_v2_monitor.md")
if not monitor_file.exists():
    monitor_file = Path(f"telemetry/training/ratio_{ratio}_monitor.md")

gpu_utils = []
gpu_temps = []
for line in monitor_file.read_text().splitlines():
    # Parse: "62 %, [N/A], [N/A], 63"
    m = re.match(r"(\d+)\s*%.*,\s*(\d+)\s*$", line.strip())
    if m:
        gpu_utils.append(int(m.group(1)))
        gpu_temps.append(int(m.group(2)))

# Also get VRAM from nvidia-smi (live)
# docker exec unsloth-headless nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Compute:
- `avg_gpu_util` — mean GPU utilization across all checks
- `peak_gpu_util` — max GPU utilization
- `avg_temp` — mean temperature
- `peak_temp` — max temperature
- `vram_used_gb` — current VRAM usage (from nvidia-smi)
- `vram_total_gb` — total VRAM capacity
- `vram_headroom_gb` — total minus used

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
    "batch_size": config["training"]["per_device_train_batch_size"],
    "grad_accum": config["training"]["gradient_accumulation_steps"],
    "dataloader_num_workers": config["training"]["dataloader_num_workers"],
    "eff_batch": config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"],
})

history_file.write_text(json.dumps(history, indent=2))
```

#### 8.5c: Apply thermal safety rules

**Thermal zones** (GPU temperature in °C):

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

#### 8.5e: Utilization scaling (only if thermal zone is COOL or COLD)

Scale based on GPU utilization headroom AND VRAM headroom:

```
IF avg_gpu_util < 60% AND vram_headroom_gb > 20:
    # Aggressive: double batch_size, halve grad_accum
    new_batch = min(batch_size * 2, 16)  # cap at 16
    new_accum = max(eff_batch // new_batch, 1)

ELIF avg_gpu_util < 75% AND vram_headroom_gb > 10:
    # Moderate: increase batch_size by 50%
    new_batch = min(int(batch_size * 1.5), 16)
    new_accum = max(eff_batch // new_batch, 1)

ELIF avg_gpu_util > 90%:
    # Already well-utilized, hold steady
    new_batch = batch_size
    new_accum = grad_accum

ELSE:
    # Reasonable utilization, no change
    new_batch = batch_size
    new_accum = grad_accum
```

Also adjust `dataloader_num_workers`:
- If `avg_gpu_util < 70%` and workers < 8: increase workers (CPU may be bottleneck)
- Cap at `min(cpu_count // 2, 16)`

#### 8.5f: Apply and log adjustment

```python
import yaml

base_config = yaml.safe_load(open("config/train_config.yaml"))
old_batch = base_config["training"]["per_device_train_batch_size"]
old_accum = base_config["training"]["gradient_accumulation_steps"]
old_workers = base_config["training"]["dataloader_num_workers"]

base_config["training"]["per_device_train_batch_size"] = new_batch
base_config["training"]["gradient_accumulation_steps"] = new_accum
base_config["training"]["dataloader_num_workers"] = new_workers

yaml.dump(base_config, open("config/train_config.yaml", "w"))

# Log the adjustment
adjustment_log = f"""
### Adaptive adjustment after {ratio}
- Thermal zone: {zone} (peak={peak_temp}°C, avg={avg_temp}°C)
- GPU util: avg={avg_gpu_util}%, peak={peak_gpu_util}%
- VRAM: {vram_used_gb:.0f}/{vram_total_gb:.0f} GB ({vram_headroom_gb:.0f} GB headroom)
- batch_size: {old_batch} → {new_batch}
- grad_accum: {old_accum} → {new_accum}
- eff_batch: {old_batch * old_accum} → {new_batch * new_accum}
- workers: {old_workers} → {new_workers}
- thermal_history: {len(history)} runs recorded, {len([h for h in history if h['zone']=='WARM'])} WARM
- reason: {reason}
"""
with open("telemetry/training/adaptive_adjustments.md", "a") as f:
    f.write(adjustment_log)
print(adjustment_log)
```

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
    "vram_used_gb": 92.5,
    "batch_size": 4,
    "grad_accum": 4,
    "dataloader_num_workers": 4,
    "eff_batch": 16
  },
  {
    "ratio": "40_60",
    "zone": "WARM",
    "peak_temp": 74,
    "avg_temp": 70,
    "avg_gpu_util": 85,
    "peak_gpu_util": 97,
    "vram_used_gb": 110,
    "batch_size": 8,
    "grad_accum": 2,
    "dataloader_num_workers": 8,
    "eff_batch": 16
  }
]
```

This file persists across skill invocations — if the user re-runs `/run-training` after a context reset, the thermal history from prior runs is preserved and the adaptive logic picks up where it left off.

### Step 9: Report (after all runs complete)

**If `$TELEMETRY`:** Spawn final `wp-finetune:review-telemetry` to produce a cross-run comparison summary covering all ratios trained. Output: `telemetry/training/cross_run_summary.md` with per-ratio peak GPU temp, final loss, training duration, and any alerts.

```python
status = dgx.status_report()
print(f"\nTraining runs complete:")
for ratio in selected_ratios:
    run_name = f"qwen3-wp-{ratio}"
    adapter_exists = Path(f"adapters/{run_name}/adapter_config.json").exists()
    merged_exists = Path(f"models/{run_name}-merged/config.json").exists()
    print(f"  {run_name}: adapter={'✓' if adapter_exists else '✗'}  merged={'✓' if merged_exists else '✗'}")

if TELEMETRY:
    print(f"\nTelemetry reports: telemetry/training/")
    print(f"Cross-run summary: telemetry/training/cross_run_summary.md")

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
| Training interrupted | `execute()` returns non-zero | Re-run with `--resume` flag |
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
