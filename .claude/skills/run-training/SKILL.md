# Skill: run-training

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

### Step 0: Select dataset exports

List available ratio exports and let the user choose:

```bash
ls data/final_dataset/ratio_*/metadata.json
```

Present a selection:
```
Available dataset exports:

| # | Ratio | Gen    | Judge  | Total  | Train  |
|---|-------|--------|--------|--------|--------|
| 1 | 30/70 | 13,071 | 30,498 | 43,569 | 34,855 |
| 2 | 40/60 | 20,332 | 30,498 | 50,830 | 40,664 |
| 3 | 50/50 | 30,498 | 30,498 | 60,996 | 48,796 |
| 4 | 60/40 | 45,747 | 30,498 | 76,245 | 60,996 |
| 5 | 70/30 | 71,162 | 30,498 | 101,660| 81,328 |

Select exports to train (comma-separated, e.g. "2,3,4" or "all"):
```

Use AskUserQuestion for selection. Store as `$SELECTED_RATIOS` list.

**For each selected ratio**, execute Steps 1-8 below with run-specific paths:
- `run_name` = `qwen3-wp-{ratio}` (e.g., `qwen3-wp-50_50`)
- `data_dir` = `data/final_dataset/ratio_{ratio}/`
- `adapter_dir` = `adapters/{run_name}/`
- `merged_dir` = `models/{run_name}-merged/`

### Step 1: Configure run

Before training, create a run-specific config overlay:

```python
import yaml, shutil

base_config = yaml.safe_load(open("config/train_config.yaml"))

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
    print(f"Training failed for {run_name}. Check W&B for loss curves.")
    print("To resume: run this skill again (idempotency will skip completed runs)")
```

### Step 8: Merge adapter (per-run isolated)

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

### Step 9: Report (after all runs complete)

```python
status = dgx.status_report()
print(f"\nTraining runs complete:")
for ratio in selected_ratios:
    run_name = f"qwen3-wp-{ratio}"
    adapter_exists = Path(f"adapters/{run_name}/adapter_config.json").exists()
    merged_exists = Path(f"models/{run_name}-merged/config.json").exists()
    print(f"  {run_name}: adapter={'✓' if adapter_exists else '✗'}  merged={'✓' if merged_exists else '✗'}")

print(f"\nNext: /run-evaluation to compare model quality across ratios")
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
- Compare W&B runs side-by-side (each run has a unique name)

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
