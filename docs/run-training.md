# Skill: run-training

Run the complete training pipeline via the DGX Toolbox execution engine. The skill declares intent — `dgx_toolbox.py` resolves paths, validates state, manages containers, and executes dynamically.

## Architecture

```
Skill (this file — intent + recovery logic)
  → dgx_toolbox.py (resolve paths, validate state, manage containers, execute)
    → Docker commands (generated dynamically from config, not hardcoded)
      → Python scripts (inside container, idempotent)
        → Output (adapters/, models/)
```

## Telemetry

> **Recommended:** Say `/observe-training` before starting to spawn background telemetry agents.
> They consume `dgx.status_report()` for GPU health, training metrics, and checkpoint monitoring.

## Trigger

User says: "run training", "train the model", "start DGX training", "/run-training"

## Process

Execute these steps in order. Each step uses `dgx_toolbox.py` — never hardcode paths or docker commands.

### Step 1: Validate

```python
from scripts.dgx_toolbox import get_toolbox
dgx = get_toolbox()

result = dgx.validate(["toolbox", "training_data", "config", "memory:70"])
print(result.report())
if not result.ok:
    # Report failures to user with actionable fixes
    for f in result.failures:
        print(f"  FIX: {f.name} — {f.message}")
        if f.details.get("running_containers"):
            print(f"    Running containers: {f.details['running_containers']}")
        if f.details.get("top_processes"):
            print(f"    Top memory: {f.details['top_processes'][:3]}")
    # STOP — do not proceed until all checks pass
```

If validation fails, tell the user what to fix and stop. Do NOT proceed with partial validation.

### Step 2: Ensure container ready

```python
ready = dgx.ensure_ready("unsloth_studio")
print(ready.report())
if not ready.ok:
    # Container failed to start/mount/install deps
    for f in ready.failures:
        print(f"  ISSUE: {f.name} — {f.message}")
    # STOP
```

This automatically:
- Starts the Unsloth Studio container via dgx-toolbox (with EXTRA_MOUNTS)
- Waits for setup
- Verifies project is mounted
- Installs pinned deps if missing
- Checks GPU access

### Step 3: Download model (idempotent)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.download_model",
    idempotency_check="models/Qwen3-30B-A3B/config.json",
)
print(result.summary())
if not result.ok:
    # Download failed — network? disk space?
    print(f"STDERR: {result.stderr[-500:]}")
    # Can retry — download_model.py has resume support
```

### Step 4: Extend tokenizer (idempotent)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.prepare_tokenizer",
    idempotency_check="adapters/tokenizer/tokenizer_config.json",
)
print(result.summary())
if not result.ok:
    print(f"STDERR: {result.stderr[-500:]}")
```

### Step 5: Dry run (validate config before committing to hours of training)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.train_model", "--dry-run",
    capture=True,
)
print(result.stdout)
if not result.ok:
    # Config issue — fix before proceeding
    print(f"Dry run failed: {result.stderr[-500:]}")
    # STOP — do not start real training
```

**Present dry run output to user.** If it shows errors, fix them. If it shows a valid training summary, proceed.

### Step 6: Train (long-running, idempotent)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.train_model",
    idempotency_check="adapters/qwen3-wp/adapter_config.json",
    timeout=None,  # No timeout — training takes 6-12 hours
)
print(result.summary())
if not result.ok:
    # Check if partial — can resume
    print("Training failed. Check W&B for loss curves.")
    print("To resume: run this skill again (idempotency will skip completed steps)")
```

### Step 7: Merge adapter (idempotent)

```python
result = dgx.execute(
    "unsloth_studio",
    "python", "-m", "scripts.merge_adapter",
    idempotency_check="models/Qwen3-30B-A3B-merged/config.json",
)
print(result.summary())
if not result.ok:
    print("Merge failed. Adapter is safe at adapters/qwen3-wp/")
    print("Fallback: serve with vLLM --lora-modules")
```

### Step 8: Status report

```python
status = dgx.status_report()
print(f"Model downloaded: {status['artifacts']['model_downloaded']}")
print(f"Model shards: {status['artifacts']['model_shards']}")
print(f"Tokenizer ready: {status['artifacts']['tokenizer_ready']}")
print(f"Adapter trained: {status['artifacts']['adapter_trained']}")
print(f"Model merged: {status['artifacts']['model_merged']}")
print(f"Memory: {status['memory']}")
print(f"\nExecution log:")
for entry in status['execution_log']:
    print(f"  {entry['status']}")
print(f"\nNext: /gsd:execute-phase 4 (evaluation via dgx-toolbox eval-toolbox)")
```

## Recovery Logic

The skill should handle these failure modes:

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

**Key principle:** Every step is idempotent. If the skill fails at step 5, re-running starts from step 5 (steps 1-4 are skipped via idempotency checks). The user never needs to manually track where things left off.

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

This structured output feeds directly into the observe-training agents without them needing to parse logs or guess state.

## Key Constraints

- `load_in_4bit=False` — QLoRA off-limits for MoE
- `output_router_logits=True` — MoE load balancing monitoring
- `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings
- All config from `config/train_config.yaml`
- All paths from `config/dgx_toolbox.yaml`
- Adapter saved separately before merge (defense-in-depth)
- Pinned versions: transformers==4.56.2, trl==0.24.0, datasets==4.3.0, bitsandbytes==0.48.0
