# Skill: run-training

Run the complete training pipeline on DGX Spark via the DGX Toolbox Unsloth Studio container. Single invocation, handles failures and resume.

All GPU steps run inside the container — the skill handles `docker exec` automatically.

## Idempotency

Every step is safe to re-run. If the pipeline fails partway through, just re-invoke "run training" — completed steps are automatically skipped:

| Step | Skip condition |
|------|---------------|
| Download model | `models/Qwen3-30B-A3B/*.safetensors` exist |
| Extend tokenizer | `adapters/tokenizer/tokenizer_config.json` exists with special tokens in vocab |
| Train model | `adapters/qwen3-wp/adapter_config.json` exists (use `--resume` to continue partial training) |
| Merge adapter | `models/Qwen3-30B-A3B-merged/config.json` exists and special tokens verify as single-token IDs |

## Trigger

User says: "run training", "train the model", "start DGX training", "/run-training"

## Prerequisites

Before running, verify:
1. DGX Toolbox installed: `python scripts/dgx_toolbox.py`
2. Training data exists: `ls data/final_dataset/openai_train.jsonl`
3. Config exists: `ls config/train_config.yaml`
4. Sufficient memory: `free -h` (need 70GB+ available)

## Container Setup

All GPU steps run inside the DGX Toolbox Unsloth Studio container. The project is bind-mounted via `EXTRA_MOUNTS`:

```bash
# Set the extra mount so unsloth-studio.sh binds our project
export EXTRA_MOUNTS="$(pwd):/workspace/wp-finetune"

# Launch (or restart) the container via DGX Toolbox
# If container is already running, stop and re-launch to pick up the mount
docker stop unsloth-studio 2>/dev/null; docker rm unsloth-studio 2>/dev/null
~/dgx-toolbox/containers/unsloth-studio.sh
```

Wait for the container to finish setup (~30s for pip installs), then install training deps:

```bash
docker exec unsloth-studio pip install --no-deps "transformers==4.56.2" "trl==0.24.0" "datasets==4.3.0" "bitsandbytes==0.48.0" pyyaml python-dotenv scipy wandb peft hf_transfer 2>&1 | tail -3
```

**Verify container sees the project:**
```bash
docker exec unsloth-studio ls /workspace/wp-finetune/config/train_config.yaml
```

### Helper: DCRUN

All GPU commands use this pattern. Define a helper:

```bash
DCRUN="docker exec -w /workspace/wp-finetune unsloth-studio"
```

Then every step is just `$DCRUN python -m scripts.xxx`.

## Process

### 1. Pre-flight Check

```bash
# Host-side checks
echo "=== Pre-flight ===" &&
python scripts/dgx_toolbox.py &&
test -f data/final_dataset/openai_train.jsonl && echo "Training data: OK" || echo "ERROR: No training data" &&
test -f config/train_config.yaml && echo "Config: OK" || echo "ERROR: No config" &&
free -h | head -2

# Container-side checks
$DCRUN python -c "import unsloth; print('Unsloth OK')"
$DCRUN python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
$DCRUN nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

If any check fails, stop and report the issue.

### 2. Download Model

```bash
$DCRUN python -m scripts.download_model
```

- **Idempotent:** Skips if safetensors shards already exist
- Downloads Qwen3-30B-A3B (~60GB) with resume support
- Downloads to `models/Qwen3-30B-A3B/` (visible on host via bind mount)

**Verify:** `$DCRUN ls models/Qwen3-30B-A3B/*.safetensors | wc -l` (expect 16 or 33 shards)

### 3. Extend Tokenizer

```bash
$DCRUN python -m scripts.prepare_tokenizer
```

- **Idempotent:** Skips if `adapters/tokenizer/` already has special tokens
- Adds `<wp_gen>` and `<wp_judge>` special tokens
- Mean-initializes new embedding rows
- Saves extended tokenizer to `adapters/tokenizer/`
- Runs smoke test

**Verify:**
```bash
$DCRUN python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('adapters/tokenizer')
gen_ids = tok.encode('<wp_gen>', add_special_tokens=False)
judge_ids = tok.encode('<wp_judge>', add_special_tokens=False)
assert len(gen_ids) == 1 and len(judge_ids) == 1
print(f'OK: wp_gen={gen_ids[0]}, wp_judge={judge_ids[0]}')
"
```

### 4. Train Model

```bash
$DCRUN python -m scripts.train_model
```

- **Idempotent:** Skips if adapter_config.json exists (use `--resume` for partial)
- **Memory pre-check:** Blocks if <70GB available, lists top consumers
- Loads model via Unsloth FastLanguageModel (BF16, no QLoRA)
- Applies LoRA with `modules_to_save=["embed_tokens", "lm_head"]`
- Sets `output_router_logits=True` for MoE monitoring
- Trains on `data/final_dataset/openai_train.jsonl` (4,766 examples)
- Logs to W&B
- Saves adapter to `adapters/qwen3-wp/`

**Expected duration:** ~6-12 hours

**Dry run first (recommended):**
```bash
$DCRUN python -m scripts.train_model --dry-run
```

**Resume from checkpoint:**
```bash
$DCRUN python -m scripts.train_model --resume
```

**Monitor:**
- W&B dashboard: loss curves, router_aux_loss
- `docker exec unsloth-studio nvidia-smi` for GPU memory

### 5. Merge Adapter

```bash
$DCRUN python -m scripts.merge_adapter
```

- **Idempotent:** Skips if merged model exists with verified special tokens
- Loads base model + LoRA adapter
- Calls `merge_and_unload()` (fix confirmed in unsloth-zoo 2026.3.5)
- Saves merged model to `models/Qwen3-30B-A3B-merged/`
- Verification roundtrip: reload, check `<wp_gen>` and `<wp_judge>` are single-token IDs
- If verification fails: prints vLLM `--lora-modules` fallback

### 6. Post-Training Summary

```bash
echo "=== Training Complete ===" &&
ls -lh adapters/qwen3-wp/adapter_config.json &&
ls models/Qwen3-30B-A3B-merged/*.safetensors 2>/dev/null | wc -l &&
echo "Next: /gsd:execute-phase 4 (evaluation)"
```

## Error Recovery

| Error | Fix |
|-------|-----|
| "Unsloth cannot find any torch accelerator" | Must run inside container: `$DCRUN python -m scripts.xxx` |
| Download interrupted | Re-run `$DCRUN python -m scripts.download_model` (resumes) |
| datasets version mismatch | `docker exec unsloth-studio pip install "datasets==4.3.0"` |
| huggingface-hub version mismatch | `docker exec unsloth-studio pip install "huggingface-hub==0.34.1"` |
| hf_transfer missing | `docker exec unsloth-studio pip install hf_transfer` |
| bitsandbytes missing | `docker exec unsloth-studio pip install "bitsandbytes==0.48.0"` |
| OOM during training | Reduce `max_seq_length` in config/train_config.yaml (4096 → 2048) |
| Training divergence | Reduce `learning_rate` (2e-4 → 1e-4) in config |
| Merge verification fails | Use adapter: `vllm serve models/Qwen3-30B-A3B --lora-modules qwen3-wp=adapters/qwen3-wp` |
| Container stopped | `EXTRA_MOUNTS="$(pwd):/workspace/wp-finetune" ~/dgx-toolbox/containers/unsloth-studio.sh` |

## DGX Toolbox Integration

This skill uses the DGX Toolbox `EXTRA_MOUNTS` feature (added in `build_extra_mounts()` in lib.sh) to bind-mount the project into the Unsloth Studio container. The mount spec is:

```
EXTRA_MOUNTS="/path/to/wp-finetune:/workspace/wp-finetune"
```

This is set automatically by the skill. The `config/dgx_toolbox.yaml` file points to the toolbox location (override with `DGX_TOOLBOX_PATH` env var).

## Key Constraints

- All GPU steps run inside `unsloth-studio` container (never on host)
- `load_in_4bit=False` — QLoRA is off-limits for MoE models
- `output_router_logits=True` — required for MoE load balancing loss
- `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings must train
- Config from `config/train_config.yaml` — nothing hardcoded
- Scripts use `scripts/dgx_toolbox.py` resolver — never hardcode DGX paths
- Adapter saved separately before merge (defense-in-depth)
- Pinned versions: transformers==4.56.2, trl==0.24.0, datasets==4.3.0, bitsandbytes==0.48.0
