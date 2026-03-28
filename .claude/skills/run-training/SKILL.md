# Skill: run-training

Run the complete Phase 3 training pipeline on DGX Spark — download model, extend tokenizer, train LoRA adapter, and merge with verification. Single invocation, handles failures and resume.

## Trigger

User says: "run training", "train the model", "start DGX training", "run phase 3", "/run-training"

## Idempotency

Every step is safe to re-run. If the pipeline fails partway through, just re-invoke "run training" — completed steps are automatically skipped:

| Step | Skip condition |
|------|---------------|
| Download model | `models/Qwen3-30B-A3B/*.safetensors` exist |
| Extend tokenizer | `adapters/tokenizer/tokenizer_config.json` exists with special tokens in vocab |
| Train model | `adapters/qwen3-wp/adapter_config.json` exists (use `--resume` to continue partial training) |
| Merge adapter | `models/Qwen3-30B-A3B-merged/config.json` exists and special tokens verify as single-token IDs |

## Prerequisites

Before running, verify:
1. DGX Toolbox Unsloth Studio container is running: `docker ps | grep unsloth-studio`
2. Non-essential containers stopped (free memory for 63GB peak): `free -h`
3. Training data exists: `ls data/final_dataset/openai_train.jsonl`
4. Config exists: `ls config/train_config.yaml`
5. DGX Toolbox resolver works: `python scripts/dgx_toolbox.py`

## Process

### 1. Pre-flight Check

```bash
echo "=== Pre-flight ===" &&
python scripts/preflight.py &&
python scripts/dgx_toolbox.py &&
test -f data/final_dataset/openai_train.jsonl && echo "Training data: OK" || echo "ERROR: No training data" &&
test -f config/train_config.yaml && echo "Config: OK" || echo "ERROR: No config" &&
free -h | head -2
```

If any check fails, stop and report the issue.

### 2. Download Model

```bash
python -m scripts.download_model
```

This downloads Qwen3-30B-A3B (~60GB) from HuggingFace with resume support.
- **Idempotent:** Checks for existing safetensor shards — skips entirely if already downloaded
- Downloads to `models/Qwen3-30B-A3B/`
- Uses HuggingFace `snapshot_download` with resume capability

**Verify:** `ls models/Qwen3-30B-A3B/*.safetensors | wc -l` should show 16 shards.

If the download fails (network issue, disk space):
- Re-run the same command — it resumes from where it left off
- Check disk space: `df -h .`

### 3. Extend Tokenizer

```bash
python -m scripts.prepare_tokenizer
```

This extends the tokenizer with `<wp_gen>` and `<wp_judge>` special tokens:
- **Idempotent:** Checks if `adapters/tokenizer/` already has special tokens in vocab — skips if so
- Loads the base model and tokenizer
- Adds special tokens via `add_special_tokens`
- Resizes model embeddings
- Initializes new token embeddings to the mean of existing embeddings (not random)
- Saves extended tokenizer to `adapters/tokenizer/`
- Runs smoke test: verifies both tokens encode as single-token IDs

**Verify:**
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('adapters/tokenizer')
gen_ids = tok.encode('<wp_gen>', add_special_tokens=False)
judge_ids = tok.encode('<wp_judge>', add_special_tokens=False)
assert len(gen_ids) == 1, f'wp_gen: {gen_ids}'
assert len(judge_ids) == 1, f'wp_judge: {judge_ids}'
print(f'OK: wp_gen={gen_ids[0]}, wp_judge={judge_ids[0]}')
"
```

### 4. Train Model

```bash
python -m scripts.train_model
```

This runs Unsloth LoRA SFT on DGX Spark:
- **Idempotent:** Checks if `adapters/qwen3-wp/adapter_config.json` exists — skips if trained (use `--resume` to continue partial training)
- Loads model via `FastLanguageModel.from_pretrained` (BF16, no QLoRA)
- Applies LoRA with `modules_to_save=["embed_tokens", "lm_head"]`
- Sets `output_router_logits=True` for MoE load balancing monitoring
- Trains on `data/final_dataset/openai_train.jsonl` (4,766 examples)
- Logs to W&B (project: wp-qwen3-moe)
- Saves adapter checkpoint to `adapters/qwen3-wp/`

**Expected duration:** ~6-12 hours depending on GPU utilization.

**Monitoring during training:**
- W&B dashboard: check loss curves, router_aux_loss, learning rate schedule
- `nvidia-smi`: check GPU memory (~63GB peak expected)
- Training logs: `tail -f adapters/qwen3-wp/training.log` (if configured)

**If training fails (OOM, divergence):**
- Resume from last checkpoint: `python -m scripts.train_model --resume`
- If OOM: reduce `per_device_train_batch_size` to 1 in config/train_config.yaml (already 1)
- If OOM persists: reduce `max_seq_length` from 4096 to 2048

**Verify:**
```bash
ls adapters/qwen3-wp/adapter_config.json && echo "Adapter saved OK"
```

### 5. Merge Adapter

```bash
python -m scripts.merge_adapter
```

This merges the LoRA adapter into the base model with a verification roundtrip:
- **Idempotent:** Checks if merged model exists with verified special tokens — skips if already merged successfully
1. Loads base model + LoRA adapter
2. Calls `merge_and_unload()`
3. Saves merged model to `models/Qwen3-30B-A3B-merged/`
4. Reloads and verifies `<wp_gen>` and `<wp_judge>` are still single-token IDs
5. If verification passes: prints "MERGE VERIFICATION PASSED"
6. If verification fails: prints vLLM `--lora-modules` fallback command

**Verify:**
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen3-30B-A3B-merged')
gen_ids = tok.encode('<wp_gen>', add_special_tokens=False)
judge_ids = tok.encode('<wp_judge>', add_special_tokens=False)
assert len(gen_ids) == 1 and len(judge_ids) == 1
print('Merged model special tokens: OK')
"
```

### 6. Post-Training Summary

After all steps complete, report:

```bash
echo "=== Training Complete ===" &&
echo "Model: models/Qwen3-30B-A3B" &&
echo "Adapter: adapters/qwen3-wp/" &&
echo "Merged: models/Qwen3-30B-A3B-merged/" &&
echo "Tokenizer: adapters/tokenizer/" &&
echo "Config: config/train_config.yaml" &&
ls -lh adapters/qwen3-wp/adapter_config.json &&
ls -lh models/Qwen3-30B-A3B-merged/*.safetensors 2>/dev/null | wc -l &&
echo "Tests: $(python3 -m pytest tests/ -q 2>&1 | tail -1)" &&
echo "" &&
echo "Next: /gsd:execute-phase 4 (evaluation)"
```

## Error Recovery

| Error | Fix |
|-------|-----|
| Download interrupted | Re-run `python -m scripts.download_model` (resumes) |
| Tokenizer fails | Check model is fully downloaded: `ls models/Qwen3-30B-A3B/*.safetensors \| wc -l` |
| OOM during training | Reduce `max_seq_length` in config/train_config.yaml (4096 → 2048) |
| Training divergence | Check W&B for loss spike, reduce `learning_rate` (2e-4 → 1e-4) |
| Resume training | `python -m scripts.train_model --resume` |
| Merge verification fails | Use adapter directly: `vllm serve models/Qwen3-30B-A3B --lora-modules qwen3-wp=adapters/qwen3-wp` |
| Docker memory pressure | Stop non-essential containers: `docker stop <name>` |

## Key Constraints

- `load_in_4bit=False` — QLoRA is off-limits for MoE models
- `output_router_logits=True` — required for MoE load balancing loss in W&B
- `modules_to_save=["embed_tokens", "lm_head"]` — special token embeddings must train
- All config from `config/train_config.yaml` — nothing hardcoded
- All scripts use `scripts/dgx_toolbox.py` resolver — never hardcode DGX paths
- Adapter saved separately before merge attempt (defense-in-depth)
