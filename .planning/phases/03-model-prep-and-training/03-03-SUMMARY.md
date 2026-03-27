---
phase: 03-model-prep-and-training
plan: "03"
subsystem: training
tags: [training, lora, unsloth, merge, wandb, moe, qwen3]
dependency_graph:
  requires: [03-01, 03-02]
  provides: [scripts/train_model.py, scripts/merge_adapter.py]
  affects: [adapters/qwen3-wp/, models/Qwen3-30B-A3B-merged/]
tech_stack:
  added: [unsloth, trl, peft, wandb]
  patterns:
    - "FastLanguageModel.from_pretrained with load_in_4bit=False (LOCKED for MoE)"
    - "output_router_logits=True in model_kwargs AND model.config (belt-and-suspenders)"
    - "modules_to_save=['embed_tokens', 'lm_head'] for special token training"
    - "SFTTrainer + SFTConfig with report_to=wandb"
    - "PeftModel.from_pretrained + merge_and_unload with verification roundtrip"
    - "vLLM --lora-modules fallback if merge verification fails"
key_files:
  created:
    - scripts/train_model.py
    - scripts/merge_adapter.py
  modified: []
decisions:
  - "output_router_logits=True set both in model_kwargs and model.config — Unsloth version inconsistency protection"
  - "Merge script falls back to vLLM --lora-modules on special-token assertion failure — adapter always stays safe"
  - "Tokenizer loaded from adapters/tokenizer (extended), not base model dir — special tokens must be present at training time"
  - "--resume with no arg auto-detects latest checkpoint in output_dir"
metrics:
  duration_minutes: 2
  completed_date: "2026-03-28"
  tasks_completed: 1
  tasks_total: 2
  files_created: 2
  files_modified: 0
  tests_passing: 75
---

# Phase 3 Plan 03: Training Scripts Summary

**One-liner:** Unsloth LoRA SFT training script with MoE-safe constraints and defense-in-depth merge with special-token verification roundtrip.

## What Was Built

### scripts/train_model.py

Full training pipeline for Qwen3-30B-A3B on DGX Spark:

- **Model loading:** `FastLanguageModel.from_pretrained` with `load_in_4bit=False` (LOCKED — QLoRA destabilizes MoE routing) and `output_router_logits=True` in both `model_kwargs` and `model.config`
- **Tokenizer:** Loads extended tokenizer from `adapters/tokenizer/` (created by prepare_tokenizer.py), asserts `<wp_gen>` and `<wp_judge>` encode as single tokens
- **LoRA:** `FastLanguageModel.get_peft_model` with r=32, alpha=64, dropout=0.05, `modules_to_save=["embed_tokens", "lm_head"]` from config (LOCKED)
- **Dataset:** `load_dataset("json")` from `data/final_dataset/openai_train.jsonl` and `openai_val.jsonl`
- **Trainer:** `SFTTrainer` + `SFTConfig` with `report_to="wandb"` (W&B project: wp-qwen3-moe)
- **CLI:** `--resume [CHECKPOINT_DIR]` (auto-detects latest if no path given), `--dry-run` (print summary, skip training)
- **ALL config from** `config/train_config.yaml` — no hardcoded hyperparameters

### scripts/merge_adapter.py

Defense-in-depth post-training merge:

1. Load base model + adapter via `PeftModel.from_pretrained`
2. `merge_and_unload()` → save merged model + extended tokenizer
3. Reload and verify `<wp_gen>` and `<wp_judge>` are still single tokens (`assert len(ids) == 1`)
4. On failure: print vLLM `--lora-modules qwen3-wp=<adapter_dir>` fallback command and `sys.exit(1)`
- **CLI:** `--adapter-dir` (default: training.output_dir), `--output-dir` (default: model.local_dir + "-merged")

## Deviations from Plan

None - plan executed exactly as written.

## Test Results

75 tests passing (previously 74 — `TestRouterLogitsEnabled::test_router_logits_string_present` now active since `scripts/train_model.py` was created).

## Key Links

| From | To | Via |
|------|----|-----|
| scripts/train_model.py | config/train_config.yaml | yaml.safe_load |
| scripts/train_model.py | data/final_dataset/openai_train.jsonl | load_dataset |
| scripts/train_model.py | adapters/qwen3-wp/ | save_pretrained |
| scripts/merge_adapter.py | adapters/qwen3-wp/ | PeftModel.from_pretrained |
| scripts/merge_adapter.py | models/Qwen3-30B-A3B-merged/ | merge_and_unload |

## Status

Task 1 committed (98e7596). Paused at Task 2 (checkpoint:human-verify) — awaiting human review before DGX execution.

## Self-Check: PASSED

- scripts/train_model.py: EXISTS
- scripts/merge_adapter.py: EXISTS
- commit 98e7596: EXISTS
- 75 tests passing: CONFIRMED
