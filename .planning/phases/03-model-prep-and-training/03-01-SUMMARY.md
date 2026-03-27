---
phase: 03-model-prep-and-training
plan: 01
subsystem: training
tags: [qwen3, huggingface, tokenizer, lora, moe, bfloat16, safetensors]

# Dependency graph
requires:
  - phase: 02-dataset-production
    provides: openai_train.jsonl / openai_val.jsonl / openai_test.jsonl at data/final_dataset/

provides:
  - scripts/download_model.py — snapshot_download with resume support for Qwen3-30B-A3B
  - scripts/prepare_tokenizer.py — full pipeline (download, extend, mean-init, smoke test)
  - config/train_config.yaml — all Phase 3 hyperparameters externalized
  - tests/test_prepare_tokenizer.py — 7 wave-0 tokenizer tests (mocked, no GPU)
  - tests/test_train_model.py — 11 wave-0 training config tests (mocked, no GPU)

affects:
  - 03-02 (training script — reads train_config.yaml, uses adapters/tokenizer)
  - 03-03 (evaluation — uses adapters/tokenizer for special token IDs)

# Tech tracking
tech-stack:
  added:
    - huggingface_hub (snapshot_download with resume support)
    - transformers (AutoModelForCausalLM, AutoTokenizer)
    - torch (bfloat16, no-grad mean-init)
  patterns:
    - load_in_4bit=False LOCKED for MoE models (no QLoRA for Qwen3 MoE architecture)
    - Mean embedding initialization (not random) for new special tokens
    - All config read from train_config.yaml — no hardcoded hyperparameters in scripts
    - from scripts.dgx_toolbox import get_toolbox at top of all training scripts

key-files:
  created:
    - config/train_config.yaml
    - scripts/download_model.py
    - scripts/prepare_tokenizer.py
    - tests/test_prepare_tokenizer.py
    - tests/test_train_model.py
  modified: []

key-decisions:
  - "load_in_4bit=False LOCKED in prepare_tokenizer.py — no QLoRA for MoE (Qwen3-30B-A3B is MoE, QLoRA destabilizes routing)"
  - "Mean embedding init: new token rows set to mean of existing embed_tokens rows (not random) for stable early training"
  - "Model saved back to local_dir after embedding resize — ensures saved model and tokenizer vocab sizes are consistent"
  - "Smoke test generates 50 tokens from <wp_gen> prompt and asserts >10 new tokens produced"
  - "test_router_logits_enabled uses static analysis (grep source file) — tests future train_model.py before it exists via pytest.skip"

patterns-established:
  - "All scripts: from scripts.dgx_toolbox import get_toolbox at module top"
  - "All paths resolved via PROJECT_ROOT = Path(__file__).resolve().parent.parent"
  - "Config loaded via yaml.safe_load(open(PROJECT_ROOT / 'config' / 'train_config.yaml'))"
  - "Wave 0 TDD: test files written before implementation, all pass with mocks/fixtures"

requirements-completed: [MODL-01, MODL-02, MODL-03, MODL-04]

# Metrics
duration: 12min
completed: 2026-03-28
---

# Phase 3 Plan 01: Model Download and Tokenizer Preparation Summary

**Qwen3-30B-A3B download script (resume-capable), tokenizer extended with <wp_gen>/<wp_judge> via mean-init embeddings, all hyperparameters in train_config.yaml, 18 wave-0 tests passing**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-28T00:00:00Z
- **Completed:** 2026-03-28T00:12:00Z
- **Tasks:** 2 of 2
- **Files modified:** 5 created, 0 modified

## Accomplishments

- config/train_config.yaml created with model, tokenizer, lora, training, data, and eval sections fully externalized
- scripts/download_model.py: snapshot_download with resume_download=True, ignores .msgpack/.h5, reads from config, skips if shards present
- scripts/prepare_tokenizer.py: full pipeline — download, AutoModelForCausalLM (bfloat16, no QLoRA), extend tokenizer, mean-init embeddings, save, smoke test, CLI flags
- 18 wave-0 tests written with mocks/fixtures — all pass without GPU or model download (17 pass, 1 skip pending train_model.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create train_config.yaml, download script, and test scaffolds** - `e3ca0bf` (feat)
2. **Task 2: Create prepare_tokenizer.py** - `da0460b` (feat)

## Files Created/Modified

- `config/train_config.yaml` — All Phase 3 hyperparameters: model name/path, tokenizer save dir, LoRA config (r=32, alpha=64, target_modules, modules_to_save), training schedule, data paths, eval thresholds
- `scripts/download_model.py` — HuggingFace snapshot_download with resume support; count_safetensors() helper for idempotent skip logic; reads model.name and model.local_dir from config
- `scripts/prepare_tokenizer.py` — Full prep pipeline: download (optional), load model, extend tokenizer, mean-init embeddings, save tokenizer + model, smoke test; --skip-download and --smoke-only CLI flags
- `tests/test_prepare_tokenizer.py` — 7 tests: special tokens added, no duplicates, mean-init correctness, mean not zero/random, single-token encoding for both <wp_gen> and <wp_judge>
- `tests/test_train_model.py` — 11 tests: count_safetensors() logic, load_config(), lora params (r, bf16, lr_scheduler), modules_to_save, dataset schema (messages, role, content, valid roles), static analysis for output_router_logits

## Decisions Made

- **load_in_4bit=False LOCKED**: Qwen3-30B-A3B is a MoE architecture; QLoRA (4-bit) destabilizes expert routing. This constraint is coded and commented.
- **Mean embedding init**: New rows set to mean of all prior embed_tokens rows. Prevents new tokens starting at noise, stabilizes early fine-tuning gradient steps.
- **Model re-saved after resize**: After embedding resize, model saved back to local_dir so safetensors and tokenizer vocab sizes remain consistent for loading.
- **Static analysis test for output_router_logits**: test_router_logits_enabled uses pytest.skip if train_model.py not present — pre-writes the constraint test before Plan 03-02 implements the script.
- **Smoke test threshold**: >10 new tokens generated from <wp_gen> prompt — conservative enough to not require specific model quality, just that generation runs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The pre-existing test_eval_gen.py, test_eval_judge.py, and test_eval_gate.py fail with ModuleNotFoundError (eval module not yet created — Phase 4). These failures predate this plan and are unrelated to Phase 3 work.

## User Setup Required

None - no external service configuration required. Model download (scripts/download_model.py) must be run manually on DGX Spark where 60 GB disk space and HuggingFace credentials are available.

## Next Phase Readiness

- config/train_config.yaml ready for Plan 03-02 (training script) and 03-03 (eval thresholds)
- scripts/prepare_tokenizer.py ready to run on DGX Spark with --skip-download if model already cached
- Wave-0 tests provide regression protection for tokenizer changes during training script development
- Blocker: model download requires DGX Spark execution (60 GB, not runnable in dev environment)

## Self-Check: PASSED

- FOUND: config/train_config.yaml
- FOUND: scripts/download_model.py
- FOUND: scripts/prepare_tokenizer.py
- FOUND: tests/test_prepare_tokenizer.py
- FOUND: tests/test_train_model.py
- FOUND: 03-01-SUMMARY.md
- FOUND commit e3ca0bf (Task 1: config + download script + test scaffolds)
- FOUND commit da0460b (Task 2: prepare_tokenizer.py)

---
*Phase: 03-model-prep-and-training*
*Completed: 2026-03-28*
