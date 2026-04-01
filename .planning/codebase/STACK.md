# Technology Stack

**Analysis Date:** 2026-03-31

## Languages

**Primary:**
- Python 3.11+ - All pipeline scripts, training, evaluation, orchestration (uses `dict[str, Any]`, `X | None` syntax)
- PHP - Code extraction (`scripts/php_extract_functions.php`), PHPCS/PHPStan evaluation targets

**Secondary:**
- YAML - All configuration (`config/*.yaml`, 12 config files)
- Jinja2 - Chat templates for tokenizer (`adapters/tokenizer/chat_template.jinja`)
- Bash - DGX Toolbox container scripts (external, referenced via `config/dgx_toolbox.yaml`)
- Markdown - System prompts (`config/judge_system.md`), documentation

## Runtime

**Environment:**
- Python 3.11+ (type hint syntax requires it)
- NVIDIA DGX Spark (unified memory architecture, 128GB system RAM shared between CPU and GPU)
- Docker containers managed by DGX Toolbox (`~/dgx-toolbox/`)
- Container `unsloth-headless` is the primary training environment

**Package Manager:**
- pip (no `requirements.txt` or `pyproject.toml` in repo)
- Dependencies pinned in `config/dgx_toolbox.yaml` under `pinned_versions`
- Lockfile: missing (config-based pinning instead)

## Frameworks

**Core ML (Training):**
- Unsloth (`FastLanguageModel`) - Model loading with MoE support, LoRA application, gradient checkpointing mode "unsloth"
- HuggingFace Transformers 4.56.2 - `AutoTokenizer`, `AutoModelForCausalLM`, `TrainerCallback`
- TRL 0.24.0 - `SFTTrainer`, `SFTConfig` for supervised fine-tuning
- PEFT - `PeftModel` for LoRA adapter loading, merging via `merge_and_unload()`
- PyTorch (bfloat16) - Tensor operations, model inference, gradient computation

**Data:**
- HuggingFace Datasets 4.3.0 - Dataset loading from JSONL files via `load_dataset("json", ...)`
- HuggingFace Hub 0.34.1 - Model download via `snapshot_download()` with resume support

**Evaluation:**
- PHPCS (3 standards: WordPress, WordPressVIPMinimum, Security) - Static analysis via subprocess in `eval/rubric_scorer.py`
- PHPStan (level 5) - Type analysis via subprocess in `eval/rubric_scorer.py`
- scipy.stats.spearmanr - Spearman correlation for judge evaluation in `eval/eval_judge.py`
- OpenAI Python client - Talks to local vLLM endpoint (not OpenAI servers)

**Experiment Tracking:**
- MLflow - Local SQLite store at `mlruns.db`, experiment name "wp-qwen3-moe", configured in `scripts/train_model.py`

**Testing:**
- pytest (inferred from `tests/test_*.py` structure)

## Key Dependencies

**Critical (pinned in `config/dgx_toolbox.yaml`):**
- `transformers==4.56.2` - Core model loading and tokenization
- `trl==0.24.0` - SFTTrainer for fine-tuning
- `datasets==4.3.0` - Training data loading
- `bitsandbytes==0.48.0` - Quantization support (load_in_4bit is LOCKED to False for MoE)
- `huggingface_hub==0.34.1` - Model download with resume

**Infrastructure (installed as extra_deps in containers):**
- `unsloth` - FastLanguageModel wrapper, gradient checkpointing; NOT version-pinned (container-provided)
- `peft` - LoRA adapter management and merge
- `mlflow` - Experiment tracking to local SQLite
- `pyyaml` - Config file parsing (used in every script)
- `python-dotenv` - `.env` file loading for API keys
- `scipy` - Spearman correlation in eval
- `hf_transfer` - Faster model downloads

**Data Pipeline Only:**
- `anthropic` - Claude API client for synthetic generation, judging (`scripts/utils.py`, `scripts/phase2_generate.py`)
- `openai` - OpenAI-compatible client for vLLM inference endpoint (`eval/eval_gen.py`, `eval/eval_judge.py`)

## Configuration

**Environment:**
- `.env` file present - contains `ANTHROPIC_API_KEY` (for data pipeline phases, not training)
- `.env.example` documents required variables
- `DGX_TOOLBOX_PATH` env var - overrides toolbox location (default: `~/dgx-toolbox`)

**Training configs (ratio variants for ablation studies):**
- `config/train_config.yaml` - Default training configuration (active)
- `config/train_config_30_70.yaml` through `config/train_config_70_30.yaml` - Gen/judge ratio variants
- All share identical LoRA and model settings; differ only in data paths, output_dir, and dataloader settings

**Pipeline configs:**
- `config/dgx_toolbox.yaml` - Container mapping, pinned versions, ports, validation paths, required imports
- `config/repos.yaml` - WordPress repository list (core + plugins with quality tiers)
- `config/taxonomy.yaml` - Training tag taxonomy and coverage minimums
- `config/synthetic_prompts.yaml` - Prompt templates for synthetic generation
- `config/judge_system.md` - 9-dimension judging rubric system prompt
- `config/wp-bench.yaml` - wp-bench evaluation configuration (vLLM endpoint, grader config)

## Model Specification

**Base Model:**
- Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B` on HuggingFace)
- Native MoE architecture: ~30B total params, ~3B active per forward pass
- 128 experts, top-8 routing
- Stored at `models/Qwen3-30B-A3B/` (16 safetensors shards, ~60GB)
- BF16 precision (load_in_4bit=False is LOCKED -- no QLoRA for MoE)
- MoE load balancing: `output_router_logits=True` set on model.config

**LoRA Configuration (from `config/train_config.yaml`):**
- r=32, alpha=64, dropout=0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_up_proj`, `down_proj`
- modules_to_save: `embed_tokens`, `lm_head` (LOCKED for custom token embeddings)
- Gradient checkpointing: `use_gradient_checkpointing="unsloth"` mode
- bias: "none"

**Custom Tokens:**
- `<wp_gen>` - Routes to code generation mode
- `<wp_judge>` - Routes to structured critique/scoring mode
- Mean-initialized from existing embedding average in `scripts/prepare_tokenizer.py`

**Training Hyperparameters:**
- SFT with cosine LR schedule, warmup_ratio=0.05
- Effective batch size: 16 (per_device_batch=4 x gradient_accumulation=4)
- Max sequence length: 4096
- 2 epochs, learning_rate=2e-4
- BF16 mixed precision (fp16=False explicitly)
- Checkpoints saved every 400 steps, eval every 200 steps
- Dataloader: 3 workers, persistent, prefetch_factor=3

**Adapter Outputs:**
- Training saves to `adapters/qwen3-wp/` (or ratio-specific dirs like `adapters/qwen3-30b-wp-40_60/`)
- Checkpoints at `adapters/qwen3-wp/checkpoint-{N}/` (every save_steps)
- Merged models at `models/qwen3-30b-wp-{ratio}-merged/`

## Memory Management

**MemoryWatchdogCallback** (`scripts/train_model.py` line 261):
- Custom `TrainerCallback` that monitors `/proc/meminfo` every training step
- Threshold: 2048 MB available RAM (OOM_WATCHDOG_THRESHOLD_MB)
- On trigger: sets `control.should_save=True` and `control.should_training_stop=True`
- Saves emergency checkpoint before OOM killer strikes
- Critical for DGX Spark unified memory where GPU/CPU compete for same pool

**Pre-training Memory Check** (`scripts/train_model.py` line 59):
- Reads `/proc/meminfo` (fallback: psutil) before loading 63GB model
- Minimum 70GB free required (MIN_FREE_MEMORY_GB)
- Shows top memory consumers and Docker containers if insufficient
- Exits with actionable suggestions if check fails

## Telemetry System

**Observer Agents** (documented in `docs/wp-finetune:observe-training.md`):
- 6-agent team spawned as background Claude Code agents
- GPU metrics, thermal/throttling, training metrics, disk I/O, checkpoint integrity, container monitor
- Write to `telemetry/training/{timestamp}/` as Markdown files
- 30-second polling intervals
- Stop via `_stop` sentinel file

**Review** (documented in `docs/wp-finetune:review-telemetry.md`):
- Reads telemetry files and produces consolidated `_summary.md`
- Extracts WARNING/CRITICAL flags, key metrics, timeline

## Platform Requirements

**Development:**
- Python 3.11+
- PHP + PHPCS with WordPress-Extra, WordPressVIPMinimum, Security standards
- PHPStan (optional, for full rubric scoring)
- Docker (for DGX Toolbox container management)
- ~60GB disk for base model, ~2GB per adapter checkpoint

**Training (Production):**
- NVIDIA DGX Spark with unified memory (128GB)
- Docker with NVIDIA runtime
- DGX Toolbox (`~/dgx-toolbox/`) for container orchestration
- Minimum 70GB free system RAM before model loading
- Container: `unsloth-headless` via `~/dgx-toolbox/containers/unsloth-headless-sync.sh`
- Training duration: 6-12 hours per run

**Inference (Production):**
- vLLM server (via DGX Toolbox) on port 8020
- LiteLLM proxy on port 4000 (optional)
- OpenAI-compatible API endpoint
- Supports both merged model and adapter-only serving (`--lora-modules` fallback)

---

*Stack analysis: 2026-03-31*
