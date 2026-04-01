# External Integrations

**Analysis Date:** 2026-03-31

## APIs & External Services

**Anthropic Claude API:**
- Purpose: Code quality assessment, synthetic generation, chain-of-thought reasoning in data pipeline
- SDK/Client: `anthropic` Python package
- Auth: `ANTHROPIC_API_KEY` env var (loaded via `python-dotenv` from `.env`)
- Models: `claude-sonnet-4-6` (judging, generation), `claude-opus-4-6` (CoT reasoning)
- Files using it:
  - `scripts/utils.py` - `call_with_backoff()`, batch API helpers (`submit_batch`, `poll_batch`, `parse_batch_results`)
  - `scripts/phase1_judge.py` - 9-dimension code quality assessment
  - `scripts/phase2_generate.py` - Synthetic code generation to fill taxonomy gaps
  - `scripts/phase2_judge.py` - Quality judgment of synthetic examples
  - `scripts/phase2_judge_dataset.py` - Rubric scoring for judge training data
  - `scripts/phase3_cot.py` - Instruction synthesis and chain-of-thought reasoning
- Rate limiting: Exponential backoff with jitter in `scripts/utils.py` (`call_with_backoff()`)
- Batch API: Used for large batches (>50 items) via `scripts/utils.py` (`batch_or_direct()`)

**vLLM (Local Inference Server):**
- Purpose: Serve fine-tuned model for evaluation
- Client: `openai` Python package (OpenAI-compatible API)
- Endpoint: `http://localhost:8020/v1` (resolved via `dgx.vllm_endpoint()`)
- Auth: `api_key="none"` (local server, no auth)
- Files using it:
  - `eval/eval_gen.py` - Generate PHP code for rubric evaluation
  - `eval/eval_judge.py` - Get judge scores for Spearman correlation
- Model name in requests: `"openai/qwen3-wp"`

**LiteLLM (Optional Proxy):**
- Purpose: Unified API for cross-model evaluation
- Endpoint: `http://localhost:4000/v1` (resolved via `dgx.litellm_endpoint()`)
- Configured in `config/dgx_toolbox.yaml` but not directly used in current evaluation scripts

**HuggingFace Hub:**
- Purpose: Model download
- SDK: `huggingface_hub.snapshot_download()`
- Auth: None required (public model)
- File: `scripts/download_model.py`
- Model: `Qwen/Qwen3-30B-A3B` (~60GB, 16 safetensors shards)
- Resume support: `resume_download=True`

**GitHub (Repository Hosting):**
- Purpose: Clone WordPress plugins, themes, and WordPress Core for training data extraction
- Method: Git CLI shallow clone (`--depth=1`) via subprocess
- Auth: Public HTTPS URLs, no authentication
- Config: Repository URLs in `config/repos.yaml` (core + plugins with quality tiers)
- File: `scripts/phase1_clone.py`

## Data Storage

**Databases:**
- MLflow SQLite: `mlruns.db` at project root
  - Connection: `sqlite:///mlruns.db` (set via `mlflow.set_tracking_uri()`)
  - Purpose: Training experiment tracking, metrics logging
  - Experiment: "wp-qwen3-moe"
  - File: `scripts/train_model.py` line 318

**File Storage:**
- Local filesystem only, structured under `data/` prefix:
  - `data/phase1_extraction/repos/` - Cloned source repositories
  - `data/phase1_extraction/output/extracted/` - Extracted functions (JSON per repo)
  - `data/phase1_extraction/output/passed/` - Passing functions (JSON per repo)
  - `data/phase1_extraction/output/failed/` - Failing functions (JSON per repo)
  - `data/phase2_synthetic/output/generated/` - Synthetic examples (JSON)
  - `data/phase2_synthetic/output/judged/` - Judged synthetic examples (JSON)
  - `data/phase2_synthetic/output/mutated/` - Automated mutation contrastive pairs (JSON)
  - `data/phase2_synthetic/output/judge_training/` - Rubric-scored judge training data (JSON)
  - `data/phase3_cot/output/` - CoT processing output
  - `data/final_dataset/` - Final training dataset (OpenAI JSONL, Alpaca JSON, raw JSONL)
  - `data/checkpoints/` - Pipeline checkpoint state (atomic JSON writes)

**Model Storage:**
- `models/Qwen3-30B-A3B/` - Base model (16 safetensors shards, ~60GB)
- `models/qwen3-30b-wp-{ratio}-merged/` - Merged adapter models (13 shards)
- `adapters/tokenizer/` - Extended tokenizer with `<wp_gen>` and `<wp_judge>` tokens
- `adapters/qwen3-wp/` - LoRA adapter checkpoints (default output)
- `adapters/qwen3-30b-wp-{ratio}/` - Ratio-specific adapter checkpoints

**Caching:**
- `unsloth_compiled_cache/` - Unsloth compiled trainer modules (auto-generated)
- `~/.cache/huggingface` - HuggingFace model cache (shared via DGX Toolbox `shared_dirs`)
- Pipeline checkpoints in `data/checkpoints/` for resumable processing

## DGX Toolbox Container Orchestration

**Architecture:**
- `scripts/dgx_toolbox.py` - Python execution engine (singleton via `get_toolbox()`)
- `config/dgx_toolbox.yaml` - Full container configuration
- Pattern: Skill (intent) -> dgx_toolbox.py (resolve + validate + execute) -> Docker commands

**Containers (from `config/dgx_toolbox.yaml`):**

| Component Key | Container Name | Purpose | Port |
|---|---|---|---|
| `unsloth_studio` | `unsloth-headless` | Model download, tokenizer extension, LoRA training, adapter merge | 8000 |
| `eval_toolbox` | `eval-toolbox` | Evaluation suite (PHPCS, Spearman, wp-bench) | - |
| `vllm` | `vllm` | Model serving for eval and inference | 8020 |

**Additional Services (components, not always running):**
- `litellm` - Unified API proxy (port 4000)
- `open_webui` - Web UI for chat (port 12000)
- `ollama_setup` - Ollama remote setup (port 11434)
- `label_studio` - Human annotation (port 8081)
- `argilla` - Data annotation (port 6900)
- `triton` - TensorRT-LLM inference (ports 8010/8011)

**Validation Engine** (`dgx_toolbox.py` `validate()` method):
- `"toolbox"` - dgx-toolbox directory exists
- `"training_data"` - Training JSONL file exists
- `"config"` - Config YAML exists
- `"memory:N"` - At least N GB available RAM
- `"container:name"` - Container running (with restart loop detection)
- `"mounted:name"` - Project bind-mounted in container
- `"gpu"` - GPU accessible (via nvidia-smi in container)
- `"deps:name"` - Pinned deps importable in container

**Container Lifecycle** (`ensure_ready()` method):
1. Check container running (detect restart loops)
2. Verify project mounted at workdir
3. Install pinned dependencies if missing
4. Final validation of all checks

**Execution** (`execute()` method):
- Runs commands inside containers via `docker exec`
- Supports idempotency checks (skip if output exists)
- Captures stdout/stderr with timing
- Maintains execution log for telemetry

## Authentication & Identity

**Auth Providers:**
- Anthropic API Key (`ANTHROPIC_API_KEY` env var) - Data pipeline only
- No auth for local vLLM/LiteLLM endpoints
- No auth for public GitHub repos

**Secrets Location:**
- `.env` file at project root (gitignored)
- `.env.example` documents required variables

## Monitoring & Observability

**MLflow Experiment Tracking:**
- Location: `mlruns.db` (SQLite) and `mlruns/` directory
- Experiment: "wp-qwen3-moe"
- Metrics: Training loss, eval loss (via SFTTrainer `report_to="mlflow"`)
- File: `scripts/train_model.py` line 318-319

**Telemetry Agent System:**
- Architecture: Background Claude Code agents write Markdown reports to `telemetry/{stage}/{timestamp}/`
- 6 observer agents for training: gpu-metrics, thermal-throttling, training-metrics, disk-io, checkpoint-integrity, container-monitor
- Skills documented in `docs/wp-finetune:observe-training.md`
- Consolidated summaries via `docs/wp-finetune:review-telemetry.md`

**DGX Toolbox Status Reports:**
- `dgx.status_report()` returns structured JSON with:
  - Container states (from `docker ps --format json`)
  - Memory stats (from `/proc/meminfo`)
  - Pipeline artifact status (config-driven checks)
  - Execution log (last 20 commands)
  - Service endpoints (vLLM, LiteLLM)

**Memory Watchdog:**
- `MemoryWatchdogCallback` in `scripts/train_model.py` line 261
- Reads `/proc/meminfo` every training step
- Triggers emergency checkpoint save when available RAM < 2GB
- Critical for DGX Spark unified memory architecture

**Error Tracking:**
- No centralized error tracking service
- Console/stderr output from Python scripts
- Structured JSON in intermediate pipeline files

## Evaluation Infrastructure

**Rubric Scoring Engine** (`eval/rubric_scorer.py`):
- 4-tool pipeline: PHPCS (3 standards) -> PHPStan -> Regex -> LLM (TODO)
- 9 scoring dimensions with weights, floor rules, N/A detection
- External tool deps: `phpcs`, `phpstan` (via subprocess)

**Evaluation Suite** (`eval/` module):
- `eval/eval_gen.py` - Generate PHP via vLLM, score with rubric engine
- `eval/eval_judge.py` - Compare model judge scores to rubric ground truth (Spearman)
- `eval/eval_gate.py` - Quality gate: thresholds from `config/train_config.yaml` eval section
- `eval/rubric_definitions.py` - Check registry, dimension weights, floor rules, regex patterns

**wp-bench** (configured in `config/wp-bench.yaml`):
- Docker-based WordPress environment grader
- Tests against `openai/qwen3-wp` model via vLLM endpoint

## CI/CD & Deployment

**Hosting:**
- NVIDIA DGX Spark (local hardware, not cloud)
- Docker containers via DGX Toolbox

**CI Pipeline:**
- No automated CI/CD
- Manual execution via Claude Code skills (`/run-training`, `/observe-training`)

**Deployment Pattern:**
1. Train adapter in `unsloth-headless` container
2. Merge adapter via `scripts/merge_adapter.py` (with verification roundtrip)
3. Serve merged model via vLLM (`~/dgx-toolbox/inference/start-vllm.sh`)
4. Fallback: serve adapter directly with `vllm serve <base> --lora-modules qwen3-wp=<adapter>`

## Environment Configuration

**Required env vars:**
- `ANTHROPIC_API_KEY` - For data pipeline phases (Claude API)

**Optional env vars:**
- `DGX_TOOLBOX_PATH` - Override dgx-toolbox location (default: `~/dgx-toolbox`)

**Secrets location:**
- `.env` file at project root (gitignored)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Pipeline Orchestration

**`scripts/pipeline_orchestrator.py`:**
- Scans all output directories to determine current pipeline state
- Computes percentage-based targets from actual data counts
- Produces structured JSON action plans for Claude Code agents
- Commands: `status`, `plan`, `plan-json`, `status-json`
- Manages 3 phases: Extract/Judge -> Synthetic/Judge Training -> CoT/Export

**Checkpoint System** (`scripts/utils.py`):
- Atomic JSON writes (write to `.tmp`, rename to `.json`)
- Per-phase checkpoints in `data/checkpoints/`
- Tracks completed, failed, batch_job_ids, timestamp

---

*Integration audit: 2026-03-31*
