---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
# External Integrations

**Analysis Date:** 2026-10-09

## APIs & External Services

**Fine-tuning / RL backend:**
- Tinker (Thinking Machines) - Hosted LoRA SFT, data capture, and RL training.
  - SDK/Client: `tinker`, `tinker_cookbook` (imported in about 90 scripts, e.g. `scripts/rl_train.py`, `scripts/capture_judge_responses_tinker.py`, `scripts/tinker_fs_gate.py`, `scripts/tinker_reasoning_sft_v4.py`)
  - Auth: `TINKER_API_KEY` (checked in `scripts/build_base20_probe_adapter.py`; the probe falls back to a local zero-init run when it is absent)
  - Site: https://thinkingmachines.ai (linked from `README.md`)

**Anthropic / Claude:**
- Claude Code CLI (`claude --print` / `claude -p`) - LLM work for the data pipeline (relabel, scoring, critique generation) and the RL judge fallback.
  - Wrapper: `scripts/claude_agent.py` (`generate()` and `generate_json()` via subprocess). Used by `scripts/rl_judge_dispatch.py` when no `base_url` is given, and by `eval/llm_checks.py` when `LLM_BACKEND` is `claude` (the default).
  - Auth: the subprocess env scrubs `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` (`scripts/claude_agent.py`, `_agent_env`).
  - Billing: the module docstring in `scripts/claude_agent.py` (dated 2026-06-22) states that `claude -p` and the Agent SDK always incur direct Anthropic API spend, not subscription coverage. A ~$90 run is cited. `README.md` ("How it was built") still says LLM steps run on a subscription with no per-token cost. The two statements conflict. The code comment is the newer one.
- Anthropic Python SDK - Test-only import (`tests/test_phase2_judge_dataset.py`, `tests/test_generate_critique_then_fix.py`, `tests/test_generate_deep_judge_cot.py`).
- `.env.example` - `ANTHROPIC_API_KEY` is a documented fallback "for Batch API". No runtime Batch API call was found in `scripts/` or `eval/`.

**Local OpenAI-compatible inference (in-house endpoints):**
- vLLM server - Judge, generator, and capture endpoints on `http://localhost:8020/v1` (`config/dgx_toolbox.yaml` `ports.vllm`, `config/wp-bench.yaml`, `scripts/serve_*_vllm.sh`). Other ports seen: 8000 (`eval/llm_checks.py` default `DEFAULT_VLLM_BASE_URL`), 8001 (docstring examples in `scripts/rl_judge_dispatch.py`).
  - SDK/Client: `openai` Python SDK with `base_url=<local>` and `api_key="none"`. Examples: `scripts/generate_cot_vllm.py`, `scripts/capture_reasoning_responses.py`, `scripts/fidelity_gate_v3.py`, `scripts/verify_moe_merge_ground_truth.py`, `scripts/rl_judge_dispatch.py` (`_get_vllm_client`, one cached client per base URL). Raw HTTP is also used via `urllib.request` in `scripts/drive_v4_routing_profile.py` (`/v1/models`, `/v1/completions`).
  - Auth: none. The local key is the literal `"none"`, with `_VLLM_API_KEY` in `scripts/rl_judge_dispatch.py`.
- llama.cpp `llama-server` - Serves the shipped GGUF at `http://127.0.0.1:8020/v1/chat/completions` (`README.md` Quickstart). OpenAI-compatible, `--jinja`, `--spec-type draft-mtp`.
- Ollama - Named as an alternative GGUF server (`README.md`, `config/dgx_toolbox.yaml` `ollama: 11434`). No code path found.
- LiteLLM proxy - Component `inference/start-litellm.sh` (port 4000) in `config/dgx_toolbox.yaml`. wp-bench calls `litellm.completion()` directly. Routing is configured by `scripts/run_eval_reasoning.py` and `scripts/_wpbench_pth/usercustomize.py` (patches `litellm` to send `extra_body`).
- Open WebUI, Label Studio (port 8081), Argilla (port 6900), Triton (ports 8010/8011) - Listed as DGX Toolbox components in `config/dgx_toolbox.yaml`. No direct client code in this repo.

**Benchmark harness:**
- wp-bench - External WordPress code-quality benchmark run through Docker (`config/wp-bench.yaml` `grader.kind: docker`, `wp_env_dir: ./wp-bench/runtime`, suite `wp-core-v1`). Invoked by `scripts/run_eval_reasoning.py` and `scripts/build_exp4_wpbench.py`, among others. Results go to `output/wp-bench-results.json`.

**Data source services (pipeline inputs, read over the network at build time):**
- GitHub - Source repos cloned from `https://github.com/...` URLs in `config/repos.yaml` (e.g. `https://github.com/WordPress/wordpress-develop.git`, `https://github.com/Yoast/wordpress-seo.git`). About 2,400 github.com references in code and docs. Cloning is documented as "clone 236 WordPress repos" in `README.md`. Entry points include `scripts/csv_to_repos.py` (see `tests/test_csv_to_repos.py`).
- WordPress.org - Plugin and theme metadata and source, plus core (`config/repos.yaml`, `scripts/` plugin fetchers, `deps/wp-finetune-data/wp_plugins_raw.json`, `deps/wp-finetune-data/wp_top1000_plugins_final.csv`). Also `https://core.trac.wordpress.org` and `https://developer.wordpress.org` references in rubric docs.
- Security advisory sources cited in rubric and data notes: Wordfence, Patchstack, pluginvulnerabilities.com (URLs appear in `config/rubric*`, `data/` notes). Reference only; no runtime API client was found.

**Model hosting (HuggingFace):**
- HuggingFace Hub - Download of base models and upload of shipped artifacts.
  - Download: `hf download` (README), `huggingface_hub` 0.34.1 (`scripts/smoke_load_base20.py`), `models/` local cache. `config/dgx_toolbox.yaml` `shared_dirs.huggingface_cache` is `~/.cache/huggingface`.
  - Upload: `hf upload <repo> <file> <remote>` in `scripts/_pub03_upload.sh` and `scripts/_pub4_upload.sh` (sequential per-file). `scripts/pub4_validate_upload.py` checks uploaded files with `requests`.
  - Published repos: `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` (canonical), `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` (superseded, still live).
  - Auth: HuggingFace token via the `hf` CLI login (no token variable name found in code).

**Telemetry and tracking:**
- MLflow - Experiment tracking, imported in `scripts/train_model.py` (`import mlflow` in a guarded block). Listed in `required_imports` and `extra_deps` in `config/dgx_toolbox.yaml`. No tracking URI found in repo config (defaults apply).
- DGX Toolbox telemetry - Local package, not a network service. `telemetry.sampler.GPUSampler` and `telemetry.failure_classifier` are used by `scripts/train_model.py` and `scripts/adaptive_planner.py`. Mounted from `~/dgx-toolbox/telemetry`.

## Data Storage

**Databases:**
- None. No database server, ORM, or DB client is used.
- Data lives in files: JSON, JSONL, CSV, YAML, and `.npy` masks under `data/`, `output/`, `deps/wp-finetune-data/`.

**File Storage:**
- Local filesystem only.
  - Models: `./models/` (git-ignored, `.gitignore`), e.g. `./models/Qwen3.6-35B-A3B` (`config/train_config_v4.yaml`).
  - Adapters: `./adapters/` (git-ignored), e.g. `./adapters/qwen3-wp` and `./adapters/tokenizer`.
  - Outputs and receipts: `output/` (committed, e.g. `output/packaging/MODEL_CARD.md`, `output/bench17/`, `output/pkg-v4/`).
  - Shared host dirs from DGX Toolbox: `~/unsloth-data`, `~/eval/datasets`, `~/eval/models`, `~/eval/runs`, `~/data/exports` (`config/dgx_toolbox.yaml` `shared_dirs`).

**Caching:**
- None as a service. Local caches: `eval/` and `scripts/` judge-score caches (`scripts/rl_judge_dispatch.py` `score_with_cache`), HuggingFace cache under `~/.cache/huggingface`, and `XDG_STATE_HOME` for the DGX Toolbox watcher state (`scripts/dgx_toolbox_watch.py`).

## Authentication & Identity

**Auth Provider:**
- None for end users. The served model has no auth layer; the llama.cpp and vLLM endpoints bind to `127.0.0.1` or `localhost`. The client key is the placeholder `"none"`.
- Service credentials are read from environment variables or a `.env` file (`python-dotenv`):
  - `TINKER_API_KEY` (Tinker)
  - `ANTHROPIC_API_KEY` (documented in `.env.example`; scrubbed from `claude` subprocess env)
  - HuggingFace login (via `hf` CLI)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Rollbar, or similar client).

**Logs:**
- Python `logging` (13 modules) and `print`-based progress output.
- Debug switches: `WP_JUDGE_DEBUG_DUMP` (judge responses), `ADAPTIVE_THERMAL_LOG` (thermal log path), `SIEVE_PROFILE_OUT` / `SIEVE_PROFILE_FLUSH_SECS` (routing profile output).
- Run receipts: `output/` JSON and Markdown receipts. Benchmarks and gates write these as their audit trail (see `PIPELINE.md`).
- MLflow (see above) for training runs.
- GPU telemetry: DGX Toolbox `telemetry` package (`scripts/train_model.py`).

## CI/CD & Deployment

**Hosting:**
- Self-hosted on a single NVIDIA DGX Spark (GB10). Served from vLLM or llama.cpp on port 8020 (`config/dgx_toolbox.yaml`).
- Model distribution: HuggingFace (`iamchum` account), Apache 2.0.
- Source code: GitHub (`https://github.com/dr-robert-li/dgx-toolbox` for the toolbox submodule; project repo is not named in the code).

**CI Pipeline:**
- None detected. No `.github/workflows/`, no `.gitlab-ci.yml`, no `Makefile`. Tests run locally with `pytest tests/`.

## Environment Configuration

**Required env vars (names only):**
- `TINKER_API_KEY` - needed for Tinker-backed scripts (`scripts/build_base20_probe_adapter.py`, `scripts/tinker_*`).
- `LLM_BACKEND` - `claude` (default) or `vllm`. Selects the LLM backend for `eval/llm_checks.py`.
- `LLM_VLLM_BASE_URL` / `VLLM_BASE_URL` / `LLM_VLLM_MODEL` / `VLLM_MODEL` - local vLLM endpoint and served model name (`eval/llm_checks.py`, `scripts/`).
- `EVAL_JUDGE_BASE_URL` - judge endpoint for `eval/eval_judge.py`. Falls back to DGX Toolbox `vllm_endpoint()`.
- `EVAL_GEN_BASE_URL` - generator endpoint for `eval/eval_gen.py`.
- `RUBRIC_USE_LLM_CHECKS` - enables the 41 LLM-assisted rubric checks (`scripts/phase1b_stratified_rejudge.py`).
- `DGX_TOOLBOX_PATH` - overrides `dgx_toolbox_path` in `config/dgx_toolbox.yaml`.
- `ANTHROPIC_API_KEY` - optional, per `.env.example`.

**Secrets location:**
- A `.env` file at the repo root, loaded by `python-dotenv`. The template `.env.example` holds a placeholder for `ANTHROPIC_API_KEY`. The real `.env` file is not read for this document.
- HuggingFace credentials are kept by the `hf` CLI in its own user config.
- `.env` and `.env.*` are excluded by `.gitignore`.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server is part of this repo. Model endpoints are served by external vLLM / llama.cpp processes and only called as clients.

**Outgoing:**
- None. No webhook sender was found.
- Outbound HTTP is limited to: model endpoints on localhost, HuggingFace `hf upload` / `hf download`, `git clone` of source repos listed in `config/repos.yaml`, the Claude CLI (`claude -p`), and the Tinker API.

---

*Integration audit: 2026-10-09*
