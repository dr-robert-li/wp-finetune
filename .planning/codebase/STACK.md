---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
# Technology Stack

**Analysis Date:** 2026-10-09

## Languages

**Primary:**
- Python 3.10+ - All pipeline, training, merge, quantization, eval, and test code (`scripts/` ~337 Python files across repo, `eval/`, `tests/`, `config/`). Minimum version stated in `README.md` ("Requirements").
- Bash - Serving, upload, and run wrappers (`scripts/serve_v4_judge_vllm.sh`, `scripts/serve_30_70_vllm.sh`, `scripts/_pub4_upload.sh`, `scripts/_pub03_upload.sh`, `scripts/eval4_ext_gguf_convert.sh`, 37 `.sh` files total)

**Secondary:**
- PHP - Data-pipeline static checks. PHP CLI with `tokenizer` extension plus PHP_CodeSniffer and WordPress-Coding-Standards are required (`README.md`, `scripts/build_antihack_set.py`, `scripts/reward_pipeline.py`, `eval/rubric_scorer.py`). Training corpus is WordPress PHP code (`tests/fixtures/reward_integration_cases/known_good_php/`, `known_bad_php/`).
- YAML - Configs in `config/*.yaml` (training, rubric, taxonomy, repos, DGX Toolbox) and `recipes/*.yaml` (vLLM serving recipes).
- Markdown - Rubric prompt `config/judge_system.md`, and the project docs (`PIPELINE.md`, `PROJECT.md`, `JOURNAL.md`).
- JSON / JSONL - Datasets, receipts, and manifests (`data/`, `output/`, `config/*.json`).

## Runtime

**Environment:**
- Two execution environments, both on a single NVIDIA DGX Spark (GB10, aarch64):
  - Training/merge/profiling: NGC PyTorch container `nvcr.io/nvidia/pytorch:25.11-py3`, run through the `unsloth-studio` container from DGX Toolbox (`config/requirements-profiling.txt` header, `config/dgx_toolbox.yaml`).
  - Serving/eval: vLLM and llama.cpp (`llama-server`) on the host or in DGX Toolbox containers.
- A `.venv/` and `venv/` are git-ignored (`.gitignore`), but no venv is committed.
- PyTorch is used via the NGC container image. It is not pinned in-repo.

**Package Manager:**
- pip (inside the container). Installs use `pip install --no-deps -r config/requirements-profiling.txt`.
- Lockfile: missing. No `requirements.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, `poetry.lock`, or `uv.lock` at the repo root. Dependencies are documented in `README.md`, `config/requirements-profiling.txt`, and the `pinned_versions` / `extra_deps` blocks in `config/dgx_toolbox.yaml`.
- `deps/dgx-toolbox` is a git submodule (`.gitmodules`, URL `https://github.com/dr-robert-li/dgx-toolbox.git`). The directory is currently empty, so the submodule is not initialized.

## Frameworks

**Core (training and model stack):**
- Hugging Face `transformers` 4.56.2 - Model loading, tokenization, and merge. Pinned in `config/requirements-profiling.txt` and `config/dgx_toolbox.yaml`.
- `peft` 0.18.1 - LoRA adapters. The MoE fused-expert LoRA uses `target_parameters` (`config/train_config_v4.yaml`).
- `trl` 0.26.1 - SFT trainer (`required_imports` in `config/dgx_toolbox.yaml`).
- `unsloth` / `unsloth_zoo` >=2026.3.5,<2026.5 - Training-time stack. The upper bound is deliberate, since 2026.5.x needs transformers 5.x (`config/requirements-profiling.txt`).
- `accelerate` >=1.0.0,<2 - Distributed/training runtime.
- `datasets` 4.3.0; `huggingface_hub` 0.34.1; `tokenizers` >=0.22,<0.23; `safetensors` >=0.4,<0.5.
- `bitsandbytes` 0.48.0 - Pinned for 4/8-bit safety (`config/requirements-profiling.txt`).
- `tinker` and `tinker_cookbook` - Hosted fine-tuning / RL backend (Thinking Machines). Imported across roughly 90 scripts, for example `scripts/rl_train.py`, `scripts/tinker_reasoning_sft_v4.py`, `scripts/tinker_reasoning_data_v4.py`. A git-ignored `.venv-tinker/` is the expected local env for these scripts (`.gitignore`); it is not committed.

**Serving and inference:**
- vLLM - OpenAI-compatible server, bf16 serving on port 8020 (`scripts/serve_v4_judge_vllm.sh`, `scripts/serve_30_70_vllm.sh`, `scripts/serve_base20_vllm.sh`, `scripts/serve_consistency_vllm.sh`, `scripts/serve_reasoning_v3_vllm.sh`, `scripts/serve_v4_profile_vllm.sh`). Configured by `recipes/*-vllm.yaml`.
- llama.cpp `llama-server` (build b9180+) - Serves the shipped GGUF on port 8020 with `-c 16384 --jinja`, with `--spec-type draft-mtp` for MTP speculative decoding (`README.md` Quickstart).
- Ollama - Named as an alternative server for GGUF (`README.md`). No repo code path found for it.
- `litellm` - Used by the wp-bench harness through `scripts/run_eval_reasoning.py` and `scripts/_wpbench_pth/usercustomize.py`. `config/dgx_toolbox.yaml` lists `litellm` as a component on port 4000.

**Quantization and export:**
- `gguf` Python package and llama.cpp convert/quant tools - GGUF conversion and the Q8_0 / Q6_K / Q5_K_M quantization ladder (`scripts/eval4_ext_gguf_convert.sh`, `scripts/pkg4_quant_type_check.py`).
- `scripts/aimer_prune_v4.py` - Expert pruning (AIMER k=224 variant).

**Evaluation and analysis:**
- `numpy` >=1.26,<3; `scipy` >=1.11; `scikit-learn` >=1.5,<1.7; `xgboost` >=2.1,<3 - Dual-head rubric calibration model (`config/requirements-profiling.txt`, `scripts/` calibration scripts). `statsmodels` and `pandas` are imported by some analysis scripts.
- PHP_CodeSniffer + WordPress-Coding-Standards - PHPCS pre-filter and scoring for the PHP data pipeline and reward signal (`eval/rubric_scorer.py`, `scripts/reward_pipeline.py`).
- `wp-bench` - External WordPress benchmark harness. Run through `config/wp-bench.yaml` (grader `kind: docker`, suite `wp-core-v1`). Appears about 189 times in scripts and eval code.

**Testing:**
- `pytest` - Test runner. `tests/` has about 63 test files, with `tests/conftest.py` and `tests/fixtures/`. The `unittest` module is also used in some tests.
- Run command (no config file committed): `pytest tests/`

**Build/Dev:**
- No build system (no Makefile, Dockerfile, or CI config in the repo).
- DGX Toolbox - Execution engine for training, eval, and serving containers. Config lives in `config/dgx_toolbox.yaml`, with `dgx_toolbox_path: ~/dgx-toolbox`. Overridable with `DGX_TOOLBOX_PATH`.
- Claude Code - Drives LLM-assisted pipeline steps and project skills in `.claude/skills/wp-finetune:*/`.

## Key Dependencies

**Critical (pinned because they break adapter loading or training):**
- `transformers==4.56.2` / `peft==0.18.1` / `trl==0.26.1` - Exact versions the 30/70 adapter was trained against. The NGC `pytorch:26.02-py3` container is explicitly excluded because its transformers 5.8.0 stack is incompatible (`config/requirements-profiling.txt`).
- `unsloth` <2026.5 - Upper bound to avoid transformers 5.x breakage.
- `scikit-learn` <1.7 - sklearn 1.7+ removed `_estimator_type`, which xgboost 2.1.x relies on at `save_model` time.
- `bitsandbytes==0.48.0` - Avoids the 0.49.2 missing `libbitsandbytes_cuda131.so` error.

**Infrastructure:**
- `pyyaml` >=6 - YAML config loading (`README.md` Requirements, `config/requirements-profiling.txt`).
- `python-dotenv` - Listed in `README.md` and `extra_deps` in `config/dgx_toolbox.yaml`. Loads `.env` (`.env.example` shows one key, `ANTHROPIC_API_KEY`).
- `mlflow` - Experiment tracking. Imported in `scripts/train_model.py`. Listed in `extra_deps`.
- `openai` (Python SDK) - Client for the local OpenAI-compatible vLLM / llama.cpp endpoints. Used with `api_key="none"` (for example `scripts/generate_cot_vllm.py`, `scripts/rl_judge_dispatch.py`, `eval/llm_checks.py`). Listed in `extra_deps`.
- `requests` - HTTP client in `scripts/pub4_validate_upload.py`.
- `hf` CLI (`hf download`, `hf upload`) - Model download and publishing to HuggingFace (`README.md`, `scripts/_pub03_upload.sh`, `scripts/_pub4_upload.sh`).
- `anthropic` Python SDK - Imported only in tests (`tests/test_phase2_judge_dataset.py`, `tests/test_generate_critique_then_fix.py`, `tests/test_generate_deep_judge_cot.py`). Runtime Anthropic access goes through the `claude` CLI (see INTEGRATIONS.md).
- `telemetry` package (DGX Toolbox) - GPU sampling and failure classification (`scripts/train_model.py`, `scripts/adaptive_planner.py`). Mounted into containers from `~/dgx-toolbox/telemetry`.

## Configuration

**Environment:**
- Config is layered YAML files in `config/`. Training variants are `config/train_config*.yaml` (for example `config/train_config_v4.yaml`, `config/train_config_reasoning.yaml`, `config/train_config_30_70.yaml`).
- Runtime settings come from environment variables read with `os.environ` / `os.getenv` (names only; see INTEGRATIONS.md for the list):
  - Backend selection: `LLM_BACKEND` (`claude` default, or `vllm`), `LLM_VLLM_BASE_URL`, `VLLM_BASE_URL`, `VLLM_MODEL`, `LLM_VLLM_MODEL`
  - Eval endpoints: `EVAL_JUDGE_BASE_URL`, `EVAL_GEN_BASE_URL`
  - Toolbox path: `DGX_TOOLBOX_PATH`
  - Behavior flags: `RUBRIC_USE_LLM_CHECKS`, `REWARD_SKIP_PHPCS_ASSERT`, `WP_JUDGE_DEBUG_DUMP`, `WPBENCH_LIMIT`, `PILOT_TARGET`, `SIEVE_PROFILE_*`, `ADAPTIVE_THERMAL_LOG`, `SWEEP_RESULTS_PATH`
- `.env` file: loaded by `python-dotenv`. Template at `.env.example` (`ANTHROPIC_API_KEY` placeholder). The actual `.env` is not read for this document.
- Required secrets: `TINKER_API_KEY` for the Tinker backend (`scripts/build_base20_probe_adapter.py`).

**Build:**
- `config/dgx_toolbox.yaml` - Container map, component paths, shared directories, ports, pinned versions, `required_imports`, `extra_deps`.
- `config/requirements-profiling.txt` - Pinned HF / training / eval stack.
- `config/wp-bench.yaml` - wp-bench run config.
- `config/repos.yaml` - Source repos for the data pipeline (236 WordPress repos per `README.md`).
- `config/adaptive_planning.yaml`, `config/rubric_calibration.yaml`, `config/taxonomy.yaml`, `config/synthetic_prompts.yaml`, `config/judge_system.md` - Pipeline and rubric configuration.
- `recipes/qwen3.6-35b-a3b-vllm.yaml`, `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml`, `recipes/qwen3-30b-wp-30_70-vllm.yaml` - vLLM serving recipes.

## Models

- Base (canonical v4): `Qwen/Qwen3.6-35B-A3B` (native MoE, 256 experts, top-8 routing). Local path `./models/Qwen3.6-35B-A3B` (`config/train_config_v4.yaml`). FP8 variant `Qwen/Qwen3.6-35B-A3B-FP8` is used by `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml`.
- Base (v1.3 and 30/70 experiments): `Qwen/Qwen3-30B-A3B` (`config/train_config*.yaml`, `config/dgx_toolbox.yaml` `status_artifacts`).
- Shipped artifact: `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` on HuggingFace (`README.md`). Recommended file `wp-judge-v4-unpruned.Q5_K_M.gguf` (23.61 GiB). Pruned file `wp-judge-v4-pruned-k224.Q6_K.gguf` (23.47 GiB).
- Special tokens added to the tokenizer: `<wp_gen>`, `<wp_judge>` (`config/train_config_v4.yaml`).
- Training: rank-32 MoE-only LoRA, `lora_alpha` 64, `lora_dropout` 0.05, `max_seq_length` 4096, learning rate 2e-4 cosine, 2 epochs in v4 (`config/train_config_v4.yaml`).

## Platform Requirements

**Development:**
- Linux with an NVIDIA GPU. The documented target is a DGX Spark (GB10, aarch64, CUDA).
- Python 3.10+, PHP CLI with `tokenizer`, PHP_CodeSniffer, and WordPress-Coding-Standards.
- Docker (DGX Toolbox containers, `wp-bench` grader with `kind: docker`).
- Claude Code CLI (`claude`) for LLM-assisted pipeline steps.
- `hf` CLI for HuggingFace downloads and uploads.

**Production:**
- Self-hosted inference. No external API is required to run the judge (`README.md`).
- Served as either GGUF via llama.cpp b9180+ or Ollama, or as bf16 via vLLM. The default port is 8020 (`config/dgx_toolbox.yaml` `ports`).
- Published to HuggingFace under the `iamchum` account (Apache 2.0).

---

*Stack analysis: 2026-10-09*
