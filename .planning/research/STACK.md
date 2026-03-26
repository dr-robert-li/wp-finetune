# Stack Research

**Domain:** LLM fine-tuning pipeline — WordPress code data + dense-to-MoE conversion on DGX Spark
**Researched:** 2026-03-26
**Confidence:** MEDIUM-HIGH (DGX Toolbox integration verified; MoE conversion approach is MEDIUM due to evolving tooling)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ (3.11 preferred) | Pipeline orchestration, training scripts | DGX Spark playbooks tested against 3.11; Unsloth requires 3.10+; 3.13 supported via uv |
| Unsloth | 2026.3.x (latest) | LoRA fine-tuning of Qwen3-8B on DGX Spark | Official DGX Spark integration via `nvidia/dgx-spark-playbooks`; 2x faster than standard HuggingFace training; 70% less VRAM; verified Qwen3-8B support |
| TRL (HuggingFace) | >=0.26.1 | SFTTrainer for supervised fine-tuning | Required by Unsloth DGX playbook (`trl==0.26.1`); SFTTrainer handles chat template formatting and multi-format dataset ingestion |
| Transformers (HuggingFace) | >=4.56.2 (pinned at 4.56.2 in DGX playbook) | Model loading, tokenizer, Qwen3 architecture | Qwen3 requires `>=4.51.0`; DGX playbook pins at 4.56.2 for Blackwell stability |
| PEFT (HuggingFace) | >=0.14.0 | LoRA adapter management | Required by Unsloth for LoraConfig; integrates with SFTTrainer natively |
| CMoE | Research code (arxiv:2502.04416) | Dense-to-MoE conversion of Qwen3-8B | Training-free conversion in under 5 minutes on a single GPU; analytically constructs router from activation statistics; no continual pre-training required; supports 8-expert configurations matching project spec |
| vLLM | >=0.9.0 | Production inference serving (AWQ) | Native Qwen3 + Qwen3MoE support from v0.9.0; AWQ+Marlin kernel delivers 741 tok/s; official DGX Spark vLLM playbook available |
| Ollama | Latest (>=0.6.x) | Local GGUF serving for developer access | One-command serving: `ollama run hf.co/Qwen/Qwen3-8B-GGUF:Q8_0`; fits within DGX Toolbox inference stack |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| anthropic (Python SDK) | >=0.50.0 | Claude API integration for judging and generation | All pipeline phases that call Claude (phase1_judge, phase2_generate, phase2_judge, phase3_cot); use `claude-sonnet-4-6` for bulk judging, `claude-opus-4-6` for CoT reasoning |
| pyyaml | >=6.0 | YAML config parsing | Reading `repos.yaml`, `taxonomy.yaml`, `synthetic_prompts.yaml` |
| bitsandbytes | 0.48.0 (pinned by DGX playbook) | 4-bit/8-bit quantization during fine-tuning | QLoRA training path; enables 4-bit loading to reduce VRAM to ~20GB for Qwen3-8B |
| datasets (HuggingFace) | 4.3.0 (pinned by DGX playbook) | Dataset loading and formatting for SFTTrainer | Convert JSONL training data to HuggingFace Dataset format before passing to Unsloth SFTTrainer |
| lm-evaluation-harness | >=0.4.5 | Model evaluation (HumanEval, code benchmarks) | Post-training evaluation; part of DGX eval-toolbox; supports HumanEval for PHP code quality proxy |
| torch | From container (nvcr.io/nvidia/pytorch:25.11-py3) | GPU compute backbone | Do NOT install separately — use NVIDIA's PyTorch container which is patched for Blackwell |
| flash-attn | Built from source for Blackwell (Triton + xFormers) | Flash Attention 2/3 for memory-efficient attention | Required for Qwen3 context lengths beyond 4k; Blackwell requires source build, not pip wheel |
| accelerate | >=1.0.0 | Distributed training orchestration | Single-GPU on DGX Spark; needed for gradient accumulation and mixed precision |
| sentencepiece / tiktoken | >=0.2.0 | Tokenizer backend for Qwen3 | Qwen3 uses tiktoken-based tokenizer; required for adding `<wp_gen>` and `<wp_judge>` special tokens |

### Data Pipeline Tools (External Executables)

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| PHP CLI | 8.1+ (with tokenizer extension) | Function extraction from WordPress source | PHP 8.1+ recommended; `php-tokenizer` extension must be enabled; verify with `php -m | grep tokenizer` |
| PHP_CodeSniffer (phpcs) | >=3.9.0 | WPCS compliance pre-filtering | Install via Composer only (pip not available); `squizlabs/php_codesniffer` |
| WordPress-Coding-Standards | >=3.1.0 | PHPCS ruleset for WordPress conventions | Install via `composer require --dev wp-coding-standards/wpcs:"^3.0"`; Composer auto-registers rulesets |
| git | >=2.30 | Shallow repo cloning | Used via subprocess in `phase1_clone.py`; `--depth=1` clones only |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Docker (NVIDIA Container Runtime) | Isolates Unsloth + PyTorch environment on DGX Spark | Use `nvcr.io/nvidia/pytorch:25.11-py3` base; launch with `--gpus=all --ulimit memlock=-1 --ipc=host` |
| Composer | PHP dependency management | Required for PHPCS + WPCS; global install with `composer global require` |
| Jupyter Notebook | Interactive training session management | Launched inside Docker container per DGX Spark Unsloth playbook |
| wandb / W&B | Training metrics logging | Optional but strongly recommended for LoRA loss curves and VRAM tracking |
| huggingface_hub CLI | HuggingFace model upload and download | Final packaging step; `huggingface-cli upload` for public release |

---

## Installation

```bash
# 1. Pull NVIDIA PyTorch base container (Blackwell-patched)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# 2. Inside container: install Python dependencies
pip install unsloth unsloth_zoo
pip install "transformers==4.56.2" "trl==0.26.1" "datasets==4.3.0"
pip install "bitsandbytes==0.48.0" --no-deps
pip install peft accelerate sentencepiece
pip install "anthropic>=0.50.0" pyyaml

# 3. Build Triton + xFormers from source for Blackwell Flash Attention
# (follow nvidia/dgx-spark-playbooks Dockerfile)

# 4. PHP data pipeline dependencies
composer global require --dev wp-coding-standards/wpcs:"^3.0"
# Installs phpcs + WPCS; registers rulesets automatically

# 5. Verify setup
php -m | grep tokenizer        # must print "tokenizer"
phpcs --version                # must print PHP_CodeSniffer >= 3.9.x
python -c "import unsloth; print('unsloth ok')"
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| CMoE (training-free conversion) | LLaMA-MoE (continual pre-training) | If you need higher post-conversion quality and have 200B token training budget; LLaMA-MoE recovers more capability but requires days of GPU time vs 5 minutes for CMoE |
| CMoE (training-free conversion) | ToMoE (dynamic structural pruning) | ToMoE has slightly better structural pruning quality and was tested on Qwen-2.5; valid alternative if CMoE activation profiling produces poor routing for PHP code tasks |
| Unsloth + TRL SFTTrainer | Axolotl | Axolotl has broader config file support but no DGX Spark official playbook; Unsloth is the officially supported path for DGX Spark |
| Unsloth + TRL SFTTrainer | LLaMA-Factory | LLaMA-Factory supports more training paradigms (DPO, RLHF) but adds complexity not needed for SFT-only v1; revisit for v2 DPO refinement |
| vLLM + AWQ (production serving) | llama.cpp + GGUF | llama.cpp is better for CPU-only or offline scenarios; on DGX Spark with A100/GB10, vLLM+AWQ+Marlin is 10x faster; use GGUF only for Ollama developer access |
| claude-sonnet-4-6 for bulk judging | claude-opus-4-6 for bulk judging | Opus is 5x more expensive ($25/MTok output vs $15/MTok); reserve Opus for chain-of-thought generation (phase3_cot) where reasoning depth matters; Sonnet sufficient for 9-dimension scoring |
| PHP_CodeSniffer + WPCS 3.x | WPCS 2.x | WPCS 3.x changed installation (Composer-only); ruleset naming changed; if an existing global install shows WPCS 2.x, upgrade — 2.x is unmaintained |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pip install torch` inside DGX container | Overwrites NVIDIA's Blackwell-patched PyTorch; Flash Attention will silently fall back to slow path or crash | Use `nvcr.io/nvidia/pytorch:25.11-py3` container; torch is pre-installed and patched |
| QLoRA 4-bit on MoE models after conversion | Unsloth documentation explicitly states "MoE QLoRA 4-bit is not recommended due to BitsandBytes limitations"; training on the converted MoE with QLoRA may produce corrupted gradients | Fine-tune the dense Qwen3-8B with LoRA FIRST, then convert to MoE; or use full-precision LoRA on MoE |
| Bare-metal Unsloth install (no Docker) | DGX Spark Blackwell GPUs require custom Triton and xFormers builds; bare-metal pip install will not get Flash Attention working correctly | Use the NVIDIA DGX Spark Docker container approach |
| WPCS 2.x / global phpcs without Composer plugin | WPCS 3.0+ changed ruleset registration; the dealerdirect Composer installer plugin is required; manual PATH tricks from pre-3.0 guides will fail | `composer config allow-plugins.dealerdirect/phpcodesniffer-composer-installer true` then `composer require wp-coding-standards/wpcs:"^3.0"` |
| `transformers > 4.56.2` during DGX training | DGX Spark playbook pins at 4.56.2 for compatibility; later versions may break bitsandbytes 0.48.0 or TRL 0.26.1 integration | Pin to `transformers==4.56.2` in the training container |
| GGUF with vLLM for production serving | vLLM documentation shows GGUF inference at 93 tok/s vs AWQ+Marlin at 741 tok/s; GGUF is designed for llama.cpp, not vLLM | Use `Qwen/Qwen3-8B-AWQ` with vLLM; keep GGUF only for Ollama developer access |
| LLaMA-MoE for dense-to-MoE if no extra compute budget | LLaMA-MoE requires 200B token continual pre-training to recover performance; impractical on a single DGX Spark for this project | Use CMoE (training-free, 5 min on single GPU) |

---

## Stack Patterns by Phase

**Phase 1-3 (Data Pipeline Execution):**
- Pure Python + PHP subprocess + Claude API — no GPU required
- Rate-limit Claude API calls to 40-50 RPM
- Use `claude-sonnet-4-6` for judging, `claude-opus-4-6` for phase3_cot only
- PHPCS pre-filter runs before any Claude API call to reduce cost ~60%

**MoE Conversion (post data pipeline):**
- Run CMoE conversion script against downloaded `Qwen/Qwen3-8B` checkpoint
- Use 8-sample WikiText-2 calibration for activation profiling
- Target configuration: S2A2E8 (2 shared + 2 active / 8 total experts, 50% activation) or S1A1E8 for maximum sparsity
- Conversion completes in ~5 minutes; validate with perplexity check before fine-tuning

**Fine-tuning (Unsloth + TRL SFTTrainer):**
- Use converted MoE checkpoint OR fine-tune dense Qwen3-8B first (safer; see What NOT to Use)
- LoRA configuration: `r=64`, `lora_alpha=128`, `target_modules="all-linear"` (excluding router layers), `task_type="CAUSAL_LM"`
- Add `<wp_gen>` and `<wp_judge>` to tokenizer BEFORE training; save `embed_tokens` and `lm_head` in `modules_to_save`
- Max sequence length: 4096 for training (covers typical WordPress function + docstring + instructions)
- Disable thinking mode in training: training data should not include `<think>...</think>` blocks for SFT
- Multi-task: interleave `<wp_gen>` and `<wp_judge>` examples (50/50 split) in dataset shuffle

**Evaluation:**
- PHPCS pass rate: pipe model outputs through `phpcs` CLI, measure `>95%` target
- Judge correlation: compare model judge scores vs Claude baseline on held-out set, target `>0.85` Pearson
- lm-evaluation-harness for standard code benchmarks (HumanEval as proxy)

**Packaging and Deployment:**
- GGUF: convert via `llama.cpp` `convert_hf_to_gguf.py`, Q4_K_M quantization, deploy via Ollama
- AWQ: use `autoawq` library to quantize to 4-bit, serve via `vllm serve Qwen3-8B-AWQ`
- HuggingFace: `huggingface-cli upload` adapter + merged checkpoint

---

## Key Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| unsloth-2026.3.x | transformers>=4.57.1, trl>=0.25.0 | Latest Unsloth relaxes pins; DGX playbook pins older |
| trl==0.26.1 | transformers==4.56.2, peft>=0.14.0 | Per NVIDIA DGX Spark playbook (updated 2025-12-15) |
| bitsandbytes==0.48.0 | torch from nvcr.io container, CUDA 13.0 | DGX playbook pins this; later versions may conflict |
| datasets==4.3.0 | transformers==4.56.2 | DGX playbook pin; dataset API is stable at this version |
| vllm>=0.9.0 | Qwen3 dense + MoE, AWQ, FP8 | v0.8.4 is minimum for Qwen3 support; 0.9.0 adds FP8 Marlin |
| PHP_CodeSniffer>=3.9.0 | WordPress-Coding-Standards>=3.1.0 | WPCS 3.x requires PHPCS 3.x; the Composer installer handles this |
| Qwen3-8B | transformers>=4.51.0 | Model architecture not recognized in older versions |

---

## Claude API Model Names (Current)

The pipeline scripts reference model strings directly. Use these verified current IDs:

| Use Case | Model ID | Cost (output) |
|----------|----------|---------------|
| Bulk judging (phases 1-2) | `claude-sonnet-4-6` | $15/MTok |
| Chain-of-thought generation (phase 3) | `claude-opus-4-6` | $25/MTok |
| (Legacy — existing scripts) | `claude-sonnet-4-6-20250514` | Still valid alias |

---

## Sources

- [Unsloth Qwen3 Fine-tune Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — Qwen3-8B LoRA configuration, MoE router notes (HIGH confidence)
- [Unsloth DGX Spark Guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth) — Docker image, pinned dependency versions (HIGH confidence)
- [NVIDIA DGX Spark Playbooks — Unsloth](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/unsloth) — `nvcr.io/nvidia/pytorch:25.11-py3` image, exact dependency pins (HIGH confidence)
- [CMoE Paper (arxiv:2502.04416)](https://arxiv.org/abs/2502.04416) — Training-free conversion methodology, S1A1E8 / S2A2E8 configs, 5-min conversion time (MEDIUM confidence — research paper, not production library)
- [ToMoE Paper (arxiv:2501.15316)](https://arxiv.org/abs/2501.15316) — Alternative MoE conversion method tested on Qwen-2.5 (MEDIUM confidence)
- [LLaMA-MoE Paper (arxiv:2406.16554)](https://arxiv.org/abs/2406.16554) — Continual pre-training methodology (HIGH confidence for methodology; LOW confidence for this project given compute constraints)
- [vLLM Qwen3 Usage Guide](https://github.com/vllm-project/vllm/issues/17327) — v0.8.4+ Qwen3 support, AWQ+Marlin 741 tok/s (HIGH confidence)
- [HuggingFace Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) — Architecture (36 layers, 8.2B params), transformers>=4.51.0 requirement, Apache 2.0 license (HIGH confidence)
- [Anthropic Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview) — Current model IDs: claude-sonnet-4-6, claude-opus-4-6 (HIGH confidence)
- [WordPress-Coding-Standards GitHub](https://github.com/WordPress/WordPress-Coding-Standards) — WPCS 3.x Composer-only install, `^3.0` requirement (HIGH confidence)
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer) — LoraConfig parameters, `modules_to_save` for special tokens (HIGH confidence)

---

*Stack research for: wp-qwen3-moe (WordPress fine-tuning + MoE conversion on DGX Spark)*
*Researched: 2026-03-26*
