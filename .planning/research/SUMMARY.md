# Project Research Summary

**Project:** wp-qwen3-moe
**Domain:** WordPress-specific code generation and judgment LLM via LoRA fine-tuning + dense-to-MoE conversion on DGX Spark
**Researched:** 2026-03-26
**Confidence:** MEDIUM-HIGH (stack and features HIGH; MoE conversion MEDIUM due to research-paper-stage tooling)

## Executive Summary

This project fine-tunes a Qwen3-8B model to produce a dual-mode WordPress code specialist that can both generate PHPCS-compliant PHP and act as a structured 9-dimension code judge — capabilities that no existing tool (CodeWP, GitHub Copilot) combines in a single self-hostable model. The recommended expert consensus approach is: run the existing 3-phase data pipeline to produce ~13,500 training examples, convert the dense Qwen3-8B checkpoint to an 8-expert MoE architecture via CMoE (training-free, ~5 minutes), extend the tokenizer with `<wp_gen>` and `<wp_judge>` task tokens, and then apply Unsloth LoRA SFT on DGX Spark inside NVIDIA's official Docker container. Deployment targets two backends: AWQ quantization for vLLM high-throughput serving and GGUF for Ollama developer access. The pipeline, training stack, and serving stack all have official NVIDIA DGX Spark playbooks, which substantially reduces integration risk.

The key risk is data pipeline fragility, not model architecture. The existing pipeline code has several critical bugs that will silently corrupt the training dataset at production scale: a brittle JSON parser that substitutes stub objects on Claude response format variations, missing checkpointing that requires full restarts from scratch on any failure, a PHPCS silent-degradation bug that accepts unverifiable mutations when the tool is unavailable, and no rate-limiting in `phase2_judge_dataset.py`. These must be fixed before the first full pipeline run. A secondary risk is the MoE conversion: CMoE is research code (arxiv:2502.04416), not a production library, and the conversion-then-fine-tune ordering is non-negotiable — fine-tuning the dense model first and converting afterward defeats the entire specialization purpose.

The project has a clear quality gate strategy: PHPCS pass rate >95% and judge correlation >0.85 (Pearson) against a Claude reference on a held-out test set. These thresholds block packaging and deployment, preventing a substandard model from being published. The evaluation suite must be designed and written before training starts, not after — otherwise the metrics cannot be used as a blocking gate.

## Key Findings

### Recommended Stack

The training stack is dictated by NVIDIA's official DGX Spark playbooks, which pins key dependencies: `nvcr.io/nvidia/pytorch:25.11-py3` base container (do not install torch separately), `transformers==4.56.2`, `trl==0.26.1`, `bitsandbytes==0.48.0`, `datasets==4.3.0`. Deviating from these pins on Blackwell hardware risks silent Flash Attention fallback or crashes. Unsloth is the officially supported fine-tuning framework for DGX Spark and provides verified Qwen3-8B + MoE support. The dense-to-MoE conversion uses CMoE (arxiv:2502.04416) — training-free, ~5 minutes on a single GPU, targeting S2A2E8 (2 shared + 2 active of 8 total experts). Inference is split: vLLM >=0.9.0 with AWQ+Marlin kernel for production (741 tok/s), Ollama for developer access (GGUF Q4_K_M). The data pipeline uses Claude API with a critical model split: `claude-sonnet-4-6` for bulk judging in phases 1-2, `claude-opus-4-6` for chain-of-thought generation in phase 3 only (3x the cost — must have a pre-run cost estimate and `--max-cot` cap).

**Core technologies:**
- Unsloth (2026.3.x): LoRA SFT on DGX Spark — official NVIDIA playbook, 2x faster + 70% less VRAM than vanilla HuggingFace training
- TRL SFTTrainer (0.26.1): supervised fine-tuning framework — required by Unsloth DGX playbook, handles chat template and dataset formatting
- CMoE (arxiv:2502.04416): dense-to-MoE conversion — training-free, 5 minutes, 8-expert config matching project spec; only MEDIUM confidence as research code
- vLLM (>=0.9.0): production inference with AWQ+Marlin — native Qwen3+Qwen3MoE support, 741 tok/s on DGX Spark
- PHP_CodeSniffer + WPCS 3.x: code quality pre-filter — Composer-only install (WPCS 3.x breaking change from 2.x); must be present before any data pipeline run
- Anthropic Python SDK (>=0.50.0): Claude API for judging and generation — `claude-sonnet-4-6` for bulk, `claude-opus-4-6` for CoT only

### Expected Features

The model's core value proposition is its dual-mode capability with structured judgment output — a capability gap confirmed against all identified competitors.

**Must have (table stakes):**
- PHPCS-passing PHP generation (WPCS compliance) — generated code that fails PHPCS is immediately rejected by CI pipelines; the model is worse than PHPCS itself without this
- SQL safety via `$wpdb->prepare()` with typed placeholders — top WordPress plugin vulnerability class (Patchstack 2025); non-negotiable trust signal
- Nonce generation/verification and capability checks — CSRF and privilege escalation prevention; missing these patterns in generated code is an instant trust-breaker
- Context-appropriate output escaping (`esc_html`, `esc_attr`, `esc_url`, `wp_kses`) — wrong function for context is a functional bug, not just style
- WP_Query for post queries, hook registration with correct signature, `register_rest_route()` with `permission_callback` — the three most common WP-specific coding tasks
- `<wp_judge>` mode returning 9-dimension JSON (wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility) with verdict and critical_failures — the primary differentiator; no competitor has this
- CoT reasoning for SQL and security patterns — makes the model explainable, not just generative

**Should have (competitive):**
- Contrastive defect explanation (bad → good mutation pairs with CoT annotations) — trained on Phase 2 mutation data; enables automated code-review use case
- Multisite awareness (switch_to_blog, per-site table prefixes, network vs site options) — nearly absent from general code model training data
- Taxonomy-grounded coverage across all 12 WP concept categories — enforces minimum coverage of underrepresented patterns (multisite, cron)
- Admin UI generation (settings pages, meta boxes, list table columns) — high developer demand

**Defer (v2+):**
- DPO/RLHF preference optimization — requires new training infrastructure; SFT alone is sufficient for v1 if PHPCS >95% and correlation >0.85
- JavaScript/Gutenberg block generation (`<wp_block>` task token) — entirely different domain; mixing JS/React training dilutes WP PHP quality
- WooCommerce-specific expert pathway (`<wc_gen>`) — warrants separate task token and dedicated training data
- Multi-lingual comment support — requires translation validation pipeline

### Architecture Approach

The system is a strictly sequential 5-layer pipeline: data pipeline (phases 1-3) produces training data, model preparation (CMoE conversion + tokenizer extension) produces the base MoE model, training (Unsloth SFT) produces LoRA adapters and a merged checkpoint, evaluation (custom domain scripts + standard benchmarks) gates deployment, and packaging (AWQ + GGUF + HF upload) produces serving artifacts. Component communication is entirely via file system handoffs — no network APIs between layers. The critical build constraint is ordering: convert dense to MoE first, then extend tokenizer, then fine-tune. Model preparation can run in parallel with data pipeline execution (independent inputs) but must complete before training begins. The evaluation-to-packaging boundary is a deliberate manual human gate, not an automated trigger.

**Major components:**
1. Data Pipeline (Phase 1-3) — 10 existing Python scripts producing ~13,500 training examples across gen/judge/CoT formats; needs bug fixes before first full run
2. Model Preparation (model_prep/) — CMoE conversion script + tokenizer extension + smoke-test verification; all scripts to be written
3. Training (training/) — Unsloth Studio Docker notebook on DGX Spark; LoRA r=64, all-linear targets, router frozen, `modules_to_save=["embed_tokens","lm_head"]`
4. Evaluation (eval/) — custom PHPCS pass rate and judge correlation scripts + lm-eval-harness standard benchmarks; scripts to be written
5. Packaging + Deployment — AWQ quantization for vLLM, GGUF for Ollama, HF Hub upload; all scripts to be written

### Critical Pitfalls

1. **Silent JSON parse failures poison training data** — Replace brittle `split("```json")` pattern across all three judge scripts with a multi-strategy extractor (JSON fence with hint, bare fence, regex for outermost `{...}`, raw `json.loads`). Hard-reject on all-strategy failure; abort if >2% of responses fail. Never substitute stub objects.

2. **No pipeline checkpointing makes multi-hour runs unrecoverable** — Add checkpoint JSON every 100 examples across `phase1_judge.py`, `phase2_generate.py`, and `phase3_cot.py`. Add `--sample N` flag before first full run; this is also required for cheap end-to-end pipeline testing.

3. **PHPCS silent degradation corrupts mutation training data** — `phase2_mutate.py` returns `True` (mutation accepted) when PHPCS is unavailable. Add a `verify_phpcs_available()` pre-flight check that hard-exits on failure. Never accept mutations silently.

4. **Special token embeddings stay randomly initialized unless explicitly trained** — LoRA freezes `embed_tokens` by default; new `<wp_gen>` and `<wp_judge>` rows receive zero gradient updates. Set `modules_to_save=["embed_tokens","lm_head"]` in LoRA config before training. Verify embedding norms are non-zero after first 100 steps.

5. **MoE routing collapse renders conversion useless** — Without load balancing loss, the top-2 router collapses to always routing to 1-2 experts. Keep auxiliary load balancing loss active during SFT (reduced coefficient, not zeroed). Monitor per-expert token distribution at each checkpoint; abort and adjust if any expert exceeds 30% of tokens.

## Implications for Roadmap

Based on the strict sequential dependency structure and identified bugs, the recommended phase structure is as follows. The data pipeline must be fixed and validated before any model work begins. Model preparation and training must be instrumented before execution to avoid costly recoveries.

### Phase 1: Data Pipeline Hardening and Execution

**Rationale:** The existing pipeline scripts have critical bugs (JSON parse failures, missing checkpoints, PHPCS silent degradation, missing rate limiting in `phase2_judge_dataset.py`) that will corrupt the training dataset at scale. These must be fixed first because downstream phases (model prep, training) are worthless if trained on bad data. The data pipeline is also the primary cost driver via Claude API calls — fixing it before running saves money and time.

**Delivers:** ~13,500 clean, PHPCS-validated, Claude-judged training examples in `final_dataset/` with train/val/test splits in three formats (OpenAI JSONL, Alpaca JSON, raw JSONL); N/A dimension scoring fixed; no train/val code leakage; taxonomy coverage verified.

**Addresses:** All P1 generation features depend on training data quality; structured 9-dimension judge training data for `<wp_judge>` mode; CoT examples for complex patterns.

**Avoids:** Pitfalls 1 (parse failures), 2 (rate limit crashes), 3 (no checkpointing), 4 (PHPCS silent degradation), 9 (train/val leakage), 10 (Opus cost shock), 12 (N/A score inflation).

**Research flag:** Standard patterns — existing scripts are the source of truth. No additional research needed; the CONCERNS.md codebase audit already identifies all issues.

### Phase 2: Model Preparation (MoE Conversion + Tokenizer Extension)

**Rationale:** Model preparation is an independent prerequisite for training. It can be developed in parallel with Phase 1 data pipeline work but must complete before training begins. The conversion-then-fine-tune ordering is non-negotiable; reversing it defeats specialization. The smoke-test verification step is mandatory before investing in a training run.

**Delivers:** `./qwen3-8b-moe-tokenized/` — an 8-expert Qwen3-8B-MoE checkpoint with `<wp_gen>` and `<wp_judge>` tokens added, embeddings resized, and routing verified via a forward-pass smoke test confirming all 8 experts fire.

**Uses:** CMoE (arxiv:2502.04416) for training-free conversion; HuggingFace `add_special_tokens` + `resize_token_embeddings` for tokenizer extension; `nvcr.io/nvidia/pytorch:25.11-py3` container.

**Implements:** Model Preparation architecture layer (model_prep/ directory — all scripts to be written).

**Avoids:** Architecture anti-pattern of training dense then converting; pitfall of skipping smoke test; pitfall 5 (special token embeddings untrained) by designing the LoRA config correctly from the start.

**Research flag:** Needs research during planning — CMoE is research code, not a production library. Verify the exact API, confirm Qwen3-8B is a tested architecture for CMoE, and identify whether ToMoE is a safer fallback if CMoE activation profiling produces poor routing on PHP code inputs.

### Phase 3: Evaluation Suite Design (Before Training)

**Rationale:** The evaluation scripts must be written before training begins, not after. The deployment gate (PHPCS >95%, judge correlation >0.85) is only meaningful if the evaluation scripts are ready to measure it. Writing eval scripts post-training means the gate cannot function as designed. This phase is short but has a hard ordering dependency: it must precede training.

**Delivers:** `eval/eval_phpcs.py`, `eval/eval_judge_correlation.py`, `eval/eval_benchmarks.sh` — all three runnable against any merged model checkpoint. Baseline Qwen3-8B dense scores captured before conversion for regression comparison.

**Uses:** PHP_CodeSniffer + WPCS (must be installed in eval environment), lm-eval-harness or bigcode-evaluation-harness, `final_dataset/openai_test.jsonl` as the held-out test set.

**Avoids:** Pitfall 7 (PHPCS-only evaluation metric overfitting); architecture anti-pattern of evaluating only on validation loss during training.

**Research flag:** Partially standard — lm-eval-harness is well-documented. The judge correlation script is custom and needs a decision on whether to use the same Claude model as training judge or a different model to avoid circularity. This should be resolved before training starts.

### Phase 4: Fine-Tuning on DGX Spark

**Rationale:** Training is gated on both the training dataset (Phase 1) and the model checkpoint (Phase 2) being ready, and on the evaluation suite (Phase 3) being in place to measure results. This is the most compute-intensive phase and the one with the highest recovery cost if setup is wrong.

**Delivers:** LoRA adapter checkpoints + merged `./wp-qwen3-8b-moe-merged/` in BF16 — the primary model artifact.

**Uses:** Unsloth Studio Docker on DGX Spark; `nvcr.io/nvidia/pytorch:25.11-py3` base; TRL SFTTrainer; LoRA r=64, `target_modules="all-linear"`, router layers frozen, `modules_to_save=["embed_tokens","lm_head"]`; 50/50 gen/judge training split; wandb for loss curve monitoring.

**Implements:** Training architecture layer (training/ directory — Jupyter notebook + train_config.yaml).

**Avoids:** Pitfall 5 (modules_to_save), pitfall 6 (MoE routing collapse via load balancing loss monitoring), pitfall 8 (multi-task interference via per-task loss tracking), STACK.md "What NOT to Use" items (QLoRA on MoE, bare-metal install, transformers version override).

**Research flag:** Partially standard — Unsloth DGX Spark playbook is well-documented and official. Needs validation during planning on: (a) whether to fine-tune the dense model first then convert vs. convert then fine-tune (research confirms convert-first, but verify with Unsloth MoE LoRA docs); (b) load balancing loss coefficient for SFT (not pretraining scale); (c) optimal LoRA rank for MoE vs. dense (r=64 is a starting point, not validated for this MoE config).

### Phase 5: Evaluation and Quality Gate

**Rationale:** Evaluation must occur on the held-out test split using the scripts from Phase 3. Both thresholds (PHPCS >95%, correlation >0.85) must pass before packaging begins. If either fails, the recovery path is to resume training from the last checkpoint with adjusted hyperparameters — which requires the LoRA adapter to still exist separately from the base model (do not merge early).

**Delivers:** Numeric eval results confirming or blocking deployment; regression comparison against Qwen3-8B dense baseline; decision on whether to proceed to packaging.

**Avoids:** Pitfall 7 (PHPCS-only metric); architecture anti-pattern of merging LoRA before evaluation passes; architecture anti-pattern of validating only on training loss.

**Research flag:** Standard patterns — eval scripts are custom but the evaluation methodology is well-defined in the project spec.

### Phase 6: Packaging and Deployment

**Rationale:** Packaging is downstream of evaluation passing. AWQ and GGUF quantization can run in parallel. HuggingFace Hub upload is last. GGUF goes to Ollama; AWQ+Marlin goes to vLLM. These must not be mixed — GGUF in vLLM is experimental and ~8x slower than AWQ+Marlin.

**Delivers:** `./wp-qwen3-8b-moe-awq/` for vLLM production serving; `./wp-qwen3-8b-moe.gguf` for Ollama developer access; HuggingFace Hub release with model card including eval metrics.

**Uses:** llm-compressor or AutoAWQ for AWQ quantization; llama.cpp for GGUF; huggingface-cli for upload; DGX Toolbox vLLM and Ollama containers.

**Avoids:** Pitfall 11 (GGUF in vLLM); architecture anti-pattern of using GGUF in vLLM for performance benchmarking.

**Research flag:** Needs partial research — AWQ+Marlin for MoE models must be verified; not all quantization tools support MoE routing tables. Confirm AutoAWQ or llm-compressor produces valid quantized weights for Qwen3MoE architecture before committing to this path.

### Phase Ordering Rationale

- **Data pipeline before everything else:** Training data quality is the primary determinant of model quality; fixing and running the pipeline is the critical path. All other phases depend on it.
- **Model prep can overlap Phase 1:** CMoE conversion and tokenizer extension scripts can be written and tested (on a small model) while the full data pipeline run is in progress, reducing elapsed time.
- **Eval suite before training:** The deployment gate is only meaningful if eval scripts exist before training outputs are produced. The temptation to write eval "later" must be resisted.
- **Don't merge LoRA early:** Keep adapter and base model separate until evaluation passes. This is a recovery-cost decision, not a technical one.
- **Phase 3 and 4 can partially overlap:** Eval suite can be written during the training run setup period, as long as it is complete before the training run produces a merged checkpoint.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Model Preparation):** CMoE is research code; need to verify Qwen3-8B is a tested architecture, confirm the exact Python API, and identify ToMoE as a validated fallback.
- **Phase 4 (Fine-Tuning):** Need to confirm load balancing loss coefficient recommendations for SFT-scale fine-tuning (not pretraining); verify Unsloth MoE LoRA documentation for router-freezing behavior.
- **Phase 6 (Packaging):** AWQ quantization support for Qwen3MoE architecture must be verified; MoE routing table preservation in quantized weights is not guaranteed by all tools.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Data Pipeline):** All issues are already identified in CONCERNS.md; fixes are implementation work, not research.
- **Phase 3 (Evaluation Suite):** Evaluation methodology is defined in PROJECT.md; the judge correlation circularity question should be resolved in requirements, not research.
- **Phase 5 (Evaluation + Gate):** Straightforward execution of Phase 3 scripts against Phase 4 outputs.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core training stack (Unsloth, TRL, transformers versions) verified against official NVIDIA DGX Spark playbooks. vLLM Qwen3MoE support confirmed from official docs. Only CMoE is MEDIUM (research paper, not production library). |
| Features | HIGH | Feature spec is grounded in the existing pipeline config files (judge_system.md, taxonomy.yaml) as primary source. Competitor analysis confirmed against multiple practitioner sources. Security claims backed by Patchstack 2025 primary research. |
| Architecture | HIGH (pipeline), MEDIUM (MoE) | Pipeline architecture is directly from existing codebase — highest confidence. MoE conversion approach is confirmed in principle from multiple peer-reviewed papers but the specific CMoE implementation is research code. DGX eval-toolbox specifics remain LOW — fallback to lm-eval-harness is the safe path. |
| Pitfalls | HIGH (pipeline bugs), MEDIUM (training pitfalls) | Pipeline bugs (pitfalls 1-7) are grounded in direct codebase audit (CONCERNS.md). Training pitfalls (8-12) are from community research and peer-reviewed MoE literature. Routing collapse and task interference are real documented phenomena. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **CMoE implementation availability:** CMoE is a research paper (Feb 2025); its Python implementation may not have a stable public release. Validate the repository and API before committing to CMoE over ToMoE. If CMoE has no public code, ToMoE (arxiv:2501.15316, tested on Qwen-2.5) is the validated fallback.
- **AWQ quantization for Qwen3MoE:** AutoAWQ and llm-compressor support for MoE models with routing tables needs explicit verification before the packaging phase begins. Quantization tools may silently drop routing weights.
- **Judge correlation circularity:** The judge correlation metric compares the fine-tuned model's scores against a Claude reference. If the same Claude model was used during training data generation, the metric is circular. Decide during requirements whether to use a different Claude model or a human-scored subset for the correlation evaluation.
- **DGX eval-toolbox specifics:** Whether the DGX Spark eval-toolbox container includes bigcode-evaluation-harness or requires manual lm-eval-harness setup is unconfirmed. Default to installing lm-eval-harness directly in the eval environment.
- **Load balancing loss during SFT:** Pretraining-scale MoE load balancing coefficients (typically 0.01) may be too high or too low for a narrow-domain SFT pass. The right coefficient for this use case is not documented in the tooling; needs empirical validation during training.

## Sources

### Primary (HIGH confidence)
- [NVIDIA DGX Spark Playbooks — Unsloth](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/unsloth) — dependency pin versions, Docker image, Flash Attention build requirements
- [Unsloth Qwen3 Fine-tune Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — LoRA configuration, MoE router notes, special token handling
- [Unsloth DGX Spark Guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth) — Docker container setup, Jupyter integration
- [HuggingFace Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) — architecture (36 layers, 8.2B params), transformers>=4.51.0, Apache 2.0 license
- [vLLM Qwen3 support](https://github.com/vllm-project/vllm/issues/17327) — Qwen3+Qwen3MoE support from v0.8.4, AWQ+Marlin 741 tok/s
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer) — LoraConfig parameters, modules_to_save
- [WordPress-Coding-Standards GitHub](https://github.com/WordPress/WordPress-Coding-Standards) — WPCS 3.x Composer-only install
- [State of WordPress Security 2025 — Patchstack](https://patchstack.com/whitepaper/state-of-wordpress-security-in-2025/) — vulnerability class rankings
- [Anthropic Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview) — current model IDs and pricing
- [Anthropic Rate Limits Documentation](https://platform.claude.com/docs/en/api/rate-limits) — RPM/TPM/OTPM limits
- Codebase audit: `.planning/codebase/CONCERNS.md` — pipeline bug analysis
- Project config: `config/judge_system.md`, `config/taxonomy.yaml` — feature spec ground truth
- [LLaMA-MoE GitHub (EMNLP 2024)](https://github.com/pjlab-sys4nlp/llama-moe) — upcycling methodology
- [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) — PHP evaluation tasks

### Secondary (MEDIUM confidence)
- [CMoE Paper (arxiv:2502.04416)](https://arxiv.org/abs/2502.04416) — training-free conversion methodology, S2A2E8 config, 5-min conversion; MEDIUM because research paper, not production library
- [ToMoE Paper (arxiv:2501.15316)](https://arxiv.org/abs/2501.15316) — alternative MoE conversion, tested on Qwen-2.5; MEDIUM, peer-reviewed but not a production release
- [Llama 3 Meets MoE (arXiv 2412.09952)](https://arxiv.org/abs/2412.09952) — upcycling patterns
- [Stabilizing MoE RL by Aligning Training and Inference Routers (arxiv:2510.11370)](https://arxiv.org/abs/2510.11370) — routing collapse research
- [LLM-as-a-Judge complete guide — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) — structured rubric design
- [Fine-tuning LLMs for secure code generation — Springer/ESE](https://link.springer.com/article/10.1007/s10664-026-10803-9) — peer-reviewed 2026, code security training
- [vLLM Quantization Performance — GPUStack](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/) — AWQ vs GGUF throughput comparison
- [Practical Tips for Finetuning LLMs Using LoRA — Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — LoRA configuration guidance

### Tertiary (LOW confidence)
- [DGX eval-toolbox specifics] — availability of bigcode-eval-harness in DGX container is unconfirmed; use lm-eval-harness as fallback
- [CMoE public implementation] — paper published Feb 2025; production-ready Python library status unverified; needs validation before Phase 2 begins

---
*Research completed: 2026-03-26*
*Ready for roadmap: yes*
