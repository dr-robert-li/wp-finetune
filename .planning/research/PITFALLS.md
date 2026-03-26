# Pitfalls Research

**Domain:** LLM fine-tuning pipeline — WordPress code, MoE conversion, LoRA SFT, API data pipelines
**Researched:** 2026-03-26
**Confidence:** HIGH (pitfalls 1-7 grounded in codebase analysis), MEDIUM (pitfalls 8-12 from community research)

---

## Critical Pitfalls

### Pitfall 1: Training Data Poisoned by Silent Parse Failures

**What goes wrong:**
The judge response parser falls back to `{"verdict": "FAIL"}` stub objects when Claude returns JSON in an unexpected format. These stubs propagate silently through the pipeline and can produce two corrupting effects: high-quality functions are discarded (filter is too aggressive), or the stub itself enters the training set as a malformed judge example.

**Why it happens:**
The brittle `split("```json")` / `split("```")` pattern in `phase1_judge.py`, `phase2_judge.py`, and `phase2_judge_dataset.py` has no fallback extraction strategy and no logging when it fails. Claude occasionally varies its markdown wrapper (adds a language hint, uses plain backticks, or omits the fence entirely) — each variation silently breaks the parser.

**How to avoid:**
Extract parsing to a shared utility that tries multiple extraction strategies in order: JSON fence with language hint, bare fence, regex for outermost `{...}`, raw `json.loads`. If all fail, log the raw response with the function ID and hard-reject the example (do not substitute a stub). Add a parse failure counter; abort if >2% of responses fail.

**Warning signs:**
- Judge pass rate unexpectedly low despite apparently good code (stubs defaulting to FAIL)
- Training examples with `{"verdict": "FAIL", "scores": {}}` but no reasoning text
- Dataset shrinks more than expected during Phase 1

**Phase to address:** Data Pipeline execution (Phase 1 fixes before first full run)

---

### Pitfall 2: API Rate Limit Cascade Crashes Multi-Hour Pipelines

**What goes wrong:**
`phase2_judge_dataset.py` has no rate limiting between scoring calls. In a batch of 4,000 examples it fires requests as fast as the loop runs. Anthropic enforces limits at the organization level across RPM, input TPM, and output TPM. Hitting any one of them returns 429. Without retry logic, the script crashes mid-pipeline after potentially hours of compute and cost.

**Why it happens:**
Rate limiting was added to most scripts but missed in `phase2_judge_dataset.py`. Phase 3 uses both Sonnet and Opus without per-model rate tracking — Opus has lower default TPM limits, so mixing models without separate counters means the Opus budget can silently exhaust while Sonnet headroom remains.

**How to avoid:**
- Add `time.sleep(REQUEST_INTERVAL)` after every API call in `phase2_judge_dataset.py`
- Implement exponential backoff with jitter for 429 responses, reading the `retry-after` header (Anthropic includes exact wait time)
- Track separate rate counters for Sonnet vs. Opus (different TPM allocations)
- For Phase 2 batch generation, use the Anthropic Batch API (50% cost reduction, async, no RPM pressure) for non-interactive generation tasks

**Warning signs:**
- First 429 appears very quickly after starting `phase2_judge_dataset.py`
- Errors at irregular intervals (suggests TPM not RPM — triggered by large code blocks)
- Opus calls failing before Sonnet calls in Phase 3

**Phase to address:** Data Pipeline execution (before Phase 2/3 runs)

---

### Pitfall 3: Pipeline Crashes Leave No Resumption Point

**What goes wrong:**
`phase1_judge.py`, `phase2_generate.py`, and `phase3_cot.py` process their full datasets in a single pass with no checkpointing. A crash at example 3,000/5,000 requires a full restart. Given API costs and Claude model versioning, a restart does not reproduce identical outputs — the pipeline loses both time and determinism.

**Why it happens:**
Checkpointing is treated as a nice-to-have. On small datasets it is irrelevant; at 5,000–13,500 examples it becomes critical. There is also a secondary issue: without checkpointing it is impossible to run a cheap "sample 100 examples" test to validate the pipeline end-to-end before committing to the full run.

**How to avoid:**
Write a checkpoint JSON every 100 examples: `{phase, last_completed_index, timestamp}`. At startup, check for an existing checkpoint and resume from `last_completed_index + 1`. Add a `--sample N` flag to process only the first N examples (critical for pre-run validation). Add a `--start-from N` flag as a manual override.

**Warning signs:**
- Script has been running for >2 hours with no intermediate output files
- Any crash requires a cost re-estimation conversation ("do we re-run from scratch?")
- Developer hesitates to test changes because a full run is required to verify

**Phase to address:** Data Pipeline execution (before first full run, highest priority fix)

---

### Pitfall 4: PHPCS Silent Degradation Pollutes Judge Training Data

**What goes wrong:**
`phase2_mutate.py` catches `FileNotFoundError` from PHPCS and returns `True` (mutation accepted as detectable) when PHPCS is unavailable. This means undetectable mutations — code that is wrong but PHPCS cannot flag — enter the judge training set as examples labeled "this defect is detectable." The model trains on a lie.

**Why it happens:**
Graceful degradation was chosen over fail-fast to make the script runnable without PHPCS. In development this is useful; in production data generation it silently corrupts the dataset.

**How to avoid:**
Add a `verify_phpcs_available()` pre-flight check at script startup. Exit with a clear error message if PHPCS is missing. Never silently accept mutations. If running in an environment where PHPCS is temporarily unavailable, fail loudly and block the pipeline until it is restored.

**Warning signs:**
- Script completes much faster than expected (PHPCS validation is being skipped)
- Mutation acceptance rate is 100% or near-100% (all mutations "verified" instantly)
- No PHPCS output in logs

**Phase to address:** Data Pipeline execution (pre-flight validation before any mutation run)

---

### Pitfall 5: Special Token Embeddings Destabilize the Fine-Tuned Model

**What goes wrong:**
Adding `<wp_gen>` and `<wp_judge>` special tokens resizes the embedding matrix. LoRA by default freezes the embedding layer and LM head. The new token rows are randomly initialized and receive no gradient updates during training. The model never learns what these tokens mean and either ignores them or produces garbage when they appear at inference.

**Why it happens:**
LoRA wraps linear layers with low-rank adapters but does not touch `embed_tokens` or `lm_head` by default. Randomly initialized embedding rows for new special tokens are treated as noise — routing never converges on them.

**How to avoid:**
In Unsloth, set `modules_to_save = ["embed_tokens", "lm_head"]` in the LoRA config. This fully trains (not LoRA-adapts) the embedding and output projection layers for the new tokens while keeping all other layers frozen. Verify before training: log the token IDs for `<wp_gen>` and `<wp_judge>` and confirm their embedding norms are non-zero after a few training steps.

**Warning signs:**
- Model generates same output regardless of which task token is used
- Task token embeddings have near-zero L2 norm after training
- Routing to specialized expert pathways is random rather than task-correlated

**Phase to address:** Tokenizer extension and LoRA setup (before fine-tuning starts)

---

### Pitfall 6: MoE Routing Collapse — All Tokens Route to One Expert

**What goes wrong:**
After dense-to-MoE conversion, the router can collapse: it learns to always send all tokens to one or two experts regardless of input. The remaining experts receive no gradient signal and become dead weights. The model behaves identically to a dense model (or worse, due to wasted parameters) and the specialized `<wp_gen>`/`<wp_judge>` routing never materializes.

**Why it happens:**
Without explicit load balancing loss during fine-tuning, the top-2 router settles on the path of least resistance — always routing to the experts that happen to initialize with the best activations. This is especially likely when fine-tuning on a narrow domain (WordPress PHP) since the input distribution is homogeneous.

**How to avoid:**
Keep the auxiliary load balancing loss active during fine-tuning (reduce its coefficient vs. pretraining but do not zero it). Monitor per-expert token routing counts at each checkpoint: the distribution should not have any single expert above ~30% of tokens in a top-2 routing scheme. If collapse is detected, increase the load balancing coefficient before resuming.

**Warning signs:**
- One expert handling >50% of all tokens in training logs
- Loss decreases normally but eval performance is identical to the unmodified dense model
- Expert activation entropy is low and decreasing across training steps

**Phase to address:** MoE conversion and LoRA fine-tuning setup

---

### Pitfall 7: Training-Evaluation Metric Overfitting (PHPCS Pass Rate)

**What goes wrong:**
Using PHPCS pass rate as the primary evaluation metric incentivizes the model to learn PHPCS's specific rule set rather than genuine WordPress coding quality. A model can achieve >95% PHPCS pass rate by producing syntactically correct, style-compliant PHP that is logically broken, insecure, or meaningless. The metric reports success while quality is poor.

**Why it happens:**
PHPCS is deterministic and easy to automate — it gives a clean scalar. Human judgment and semantic correctness are expensive to measure. Projects narrow to the cheap metric.

**How to avoid:**
Treat PHPCS pass rate as a necessary but not sufficient gating metric. Pair it with:
1. A held-out set of examples scored by the Claude judge (not the same judge used during training)
2. Manual review of 50 randomly sampled generated functions per eval cycle
3. A "judge correlation" metric (>0.85 target already in PROJECT.md) that checks whether the fine-tuned model's judgments agree with Sonnet/Opus on unseen examples

Do not report eval results using only PHPCS.

**Warning signs:**
- PHPCS pass rate climbs to >95% but generated code looks formulaic or repetitive
- Judge correlation metric is not improving even as PHPCS rate improves
- Generated functions pass linting but contain copy-paste patterns with no logic

**Phase to address:** Evaluation phase (define evaluation suite before training begins, not after)

---

### Pitfall 8: Multi-Task Task Interference — Generation vs. Judgment in Same Training Run

**What goes wrong:**
Training on `<wp_gen>` (generate PHP code) and `<wp_judge>` (output structured JSON scores) in a shared SFT run causes task interference. The generation task rewards verbose PHP; the judgment task rewards compact JSON. Gradients from each task push the model in conflicting directions, producing a model that is mediocre at both rather than excellent at either.

**Why it happens:**
Multi-task SFT assumes that tasks are complementary. When output formats differ radically (code vs. JSON) and tasks use distinct reasoning modes (synthesis vs. evaluation), they often interfere. This is documented in MoE literature as a particular risk when the routing mechanism has not yet converged.

**How to avoid:**
Structure the training curriculum deliberately. Option A: interleave tasks at a 50/50 ratio from the start (current plan) — acceptable but monitor judge-correlation and PHPCS metrics separately. Option B: warm-up phase where only one task is trained, then introduce the second. The MoE routing benefits from seeing both tasks early, so Option A is reasonable, but add per-task loss tracking to detect interference early.

**Warning signs:**
- Training loss for one task decreases while loss for the other plateaus or increases
- Model produces JSON-like fragments when asked to generate code, or PHP fragments in judge outputs
- Per-task eval metrics diverge significantly from early to late training

**Phase to address:** Fine-tuning curriculum design

---

### Pitfall 9: Train/Validation Leakage from Example Reuse

**What goes wrong:**
The same PHP function appears in both training and validation sets. Evaluation reports high scores but the model has simply memorized training examples. This is especially likely given the 50/50 judge/gen split — a function used to train the generation pathway could also appear as the "bad example" in a contrastive judge pair.

**Why it happens:**
`export_dataset.py` performs an 80/10/10 split but does not deduplicate across splits, and does not track whether a code example was already used in a different training example type (generation vs. judgment). CONCERNS.md explicitly flags that judge training may consume 40% of all passed examples, meaning overlap is near-certain.

**How to avoid:**
Implement content-based deduplication before export: hash the raw PHP code of each example. If the same hash appears in multiple examples (generation + judge pair), keep one copy and assign it deterministically to train. Tag every example with `used_for: [list]` in metadata. Validate in `export_dataset.py` that no PHP code block appears in both train and validation splits.

**Warning signs:**
- Validation PHPCS pass rate is unusually high from the first training checkpoint
- Validation loss is lower than training loss (classic leakage signature)
- Small validation set size relative to total dataset

**Phase to address:** Dataset export and validation

---

### Pitfall 10: API Cost Shock from Untracked Opus Usage

**What goes wrong:**
`phase3_cot.py` uses Claude Opus for all CoT generation. Opus is ~3x Sonnet's cost. The cost estimation in `phase2_generate.py` uses a hardcoded `$5M` rate (Sonnet pricing) and does not account for Opus. A developer runs Phase 3 expecting a $50 bill and receives a $300–500 bill.

**Why it happens:**
Cost estimation was added for Phase 2 but not updated when Phase 3 added Opus. The CoT tag matching logic also applies CoT to long functions (>threshold lines) in addition to tagged functions — the total Opus call count is higher than expected.

**How to avoid:**
Add a dry-run cost estimator before Phase 3 starts: count qualifying CoT examples, estimate token usage (average tokens per code block × 2 for prompt + completion), multiply by Opus pricing, and display a warning that requires explicit confirmation. Add a `--max-cot N` flag to cap CoT generation. Use Sonnet for shorter functions (< 50 lines) where CoT reasoning depth is less valuable.

**Warning signs:**
- Phase 3 takes significantly longer than Phase 2 (Opus latency is higher)
- API cost dashboard shows a spike during Phase 3
- No pre-run cost estimate displayed in terminal output

**Phase to address:** Phase 3 data pipeline execution (before first full Phase 3 run)

---

### Pitfall 11: GGUF in vLLM Has Poor Throughput — Wrong Deployment Target

**What goes wrong:**
The plan deploys GGUF for Ollama and AWQ for vLLM. This is correct. However, if GGUF is tested in vLLM for convenience, it shows ~93 tok/s vs. 741 tok/s for Marlin-AWQ. The deployment is declared "too slow" based on GGUF performance, and the correct format (AWQ with Marlin kernel) is never tested.

**Why it happens:**
GGUF is the most familiar quantization format and easy to generate with llama.cpp tools. Developers reach for it first. vLLM's GGUF support is explicitly documented as "highly experimental and under-optimized."

**How to avoid:**
Use the correct format per backend:
- **Ollama:** GGUF (Q4_K_M or Q5_K_M for quality/speed balance)
- **vLLM:** AWQ with Marlin kernel (`--quantization marlin`)
Never benchmark GGUF in vLLM as a proxy for vLLM performance.

After MoE fine-tuning, the quantization workflow must account for MoE architecture: not all quantization tools support MoE weight routing tables. Verify Unsloth's export tools support Qwen3 MoE architecture before assuming standard export works.

**Warning signs:**
- vLLM serving throughput below 200 tok/s on DGX Spark (should be 700+ with Marlin)
- `--quantization gguf` in vLLM launch command
- Export step completed without verifying MoE routing weights are preserved in quantized format

**Phase to address:** Model deployment and packaging

---

### Pitfall 12: Score Inflation from N/A Dimensions in Judge Training Data

**What goes wrong:**
Backend PHP functions that produce no HTML and have no user-facing strings receive automatic N/A (score 10) for accessibility and i18n dimensions. This inflates their average score and pushes them above the quality floor (≥7) even when other dimensions are weak. The judge model trains on an association: "pure backend functions = high quality," which is false.

**Why it happens:**
The N/A handling was added to avoid penalizing backend code for inapplicable criteria. The intent was correct but the implementation (score 10) corrupts the scoring distribution rather than excluding the dimension from the average.

**How to avoid:**
Change N/A handling: track inapplicable dimensions separately with a flag, exclude them from the average calculation entirely (proportional weighting over applicable dimensions only). Do not substitute 10. Audit the judge training dataset distribution: if >60% of examples are "backend-only" functions, the dataset is biased and generation training data needs more frontend/template examples.

**Warning signs:**
- High percentage of judge training examples have accessibility or i18n dimensions at exactly 10
- Average judge score across all examples is unusually high (>8.5)
- Model generates backend-heavy code disproportionately in generation tasks

**Phase to address:** Data pipeline validation, before judge dataset export

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded model IDs in all scripts | Simple, no config file needed | Entire pipeline breaks silently when model is deprecated; cannot systematically upgrade | Never — move to `config/models.yaml` before first full run |
| Graceful PHPCS degradation (return True) | Script runs without PHPCS installed | Undetectable mutations enter training data; judge learns wrong signal | Never for production data runs |
| No progress checkpointing | Simpler code | Re-run entire pipeline on any failure; can't test cheaply | Acceptable for scripts under 60 seconds; unacceptable at multi-hour scale |
| Fixed `time.sleep(REQUEST_INTERVAL)` without retry | Simple to implement | Pipeline dies on first 429 instead of recovering | Acceptable for initial testing only; add backoff before production runs |
| Batch API skipped in favor of realtime | Simpler response handling | 2x cost on all generation/judge calls | Acceptable for <500 examples; unacceptable at 13,500 example scale |
| No pre-run environment validation | Faster startup | Fails hours in on first PHPCS/API call; wasted compute | Never — add `verify_setup.py` before first full run |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Anthropic API rate limits | Tracking only RPM, ignoring TPM | Monitor all three limits (RPM, ITPM, OTPM); large code blocks exhaust TPM well before RPM |
| Anthropic API rate limits | Using same rate limit for Sonnet and Opus | Track separate counters; Opus has lower default TPM limits |
| Anthropic Batch API | Not using it for offline pipeline steps | All Phase 2 generation and Phase 3 CoT qualify for Batch API (50% cost, no RPM pressure) |
| PHPCS via subprocess | Catching all exceptions and continuing | Catch `FileNotFoundError` specifically and fail hard; silence hides tool misconfiguration |
| Claude response parsing | Brittle markdown fence splitting | Try multiple extraction strategies; log raw response on failure; never substitute stub objects |
| Unsloth LoRA + special tokens | Default LoRA config freezes embed_tokens | Explicitly add `modules_to_save=["embed_tokens","lm_head"]` before training |
| vLLM + GGUF | Testing GGUF performance in vLLM | GGUF is experimental in vLLM; use AWQ with Marlin kernel for production throughput |
| GGUF export + MoE | Using standard GGUF export tools | Verify tool supports MoE routing tables before assuming export is correct |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Sequential generation with `time.sleep` | Phase 2 takes 150+ seconds for 100 gaps | Use async requests or Claude Batch API | Immediately at >50 gaps |
| Inconsistent code truncation (3000 vs 4000 chars) | Long functions scored on truncated code; security-critical code cut off | Standardize at 4000 chars; reject functions >5000 chars at extraction | At any long function >3000 chars |
| No `--sample N` flag for pipeline testing | Full pipeline run required to test any change | Add sample flag before first full run | Every development iteration |
| Opus for all CoT regardless of function length | Phase 3 costs 3x more than estimated | Use Sonnet for <50-line functions; Opus for complex/long functions only | At full 13,500 example dataset |
| Single-threaded git clone in Phase 1 | Cloning 10+ repos takes minutes serially | Parallelize with `asyncio` or thread pool | At >10 repos in `repos.yaml` |

---

## Data Quality Traps

| Trap | Risk | Prevention |
|------|------|------------|
| `phase2_judge_dataset.py` consuming 40% of passed examples | Generation training starved of diverse real examples; judge set lacks nuance | Deduplicate across training splits; cap judge examples at 25% of total passed examples |
| Auto-tagging Core code without validation | SQL vulnerability in Core becomes a "reference implementation" style anchor | Sample Core functions through the full judge system; do not trust auto-tagging for security dimensions |
| Task type inferred from metadata with silent default to "gen" | Judge examples mislabeled as generation examples | Require explicit `metadata.task_type`; fail export if absent; never default |
| Taxonomy coverage not enforced post-generation | Final dataset misses concept coverage despite planning | Run coverage check in `export_dataset.py`; fail if any tag below minimum threshold |
| No PHP validity check on exported code examples | Syntactically broken PHP in training set | Parse 100% of code blocks in export validation using `php -l` (lint only, no execution) |

---

## "Looks Done But Isn't" Checklist

Before declaring any phase complete:

- [ ] **Phase 1 judge:** Verify parse failure rate is <2% — check log for `PARSE_FAIL` entries, not just final count
- [ ] **Phase 2 mutations:** Confirm PHPCS was active during full run — check acceptance rate (should not be 100%)
- [ ] **Phase 2 judge dataset:** Confirm N/A scoring is tracked separately, not inflating averages to 10
- [ ] **Phase 3 CoT:** Confirm actual Opus call count matches estimate — inspect billing console, not just script logs
- [ ] **Tokenizer extension:** Confirm `<wp_gen>` and `<wp_judge>` token IDs are non-zero in embedding matrix after first training checkpoint
- [ ] **MoE fine-tuning:** Confirm per-expert routing distribution — no single expert above 30% of tokens
- [ ] **Dataset export:** Confirm no PHP code block appears in both train and validation splits
- [ ] **Deployment:** Confirm GGUF → Ollama and AWQ+Marlin → vLLM, not GGUF in both
- [ ] **Evaluation:** Confirm evaluation uses both PHPCS and judge correlation, not PHPCS alone

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Parse failures polluted training data | HIGH | Re-run affected phase from checkpoint after fixing parser; audit and remove stub-contaminated examples from output JSONL |
| Pipeline crashed mid-run, no checkpoint | HIGH | Implement checkpoint then restart; alternatively, manually identify last successful output and pass `--start-from N` |
| API rate limit exhausted mid-run | LOW | Wait for reset window (use `retry-after` header value); add exponential backoff; resume from checkpoint |
| Routing collapse detected post-training | HIGH | Resume training with higher load balancing coefficient; may require full re-run if collapse is severe |
| Special token embeddings untrained | HIGH | Re-run fine-tuning with corrected `modules_to_save`; cannot fix post-hoc without retraining |
| Train/val leakage detected | MEDIUM | Re-export dataset with deduplication; re-evaluate; check if checkpoint metrics were inflated |
| GGUF exported but MoE routing weights missing | MEDIUM | Re-export with correct tool; verify routing table presence in exported weights before serving |
| API cost shock from Opus | LOW | No data recovery needed; add cost estimator and `--max-cot` flag before next run |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent parse failures | Data Pipeline — Phase 1 pre-run fixes | Parse failure rate <2% in Phase 1 logs |
| API rate limit cascade | Data Pipeline — before Phase 2/3 runs | 429 errors handled with backoff; no mid-run crash |
| No checkpoint/resume | Data Pipeline — before first full run | `--sample 100` test completes and produces valid output |
| PHPCS silent degradation | Data Pipeline — pre-flight validation | Pre-flight check exits with error if PHPCS unavailable |
| Special token embedding not trained | MoE conversion + tokenizer extension | Token embedding norms >0 after first 100 training steps |
| MoE routing collapse | Fine-tuning setup | Per-expert token distribution logged; no expert >30% |
| Evaluation metric overfitting | Evaluation design (before training) | Both PHPCS and judge correlation tracked independently |
| Multi-task interference | Fine-tuning curriculum | Per-task loss tracked; neither task loss stagnates |
| Train/val leakage | Dataset export | No shared PHP code hash across train and val splits |
| Opus cost shock | Phase 3 pre-run | Dry-run estimate displayed and confirmed before execution |
| GGUF in vLLM | Deployment packaging | AWQ+Marlin throughput >700 tok/s on vLLM; GGUF only in Ollama |
| N/A score inflation | Judge dataset export | No dimension scored exactly 10 via N/A; proportional averaging confirmed |

---

## Sources

- Codebase analysis: `/home/robert_li/Desktop/projects/wp-finetune/.planning/codebase/CONCERNS.md` (HIGH confidence — direct code audit)
- [Anthropic Rate Limits Documentation](https://platform.claude.com/docs/en/api/rate-limits) (HIGH confidence — official)
- [Anthropic 429 Error Guidance](https://support.claude.com/en/articles/8114527-i-m-encountering-429-errors-and-i-m-worried-my-rate-limit-is-too-low-what-should-i-do) (HIGH confidence — official)
- [Practical Tips for Finetuning LLMs Using LoRA — Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) (MEDIUM confidence — expert practitioner)
- [How to Add Special Tokens to LLMs Safely](https://langcopilot.com/posts/2025-09-23-how-to-add-special-tokens-llms) (MEDIUM confidence — community, verified against Unsloth docs)
- [Unsloth Qwen3 Fine-tune Documentation](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune) (HIGH confidence — official tool docs)
- [ToMoE: Converting Dense LLMs to MoE via Dynamic Structural Pruning](https://arxiv.org/abs/2501.15316) (HIGH confidence — peer-reviewed 2025)
- [Stabilizing MoE RL by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370) (MEDIUM confidence — routing collapse research)
- [Towards Catastrophic Forgetting-Free Multi-Domain MoE](https://aclanthology.org/2025.emnlp-main.932.pdf) (MEDIUM confidence — EMNLP 2025)
- [vLLM GGUF Documentation](https://docs.vllm.ai/en/stable/features/quantization/gguf/) (HIGH confidence — official, notes "experimental")
- [vLLM Quantization Performance — GPUStack](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/) (MEDIUM confidence — benchmark data)
- [Which Quantization Method is Right for You? — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/which-quantization-method-is-right) (MEDIUM confidence — practitioner comparison)
- [Qwen3 not stopping generation after LoRA fine-tuning — LlamaFactory Issue #7943](https://github.com/hiyouga/LlamaFactory/issues/7943) (MEDIUM confidence — real-world issue report)

---

*Pitfalls research for: WordPress code fine-tuning MoE — execution and training phase*
*Researched: 2026-03-26*
