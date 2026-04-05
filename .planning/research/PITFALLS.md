# Pitfalls Research

**Domain:** LLM fine-tuning pipeline — WordPress code, MoE conversion, LoRA SFT, API data pipelines
**Researched:** 2026-03-26 (v1.0); 2026-04-04 appended (v1.2 reasoning fine-tune)
**Confidence:** HIGH (pitfalls 1-7 grounded in codebase analysis), MEDIUM (pitfalls 8-12 from community research), MEDIUM–HIGH (pitfalls 13-21 reasoning/continued-training section)

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

## v1.2 Reasoning Fine-Tune Pitfalls

*These pitfalls apply specifically to the v1.2 milestone: adding deep judge CoT reasoning and critique-then-fix capabilities to an already-trained Qwen3-30B-A3B adapter (60:40 ratio, 43hr training, loss 0.29). The base adapter already has MoE routing shaped by phase 1 SFT and the structured JSON judge format baked in.*

---

### Pitfall 13: Continued Training Learning Rate Destroys the Existing Adapter

**What goes wrong:**
Resuming LoRA training with the same learning rate used in the initial run (`2e-4` from train_config_60_40.yaml) causes an immediate loss spike. The optimizer state is reset on checkpoint load (confirmed by Unsloth docs), so the cosine scheduler starts from scratch at a high LR against adapter weights that have already converged. The first few hundred steps overwrite useful weights before the LR decays to a safe range.

**Why it happens:**
Initial training uses a 5% warmup ratio to ramp from 0 to `2e-4`. When continued training starts fresh, the optimizer again ramps from 0 to `2e-4` — but the adapter is no longer freshly initialized. The weights are already near a local minimum, and a full-LR update at that point is too large, causing destructive gradient steps that unlearn what phase 1 trained.

**How to avoid:**
Use a starting LR of 2–5x lower than the initial run for continued training. A practical rule: if phase 1 used `2e-4`, use `4e-5` to `1e-4` for phase 2. Keep the warmup ratio at 5% or lower. Use a cosine schedule with the same total-steps calculation but clamp peak LR to the reduced value. Confirm that phase 1 performance metrics (judge Spearman, PHPCS pass rate) on the phase 1 validation set do not degrade after the first 200 continued-training steps.

**Warning signs:**
- Training loss spikes above phase 1 final loss in the first 100 steps
- Judge Spearman correlation drops measurably at first checkpoint vs. pre-training eval
- Gradient norms spike to >10 in early steps (they should be ~1–3 in stable training)

**Phase to address:** v1.2 training setup (before the continued training run begins)

---

### Pitfall 14: Format Collapse — Reasoning Chains Overwrite JSON Structure

**What goes wrong:**
The existing adapter outputs valid structured JSON (`{"verdict": ..., "scores": {...}, ...}`) for judge tasks. After continued training on reasoning data where the target output is a long chain-of-thought narrative followed by scores, the model loses the JSON format. It starts producing free-text reasoning with embedded score mentions but without parseable JSON structure. The judge pipeline breaks entirely — nothing can extract scores from the output.

**Why it happens:**
The model encounters a new training signal: long reasoning traces where the structured JSON is a small tail at the end of a large free-text block. With enough examples of this format, the attention heads that anchor the JSON structure get overwritten by weights that expect to produce natural-language prose. The effect is accelerated by the fact that reasoning examples are much longer (more tokens) than the original compact JSON examples, giving them disproportionate gradient influence per batch.

**How to avoid:**
Two mitigations work together:
1. Keep the original structured judge examples (no reasoning) in the continued training mix at a ratio of at least 30%. This prevents the model from forgetting the compact format. Use an approximate ratio of 30% original judge examples / 40% deep CoT examples / 30% critique-then-fix examples.
2. Enforce a consistent output template in the reasoning training data: the JSON block must always appear at the end of the reasoning trace, inside a clearly delimited section (e.g., `<judge_output>...</judge_output>` tags). Train the model to always end reasoning with the canonical JSON structure.

**Warning signs:**
- Inference samples from the continued-training model produce well-reasoned text but no parseable JSON
- The `}` final bracket of the judge JSON is truncated or missing
- Parse failure rate on judge examples climbs above 5% at any checkpoint

**Phase to address:** v1.2 data generation (reasoning template design) and training data mixing

---

### Pitfall 15: Reasoning Length Explosion at Inference

**What goes wrong:**
After reasoning fine-tuning, the model generates correct scores but produces 1,500–4,000 token reasoning traces for every judge call. The existing judge pipeline is designed for compact outputs (~150 tokens). Inference latency triples and memory pressure increases significantly during serving on DGX Spark. Batch scoring of 100 functions that previously took 2 minutes now takes 12+ minutes.

**Why it happens:**
SFT on reasoning data teaches the model to reason fully for every query — it does not learn to calibrate reasoning depth to query difficulty. The model has seen many examples of full-length reasoning chains and none of compact reasoning, so it defaults to maximum length even for trivial functions. This is the "overthinking" pattern documented extensively in 2025 reasoning research.

**How to avoid:**
Include a distribution of reasoning lengths in training data. For simple functions (PHPCS pass, few security concerns), include short reasoning traces (3–5 sentences per dimension). For complex or failing functions, use full-length traces. Target a 40/60 split of short-form vs. long-form reasoning examples. Add a system prompt hint at inference: "Use concise reasoning proportional to code complexity." Enforce an output length cap in the inference configuration (max_new_tokens for judge calls should be set to a reasonable ceiling, e.g., 800 tokens for compact mode, 2000 for deep mode).

**Warning signs:**
- First 10 inference samples from the reasoning model all exceed 1,000 tokens
- Batch scoring time increases by >3x vs. the phase 1 model
- Reasoning traces for trivially simple functions are as long as traces for complex ones

**Phase to address:** v1.2 data generation (include short-form reasoning examples) and inference configuration

---

### Pitfall 16: MoE Routing Distribution Shifts After Reasoning Data Injection

**What goes wrong:**
The phase 1 SFT run shaped the Qwen3-30B-A3B routing distribution around WordPress PHP patterns. Reasoning data has a fundamentally different token distribution: long natural-language chains, analytical vocabulary, hedging phrases, dimension-by-dimension structured prose. Adding this data shifts which experts handle judge tokens. The routing that v2.0 MoE-Sieve relies on (profiling for hot expert selection) was measured on the phase 1 adapter. If v1.2 significantly shifts routing, the MoE-Sieve profiling must be re-run on the v1.2 adapter, not reused from v1.0 profiling data.

**Why it happens:**
Qwen3's top-8 routing from 128 experts means ~10% of experts activate per token. Reasoning data introduces a new token distribution that activates different experts. With SFT on reasoning data, the router's input features shift because the attention hidden states (router inputs) for reasoning tokens differ from PHP code tokens. After fine-tuning, the routing is no longer what was measured before.

**How to avoid:**
Do not treat v1.0 routing profiles as valid for v1.2. After v1.2 training completes, run a fresh routing profiling pass before any MoE-Sieve work. Explicitly freeze the router during v1.2 training (Qwen3 MoE fine-tuning disables router layer fine-tuning by default — confirm this is active in Unsloth config). Freezing the router prevents routing shifts while still letting attention and FFN adapters learn the reasoning patterns.

**Warning signs:**
- Router layer gradients are non-zero in training logs (router is being fine-tuned when it should not be)
- Per-expert token distribution at v1.2 checkpoint differs by >15% from v1.0 distribution on the same validation set
- `<wp_judge>` token routing affinity changes significantly between v1.0 and v1.2 adapters

**Phase to address:** v1.2 training config (freeze router before run) and v2.0 planning (require fresh profiling after v1.2)

---

### Pitfall 17: Critique-Then-Fix Format — Model Learns to Skip the Fix

**What goes wrong:**
The critique-then-fix format has two parts: a structured critique identifying issues (what/why/severity per dimension), followed by a corrected version of the code. After training, the model reliably produces the critique section but frequently truncates or skips the corrected code block. Prompts that ask for `<corrected_code>` get the analysis but not the fix. The critique-only behavior is a shortcut: it minimizes output length and closely mirrors the existing judge training format.

**Why it happens:**
The model has 143K judge training examples that produce critique-style analysis without any code. The critique-then-fix examples are a new, smaller dataset. When gradients compete, the existing judge behavior (critique only) is a stronger attractor because it has far more training support. The fix section requires generating PHP code after a long reasoning trace — a distribution the model has not been trained on end-to-end.

**How to avoid:**
Increase the weight of the fix section in the loss calculation. Use response-masking to compute loss only on the corrected code section (not the critique) for a subset of training examples, forcing the model to attend to the fix quality independently. Structure the training data so the correct code always appears in a clearly delimited block (e.g., `<corrected_code>...(PHP)...</corrected_code>`) that the model can learn to produce as a distinct unit. Include examples that are "easy fixes" (single-line security patch) and "complex fixes" (refactor entire function) so the model learns to produce fixes of varying scope.

**Warning signs:**
- Inference samples contain `<critique>` block but empty `<corrected_code>` block
- Corrected code in training data examples is shorter than critique by a large margin (check token counts — if critique is consistently 5x longer, loss is dominated by critique tokens)
- Model produces critique-only output even when explicitly prompted to "also provide the corrected version"

**Phase to address:** v1.2 data generation (critique-then-fix template) and training loss configuration

---

### Pitfall 18: Score Calibration Drift — Reasoning Changes What Scores Mean

**What goes wrong:**
The phase 1 adapter was calibrated to produce dimension scores correlated with the Claude judge (target Spearman 0.85). After reasoning fine-tuning, the model's scores shift systematically. Scores may inflate (the model justifies higher scores after reasoning through the code more thoroughly), or deflate (the model surfaces issues it previously missed). The Spearman correlation with the ground-truth Claude judge may actually improve — but the score absolute values drift, breaking downstream systems that threshold on specific score values (e.g., the security auto-fail at <5).

**Why it happens:**
Reasoning chains cause the model to reconsider its initial impression of code quality. This is the "reflection changing verdict" effect: models that reason about a problem often produce different outputs than models that answer immediately. This is desirable for judgment quality but breaks metric continuity if calibration assumptions carry over from phase 1.

**How to avoid:**
Re-run the full evaluation suite on the v1.2 adapter using the same held-out validation set used for v1.0. Compare absolute score distributions (histogram), not just Spearman correlation. Flag any dimension where the mean score shifts by more than 0.5 points vs. v1.0 on the same examples. If security dimension scores shift downward significantly, re-check the security auto-fail threshold — it may need to be lowered from <5 if the model is now more conservative.

**Warning signs:**
- Security dimension mean score drops by >0.5 points vs. v1.0 on the same validation set
- Any dimension's score distribution shifts bimodally (model now produces extreme scores, fewer mid-range scores)
- PASS/FAIL classification accuracy vs. ground-truth changes significantly despite improved Spearman

**Phase to address:** v1.2 evaluation (mandatory recalibration check before declaring v1.2 complete)

---

### Pitfall 19: Generation Task Regression From Reasoning Data Contamination

**What goes wrong:**
The `<wp_gen>` pathway regresses after reasoning fine-tuning. Generated code starts including reasoning traces in the output — the model begins prefixing PHP code with analytical text because it has seen many examples where text precedes the important output. Alternatively, generated PHP code quality (PHPCS pass rate, security scores) drops because the training mix reduced the density of clean generation examples relative to the total dataset.

**Why it happens:**
Continued training on judge-only reasoning data shifts the overall distribution. The generation task was trained on compact PHP code examples. If the continued training mix is 100% judge+reasoning data with no generation examples, the model's generation capability degrades through forgetting. The effect is similar to the catastrophic forgetting between SFT and RL documented in literature: each new training phase overwrites the previous task if not explicitly included.

**How to avoid:**
Include a replay buffer of generation examples in the v1.2 training mix. A 20–30% inclusion of original `<wp_gen>` examples from the phase 1 training set is sufficient to prevent regression. Do not train v1.2 on judge-only data. Run the full PHPCS + judge correlation evaluation on generation samples (not just judge samples) after v1.2 training and compare to v1.0 baseline.

**Warning signs:**
- v1.2 model generates PHP code prefixed with explanatory text when `<wp_gen>` is used
- PHPCS pass rate on generated functions drops by >3 percentage points vs. v1.0
- Security dimension score on generated functions drops below the auto-fail threshold more frequently than v1.0

**Phase to address:** v1.2 training data assembly (include generation replay examples before the run)

---

### Pitfall 20: Reasoning Data Quality Circular Dependency

**What goes wrong:**
The reasoning data for v1.2 is generated by Claude Code agents using the existing judge system. If the agent-generated reasoning chains contain errors (wrong dimension analysis, false critical failure claims, incorrect fix suggestions), those errors become training signal. The model learns to reproduce the errors confidently, since they appear in the training data with no quality signal differentiating them from correct reasoning.

**Why it happens:**
The pipeline for generating reasoning data uses the same agents and prompts as the original data pipeline. At scale (regenerating judge training examples with reasoning chains means tens of thousands of examples), spot-checking is impractical. Errors in Claude's reasoning about PHP code are systematic, not random — for example, Claude may consistently misidentify a `sanitize_text_field()` usage as insufficient sanitization, and that systematic error propagates to all examples involving that pattern.

**How to avoid:**
Do not generate reasoning data in bulk without validation sampling. Sample 1% of generated reasoning examples (minimum 50) and manually review for correctness before proceeding. Focus review on the security dimension (highest stakes) and SQL safety dimension (most rule-specific). Add a "reasoning self-consistency" check: for any function where the reasoning chain concludes differently from the original phase 1 verdict, flag it for manual review before including in training. Use stricter prompts that require the agent to cite specific line numbers and pattern names when flagging issues.

**Warning signs:**
- Reasoning chains for PASS functions frequently mention "potential issues" that the phase 1 judge found none of (false positives proliferating)
- The reasoning for FAIL functions cites different critical failures than the original phase 1 judgment
- Any pattern appears in >20% of reviewed reasoning chains that was not in the original judge's reported critical failures

**Phase to address:** v1.2 data generation (validation sampling before bulk generation committed to training)

---

### Pitfall 21: Unsloth Optimizer State Reset Breaks LoRA Initialization

**What goes wrong:**
Unsloth resets the optimizer state when loading a saved LoRA adapter for continued training. This is documented behavior: the optimizer history (momentum, variance estimates) from phase 1 is discarded. This has two consequences. First, Adam's variance estimates that stabilized the late-phase-1 training are gone — the optimizer starts as if training from step 0, with no warm gradient statistics to guide step sizes. Second, if the LR scheduler uses the step counter to determine the LR, it starts from step 0 again, meaning it will ramp back up through the full warmup phase even though the model is already trained.

**Why it happens:**
Saving and loading adapters saves only the weight deltas (LoRA A and B matrices), not the optimizer state. This is by design in most PEFT implementations — the optimizer state is often as large as the model itself and not worth preserving for short fine-tuning runs. For continued training with a different objective (reasoning), this becomes a problem because the optimizer "forgets" that this adapter is already trained.

**How to avoid:**
Set a conservative initial LR (see Pitfall 13). Use a flat LR or very short warmup (1–2%) rather than the 5% warmup used in phase 1. Consider using AdaFactor instead of AdamW for continued training — AdaFactor's adaptive per-parameter LR is less sensitive to missing optimizer history. Monitor gradient norms in the first 100 steps: they should be in the same range as late-phase-1 gradient norms (<2), not the high norms seen in early-phase-1 training (5–10).

**Warning signs:**
- Gradient norms in the first 50 steps of continued training are higher than the final gradient norms from phase 1
- Loss is higher at step 100 of phase 2 than at the final step of phase 1
- The LR schedule plot shows a full warmup ramp despite loading a trained adapter

**Phase to address:** v1.2 training configuration (optimizer and scheduler settings before continued training run)

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
| Reusing phase 1 LR config for continued training | No config changes needed | Loss spike in first 200 steps overwrites phase 1 weights | Never — reduce LR by 2-5x for any continued training run |
| 100% judge+reasoning data in v1.2 mix | Maximum reasoning signal | Generation pathway regresses; PHPCS pass rate drops | Never — include 20-30% generation replay examples |
| Bulk reasoning data generation without sampling | Faster data generation | Systematic errors from Claude propagate to training | Never — always sample 1% minimum before committing bulk generation |

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
| Unsloth continued training | Using same LR as initial run | Reduce LR 2-5x; confirm optimizer state is reset and warmup does not overshoot |
| Qwen3 MoE router in fine-tuning | Accidentally training router layer | Router fine-tuning is disabled by default in Unsloth; confirm it remains disabled in continued training config |
| Reasoning training data + JSON format | Training on free-text reasoning only | Always include structured JSON tail in every reasoning example; keep 30% compact-format judge examples in mix |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Sequential generation with `time.sleep` | Phase 2 takes 150+ seconds for 100 gaps | Use async requests or Claude Batch API | Immediately at >50 gaps |
| Inconsistent code truncation (3000 vs 4000 chars) | Long functions scored on truncated code; security-critical code cut off | Standardize at 4000 chars; reject functions >5000 chars at extraction | At any long function >3000 chars |
| No `--sample N` flag for pipeline testing | Full pipeline run required to test any change | Add sample flag before first full run | Every development iteration |
| Opus for all CoT regardless of function length | Phase 3 costs 3x more than estimated | Use Sonnet for <50-line functions; Opus for complex/long functions only | At full 13,500 example dataset |
| Single-threaded git clone in Phase 1 | Cloning 10+ repos takes minutes serially | Parallelize with `asyncio` or thread pool | At >10 repos in `repos.yaml` |
| Max sequence length unchanged for reasoning data | Reasoning traces exceed 4096 token limit; truncated mid-reasoning | Increase max_seq_length to 6144-8192 for v1.2 training run | At any reasoning example longer than 4096 tokens |
| Reasoning inference without length cap | Batch scoring takes 5-10x longer post-v1.2 | Set max_new_tokens for judge calls; use compact mode for routine scoring | Immediately after v1.2 deployment |

---

## Data Quality Traps

| Trap | Risk | Prevention |
|------|------|------------|
| `phase2_judge_dataset.py` consuming 40% of passed examples | Generation training starved of diverse real examples; judge set lacks nuance | Deduplicate across training splits; cap judge examples at 25% of total passed examples |
| Auto-tagging Core code without validation | SQL vulnerability in Core becomes a "reference implementation" style anchor | Sample Core functions through the full judge system; do not trust auto-tagging for security dimensions |
| Task type inferred from metadata with silent default to "gen" | Judge examples mislabeled as generation examples | Require explicit `metadata.task_type`; fail export if absent; never default |
| Taxonomy coverage not enforced post-generation | Final dataset misses concept coverage despite planning | Run coverage check in `export_dataset.py`; fail if any tag below minimum threshold |
| No PHP validity check on exported code examples | Syntactically broken PHP in training set | Parse 100% of code blocks in export validation using `php -l` (lint only, no execution) |
| Reasoning data generated from the same functions as phase 1 training | Model memorizes code-reasoning pairs rather than learning to reason | Use separate held-out function set for reasoning data generation, not the phase 1 training set |
| Critique-then-fix pairs where original code is too easy | Model learns trivial fix patterns; cannot handle novel defect combinations | Include fix examples across all 9 dimensions and all severity levels; reject examples where the fix is a single token change |

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
- [ ] **v1.2 training config:** Confirm LR is 2-5x lower than phase 1; confirm router layer is frozen
- [ ] **v1.2 training mix:** Confirm generation replay examples are included (20-30% of mix)
- [ ] **v1.2 data quality:** Confirm 1% sample of reasoning data reviewed before bulk commit
- [ ] **v1.2 format:** Confirm every reasoning example ends with parseable JSON in canonical structure
- [ ] **v1.2 eval:** Re-run full evaluation suite on generation AND judge tasks; compare absolute score distributions to v1.0 baseline
- [ ] **v1.2 length:** Confirm max_seq_length increased to accommodate reasoning traces; no training examples are being truncated mid-reasoning

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
| v1.2 LR too high, adapter damaged | HIGH | Roll back to phase 1 checkpoint; restart v1.2 with reduced LR; 43hr wall-clock penalty |
| v1.2 format collapse (no JSON in output) | HIGH | Retrain with corrected data mix (30% compact judge examples); enforce JSON tail template in all reasoning examples |
| v1.2 generation regression detected | MEDIUM | Check if generation replay examples were included; if not, retrain with corrected mix; if replay was included, increase replay ratio to 40% |
| v1.2 reasoning length explosion | LOW | No retraining needed; add max_new_tokens cap in inference config; optionally retrain with short-form reasoning examples to internalize brevity |
| v1.2 score calibration drift | LOW | Document new calibration baseline; update thresholds in eval config if needed; does not require retraining unless PASS/FAIL accuracy degrades |

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
| v1.2 LR destruction of adapter | v1.2 training config | LR set to ≤1e-4; gradient norms <3 in first 50 steps |
| v1.2 format collapse | v1.2 data generation template + training mix | Parse success rate ≥95% on judge samples from v1.2 model |
| v1.2 reasoning length explosion | v1.2 data generation + inference config | P90 judge output length ≤800 tokens in batch scoring |
| v1.2 MoE routing shift | v1.2 training config (freeze router) | Router layer gradients = 0 in training logs |
| v1.2 critique skips fix | v1.2 data template + loss masking | ≥95% of critique-then-fix inference samples contain non-empty `<corrected_code>` block |
| v1.2 score calibration drift | v1.2 evaluation | Score distribution histogram vs. v1.0 baseline; security mean shift <0.5 |
| v1.2 generation regression | v1.2 training mix (include gen replay) | PHPCS pass rate on gen samples within 3pp of v1.0 baseline |
| v1.2 reasoning data errors | v1.2 data generation validation | 1% sample reviewed; error rate <5% before bulk generation proceeds |
| v1.2 optimizer state reset | v1.2 training config | Gradient norms at step 50 comparable to late-phase-1 norms; no loss spike above phase 1 final loss |

---

## Sources

- Codebase analysis: `/home/robert_li/Desktop/projects/wp-finetune/.planning/codebase/CONCERNS.md` (HIGH confidence — direct code audit)
- [Anthropic Rate Limits Documentation](https://platform.claude.com/docs/en/api/rate-limits) (HIGH confidence — official)
- [Anthropic 429 Error Guidance](https://support.claude.com/en/articles/8114527-i-m-encountering-429-errors-and-i-m-worried-my-rate-limit-is-too-low-what-should-i-do) (HIGH confidence — official)
- [Practical Tips for Finetuning LLMs Using LoRA — Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) (MEDIUM confidence — expert practitioner)
- [How to Add Special Tokens to LLMs Safely](https://langcopilot.com/posts/2025-09-23-how-to-add-special-tokens-llms) (MEDIUM confidence — community, verified against Unsloth docs)
- [Unsloth Qwen3 Fine-tune Documentation](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune) (HIGH confidence — official tool docs)
- [Unsloth Continued Pretraining Documentation](https://unsloth.ai/docs/basics/continued-pretraining) (HIGH confidence — official; optimizer state reset confirmed)
- [ToMoE: Converting Dense LLMs to MoE via Dynamic Structural Pruning](https://arxiv.org/abs/2501.15316) (HIGH confidence — peer-reviewed 2025)
- [Stabilizing MoE RL by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370) (MEDIUM confidence — routing collapse research)
- [Towards Catastrophic Forgetting-Free Multi-Domain MoE](https://aclanthology.org/2025.emnlp-main.932.pdf) (MEDIUM confidence — EMNLP 2025)
- [vLLM GGUF Documentation](https://docs.vllm.ai/en/stable/features/quantization/gguf/) (HIGH confidence — official, notes "experimental")
- [vLLM Quantization Performance — GPUStack](https://docs.gpustack.ai/2.0/performance-lab/references/the-impact-of-quantization-on-vllm-inference-performance/) (MEDIUM confidence — benchmark data)
- [Which Quantization Method is Right for You? — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/which-quantization-method-is-right) (MEDIUM confidence — practitioner comparison)
- [Qwen3 not stopping generation after LoRA fine-tuning — LlamaFactory Issue #7943](https://github.com/hiyouga/LlamaFactory/issues/7943) (MEDIUM confidence — real-world issue report)
- [Qwen3-MoE Fine-tuning Best Practices — QwenLM GitHub Discussion #1301](https://github.com/QwenLM/Qwen3/discussions/1301) (MEDIUM confidence — official team discussion; loss-to-zero issue documented)
- [ESFT: Expert-Specialized Fine-Tuning for Sparse Architectural LLMs](https://arxiv.org/abs/2407.01906) (HIGH confidence — EMNLP 2024; routing distribution task-concentration finding)
- [Mitigating Forgetting Between SFT and RL — arxiv 2510.04454](https://arxiv.org/html/2510.04454v1) (MEDIUM confidence — SFT-RL forgetting dynamics)
- [On the Limitations of Fine-tuned Judge Models](https://arxiv.org/html/2403.02839v2) (HIGH confidence — peer-reviewed; fine-tuned judge format collapse documented)
- [Stop Overthinking: Survey on Efficient Reasoning for LLMs](https://arxiv.org/pdf/2503.16419) (HIGH confidence — TMLR 2025; overthinking/length explosion documented)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2) (HIGH confidence — LoRA forgetting properties in continued training)
- [SMoLoRA: Dual Catastrophic Forgetting in Continual Learning — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.pdf) (MEDIUM confidence — ICCV 2025; LoRA continual forgetting dynamics)

---

*Pitfalls research for: WordPress code fine-tuning MoE — v1.0 execution/training phase + v1.2 reasoning fine-tune*
*Researched: 2026-03-26 (original); 2026-04-04 (v1.2 reasoning section appended)*
