# Project Research Summary

**Project:** wp-qwen3-moe v1.2 — Judge Reasoning Fine-Tune
**Domain:** LLM fine-tuning pipeline — WordPress PHP code quality, MoE SFT, judge reasoning with critique-then-fix
**Researched:** 2026-04-04
**Confidence:** HIGH (stack and architecture verified against codebase), MEDIUM (reasoning quality metrics, training dynamics)

---

## Executive Summary

This project adds deep reasoning capability to an already-trained WordPress PHP judge model (Qwen3-30B-A3B, 60:40 ratio, LoRA r=32). The v1.0 pipeline produced a well-calibrated judge that scores PHP code across 9 dimensions with Spearman correlation tracking. The v1.2 milestone has a focused goal: the judge should articulate *why* code is good or bad (dimension-by-dimension reasoning chains) and generate a corrected version alongside its critique. Research confirms this is achievable through continued SFT on a narrow reasoning dataset (~6,000-12,000 examples) using the existing Unsloth + TRL stack — no new training infrastructure is required.

The recommended approach is a two-stream data generation pass: (1) regenerate a representative 10% sample (~14,000) of existing judge training examples with full dimension-by-dimension reasoning chains using Claude Code agents, and (2) convert existing phase2 mutation pairs into critique-then-fix triples using the same agent pattern. Both streams feed a separate `data/reasoning_dataset/` that never mixes with the original `data/final_dataset/`. Continued training starts from the winning ratio adapter at 5-10x lower learning rate for 1-2 epochs. The key risk is format collapse: the model forgetting the compact JSON judge format and producing reasoning-only output that breaks the parsing pipeline. This is mitigated by keeping 30% original judge examples in the training mix and enforcing a canonical output template.

Three additional risks need active management: catastrophic forgetting of generation capability (mitigated by 20-30% generation replay in the training mix), reasoning length explosion at inference (mitigated by including short-form reasoning examples in training data), and score calibration drift after reasoning changes the model's analytical depth (mitigated by re-running the full eval suite and comparing absolute score distributions, not just Spearman correlation). The existing eval infrastructure (eval_judge.py, eval_gen.py, eval_gate.py) handles post-training verification without modification.

---

## Key Findings

### Recommended Stack

The v1.0 stack (Unsloth 2026.3.5, TRL 0.24.0, transformers 5.3.0, Qwen3-30B-A3B, DGX Spark) is unchanged for v1.2. The only net-new dependency is `bert-score==0.3.13` for reasoning quality evaluation; it requires only the already-installed `torch`. TRL v1.0 released 2026-03-31 — do NOT upgrade mid-milestone; v0.24.0 handles all required training formats. The one required config change is `max_seq_length: 4096 → 8192` because deep judge CoT chains routinely exceed 4096 tokens across 9 dimensions.

**Core technologies:**
- **Unsloth 2026.3.5**: LoRA SFT accelerator — confirmed installed; handles 8192 token sequences at LoRA r=32 within DGX Spark 128GB unified memory
- **TRL 0.24.0**: SFTTrainer — native support for conversational prompt-completion format; no upgrade needed for v1.2 reasoning chain training
- **anthropic SDK >=0.50.0**: Claude Code agent spawn pattern — existing spawn-until-target pattern reused for both new data generation scripts; no new agent framework needed
- **bert-score==0.3.13**: Reasoning quality evaluation — BERTScore F1 shows 59% vs BLEU's 47% human alignment on reasoning tasks (ACL 2025); only new install required
- **nltk 3.9.3**: Already installed; used for dimension coverage checks and keyword specificity metrics

**Training data format decision:** Reasoning goes in the `response` field as structured prose, not in a `<think>` block. Qwen3's `enable_thinking` is left enabled at inference, but training data must use visible reasoning so users can read the dimension-by-dimension critique. Using TRL v1.0's `"thinking"` field would produce hidden `<think>` blocks — the reasoning IS the product for v1.2, not scaffolding.

### Expected Features

**Must have (v1.2 table stakes):**
- Dimension-by-dimension reasoning in judge output — every rubric-scored judge model (Prometheus, JudgeLM, Auto-J) produces per-criterion reasoning before scores; scores-only output is a structurally defective judge format
- Verdict-after-reasoning ordering with separator token — Prometheus research establishes that generating feedback then score outperforms score-first; `[/REASONING]` separator followed by JSON scores block is the structural requirement
- Critique-then-fix format — defective code to structured critique with severity tags to corrected version in one inference call; no comparable open judge for WordPress PHP produces corrections
- Score consistency validation before training — reject examples where written severity contradicts numeric scores (e.g., "critical SQL injection" with security score 7); training on contradictory examples is actively harmful
- WordPress-specific reasoning templates — `$wpdb->prepare()`, `wp_verify_nonce()`, `check_ajax_referer()`, `esc_html()` must appear by name in reasoning chains; generic patterns do not transfer to WP developers

**Should have (v1.2 differentiators):**
- Severity-tagged issue list in critique — `severity: critical/high/medium/low` plus `dimension` tag per issue; enables downstream triage; shown to improve task transfer in Critique-Coder research
- Fix-rationale field — "what changed and why" explanation alongside corrected code; produces a judge that articulates regressions, not just pass/fail verdicts

**Defer (v2+):**
- TRACT-style regression-aware loss — requires custom loss head on top of Unsloth SFTTrainer; flag for v2.0 if Spearman plateaus below 0.85
- Multi-turn self-refinement training — appropriate after GRPO is introduced in v3.0; the RL loop provides refinement signal without multi-turn SFT format complexity
- Pairwise preference data for reasoning quality — requires human or stronger-model annotation; consider if Spearman plateaus post-v2.0

### Architecture Approach

The v1.2 integration treats the existing pipeline as strictly read-only and adds a parallel new data generation layer feeding a separate reasoning dataset. Four new scripts are required (`phase4_deep_judge_cot.py`, `phase4_critique_fix.py`, `merge_reasoning_dataset.py`, and optionally `eval_reasoning.py`), one new training config (`train_config_reasoning.yaml`), and two new output directories (`data/phase4_reasoning/`, `data/reasoning_dataset/`). The existing `train_model.py`, all eval scripts, and `merge_adapter.py` are used without modification. The reasoning dataset uses an intentionally larger 80/20 train/val split (vs 80/10/10 for the main dataset) because the smaller dataset needs a larger val slice for reliable perplexity tracking.

**Major components:**
1. **Phase 4 data generation layer** — two parallel agent scripts; `phase4_deep_judge_cot.py` sources from `data/phase1_extraction/output/{passed,failed}/`; `phase4_critique_fix.py` sources from `data/phase2_synthetic/output/mutated/`; both use the existing Claude Code agent spawn-until-target pattern
2. **Merge and format layer** — `merge_reasoning_dataset.py` combines both streams, enforces the output template, validates score consistency, and exports SFT-ready JSONL
3. **Continued training layer** — `train_model.py` loaded with `train_config_reasoning.yaml`; starts from the winning ratio adapter; 5-10x lower LR than phase 1; 1-2 epochs maximum; training mix of 40% deep CoT + 30% critique-then-fix + 30% original flat judge examples + 20% generation replay woven in
4. **Evaluation layer** — existing `eval_judge.py` (Spearman), `eval_gen.py` (PHPCS), `eval_gate.py` (gate check); `parse_judge_response()` already handles fenced JSON embedded at the end of reasoning traces; no eval script changes needed

### Critical Pitfalls

**v1.2-specific (highest priority):**

1. **Continued training LR destroys the existing adapter** — optimizer state resets on adapter load; using the same LR as phase 1 (`2e-4`) causes a loss spike that overwrites phase 1 weights. Use 4e-5 to 1e-4; monitor gradient norms in the first 100 steps (should stay below 2-3, not the 5-10 seen in early phase 1).

2. **Format collapse — reasoning chains overwrite JSON structure** — after continued training the model produces well-reasoned text but no parseable JSON, breaking the judge pipeline entirely. Keep 30% original flat judge examples in the training mix; enforce that all reasoning training examples end with a clearly delimited JSON block inside `<judge_output>` tags; monitor parse failure rate at every checkpoint (abort if >5%).

3. **Critique-then-fix model learns to skip the fix** — 143K existing judge-only examples create a strong attractor toward critique-without-code output. Structure corrected code in a clearly delimited block (`<corrected_code>...</corrected_code>`); use response-masking to compute loss on the fix section independently; include both easy (single-line patch) and complex (function refactor) fix examples.

4. **Generation task regression from reasoning data** — `<wp_gen>` pathway degrades when training mix is 100% judge+reasoning data; PHPCS pass rate can drop measurably. Include 20-30% generation replay examples from phase 1 training set in the v1.2 mix; always run `eval_gen.py` after any continued fine-tune, not just `eval_judge.py`.

5. **Reasoning data quality circular dependency** — Claude agent-generated reasoning chains can contain systematic errors that propagate to training. Sample 1% of generated reasoning (minimum 50 examples) and manually review before bulk generation is committed; flag any function where the reasoning chain contradicts the original phase 1 verdict.

**Carry-forward from v1.0 (still applicable):**

6. **Training data poisoned by silent parse failures** — use multi-strategy JSON extraction (fence with language hint, bare fence, regex, raw parse) with hard rejection and failure counter; abort pipeline if >2% of responses fail parsing.

7. **Score inflation from N/A dimensions** — backend PHP functions receiving score 10 on inapplicable dimensions inflates averages; exclude N/A dimensions from average entirely using proportional weighting over applicable dimensions only.

---

## Implications for Roadmap

The v1.2 milestone has a clear six-step sequential flow with one parallel fork.

### Phase 1: Eval Triage — Identify Winning Adapter

**Rationale:** All subsequent phases depend on knowing which adapter to continue training from. This step (`scripts/run_eval_triage.py` + `scripts/triage_ratios.py`) is already in progress but must complete before any v1.2 work begins.
**Delivers:** `output/triage_decision.md` identifying the winning ratio; confirmed adapter path for all downstream phases.
**Avoids:** Building on the wrong adapter (unrecoverable sunk cost if wrong adapter is fine-tuned for 4+ hours).
**Research flag:** No additional research needed — standard eval triage pattern using existing scripts.

### Phase 2a + 2b: Parallel Data Generation

**Rationale:** The two data generation streams are independent and run in parallel. Phase 2a (deep judge CoT) is larger and slower; phase 2b (critique-then-fix) is bounded by the mutation pool size. Both must complete before Phase 3.
**Delivers:**
- `data/phase4_reasoning/deep_judge_cot/` — 5,000-14,000 judge training examples with full dimension-by-dimension reasoning chains
- `data/phase4_reasoning/critique_fix/` — 1,000-2,000 critique-then-fix triples from existing mutation pairs
**Addresses:** Dimension-by-dimension reasoning (P1), critique-then-fix format (P1), WordPress-specific reasoning templates (P1), severity-tagged issues (P1), fix-rationale field (P2).
**Avoids:** Pitfall 20 (reasoning data quality) — sample 1% of output and manually review before proceeding; apply score consistency validation at generation time to reject contradictory examples immediately.
**Research flag:** Phase 2a needs a validated Claude agent prompt for deep judge CoT before bulk generation. The existing `judge_system.md` instructs Claude to output compact JSON — a v1.2 variant must elicit dimension-by-dimension structured reasoning with line-number specificity and WP-specific pattern names. Develop and validate on 20-50 pilot examples before bulk generation.

### Phase 3: Merge and Format Reasoning Dataset

**Rationale:** Depends on both 2a and 2b completing. Enforces the canonical output template, validates score consistency, and assembles the training mix.
**Delivers:** `data/reasoning_dataset/openai_train.jsonl` + `openai_val.jsonl` + `metadata.json`; 80/20 split.
**Uses:** `merge_reasoning_dataset.py` (new script, extends `merge_dataset.py` pattern).
**Addresses:** Score consistency validation (P1), verdict-after-reasoning format (P1), format collapse prevention.
**Avoids:** Pitfall 14 (format collapse) — enforce `<judge_output>` delimited JSON block in all reasoning examples; Pitfall 19 (generation regression) — include generation replay examples from phase 1 training set in the assembled mix.
**Research flag:** No additional research needed — extends existing merge pattern; no novel integration.

### Phase 4: Reasoning Fine-Tune

**Rationale:** Depends on Phase 3 dataset being complete and quality-validated. Continued SFT on the winning adapter using the reasoning dataset.
**Delivers:** `adapters/qwen3-30b-wp-{winning}-reasoning/`
**Uses:** `train_model.py` + `train_config_reasoning.yaml` (new config; lower LR, fewer epochs, reasoning dataset path, winning adapter as base).
**Avoids:**
- Pitfall 13 (LR destroys adapter) — use LR 4e-5 to 1e-4; monitor gradient norms first 100 steps
- Pitfall 21 (optimizer state reset) — flat LR or 1-2% warmup; consider AdaFactor
- Pitfall 16 (MoE routing shift) — confirm router layer is frozen in Unsloth config before training begins
- Pitfall 15 (reasoning length explosion) — training data must include 40/60 short-form/long-form reasoning split
**Research flag:** Needs validation of Unsloth PEFT stacking behavior on Qwen3 MoE before training begins. Specifically: does Unsloth require the base merged weights (Option B) or can it stack a second LoRA on top of a saved adapter (Option A)? Unsloth docs returned partial content during initial research. This is a blocking question — resolve before the training run.

### Phase 5: Eval Verification

**Rationale:** Hard gate before adapter merge. Run all three existing eval scripts against the reasoning adapter.
**Delivers:** Spearman correlation (must be >= v1.0 baseline), PHPCS pass rate (must be >= 95%), `eval_gate.py` pass.
**Addresses:** Score calibration drift (Pitfall 18) — compare absolute score distributions and flag any dimension with mean shift >0.5 points vs v1.0; this is a regression test, not just a Spearman check.
**Avoids:** Pitfall 5 (skipping eval_gen.py) — generation regression may be invisible in judge metrics alone; both evals are mandatory.
**Research flag:** No additional research needed — `parse_judge_response()` already handles fenced JSON embedded in reasoning traces; existing eval scripts are verified compatible.

### Phase 6: Adapter Merge

**Rationale:** Final step, conditional on Phase 5 gate passing. Uses existing `merge_adapter.py` unchanged.
**Delivers:** `models/qwen3-30b-wp-{winning}-reasoning-merged/`
**Avoids:** Pitfall 11 (GGUF in vLLM) — use AWQ with Marlin kernel for vLLM deployment; verify Unsloth export tools support Qwen3 MoE routing tables before export.
**Note for v2.0:** MoE routing profiles from v1.0 are invalidated after v1.2 training even if the router was frozen during training; fresh profiling pass is required before any MoE-Sieve work at v2.0.
**Research flag:** No additional research needed for the merge itself.

### Phase Ordering Rationale

- Phase 1 is a hard prerequisite — all adapter-specific work depends on knowing the winning ratio.
- Phases 2a and 2b are the only parallel steps; sequential execution would add 2-4 hours of wall time unnecessarily.
- Phase 3 must gate on both 2a and 2b completing; it can be scripted to begin as soon as both directories reach target example counts.
- Phase 4 must follow Phase 3 because training mix assembly in Phase 3 is what prevents format collapse and generation regression.
- Phase 5 and 6 are strictly sequential with a hard gate between them; never merge an adapter that has not passed eval_gate.py.

### Research Flags Summary

**Phases needing deeper research during planning:**
- **Phase 2a (deep judge CoT agent prompts):** Develop and validate a v1.2-specific Claude agent prompt that elicits dimension-by-dimension structured reasoning with WP-specific line-level pattern citations. Do not proceed to bulk generation without a 20-50 example pilot.
- **Phase 4 (Unsloth continued training with PEFT stacking):** Verify whether Option A (stack second LoRA on top of saved adapter) or Option B (train LoRA on merged model) is required by Unsloth for Qwen3 MoE. This is a blocking unknown.

**Phases with standard patterns (skip additional research):**
- **Phase 1:** Existing eval triage scripts; no new territory.
- **Phase 3:** Extends existing merge pattern; no novel integration.
- **Phase 5:** Existing eval scripts confirmed compatible with new response format.
- **Phase 6:** Existing merge_adapter.py; no changes required.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All installed versions confirmed via `pip show` on DGX Spark; only new dependency is bert-score which has no compatibility risk; official TRL and Unsloth docs verified (Unsloth fetch was partial — gap noted) |
| Features | HIGH (core format), MEDIUM (metrics) | Reasoning format decisions grounded in Prometheus, Critique-Coder, TRACT research; reasoning quality metrics based on ACL 2025 and preprint evidence |
| Architecture | HIGH | Based on direct codebase inspection; all integration points verified; `parse_judge_response()` compatibility confirmed against the new response format |
| Pitfalls | HIGH (v1.0 pitfalls from code inspection), MEDIUM-HIGH (v1.2 reasoning pitfalls from SFT literature) | v1.2 reasoning pitfalls derived from continued training and format collapse literature; exact severity on Qwen3-30B-A3B specifically not directly measured |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Unsloth PEFT stacking on Qwen3 MoE:** Whether Option A (nested LoRA on adapter) or Option B (LoRA on merged model) is required needs a fresh Unsloth docs fetch before Phase 4. Flag as a blocking research question in the Phase 4 plan.
- **Deep judge CoT agent prompt template:** The v1.2-specific Claude agent prompt needs to be developed and validated on a 20-50 example pilot before bulk generation in Phase 2a. Do not run bulk generation without this validation step.
- **Mutation pool exact size:** The critique-then-fix target volume (1,000-2,000 examples) assumes the mutation pool in `data/phase2_synthetic/output/mutated/` is large enough. Verify the actual count across all 7 mutation types before setting Phase 2b targets.
- **Training mix ratio feasibility:** The recommended mix (40% deep CoT + 30% critique-then-fix + 30% original judge + 20% generation replay) requires validation that dataset volumes support these ratios without oversampling. Adjust if either generation stream produces fewer than 3,000 examples.

---

## Sources

### Primary (HIGH confidence)
- Confirmed `pip show` output from DGX Spark: Unsloth 2026.3.5, TRL 0.24.0, transformers 5.3.0, nltk 3.9.3
- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/dataset_formats) — conversational prompt-completion format, reasoning field options
- [TRL v1.0 Blog Post](https://huggingface.co/blog/trl-v1) — v1.0 release 2026-03-31; minimal migration from 0.x confirmed
- [Qwen-3 Chat Template Deep Dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) — enable_thinking behavior, think tag implications for training data
- [bert-score PyPI](https://pypi.org/project/bert-score/) — v0.3.13 current, torch-only dependency
- Codebase direct inspection: `eval/eval_judge.py`, `scripts/phase2_mutate.py`, `scripts/phase2_judge_dataset.py`, `scripts/merge_dataset.py`, `data/phase3_cot/output/`, `data/final_dataset/`, `adapters/` listing

### Secondary (MEDIUM confidence)
- [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491) — feedback-before-score format, `[RESULT]` separator, rubric-guided training
- [Prometheus 2](https://arxiv.org/html/2405.01535v2) — 40K SFT pointwise format sufficient for SOTA judge quality
- [TRACT (ACL 2025)](https://arxiv.org/abs/2503.04381) — CE + regression-aware loss for judge fine-tuning; difficulty-adaptive reasoning depth
- [Critique-Coder](https://arxiv.org/abs/2509.22824) — structured severity labels improve downstream task transfer; CRL with 20% critique mix
- [J1: Incentivizing Thinking in LLM-as-a-Judge](https://arxiv.org/abs/2505.10320) — training judges to reason; 32B judge matching 671B on structured tasks
- [Training an LLM-as-a-Judge: Pipeline, Insights, Practical Lessons](https://arxiv.org/html/2502.02988v1) — dimension-level scoring format, MAE + Agr(2,2) eval metrics
- [LLM Evaluation 2025: Smarter Metrics](https://www.techrxiv.org/users/927947/articles/1304989) — BERTScore 59% vs BLEU 47% alignment on reasoning tasks (preprint)
- [Unsloth Qwen3 Docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — 75%/25% reasoning/non-reasoning mix; enable_thinking inference config (partial content retrieved — gap)
- [The Art of Repair](https://arxiv.org/abs/2505.02931) — (Instruction, Input, Output) format for code repair instruction tuning

### Tertiary (LOW confidence)
- [Stop Rewarding Hallucinated Steps](https://arxiv.org/html/2602.05897) — faithfulness hallucination rate; not directly applicable to SFT training data context
- [Fine-Tuning with Divergent Chains of Thought](https://arxiv.org/abs/2407.03181) — comparing multiple chains before verdict improves reasoning consistency (SFT context only)

---

*Research completed: 2026-04-04*
*Ready for roadmap: yes*
