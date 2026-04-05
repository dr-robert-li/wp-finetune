# Feature Research

**Domain:** Judge reasoning fine-tuning for code quality models (WordPress PHP)
**Researched:** 2026-04-04
**Confidence:** HIGH (core reasoning formats), MEDIUM (evaluation metrics), LOW (WordPress-specific judge benchmarks)

---

## Context: What Already Exists

This is a subsequent-milestone research document. The following are **already built** and are not in scope:

- 30K judge training examples with JSON scores and short explanations
- 4-way CoT split: `gen_pattern`, `judge_rubric`, `judge_contrastive`, `security`
- Contrastive mutation engine with 7 mutation types (SQL injection, CSRF, XSS, authorization, input_validation, i18n, performance)
- Multi-format export pipeline (OpenAI, Alpaca, Raw JSONL with task tokens)
- 9-dimension eval suite (241 checks, Spearman correlation)
- `phase2_mutate.py` producing verified bad→good contrastive pairs

New features must integrate with this pipeline and use the existing mutation engine output as raw material.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that any competitive judge reasoning fine-tune must have. Missing these means the v1.2 adapter produces reasoning that is structurally incomplete or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Dimension-by-dimension reasoning in judge output | Every rubric-scored judge model (Prometheus, JudgeLM, Auto-J) produces per-criterion reasoning before scores; outputting scores without per-dimension rationale is widely considered a defective judge format | MEDIUM | Requires regenerating existing 30K judge examples with expanded reasoning; existing `judge_system.md` rubric provides the 9-dimension frame |
| Verdict-after-reasoning ordering | Prometheus research established that generating feedback THEN score outperforms score-first formats; the model uses its written reasoning as context when assigning the numeric score — reversing this order degrades score accuracy | LOW | Structural requirement in data generation prompts; add a `[RESULT]` or equivalent separator token between reasoning and score |
| Issue identification with location reference | Reasoning chains that name the specific line/pattern causing the defect (e.g., "line 14: direct variable concatenation in SQL query") produce judges that generalize better than vague category labels | MEDIUM | Claude agents can identify line-level patterns; mutation engine already tags mutation type and line context |
| Score consistency across reasoning | The reasoning text must logically support the numeric score; a chain concluding "critical SQL injection present" followed by security score 8 is a reasoning hallucination — the most common failure mode in fine-tuned judges | MEDIUM | Training data validation step: reject examples where written severity contradicts final score |
| Structured output schema (JSON or tagged fields) | Required for reliable inference-time parsing; unstructured free-text reasoning cannot be reliably parsed downstream | LOW | Existing pipeline already exports JSON; extend schema to include `reasoning` field alongside `scores` and `explanation` |

### Differentiators (Competitive Advantage)

Features that go beyond standard judge fine-tuning and are specific to this use case.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Critique-then-fix format (defective code → critique → corrected code) | Combines judge capability with generative repair in a single inference call; no comparable open judge model for WordPress PHP produces corrections; directly exploits the existing mutation engine which already has verified bad→good pairs | HIGH | Core new format for v1.2; uses `phase2_mutate.py` output directly; each bad→good pair becomes one `(bad_code, structured_critique, good_code)` triple |
| Severity-tagged issue list within critique | Each identified issue gets `severity: critical/high/medium/low` plus `dimension: [security/performance/...]`; enables downstream tooling to triage review comments by severity; Critique-Coder and SRR-Judge both show that structured severity labels improve downstream task transfer | MEDIUM | Addable to the critique template without fundamental architecture change |
| Fix-rationale field in corrected output | The corrected code section includes a brief "what changed and why" explanation, not just the fixed code; this produces judges that can articulate regressions in code review rather than just flagging pass/fail | MEDIUM | Adds 2-3 sentences to each fix section; distill from Claude agents using the existing agent pipeline |
| TRACT-style two-phase training: CoT generation then regression-aware fine-tune | TRACT (ACL 2025) shows that combining CE loss for reasoning with regression-aware loss for numeric score prediction significantly outperforms CE-only fine-tuning; directly applicable to the 9-dimension scoring task | HIGH | Requires modifying Unsloth training config to use a custom loss; may be deferred if Unsloth Studio does not expose custom loss heads |
| WordPress-specific security reasoning templates | Generic judge models do not know that `$wpdb->prepare()` absence is SQL injection, or that `check_ajax_referer()` absence is CSRF; embedding WP-specific pattern names in reasoning templates produces a judge that speaks the vocabulary WordPress developers use | LOW | Template enrichment in Claude agent prompts; no architecture change required |
| Contrastive reasoning pair with causal "intent vs. effect" explanation | The contrastive `judge_contrastive` CoT type already exists but uses short explanations; expanding it to include "here is what the developer probably intended vs. what actually happens" improves the model's ability to reason about subtle authorization bugs | MEDIUM | Regenerate `judge_contrastive` examples with deeper causal reasoning; source material from existing mutation engine output |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Free-form "think out loud" reasoning with no structure | Seems more natural; closer to GPT-o1 style extended thinking | Produces reasoning chains where the model can rationalize any score; training data quality degrades because there is no schema to validate consistency between reasoning and score; faithfulness hallucination rate is high for unstructured CoT (ProcessBench reports 51.8% invalid-trace-but-correct-answer rate on challenging problems) | Use structured reasoning with labeled sections (dimension analysis, issue list, fix plan, scores) — constrains the generation space and enables automated consistency checking during data validation |
| Iterative self-refinement (model critiques its own output, then re-critiques) | Appealing for quality; Self-Refine shows 21-32pp improvement on unit test correctness | For a 30B MoE in SFT fine-tuning context, multi-turn refinement loops during training require 2-3x data volume and complicate the training format significantly; also introduces position bias (later critique always appears "more thoughtful") | Single-pass critique-then-fix is sufficient for v1.2; multi-turn self-refinement is appropriate for v3.0 GRPO where the RL loop provides the refinement signal |
| Pairwise preference format for reasoning quality (A vs B) | RLHF-style comparison seems principled for teaching reasoning quality | Requires human labelers or a stronger model to judge which reasoning chain is better; circularity problem — the model being trained cannot reliably self-evaluate reasoning quality; DPO on reasoning pairs tends to shorten reasoning rather than improve it | Use pointwise scoring with consistency validation (score matches reasoning conclusion); Prometheus 2 demonstrated that 40K SFT examples with pointwise format match pairwise RLHF judge quality |
| Exhaustive deep reasoning for every training example including trivial passes | More training data with deeper reasoning should always help | Simple high-quality functions (score 9-10 on all dimensions) produce trivial reasoning chains ("no issues found, well-formed code"); training on these dilutes the signal and inflates the dataset without improving judge capability; TRACT and Prometheus both use difficulty-adaptive reasoning depth | Reserve deep reasoning for examples with score variance (at least one dimension < 8); apply short-form reasoning or skip for uniformly high-quality examples |
| Separate reasoning model from scoring model | Clean architecture; reasoning and scoring as distinct heads | For a LoRA fine-tune on a single MoE model, adding a separate head requires architecture changes incompatible with Unsloth LoRA; increases serving complexity for what is essentially a format choice | Single-model output with tagged sections; use structured output parsing at inference time |

---

## Feature Dependencies

```
[Deep judge CoT with dimension reasoning]
    └──required by──> [Critique-then-fix format]
                          (critique sections map to the same 9 dimensions)

[Existing contrastive mutation engine (phase2_mutate.py)]
    └──provides source material──> [Critique-then-fix training pairs]
                                       (bad→good pairs become critique triples)

[Existing 30K judge examples (short explanation)]
    └──regenerated as──> [Deep judge CoT examples (long reasoning chains)]

[Severity-tagged issue list]
    └──required by──> [Fix-rationale field]
                          (fix rationale must reference which severities it resolves)

[Verdict-after-reasoning ordering]
    └──required by──> [Score consistency validation]
                          (validation step requires reasoning text to precede score)

[TRACT-style regression-aware loss]
    └──requires──> [Unsloth custom loss support]
                       (dependency on training infrastructure — may block)

[Deep judge CoT data] ──parallel with──> [Critique-then-fix data]
    (both produced in same agent generation pass; same model, different prompt templates)
```

### Dependency Notes

- **Critique-then-fix requires deep judge CoT:** The critique sections in the fix format are the same dimension-by-dimension reasoning as in standalone judge CoT. Build judge CoT format first and reuse the template for critique sections.
- **Mutation engine provides critique-then-fix source material:** `phase2_mutate.py` already produces verified `(bad_code, mutation_type, good_code)` triples for all 7 mutation types. These become the raw material for `(bad_code, structured_critique, good_code)` triples — the critique is generated by Claude agents who know the mutation type and can reason about it explicitly.
- **Regression-aware loss is optional in v1.2:** TRACT-style loss is a differentiator, not table stakes. If Unsloth Studio does not expose custom loss heads, standard CE loss on the full sequence (reasoning + scores) is acceptable. Flag for v2.0.
- **Score consistency validation must precede training:** Training on examples where reasoning contradicts scores is actively harmful. Build a validation step that rejects examples with logical inconsistency before export.

---

## MVP Definition

### Launch With (v1.2)

The minimum to deliver the milestone goal: model can articulate why code is bad, score with dimension-level justification, and generate corrected versions.

- [ ] Deep judge CoT data: Regenerate judge training examples with full reasoning chains — dimension-by-dimension analysis, specific issue identification with WP pattern names, per-dimension scores — sourced from existing 30K judge examples regenerated via Claude agents
- [ ] Critique-then-fix data: For each mutation-engine output triple `(bad_code, mutation_type, good_code)`, generate `(bad_code, structured_critique_with_severity, good_code_with_fix_rationale)` using Claude agents
- [ ] Score consistency validation: Automated check that rejects training examples where written severity labels contradict numeric scores (e.g., "critical SQL injection" + security score 7)
- [ ] Verdict-after-reasoning format with separator token: Reasoning text comes first, `[/REASONING]` or equivalent separator, then JSON score block — enforced in data generation prompts and validated in export
- [ ] Fine-tune winning ratio adapter on combined deep reasoning dataset: Standard SFT with CE loss; LoRA r=32 same as v1.0

### Add After Validation (v1.x)

- [ ] TRACT-style regression-aware loss — add if Unsloth Studio supports custom loss heads and v1.2 eval shows score calibration gaps (Spearman below 0.85)
- [ ] Difficulty-adaptive reasoning depth — skip deep reasoning for uniformly high-quality examples (all dimensions >= 9) once dataset volume is confirmed sufficient without them
- [ ] Contrastive reasoning expansion with causal "intent vs. effect" — add if v1.2 eval shows authorization dimension underperforming

### Future Consideration (v2+)

- [ ] Multi-turn self-refinement training — appropriate after GRPO is introduced in v3.0; the RL loop provides refinement signal without requiring explicit multi-turn SFT format
- [ ] GRPO for judge reasoning quality — v3.0 GRPO currently scoped as gen-only (`<wp_gen>`); extending GRPO to refine judge reasoning quality using verifiable rewards (PHPCS/security scanner for critique-then-fix corrections, separately spawned Claude evaluator agent for scoring consistency checks) is a scope consideration for v3.0, deferred until gen-only GRPO results are evaluated
- [ ] Pairwise reasoning preference data — requires human or stronger-model annotation; consider if Spearman correlation plateaus below 0.90 after v1.2 and v2.0

---

## Feature Prioritization Matrix

| Feature | Model Value | Implementation Cost | Priority |
|---------|-------------|---------------------|----------|
| Dimension-by-dimension reasoning in judge output | HIGH | MEDIUM | P1 |
| Verdict-after-reasoning ordering with separator token | HIGH | LOW | P1 |
| Critique-then-fix format (bad → critique → fixed) | HIGH | HIGH | P1 |
| Score consistency validation (reject contradictory examples) | HIGH | LOW | P1 |
| Severity-tagged issue list | MEDIUM | MEDIUM | P1 |
| WordPress-specific security reasoning templates | HIGH | LOW | P1 |
| Fix-rationale field in corrected output | MEDIUM | MEDIUM | P2 |
| Contrastive reasoning with causal "intent vs. effect" | MEDIUM | MEDIUM | P2 |
| Difficulty-adaptive reasoning depth | MEDIUM | LOW | P2 |
| TRACT regression-aware loss | HIGH | HIGH | P3 |
| Multi-turn self-refinement training | LOW | HIGH | P3 |

**Priority key:**
- P1: Required for v1.2 milestone goal
- P2: Add if agent generation capacity permits or eval shows gap
- P3: Deferred to v2.0/v3.0 or requires infrastructure prerequisites

---

## Competitor Feature Analysis (Judge Model Landscape)

| Feature | Prometheus 2 | JudgeLM / Auto-J | Our v1.2 Approach |
|---------|--------------|------------------|-------------------|
| Reasoning format | Feedback text then `[RESULT]` score | Score + justification | Dimension sections then `[/REASONING]` then JSON scores |
| Per-dimension scoring | Rubric-guided, 1-5 Likert | Single overall quality score | 9-dimension, 1-10 each, security auto-fail |
| Code repair output | None | None | Critique-then-fix with corrected PHP |
| Domain specialization | General (language quality) | General | WordPress PHP (WPCS, security patterns, WP APIs) |
| Training data scale | 100K (GPT-4 generated) | 33K-100K | ~30K regenerated + mutation engine triples |
| Score calibration approach | CE loss on full sequence | CE loss | CE loss (TRACT as P3 upgrade) |
| Mutation-based contrastive pairs | No | Some DPO pairs | Yes, 7 mutation types, PHPCS-verified |

**Key differentiator:** No open judge model produces WP-specific PHP repair suggestions. The mutation engine's verified bad→good pairs are a unique data source that generic judge models cannot replicate.

---

## Sources

- [TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge (ACL 2025)](https://arxiv.org/abs/2503.04381) — MEDIUM confidence; two-stage CE + regression loss for judge fine-tuning
- [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491) — HIGH confidence; feedback-before-score format, `[RESULT]` separator, rubric-guided training
- [Prometheus 2 (May 2024)](https://arxiv.org/html/2405.01535v2) — HIGH confidence; 40K SFT examples sufficient for SOTA pointwise judge; supports absolute and pairwise grading
- [Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning](https://arxiv.org/abs/2509.22824) — MEDIUM confidence; CRL with 20% critique mix ratio outperforms RL-only; structured critique judgment labels
- [Fine-Tuning with Divergent Chains of Thought Boosts Reasoning Through Self-Correction](https://arxiv.org/abs/2407.03181) — MEDIUM confidence; comparing multiple chains before verdict improves reasoning consistency
- [LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods](https://arxiv.org/html/2412.05579v2) — HIGH confidence; position bias, length bias, scoring bias mitigation approaches
- [Evaluating Step-by-step Reasoning Traces: A Survey](https://arxiv.org/html/2502.12289v1) — MEDIUM confidence; faithfulness metrics, hallucinated reasoning detection
- [Stop Rewarding Hallucinated Steps: Faithfulness-Aware Step-Level RL](https://arxiv.org/html/2602.05897) — LOW confidence (training data context, not direct applicability); faithfulness hallucination rate data point
- [CYCLE: Learning to Self-Refine the Code Generation](https://dl.acm.org/doi/full/10.1145/3649825) — MEDIUM confidence; iterative refinement training; confirms multi-turn is disproportionately costly for SFT
- [Finetuning LLM Judges for Evaluation (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/finetuned-judge) — MEDIUM confidence; practitioner summary of JudgeLM, Auto-J, Prometheus training patterns
- Existing codebase: `scripts/phase2_mutate.py`, `scripts/phase3_cot.py`, `scripts/phase2_judge_dataset.py` — HIGH confidence; verified source material and pipeline integration points

---

*Feature research for: wp-qwen3-moe v1.2 judge reasoning fine-tune*
*Researched: 2026-04-04*
