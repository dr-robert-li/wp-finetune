# Stack Research

**Domain:** LLM fine-tuning pipeline — WordPress code data + MoE fine-tuning on DGX Spark (v1.2 addendum: deep judge CoT + critique-then-fix)
**Researched:** 2026-04-04 (v1.2 milestone update; original 2026-03-26)
**Confidence:** HIGH for data generation additions (Claude API patterns verified), MEDIUM for reasoning quality eval metrics (no single authoritative source), HIGH for training format changes (TRL docs verified)

---

## v1.2 Milestone Scope

This document extends the original stack with additions specific to the v1.2 Judge Reasoning Fine-Tune milestone. The existing stack (Unsloth 2026.3.x, TRL 0.24.0, transformers 5.3.0, Qwen3-30B-A3B, DGX Spark infrastructure) is **unchanged**. Only new or modified components are described below.

**What v1.2 adds to the pipeline:**
1. Deep judge CoT data generation — regenerate judge training examples with full dimension-by-dimension reasoning chains
2. Critique-then-fix data generation — new training format: defective code → structured critique (what/why/severity per dimension) → corrected version
3. Reasoning quality evaluation — measure whether reasoning chains are substantive, not just syntactically valid

---

## Recommended Stack — New Components Only

### Data Generation

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| anthropic (Python SDK) | >=0.50.0 (already installed) | Generate deep reasoning chains via Claude Code agents | Already in use for phase1-3; extend same agent spawn pattern for v1.2 generation; no new dependency |
| claude-sonnet-4-6 | current | Deep judge CoT generation — dimension-by-dimension analysis | Sonnet sufficient for structured reasoning chains at this complexity level; reserve Opus only if chain quality is inadequate after spot-check |

**Data generation approach:** Use the existing Claude Code agent spawn pattern (see `docs/AGENT_PIPELINE.md`) to generate both deep judge CoT and critique-then-fix pairs. No new LLM tooling is needed. The existing `phase2_judge_dataset.py` agent loop is the template.

### Training Format Changes (No New Libraries)

The existing TRL SFTTrainer handles both new formats natively. What changes is the **JSONL schema** of training examples, not the trainer.

**Deep judge CoT format** — conversational prompt-completion with reasoning embedded in assistant content:

```jsonc
{
  "prompt": [
    {"role": "system", "content": "<wp_judge> system prompt..."},
    {"role": "user", "content": "Score this WordPress PHP code:\n```php\n{code}\n```"}
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "**Dimension Analysis:**\n\n**D1 (WPCS Compliance):** [reasoning about specific violations found] → score: 72\n**D2 (Security):** [reasoning: nonce check missing on line 14, $wpdb->prepare absent] → score: 30\n...\n**D9 (Documentation):** [reasoning] → score: 85\n\n**Verdict:** FAIL. Security dimension score 30 is below auto-fail threshold of 50. Must fix: [list]\n\n```json\n{\"wpcs_compliance\": 72, \"security_score\": 30, ...}\n```"
    }
  ]
}
```

**Critique-then-fix format** — single assistant turn combining critique and corrected code:

```jsonc
{
  "prompt": [
    {"role": "system", "content": "<wp_judge> critique and fix system prompt..."},
    {"role": "user", "content": "Review this WordPress PHP code and provide a corrected version:\n```php\n{defective_code}\n```\nMutation type: {mutation_description}"}
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "**Critique:**\n\n- D2 (Security) CRITICAL: SQL injection via unparameterized $wpdb->query on line 8. Direct variable interpolation bypasses WordPress's prepared statement API.\n- D1 (WPCS) MODERATE: Missing @param/@return PHPDoc blocks reduce maintainability.\n\n**Corrected code:**\n```php\n{fixed_code}\n```\n\n**What changed:** Replaced direct $wpdb->query with $wpdb->prepare() using %d placeholder. Added PHPDoc blocks."
    }
  ]
}
```

**Key format decisions:**
- Reasoning goes inside the `content` field as structured prose — **not** in a `<think>` block. Qwen3's `enable_thinking` is left at default (enabled) for inference, but training data uses plain-text reasoning so the model learns to output visible critiques, not hidden deliberation.
- Do NOT use the `"thinking"` field in assistant messages (supported by TRL v1.0 format). Visible critique text is the product — users need to read the reasoning. Hidden `<think>` blocks would train the model to reason privately then emit only scores, defeating the purpose.
- Source material for critique-then-fix: existing contrastive pairs from `phase2_mutate.py` (7 mutation types already produce defective→good pairs). Claude agents expand the reasoning annotation on top of existing mutations.

### Reasoning Quality Evaluation — New Script

A new evaluation script (`eval_reasoning_quality.py`) is needed to measure whether generated reasoning chains are substantive. This uses regex-based fast checks as primary gates plus a separately spawned Claude evaluator agent as a secondary quality signal for reasoning coherence evaluation. The Claude evaluator agent runs in an independent context window and receives only the generated code + reasoning as opaque inputs — no shared state with the model under test. This isolation principle mitigates circularity concerns (decision D-19 revisited: the separately spawned agent with opaque inputs is acceptable because the evaluator cannot access training data, model weights, or generation context).

| Tool | Location | Purpose | Why |
|------|----------|---------|-----|
| Separately spawned Claude evaluator agent | Claude Code (subscription) | Evaluate reasoning chain quality: coherence, dimension coverage depth, score-reasoning consistency | Claude evaluator agent provides actual reasoning quality assessment (coherence, logical consistency) rather than text similarity; spawned in an independent context window with only generated code + reasoning as opaque inputs (no shared state with model under test); mitigates circularity because the evaluator has no access to training data, model weights, or generation context; $0 cost via subscription |
| nltk | 3.9.3 (already installed) | Tokenization for reasoning chain length metrics | Already present; used for token count, sentence count, coverage of expected keywords (e.g., "nonce", "escape", "prepare") |

**What NOT to use for reasoning quality eval:**
- BLEU/ROUGE — designed for translation/summarization; punishes valid paraphrases of the same reasoning
- BLEURT — requires a trained checkpoint download (~1.4GB); overkill for this task; limited to Google's hosted checkpoint
- BERTScore — text similarity metric, not a reasoning quality evaluator; measures surface-level semantic overlap between texts, cannot assess whether a reasoning chain is logically coherent, covers all dimensions substantively, or has score-reasoning consistency
- LLM-as-judge using the model being trained — circular; we're training the judge, we can't use it to evaluate itself
- Same model family as evaluator with shared context — circular if evaluator can see training data or generation context; the separately spawned Claude evaluator agent is acceptable because it operates in an independent context window with only opaque inputs (generated code + reasoning text), no access to training data, model weights, or generation prompts

**Reasoning quality metrics (implemented in `eval_reasoning_quality.py`):**
1. **Dimension coverage** — does the chain mention all 9 dimensions by name? (exact string match, no model needed)
2. **Issue specificity** — does the chain cite specific line numbers, function names, or WordPress API violations? (regex over known patterns: `line \d+`, `$wpdb->`, `wp_verify_nonce`, etc.)
3. **Claude evaluator agent coherence score** — a separately spawned Claude evaluator agent (independent context window, opaque inputs only) evaluates a sample of reasoning chains for: (a) logical coherence of the reasoning flow, (b) whether dimension analyses substantively address the code rather than being generic filler, (c) consistency between written reasoning severity and final numeric scores
4. **Fix presence rate** (critique-then-fix only) — does the corrected code actually contain `$wpdb->prepare`, `wp_verify_nonce`, `esc_html`, etc. that the original was missing? (PHPCS + regex; no model required)

**Primary gate:** Dimension coverage >=9/9 AND issue specificity rate >=60% of examples cite at least one specific violation. Claude evaluator agent coherence is a secondary quality signal evaluated on a representative sample (~100-200 examples), not run on every example.

### Supporting Scripts — New Files in `scripts/`

No new libraries. New Python scripts that extend existing patterns:

| Script | Purpose | Extends |
|--------|---------|---------|
| `phase4_deep_judge_cot.py` | Regenerate judge training examples with full dimension-by-dimension reasoning chains | `phase2_judge_dataset.py` agent loop pattern |
| `phase4_critique_fix.py` | Generate critique-then-fix pairs from `phase2_mutate.py` outputs | `generate_cot_real.py` structure + `phase2_mutate.py` source data |
| `eval_reasoning_quality.py` | Measure dimension coverage, issue specificity, Claude evaluator agent coherence on generated chains | Standalone; uses nltk + separately spawned Claude evaluator agent (subscription) |

---

## Recommended Stack — Existing Components (Confirmed Unchanged)

The full v1 stack remains valid. Key confirmed versions from the DGX Spark environment:

| Technology | Version (confirmed) | Status |
|------------|---------------------|--------|
| Unsloth | 2026.3.5 | Confirmed installed |
| TRL | 0.24.0 | Confirmed installed; v1.0 released 2026-03-31 — do NOT upgrade mid-milestone |
| transformers | 5.3.0 | Confirmed installed |
| Python | 3.11 | Per DGX playbook |
| anthropic SDK | >=0.50.0 | In use across phases 1-3 |
| nltk | 3.9.3 | Confirmed installed |

**TRL version note:** TRL v1.0 (released 2026-03-31) is a stability release — "migration from last 0.x version is minimal." Do NOT upgrade to v1.0 mid-milestone without validating against `train_model.py`. The 0.24.0 installed version handles all v1.2 training format requirements (standard prompt-completion conversational format; no new TRL features required).

---

## Training Configuration Changes

The existing `train_config_*.yaml` files require **one change** for v1.2: max sequence length increase to accommodate longer reasoning chains.

| Parameter | v1.0 Value | v1.2 Value | Reason |
|-----------|-----------|-----------|--------|
| `max_seq_length` | 4096 | 8192 | Deep judge CoT chains (dimension analysis × 9 + issue list + JSON) routinely exceed 4096 tokens; Qwen3-30B-A3B supports 128K context; DGX Spark 128GB handles 8192 at LoRA r=32 |
| `per_device_train_batch_size` | (current) | Reduce by 50% or keep with gradient checkpointing | Longer sequences increase activation memory; use adaptive planner's existing batch coupling to maintain effective batch size |
| LoRA r | 32 (v1.2 config) | 32 | No change; consistent with v1.2 plan ("only winning ratio gets this treatment") |

**Memory estimate:** At 8192 token sequences with LoRA r=32 on Qwen3-30B-A3B (128GB unified memory), batch_size=1 with gradient_accumulation=16 is the conservative baseline. The existing adaptive planner handles batch tuning.

---

## Data Sourcing Strategy for v1.2

| Data Type | Source | Volume Target | Generation Method |
|-----------|--------|---------------|-------------------|
| Deep judge CoT | All 134K phase1 judged functions (passed + failed) + 2,720 synthetic passed | Sample 10-20K for reasoning annotation | Claude Code agents; same spawn pattern as phase2_judge_dataset.py |
| Critique-then-fix | Existing mutated pairs from `data/phase2_synthetic/output/mutated/` (7 mutation types) | All available mutated pairs (~estimated 5-10K) | Claude Code agents add critique + verified fix annotation |

**Source material decision:** The existing `phase2_mutate.py` contrastive pairs are ideal for critique-then-fix because the mutation type is known (sql_injection, csrf, xss, etc.) — Claude agents can be prompted with the mutation label to produce precise, dimension-targeted critiques instead of generic ones.

---

## Installation — New Dependencies Only

No new Python package dependencies for v1.2. The Claude evaluator agent uses the existing Claude Code subscription (same as data generation agents) and requires no additional infrastructure.

Everything required (anthropic SDK, nltk, TRL, Unsloth, transformers) is already installed.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Plain-text reasoning in `content` field | Hidden `<think>` blocks via Qwen3 enable_thinking=True training data | The reasoning IS the product for v1.2 — users need to see dimension-by-dimension critiques; hidden thinking trains covert deliberation not visible output; also requires enable_thinking flag management at inference time |
| Visible reasoning in assistant `content` | TRL v1.0 `"thinking"` field in assistant messages | The "thinking" field renders as hidden `<think>` blocks in Qwen3 chat template — same problem as above; also requires TRL upgrade mid-milestone |
| Separately spawned Claude evaluator agent for reasoning quality | BERTScore (text similarity) | BERTScore measures surface-level semantic overlap, not reasoning quality; a separately spawned Claude evaluator agent (independent context window, opaque inputs only) can evaluate coherence, dimension coverage depth, and score-reasoning consistency — actual reasoning quality signals; circularity mitigated by isolation principle (no shared state with model under test) |
| Extend `phase2_judge_dataset.py` pattern | New agent framework (LangChain, DSPy) | Zero new dependencies; existing Claude Code agent spawn pattern already validated through 143K examples in v1.0 |
| Regenerate from all 134K judged functions | Generate from scratch with new prompts | Reusing existing judged data guarantees the reasoning annotations cover real-world code, not just synthetic; mutation type labels from phase2_mutate.py provide critique anchors |
| claude-sonnet-4-6 for reasoning generation | claude-opus-4-6 | Sonnet is sufficient for structured multi-dimension reasoning at this complexity; upgrade to Opus only after spot-checking chain quality; 40% cost difference |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Qwen3 `<think>` blocks in training data | Trains hidden reasoning, not visible critique; inference-time users need to see the reasoning | Embed reasoning directly in assistant `content` as structured prose |
| TRL v1.0 upgrade mid-milestone | Released 2026-03-31; migration risk during active training; v0.24.0 handles all required formats | Stay on 0.24.0; evaluate upgrade at v2.0 milestone start |
| max_seq_length=4096 for v1.2 training data | Deep judge CoT chains exceed 4096 tokens routinely; truncation silently destroys reasoning quality | Set max_seq_length=8192; validate 95th percentile token length of generated chains before training |
| BLEU/ROUGE/BERTScore for reasoning quality | Designed for text similarity, not reasoning quality; BERTScore measures semantic overlap, not logical coherence or score-reasoning consistency | Claude evaluator agent coherence (sample-based) + dimension coverage + issue specificity regex |
| Separate "fix" and "critique" as two training examples | Creates two-call inference pattern; model needs to learn single-pass critique-then-fix | Combine critique + corrected code in one assistant turn |
| New external data for critique-then-fix | Introduces distribution shift from existing training data | Use `phase2_mutate.py` outputs — same mutation types, same code distribution, same quality filters already applied |

---

## Stack Patterns by Phase — v1.2 Additions

**Phase 4a (Deep Judge CoT Generation):**
- Claude Code agents (subscription); spawn-until-target pattern
- Prompt template: system = `<wp_judge>` judge role + dimension rubric; user = code sample; expected output = dimension-by-dimension prose analysis + JSON scores
- Quality gate: dimension coverage check (all 9 mentioned) + issue specificity >=60% + Claude evaluator agent coherence on pilot sample
- Output: `data/phase4_reasoning/deep_judge_cot.jsonl`

**Phase 4b (Critique-then-Fix Generation):**
- Claude Code agents; source = `data/phase2_synthetic/output/mutated/` existing pairs
- Prompt template: system = `<wp_judge>` critique role; user = defective code + mutation_type label; expected output = critique (what/why/severity) + corrected code
- Quality gate: PHPCS pass on corrected code + fix-presence check (mutation-specific security functions restored)
- Output: `data/phase4_reasoning/critique_fix.jsonl`

**Phase 4c (Reasoning Quality Evaluation):**
- `eval_reasoning_quality.py`: dimension coverage, issue specificity, Claude evaluator agent coherence (sample-based)
- Run before training; fail fast if <80% of chains pass dimension coverage gate
- Output: `telemetry/reasoning_quality_report.json`

**Phase 4d (Fine-tuning on Combined Reasoning Dataset):**
- Same Unsloth + TRL SFTTrainer; same DGX Spark infrastructure
- Dataset: merge deep_judge_cot.jsonl + critique_fix.jsonl + winning-ratio existing data
- Config change: max_seq_length=8192; adaptive planner handles batch sizing
- Adaptive planner already handles sequence-length-driven memory pressure

---

## Version Compatibility — v1.2 Additions

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| Claude evaluator agent (subscription) | Claude Code CLI | Separately spawned agent with independent context; used for reasoning quality eval on sample batches; $0 cost via subscription |
| max_seq_length=8192 | unsloth 2026.3.5, Qwen3-30B-A3B | Qwen3 supports 128K context; 8192 is well within Unsloth's tested range |
| TRL 0.24.0 | prompt-completion conversational format | Standard format; no upgrade needed for v1.2 reasoning chain training |

---

## Sources

- [TRL Dataset Formats](https://huggingface.co/docs/trl/main/dataset_formats) — conversational prompt-completion format, reasoning field options (HIGH confidence; official docs)
- [TRL v1.0 Blog Post](https://huggingface.co/blog/trl-v1) — v1.0 release date 2026-03-31, minimal migration from 0.x (HIGH confidence)
- [Qwen-3 Chat Template Deep Dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) — enable_thinking behavior, think tag format, implications for training data (HIGH confidence)
- [Unsloth Qwen3 Docs](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) — 75%/25% reasoning/non-reasoning dataset mix recommendation; enable_thinking config for inference (MEDIUM confidence; docs fetch returned partial content)
- [J1: Incentivizing Thinking in LLM-as-a-Judge](https://arxiv.org/abs/2505.10320) — training judges to reason with verifiable rewards; 32B judge matches 671B DeepSeek-R1 on some benchmarks (MEDIUM confidence; research paper)
- [Training an LLM-as-a-Judge: Pipeline, Insights, Practical Lessons](https://arxiv.org/html/2502.02988v1) — dimension-level scoring format, MAE + Agr(2,2) as eval metrics (MEDIUM confidence; research paper)
- [Improve LLM-as-a-Judge as General Ability](https://aclanthology.org/2025.emnlp-main.712.pdf) — jCoT (reasoning process) + jres (judge result) training format (MEDIUM confidence; peer-reviewed)
- [LLM Evaluation 2025: Smarter Metrics](https://www.techrxiv.org/users/927947/articles/1304989) — comparison of automated metrics on reasoning tasks; text similarity metrics (BERTScore, BLEU, BLEURT) have limited alignment with human judgment on reasoning quality evaluation (MEDIUM confidence; preprint)
- [The Art of Repair: Optimizing Iterative Program Repair](https://arxiv.org/abs/2505.02931) — (Instruction, Input, Output) format for code repair instruction tuning; buggy code + fix template + corrected code structure (MEDIUM confidence; research paper)
- Confirmed installed versions via `pip show` on DGX Spark: TRL 0.24.0, transformers 5.3.0, unsloth 2026.3.5, nltk 3.9.3 (HIGH confidence; direct verification)

---

*Stack research for: wp-qwen3-moe v1.2 (Deep Judge CoT + Critique-then-Fix)*
*Researched: 2026-04-04*
