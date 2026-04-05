# Architecture Research

**Domain:** ML fine-tuning pipeline — v1.2 Judge Reasoning Fine-Tune integration
**Researched:** 2026-04-04
**Confidence:** HIGH (based on direct codebase inspection, existing data, and known pipeline behavior)

## Context: What Exists vs What v1.2 Adds

This document is scoped to **milestone v1.2** — adding deep judge CoT and critique-then-fix capabilities to the winning ratio adapter. The existing v1.0/v1.1 architecture (Phase 1-4 data pipeline, multi-ratio training, eval triage) is complete. This file focuses exclusively on the integration question.

### Current State (post-Phase 4 triage)

- Phase 1-3 data pipeline fully executed: 267K merged training examples, 5-ratio exports in `data/final_dataset/`
- Four trained LoRA adapters: `adapters/qwen3-30b-wp-{30_70,40_60,50_50,60_40}/`
- Winning ratio will be determined by Phase 4 eval triage (`scripts/run_eval_triage.py` + `scripts/triage_ratios.py`)
- Existing judge training format: flat rubric scores with a short `explanation` field
- Existing CoT format: 4 types (gen_pattern, judge_rubric, judge_contrastive, security), 29,020 total in `data/phase3_cot/output/`

### What v1.2 Adds

1. **Deep judge CoT data** — new `<wp_judge>` training examples where the response is a full reasoning chain (dimension-by-dimension analysis → issue identification → fix suggestions → final scores), not just the score JSON
2. **Critique-then-fix data** — new format where the model sees defective code, produces a structured per-dimension critique, then outputs a corrected version; uses phase2 mutations as source material
3. **v1.2 reasoning dataset** — merged file combining deep judge CoT + critique-then-fix, formatted for SFT on the winning adapter
4. **Reasoning fine-tune run** — single LoRA training pass on winning adapter using reasoning dataset
5. **Eval verification** — re-run eval_judge (Spearman) and eval_gen (PHPCS) on the reasoning-enhanced adapter to confirm judge improvement without gen regression

## System Overview: v1.2 Integration Points

```
EXISTING PIPELINE (complete, read-only for v1.2)
┌─────────────────────────────────────────────────────────────────────┐
│ data/phase1_extraction/output/passed/   (93,904 passed functions)   │
│ data/phase2_synthetic/output/mutated/   (contrastive mutation pairs)│
│ data/phase2_synthetic/output/judge_training/  (143K judge examples) │
│ data/phase3_cot/output/                 (29,020 CoT examples)       │
│ adapters/qwen3-30b-wp-{winning}/        (winning ratio adapter)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  read-only sources
                               ▼
NEW DATA GENERATION LAYER (v1.2, new scripts)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌───────────────────────────────┐  ┌───────────────────────────┐  │
│  │  phase4_deep_judge_cot.py     │  │  phase4_critique_fix.py   │  │
│  │  (NEW SCRIPT)                 │  │  (NEW SCRIPT)             │  │
│  │                               │  │                           │  │
│  │  Input: passed/ + failed/     │  │  Input: mutated/          │  │
│  │    (or judge_training/)       │  │    (contrastive pairs)    │  │
│  │                               │  │                           │  │
│  │  Agent: regenerate judge      │  │  Agent: given defective   │  │
│  │  training with full reasoning │  │  code + mutation type,    │  │
│  │  chain per dimension before   │  │  produce structured       │  │
│  │  emitting score JSON          │  │  critique then corrected  │  │
│  │                               │  │  version                  │  │
│  │  Output:                      │  │  Output:                  │  │
│  │  data/phase4_reasoning/       │  │  data/phase4_reasoning/   │  │
│  │    deep_judge_cot/            │  │    critique_fix/          │  │
│  └───────────────────────────────┘  └───────────────────────────┘  │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
MERGE + FORMAT LAYER (v1.2, new or extended script)
┌─────────────────────────────────────────────────────────────────────┐
│  merge_reasoning_dataset.py  (NEW SCRIPT or extend merge_dataset.py)│
│                                                                     │
│  Combines:                                                          │
│    data/phase4_reasoning/deep_judge_cot/   → task_type: wp_judge   │
│    data/phase4_reasoning/critique_fix/     → task_type: wp_judge   │
│                                                                     │
│  Outputs: data/reasoning_dataset/                                   │
│    openai_train.jsonl  (SFT format with <wp_judge> task token)      │
│    openai_val.jsonl    (10% held-out split)                         │
│    metadata.json       (counts, source breakdown)                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
TRAINING LAYER (v1.2, new config + existing train_model.py)
┌─────────────────────────────────────────────────────────────────────┐
│  train_model.py  (EXISTING — load with new config)                  │
│                                                                     │
│  Config: config/train_config_reasoning.yaml  (NEW)                  │
│    base: adapters/qwen3-30b-wp-{winning}/    (start from adapter)   │
│    data: data/reasoning_dataset/openai_train.jsonl                  │
│    run_name: qwen3-30b-wp-{winning}-reasoning                       │
│    lr: lower (1e-5 vs 2e-4) — fine-tuning on fine-tune             │
│    epochs: fewer (1-2 vs 3) — small dataset, avoid forgetting       │
│                                                                     │
│  Output: adapters/qwen3-30b-wp-{winning}-reasoning/                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
EVALUATION LAYER (v1.2, existing eval scripts + new reasoning metric)
┌─────────────────────────────────────────────────────────────────────┐
│  eval_judge.py    (EXISTING — re-run, verify Spearman stable/up)    │
│  eval_gen.py      (EXISTING — re-run, confirm PHPCS no regression)  │
│  eval_gate.py     (EXISTING — gate check on both)                   │
│                                                                     │
│  eval_reasoning.py  (NEW — optional)                                │
│    Checks: does judge response contain reasoning structure?          │
│    Checks: are dimension-level rationales present before scores?     │
│    Metric: reasoning_present_rate (% of responses with chain)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown: New vs Modified vs Unchanged

### New Scripts (must be created)

| Script | Location | Purpose | Inputs | Outputs |
|--------|----------|---------|--------|---------|
| `phase4_deep_judge_cot.py` | `scripts/` | Generate judge training with full reasoning chains | `passed/`, `failed/`, or `judge_training/` | `data/phase4_reasoning/deep_judge_cot/*.json` |
| `phase4_critique_fix.py` | `scripts/` | Generate critique-then-fix pairs from mutations | `data/phase2_synthetic/output/mutated/` | `data/phase4_reasoning/critique_fix/*.json` |
| `merge_reasoning_dataset.py` | `scripts/` | Merge + format new reasoning data for SFT | `data/phase4_reasoning/` | `data/reasoning_dataset/` |
| `eval_reasoning.py` | `eval/` | Verify reasoning structure quality (optional) | Model inference output | Reasoning-present rate metric |

### Modified Scripts (light changes)

| Script | Change Required | Why |
|--------|----------------|-----|
| `train_model.py` | Accept `--config` flag pointing to new YAML | Already supports this via `CONFIG_PATH` override; verify it works with adapter-as-base loading pattern |
| `pipeline_orchestrator.py` | Add Phase 4 awareness | Track `data/phase4_reasoning/` counts, add v1.2 plan action items |
| `docs/AGENT_PIPELINE.md` | Add Phase 4 agent type entries | Document new agent batch patterns for deep judge CoT and critique-then-fix |

### New Configuration (must be created)

| File | Location | Purpose |
|------|----------|---------|
| `train_config_reasoning.yaml` | `config/` | Training config for reasoning fine-tune: lower LR, fewer epochs, reasoning dataset path, winning adapter as base |

### Unchanged (use as-is)

| Component | Used By v1.2 How |
|-----------|-----------------|
| `eval_judge.py` | Re-run against reasoning adapter; compare Spearman to baseline |
| `eval_gen.py` | Re-run to confirm PHPCS pass rate not regressed |
| `eval_gate.py` | Gate check on combined metrics |
| `scripts/merge_adapter.py` | Merge reasoning adapter into base if eval passes |
| `adapters/qwen3-30b-wp-{winning}/` | Starting point for reasoning fine-tune |
| `data/phase2_synthetic/output/mutated/` | Source material for critique-then-fix |
| `data/phase1_extraction/output/{passed,failed}/` | Source material for deep judge CoT |
| `config/judge_system.md` | System prompt for deep judge CoT generation (same rubric, different response format) |

## Data Format Specifications

### Deep Judge CoT Format

The existing judge training format emits flat scores + brief explanation:

```json
{
  "task_type": "wp_judge",
  "instruction": "<wp_judge> Evaluate this WordPress code:\n\n[body]",
  "response": {
    "overall_score": 82,
    "wpcs_compliance": 85,
    "security": 90,
    "explanation": "Good security but minor WPCS issues."
  },
  "quality_tier": "high"
}
```

The v1.2 deep judge CoT format adds a reasoning chain before the scores. The response becomes a multi-section text, not a raw JSON object, because the reasoning chain precedes the scores:

```json
{
  "task_type": "wp_judge",
  "instruction": "<wp_judge> Evaluate this WordPress code:\n\n[body]",
  "response": "## Dimension Analysis\n\n**WPCS Compliance:** The function follows lowercase_with_underscores naming... [3-5 sentences per dimension]\n\n**Security:** Nonce verification present via check_ajax_referer()... [3-5 sentences]\n\n**Performance:** Uses transient caching correctly...\n\n## Issues Identified\n\n1. Missing @param in PHPDoc — prevents static analysis from inferring types\n2. SELECT * on meta join — should target specific columns\n\n## Fix Suggestions\n\n1. Add @param string $post_id and @return array to the docblock\n2. Replace SELECT * with SELECT p.ID, p.post_title\n\n## Scores\n\n```json\n{\"overall_score\": 82, \"wpcs_compliance\": 85, ...}\n```",
  "quality_tier": "high",
  "source": "deep_judge_cot"
}
```

Key design decisions:
- Reasoning chain is part of the `response` field, not a separate `cot_reasoning` field — the model should learn to produce reasoning inline as part of judging, not as a metadata field
- Score JSON embedded at end of response in a fenced block — consistent with existing `parse_judge_response()` in `eval_judge.py` which already handles fenced JSON extraction
- Task token `<wp_judge>` remains in the instruction — routing behavior unchanged

### Critique-Then-Fix Format

```json
{
  "task_type": "wp_judge",
  "instruction": "<wp_judge> This WordPress code has a defect. Identify the problem, explain why it is dangerous, and provide the corrected version:\n\n```php\n[defective_body]\n```",
  "response": "## Critique\n\n**Defect category:** SQL injection\n**Dimension affected:** SQL Safety (D2), Security (D3)\n**Severity:** Critical\n\n**What is wrong:** The query concatenates `$user_id` directly into the SQL string without using `$wpdb->prepare()`. An attacker who can control this input can execute arbitrary SQL.\n\n**Why it is dangerous:** [2-3 sentences on exploitation vector]\n\n## Corrected Version\n\n```php\n[corrected_body]\n```\n\n**Changes made:** Replaced direct concatenation with `$wpdb->prepare( '... WHERE id = %d', $user_id )`. The `%d` placeholder enforces integer casting before the query executes.",
  "quality_tier": "critique_fix",
  "mutation_type": "sql_injection",
  "source_function": "function_name",
  "source": "critique_fix"
}
```

Key design decisions:
- Task token is `<wp_judge>`, not `<wp_gen>` — critique-then-fix is a judge capability (analyze defects), not a generation capability
- Instruction explicitly states "defect present" — this trains the model to produce a critique when it knows something is wrong, not to guess
- Corrected version is the original passed function from phase2_mutate.py (already available in `mutated/` alongside the defective version)
- Mutation type metadata enables filtering by defect category for targeted evaluation

### Reasoning Dataset Merge

The `data/reasoning_dataset/` directory is analogous to `data/final_dataset/` but scoped to reasoning-only examples:

```
data/reasoning_dataset/
├── openai_train.jsonl    (80% split, SFT format)
├── openai_val.jsonl      (20% split — larger val for quality monitoring)
└── metadata.json         {
                            "total": N,
                            "deep_judge_cot": N,
                            "critique_fix": N,
                            "split": "80/20",
                            "source_adapter": "qwen3-30b-wp-{winning}"
                          }
```

Intentionally 80/20 (vs 80/10/10 for main dataset) — the reasoning dataset is smaller, so a larger val slice gives more reliable perplexity tracking during the fine-tune.

## Data Flow

### v1.2 End-to-End Flow

```
Phase 2 mutations (existing)               Phase 1 passed + failed (existing)
data/phase2_synthetic/output/mutated/       data/phase1_extraction/output/{passed,failed}/
        │                                              │
        ▼                                              ▼
phase4_critique_fix.py               phase4_deep_judge_cot.py
(agents: given defective+good pair,  (agents: given code + existing scores,
 write critique and fix)              regenerate response with reasoning chain)
        │                                              │
        ▼                                              ▼
data/phase4_reasoning/critique_fix/  data/phase4_reasoning/deep_judge_cot/
        │                                              │
        └──────────────────┬────────────────────────────┘
                           ▼
                merge_reasoning_dataset.py
                (format as SFT messages, split 80/20, write metadata)
                           │
                           ▼
                data/reasoning_dataset/
                openai_train.jsonl + openai_val.jsonl
                           │
                           ▼
                train_model.py --config config/train_config_reasoning.yaml
                (LoRA SFT on winning adapter, lower LR, 1-2 epochs)
                           │
                           ▼
                adapters/qwen3-30b-wp-{winning}-reasoning/
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               eval_judge.py  eval_gen.py
               (Spearman)     (PHPCS pass rate)
                    │             │
                    └──────┬──────┘
                           ▼
                      eval_gate.py
                (gate: Spearman >= baseline, PHPCS >= baseline)
                           │
                  PASS ────┘──── FAIL → investigate forgetting
                   │
                   ▼
            merge_adapter.py
            models/qwen3-30b-wp-{winning}-reasoning-merged/
```

## Build Order for v1.2

Dependencies flow strictly: data generation must complete before merge, merge before training, training before eval.

```
Step 1: Identify winning adapter
   Run: scripts/run_eval_triage.py + scripts/triage_ratios.py
   Output: output/triage_decision.md (identifies winning ratio)
   Gate: must complete before Step 2 (determines which adapter to fine-tune)

Step 2a: Generate deep judge CoT data (parallel with 2b)
   Run: scripts/phase4_deep_judge_cot.py
   Inputs: data/phase1_extraction/output/{passed,failed}/
   Output: data/phase4_reasoning/deep_judge_cot/
   Agent pattern: same spawn-until-target as existing pipeline
   Target: ~5,000-10,000 examples (10% of judge_training pool with reasoning)

Step 2b: Generate critique-then-fix data (parallel with 2a)
   Run: scripts/phase4_critique_fix.py
   Inputs: data/phase2_synthetic/output/mutated/
   Output: data/phase4_reasoning/critique_fix/
   Agent pattern: 1 agent per mutation type batch
   Target: ~1,000-2,000 examples (bounded by mutation pool size)

Step 3: Merge reasoning dataset
   Run: scripts/merge_reasoning_dataset.py
   Inputs: data/phase4_reasoning/deep_judge_cot/ + critique_fix/
   Output: data/reasoning_dataset/
   Gate: requires steps 2a and 2b complete

Step 4: Reasoning fine-tune
   Run: python -m scripts.train_model --config config/train_config_reasoning.yaml
   Inputs: adapters/qwen3-30b-wp-{winning}/ + data/reasoning_dataset/openai_train.jsonl
   Output: adapters/qwen3-30b-wp-{winning}-reasoning/
   Gate: requires step 3 complete

Step 5: Eval verification
   Run: python -m eval.eval_judge + python -m eval.eval_gen + python -m eval.eval_gate
   Inputs: adapters/qwen3-30b-wp-{winning}-reasoning/ (via vLLM LoRA serving)
   Gate: Spearman >= baseline (from phase 4 triage), PHPCS >= 95%

Step 6: Merge adapter (if eval passes)
   Run: python -m scripts.merge_adapter
   Output: models/qwen3-30b-wp-{winning}-reasoning-merged/
```

Steps 2a and 2b are the only parallel steps. All others are sequential with hard gates.

## Integration Points with Existing Pipeline

### What Phase 4 Data Generation Inherits Unchanged

| Existing mechanism | How phase4_deep_judge_cot.py uses it |
|-------------------|--------------------------------------|
| `config/judge_system.md` | Same rubric definitions and 9 dimensions; only the response format changes |
| `scripts/utils.py` | `extract_json`, `call_with_backoff`, `load_checkpoint`, `save_checkpoint` — all reusable |
| Agent spawn-until-target pattern | Same pattern from `docs/AGENT_PIPELINE.md` — agents read batch, write JSON, orchestrator checks targets |
| `<wp_judge>` task token | Unchanged; all new examples tagged with `<wp_judge>` |
| Mutation catalog from phase2_mutate.py | critique_fix.py reads `mutated/` directory; mutation type is stored in each file entry |

### What phase4_deep_judge_cot.py Must NOT Inherit

| Pattern | Why to break it |
|---------|----------------|
| `response` field as dict (scores only) | v1.2 response is a multi-section text string; storing as dict breaks the reasoning chain |
| `explanation` field as 2-3 sentences | Deep CoT replaces the short explanation with full dimension-by-dimension reasoning — must not fall back to short form |
| Same judge system prompt from `judge_system.md` | The prompt instructs Claude to produce a JSON response; v1.2 needs a different prompt that elicits the reasoning-first format |

### train_model.py Adapter-as-Base Pattern

Existing `train_model.py` always starts from `models/Qwen3-30B-A3B/` (the base model). For v1.2, it must start from the winning ratio adapter. Two patterns are viable:

**Option A — Continue training on the existing adapter (preferred):** Load the base model, apply the existing LoRA adapter, then start a new LoRA training pass. The new LoRA is initialized randomly on top of the frozen base + existing adapter. This preserves the original adapter as a checkpoint.

**Option B — Train on the merged model:** Merge the winning adapter first (`merge_adapter.py`), then train the merged model as the new "base". This avoids nested LoRA but burns the clean merge checkpoint.

Option A is preferred because it keeps the original adapter intact for rollback and is how continued fine-tuning is typically done with PEFT. The `train_config_reasoning.yaml` must specify:
- `base_model`: path to merged model or adapter (clarify with Unsloth docs — Unsloth may require merged weights for continued training)
- `lora_r`: 16 or 32 (smaller than v1.0 r=32 is fine for a narrow capability addition)
- `learning_rate`: 1e-5 (10x lower than typical first fine-tune, to avoid catastrophic forgetting)
- `num_epochs`: 1-2 (reasoning dataset is small; more epochs risk overfitting)

### eval_judge.py Compatibility

`eval_judge.py` already handles fenced JSON extraction in `parse_judge_response()` — it tries raw JSON, then ` ```json ``` ` blocks, then generic ` ``` ``` ` blocks, then regex JSON extraction. The deep judge CoT format embeds scores in a fenced block at the end of the response. This is compatible with existing parsing without modification.

The Spearman correlation computed by `eval_judge.py` depends only on the extracted numeric scores, not on the reasoning text. No changes needed to the eval script.

## Architectural Patterns for v1.2

### Pattern 1: Reasoning-First Response Format

**What:** Training examples for judge capability use a structured text response where reasoning precedes scores. The model learns to articulate analysis before committing to numbers.
**When to use:** When the downstream use case requires explainability (code reviews, developer feedback) rather than just a score.
**Trade-offs:** Longer responses increase token cost at inference time. The response format is harder to parse than raw JSON — mitigation is that `parse_judge_response()` already handles embedded JSON extraction.

### Pattern 2: Mutation Reuse for Critique-Then-Fix

**What:** The phase2 contrastive mutations (7 mutation types: sql_injection, csrf, xss, authorization, input_validation, i18n, performance) are re-used as source material for critique-then-fix. The defective code and the original correct code are already paired in `mutated/`.
**When to use:** When generating realistic defective code is expensive — mutations provide controlled, verifiable defects at zero additional generation cost.
**Trade-offs:** The mutation catalog is limited to 7 programmatic patterns. Real-world defects are more varied. Mitigation: supplement with a small number of agent-generated defect examples targeting complex multi-dimension failures not covered by single mutations.

### Pattern 3: Narrow Dataset + Lower Learning Rate

**What:** The reasoning dataset is intentionally small (6,000-12,000 examples vs 43K-102K for v1.0) and trained with a lower learning rate on top of an existing adapter.
**When to use:** When adding a focused capability to a model that already performs well on the base task. The goal is capability addition, not capability replacement.
**Trade-offs:** Small dataset + low LR risks the model not fully generalizing the reasoning format. Mitigation: monitor val perplexity closely; if it plateaus early, stop training to avoid the SFT forgetting curve. Larger reasoning datasets are better if generation budget allows.

## Anti-Patterns

### Anti-Pattern 1: Mixing Reasoning and Non-Reasoning Judge Examples in One Dataset

**What people do:** Combine the new deep judge CoT examples with the existing ~143K flat judge training examples and re-train from scratch.
**Why it is wrong:** The existing flat judge examples (response = short JSON) and the new reasoning examples (response = multi-section text) have incompatible response formats. A model trained on both will learn to produce either format unpredictably. The existing training already produced a well-calibrated judge; the v1.2 goal is capability addition, not replacement.
**Do this instead:** The reasoning dataset (`data/reasoning_dataset/`) is separate from the main dataset (`data/final_dataset/`). The fine-tune starts from the winning adapter, not from scratch. The two datasets never merge.

### Anti-Pattern 2: Using `<wp_gen>` Token for Critique-Then-Fix

**What people do:** Tag critique-then-fix examples as `<wp_gen>` because the output contains code.
**Why it is wrong:** Critique-then-fix is fundamentally a judging task — the model must analyze defects before generating the fix. Using `<wp_gen>` puts this capability in the wrong expert routing pathway and prevents it from being strengthened by v2.0 MoE-Sieve judge-expert selection.
**Do this instead:** Use `<wp_judge>` for all critique-then-fix examples. The corrected code is the conclusion of the critique, not a standalone generation.

### Anti-Pattern 3: Regenerating All 143K Judge Training Examples

**What people do:** Re-run phase2_judge_dataset.py over all 143K examples with a new prompt that elicits reasoning chains.
**Why it is wrong:** This replicates 40+ hours of agent work to produce data that already exists in flat form. The v1.2 goal is targeted capability addition on a representative subset, not full regeneration.
**Do this instead:** Sample a representative subset (10% of the judge training pool, ~14,000 examples) for deep CoT generation. Use percentage-based targets consistent with the existing pipeline design philosophy (decision: "Percentage-based pipeline targets" in PROJECT.md).

### Anti-Pattern 4: Training the Reasoning Fine-Tune for Too Many Epochs

**What people do:** Apply the same epoch count used for the v1.0 fine-tune (3 epochs).
**Why it is wrong:** The reasoning dataset is 5-20x smaller than the v1.0 training data. At 3 epochs on a small dataset, the model will overfit to the reasoning format and lose calibration on the underlying judge scores — exactly the regression that eval_judge.py will catch.
**Do this instead:** Train for 1-2 epochs maximum. Monitor validation perplexity and stop early if it stops decreasing. The eval gate (eval_judge.py Spearman + eval_gen.py PHPCS) will catch any regression before the adapter is merged.

### Anti-Pattern 5: Skipping eval_gen.py After Reasoning Fine-Tune

**What people do:** Only run eval_judge.py after the reasoning fine-tune, since the focus is on judge capability.
**Why it is wrong:** Catastrophic forgetting in SFT is not dimension-specific — fine-tuning on judge reasoning data can degrade generation quality if the learning rate is too high or epochs too many. Generation regression may not be visible in eval_judge.py output.
**Do this instead:** Always run both eval_judge.py and eval_gen.py after any fine-tune. The eval_gate.py script already gates on both. This is a hard requirement for v1.2.

## File System Layout (v1.2 Additions)

```
wp-finetune/
├── scripts/
│   ├── phase4_deep_judge_cot.py     NEW — deep judge CoT generation
│   ├── phase4_critique_fix.py       NEW — critique-then-fix generation
│   └── merge_reasoning_dataset.py   NEW — merge + format reasoning data
├── eval/
│   └── eval_reasoning.py            NEW (optional) — reasoning structure check
├── config/
│   └── train_config_reasoning.yaml  NEW — reasoning fine-tune hyperparameters
├── data/
│   ├── phase4_reasoning/            NEW OUTPUT DIR
│   │   ├── deep_judge_cot/          deep judge CoT JSON batches
│   │   └── critique_fix/            critique-then-fix JSON batches
│   └── reasoning_dataset/           NEW OUTPUT DIR
│       ├── openai_train.jsonl        SFT-ready training split
│       ├── openai_val.jsonl          validation split
│       └── metadata.json
└── adapters/
    └── qwen3-30b-wp-{winning}-reasoning/   NEW ADAPTER
```

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Integration points | HIGH | Direct codebase inspection; existing format contracts verified |
| Data format compatibility | HIGH | `parse_judge_response()` handles fenced JSON; task tokens unchanged |
| Build order dependencies | HIGH | Standard pipeline dependency graph; no external unknowns |
| Training config for adapter-on-adapter | MEDIUM | Unsloth PEFT stacking behavior on Qwen3 MoE needs verification; Option A (load merged) vs Option B (load adapter) requires testing |
| Target example counts | MEDIUM | Derived from existing pool sizes and percentage-based design philosophy; exact counts depend on mutation pool verification |
| Catastrophic forgetting risk | MEDIUM | General SFT forgetting literature supports concern; actual severity on Qwen3-30B-A3B with LoRA at low LR is not directly verified |

## Sources

- Existing codebase: `scripts/phase2_mutate.py`, `scripts/phase2_judge_dataset.py`, `scripts/phase3_cot.py`, `scripts/merge_dataset.py`, `eval/eval_judge.py`, `docs/AGENT_PIPELINE.md` — HIGH confidence, direct inspection
- `data/phase3_cot/output/` file listing — confirms existing CoT types and batch structure
- `data/final_dataset/` structure — confirms merge output format
- `adapters/` listing — confirms winning ratio candidates
- `PROJECT.md` v1.2 milestone spec — defines deep judge CoT and critique-then-fix requirements

---
*Architecture research for: wp-qwen3-moe v1.2 Judge Reasoning Fine-Tune integration*
*Researched: 2026-04-04*
