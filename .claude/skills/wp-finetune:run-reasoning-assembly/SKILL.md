# Skill: wp-finetune:run-reasoning-assembly

Run the reasoning dataset assembly pipeline end-to-end: consistency validation, canonical template formatting, training mix assembly, stratified split, and export. Single invocation, no prompting required.

## Telemetry

> **Recommended:** Say `/observe-data-pipeline` before starting to spawn background telemetry agents.
> Optional — the pipeline runs fine without it. Output: `telemetry/reasoning-assembly/{timestamp}/`

## Trigger

User says: "run reasoning assembly", "assemble reasoning dataset", "/run-reasoning-assembly"

## Process

### 1. Run Consistency Validation

Execute `scripts/validate_reasoning_consistency.py` to validate all reasoning examples through Claude Code agents.

```bash
python scripts/validate_reasoning_consistency.py --source both
```

This reads CoT examples from `data/phase4_reasoning/deep_judge_cot/deep_judge_cot_bulk.json` and CtF examples from `data/phase4_reasoning/critique_then_fix/critique_then_fix_bulk.json`, routes ALL through Claude Code agents (no heuristic pre-filter per D-01), and writes:
- `data/reasoning_dataset/consistency_valid.jsonl` — examples that passed validation
- `data/reasoning_dataset/consistency_rejected.jsonl` — examples with score-reasoning contradictions

Display the validation results to the user.

### 2. Run Assembly

Execute `scripts/assemble_reasoning_dataset.py` to assemble and export the dataset.

```bash
python scripts/assemble_reasoning_dataset.py
```

This reads consistency-valid examples, applies canonical template formatting (dimension-by-dimension analysis prose + `[/REASONING]` separator + `<judge_output>` JSON block), assembles the 60/25/15 training mix (CoT/CtF/replay), performs stratified 80/20 split, and exports:
- `data/reasoning_dataset/openai_train.jsonl` — 80% training data
- `data/reasoning_dataset/openai_val.jsonl` — 20% validation data
- `data/reasoning_dataset/metadata.json` — rejection stats, mix percentages, taxonomy coverage

### 3. Verify Output

Check metadata.json ratios and verify template compliance on sample.

```bash
python -c "
import json
meta = json.load(open('data/reasoning_dataset/metadata.json'))
print(f\"Total: {meta['total_examples']}\")
print(f\"Mix: CoT={meta['mix']['cot_percent']}%, CtF={meta['mix']['ctf_percent']}%, Replay={meta['mix']['replay_percent']}%\")
print(f\"Train: {meta['split']['train_count']}, Val: {meta['split']['val_count']}\")
print(f\"Rejections: {meta['rejection_counts']}\")
"
```

Verify template compliance:

```bash
head -3 data/reasoning_dataset/openai_train.jsonl | python -c "
import sys, json
for line in sys.stdin:
    ex = json.loads(line)
    content = ex['messages'][1]['content']
    has_sep = '[/REASONING]' in content
    has_judge = '<judge_output>' in content
    print(f\"  separator={has_sep}, judge_output={has_judge}\")
"
```

### 4. Regenerate Rejected Examples (if needed)

If rejection count is non-zero and needs to be reduced, re-queue rejected examples for regeneration:

```bash
python scripts/validate_reasoning_consistency.py --source both --auto-regenerate
```

This writes `data/reasoning_dataset/requeue_for_regeneration.json` with `regenerate_prompt` field containing corrected guidance for each rejected example.

After regeneration, re-run consistency validation and assembly.

## Key Rules

- ALL LLM work via Claude Code agents (no Anthropic API calls) — per D-01, no deterministic heuristics for consistency validation
- Non-LLM steps (template formatting, splitting, JSONL writing) run as Python scripts directly
- Spawn agents in parallel (run_in_background=true) for regeneration — each batch of ~20 examples is an agent call
- Pattern: validate -> assemble -> verify -> regenerate if needed
- Output: `data/reasoning_dataset/` with `openai_train.jsonl`, `openai_val.jsonl`, and `metadata.json`
- Every example MUST go through a Claude Code agent for consistency judgment — no auto-pass or auto-fail
- Canonical template enforcement (D-04): every example must have dimension-by-dimension analysis prose, `[/REASONING]` separator, and `<judge_output>` JSON block

## Outputs

| File | Description |
|------|-------------|
| `data/reasoning_dataset/consistency_valid.jsonl` | Examples passing agent-validated consistency |
| `data/reasoning_dataset/consistency_rejected.jsonl` | Examples with score-reasoning contradictions |
| `data/reasoning_dataset/openai_train.jsonl` | 80% training data in OpenAI JSONL format |
| `data/reasoning_dataset/openai_val.jsonl` | 20% validation data in OpenAI JSONL format |
| `data/reasoning_dataset/metadata.json` | Rejection counts, mix percentages, taxonomy coverage |
| `data/reasoning_dataset/requeue_for_regeneration.json` | (Optional) Rejected examples for regeneration |

## Targets

| Metric | Target |
|--------|--------|
| Consistency validation | 100% agent-validated (no heuristic pre-filter) |
| Training mix | CoT ~60%, CtF ~25%, Replay ~15% |
| Split ratio | 80/20 (stratified by domain) |
| Template compliance | Every example has `[/REASONING]` + `<judge_output>` |
| Rejection rate | 0-5% of input (expected natural variation) |

Run `python scripts/validate_reasoning_consistency.py` and `python scripts/assemble_reasoning_dataset.py` to execute.
