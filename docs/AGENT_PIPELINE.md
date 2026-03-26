# Claude Code Agent Pipeline

This document describes the agent-based execution model used for all LLM-heavy pipeline steps. Instead of calling the Anthropic API directly, this project uses Claude Code agents (covered by subscription) to process data in parallel batches.

## Why Agents Over API

| Dimension | Anthropic Batch API | Claude Code Agents |
|-----------|--------------------|--------------------|
| Cost | Pay-per-token (~$35-60/run) | $0 (subscription) |
| Parallelism | Native batch parallelism | Spawn N agents in parallel |
| Context | 200K per request | Full agent context per batch |
| Reliability | Checkpoint/resume built-in | Agent failures need re-spawn |
| Quality | Fixed model, fixed prompt | Agent can read files, adapt, iterate |

## Execution Model

### Continuous Spawning Until Target

The pipeline uses a **spawn-until-target** pattern:

1. Define target counts (e.g., 1,500 judge high-score examples)
2. Spawn N parallel agents, each processing a batch
3. When agents complete, check totals against targets
4. If targets not met, spawn more agents for remaining gaps
5. Repeat until all targets are met

```
while target_not_met:
    gap = target - current_count
    batch_size = min(200, gap)
    num_agents = ceil(gap / batch_size)
    spawn(num_agents, batch_size)
    wait_for_completion()
    current_count = recount()
```

### Agent Types by Pipeline Step

#### 1. Code Judging (Phase 1 Judge)
- **Input:** Extracted function JSON from `phase1_extraction/output/extracted/{repo}.json`
- **Rubric:** `config/judge_system.md` (9 dimensions, threshold >= 8, security auto-FAIL)
- **Output:** `phase1_extraction/output/passed/{repo}.json` and `failed/{repo}.json`
- **Batch size:** 1 repo per agent (small repos can be batched)
- **Parallelism:** 4-5 agents

#### 2. Synthetic Generation (Phase 2 Generate)
- **Input:** Gap report from `phase2_synthetic/gap_report.json` + style anchors from passed code
- **Templates:** `config/synthetic_prompts.yaml` (including rejection templates)
- **Output:** `phase2_synthetic/output/generated/{category}_synthetic.json`
- **Batch size:** 20-30 examples per agent, grouped by taxonomy tag
- **Parallelism:** 4-5 agents covering different tag categories

#### 3. Synthetic Judging (Phase 2 Judge)
- **Input:** Generated synthetic files from `phase2_synthetic/output/generated/`
- **Rubric:** Same as Phase 1 judge
- **Output:** `phase2_synthetic/output/judged/passed_synthetic_{batch}.json` and `failed_synthetic_{batch}.json`
- **Batch size:** 50-80 examples per agent
- **Parallelism:** 3 agents

#### 4. Judge Training Data (Phase 2 Judge Dataset)
- **Input:** Passed functions (high-score), failed functions (low-score), judged synthetics
- **Rubric:** 0-100 scale across 6 dimensions
- **Output:** `phase2_synthetic/output/judge_training/{quality}_{batch}.json`
- **Batch size:** 150-200 examples per agent
- **Parallelism:** 3-5 agents per quality tier (high/low/synthetic)
- **Target:** ~1,500 high + ~1,000 low + ~1,500 synthetic = ~4,000 total

#### 5. CoT Reasoning (Phase 3 CoT)
- **Input:** Passed functions (instruction synthesis), failed functions (contrastive), judged synthetics
- **Output:** `phase3_cot/output/cot_{type}_{batch}.json`
- **Types:** real code (instruction-completion), contrastive (bad→fix), synthetic (with rejection CoT)
- **Batch size:** 40-100 examples per agent
- **Parallelism:** 3-5 agents
- **Target:** ~300 real + ~150 contrastive + ~50 synthetic = ~500 total

#### 6. Non-Agent Steps (Pure Python)
These run as regular scripts, no LLM needed:
- `phase1_clone.py` — Git clone
- `phase1_extract.py` — PHP tokenizer extraction
- `phase2_gap_analysis.py` — Taxonomy coverage analysis
- `phase2_mutate.py` — Automated contrastive mutations
- `export_dataset.py` — Format conversion, splits, metadata

## Output Format Contracts

All agents must write output matching the exact JSON format of existing pipeline files. Read one example file before writing to verify structure.

### Passed/Failed Function Format
```json
[
  {
    "function_name": "...",
    "body": "...",
    "source_repo": "...",
    "source_file": "...",
    "quality_tier": "assessed",
    "assessment": {
      "verdict": "PASS|FAIL",
      "scores": { "wpcs_compliance": 1-10, ... },
      "critical_failures": [],
      "training_tags": [],
      "notes": "..."
    }
  }
]
```

### Judge Training Format
```json
{
  "task_type": "wp_judge",
  "instruction": "<wp_judge> Evaluate this WordPress code:\n\n[body]",
  "response": {
    "overall_score": 0-100,
    "wpcs_compliance": 0-100,
    ...6 dimensions...
    "must_fix_issues": [],
    "passes_threshold": true/false,
    "explanation": "..."
  },
  "quality_tier": "high|low"
}
```

### CoT Format
```json
{
  "task_type": "wp_gen",
  "instruction": "<wp_gen> [Prompt]",
  "response": "[Code or review]",
  "cot_reasoning": "Step-by-step:\n1. ...",
  "complexity": "simple|medium|complex"
}
```

## Target Counts (v1)

| Category | Target | Source |
|----------|--------|--------|
| Real code (passed judge) | 15,000+ | Extracted from 60+ repos |
| Synthetic (passed judge) | 200+ | Gap-fill for taxonomy coverage |
| Judge training (high) | ~1,500 | Rubric-scored passed code |
| Judge training (low) | ~1,000 | Rubric-scored failed code |
| Judge training (synthetic) | ~1,500 | Rubric-scored synthetic code |
| CoT (real code) | ~300 | Instruction-completion pairs |
| CoT (contrastive) | ~150 | Bad→fix pairs with explanation |
| CoT (synthetic) | ~50 | Including rejection examples |
| **Total dataset** | **~20,000+** | **40/60 gen/judge split** |

## Scaling for Future Iterations

To increase dataset size in future iterations:

1. **Add more repos:** Update `config/repos.yaml` via `scripts/csv_to_repos.py` with relaxed or expanded filters
2. **Spawn more judge agents:** Each new repo gets judge agents applied
3. **Re-run gap analysis:** `python scripts/phase2_gap_analysis.py` identifies new deficits
4. **Spawn more synthetic agents:** Target the specific taxonomy gaps
5. **Scale judge training:** Proportionally increase high/low/synthetic scoring agents
6. **Scale CoT:** More instruction-completion and contrastive pairs from new passed/failed code

The pattern is always: **identify gap → spawn agents → check totals → repeat**.
