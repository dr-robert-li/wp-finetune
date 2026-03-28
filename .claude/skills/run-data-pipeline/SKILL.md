# Skill: run-data-pipeline

Run the complete wp-qwen3-moe data pipeline end-to-end using Claude Code agents for all LLM work. Single invocation, no prompting required.

## Telemetry

> **Recommended:** Say `/observe-data-pipeline` before starting to spawn background telemetry agents.
> Optional — the pipeline runs fine without it. Output: `telemetry/data-pipeline/{timestamp}/`

## Trigger

User says: "run the pipeline", "generate training data", "run data pipeline", "/run-pipeline"

## Process

### 1. Get Current State

```bash
python scripts/pipeline_orchestrator.py status
```

Display the status to the user.

### 2. Get Action Plan

```bash
PLAN=$(python scripts/pipeline_orchestrator.py plan-json)
```

Parse the JSON plan. If `phase` is `complete`, stop — pipeline is done.

### 3. Execute Actions in Order

For each action in the plan:

**If type is "script":**
Run the command directly:
```bash
python scripts/{script}.py
```

**If type is "agent" and step is "judge_repos":**
For each batch in `batches`, spawn a parallel `general-purpose` agent:
```
Agent(
  description="Judge repos: {batch_repos}",
  prompt="You are a WordPress code quality judge. Read config/judge_system.md.
  Judge all functions in these repos: {batch_repos}.
  For each repo, read data/phase1_extraction/output/extracted/{repo}.json.
  Read one existing passed file first for format: data/phase1_extraction/output/passed/wp-super-cache.json.
  Split into passed/failed, write to data/phase1_extraction/output/passed/{repo}.json and failed/{repo}.json.
  PASS requires ALL scores >= 8, no critical failures. Security < 5 = auto-FAIL.",
  run_in_background=true
)
```

**If type is "agent" and step starts with "judge_training_":**
Spawn `agent_count` agents, each generating `batch_size` examples:
- For `judge_high`: Score PASSED functions 75-100
- For `judge_low`: Score FAILED functions 10-65
- For `judge_synth`: Score mixed PASSED/FAILED on 0-100

**If type is "agent" and step is "synthetic_generation":**
Read the gap report, group gaps by category, spawn agents per category.
Each agent generates 20-30 examples for its assigned taxonomy tags.

**If type is "agent" and step is "judge_synthetics":**
Spawn agents to assess generated synthetic files against the judge rubric.

**If type is "agent" and step is "cot_gen_pattern":**
Spawn agents for Gen Pattern CoT from passed functions.
Each agent reads passed functions from `source_dir`, generates examples:
- instruction: "Implement a WordPress function that [requirement derived from code]"
- reasoning: "Pattern selection → why this WP API → implementation steps → edge cases"
- response: the actual function code
Write to: `{output_dir}/{output_prefix}_batch{N}.json`

**If type is "agent" and step is "cot_judge_rubric":**
Spawn agents for Judge Rubric CoT from mixed passed+failed functions.
Each agent reads functions from both passed/ and failed/ dirs, generates examples:
- instruction: "<wp_judge> Evaluate this WordPress code:\n\n{function body}"
- reasoning: "D1 WPCS check → D2 Security check → D3 SQL → ... → D9 Structure → overall score → verdict"
- response: structured rubric score JSON (matching eval judge training format)
Write to: `{output_dir}/{output_prefix}_batch{N}.json`

**If type is "agent" and step is "cot_judge_contrastive":**
Spawn agents for Judge Contrastive CoT from failed functions.
Each agent reads failed functions from `source_dir`, generates examples:
- instruction: "Review this WordPress code and identify issues:\n\n{bad code}"
- reasoning: "Issue 1: [what's wrong] → Fix: [how to fix] → Issue 2: ..."
- response: fixed version of the code with explanation
Write to: `{output_dir}/{output_prefix}_batch{N}.json`

**If type is "agent" and step is "cot_security":**
Spawn agents for Security CoT from security-tagged functions (both passed and failed).
Each agent reads functions with security surface (superglobals, nonces, escaping), generates examples:
- instruction: "Analyze the security of this WordPress code:\n\n{function body}"
- reasoning: "Nonce check → capability check → input sanitization → output escaping → verdict"
- response: security assessment with specific findings
Write to: `{output_dir}/{output_prefix}_batch{N}.json`

### 4. Wait for All Agents

Wait for all background agents to complete. Report any failures.

### 5. Re-check State

```bash
python scripts/pipeline_orchestrator.py status
```

### 6. Loop If Targets Not Met

```bash
PLAN=$(python scripts/pipeline_orchestrator.py plan-json)
```

If `all_targets_met` is false, go back to step 3 with the new plan.
This is the **spawn-until-target** loop.

### 7. Merge and Export

When all targets are met:

```bash
python scripts/merge_dataset.py  # Merge all sources into wordpress_finetune.jsonl
python scripts/export_dataset.py  # Apply ratio, dedup, splits, formats
```

### 8. Report Final Status

```bash
python scripts/pipeline_orchestrator.py status
```

Display final dataset statistics. Pipeline complete.

## Key Rules

- ALL LLM work via Claude Code agents (no Anthropic API calls)
- Non-LLM steps run as Python scripts directly
- Spawn agents in parallel (run_in_background=true) for throughput
- After each wave of agents, re-run orchestrator to check targets
- Keep spawning until ALL targets in pipeline_orchestrator.py are met
- Judge rubric: config/judge_system.md, threshold >= 8, security auto-FAIL
- Output format: read existing files to match structure exactly
- 40/60 gen/judge split enforced during export

## Targets (percentage-based)

All targets are dynamic — computed from actual judged function counts by `pipeline_orchestrator.py`.

| Category | Formula | Floor |
|----------|---------|-------|
| Real code judging | Judge ALL extracted repos | — |
| Synthetic generation | Fill gap_report.json deficit to 0 | — |
| Judge high-score | 10% of passed functions | — |
| Judge low-score | 10% of failed functions | — |
| Judge synth-scored | 10% of synthetic passed | — |
| CoT: Gen Pattern | max(500, 10% of passed) | 500 |
| CoT: Judge Rubric | max(500, 10% of total judged) | 500 |
| CoT: Judge Contrastive | max(500, 10% of failed) | 500 |
| CoT: Security | max(500, 10% of security-tagged) | 500 |

Run `python scripts/pipeline_orchestrator.py status` to see current values.
