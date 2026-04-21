---
quick_task: 260422-p41-review-followups
created_at: 2026-04-22
status: in-progress
phase: 4.1 review follow-ups
---

# Phase 4.1 Review Follow-ups — Execution Plan

Execute 4 action items from the Gemini + Codex cross-review sequentially.

## Current State (as of 2026-04-22 investigation)

| Metric | Value |
|--------|-------|
| CoT examples (final batches) | 1,163 |
| CtF examples (final batches) | 200 |
| Total | 1,363 |
| CoT function IDs | 196 unique |
| CtF function IDs | 180 unique |
| CoT _input_batches (no reasoning) | 280 items (intermediate, not in final count) |
| Citation audit subset | 196 examples (deep_judge_cot_bulk.json) |
| Pilot CoT | 20 examples, 9 dims covered, citation_validity=0.85 |
| ACCEPTANCE REPORT | MISSING (no 04.1-OUTPUTS file exists) |

## Gap Investigation Findings

**The 171 gap**: 1,163 generated - 992 judged (741 PASS + 251 FAIL) = 171
- The 741/251 comes from judge evaluation, not from generation
- 171 = judge parse failures or abstentions (no parseable verdict extracted)
- The `_input_batch_*.json` files (280 items) are intermediate orchestrator state, NOT part of the 1,163
- **No explicit rejection log exists** — the gap is unexplained in the pipeline

## Tasks

### 1. Account for 171-example gap
**Goal**: Document what the 171 gap is and create an explanation
**Approach**: Analyze judge parse rate, check for rejection logs, document the gap

### 2. Freeze global exclusion manifest
**Goal**: Lock function_id manifest for train/eval contamination prevention
**Approach**: Merge CoT + CtF IDs, write to canonical location with documentation

### 3. Audit 2-exemplar vs full-seed
**Goal**: Compare reasoning quality between 2-exemplar bulk and pilot
**Approach**: Sample matched functions, compare verdict agreement, rubric coverage, citation accuracy

### 4. Draft acceptance report
**Goal**: Produce final Phase 4.1 acceptance report with explicit denominators
**Approach**: Write report with generated/parsed/accepted/rejected/deduped/audited/exported counts
