# Audit: 2-Exemplar Prompt vs Full-Seed Pilot

**Date**: 2026-04-22  
**Audited by**: Quick task 260422-p41-review-followups

## Setup

| | Pilot (seed_few_shot_agent) | Bulk (claude_code_agent_real_v3) |
|--|--|---|
| Exemplars | 118 golden seeds (full set available) | 2 inline exemplars |
| N | 20 | 196 (citation audit subset) |
| Sample type | Human-curated boundary cases | Randomly sampled from Phase 1 |

## Findings

### 1. Rubric Coverage — CONCERN

| Metric | Pilot | Bulk |
|--|--|
| 9-dim coverage rate | 20/20 (100%) | 166/196 (84.7%) |
| 6-dim | 0/20 | 10/196 (5.1%) |
| 7-dim | 0/20 | 11/196 (5.6%) |
| 8-dim | 0/20 | 3/196 (1.5%) |
| 5-dim | 0/20 | 6/196 (3.1%) |

**Finding**: The pilot achieved 100% full-rubric coverage. The bulk shows 15.3% examples addressing fewer than 9 dimensions. This suggests the 2-exemplar prompt occasionally collapses rubric breadth.

### 2. Citation Accuracy — IMPROVEMENT

| Metric | Pilot | Bulk |
|--|--|
| Mean hallucination_ratio | 0.118 | 0.001 |
| Zero-hallucination rate | 13/20 (65%) | 195/196 (99.5%) |
| Max hallucination_ratio | 0.500 | 0.250 |
| >0 hallucination | 7/20 (35%) | 1/196 (0.5%) |

**Finding**: Citation accuracy improved dramatically in bulk output. This may be because the 2-exemplar prompt was specifically designed to minimize hallucinated citations.

### 3. Matched Functions (N=2)

| Function | Pilot dims | Bulk dims | Pilot cit_acc | Bulk cit_acc |
|--|--|---|---|---|
| DatabaseChanges::getCreateQueryFieldConfigDefaultsValues | 9 | 9 | 0.2 | 0.0 |
| get_question_object | 9 | 9 | 0.0 | 0.0 |

Both matched functions received identical rubric coverage. Citation accuracy improved for one (0.2→0.0) and stayed perfect for the other.

### 4. Generation Method Difference

- Pilot: `seed_few_shot_agent` — uses 118 golden seeds with full context
- Bulk: `claude_code_agent_real_v3` — uses 2 inline exemplars, context-reduced from ~241KB to ~3KB

## Conclusion

The 2-exemplar prompt **improved citation accuracy** (near-perfect) but **slightly reduced rubric coverage** (15.3% of bulk examples address fewer than all 9 dimensions). The rubric reduction is minor but not negligible — it should be monitored. The citation improvement is significant and likely intentional.

## Recommendation

1. The 2-exemplar prompt is acceptable for bulk generation IF rubric coverage ≥ 8 dimensions
2. For Phase 4.2+ generation, consider rotating exemplars (as Gemini suggested) to prevent style collapse
3. Require `dimensions_addressed.length >= 8` as a hard acceptance gate
