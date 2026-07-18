# Reward-Validity Oracle — Step-1 finding (Phase 08.2, 2026-07-01)

**The gate 08.1 was missing**: does a candidate RL reward's per-checkpoint trajectory rank-correlate
with the validated target (judge teacher-Spearman)? If not, optimizing it cannot move the target.

Run: `scripts/_reward_validity_oracle.py` (CPU, $0). Inputs: 11 existing seedA captures
(`output/rl_eval/*/judge_responses.jsonl`), teacher GT from `openai_val.jsonl`, fix-corr series
from `READS_TALLY.md`. n_common_aligned=85. Output: `output/reward_validity/reward_validity_oracle.json`.

## Result — correlation of each reward's checkpoint trajectory vs teacher-Spearman
| reward form | Spearman vs target | 95% CI | valid (CI_lo>0) |
|---|---:|---|---|
| **fix_correctness (seedA's optimized proxy)** | **−0.240** | [−0.872, +0.417] | **NO — Goodhart confirmed** |
| **pairwise_rank_agreement vs teacher** | **+0.700** | [+0.147, +0.935] | **YES** |
| in_group_spearman vs teacher | +1.000 | [1.0, 1.0] | YES (by construction = batch-level target) |
| listwise_ndcg@10 | +0.046 | [−0.710, +0.780] | no |
| neg_abs_calibration (point) | +0.300 | [−0.454, +0.916] | no |

## Verdict
1. **Root cause PROVEN empirically.** The reward seedA maximized (fix-correctness) does NOT track
   teacher-Spearman — correlation is slightly NEGATIVE. The flat/failed RLEV-01 result was structurally
   guaranteed: the reward and the target were ~orthogonal. No LR change fixes this.
2. **Replacement identified: per-group pairwise rank-agreement vs teacher GT** — the only DENSE,
   per-completion-decomposable form that provably tracks the target (corr +0.70, CI lower +0.15>0).
   `in_group_spearman` is the batch-level target itself (valid but not per-group-dense → use as the
   acceptance metric, not the gradient signal). Point calibration + NDCG rejected (CI includes 0).

## Caveats / ongoing use
- n=11 checkpoints from ONE run (all near each other) → wide CIs; this is directional + a root-cause
  confirmation, not a high-power estimate. Expand the checkpoint/behavior set for future candidates.
- The calibration forms correlate by construction (monotone w/ Spearman) — that is the DESIRED property;
  the discriminating result is fix-correctness's non-correlation.
- **Standing rule (the 08.2 gate): no reward goes to GPU until its oracle corr CI-lower > 0.**
  pairwise_rank_agreement PASSES this gate; fix-correctness FAILS it.
