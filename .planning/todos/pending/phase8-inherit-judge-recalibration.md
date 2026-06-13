---
id: phase8-inherit-judge-recalibration
created: 2026-06-14
source_phase: "04.4"
resolves_phase: "8"
priority: high
tags: [reward-pipeline, judge-calibration, D-V4-09]
---

# Phase 8 reward pipeline MUST inherit the v4-winner judge calibration offset

**Decision:** D-V4-09 (`.planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-CONTEXT.md`).

The Phase 04.4 mechanism diagnosis established that the merged v4-winner endpoint applies a
**significant, rank-preserving judge calibration offset of −3.58 pt** vs the pre-merge reference
(95% CI [−6.09, −1.24], ~2.9 SE). The correction constant is frozen in:

  `output/eval_reasoning_v4_winner/judge_recalibration.json`  →  `score_offset = +3.58`

**Requirement:** Phase 8's composite reward pipeline (`scripts/reward_pipeline.py`) uses the frozen
local `wp_judge` model for the **30% judge reward component** (D-11). That judge component MUST apply
the `+3.58` recalibration (equivalent: evaluate any absolute pass/fail at threshold 70−3.58 = 66.42)
so the RL-reward-time judge calibration matches the 04.4 gate-time calibration. Without it, Phase 8
scores rollouts with the uncorrected (−3.58pt-stricter) judge while 04.4 promoted under the corrected
one — gate/reward divergence.

**Note (mechanics):** the offset is rank-invariant — under MO-GRPO within-group normalization a uniform
offset largely cancels in the *relative* reward, so the impact is concentrated in any **absolute**
judge threshold the reward applies (hard pass/fail cutoffs, anti-hack gates). Apply it there.

**Acceptance:** Phase 8 reward-pipeline spec/PR references `judge_recalibration.json` as a hard input;
a test asserts the judge component applies the +3.58 offset (or the 66.42 effective threshold).
