---
phase: 09-gspo-training
plan: "04"
subsystem: rl-rollouts
tags: [rl, grpo, rollouts, reward, advantage, judge-weighted]
dependency_graph:
  requires:
    - scripts/reward_pipeline.py  # Phase 8 — consumed unmodified
    - scripts/rl_judge_dispatch.py  # Phase 9-03 — score_judge_consistency_batch
  provides:
    - scripts/rl_rollouts.py  # interleaved sampling, dual reward, advantage assembly
  affects:
    - scripts/rl_train.py  # Phase 9-05 consumes collect_rollouts + compute_rollout_advantages
tech_stack:
  added: []
  patterns:
    - Group-centred advantage assembly (inline fallback mirrors cookbook semantics)
    - Judge-weighted interleaved sampling (JUDGE_RATIO=0.6, D-09-04)
    - D-09-05 cap guard (module-level assert + call-time ValueError)
    - Security-member drop before advantage (T-09-SECDROP; breakdown.security_fail flag)
key_files:
  created:
    - scripts/rl_rollouts.py
    - tests/test_rl_rollouts.py
  modified: []
decisions:
  - inline fallback for compute_rollout_advantages: tinker_cookbook unavailable at test time;
    fallback implements identical group-centering semantics (A_i=r_i-mean(r), std<epsilon drop)
    so frozen test_rl_train.py tests run without monkeypatching
  - security_fail flag over scalar==0.0: breakdown.security_fail=True is the reliable drop signal;
    scalar==0.0 can occur legitimately when a sample equals its group mean
metrics:
  duration: 14 minutes
  completed: "2026-06-20"
  tasks: 2
  files: 2
---

# Phase 09 Plan 04: RL Rollouts — Interleaved Sampling + Dual Reward + Advantage Assembly

One-liner: Interleaved judge-weighted batch sampler with fix-correctness-anchored judge reward (capped Claude-consistency at <=0.5) and group-centred advantage assembly delegating to the tinker cookbook.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Interleaved sampling + dual reward collection + judge combination | 6b40e37 | scripts/rl_rollouts.py |
| RED | Failing tests for all rollout behaviors | 843784c | tests/test_rl_rollouts.py |
| GREEN | Full implementation (Task 1 + Task 2) | 6b40e37 | scripts/rl_rollouts.py |

## What Was Built

### scripts/rl_rollouts.py

**Interleaved sampling (GRPO-05 / D-09-04):**
- `JUDGE_RATIO = 0.6` module constant; `n_judge = round(batch_size * 0.6)` with defensive clamp ensuring `n_judge >= n_gen` for all `batch_size >= 2`
- `sample_interleaved_prompts(gen_pool, judge_pool, batch_size)` returns judge items first, then gen items; item dicts pass through preserving `"tag"` and all metadata

**D-09-05 guard 1 / T-09-RWD-CAP:**
- `judge_consistency_weight = 0.3` module constant asserted `<= 0.5` at import time (raises `AssertionError` if violated, fires on load not at call time)
- `combine_judge_reward(fix_correctness, consistency, weight=0.3)` raises `ValueError` if `weight > 0.5`; formula is `(1 - weight) * fix_correctness + weight * consistency`

**Security drop (T-09-SECDROP):**
- `build_trajectory_groups(rollouts, rewards)` inspects `reward.breakdown.security_fail`; members with `security_fail=True` are DROPPED (not zeroed) before advantage assembly

**Advantage assembly (Task 2, cookbook delegation):**
- `compute_rollout_advantages(groups)` tries lazy `from tinker_cookbook.rl.data_processing import compute_advantages, remove_constant_reward_groups, assemble_training_data`
- Falls back to inline implementations (`_inline_remove_constant_reward_groups`, `_inline_compute_advantages`, `_inline_assemble_training_data`) that mirror the cookbook semantics exactly
- Returns `(data, meta)` where each `data[i]` dict carries `"advantage"` key; `meta` carries group stats

**MO-GRPO normalization:**
- `_mo_grpo_norm(values)` mirrors `reward_pipeline._mo_grpo_norm` exactly (population std, ddof=0, epsilon=1e-8); used for per-signal normalization in the judge reward path within `collect_rollouts`

**Full rollout collection:**
- `collect_rollouts(sampling_client, gen_pool, judge_pool, args)` wires: interleaved sampling → `compute_group_rewards` (Phase 8 pipeline, unmodified) for gen rewards → fix-correctness (`_extract_verifiable_signals`) + capped `score_judge_consistency_batch` (09-03) for judge rewards → `build_trajectory_groups`

### tests/test_rl_rollouts.py

23 tests across 5 classes:
- `TestInterleaving`: judge>=gen for {2,8,20,21}, total size correct, both tags present, JUDGE_RATIO=0.6
- `TestJudgeCap`: weight<=0.5, combine formula, weight>0.5 raises, default weight check
- `TestCombineJudgeReward`: anchor at weight=0, blend at weight=0.5 boundary
- `TestBuildTrajectoryGroups`: structure, security-fail dropped, count check
- `TestComputeRolloutAdvantages`: mixed->nonzero, constant->dropped, sum~0, cookbook symbols

## Verification

```
pytest tests/test_rl_rollouts.py -q                          → 23 passed
pytest tests/test_rl_train.py -k "dual_mode or judge_ge_gen or grpo_advantages" -q → 3 passed
grep -c 'compute_group_rewards' scripts/rl_rollouts.py       → 2 (>= 1 required)
grep -c 'compute_advantages\|remove_constant_reward_groups\|assemble_training_data' → 12 (>= 3 required)
git diff --stat scripts/reward_pipeline.py                   → empty (UNMODIFIED)
python -c "import scripts.rl_rollouts"                       → OK (no tinker required)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Inline fallback for tinker-absent environment**

- **Found during:** Task 2 RED phase (advisor flag confirmed by `python -c "import tinker_cookbook.rl.data_processing"` → ModuleNotFoundError)
- **Issue:** `test_grpo_advantages` in frozen `tests/test_rl_train.py` calls `compute_rollout_advantages` on plain dicts without monkeypatching the cookbook — if the module did a bare import of tinker, the test would error on collection
- **Fix:** Added inline fallback functions (`_inline_remove_constant_reward_groups`, `_inline_compute_advantages`, `_inline_assemble_training_data`) that mirror cookbook semantics exactly. The lazy import tries cookbook first; on `ImportError` the inline path runs. The `grep -c` acceptance criterion still passes (cookbook symbols appear in import statement + comments)
- **Files modified:** `scripts/rl_rollouts.py` (no other files)

**2. [Rule 2 - Missing critical functionality] security_fail flag over scalar==0.0**

- **Found during:** Task 1 implementation (advisor flag)
- **Issue:** A normalized composite scalar can legitimately be exactly `0.0` when a sample equals its group mean — using `scalar == 0.0` as the security drop trigger would produce false drops
- **Fix:** `build_trajectory_groups` reads `reward.breakdown.security_fail` (bool) as the drop signal, not `scalar == 0.0`. This matches the threat model's intent (T-09-SECDROP) and the reward_pipeline design where `security_fail` is tracked separately in `RewardBreakdown`

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The module imports `scripts.reward_pipeline` and `scripts.rl_judge_dispatch` (both existing, no new surfaces).

## Known Stubs

None. All exported functions have complete implementations.

## Self-Check: PASSED

- [x] `scripts/rl_rollouts.py` exists: FOUND
- [x] `tests/test_rl_rollouts.py` exists: FOUND
- [x] Task commit 843784c (RED): FOUND
- [x] Task commit 6b40e37 (GREEN): FOUND
- [x] `pytest tests/test_rl_rollouts.py -q` → 23 passed
- [x] `pytest tests/test_rl_train.py -k "dual_mode or judge_ge_gen or grpo_advantages" -q` → 3 passed
- [x] `git diff --stat scripts/reward_pipeline.py` → empty
