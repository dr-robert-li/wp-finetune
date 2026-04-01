---
phase: 06-adaptive-training-planner
plan: 05
subsystem: adaptive-planner
tags: [tdd, gap-closure, downscale, thermal, batch-sizing]
dependency_graph:
  requires: []
  provides: [apply_ladder_downscale_path]
  affects: [scripts/adaptive_planner.py, config/adaptive_planning.yaml, tests/test_adaptive_planner.py]
tech_stack:
  added: []
  patterns: [TDD red-green, batch-grad_accum coupling, downscale-zones guard]
key_files:
  created: []
  modified:
    - scripts/adaptive_planner.py
    - config/adaptive_planning.yaml
    - tests/test_adaptive_planner.py
decisions:
  - "Downscale to floor directly (not step-by-step): CAPPED/THROTTLED are safety actions, not gradual tuning"
  - "downscale_floor=1 in config/adaptive_planning.yaml under ladder section — configurable, not hardcoded"
  - "rung_1_batch_downscale distinct from rung_1_batch — different semantics, different audit trail"
  - "TARGET zone returns empty delta — no downscale or upscale (happy path, already optimal)"
metrics:
  duration_min: 8
  completed_date: "2026-04-01"
  tasks_completed: 2
  files_changed: 3
---

# Phase 6 Plan 5: Batch Downscale for CAPPED/THROTTLED Zones Summary

**One-liner:** JWT-free thermal safety gate: apply_ladder() now reduces batch to configurable floor=1 with coupled grad_accum for CAPPED/THROTTLED power zones.

## What Was Built

Added a batch downscale path to `apply_ladder()` in `scripts/adaptive_planner.py` that triggers when `power_zone` is `"CAPPED"` or `"THROTTLED"`. This closes the BLOCKER identified in the gap analysis: previously, high-power/high-temp zones returned an empty delta — no action to shed thermal load.

The downscale path:
- Reads `downscale_floor` from `thresholds["ladder"]` (default 1 via `config/adaptive_planning.yaml`)
- If `current_batch > downscale_floor`, sets `per_device_train_batch_size = downscale_floor`
- Computes `gradient_accumulation_steps` via `couple_batch_grad_accum()` to preserve `effective_batch`
- Appends `"rung_1_batch_downscale"` to `rungs_applied`
- Returns immediately (no upscale rungs applied in downscale zones)

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add downscale_floor config and failing tests (RED) | 86cfc97 | config/adaptive_planning.yaml, tests/test_adaptive_planner.py |
| 2 | Implement batch downscale path in apply_ladder() | 203085e | scripts/adaptive_planner.py |

## Verification Results

```
pytest tests/test_adaptive_planner.py -v
34 passed (28 existing + 6 new downscale tests)
```

All success criteria confirmed:
- `apply_ladder("CAPPED", batch=4, grad_accum=4)` returns batch=1, grad_accum=16
- `apply_ladder("THROTTLED", batch=4, grad_accum=4)` returns batch=1, grad_accum=16
- `apply_ladder("CAPPED", batch=1, grad_accum=16)` returns empty rungs (already at floor)
- `apply_ladder("TARGET", ...)` returns empty rungs (no action)
- `apply_ladder("MODERATE", ...)` still returns upscale actions (no regression)

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- scripts/adaptive_planner.py: FOUND (contains downscale_zones, rung_1_batch_downscale)
- config/adaptive_planning.yaml: FOUND (contains downscale_floor: 1)
- tests/test_adaptive_planner.py: FOUND (contains TestApplyLadderDownscale with 6 tests)
- Commit 86cfc97: FOUND
- Commit 203085e: FOUND
- All 34 tests pass
