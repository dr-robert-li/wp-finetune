---
phase: 06-adaptive-training-planner
plan: 01
subsystem: training
tags: [adaptive-planner, thermal-routing, pytorch, qwen3, unsloth, telemetry]

# Dependency graph
requires:
  - phase: 03-model-prep-and-training
    provides: train_config.yaml hyperparameters and Qwen3-30B-A3B training setup
provides:
  - scripts/adaptive_planner.py — testable Python module with routing, coupling, and ladder logic
  - config/adaptive_planning.yaml — centralised v4.0 threshold configuration
  - tests/test_adaptive_planner.py — 28 decision-table tests covering all core functions
affects:
  - 06-02 (telemetry integration)
  - 06-03 (adaptive-planner skill — thin wrapper around this module)
  - 06-04 (run-training integration)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All algorithmic thresholds centralised in config/adaptive_planning.yaml (no hardcoded values in scripts)"
    - "TDD: RED tests committed before GREEN implementation"
    - "Telemetry stubs in tests so hardware is not required"
    - "round() for batch/grad_accum coupling to avoid float truncation (0.60/0.40=1.4999)"

key-files:
  created:
    - scripts/adaptive_planner.py
    - config/adaptive_planning.yaml
    - tests/test_adaptive_planner.py
  modified: []

key-decisions:
  - "thermal_brake.warning_temp=82 (raised from 80C per empirical GB10/OEM cooler data)"
  - "thermal_brake.emergency_temp=85 (raised from 83C per empirical data)"
  - "couple_batch_grad_accum uses round() not // to avoid float precision truncation"
  - "classify_power_zone returns MODERATE only when BOTH has_batch_headroom AND has_mem_headroom are True"
  - "IO bottleneck guard: gpu_util<30 AND watts<30 skips Rung 1 (batch) to avoid thrashing during dataloader stalls"
  - "compute_batch_ceiling calls telemetry.effective_scale.compute (NOT compute_effective_scale — does not exist)"

patterns-established:
  - "Adaptive routing: thermal brake (peak_temp>=82) overrides all watt/util signals"
  - "Ladder rungs in v4.0 order: batch(1) > prefetch(2) > workers(3) > save_steps(4) > eval_steps(5)"
  - "pin_memory always False on UMA (DGX Spark unified memory)"
  - "persistent_workers: sticky — never reverts once True"

requirements-completed: [ADPT-01, ADPT-02, ADPT-03, BTCH-01]

# Metrics
duration: 18min
completed: 2026-04-01
---

# Phase 6 Plan 1: Adaptive Training Planner Core Module Summary

**Testable Python module with thermal power-zone routing (82/85C thresholds), round()-based batch coupling, 5-rung thermal exploitation ladder, and centralised v4.0 YAML config — addresses HIGH review concern about untestable skill logic**

## Performance

- **Duration:** 18 min
- **Started:** 2026-04-01T05:00:00Z
- **Completed:** 2026-04-01T05:18:35Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `scripts/adaptive_planner.py` with 5 core functions as testable pure Python (no GPU required)
- Created `config/adaptive_planning.yaml` with all v4.0 thresholds, 82/85C empirical GB10 temps
- 28 decision-table unit tests covering all routing zones, coupling edge cases, ladder rung order, and telemetry parsing
- Used TDD (RED commit before GREEN): tests written first, confirmed failing, then implementation

## Task Commits

1. **Task 1 (TDD RED): Failing tests for adaptive_planner** - `c631ce4` (test)
2. **Task 1 (TDD GREEN): Implement adaptive_planner.py** - `1ae4bbb` (feat)
3. **Task 2: Create config/adaptive_planning.yaml** - `f8fe410` (feat)

## Files Created/Modified
- `scripts/adaptive_planner.py` — classify_power_zone, couple_batch_grad_accum, compute_batch_ceiling, apply_ladder, parse_telemetry_jsonl
- `config/adaptive_planning.yaml` — centralised v4.0 thresholds (82/85C temps, all 9 sections)
- `tests/test_adaptive_planner.py` — 28 decision-table tests with telemetry stubs

## Decisions Made
- thermal_brake.warning_temp raised to 82C (from 80C): GB10 OEM coolers run 80-82C with no throttling — old 80C threshold was too conservative and would false-trigger THROTTLED
- thermal_brake.emergency_temp raised to 85C (from 83C): same empirical evidence, 83C was over-cautious
- `couple_batch_grad_accum` uses `round()` not `//`: float truncation bug documented (0.60/0.40=1.4999... → int gives 1 not 2)
- MODERATE requires BOTH headrooms (batch AND mem): single-headroom-only situations are still TARGET since one constraint would prevent safe climbing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- RTK token filter intercepted pytest output; used `rtk proxy python -m pytest` to bypass filtering and see full test output

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `scripts/adaptive_planner.py` is importable and tested — Plan 02 (telemetry integration) and Plan 03 (skill wrapper) can import and call these functions directly
- Config file at `config/adaptive_planning.yaml` provides canonical threshold source for all downstream consumers
- Plan 02 should wire `parse_telemetry_jsonl` output to `classify_power_zone` and `apply_ladder` calls

---
*Phase: 06-adaptive-training-planner*
*Completed: 2026-04-01*
