---
phase: 06-adaptive-training-planner
plan: 03
subsystem: training
tags: [adaptive-planner, telemetry, skill, anchor-store, warmup-probe, dgx-toolbox]

requires:
  - phase: 06-adaptive-training-planner/06-01
    provides: scripts/adaptive_planner.py with classify_power_zone, apply_ladder, parse_telemetry_jsonl, etc.
  - phase: 06-adaptive-training-planner/06-02
    provides: telemetry package integration (MemoryWatchdogCallback, GPUSampler, AnchorStore APIs)

provides:
  - .claude/skills/wp-finetune:adaptive-planner/SKILL.md — thin skill wrapper calling scripts/adaptive_planner.py
  - Updated run-training Step 8.5 delegating to adaptive-planner skill
  - Step 6 warmup probe protocol in run-training
  - dgx_toolbox.yaml with telemetry mount and container PYTHONPATH

affects:
  - run-training skill (Step 8.5 replaced, Step 6 extended)
  - Any future skill that calls the adaptive-planner skill
  - dgx_toolbox.py (reads container_env for PYTHONPATH injection)

tech-stack:
  added: []
  patterns:
    - "Skill as thin orchestration wrapper: all algorithm logic in tested Python module, skill is pure invocation"
    - "Warmup probe protocol: flag file _warmup_probe_required gates batch increases across runs"
    - "AnchorStore integration: apply_override/lookup/compute_config_hash for config history tracking"

key-files:
  created:
    - .claude/skills/wp-finetune:adaptive-planner/SKILL.md
  modified:
    - .claude/skills/wp-finetune:run-training/SKILL.md
    - config/dgx_toolbox.yaml

key-decisions:
  - "adaptive-planner skill is a thin wrapper: all decision logic stays in scripts/adaptive_planner.py (HIGH review concern)"
  - "Canonical JSONL schema updated to GPUSampler fields (watts, temperature_c, gpu_util_pct, mem_available_gb) — old field names deprecated"
  - "Thermal thresholds updated to match adaptive_planning.yaml: WARNING >82C, CRITICAL >=85C (was 80/83)"
  - "Warmup probe flag file (_warmup_probe_required) is the contract between Step 8.5 and Step 6 across run boundaries"

patterns-established:
  - "Skill-to-Python delegation: skill contains only orchestration steps, Python module contains all testable logic"
  - "Probe protocol: prepare_probe creates flag, evaluate_probe checks results, flag cleared regardless of outcome"
  - "AnchorStore usage: compute_config_hash -> lookup -> apply_override (no record_run, no has_anchor_for)"

requirements-completed: [PROB-01, PROB-02, PROB-03]

duration: 8min
completed: 2026-04-01
---

# Phase 06 Plan 03: Skill Integration Summary

**adaptive-planner Claude Code skill wired into run-training Step 8.5, delegating to scripts/adaptive_planner.py via prepare_probe/evaluate_probe/AnchorStore APIs from dgx-toolbox Phase 13**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-01T05:33:48Z
- **Completed:** 2026-04-01T05:42:04Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `.claude/skills/wp-finetune:adaptive-planner/SKILL.md` as a thin orchestration wrapper that calls `scripts/adaptive_planner.py` functions — no algorithm logic in markdown
- Updated run-training Step 8.5 to invoke the adaptive-planner skill instead of containing inline algorithm code (addresses HIGH review concern from cross-AI review)
- Added Step 6 warmup probe protocol using `telemetry.probe.prepare_probe`/`evaluate_probe` (PROB-01, PROB-03)
- Added `dgx_telemetry` mount and `container_env.PYTHONPATH` to `config/dgx_toolbox.yaml` for container access to telemetry package

## Task Commits

1. **Task 1: Create adaptive-planner skill and update dgx_toolbox.yaml** - `f04f980` (feat)
2. **Task 2: Replace run-training Step 8.5 with adaptive-planner invocation** - `2b6546f` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `.claude/skills/wp-finetune:adaptive-planner/SKILL.md` — New skill: thin wrapper over scripts/adaptive_planner.py with 7 orchestration steps (guard, parse, Unsloth actuals, failure classification, power zone + ladder, probe protocol, config write + anchor record)
- `.claude/skills/wp-finetune:run-training/SKILL.md` — Step 8.5 replaced with skill invocation + deprecated reference; Step 6 extended with warmup probe; thermal thresholds updated; JSONL schema updated to GPUSampler fields; Step 0c power draw note added
- `config/dgx_toolbox.yaml` — Added `extra_mounts.dgx_telemetry` and `container_env.PYTHONPATH` sections

## Decisions Made

- Step 8.5 inline algorithm preserved in run-training as "deprecated reference only" rather than deleted — provides context for why the delegation exists without causing algorithm duplication
- Thermal thresholds in skill match `config/adaptive_planning.yaml`: WARNING >82C (was >80C), CRITICAL >=85C (was >=83C) — raised per empirical GB10/OEM cooler data from Phase 06-01
- Canonical JSONL schema updated throughout run-training to use GPUSampler field names (`watts`, `temperature_c`, `gpu_util_pct`, `mem_available_gb`) — old names (`gpu_util`, `temp`, `vram_used_mb`) are deprecated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Troubleshooting section mentioned forbidden API names**
- **Found during:** Task 1 (SKILL.md creation, verification check)
- **Issue:** The troubleshooting section used `run_probe` and `record_run` as negative examples in error messages. The verification script (from plan's `<verify>` block) does a plain string match — it failed because the strings appeared in the file even in a "don't use this" context.
- **Fix:** Rewrote troubleshooting entries to avoid using the forbidden API names literally. Replaced "run_probe" and "record_run" error titles with generic error descriptions.
- **Files modified:** `.claude/skills/wp-finetune:adaptive-planner/SKILL.md`
- **Verification:** Verification script passed after rewording.
- **Committed in:** f04f980 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in troubleshooting section wording)
**Impact on plan:** Minor wording change only. No functional change to the skill logic.

## Issues Encountered

None beyond the auto-fixed deviation above.

## Known Stubs

None. Both skill files contain fully wired logic paths. The adaptive-planner skill correctly delegates to `scripts/adaptive_planner.py` functions. No placeholder data or hardcoded empty values.

## Next Phase Readiness

- Plan 04 (the final plan in Phase 06) can now verify the full integration: adaptive-planner skill -> Python module -> telemetry APIs
- The skill system is complete: run-training delegates to adaptive-planner, which delegates to scripts/adaptive_planner.py
- dgx_toolbox.yaml is ready for container execution with telemetry package available

## Self-Check: PASSED

- FOUND: `.claude/skills/wp-finetune:adaptive-planner/SKILL.md`
- FOUND: `.claude/skills/wp-finetune:run-training/SKILL.md`
- FOUND: `config/dgx_toolbox.yaml`
- FOUND: `.planning/phases/06-adaptive-training-planner/06-03-SUMMARY.md`
- FOUND commit: `f04f980` (Task 1)
- FOUND commit: `2b6546f` (Task 2)

---
*Phase: 06-adaptive-training-planner*
*Completed: 2026-04-01*
