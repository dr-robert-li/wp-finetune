---
phase: quick
plan: 260328-h1d
subsystem: dgx-toolbox
tags: [refactor, config-extraction, project-agnostic]
dependency_graph:
  requires: []
  provides: [project-agnostic-dgx-engine]
  affects: [run-training, observe-training, review-telemetry]
tech_stack:
  added: []
  patterns: [config-driven-engine, backward-compatible-defaults]
key_files:
  created: []
  modified:
    - config/dgx_toolbox.yaml
    - scripts/dgx_toolbox.py
decisions:
  - "All 8 couplings use .get() with backward-compatible defaults -- old YAML files still work"
  - "project_root defaults to cwd (not __file__.parent.parent) so any project can use the engine"
  - "CONFIG_PATH computed from cwd instead of __file__ to support external project usage"
metrics:
  duration_minutes: 4
  completed: "2026-03-28T02:22:43Z"
  tasks_completed: 2
  tasks_total: 2
---

# Quick Task 260328-h1d: Make dgx_toolbox.py Project-Agnostic Summary

Extracted all 8 hardcoded project couplings (CONTAINER_MAP, PROJECT_ROOT, validation paths, import lists, artifact checks, extra deps) from dgx_toolbox.py into dgx_toolbox.yaml, converting the Python module into a pure config-driven execution engine.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Expand dgx_toolbox.yaml with all 8 project couplings | 0c624c8 | config/dgx_toolbox.yaml |
| 2 | Refactor dgx_toolbox.py to read all couplings from config | efe9135 | scripts/dgx_toolbox.py |

## What Changed

### config/dgx_toolbox.yaml
Added 6 new top-level sections (69 lines):
- `containers` -- 3 container definitions (unsloth_studio, eval_toolbox, vllm)
- `validation_paths` -- training_data and config relative paths
- `mount_check_file` -- file checked inside container to verify mount
- `required_imports` -- 7 Python packages to verify in container
- `status_artifacts` -- 6 pipeline artifact definitions with path/type/pattern
- `extra_deps` -- 6 pip packages beyond pinned_versions

### scripts/dgx_toolbox.py
Removed 65 lines, added 47 lines (net -18 lines):
- Removed `PROJECT_ROOT` global constant
- Removed `CONTAINER_MAP` global dict (20 lines)
- Added `self._project_root` (resolved from config, defaults to cwd)
- Added `self._containers` (loaded from config)
- All 8 methods updated to read from `self._config` with backward-compatible defaults
- `status_report` artifacts now use config-driven loop instead of hardcoded paths
- CLI block updated to use instance `dgx._containers` instead of global

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed hardcoded mount path in run_service**
- **Found during:** Task 2
- **Issue:** `run_service` had hardcoded `/workspace/wp-finetune` in EXTRA_MOUNTS env var
- **Fix:** Changed to read workdir from container config: `self._containers.get(component, {}).get("workdir", "/workspace")`
- **Files modified:** scripts/dgx_toolbox.py
- **Commit:** efe9135

**2. [Rule 2 - Completeness] Updated docstring to be project-agnostic**
- **Found during:** Task 2
- **Issue:** Docstring for validate() listed hardcoded paths like `data/final_dataset/openai_train.jsonl`
- **Fix:** Changed to generic descriptions ("path from config")
- **Files modified:** scripts/dgx_toolbox.py
- **Commit:** efe9135

## Verification Results

1. `get_toolbox()` returns valid info dict with containers from config
2. No `PROJECT_ROOT` or `CONTAINER_MAP` globals exist in module
3. Zero matches for `Qwen3`, `qwen3-wp`, `unsloth,trl,peft` in Python file
4. API surface unchanged: validate, ensure_ready, execute, run_service, status_report, info all present
5. Backward-compatible defaults verified: all `.get()` calls have fallback values matching original hardcoded values

## Self-Check: PASSED
