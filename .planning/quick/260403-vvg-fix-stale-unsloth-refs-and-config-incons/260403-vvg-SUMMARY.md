---
phase: quick
plan: 260403-vvg
subsystem: config, scripts
tags: [unsloth-removal, config-consistency, dgx-toolbox, dataloader]
dependency_graph:
  requires: []
  provides: [clean-dgx-toolbox-config, consistent-train-configs]
  affects: [scripts/dgx_toolbox.py, config/dgx_toolbox.yaml, config/train_config_30_70.yaml, config/train_config_40_60.yaml]
tech_stack:
  added: []
  patterns: [Path(__file__).resolve() for script-relative config resolution]
key_files:
  created: []
  modified:
    - config/dgx_toolbox.yaml
    - scripts/dgx_toolbox.py
    - config/train_config_30_70.yaml
    - config/train_config_40_60.yaml
    - CHANGELOG.md
decisions:
  - Unsloth removed from required_imports (not installed in eval-toolbox container)
  - CONFIG_PATH now script-relative (matches pattern in train_model.py commit 5276e4b)
  - dataloader_prefetch_factor set to 2 for 30_70 and 40_60 (train_model.py default)
metrics:
  duration_min: 5
  completed_date: "2026-04-03"
  tasks_completed: 2
  files_modified: 5
---

# Phase quick Plan 260403-vvg: Fix Stale Unsloth Refs and Config Inconsistencies Summary

**One-liner:** Removed unsloth from dgx_toolbox required_imports and fallback, fixed CONFIG_PATH to use `Path(__file__)`, and added missing dataloader fields to 30_70/40_60 train configs for full 6-config consistency.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Remove stale unsloth references and fix CONFIG_PATH resolution | 46fbc88 | config/dgx_toolbox.yaml, scripts/dgx_toolbox.py |
| 2 | Add missing dataloader fields to train configs and update CHANGELOG | f340b22 | config/train_config_30_70.yaml, config/train_config_40_60.yaml, CHANGELOG.md |

## What Was Done

### Task 1: Unsloth removal + CONFIG_PATH fix

- **config/dgx_toolbox.yaml**: Removed `- unsloth` from `required_imports` list. The eval-toolbox container does not have Unsloth installed; leaving it caused `_check_deps` to fail at runtime.
- **scripts/dgx_toolbox.py** (line 329): Removed `"unsloth"` from the hardcoded fallback list in `_check_deps`. Fallback is now `["trl", "peft", "datasets", "mlflow", "yaml", "scipy", "dotenv"]`.
- **scripts/dgx_toolbox.py** (line 43): Replaced `CONFIG_PATH = Path.cwd() / "config" / "dgx_toolbox.yaml"` with a `PROJECT_ROOT = Path(__file__).resolve().parent.parent` pattern, matching the fix already applied to `train_model.py` (commit 5276e4b). Prevents config load failure when the container workdir differs from the project root.

### Task 2: Dataloader field consistency

- **config/train_config_30_70.yaml**: Added `dataloader_persistent_workers: true` and `dataloader_prefetch_factor: 2` after `dataloader_num_workers: 4`.
- **config/train_config_40_60.yaml**: Added `dataloader_prefetch_factor: 2` (already had `dataloader_persistent_workers`).
- All 6 ratio configs (base, 30_70, 40_60, 50_50, 60_40, 70_30) now contain both dataloader fields. Values vary per-ratio (2, 2, 3, 4, 3, 3) — intentional per-ratio tuning.
- **CHANGELOG.md**: Added three entries under `## [Unreleased] ### Fixed` documenting all changes.

## Verification

- `tests/test_config.py` + `tests/test_train_model.py`: **15/15 passed**
- Zero unsloth references in `required_imports` section of dgx_toolbox.yaml
- Zero unsloth references in dgx_toolbox.py
- `Path(__file__).resolve().parent.parent` confirmed in dgx_toolbox.py
- All 6 train_config*.yaml files confirmed to have both `dataloader_persistent_workers` and `dataloader_prefetch_factor`

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- [x] config/dgx_toolbox.yaml modified and committed
- [x] scripts/dgx_toolbox.py modified and committed
- [x] config/train_config_30_70.yaml modified and committed
- [x] config/train_config_40_60.yaml modified and committed
- [x] CHANGELOG.md modified and committed
- [x] Commits 46fbc88 and f340b22 exist
- [x] All 15 tests pass
