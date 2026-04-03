---
phase: quick
plan: 260403-vvg
type: execute
wave: 1
depends_on: []
files_modified:
  - config/dgx_toolbox.yaml
  - scripts/dgx_toolbox.py
  - config/train_config_30_70.yaml
  - config/train_config_40_60.yaml
  - CHANGELOG.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "dgx_toolbox.yaml required_imports does not contain unsloth"
    - "dgx_toolbox.py hardcoded fallback does not contain unsloth"
    - "dgx_toolbox.py CONFIG_PATH resolves relative to __file__, not cwd"
    - "All 6 train_config*.yaml files contain dataloader_persistent_workers and dataloader_prefetch_factor"
  artifacts:
    - path: "config/dgx_toolbox.yaml"
      contains: "required_imports"
    - path: "scripts/dgx_toolbox.py"
      contains: "PROJECT_ROOT = Path(__file__).resolve().parent.parent"
  key_links: []
---

<objective>
Fix four stale references and config inconsistencies from recent functionality shifts: remove unsloth from dgx_toolbox config and script, fix CONFIG_PATH resolution in dgx_toolbox.py, and add missing dataloader fields to two train configs.

Purpose: Prevent container runtime failures from stale unsloth references and ensure consistent training config across all ratio variants.
Output: Clean config/scripts with no unsloth refs, consistent dataloader settings across all configs.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@config/dgx_toolbox.yaml
@scripts/dgx_toolbox.py
@config/train_config_30_70.yaml
@config/train_config_40_60.yaml
@CHANGELOG.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Remove stale unsloth references and fix CONFIG_PATH resolution</name>
  <files>config/dgx_toolbox.yaml, scripts/dgx_toolbox.py</files>
  <action>
1. In `config/dgx_toolbox.yaml` line 118: remove `- unsloth` from the `required_imports` list. Keep trl, peft, datasets, mlflow, yaml, scipy.

2. In `scripts/dgx_toolbox.py` line 329: remove `"unsloth"` from the hardcoded fallback list in `_check_deps`. Change:
   ```python
   imports = self._config.get("required_imports", ["unsloth", "trl", "peft", "datasets", "mlflow", "yaml", "scipy", "dotenv"])
   ```
   to:
   ```python
   imports = self._config.get("required_imports", ["trl", "peft", "datasets", "mlflow", "yaml", "scipy", "dotenv"])
   ```

3. In `scripts/dgx_toolbox.py` line 43: replace `CONFIG_PATH = Path.cwd() / "config" / "dgx_toolbox.yaml"` with:
   ```python
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   CONFIG_PATH = PROJECT_ROOT / "config" / "dgx_toolbox.yaml"
   ```
   This matches the pattern already applied in train_model.py (commit 5276e4b).
  </action>
  <verify>
    <automated>grep -c "unsloth" config/dgx_toolbox.yaml scripts/dgx_toolbox.py | grep -v ":0$" && echo "FAIL: unsloth still present" && exit 1 || echo "PASS: no unsloth refs"; grep -q "Path(__file__).resolve().parent.parent" scripts/dgx_toolbox.py && echo "PASS: CONFIG_PATH uses __file__" || (echo "FAIL" && exit 1)</automated>
  </verify>
  <done>No unsloth references remain in dgx_toolbox.yaml or dgx_toolbox.py. CONFIG_PATH resolves relative to script location.</done>
</task>

<task type="auto">
  <name>Task 2: Add missing dataloader fields to train configs and update CHANGELOG</name>
  <files>config/train_config_30_70.yaml, config/train_config_40_60.yaml, CHANGELOG.md</files>
  <action>
1. In `config/train_config_30_70.yaml`: add two lines after `dataloader_num_workers: 4` (line 42), maintaining alphabetical order within the training block:
   ```yaml
     dataloader_persistent_workers: true
     dataloader_prefetch_factor: 2
   ```
   Use value 2 matching train_model.py defaults (other configs vary: 50_50 and 70_30 use 3, 60_40 uses 4 -- those are intentional per-ratio tuning).

2. In `config/train_config_40_60.yaml`: add one line after `dataloader_persistent_workers: true` (line 43):
   ```yaml
     dataloader_prefetch_factor: 2
   ```

3. In `CHANGELOG.md`: add entries under `## [Unreleased]` in the existing `### Fixed` section:
   ```
   - **`config/dgx_toolbox.yaml` / `scripts/dgx_toolbox.py`** -- Removed stale `unsloth` from `required_imports` list and hardcoded fallback. Eval-toolbox container does not have Unsloth installed
   - **`scripts/dgx_toolbox.py`** -- `CONFIG_PATH` now resolved via `Path(__file__)` instead of `Path.cwd()`, matching the fix already applied to `train_model.py`
   - **`config/train_config_30_70.yaml` / `config/train_config_40_60.yaml`** -- Added missing `dataloader_persistent_workers` and `dataloader_prefetch_factor` fields for consistency with other ratio configs
   ```
  </action>
  <verify>
    <automated>grep -c "dataloader_prefetch_factor" config/train_config_30_70.yaml config/train_config_40_60.yaml | grep ":0$" && echo "FAIL: missing fields" && exit 1 || echo "PASS: fields present"; grep -c "dataloader_persistent_workers" config/train_config_30_70.yaml | grep -q ":1" && echo "PASS" || (echo "FAIL" && exit 1)</automated>
  </verify>
  <done>All 6 train config files have both dataloader_persistent_workers and dataloader_prefetch_factor. CHANGELOG updated with all fixes.</done>
</task>

</tasks>

<verification>
Run existing tests to confirm nothing is broken:
```bash
cd /home/robert_li/Desktop/projects/wp-finetune && python -m pytest tests/test_config.py tests/test_train_model.py -x -q
```
</verification>

<success_criteria>
- Zero references to "unsloth" in dgx_toolbox.yaml and dgx_toolbox.py
- dgx_toolbox.py CONFIG_PATH uses Path(__file__).resolve().parent.parent
- All ratio configs (30_70, 40_60, 50_50, 60_40, 70_30, base) contain both dataloader fields
- Existing tests pass
- CHANGELOG.md updated
</success_criteria>

<output>
After completion, create `.planning/quick/260403-vvg-fix-stale-unsloth-refs-and-config-incons/260403-vvg-SUMMARY.md`
</output>
