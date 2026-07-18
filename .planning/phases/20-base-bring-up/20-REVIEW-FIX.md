---
phase: 20-base-bring-up
fixed_at: 2026-07-13T04:02:51Z
review_path: .planning/phases/20-base-bring-up/20-REVIEW.md
iteration: 1
findings_in_scope: 11
fixed: 11
skipped: 0
status: all_fixed
---

# Phase 20: Code Review Fix Report

**Fixed at:** 2026-07-13T04:02:51Z
**Source review:** .planning/phases/20-base-bring-up/20-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 11 (1 critical, 10 warning; Info findings out of scope per instructions)
- Fixed: 11
- Skipped: 0

## Fixed Issues

### CR-01: `train_config_v4.yaml` LoRA `target_modules` do not match the real Qwen3.6-35B-A3B module tree

**Files modified:** `config/train_config_v4.yaml`
**Commit:** 1703cc3
**Applied fix:** Verified module names live against `models/Qwen3.6-35B-A3B/model.safetensors.index.json`
(no weights loaded — index-only). `gate_up_proj` matched zero real `nn.Module`s (fused
`nn.Parameter` on the routed experts) and `down_proj` only reached `shared_expert.down_proj`.
Corrected `target_modules` to the real `shared_expert.{gate_proj,up_proj,down_proj}`
`nn.Linear` submodules, and added `target_parameters: [mlp.experts.gate_up_proj,
mlp.experts.down_proj]` for the fused routed-expert tensors — this mirrors the existing,
already-working precedent in `config/train_config_reasoning.yaml` for the same A3B-family
fused-expert architecture, and `scripts/train_model.py::apply_lora` already plumbs
`lora_cfg.get("target_parameters")` into `FastLanguageModel.get_peft_model`, so no code
change was needed there — only the config was wrong.
**Note:** This is a **requires human verification** item in the sense that the actual LoRA
attach count can only be confirmed by loading the model in `train_model.py --dry-run`
(`assert_router_frozen_and_report`), which needs GPU/CPU hardware this fixer does not have
runtime access to. The fix itself was verified purely against static artifacts
(`model.safetensors.index.json`, no weights loaded), consistent with the review's own
verification method.

### WR-01: `download_model.py` idempotency check treats any partial download as complete

**Files modified:** `scripts/download_model.py`, `tests/test_download_model_v4.py`
**Commit:** 6947148
**Applied fix:** Added `expected_shard_count()` reading `model.safetensors.index.json`'s
`weight_map` when present; `download_model()` now only skips when
`existing_shards >= expected_shards` (or falls back to the old any-shard heuristic when no
index.json exists yet, matching the pre-existing test's fixture which has no index.json).
Added two new tests: incomplete-download-with-index resumes, complete-download-with-index
skips. All 9 tests in `tests/test_download_model_v4.py` pass.

### WR-02: `merge_adapter.py` narrows `target_modules` to bare leaf names, not scoped per-layer paths

**Files modified:** `scripts/merge_adapter.py`
**Commit:** 558d783
**Applied fix:** `kept_leaf_names` (bare leaf names, e.g. `"q_proj"`) replaced with
`kept_module_paths` (full dotted per-layer paths, e.g.
`"model.language_model.layers.11.self_attn.q_proj"`), computed from the same `remapped` dict
keys already available. This routes PEFT's `check_target_module_exists` through its exact
`key in config.target_modules` branch instead of the `key.endswith(f".{target_key}")`
suffix-match branch, removing the collision risk the review flagged.

### WR-03: Module-count guard only checks for missing modules, not unexpected/extra ones

**Files modified:** `scripts/merge_adapter.py`
**Commit:** fa60ea2
**Applied fix:** `_guard_merge_completeness` now includes `unexpected_modules` in the abort
condition (`merged_count != expected_count or merged_count <= 0 or unexpected_modules`), not
just a printed warning. Also added `unexpected_module_count`/`unexpected_modules_sample` to
the written guard receipt for post-mortem visibility.

### WR-04: Merge idempotency short-circuit is effectively dead for the v4 base

**Files modified:** `scripts/merge_adapter.py`
**Commit:** 660130b
**Applied fix:** The idempotency probe now calls `_select_serving_tokenizer` (the same
function `_verify_merged_model` uses post-merge) to determine `check_special_tokens` before
deciding whether the special-token assertion applies. When the base has no extended-vocab
tokenizer (the v4 base's current state), the probe now correctly short-circuits instead of
always falling through to a full re-merge. Cost: one cheap tokenizer-files-only load of the
base tokenizer, no model weights.

### WR-05: `merge_adapter.py` has no top-level failure handling / receipt

**Files modified:** `scripts/merge_adapter.py`
**Commit:** d15c2e0
**Applied fix:** Wrapped `main(args)` in the `__main__` block with `try/except Exception`,
writing `output/base20/_merge_adapter_result.json` (`status: fail`, `error`) before
`sys.exit(1)`. `SystemExit` (the script's own diagnosed abort paths in
`_guard_merge_completeness` / `_verify_merged_model`, both of which already print their own
diagnostics/receipts) is re-raised unchanged, matching the sibling gate scripts' pattern
without duplicating their already-correct receipts.

### WR-06: Non-atomic config.json rewrites risk leaving a corrupted config on crash

**Files modified:** `scripts/check_token_alignment.py`, `scripts/merge_adapter.py`
**Commit:** 3ec4b80
**Applied fix:** Both the BASE-02 token-alignment JSON surgery and the BASE-04 VL-config
repair now write to a `.json.tmp` sibling file and `Path.replace()` (atomic, `os.replace`
semantics) it into place, instead of writing directly to the target `config.json`.

### WR-07: `boot_vllm()` discards the launch subprocess's stdout/stderr

**Files modified:** `scripts/_p0_vllm_smoke_serve.py`
**Commit:** 91c818b
**Applied fix:** Replaced `subprocess.run(..., check=True, stdout=DEVNULL, stderr=STDOUT)`
with `capture_output=True, text=True` plus a manual `returncode != 0` check that raises a
`RuntimeError` including the captured stdout/stderr, so a launch-time failure (bad
`MODEL_DIR`, missing `docker`, etc.) now surfaces its diagnostics.

### WR-08: Container-liveness check uses substring containment, not exact match

**Files modified:** `scripts/_p0_vllm_smoke_serve.py`
**Commit:** 91c818b (same commit as WR-07 — same file, adjacent lines)
**Applied fix:** `if name not in alive:` → `if name not in alive.splitlines():`.

### WR-09: `recipes/qwen3.6-35b-a3b-vllm.yaml` documents settings that don't match the script actually used

**Files modified:** `recipes/qwen3.6-35b-a3b-vllm.yaml`
**Commit:** 364c8bc
**Applied fix:** Updated the recipe's `max_model_len`/`max_num_batched_tokens` from `16384`
to `8192` to match `serve_base20_vllm.sh`'s actual default (the review's first suggested
option — changing the doc to match the passing gate runs, rather than changing the script's
default and risking an unvalidated memory/timing profile).

### WR-10: Tar extraction in the Tinker probe adapter download doesn't validate member type

**Files modified:** `scripts/build_base20_probe_adapter.py`
**Commit:** 0161ebd
**Applied fix:** Added `and m.isfile()` to the extraction condition, so a tar entry named
`adapter_config.json`/`adapter_model.safetensors` that is actually a symlink/hardlink/device
is no longer extracted as such.

## Skipped Issues

None — all 11 in-scope findings (CR-01, WR-01 through WR-10) were fixed. Info-level findings
(IN-01 through IN-05) are out of scope per the fix instructions and were left untouched.

## Verification

- `python3 -m pytest tests/test_download_model_v4.py tests/test_check_token_alignment.py -x -q`
  → **20 passed** (9 + 11, including 2 new WR-01 tests).
- Every modified `.py` file passed `python3 -c "import ast; ast.parse(...)"`.
- Every modified `.yaml` file passed a `yaml.safe_load()` round-trip.
- `config/train_config_v4.yaml`'s new `target_modules`/`target_parameters` were verified
  against `models/Qwen3.6-35B-A3B/model.safetensors.index.json` (module-name presence only —
  no model weights were loaded, per the fix-agent's hardware constraint).

---

_Fixed: 2026-07-13T04:02:51Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
