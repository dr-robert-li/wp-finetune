---
phase: 21-sft-training-generation-judge-models
fixed_at: 2026-07-14T12:30:00Z
review_path: .planning/phases/21-sft-training-generation-judge-models/21-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 9
skipped: 1
status: partial
---

# Phase 21: Code Review Fix Report

**Fixed at:** 2026-07-14T12:30:00Z
**Source review:** `.planning/phases/21-sft-training-generation-judge-models/21-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (3 CRITICAL + 6 WARNING; INFO included per explicit instruction)
- Fixed: 9
- Skipped: 1 (IN-01)

All fixes are forward-looking only. No historical Phase 21 receipts
(`moe_merge_probe.json`, `gen03_merge.json`, `judge03_rho.json`,
`judge03_capture_rho.json`, `gen03_wpbench.json`, etc.) were rewritten or
touched -- they remain the historical evidence of the runs that already
happened. The fixes protect *future* re-runs, especially Phase 23/27 reuse
of `merge_adapter.py` and the merge+serve scripts.

## Fixed Issues

### CR-01: `boot_vllm()` called outside the `try/finally` that tears down the container

**Files modified:** `scripts/build_base21_moe_probe_adapter.py`,
`scripts/build_gen03_merge.py`, `scripts/build_judge03_merge_serve.py`,
`scripts/verify_moe_merge_ground_truth.py`
**Commit:** `c036487`
**Applied fix:** Moved `boot_vllm(...)` inside the `try:` block at all 5
cited sites (matching `build_gen03_wpbench.py`'s already-correct pattern),
so a `boot_vllm()` failure after the container has already started still
routes through `stop_vllm()` in `finally`.

### CR-02: `base_vs_merged_differs` is vacuously True when the merged model returns empty output

**Files modified:** `scripts/build_base21_moe_probe_adapter.py`,
`scripts/build_gen03_merge.py`, `scripts/build_judge03_merge_serve.py`
**Commit:** `b440e22`
**Applied fix:** Added `if not merged_out: return False` before the
`merged_out != base_out` diff comparison in all 3 sites, so an empty
merged-model generation can no longer pass the "differs from base" gate
purely because `"" != base_out` is trivially true.

### CR-03: `merge_adapter.py`'s idempotency short-circuit can silently serve a stale merge for a new promoted checkpoint

**Files modified:** `scripts/merge_adapter.py`
**Commit:** `3b11939`
**Applied fix:** Added `_adapter_content_hash()` (sha256 of
`adapter_model.safetensors`) and a `.merged_from.json` marker written into
`output_dir` after every successful merge (both the `tinker_cookbook` MoE
path and the older PEFT path). The idempotency short-circuit now only
fires when the marker exists AND its hash matches the adapter being merged
THIS call -- any missing marker (pre-fix merged dir, or a killed-mid-merge
partial write), or a hash mismatch (stale merge from an earlier checkpoint
promotion reused at the same fixed `output_dir`), falls through to a full
re-merge instead of silently skipping. Self-checked with a synthetic
adapter directory (hash differs for different content, stable for
identical content) and the existing routing test suite still passes.

## Fixed Warnings

### WR-01: Bootstrap CI uses an unseeded RNG

**File modified:** `scripts/build_gen03_wpbench.py`
**Commit:** `903c344`
**Applied fix:** Seeded `np.random.default_rng(1337)` and recorded
`bootstrap_seed` in both the CI dict and the final `gen03_wpbench.json`
receipt. Verified: two calls against identical synthetic `results_json`
now produce byte-identical `ci_lower`/`ci_upper`.

### WR-02: Quality dimension held fixed inside the bootstrap

**File modified:** `scripts/build_gen03_wpbench.py`
**Commit:** `4df4483`
**Applied fix:** `quality` is now resampled per bootstrap iteration
(`rng.choice(quality, ...)`) the same way `knowledge`/`correctness` are,
falling back to the `None`-mean no-op when `quality.size == 0` (matches the
actual Phase 21 receipt shape, which had no per-test quality signal).
Verified with synthetic data both with and without a populated quality
array.

### WR-03: Generic-exception fail receipt ignores the caller-specific receipt path

**File modified:** `scripts/merge_adapter.py`
**Commit:** `200b380`
**Applied fix:** The `__main__` exception handler now derives the fail
receipt's directory from `args.guard_receipt_path` (resolved via
`resolve_path`) when the caller passed one, falling back to the original
hardcoded `output/base20/_merge_adapter_result.json` only when no explicit
path was given. Verified end-to-end: a forced config-load failure with
`--guard-receipt-path` set writes the fail receipt next to it, not to
`base20/`.

### WR-04: Generation/sampling errors silently folded into "parse fail" / rho measurements

**Files modified:** `scripts/smoke_judge_format_base21.py`,
`scripts/capture_judge_responses_tinker.py`
**Commit:** `c20bd12`
**Applied fix:** `judge_generate()` now returns `(outs, infra_error_idx)`;
`run_smoke()` adds `n_infra_error` to `judge01_format_smoke.json`.
`capture_judge_responses_tinker.py` tags each captured row with an
additive `infra_error` bool (the unmodified `eval_relabel.py` only reads
`index`/`response` per row and ignores unknown keys) and writes a
`<out>.capture_summary.json` sidecar carrying `n_infra_error`, written
last so the existing per-row streaming writes stay crash-resilient.
Verified both with mocked failure injection.

### WR-05: Judge seed promotion uses raw point-estimate rho, not CI-aware

**File modified:** `scripts/build_judge03_capture_rho.py`
**Commit:** `09b96f0`
**Applied fix:** Added `runner_up_seed` and `ci_overlaps_runner_up` to
`judge03_capture_rho.json`, computed from the top-2 seeds by point-estimate
rho. `best_single_seed` keeps its existing `seed`/`rho` keys (the only
fields `build_judge03_merge_serve.py` reads), so this is additive and
non-breaking. Verified with synthetic per-seed CI data (overlapping and
matching the expected result).

### WR-06: Expected-modules-manifest fallback defaults to Phase-20's attention-only file

**File modified:** `scripts/merge_adapter.py`
**Commit:** `3608b42`
**Applied fix:** `_merge_via_tinker_cookbook` (which only ever runs for
routed-MoE-expert adapters) now raises `SystemExit` immediately when
`--expected-modules-manifest` is not supplied, instead of silently falling
back to Phase 20's attention-only manifest. **Scoped to this function
only** -- the older PEFT `merge_adapter()` path's structurally identical
fallback (line ~607) was left unchanged, because that path never handles
routed-MoE adapters (`_adapter_has_routed_expert_params` always routes
those through `_merge_via_tinker_cookbook` first), so Phase 20's manifest
IS the architecturally correct default there. Every current Phase 21
caller already passes the manifest explicitly, so this is a no-op for
existing usage (verified: routing tests still pass).

## Skipped Issues

### IN-01: Duplicated `BASE_MODEL` string literal instead of importing the canonical constant

**File:** `scripts/build_judge03_capture_rho.py:39`
**Reason:** The suggested fix (`from tinker_reasoning_data_v4 import
BASE_MODEL as V4_BASE_MODEL`) was checked against the actual code and does
NOT apply cleanly. `tinker_reasoning_data_v4.py` does a top-level,
unconditional `from tinker_cookbook import renderers` import.
`build_judge03_capture_rho.py`'s own docstring states it runs under "the
project/conda env" (it shells out to `.venv-tinker` itself only for the
capture subprocess step) -- and `tinker_cookbook` is confirmed NOT
installed in that env (`ModuleNotFoundError` verified directly in this
repo's conda env). Applying the suggested fix would introduce a hard
import-time dependency that breaks the script in its documented running
environment. The "duplicated literal" is a deliberate Chesterton's-fence
avoidance of that dependency, not an oversight -- left unchanged.
**Original issue:** `V4_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"` duplicates
`tinker_reasoning_data_v4.BASE_MODEL`; if the v4 base model name ever
changes, this file would silently continue passing the stale name to
`capture_judge_responses_tinker.py --base-model`.

## Verification

- `python -c "import ast; ast.parse(...)"` (Tier 2 syntax check): all 9
  modified files pass.
- `pytest tests/test_merge_adapter_moe_routing.py`: 2 passed, 2 skipped
  (the 2 skips require `tinker_cookbook`, unavailable in this env --
  pre-existing, unrelated to these fixes).
- `pytest tests/test_tinker_reasoning_data_v4.py`: collection error in
  this env (`ModuleNotFoundError: tinker_cookbook`) -- confirmed
  pre-existing on the pre-fix commit too (this file was not touched by any
  fix; the test requires `.venv-tinker`, not present in this worktree).
- Functional self-checks (ad hoc, not committed as test files per YAGNI --
  each fix is either a trivial mechanical change covered by the syntax
  check, or was additionally exercised with synthetic data/mocks inline
  during this session): `_adapter_content_hash` hash-stability/hash-diff
  (CR-03), bootstrap reproducibility + quality-resampling (WR-01/WR-02),
  fail-receipt path derivation (WR-03), infra-error tracking in both
  scripts (WR-04), CI-overlap computation (WR-05).

---

_Fixed: 2026-07-14T12:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
