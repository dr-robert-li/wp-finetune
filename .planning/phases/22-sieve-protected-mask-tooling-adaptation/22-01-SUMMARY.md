---
phase: 22-sieve-protected-mask-tooling-adaptation
plan: 01
subsystem: infra
tags: [moe-sieve, protected-mask, k-sweep, vllm, qwen3.6-35b-a3b, torch, transformers]

# Dependency graph
requires:
  - phase: 20-base-bring-up
    provides: "20-04 empirical fact — LIVE module tree is FLAT model.model.layers.*, not the nested language_model.* on-disk convention"
  - phase: 21-sft-training-generation-judge-models
    provides: "v4 judge merged checkpoints (judge-s0/s1/s2) that Plan 22-02 will profile"
provides:
  - "scripts/sieve_arch.py: single arch-awareness helper (arch_dims, layer_strata, resolve_moe_layers, infer_dims_from_records, resolve_task_token_ids) used by every Sieve/mask/k-sweep script"
  - "All 6 profiler/mask/k-sweep/vLLM-patch scripts adapted to derive (n_layers, n_experts) from config/data instead of hardcoding v3's 48/128"
  - "Per-stratum (deltanet/attention) E_eff reporting in profile_merged_model.py, alongside existing collapsed stats"
  - "vLLM router-mask patch resolves the qwen3_5_moe/qwen3_next MoE-block class first, qwen3_moe (v3) fallback, fail-loud on zero resolution"
affects: [22-02-vllm-smoke, 25-k-sweep, 26-prune]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sieve_arch._cfg()/_text_config() helper: supports both dict-like test-fixture configs (.get()) and real transformers PretrainedConfig objects (attribute-only, no .get())"
    - "resolve_moe_layers()/resolve_task_token_ids(): raise-not-empty and None-degrade contracts respectively, so a wrong traversal path or missing task-token extension fails loud/degrades gracefully instead of silently mis-profiling"
    - "sys.path bootstrap (PROJECT_ROOT insert + noqa: E402) for scripts/*.py that import scripts.* and must also run as `python scripts/foo.py` directly — established repo convention (scripts/sieve_ksweep_run.py), applied to 3 sieve scripts here"

key-files:
  created:
    - scripts/sieve_arch.py
    - tests/test_sieve_arch.py
    - tests/test_sieve_vllm_patch.py
    - .planning/phases/22-sieve-protected-mask-tooling-adaptation/deferred-items.md
  modified:
    - scripts/profile_base_model.py
    - scripts/profile_merged_model.py
    - scripts/extract_protected_mask.py
    - scripts/sieve_cross_seed_overlap.py
    - scripts/sieve_expert_mask_inference.py
    - scripts/sieve_protected_retention.py
    - scripts/_sieve_vllm_patch/sitecustomize.py
    - tests/test_protected_mask.py
    - tests/test_sieve_cross_seed_overlap.py
    - tests/test_sieve_ksweep_mask.py
    - tests/test_sieve_protected_retention.py

key-decisions:
  - "resolve_moe_layers tries model.layers -> model.language_model.layers -> language_model.layers in that order (flat-first, per 20-04 empirical fact), NOT the ROADMAP's literal nested-first guess"
  - "layer_strata's v3 fallback (neither layer_types nor full_attention_interval present) returns all-DELTANET_STRATUM per the plan's explicit spec, even though v3 has no DeltaNet layers at all — the label is a moot uniform-fallback choice on v3, not a semantic claim"
  - "resolve_task_token_ids validates BOTH '<wp_gen>' and '<wp_judge>' are in tokenizer.get_vocab() before returning the caller-supplied numeric defaults, else (None, None) — the v4 judge (vocab 248320) has neither token, so RoutingCollector degrades to total-only tagging"
  - "sieve_protected_retention.py's mask.shape==(48,128) and mask.sum()==1480 asserts replaced with dtype==bool + non-empty — the v4 mask is a fresh Phase-25 profile of unknown shape/count, those v3-specific literals cannot be asserted generically"
  - "_resolve_moe_block_class tolerates an unknown-but-present class name by scanning a resolved module for exactly one *SparseMoeBlock class; an ambiguous scan (2+ matches) does NOT guess — falls through to the next candidate or raises"

requirements-completed: [GATE4-02]

coverage:
  - id: D1
    description: "sieve_arch.py provides config/data-derived dims (arch_dims), strata (layer_strata), traversal (resolve_moe_layers), JSONL-inferred dims (infer_dims_from_records), and task-token resolution (resolve_task_token_ids) — no v3 hardcode in a load-bearing position"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_sieve_arch.py (13 tests)"
        status: pass
      - kind: unit
        ref: "scripts/sieve_arch.py --self-check (demo())"
        status: pass
    human_judgment: false
  - id: D2
    description: "profile_base_model.py / profile_merged_model.py derive (n_layers, n_experts) from model.config via sieve_arch.arch_dims and register hooks via sieve_arch.resolve_moe_layers with a hook-count==n_layers assert; RoutingCollector gen_id/judge_id are now resolved via sieve_arch.resolve_task_token_ids"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_routing_collector.py, tests/test_jaccard_stability.py, tests/test_eeff.py (v3 regression suite, unchanged behavior confirmed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "extract_protected_mask.py / sieve_cross_seed_overlap.py infer (n_layers, n_experts) from the routing_report.jsonl itself via sieve_arch.infer_dims_from_records, not hardcoded 48/128"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_protected_mask.py::TestExtractAndReportDimInference::test_v4_scale_dims_inferred_from_jsonl"
        status: pass
      - kind: unit
        ref: "tests/test_sieve_cross_seed_overlap.py::TestLoadSeedCounts::test_infers_dims_from_v4_scale_records"
        status: pass
    human_judgment: false
  - id: D4
    description: "profile_merged_model.py reports per-layer E_eff split by stratum (deltanet vs attention), not collapsed into one mean — added to routing_report.jsonl's jaccard_stability.json sidecar and the function's return dict"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_sieve_arch.py::TestLayerStrata (strata computation the aggregation consumes)"
        status: pass
    human_judgment: true
    rationale: "The strata_eeff aggregation itself has no dedicated unit test isolating profile_merged_model's internal wiring (it runs inside a GPU forward-pass function not exercised by CPU-only tests) — code-reviewed against the plan's behavior spec and the underlying layer_strata call is unit-tested, but empirical proof on a real forward pass is Plan 22-02's job."
  - id: D5
    description: "sieve_protected_retention.py derives the expected mask shape from the loaded mask itself (dtype==bool + non-empty), dropping the v3-specific shape==(48,128)/sum==1480 asserts"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_sieve_protected_retention.py::TestRetentionCheck::test_v4_scale_256_experts_retention"
        status: pass
      - kind: unit
        ref: "scripts/sieve_protected_retention.py --self-check"
        status: pass
    human_judgment: false
  - id: D6
    description: "The vLLM router-mask patch resolves the qwen3_5_moe/qwen3_next MoE-block class first (v4), qwen3_moe fallback (v3), and fails LOUD (RuntimeError) if no candidate resolves while SIEVE_KEEP_MASK_NPY is set — the -inf masking hook math is unchanged"
    requirement: "GATE4-02"
    verification:
      - kind: unit
        ref: "tests/test_sieve_vllm_patch.py (5 tests: exact-match pick, scan-fallback, ambiguous-scan non-resolution, exhausted-list raise, v4-before-v3 ordering)"
        status: pass
    human_judgment: true
    rationale: "The exact qwen3_5_moe/qwen3_next module+class names are best-known, not yet confirmed against the installed vLLM (vLLM is not importable on this host) — live confirmation is Plan 22-02's job inside the serving container. The resolver logic itself is fully unit-tested with stubs."

# Metrics
duration: ~20min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 01: Sieve/Protected-Mask Tooling Adaptation Summary

**scripts/sieve_arch.py (new arch-awareness helper) wires config/data-derived dims + strata into all 6 Sieve/profiler/mask/k-sweep scripts + the vLLM router-mask patch, replacing every v3-hardcoded 48-layer/128-expert literal with a load-bearing config or JSONL derivation — 47/47 tests green, v3 [48,128]/1480 fixtures still pass unchanged.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-07-15
- **Tasks:** 3
- **Files modified:** 11 modified, 4 created

## Accomplishments

- `scripts/sieve_arch.py`: `arch_dims`, `layer_strata`, `resolve_moe_layers`, `infer_dims_from_records`, `resolve_task_token_ids` — one dependency-light (numpy+stdlib) helper every consumer now imports; `layer_strata` verified against the REAL v4 `config.json` `layer_types` array (attention at exactly `[3,7,11,15,19,23,27,31,35,39]`)
- `profile_base_model.py` / `profile_merged_model.py`: hook registration now goes through `sieve_arch.resolve_moe_layers(model)` with a `hook-count == n_layers` assertion (raises, doesn't silently zero-hook — the exact 20-04 failure mode); `RoutingCollector` gained `gen_id`/`judge_id` constructor params resolved via `sieve_arch.resolve_task_token_ids`
- `profile_merged_model.py` additionally reports per-stratum (`deltanet`/`attention`) E_eff mean/max/var in `jaccard_stability.json` and its return dict, alongside the existing collapsed stats
- `extract_protected_mask.py` / `sieve_cross_seed_overlap.py`: count-array dims inferred from the routing JSONL itself via `infer_dims_from_records`, not hardcoded
- `sieve_protected_retention.py`: the v3-specific `mask.shape==(48,128)` / `mask.sum()==1480` asserts replaced with `dtype==bool` + non-empty (the v4 mask is a fresh Phase-25 profile of unknown shape/count)
- `scripts/_sieve_vllm_patch/sitecustomize.py`: `_resolve_moe_block_class` extracted and moved to module scope (out of the `if SIEVE_KEEP_MASK_NPY:` block) so it's unit-testable without vLLM installed; ordered candidate list tries qwen3_5_moe/qwen3_next (v4) first, qwen3_moe (v3) fallback, fail-loud `RuntimeError` preserved when no candidate resolves
- No load-bearing `48`/`128`/`1480` literal remains anywhere in the six adapted scripts (confirmed by grep sweep — only defaults, back-compat module constants, and comments remain)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scripts/sieve_arch.py** - `33b8615` (feat)
2. **Task 2: Wire sieve_arch into the profiler + mask + k-sweep consumers** - `0120143` (feat)
3. **Task 3: Adapt the vLLM router-mask patch class resolution** - `a04491a` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `scripts/sieve_arch.py` - New: arch_dims, layer_strata, resolve_moe_layers, infer_dims_from_records, resolve_task_token_ids
- `tests/test_sieve_arch.py` - New: 13 tests covering all 5 functions on v3/v4 fixtures + stub models
- `scripts/profile_base_model.py` - RoutingCollector gen_id/judge_id params; hook registration via sieve_arch.resolve_moe_layers + assert; dims via sieve_arch.arch_dims
- `scripts/profile_merged_model.py` - Same wiring as profile_base_model + per-stratum E_eff aggregation (strata_eeff dict) in the jaccard sidecar and return dict
- `scripts/extract_protected_mask.py` - n_layers/n_experts derived from infer_dims_from_records(records) instead of hardcoded 48/128
- `scripts/sieve_cross_seed_overlap.py` - load_seed_counts infers dims from the file; N_LAYERS/N_EXPERTS kept as empty-file fallback only; sys.path bootstrap added
- `scripts/sieve_expert_mask_inference.py` - N_LAYERS/N_EXPERTS documented as non-load-bearing import back-compat; sys.path bootstrap added
- `scripts/sieve_protected_retention.py` - mask shape/1480-count asserts replaced with dtype+non-empty; sys.path bootstrap added
- `scripts/_sieve_vllm_patch/sitecustomize.py` - _resolve_moe_block_class + _install moved to module scope; ordered v4-first candidate list
- `tests/test_sieve_vllm_patch.py` - New: 5 tests for the resolver (stub modules, no vLLM/GPU needed)
- `tests/test_protected_mask.py`, `tests/test_sieve_cross_seed_overlap.py`, `tests/test_sieve_ksweep_mask.py`, `tests/test_sieve_protected_retention.py` - v4 (40,256)-scale fixture cases added alongside existing v3 (48,128) cases
- `.planning/phases/22-sieve-protected-mask-tooling-adaptation/deferred-items.md` - New: logs out-of-scope pre-existing failures found during the full-suite regression sweep

## Decisions Made

- `resolve_moe_layers` candidate order is `model.layers` → `model.language_model.layers` → `language_model.layers` (flat-first), matching the 20-04 empirical LIVE-tree fact, not the ROADMAP's literal nested-path guess (22-VALIDATION.md's reconciled conflict, already binding per the plan)
- `_cfg()`/`_text_config()` helper in sieve_arch.py supports both `.get()`-style dict fixtures AND real `transformers.PretrainedConfig` objects (attribute-only, confirmed via a direct interpreter check that `PretrainedConfig` does NOT implement `.get()`) — the plan's literal `config.get("text_config", config)` wording would break on a real loaded model.config, so the helper generalizes it
- `resolve_task_token_ids` gates on tokenizer vocab presence of `<wp_gen>`/`<wp_judge>` and returns the caller-supplied numeric defaults (not a fresh vocab lookup) — matches the existing WP_GEN_ID/WP_JUDGE_ID constants semantics exactly
- Per-stratum E_eff is captured from the FULL-pass collector state before the Jaccard subsample pass resets it (the subsample pass mutates collector state in-place)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] sys.path bootstrap for direct-script self-check execution**
- **Found during:** Task 2 verification (`.venv-tinker/bin/python scripts/sieve_expert_mask_inference.py --self-check` etc.)
- **Issue:** `scripts/sieve_expert_mask_inference.py` and `scripts/sieve_protected_retention.py` already imported `from scripts.sieve_cross_seed_overlap import ...` at module level pre-existing this plan; running them as a direct file path (not `python -m`) never had the project root on `sys.path`, so `ModuleNotFoundError: No module named 'scripts'` fired immediately, blocking the plan's own `<verify>` block. Additionally, adding `from scripts.sieve_arch import infer_dims_from_records` to `sieve_cross_seed_overlap.py` (Task 2's own change) newly broke ITS previously-working direct-script self-check the same way.
- **Fix:** Added the established repo convention (`PROJECT_ROOT = Path(__file__).resolve().parent.parent; if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))`, already used in `scripts/sieve_ksweep_run.py` and a dozen other scripts) to all three files, before the `scripts.*` import (`# noqa: E402`).
- **Files modified:** scripts/sieve_cross_seed_overlap.py, scripts/sieve_expert_mask_inference.py, scripts/sieve_protected_retention.py
- **Verification:** All three `--self-check` invocations print `self-check OK`
- **Committed in:** `0120143` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — pre-existing gap on 2 files + a regression on a 3rd introduced by this plan's own edit, both fixed with the same established pattern)
**Impact on plan:** Necessary to satisfy the plan's own `<verify>` block for Task 2. No scope creep — same fix pattern already standard in this codebase.

## Issues Encountered

- Full-suite regression sweep (`pytest tests/ --ignore=tests/test_preflight.py`) surfaced 7 unrelated pre-existing failures (Tinker auth env, `dotenv` missing) — confirmed via `git stash` to reproduce identically on the unmodified tree, logged to `deferred-items.md`, not fixed (out of scope per SCOPE BOUNDARY).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 22-02 (GB10 smoke) can now load the real v4 judge model and call `sieve_arch.resolve_moe_layers`/`arch_dims`/`layer_strata` against the live `model.config` to empirically confirm 40 hooks resolve, `router_logits` last-dim is 256, and the strata pattern matches — this plan's code+unit-test layer is the prerequisite, not yet the empirical proof
- The vLLM patch's exact qwen3_5_moe/qwen3_next module+class names remain best-known pending live confirmation inside the serving container (Plan 22-02's job) — the resolver's scan-fallback (single `*SparseMoeBlock` class in a resolved module) gives it a second chance even if the literal class name misses
- No blockers for Plan 22-02

---
*Phase: 22-sieve-protected-mask-tooling-adaptation*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 6 claimed files found on disk; all 3 task commit hashes (33b8615, 0120143, a04491a) found in git log.
