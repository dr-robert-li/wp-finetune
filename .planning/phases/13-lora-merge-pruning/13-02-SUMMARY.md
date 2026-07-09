---
phase: 13-lora-merge-pruning
plan: 02
subsystem: infra
tags: [moe, pruning, vllm, gate-before-remove, reap, qwen3-30b-a3b]

requires:
  - phase: 11-compression-packaging
    provides: scripts/sieve_expert_mask_inference.py (build_ksweep_mask/apply_mask), scripts/serve_30_70_vllm.sh, scripts/_sieve_vllm_patch/sitecustomize.py (SIEVE_MASK_NPY router patch), scripts/sieve_capture_judge_http.py, the _reset_wpbench_grader pattern (fix 8c4b167), judge max_tokens 2048 (fix cd36a5e) — all reused unchanged
  - phase: 13-lora-merge-pruning
    plan: 01
    provides: output/prune/aimer_scores_{gen,judge}.npy — real [48,128] AIMER score arrays consumed by --dry-run and (eventually) real gate runs
provides:
  - scripts/prune_gated_eval.py — gate-before-remove eval driver (PRUNE-03): score-array + ratio -> keep-mask -> vLLM serve + wp-bench + 3-seed judge, gated on vLLM-measured bars
  - scripts/reap_prune.py — REAP calibration-saliency scorer module (PRUNE-02): REAPCollector class, unit-tested; compute_reap_scores() calibration entry point deliberately NOT run
  - tests/test_reap_prune.py — Wave-0 unit tests for REAPCollector
affects: [13-04-gate-before-remove-execution, 13-05-reap-calibration-conditional, 13-06-physical-pruning]

tech-stack:
  added: []
  patterns:
    - "Gate-before-remove driver: any per-expert score array (routing counts, AIMER norms, REAP saliency) feeds build_ksweep_mask 1:1 (Pattern 1, zero-diff mask reuse) — the driver never needs its own mask-building logic, only its own regression-bar gating logic"
    - "Protected-mask sha256 re-verification immediately before every serve (T-13-03): defense-in-depth on top of build_ksweep_mask's own protected-union guarantee"

key-files:
  created:
    - scripts/prune_gated_eval.py
    - scripts/reap_prune.py
    - tests/test_reap_prune.py
  modified: []

key-decisions:
  - "prune_gated_eval.py's CLI separates --dry-run (both axes, CPU-only, prints planned arms) from a real --axis {gen,judge} gate run (GPU, executes in 13-04/13-05) — matches the plan's explicit CPU-verifiable-now vs GPU-later split"
  - "Judge parse-rate is computed as fraction of the 121-item val set that produced a parseable score in the ensemble (len(ensemble)/len(labels)), consistent with how sieve_ksweep_run.py's val_labels_v1 alignment already works"
  - "reap_prune.py's compute_reap_scores() raises NotImplementedError by design — it documents the real 13-05 calibration entry point contract (checkpoint_dir, calibration_jsonl_paths, sample_count) without executing any forward pass in this plan; only REAPCollector's accumulation logic is exercised by tests"
  - "REAPCollector's paired make_gate_hook/make_expert_hook are provided as the intended real-model wiring extending profile_base_model.RoutingCollector's hook pattern, but are marked with a ponytail: comment noting the exact HF Qwen3MoE per-expert token-subset alignment must be verified against the real model at 13-05 calibration time — not exercised here (no GPU/model in this plan)"

patterns-established:
  - "Any new per-expert scoring method (weight-based like AIMER, or calibration-based like REAP) plugs into prune_gated_eval.py's build_gated_mask() without modification — score array is the only interface contract"

requirements-completed: [PRUNE-02, PRUNE-03]

coverage:
  - id: D1
    description: "Gate-before-remove driver builds a keep-mask from any score array + ratio (k=96/64/32 for 25/50/75%), asserts the protected subset holds, re-verifies protected-mask sha256 against prune_set_for_phase13.json before any serve, and gates gen/judge results against the vLLM-measured bars (0.4284/0.7555/0.95), with Tinker-native 0.842/0.827 absent from bar logic"
    requirement: "PRUNE-03"
    verification:
      - kind: unit
        ref: "scripts/prune_gated_eval.py --self-check (mask build + protected-subset assertion + sha256 tamper-detection + bar-constant checks)"
        status: pass
      - kind: other
        ref: "scripts/prune_gated_eval.py --dry-run --ratio {25,50,75} (real AIMER score arrays from 13-01): prints planned gen/judge arms and per-layer keep counts without serving"
        status: pass
    human_judgment: false
  - id: D2
    description: "REAP calibration-saliency scorer (REAPCollector: sum(gate_weight*output_norm)/count, zero for inactive experts) implemented and unit-tested against a synthetic hook-event fixture; the real calibration forward pass is deferred to 13-05"
    requirement: "PRUNE-02"
    verification:
      - kind: unit
        ref: "tests/test_reap_prune.py#test_scores_shape, test_hand_computed_saliency_mean, test_inactive_expert_scores_zero_no_divide_by_zero, test_reset_clears_accumulators"
        status: pass
      - kind: unit
        ref: "scripts/reap_prune.py --self-check"
        status: pass
      - kind: other
        ref: "grep confirms compute_reap_scores() is never called anywhere in this plan and raises NotImplementedError if invoked"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 02: Gate-Before-Remove Eval Driver + REAP Calibration-Saliency Scorer Summary

**Built the gate-before-remove vLLM driver (score array + ratio -> masked serve + wp-bench + 3-seed judge, gated on 0.4284/0.7555/0.95) and the REAP saliency-scorer module (REAPCollector, unit-tested), with zero real GPU serving in this plan.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-10
- **Tasks:** 2/2
- **Files modified:** 3 (2 new scripts, 1 new test file)

## Accomplishments
- `scripts/prune_gated_eval.py` builds a keep-mask from ANY per-expert score array (`build_gated_mask`), reusing `build_ksweep_mask` from Phase 11 unchanged (scores replace routing counts 1:1)
- Protected-mask sha256 is re-verified against `output/sieve/prune_set_for_phase13.json` immediately before any serve (`verify_protected_sha`), on top of the protected-subset assertion already guaranteed by `build_ksweep_mask`
- Gen-axis gating reuses `run_eval_reasoning._wpbench_with_boot` + the Phase-11 grader-reset fix (8c4b167); judge-axis gating serves 3 seeds sequentially (GB10 one-model-at-a-time) with the shared mask and captures via `sieve_capture_judge_http` at `max_tokens=2048` (fix cd36a5e)
- Regression bars hard-coded to the vLLM-measured values from `prune_set_for_phase13.json` (gen wp-bench 0.4284, judge ensemble rho 0.7555, judge parse-rate 0.95, s1 fallback bar 0.7497 recorded alongside) — Tinker-native 0.842/0.827 confirmed absent from bar logic by both a self-check assertion and a manual grep
- `--dry-run` exercised against the real `output/prune/aimer_scores_{gen,judge}.npy` arrays from 13-01 for all three ratios (25/50/75), printing correct per-layer keep counts (e.g. ratio=25: 5076 gen / 5075 judge total experts kept, min/max per layer 99/112) with `protected_retained=True` in every case
- `scripts/reap_prune.py::REAPCollector` implements the REAP formula (`S_j = mean(gate_weight * ||expert_output||_2)`) via `sum/max(count,1)`, guaranteeing a defined 0.0 for never-activated experts (no divide-by-zero) — verified by 4/4 synthetic-fixture tests and a self-check
- `compute_reap_scores()` is provided as the documented 13-05 calibration entry point (signature: `checkpoint_dir, calibration_jsonl_paths, sample_count`) but deliberately raises `NotImplementedError` and is never invoked in this plan, per the plan's explicit "DO NOT run it here" instruction

## Task Commits

Each task was committed atomically:

1. **Task 1: Gate-before-remove eval driver (PRUNE-03)** - `a3c0fb6` (feat)
2. **Task 2: REAP scorer tests** - `618c9b5` (test — RED, module-level importorskip skips cleanly until implementation lands)
3. **Task 2: REAP scorer implementation** - `6483b8e` (feat — GREEN, 4/4 tests pass, self-check exits 0)

_TDD gate sequence for Task 2: test(618c9b5) -> feat(6483b8e), confirmed in git log._

## Files Created/Modified
- `scripts/prune_gated_eval.py` - gate-before-remove driver: `build_gated_mask`, `verify_protected_sha`, `run_gen_gate`/`_capture_judge_seed`/`score_judge_gate` (real-serve path, GPU, executes in 13-04/13-05), `_dry_run` (CPU-only), `--self-check`
- `scripts/reap_prune.py` - `REAPCollector` (record/scores/reset), `make_gate_hook`/`make_expert_hook` (intended real-model wiring, unexercised here), `compute_reap_scores` (13-05 entry point stub, raises if called), `--self-check`
- `tests/test_reap_prune.py` - synthetic hook-event fixture; shape, hand-computed saliency mean, inactive-expert=0/no-divide-by-zero, reset

## Decisions Made
- CLI split between `--dry-run` (both axes, CPU-only, works today against real 13-01 score arrays) and a real `--axis {gen,judge}` gate run (GPU, deferred to 13-04/13-05) — keeps this plan fully CPU-verifiable while the real driver code is ready to execute unchanged later
- Judge parse-rate defined as `len(ensemble)/len(labels)` over the 121-item val set, consistent with the existing `sieve_ksweep_run.py` alignment convention (no new parsing logic needed)
- `compute_reap_scores()` intentionally left as a documented stub (raises `NotImplementedError`) rather than a real forward-pass implementation — the exact HF Qwen3MoE per-expert hook wiring is non-trivial to get right without a live model to validate against, and the plan explicitly defers the calibration run itself to 13-05 conditional on AIMER@25% passing

## Deviations from Plan

None - plan executed exactly as written. The plan itself explicitly scoped `compute_reap_scores` as "provide the entry point but DO NOT run it here" — implementing it as a documented, non-executing stub with a fully tested `REAPCollector` beneath it satisfies that instruction directly, not a deviation from it.

## Issues Encountered

One self-check bug found and fixed during development (not a plan deviation, an implementation bug caught by its own verification loop): the initial `_self_check()` for `prune_gated_eval.py` used a 16-expert synthetic array while `RATIO_TO_K` maps to k=96/64/32 (real 128-expert budgets) — numpy's `argsort(...)[-96:]` on a 16-element array silently clips to all 16 elements, so the assertion `kept[0].sum() >= expected_k` failed. Fixed by sizing the self-check fixture to the real `N_EXPERTS=128`. Confirmed working before the task commit.

## User Setup Required

None - no external service configuration required.

## Threat Flags

None - no new network endpoints, auth paths, or trust-boundary schema changes. The one new trust boundary this plan touches (built keep-mask -> vLLM router patch, T-13-03) is explicitly in the plan's threat model and mitigated by the sha256 re-verification + protected-subset assertion implemented here.

## Next Phase Readiness

- `scripts/prune_gated_eval.py` is ready for 13-04 to invoke with real `--axis gen`/`--axis judge` runs against the AIMER score arrays (and later REAP scores, once 13-05 runs `compute_reap_scores` — not yet implemented, tracked as a future task, not a blocker for 13-04's AIMER-only gate runs)
- `scripts/reap_prune.py::REAPCollector` is ready to accept real hook-event data once 13-05's forward-pass wiring against the live Qwen3MoE model is built and validated (conditional on AIMER@25% passing gates per 13-CONTEXT)
- No blockers for 13-04 (gate-before-remove execution) or 13-03 (overlap analysis, selection rule, physical surgery planning)

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

All created files exist on disk (`scripts/prune_gated_eval.py`, `scripts/reap_prune.py`, `tests/test_reap_prune.py`); all 3 task commit hashes (`a3c0fb6`, `618c9b5`, `6483b8e`) found in git log.
