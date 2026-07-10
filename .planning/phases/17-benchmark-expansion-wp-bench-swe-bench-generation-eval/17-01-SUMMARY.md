---
phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
plan: 01
subsystem: eval
tags: [wp-bench, vllm, benchmark, qwen3-moe, bf16]

requires:
  - phase: 15-packaging
    provides: gate1_bf16_baseline.json (the 0.4484 comparison reference) and the bf16/Q8 GGUF packaging lineage
provides:
  - "Fresh, full 344-test wp-bench score for the v1.2 gen model on the actual vLLM bf16 shipping stack"
  - "Explicit same-stack delta vs the 0.4484 Gate-1 reference (-0.0119, within the 5.20pp seed-noise floor)"
  - "Restored scripts/_wpbench_pth + scripts/_wpbench_shim active dependencies (mis-archived in Phase 16 cleanup)"
affects: [18-production-repo-sweep-hf-publication]

tech-stack:
  added: []
  patterns:
    - "Real-generation warm-up gate before capture (not /health) — Phase 15 lesson, now applied via scripts/_p0_vllm_smoke_serve.generate()"
    - "wp-bench enable_thinking=False injection via PYTHONPATH usercustomize.py monkeypatch (wp-bench's config schema has no native field for chat_template_kwargs)"

key-files:
  created:
    - scripts/bench17_wpbench_full_rerun.py
    - output/bench17/wpbench_full_gate_rerun.json
    - output/bench17/full_gate_rerun/wp_bench_results_20260711_050328.json
  modified:
    - deprecated/README.md

key-decisions:
  - "Restored scripts/_wpbench_pth/usercustomize.py and scripts/_wpbench_shim/npx from deprecated/ back to scripts/ — they are active runtime dependencies of run_eval_reasoning.py referenced via string path construction, which the Phase 16 cleanup's import-only grep missed"
  - "Reused scripts/run_eval_reasoning.py::_run_wpbench exactly (same request_timeout/max_tokens/concurrency/enable_thinking) rather than inventing a new wp-bench invocation, monkeypatching only the module-level PORT to 8020 to match the dgx_toolbox.yaml documented default"
  - "Copied raw results to output/wp-bench-results.json (untracked) purely to satisfy the plan's literal verify path — wp-bench actually writes a timestamped filename, a known quirk already documented in run_eval_reasoning.py"

requirements-completed: [BENCH-01]

coverage:
  - id: D1
    description: "Full (unlimited, 344-test) wp-bench run completed on the v1.2 gen model served via vLLM bf16, gated on real-generation warm-up"
    requirement: "BENCH-01"
    verification:
      - kind: other
        ref: "output/bench17/full_gate_rerun/wp_bench_results_20260711_050328.json (344 results: 320 knowledge + 24 execution)"
        status: pass
    human_judgment: false
  - id: D2
    description: "BENCH-01 receipt recorded with score, config, seed, serving-stack attestation, and explicit delta vs the 0.4484 Gate-1 reference"
    requirement: "BENCH-01"
    verification:
      - kind: other
        ref: "output/bench17/wpbench_full_gate_rerun.json"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-11
status: complete
---

# Phase 17 Plan 01: BENCH-01 Full wp-bench Rerun Summary

**Fresh full 344-test wp-bench score (0.4365) on the v1.2 gen model's vLLM bf16 shipping stack, confirming the 0.4484 Gate-1 figure reproduces within the project's own 5.20pp seed-noise floor — no regression, no stack anomaly.**

## Performance

- **Duration:** 25 min (including investigation of a broken dependency)
- **Started:** 2026-07-11T04:39:00Z
- **Completed:** 2026-07-11T05:04:00Z
- **Tasks:** 2 completed
- **Files modified:** 8 (2 restored + 6 new)

## Accomplishments
- Served `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (bf16, 57GB) via the existing `serve_30_70_vllm.sh` vLLM container pattern on port 8020, gated capture on a real one-word generation succeeding (not `/health`)
- Ran the full unmodified 344-test wp-core-v1 suite (320 knowledge + 24 execution) with `enable_thinking=False`, identical sampling config to the code path that produced the 0.4484 reference figure
- Result: overall **0.4365** (knowledge 0.490625, correctness/execution 0.395833) — delta **-0.0119** vs 0.4484, well within the 5.20pp seed-noise floor established in `output/sieve/optimal_k.json`
- Traced the provenance of 0.4484 through `gate1_bf16_baseline.json` -> `eval3_final_comparison.json` -> `sieve/optimal_k.json`'s `full_arm` -> `sieve_ksweep_run.py`'s k=full arm, confirming it is the same `_run_wpbench()` code path measured here
- Wrote the BENCH-01 receipt (`output/bench17/wpbench_full_gate_rerun.json`) with same-stack attestation: vLLM bf16, no GGUF/llama.cpp path anywhere
- Stopped the vLLM container after the run — GPU free for wave 2 (SWE-bench plans)

## Task Commits

1. **Task 1: Serve v1.2 gen model on vLLM bf16 and run the full wp-bench suite** - `4baebb3` (feat)
2. **Task 2: Build the BENCH-01 receipt and compare to the 0.4484 Gate-1 number** - `d8b9b2e` (feat)

**Preceding fix commit:** `3116665` (fix — restore mis-archived active dependency, required before Task 1 could run)

## Files Created/Modified
- `scripts/bench17_wpbench_full_rerun.py` - thin driver: boot vLLM at :8020 -> real-generation warm-up gate -> reuse `_run_wpbench` -> stop container
- `scripts/_wpbench_pth/usercustomize.py` - restored (was wrongly moved to `deprecated/` in Phase 16); PYTHONPATH monkeypatch injecting `enable_thinking=false`
- `scripts/_wpbench_shim/npx` - restored (was wrongly moved to `deprecated/` in Phase 16); PATH shim so wp-bench's `npx wp-env` resolves to the global `wp-env` bin
- `deprecated/README.md` - corrected to document the restore and the import-only-grep gap that missed these two string-referenced dependencies
- `output/bench17/wpbench_full_gate_rerun.json` - the BENCH-01 receipt (score + full serving stack + delta vs 0.4484)
- `output/bench17/full_gate_rerun/wp_bench_results_20260711_050328.json` + `.jsonl` + `wp_bench_run.log` + `wp_bench_config_tmp.yaml` - raw wp-bench harness output
- `output/bench17/raw_run_meta.json` + `driver_run.log` - driver script metadata/log

## Decisions Made
- Reused the exact serving + wp-bench invocation code path from `scripts/run_eval_reasoning.py::_run_wpbench` (via `scripts/sieve_ksweep_run.py`'s precedent) rather than writing a fresh wp-bench call, so sampling config (temp 0.0, max_tokens 2048, timeout 1800s, concurrency 4, enable_thinking=false) is provably identical to the code path that produced the 0.4484 comparison figure.
- Only the container port changed (8020 here vs. 8021 in the sieve driver) to match `config/wp-bench.yaml`/`config/dgx_toolbox.yaml`'s documented default — a cosmetic difference with no effect on generation quality.
- Restored the wrongly-deprecated `_wpbench_pth`/`_wpbench_shim` helpers to `scripts/` rather than pointing the code at their `deprecated/` location, matching this project's established convention (three other underscore-prefixed files were already restored during the Phase 16 cleanup for the same reason).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored `scripts/_wpbench_pth/usercustomize.py` and `scripts/_wpbench_shim/npx`, wrongly archived in Phase 16 cleanup**
- **Found during:** Task 1 pre-flight (reading `run_eval_reasoning.py` before serving)
- **Issue:** The Phase 16 pipeline-lockdown commit (`c236edf`) moved these two helper dirs to `deprecated/scripts/`, believing (per its own commit message) that "post-move grep shows zero active imports of any moved file." That grep checked Python `import` statements only. `scripts/run_eval_reasoning.py::_run_wpbench()` references them via runtime string path construction (`PROJECT_ROOT / "scripts" / "_wpbench_pth"`), which a static-import grep cannot catch. Without `_wpbench_pth`, the reasoning model's Qwen3 chat template defaults to `enable_thinking=true`, generation opens an unterminated `<think>` block on gen-style prompts, and wp-bench either times out or scores unparseable `<think>...` prefixes — exactly the failure mode this same helper was built to fix in Phase 4.4. Without `_wpbench_shim`, wp-env's `npx wp-env` shellout 404s/hangs on this host.
- **Fix:** `git mv deprecated/scripts/_wpbench_pth scripts/_wpbench_pth` and `git mv deprecated/scripts/_wpbench_shim scripts/_wpbench_shim`; corrected `deprecated/README.md` to document the restore and the grep gap.
- **Files modified:** `scripts/_wpbench_pth/usercustomize.py`, `scripts/_wpbench_shim/npx`, `deprecated/README.md`
- **Verification:** Task 1's wp-bench run completed cleanly with `enable_thinking=false` confirmed applied (no `<think>` contamination in any of the 344 results) and the execution-test grader's wp-env shellout succeeded (24/24 execution tests scored, not errored).
- **Committed in:** `3116665` (standalone fix commit, preceding the Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — missing referenced file)
**Impact on plan:** Essential for Task 1 to run at all; no scope creep. The restore is a pure bug fix in a prior phase's cleanup, not new functionality.

## Issues Encountered
None beyond the deviation above. The full run completed in a single pass with no retries, timeouts, or GPU/memory pressure (114GB available before load; 57GB bf16 model + KV cache at gpu_mem_util=0.55 fit comfortably).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- BENCH-01 (wp-bench full rerun) is complete and closed. GPU/host is free (vLLM container stopped, no leftover llama-server processes) for the wave-2 SWE-bench generation-eval plans (BENCH-02/03), which the phase's own research flagged as needing a Wave-0 throughput probe and an aarch64 Docker `arch` wrapper before scope can be locked.
- No blockers carried forward from this plan.

---
*Phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval*
*Completed: 2026-07-11*

## Self-Check: PASSED
All claimed files verified present on disk; all claimed commit hashes verified present in `git log --oneline --all`.
