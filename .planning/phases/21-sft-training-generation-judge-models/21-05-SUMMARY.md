---
phase: 21-sft-training-generation-judge-models
plan: 05
subsystem: evaluation
tags: [gen-model, merge, wp-bench, ci-aware, qwen3.6, moe, vllm, wave-3]

# Dependency graph
requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "21-01: routed MoE-expert merge path proven (merge_adapter.py -> tinker_cookbook build_hf_model, 240/240); 21-02: wp-gen-v4-manifest.json with promoted wp-gen-v4-ep3 sampler checkpoint"
provides:
  - "models/Qwen3.6-35B-A3B-gen-v4-merged -- merged generation model (66.99 GiB, local)"
  - "output/base21/gen03_merge.json -- gen merge receipt: merge_ok=true, 240/240 modules, base_vs_merged_differs=true"
  - "output/base21/gen03_wpbench.json -- GEN-03 wp-bench receipt: 0.372 overall / CI-lower 0.2847 vs floor 0.4286, pass=false (RECORDED MISS), fresh raw new-base anchor 0.4897"
  - "scripts/build_gen03_merge.py + scripts/build_gen03_wpbench.py -- reusable merge+bench drivers for the v4 base"
  - "scripts/serve_base20_vllm.sh SERVED_MODEL_NAME env toggle (needed by any future wp-bench run against a serve_base20-served model)"
affects: [21-06-judge-eval, phase-21-verification, v4.0-milestone-disposition]

tech-stack:
  added: []
  patterns:
    - "wp-bench's litellm config hardcodes served model name wp-30_70 -- any serve script other than serve_30_70_vllm.sh must set --served-model-name wp-30_70 or every bench request 404s AFTER wp-env boots (silent-looking failure, error only in wp_bench_run.log)"
    - "run_eval_reasoning._run_wpbench returns ran=True even when the wp-bench subprocess exits non-zero -- callers must check wpbench_score is not None, not just ran"
    - "urllib.request.urlretrieve has no retry/resume -- multi-GiB Tinker archive downloads need curl -L --fail -C - --retry (curl exit 18 semantics prevent silently-truncated tars reaching extraction)"
    - "CI-aware overall-score bootstrap must be stratified: wp-bench overall is a 0.3/0.4/0.3 weighted combination of unequal-size strata (320 knowledge / 24 execution), so resample per-stratum and recombine via the exact wp_bench.scoring formula each replicate"

key-files:
  created:
    - scripts/build_gen03_merge.py
    - scripts/build_gen03_wpbench.py
    - output/base21/gen03_merge.json
    - output/base21/gen03_wpbench.json
  modified:
    - scripts/serve_base20_vllm.sh

key-decisions:
  - "GEN-03 disposition: RECORDED MISS (per plan's explicit failure disposition, not forced/retried). Merged gen model wp-bench 0.372, CI lower 0.2847 < floor 0.4286."
  - "Inherited floor 0.4286 STANDS: the fresh raw new-base anchor measured 0.4897 -- ABOVE the floor, so the V4-RERUN-ROADMAP fresh-floor escape hatch (for a downward noise-band shift) does not apply; swapping floors was neither justified nor performed."
  - "The 21-01 moe_merge_probe.json receipt is a valid --expected-modules-manifest for the real gen adapter: the attached-module set (240 = 120 routed-expert + 120 shared_expert) is architecture-driven, not weight-value-driven, and GEN-02 trained with the identical train_mlp=True/train_attn=False/train_unembed=False topology."

metrics:
  duration: ~150min (dominated by 4 vLLM boots ~6.5min each + 2 full 344-test bench runs ~15min each + 2 GiB adapter download + 67 GiB merge)
  completed: 2026-07-14

status: complete
---

# Phase 21 Plan 05: GEN-03 Merge + wp-bench Codegen-Preservation Gate Summary

**The promoted gen adapter (wp-gen-v4-ep3) merged cleanly onto the local new base via the proven fused-expert path (240/240 modules, base-vs-merged generation visibly differs), but the merged model MISSED the wp-bench floor: 0.372 overall / CI-lower 0.2847 vs 0.4286 — while the RAW new base measured 0.4897 on the identical harness. The reasoning-mix SFT regressed the new base's codegen by ~11.8pp (beyond the 5.2pp seed-noise floor), echoing the v1.2 RC-B interference signature. Recorded as a valid miss per the plan's explicit failure disposition — nothing forced, floors not swapped.**

## Performance

- **Duration:** ~150 min wall (includes one download-failure retry and one bench-invocation fix + rerun)
- **Tasks:** 2/2
- **GB10 discipline:** sole user; every vLLM serve torn down in a finally block before the next residency (verified no stray containers / GPU processes after each step)

## Accomplishments

### Task 1 — Merge promoted gen adapter onto the new base (commits `656b8f7`, `daa2b5f`)

- Downloaded the promoted `wp-gen-v4-ep3` sampler checkpoint archive (2.08 GB) from Tinker via the REST archive-URL path with WR-10 member-validated extraction.
- Merged via `merge_adapter.py --config-path config/train_config_v4.yaml` — routed automatically through `tinker_cookbook.weights.build_hf_model` (the 21-01 gap-closure path). **Module-count guard: 240/240, 0 drops.**
- Real-generation base-vs-merged diff on a WordPress prompt: merged model emits direct clean code (`function get_current_user_display_name() {...}`); raw base emits thinking-process prose. `base_vs_merged_differs=true` — the adapter delta demonstrably landed.
- Merged model: `models/Qwen3.6-35B-A3B-gen-v4-merged` (66.99 GiB). Receipt: `output/base21/gen03_merge.json` (all plan assertions pass).

### Task 2 — Serve + full wp-bench vs the CI-aware floor (commits `eb0aeab`, `fa0b760`)

- Served the merged model via `serve_base20_vllm.sh` (LANGUAGE_MODEL_ONLY + new SERVED_MODEL_NAME=wp-30_70), Phase 15 LOCKED real-generation warm-up gate passed, then ran the exact reused `run_eval_reasoning._run_wpbench` harness: full 344-test suite (320 knowledge / 24 execution), `enable_thinking=False` via `_wpbench_pth`, max_tokens 2048, temperature 0.0, seed 1337, concurrency 4, request_timeout 1800s.
- **Merged gen model: overall 0.372, stratified-bootstrap 95% CI [0.2847, 0.4753] (n_boot=1000).** CI lower bound 0.2847 < floor 0.4286 → **pass=false**.
- Per the plan's fresh-floor clause, ran the measured raw new-base anchor (same harness, RAW `models/Qwen3.6-35B-A3B`): **0.4897 overall** — 6.1pp ABOVE the inherited floor. No downward noise-band shift exists, so the inherited floor stands and the miss is final. `floor_source="inherited"`.
- Receipt: `output/base21/gen03_wpbench.json` (plan's automated verify passes; both raw results files + run logs preserved under `output/base21/gen03_full/` and `gen03_fresh_new_base_anchor/`).

## The headline finding

| Measurement | wp-bench overall |
|---|---|
| RAW Qwen3.6-35B-A3B (new base, fresh anchor, this run) | **0.4897** |
| Merged gen model (reasoning-mix SFT, this run) | **0.372** |
| Floor (carried v3.0 acceptance bar) | 0.4286 |
| OLD base raw anchor (Qwen3-30B-A3B, 2026-07-12) | 0.4033 |
| v1.2 shipped gen model (Gate-1) | 0.4484 |

The new base is substantially stronger at WordPress codegen out of the box (0.4897 vs the old base's 0.4033), but the v1.2-recipe reasoning-mix SFT **damaged** it — an 11.8pp drop vs its own raw base, well beyond the 5.2pp seed-noise floor. This is the same interference direction as the historical RC-B finding (v1.2's first reasoning merge: 0.4537 → 0.3716) which was eventually mitigated by recipe tuning on the old base. GEN-03's purpose was exactly to measure this rather than assume it — measured, recorded, not papered over.

## Task Commits

1. **Task 1: GEN-03 merge** — `656b8f7` (fix: resumable download), `daa2b5f` (feat: merge receipt)
2. **Task 2: wp-bench gate** — `eb0aeab` (fix: served-model-name + error surfacing), `fa0b760` (feat: wp-bench receipt)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `urllib.request.urlretrieve` died mid-download of the 2.08 GB adapter archive**
- **Found during:** Task 1, first run
- **Issue:** `ContentTooShortError` at 1.42/2.08 GB — transient network truncation; urlretrieve has no retry/resume.
- **Fix:** replaced with `curl -L --fail -C - --retry 5 --retry-delay 10` (resumes the partial, retries resets; curl exit-18 semantics guarantee a truncated tar cannot reach extraction).
- **Files modified:** `scripts/build_gen03_merge.py`
- **Commit:** `656b8f7`

**2. [Rule 1 - Bug] wp-bench 404'd on every request: served-model-name mismatch**
- **Found during:** Task 2, first run
- **Issue:** `_run_wpbench`'s litellm config hardcodes model name `wp-30_70` (set by `serve_30_70_vllm.sh`, the working base-anchor precedent). The v4 `serve_base20_vllm.sh` sets no `--served-model-name`, so vLLM served as `/workspace/model` — every bench request failed with `The model wp-30_70 does not exist` after wp-env boot. My wrapper also only checked `ran` (which is True even on bench-subprocess failure), so the error surfaced as a confusing results-glob crash.
- **Fix:** added an optional `SERVED_MODEL_NAME` env toggle to `serve_base20_vllm.sh` (default unset — prior callers unaffected); `build_gen03_wpbench.py` passes `SERVED_MODEL_NAME=wp-30_70` and now halts loudly on `wpbench_score=None` with the harness's own error.
- **Files modified:** `scripts/serve_base20_vllm.sh`, `scripts/build_gen03_wpbench.py`
- **Commit:** `eb0aeab`

### Non-deviation notes

- The fresh raw new-base anchor run was IN-PLAN (the plan's own conditional: "If the merged score is near/below 0.4286 ... derive a fresh floor by running the same harness on the RAW new base"). It ran, and the measurement showed the escape hatch does NOT apply (base moved UP, not down) — so the floor was not swapped.

## Threat Model Compliance

- **T-21-12 (silent all-zero/partial merge scores as "no regression"):** mitigated for real — 240/240 module guard + base-vs-merged real-generation diff both asserted in `gen03_merge.json` BEFORE any eval; and the eval indeed found a regression the guard proves is not a merge artifact.
- **T-21-13 (thinking-on/truncation skews wp-bench):** exact reused harness — `enable_thinking=False` via `_wpbench_pth` (recorded in the receipt), max_tokens 2048, seed 1337, real-generation warm-up gate before each bench.
- **T-21-14 (GB10 memory collision):** sole GB10 user; all four vLLM serves stopped in finally blocks; no stray containers or GPU processes after completion.

## Known Stubs

None — all measurements are real (two full 344-test wp-bench runs, real Tinker archive download, real 67 GiB merge). No mocked calls, no fabricated receipts.

## Next Phase Readiness

- **GEN-03: SATISFIED as a RECORDED MISS** (the plan's success criterion is "pass or recorded miss" — codegen preservation is measured, not assumed).
- **Milestone implication (for phase verification / v4.0 disposition):** the reasoning-mix SFT recipe that worked on the old base regresses codegen on Qwen3.6-35B-A3B. Candidate levers, if the milestone requires clearing the floor: epoch selection (ep1/ep2 checkpoints persist in `wp-gen-v4-manifest.json` — less SFT may mean less damage), more wp_gen codegen replay in the mix, or lower rank/LR — the same lever family that closed RC-B on the old base. Also note the raw new base (0.4897) ALREADY clears the v1.2 shipped figure (0.4484); "what the fine-tune buys" needs re-framing against a much stronger base.
- The merged model stays on disk for any follow-up probing; ep1/ep2 sampler checkpoints remain available on Tinker for a cheap epoch-sweep without retraining.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-14*

## Self-Check: PASSED

All 4 created artifact files + merged model dir verified present on disk; all 4 task commit hashes (`656b8f7`, `daa2b5f`, `eb0aeab`, `fa0b760`) verified present in `git log`.
