---
phase: 21-sft-training-generation-judge-models
plan: 06
subsystem: evaluation
tags: [judge-rho, tinker-capture, vllm, merge, ci-aware, qwen3.6, ensemble, wave-4]

# Dependency graph
requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "21-03: 3 judge seed manifests (wp-judge-v4-s{1,0,2}, promoted ep3 each); 21-05: proven download/merge/serve patterns (curl -C - resumable, SERVED_MODEL_NAME toggle, finally-teardown); 21-04: raw-base 30/30 parse-fail baseline for interpretation"
provides:
  - "output/base21/judge03_capture_rho.json -- cheap-path receipt: per-seed 8192-cap Tinker-capture rho (s1 0.8358 best, s0 0.7649, s2 0.7926, all parse_fail 0) + 3-seed median ensemble 0.8160"
  - "output/base21/judge03_rho.json -- final JUDGE-03 receipt: vLLM-served s1 rho 0.7872 CI [0.7125, 0.8405] at 8192 cap + CI-aware verdict overall_pass=false, disposition=valid_recorded_miss, re-open condition recorded (condition_met=false)"
  - "models/Qwen3.6-35B-A3B-judge-v4-s1-merged -- merged promoted judge seed (66.99 GiB, 240/240 modules, local)"
  - "scripts/capture_judge_responses_tinker.py --base-model/--renderer flags (v4-aware, back-compat v3 default)"
  - "scripts/build_judge03_capture_rho.py + scripts/build_judge03_merge_serve.py -- reusable capture-rho and merge+serve+score drivers"
affects: [phase-21-verification, v4.0-milestone-disposition, phase-27-packaging-ensemble]

tech-stack:
  added: []
  patterns:
    - "vLLM MAX_MODEL_LEN must exceed longest-prompt + generation cap: an 8192-token completion cap on an 8192-context serve silently re-truncates long prompts -- the exact Pitfall-4 failure moved from capture side to serve side; this serve used MAX_MODEL_LEN=16384 (longest wp_judge val prompt 2288 tokens)"
    - "eval_relabel.py writes its summary to a SHARED eval_summary.json next to the capture -- multi-capture callers must sidecar-copy each summary before the next scoring run overwrites it"

key-files:
  created:
    - scripts/build_judge03_capture_rho.py
    - scripts/build_judge03_merge_serve.py
    - output/base21/judge03_capture_rho.json
    - output/base21/judge03_rho.json
    - output/base21/judge_capture_s1.jsonl
    - output/base21/judge_capture_s0.jsonl
    - output/base21/judge_capture_s2.jsonl
    - output/base21/judge_capture_vllm_s1.jsonl
    - output/base21/judge03_capture_ensemble.json
  modified:
    - scripts/capture_judge_responses_tinker.py

key-decisions:
  - "JUDGE-03 disposition: VALID RECORDED MISS (not forced). vLLM-served s1 CI-lower 0.7125 < 0.85; cheap-path ensemble CI-lower 0.7563 < 0.87. Neither pre-registered target cleared CI-aware."
  - "single_seed_pass judged on the vLLM-served figure (the pre-registered criteria's own methodology: 'measured the same way as the v3.0 shipping figure'); ensemble_pass judged on the cheap-path ensemble (the only ensemble measurable in Phase 21 -- the vLLM-served 3-merged ensemble is deferred to packaging). Each figure methodology-labeled in the receipt (T-21-17), never conflated."
  - "Discretion-item-2 re-open condition recorded verbatim with condition_met=false: (a) saturated-below-target is met, but (b) a gap-closure diagnostic (capacity/loss-shape/data-cleaning) has NOT been run on Qwen3.6-35B-A3B -- both are required, so no relabel-campaign re-open is warranted from this plan"
  - "Served at MAX_MODEL_LEN=16384 (not the script default 8192) so the literal 8192-token generation cap physically fits above the longest 2288-token wp_judge prompt -- otherwise the serve would re-introduce the exact truncation Pitfall 4 exists to prevent"

metrics:
  duration: ~120min (Task 1 ~60min remote Tinker captures + scoring; Task 2 ~50min: 2 GiB download + 67 GiB merge + 3 vLLM serve cycles + 121-prompt served capture; Task 3 ~1min)
  completed: 2026-07-14

requirements-completed: [JUDGE-03]

status: complete
---

# Phase 21 Plan 06: JUDGE-03 Judge-Rho Measurement (Both Methodologies) Summary

**Judge rho measured on both pre-registered methodologies for the 3 relabel-SFT seeds on Qwen3.6-35B-A3B: cheap Tinker-capture path (best single s1 0.8358, 3-seed median ensemble 0.8160, zero parse failures at 8192 cap) and the literal vLLM-served path (merged s1, 240/240 modules, rho 0.7872 CI [0.7125, 0.8405]). Neither the >0.85 single nor >0.87 ensemble target cleared CI-aware — recorded as a VALID MISS with the discretion-item-2 re-open condition explicitly unmet (no gap-closure diagnostic run on this base yet). All 3 seed checkpoints preserved for the deferred packaging ensemble.**

## Performance

- **Duration:** ~120 min wall (3 sequential remote Tinker captures ~20 min each; 67 GiB merge; 3 vLLM serve cycles ~6.5 min boot each; sole GB10 user throughout, all serves torn down in finally blocks)
- **Tasks:** 3/3
- **Cost:** remote Tinker sampling only (3 x 121 prompts, no training spend)

## The headline numbers

| Measurement | rho | CI lower | Methodology | Target | Verdict |
|---|---|---|---|---|---|
| s1 single (served) | **0.7872** | 0.7125 | vLLM-served, 8192 cap | >0.85 | MISS |
| 3-seed median ensemble | **0.8160** | 0.7563 | Tinker-capture, 8192 cap | >0.87 | MISS |
| s1 single (capture) | 0.8358 | 0.7740 | Tinker-capture (promotion path, non-gating) | — | reference |
| s0 / s2 (capture) | 0.7649 / 0.7926 | — | Tinker-capture | — | reference |
| v3.0 shipping wall | 0.8075 ens / 0.8017 single | — | vLLM-served (old base) | — | the bar to beat |
| Ceiling | 0.984 | — | label attenuation | — | — |

Context the receipt records:

- **Format training landed decisively:** raw new base parse-fails 30/30 wp_judge prompts (JUDGE-01); all trained seeds parse 121/121 on BOTH the Tinker-capture and vLLM-served paths at 8192 cap.
- **Capture→served attenuation reproduces the old base's pattern:** s1 0.8358 capture → 0.7872 served (old base: 0.8274 capture → 0.8017 served single). The two methodologies are kept distinct in the receipt (T-21-17); promotion used capture, the gate used served.
- **The new base did not move the judge-rho wall.** The served single (0.7872) sits below the v3.0 served single (0.8017); the cheap ensemble (0.8160) is comparable to v3.0's 0.8075. Combined with 21-05's codegen regression, the v1.2/v1.3 recipe does not transfer its gains to Qwen3.6-35B-A3B unchanged.

## Accomplishments

### Task 1 — Cheap Tinker-capture rho per seed + ensemble median (commit `e3e7155`)

- `capture_judge_responses_tinker.py` gained optional `--base-model`/`--renderer` flags — smallest diff, defaults preserve the v3 import path byte-for-byte; passing the v4 base imports `tinker_reasoning_data_v4`'s `RENDERER_NAME` (`qwen3_5_disable_thinking`). Index filter (`wp_judge_startswith`) and file-order index contract untouched (T-21-16).
- All 3 seeds captured at `--max-tokens 8192` (Pitfall 4: never the 1024 default), temperature 0.0, against each seed's manifest-resolved promoted ep3 sampler.
- Scored per seed with the UNMODIFIED `eval_relabel.py`; ensemble median with the UNMODIFIED `eval_relabel_ensemble.py` (already implements exactly the per-item median + same 2000-resample bootstrap).
- Receipt `judge03_capture_rho.json`: per_seed (all n=121, parse_fail=0), best_single_seed (s1), ensemble_median, max_tokens=8192, method=tinker_capture.

### Task 2 — Merge promoted s1 + vLLM-serve at 8192 cap + literal rho (commit `c923c62`)

- `build_judge03_merge_serve.py` mirrors 21-05's proven `build_gen03_merge.py` pattern: resumable curl download of the 2 GiB s1 archive (WR-10 member-validated extraction), `merge_adapter.py --config-path config/train_config_v4.yaml` fused-expert merge — **240/240 module guard passed**, real-generation base-vs-merged diff **differs** (both asserted before any scoring, T-21-18).
- Served via `serve_base20_vllm.sh` (LANGUAGE_MODEL_ONLY, MAX_MODEL_LEN=16384) with the Phase 15 LOCKED real-generation warm-up gate; captured all 121 wp_judge val prompts at max_tokens=8192 via the existing `sieve_capture_judge_http.capture()` (RC-A `enable_thinking=False` guard, same index contract); scored with unmodified `eval_relabel.py`.
- **vLLM-served s1: rho 0.7872, CI [0.7125, 0.8405], n=121, parse_fail 0.**
- Serve stopped in finally; teardown verified (no containers, port 8025 free, GPU idle). Sole 67 GiB residency in Wave 4.
- All 3 seed manifests verified still on disk; `ensemble_vllm_served: "deferred_to_packaging"` recorded.

### Task 3 — CI-aware verdict + failure disposition (commit `9673294`)

- Targets recorded `{single: 0.85, ensemble: 0.87}`; CI LOWER bound must clear the bar (D-V4-10 hardening), same bootstrap on both paths.
- `single_seed_pass=false` (served CI-lower 0.7125), `ensemble_pass=false` (cheap-path ensemble CI-lower 0.7563), `overall_pass=false`.
- `disposition="valid_recorded_miss"`; discretion-item-2 re-open condition recorded VERBATIM with `condition_met=false` — (a) saturated-below-target holds, (b) gap-closure diagnostic on THIS base has not been run; both required, so no re-open triggered (recording only, per plan).
- Framed vs `{v30_ensemble: 0.8075, v30_single: 0.8017, ceiling: 0.984}`; every figure labeled with its source methodology.

## Task Commits

1. **Task 1: cheap-path capture rho** — `e3e7155` — `feat(21-06): JUDGE-03 cheap-path Tinker-capture rho -- 3 seeds at 8192 cap + ensemble median`
2. **Task 2: merge + serve + literal rho** — `c923c62` — `feat(21-06): JUDGE-03 literal path -- s1 merged (240/240) + vLLM-served rho 0.7872 at 8192 cap`
3. **Task 3: CI-aware verdict** — `9673294` — `feat(21-06): JUDGE-03 CI-aware verdict -- valid recorded miss vs pre-registered 0.85/0.87 targets`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Raised the serve's MAX_MODEL_LEN to 16384 for the judge capture**
- **Found during:** Task 2 (design, before serving)
- **Issue:** `serve_base20_vllm.sh` defaults `MAX_MODEL_LEN=8192`. The plan's literal "8192-token cap" applies to the GENERATION budget; with an 8192-token context the longest wp_judge val prompt (measured: 2288 tokens) would leave only ~5.9K completion tokens — silently re-introducing the exact truncation-as-quality-regression failure Pitfall 4 / T-21-15 exists to prevent, just moved from the capture side to the serve side.
- **Fix:** passed `MAX_MODEL_LEN=16384` (env toggle already existed in the serve script — no script change needed) so prompt + 8192-token completion always fit; recorded as `max_model_len_served` in the receipt.
- **Files modified:** none (env-only)
- **Commit:** `c923c62`

### Non-deviation notes

- Two new orchestrator scripts (`build_judge03_capture_rho.py`, `build_judge03_merge_serve.py`) are not in the plan's `files_modified` list but follow the established 21-05 convention (`build_gen03_merge.py`/`build_gen03_wpbench.py`) — thin drivers around existing, unmodified scorers and helpers.
- `eval_relabel.py` and `eval_relabel_ensemble.py` needed zero changes, exactly as the plan predicted.

## Threat Model Compliance

- **T-21-15 (1024-default truncation):** every capture ran at `--max-tokens 8192`, recorded in both receipts; parse_fail tracked per seed (0 everywhere — no truncation signal).
- **T-21-16 (index misalignment):** the `wp_judge_startswith` filter + file-order index contract untouched by the v4 flag diff; served capture reused the existing `sieve_capture_judge_http` implementation of the same contract.
- **T-21-17 (methodology conflation):** every rho in `judge03_rho.json` carries a `methodology` label; promotion=tinker_capture, literal gate=vllm_served, explicitly annotated as non-interchangeable.
- **T-21-18 (silent bad merge):** 240/240 module guard + base-vs-merged real-generation diff asserted BEFORE the served capture; script refuses to score an unverified merge.
- **T-21-19 (GB10 collision):** sole GB10 user; merge (CPU) fully exited before each serve; every serve stopped in a finally block; teardown verified clean after completion.

## Known Stubs

None — all measurements are real (363 remote Tinker generations, real 2 GiB archive download, real 67 GiB merge, 3 real vLLM serve cycles, 121 served generations). No mocked calls, no fabricated receipts.

## Next Phase Readiness

- **JUDGE-03: SATISFIED as a VALID RECORDED MISS** (the plan's success criterion is "a pass or a valid recorded miss") — judge rho measured on both methodologies, compared CI-aware, disposition committed.
- **Milestone implication (with 21-05's GEN-03 miss):** the v1.2/v1.3 recipe transfers format compliance (0/121 parse fails vs 30/30 raw) but neither the codegen floor nor the judge-rho targets clear on Qwen3.6-35B-A3B. The re-open condition for the relabel campaign is NOT met until a gap-closure diagnostic (capacity/loss-shape/data-cleaning, mirroring `output/relabel/gap_closure_summary.json`) is run on THIS base and rules out recipe causes — that diagnostic is the natural next investigation if the milestone pursues the judge targets.
- **Preserved for Phase 27 packaging:** all 3 seed manifests + Tinker checkpoints (9 sampler checkpoints), the merged s1 model on disk, and the per-seed captures — the vLLM-served 3-merged ensemble measurement needs only these.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-14*

## Self-Check: PASSED

All 9 created files + merged model dir verified present on disk; all 3 task commit hashes (`e3e7155`, `c923c62`, `9673294`) verified present in `git log`.
