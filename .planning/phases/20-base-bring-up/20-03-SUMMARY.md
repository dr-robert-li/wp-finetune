---
phase: 20-base-bring-up
plan: 03
subsystem: infra
tags: [vllm, deltanet, cuda-graph-capture, qwen3.6, gb10, serving-smoke]

# Dependency graph
requires:
  - phase: 20-base-bring-up
    provides: "models/Qwen3.6-35B-A3B/ downloaded + load-verified (20-01); config.json eos/pad aligned in place (20-02)"
provides:
  - "recipes/qwen3.6-35b-a3b-vllm.yaml — bf16 vLLM recipe for the LOCAL checkpoint (gpu_memory_utilization 0.80, no fp8)"
  - "scripts/serve_base20_vllm.sh — v4 serve script with LANGUAGE_MODEL_ONLY + ENFORCE_EAGER env toggles, 0.80 mem-util default, no --served-model-name hardcode"
  - "scripts/_p0_vllm_smoke_serve.py boot_vllm(serve_script=, extra_env=) — backward-compatible params so callers can select any serve script and pass env toggles"
  - "scripts/smoke_deltanet_base20.py — BASE-03 DeltaNet serving smoke (version assert + capture-enabled boot + warm-up gate + eager fallback path)"
  - "output/base20/deltanet_smoke.json — BASE-03 gate receipt (status=pass, cuda_graph_capture=enabled, vllm 0.20.2rc1, use_kernels=false)"
affects: [20-04-vl-merge, 21-sft]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Serve-script env toggles (LANGUAGE_MODEL_ONLY/ENFORCE_EAGER) default UNSET so the first serve attempt always exercises CUDA-graph capture — eager-only smokes are false passes for capture-phase crashes (vLLM #35945)"
    - "Container-resolved tool versions are recorded facts in gate receipts, never assumptions from a :latest tag — resolve via docker run before any dependent test"

key-files:
  created:
    - recipes/qwen3.6-35b-a3b-vllm.yaml
    - scripts/serve_base20_vllm.sh
    - scripts/smoke_deltanet_base20.py
    - output/base20/deltanet_smoke.json
  modified:
    - scripts/_p0_vllm_smoke_serve.py

key-decisions:
  - "use_kernels=False locked for this phase: the Atlas-Inference/gdn community Hub kernel (SUS, non-allowlisted, needs trust_remote_code) declined for a 1.38x prefill-only speedup; flipping to True later requires a checkpoint:human-verify (gate=blocking-human) per T-20-03a — decision + rationale recorded in the receipt"
  - "CUDA-graph capture ENABLED path succeeded on the first attempt on GB10/aarch64 (vLLM #35945 did NOT reproduce on vllm 0.20.2rc1.dev196) — --enforce-eager fallback path exists in the smoke script but was not exercised, fallback_used=false"
  - "Container vLLM version 0.20.2rc1.dev196+g84f7a5534 resolved as the FIRST smoke action and asserted >=0.19.0 — Assumption A3 discharged as a recorded fact"

requirements-completed: [BASE-03]

coverage:
  - id: D1
    description: "bf16 recipe exists (gpu_memory_utilization 0.80, no fp8 quantization), serve_base20_vllm.sh passes bash -n with LANGUAGE_MODEL_ONLY/ENFORCE_EAGER gating and no wp-30_70/SIEVE carryover, boot_vllm gains serve_script/extra_env without breaking any of the 7 existing callers (all call positionally through gpu_mem_util only)"
    requirement: "BASE-03"
    verification:
      - kind: other
        ref: "bash -n scripts/serve_base20_vllm.sh && python -c \"... assert r['defaults']['gpu_memory_utilization']==0.80 ... assert 'serve_script' in p and 'extra_env' in p\" (exits 0)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The container's resolved vLLM version (0.20.2rc1.dev196+g84f7a5534.d20260510) was logged as the FIRST smoke action and asserted >=0.19.0; docker/nvidia-smi liveness (A1) checked before any DeltaNet test"
    requirement: "BASE-03"
    verification:
      - kind: other
        ref: "output/base20/deltanet_smoke.json vllm_version field; smoke log lines '[A1] docker + nvidia-smi liveness check OK' + '[A3] container vLLM version: 0.20.2rc1...'"
        status: pass
    human_judgment: false
  - id: D3
    description: "Qwen3.6-35B-A3B served text-only (--language-model-only) via vLLM on GB10 WITH CUDA-graph capture enabled (no --enforce-eager on first attempt); healthy after 385s; a real warm-up generation returned non-empty output — DeltaNet layers execute on aarch64 under graph capture"
    requirement: "BASE-03"
    verification:
      - kind: other
        ref: "output/base20/deltanet_smoke.json: cuda_graph_capture=enabled, fallback_used=false, warm_gen_ok=true, warm_gen_sample non-empty"
        status: pass
    human_judgment: false
  - id: D4
    description: "use_kernels=False decision recorded with non-empty rationale in the receipt (community kernel declined, blocking-human checkpoint required to flip); serving processes killed cleanly, no orphan containers/ports"
    requirement: "BASE-03"
    verification:
      - kind: other
        ref: "python -c \"import json; d=json.load(open('output/base20/deltanet_smoke.json')); assert d['use_kernels'] is False and d['use_kernels_rationale']\" (exits 0); docker ps -a shows no base20/vllm containers; port 8020 free"
        status: pass
    human_judgment: false

duration: 16min
completed: 2026-07-13
status: complete
---

# Phase 20 Plan 03: v4 Base Bring-Up — DeltaNet vLLM Serving Smoke Summary

**Qwen3.6-35B-A3B's Gated-DeltaNet layers serve on aarch64/GB10 via vLLM 0.20.2rc1 WITH CUDA-graph capture enabled on the first attempt (vLLM #35945 did not reproduce; --enforce-eager fallback built but unused), a real generation returned non-empty output, and the use_kernels=False decision (community Atlas-Inference/gdn kernel declined) is recorded in output/base20/deltanet_smoke.json — the reusable v4 serving harness (bf16 recipe + serve script + boot_vllm extension) is ready for plan 20-04's merge round-trip.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-07-13T02:06:00Z
- **Completed:** 2026-07-13T02:21:57Z
- **Tasks:** 2
- **Files modified:** 5 (4 created, 1 modified)

## Accomplishments
- `recipes/qwen3.6-35b-a3b-vllm.yaml` (new): bf16 sibling of the FP8 recipe — points at the LOCAL `models/Qwen3.6-35B-A3B` checkpoint, `gpu_memory_utilization: 0.80` (Pitfall 2 GB10-stable value, not the FP8 recipe's 0.55 or vLLM's 0.90 default), `trust_remote_code: true`, no `--quantization fp8`
- `scripts/serve_base20_vllm.sh` (new): modeled on `serve_30_70_vllm.sh` (same no-`--rm`/logs-retrievable-on-crash behavior) with `LANGUAGE_MODEL_ONLY=1` → `--language-model-only` and `ENFORCE_EAGER=1` → `--enforce-eager` env gating, both default UNSET so the first attempt runs with CUDA-graph capture; no hardcoded `--served-model-name`, no SIEVE mask block
- `boot_vllm()` in `scripts/_p0_vllm_smoke_serve.py` extended with `serve_script` (default: existing constant) and `extra_env` (default None) params — verified additive: all 7 existing callers pass positionally through `gpu_mem_util` only
- `scripts/smoke_deltanet_base20.py` (new) ran end-to-end: A1 liveness (docker + nvidia-smi) → vLLM version resolve + `>=0.19.0` assert (FIRST action) → capture-enabled boot (`LANGUAGE_MODEL_ONLY=1`, 0.80 mem-util, 1200s timeout for the 67 GiB base) → healthy at 385s → real-generation warm-up gate (non-empty) → receipt → `stop_vllm` in finally
- BASE-03 receipt: `status=pass`, `cuda_graph_capture=enabled`, `fallback_used=false`, `vllm_version=0.20.2rc1.dev196+g84f7a5534.d20260510`, `use_kernels=false` + rationale

## Task Commits

Each task was committed atomically:

1. **Task 1: bf16 recipe + serve_base20_vllm.sh + boot_vllm serve_script/extra_env params** - `8ae5c2a` (feat)
2. **Task 2: BASE-03 DeltaNet serving smoke with CUDA-graph capture + receipt** - `c336fd7` (feat)

**Plan metadata:** pending (docs: complete plan, this commit)

## Files Created/Modified
- `recipes/qwen3.6-35b-a3b-vllm.yaml` - bf16 vLLM recipe for the local v4 base checkpoint
- `scripts/serve_base20_vllm.sh` - v4 serve script (LANGUAGE_MODEL_ONLY/ENFORCE_EAGER toggles, 0.80 mem-util)
- `scripts/_p0_vllm_smoke_serve.py` - `boot_vllm()` +serve_script/+extra_env (backward compatible)
- `scripts/smoke_deltanet_base20.py` - BASE-03 DeltaNet serving smoke script
- `output/base20/deltanet_smoke.json` - BASE-03 gate receipt (force-added per the 20-01 `output/base20/*.json` gate-receipt precedent)

## Decisions Made
- **use_kernels=False** (default, per 20-RESEARCH.md Alternatives Considered): the `Atlas-Inference/gdn` community Hub kernel requires `trust_remote_code=True` against a non-allowlisted repo (SUS verdict) for only a 1.38x prefill speedup (decode flat, memory-bandwidth bound). `kernels` was not installed; the Hub kernel was never loaded. Flipping to `True` in any later phase requires a `checkpoint:human-verify` (`gate="blocking-human"`) per threat-model entry T-20-03a. Decision + rationale recorded in the receipt.
- **CUDA-graph capture path is the recorded pass** — vLLM #35945 (`AssertionError` in `causal_conv1d_update` during capture) did not reproduce on this container's vLLM 0.20.2rc1.dev196 on GB10/aarch64. The one-shot `--enforce-eager` retry path exists in the smoke script (exercisable if a future container regresses) but `fallback_used=false`.
- **Assumption A3 discharged as a recorded fact:** the `:latest` nightly container resolves to vLLM `0.20.2rc1.dev196+g84f7a5534.d20260510`, comfortably above the `>=0.19.0` vendor-recommended floor — logged as the FIRST smoke action, not assumed from the tag.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. Boot was clean (0 errors in docker logs), healthy in 385s — well under both the 1200s smoke timeout and the ~9 min historical GB10 boot anchor.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The v4 serving harness (bf16 recipe + `serve_base20_vllm.sh` + `boot_vllm(serve_script=, extra_env=)`) is exactly what plan 20-04's merge round-trip consumes — no further serving-infra work needed.
- BASE-03 is satisfied: the architecture-specific serving risk (DeltaNet under CUDA-graph capture on aarch64) is resolved before any SFT/eval serves this base.
- All serving processes/containers killed and verified clean (no orphan containers, port 8020 free).
- No blockers.

---
*Phase: 20-base-bring-up*
*Completed: 2026-07-13*

## Self-Check: PASSED

All 4 created files verified present on disk; both task commit hashes (8ae5c2a, c336fd7) verified present in git log; automated verify commands for both tasks exit 0.
