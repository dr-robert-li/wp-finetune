---
phase: 07-router-profiling-protected-expert-set
plan: 02
subsystem: profiling
tags: [moe, router, eeff, jaccard, bootstrap-ci, protected-mask, qwen3-moe, dgx]

requires:
  - phase: 07-router-profiling-protected-expert-set (plan 01)
    provides: profile_merged_model.py, compute_concentration.py, extract_protected_mask.py, run-profiling skill (test-certified)
provides:
  - Per-layer routing report (routing_report.jsonl, 48 layers, ratio 30_70)
  - PROF-03 Jaccard CI gate result (jaccard_ci_lower=0.9426 >= 0.94, PASS)
  - PROF-04 concentration report (E_eff/CV/coverage/depth-skew + D-08 delta vs base, 48 rows)
  - D-03 protected-expert mask [48,128] bool + sidecar + D-04 sensitivity table
  - PROF-05/GATE-01 N/A rationale
affects: [phase-08-reward, phase-11-moe-sieve, phase-13-aimer-reap-pruning]

tech-stack:
  added: []
  patterns:
    - "Headless ngc-pytorch container run (no -it) replicating the interactive launcher's mounts"
    - "CI-aware gate disposition: bootstrap lower bound clears the bar, not the point estimate (D-09)"

key-files:
  created:
    - output/profiling/reasoning-merged-v4/routing_report.jsonl
    - output/profiling/reasoning-merged-v4/jaccard_stability.json
    - output/profiling/reasoning-merged-v4/concentration_report.json
    - output/profiling/reasoning-merged-v4/protected_expert_mask.npy
    - output/profiling/reasoning-merged-v4/protected_expert_mask.json
    - output/profiling/reasoning-merged-v4/sensitivity_table.json
    - output/profiling/reasoning-merged-v4/protected_mask_result.json
    - output/profiling/reasoning-merged-v4/routing_report_rationale.md
    - .planning/phases/07-router-profiling-protected-expert-set/07-HUMAN-REVIEW.md
  modified: []

key-decisions:
  - "PROF-03 CI gate PASSES at jaccard_ci_lower=0.9426; point gate would FAIL (6/48 layers <0.94, L35=0.60) — CI-aware disposition (D-09) carries it"
  - "D-08 E_eff delta non-empty (48 matched rows); late-layer broadening L45-47 ~+7 is the largest structured effect"
  - "PROF-05/GATE-01 documented N/A (single survivor 30/70), no fabricated matrix"
  - "Protected mask: 1480 experts (mean 30.8/layer) under conservative co-activation"

patterns-established:
  - "Multi-hour GPU run driven headless from host with file-based liveness + milestone monitoring; .complete idempotency markers per step"

requirements-completed: [PROF-05, GATE-01]

duration: ~6h40m (incl. 6h30m GPU)
completed: 2026-06-15
---

# Phase 7 Plan 02: DGX Profiling Run Summary

**Profiled the promoted v1.2 30B MoE on the matched 30/70 training stimulus — all automated gates green (Jaccard CI 0.9426 ≥ 0.94, D-08 delta 48 rows, mask [48,128]); human sign-off APPROVED 2026-06-19 (council-reviewed). Phase 7 CLOSED.**

## Status

- ✅ **Task 1** — Profiling run + output validation. All artifacts written; Task-1 verify one-liner exits 0 (`OK mask (48, 128) rows 48 jaccard_ci_lower 0.9426`). Baseline untouched.
- ✅ **Task 2** — PROF-05 + GATE-01 N/A rationale written; Task-2 verify one-liner prints `rationale OK`.
- ✅ **Task 3 (gate closed)** — Human sign-off on E_eff comparison + protected expert set. **APPROVED 2026-06-19** (Dr. Robert Li; SOTA council GPT-5.5 / Opus 4.8 / Gemini 3.1 Pro unanimous on both judgment items — L35 Jaccard 0.60 ACCEPT under D-09, L45–47 E_eff +7 ACCEPT as routing-shift). Disposition recorded in `07-HUMAN-REVIEW.md` §5. Phase 7 closed; Phases 8 + 11 unblocked.

## Run

- Model: `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (Qwen3-MoE 48×128, top-8).
- Stimulus: `data/final_dataset/ratio_30_70/openai_train.jsonl` (34,855 ex full reference + 10% subsample).
- Headless `ngc-pytorch` container, GB10, 6h 30m, rc=0. 785.8M tokens profiled (117.4M gen / 663.4M judge).

## Results (detail in 07-HUMAN-REVIEW.md)

- **PROF-03:** `jaccard_ci_lower=0.9426 ≥ 0.94` PASS (point mean 0.9685, min 0.60 @ L35; 6 layers sub-threshold — carried by CI).
- **PROF-04:** E_eff total 72.58 / gen 60.69 / judge 72.65 (gen more concentrated than judge); CV 1.20; depth-skew 1.107.
- **D-08:** 48 matched delta rows, mean +2.75 (range −2.57…+7.31); late-layer broadening L45/46/47 ≈ +7.
- **D-03/D-04:** 1480 protected experts (25–40/layer); sensitivity mean=1480 / median=2477 / top16=595.

## Deviations

- **Container launcher is interactive-only.** `deps/dgx-toolbox/containers/ngc-pytorch.sh` hardcodes `-it` + `exec bash` and ignores command args, so the skill's documented headless invocation does not work as written. Drove the run via a direct `docker run` (no `-it`) replicating the launcher's mounts + `install-deps.py`, substituting the profiling command for `exec bash`. User-approved approach.
- **Forward pass is silent.** No `logging.basicConfig` in the profiler → `logger.info` progress (hook count, pass transitions) does not emit. Liveness tracked via GPU util + artifact appearance instead. Non-blocking; outputs validate correctness.

## Forward obligations

- Phases 11 (MoE-Sieve) / 13 (AIMER/REAP) consume `protected_expert_mask.npy` as the must-not-prune set. Mask is immutable once signed off.
- If the reviewer judges the 0.3pt Jaccard CI margin too thin, D-06 fallback = re-profile with `--subsample 0.25`.
