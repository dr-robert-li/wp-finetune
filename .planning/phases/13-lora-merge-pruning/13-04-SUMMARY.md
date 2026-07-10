---
phase: 13-lora-merge-pruning
plan: 04
subsystem: infra
tags: [moe, pruning, aimer, vllm, gate-before-remove, wp-bench, qwen3-30b-a3b]

requires:
  - phase: 13-lora-merge-pruning
    plan: 01
    provides: output/prune/aimer_scores_{gen,judge}.npy — the [48,128] AIMER score arrays both gate masks were built from
  - phase: 13-lora-merge-pruning
    plan: 02
    provides: scripts/prune_gated_eval.py gate-before-remove driver (real --axis gen/judge GPU paths executed here for the first time)
  - phase: 11-compression-packaging
    provides: serve_30_70_vllm.sh + SIEVE_MASK_NPY sitecustomize router patch, grader-reset fix 8c4b167, judge max_tokens 2048 fix cd36a5e, kfull judge captures (reused as D2_security baseline)
provides:
  - output/prune/masks/aimer_gen_k96.npy, output/prune/masks/aimer_judge_k96.npy — the AIMER@25% keep-masks (gating only, no weights removed)
  - output/prune/gated/aimer_25_gen.json — gen gate record wp_bench 0.1577 FAIL vs 0.4284
  - output/prune/gated/aimer_25_judge.json — judge gate record ens rho 0.165 / parse 0.446 FAIL vs 0.7555/0.95, all 3 per-seed rho
  - output/prune/gated/aimer_25_d2.json — d2_security_retention/baseline record (13-03 prune_selection forward dep), annotated unreliable under parse collapse
affects: [13-05-expansion-conditional, 13-06-selection]

tech-stack:
  added: []
  patterns:
    - "Gate records carry a top-level 'pass' key plus per-bar pass_* booleans; numpy scalars serialized via json default=lambda o: o.item()"
    - "vLLM boot failures now leave diagnosable logs: serve script no longer --rm's the container, wait_healthy dumps docker logs to logs/vllm_boot_failures/ before raising"

key-files:
  created:
    - output/prune/masks/aimer_gen_k96.npy
    - output/prune/masks/aimer_judge_k96.npy
    - output/prune/gated/aimer_25_gen.json
    - output/prune/gated/aimer_25_judge.json
    - output/prune/gated/aimer_25_d2.json
    - scripts/prune_run_13_04.sh
  modified:
    - scripts/prune_gated_eval.py
    - scripts/_p0_vllm_smoke_serve.py
    - scripts/serve_30_70_vllm.sh

key-decisions:
  - "AIMER@25% is a decisive MEASURED FAIL on both axes — the phase's pivot experiment answers NO: weight-norm ranking finds no prunable subset routing-coldness missed"
  - "Judge record rescored from the on-disk seed captures after the post-GPU serializer crash — zero re-serving, all 3 seeds' GPU work preserved"
  - "D2_security baseline taken from Phase 11's existing kfull judge captures (no new GPU spend); masked retention value annotated UNRELIABLE because parse collapse corrupts the score scale"

patterns-established:
  - "Any long GPU eval record write must serialize numpy scalars (default=.item()) — a TypeError after hours of serving is the most expensive possible crash site"

requirements-completed: [PRUNE-03]

coverage:
  - id: D1
    description: "AIMER@25% gen-axis gate-evaluated via gating mask (no weights removed): mask built from AIMER scores, protected sha re-verified pre-serve, grader containers reset, full wp-bench 344-test suite run against masked vLLM serve; result recorded with pass/fail vs the vLLM-measured 0.4284 bar"
    requirement: "PRUNE-03"
    verification:
      - kind: other
        ref: "output/prune/gated/aimer_25_gen.json — wp_bench 0.1577, protected_retained true, pass false (plan's automated verify one-liner passes)"
        status: pass
    human_judgment: false
  - id: D2
    description: "AIMER@25% judge-axis gate-evaluated on ALL 3 seeds sequentially (shared mask, HARD CONSTRAINT 6): 121/121 captured per seed at max_tokens 2048; ensemble rho, per-seed rho, s1 fallback rho, parse-rate recorded with pass/fail vs 0.7555/0.95 bars"
    requirement: "PRUNE-03"
    verification:
      - kind: other
        ref: "output/prune/gated/aimer_25_judge.json — ens rho 0.165, per_seed_rho len 3, parse 0.446, protected_retained true, pass false (plan's automated verify one-liner passes)"
        status: pass
    human_judgment: false

duration: ~5h (gen arm ~1.5h wp-bench + 3 judge seeds ~2h serve/capture + one transient boot failure retry)
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 04: AIMER@25% Gate-Before-Remove Eval Summary

**AIMER@25% (keep 96/128 experts/layer) is a decisive measured FAIL on both models — gen wp-bench collapses to 0.1577 (bar 0.4284) and the judge parse-collapses to 0.446 (bar 0.95) with ensemble rho 0.165 (bar 0.7555) — answering the phase's live question: weight-norm expert ranking finds no prunable subset that routing-coldness could not.**

## AIMER@25 Results Table

| Axis | Metric | Measured | Bar | Pass |
|------|--------|----------|-----|------|
| gen | wp_bench overall | **0.1577** | >= 0.4284 | **FAIL** |
| gen | wp_bench knowledge | 0.3125 | — | (detail) |
| gen | wp_bench correctness | 0.0417 | — | (detail) |
| judge | ensemble rho (3-seed median) | **0.1651** | >= 0.7555 | **FAIL** |
| judge | parse rate (121-item val) | **0.4463** | >= 0.95 | **FAIL** |
| judge | s1 rho (fallback) | 0.3048 | >= 0.7497 | **FAIL** |
| judge | per-seed rho | s0 -0.234 / s1 0.305 / s2 0.444 | — | (detail) |
| both | protected_retained | true | must be true | PASS |

Per-seed parse counts: s0 26/121, s1 34/121, s2 5/121 — the parse collapse is the judge's known failure mode (13-RESEARCH Pitfall 2, judge worst-layer E_eff ~99 > 96 kept), reproduced exactly.

## What this means for 13-05 / 13-06

- **13-05 short-circuits.** 25% was the only genuinely-open region (96 kept > E_eff ~90); it failed catastrophically, not marginally (gen -27pp below bar, judge rho -59pp below bar). 50%/75% are strictly more aggressive and Phase 11 already mapped that collapse territory. REAP's domain comparison is moot — nothing is prunable to compare.
- **13-06 selection** will see one variant, ineligible on 3 independent gates (gen bar, judge rho bar, parse bar) → `no_winner` verdict territory; the phase ships unpruned, consistent with Phase 11's optimal_k=full sign-off.
- Interpretation for the record: gen collapse at 25% means weight-norm (AIMER) ranking fares no better than routing-cold ranking on this workload — consistent with Phase 11's finding that the model's routing is too distributed (E_eff ~88-99/128) for any expert-subset compression.

## Performance

- **Duration:** ~5h wall-clock (one transient vLLM boot failure + retry; gen wp-bench ~1.5h; 3 judge seeds ~2h sequential serve/capture; rescore from captures after serializer crash — no re-serving)
- **Completed:** 2026-07-10
- **Tasks:** 2/2
- **Files modified:** 9 (5 result/mask artifacts, 3 scripts fixed, 1 driver script)

## Task Commits

1. **Infra fixes pre-run (boot-log capture, D2 record, mask path, infra-fail exit code)** - `a2e5aea` (fix)
2. **Tasks 1+2: AIMER@25 gen + judge gate results + post-GPU serializer fix** - `32ce674` (feat)

## Files Created/Modified
- `output/prune/masks/aimer_gen_k96.npy`, `output/prune/masks/aimer_judge_k96.npy` - [48,128] bool keep-masks, per-layer keep 99-112 (top-96-by-AIMER-score UNION protected), protected subset verified
- `output/prune/gated/aimer_25_gen.json` - gen gate record (wp_bench, detail scores, pass/fail vs 0.4284, protected_retained)
- `output/prune/gated/aimer_25_judge.json` - judge gate record (ensemble rho, all 3 per-seed rho, s1 fallback, parse rate, pass/fail vs 0.7555/0.95)
- `output/prune/gated/aimer_25_d2.json` - d2_security_retention/baseline for prune_selection.load_variant_records (13-03 forward dependency), annotated unreliable
- `scripts/prune_gated_eval.py` - real gate paths executed; fixes: numpy-safe JSON writer, per_seed_rho (all 3 seeds), top-level `pass` key, D2 record emission, masks to output/prune/masks/, exit 1 on infra failure
- `scripts/serve_30_70_vllm.sh` - dropped `--rm` so crashed boots leave retrievable docker logs
- `scripts/_p0_vllm_smoke_serve.py` - wait_healthy dumps docker logs to logs/vllm_boot_failures/ before raising VllmBootTimeout
- `scripts/prune_run_13_04.sh` - chained background driver (gen arm → measured-guard → judge arm)

## Decisions Made
- D2_security baseline sourced from Phase 11's existing kfull judge captures (`output/sieve/ksweep/judge_kfull/s{0,1,2}/`) — same val set, same max_tokens, zero additional GPU spend
- Judge record rescored from the on-disk captures after the serializer crash rather than re-serving (all GPU work had completed; the crash was purely at the JSON write)
- The d2 retention value (25.1 vs baseline 6.98 on a 0-10 scale) is annotated UNRELIABLE in the record: under parse collapse the few parsed D2 values ride corrupted scales (up to 80); the variant is already ineligible on three other gates, so the annotation prevents 13-06 misreading retention>baseline as a real D2 pass

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Transient vLLM boot failure destroyed its own logs**
- **Found during:** Task 1 (first gen arm attempt)
- **Issue:** container `prune-gated-gen-vllm` exited during boot; `--rm` auto-removed it, losing the logs; the driver wrote a null-metric record and exited 0
- **Fix:** removed `--rm` from serve_30_70_vllm.sh (rm -f at start/stop already handles cleanup); wait_healthy now dumps docker logs to logs/vllm_boot_failures/ before raising; prune_gated_eval exits 1 on null primary metric so the chained driver stops. Retry with identical config booted cleanly (mask patch installed on all 48 layers) — consistent with a transient (Phase 9 precedent: judge_vllm_restart_0702.log)
- **Files modified:** scripts/serve_30_70_vllm.sh, scripts/_p0_vllm_smoke_serve.py, scripts/prune_gated_eval.py
- **Commit:** a2e5aea

**2. [Rule 1 - Bug] np.bool_ crashed the judge record write AFTER all GPU work**
- **Found during:** Task 2 (record write, post-serve)
- **Issue:** `spearmanr().statistic` returns np.float64; `ens_rho >= floor` yields np.bool_, which stdlib json rejects — TypeError after ~2h of judge serving
- **Fix:** `_write_result` uses `json.dumps(..., default=lambda o: o.item())`; judge record rescored from the intact on-disk captures (no re-serving)
- **Files modified:** scripts/prune_gated_eval.py
- **Commit:** 32ce674

**3. [Rule 2 - Missing critical functionality] Records lacked the plan's verify contract fields**
- **Found during:** Task 1/2 verification
- **Issue:** the 13-02 driver emitted pass_gen_wp_bench / pass_judge_* but not the top-level `pass` key the plan's automated verify asserts, and no per-seed rho (HARD CONSTRAINT 6 requires all-3-seed validation); it also lacked the d2_security record 13-03's prune_selection.load_variant_records expects
- **Fix:** added `pass` to both records, `per_seed_rho` (all 3 seeds) to score_judge_gate, and `{method}_{ratio}_d2.json` emission (baseline from Phase-11 kfull captures); masks saved to `output/prune/masks/{method}_{axis}_k{K}.npy` matching the plan's files_modified contract
- **Files modified:** scripts/prune_gated_eval.py
- **Commits:** a2e5aea, 32ce674

### Known Limitation (documented, not fixed)

- **Gen-axis 9-dim retention not measurable from this arm:** wp-bench emits knowledge/correctness/quality, not the D1-D9 rubric dims; the 13-02 driver (approved machinery) never captured gen-side rubric dims and doing so would require a separate judge-scored eval serve. The 9-dim signal this plan carries lives on the judge axis (parsed dimension scores feeding the d2 record). Given the decisive gen FAIL, the additional GPU spend was not warranted; 13-06's D2 eligibility check is satisfied fail-closed by the annotated d2 record.

## Issues Encountered
- First gen-arm boot failure was undiagnosable post-hoc (logs auto-removed) — the infra fix above makes any recurrence diagnosable; the visible JSONDecodeError in boot logs is vLLM's usage-telemetry daemon thread failing on GB10 cpuinfo, nonfatal noise present in successful boots too

## User Setup Required
None.

## Threat Flags
None - no new network endpoints or trust-boundary changes. T-13-03 mitigation exercised for real: protected mask sha256 re-verified before every serve (4 serves total), protected_retained true in all three records. T-13-02 honored: bars are the vLLM-measured 0.4284/0.7555/0.95; both FAILs recorded honestly (the JSON's superseded recalibrated_floor 0.425 never entered gating logic).

## Next Phase Readiness
- 13-05 branches on this result: AIMER@25 failed both axes decisively → short-circuit (no 50/75% expansion, no REAP calibration — its gate condition "AIMER@25 passes" is false)
- 13-06 has everything it needs: `output/prune/gated/aimer_25_{gen,judge,d2}.json` merge cleanly through prune_selection.load_variant_records → expected verdict no_winner (phase ships unpruned)
- All Phase-13 gate machinery is now battle-tested on real hardware (masked serve, wp-bench, 3-seed judge capture, rescore-from-captures path)

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

All 5 artifacts + driver script + SUMMARY exist on disk; both task commits (a2e5aea, 32ce674) found in git log.
