---
phase: 23-final-evaluation
plan: 02
subsystem: evaluation
tags: [judge-rho, q8-gguf, llama-cpp, paired-bootstrap, shipped-stack, pre-registration]

requires:
  - phase: 23-final-evaluation
    provides: "23-01 EVAL4-01 verdict (judge valid_recorded_miss on bf16-vLLM) + DIAGNOSTIC_SYNTHESIS engine-numerics-ceiling finding"
  - phase: 21-sft-training-generation-judge-models
    provides: "wp-judge-v4-s{0,1,2} promoted ep3 checkpoints (Tinker manifests) + s1 merged model + proven fused-expert merge path"
provides:
  - "output/eval4/ext_q8_preregistration.md -- pre-registered primary metric + unequivocal-win rule (committed BEFORE measurement)"
  - "output/eval4/ext_q8_results.json -- machine-readable verdict: v4 Q8 ensemble 0.8067 vs v3 0.8056, paired delta CI spans 0, unequivocal_win=false"
  - "output/eval4/ext_q8/ -- per-seed captures, rho receipts, ensemble receipt"
  - "models/_gguf/wp-v4-judge-s{0,1,2}.Q8_0.gguf -- v4 judge Q8 artifacts (37.8 GiB each)"
  - "models/Qwen3.6-35B-A3B-judge-v4-s{0,2}-merged -- newly merged seeds (240/240 guard)"
  - "VERDICT-EVAL4.md section 6 -- judge-only-ship implication (v3 stays canonical)"
affects: [24-conditional-gate, 25-conditional-gate, 26-conditional-gate, 27-packaging]

tech-stack:
  added: []
  patterns:
    - "Pre-register-then-measure: decision rule (point AND paired-bootstrap-CI) committed to git before any merge/conversion/serve for the run"
    - "Paired per-item bootstrap (same resampled indices both arms, 10k, seed 1337) to isolate the rho delta from independent sampling noise"
    - "Shipped-stack parity eval: measure the candidate on the incumbent's EXACT serving stack (llama.cpp Q8 @8192) before any ship call"

key-files:
  created:
    - output/eval4/ext_q8_preregistration.md
    - output/eval4/ext_q8_results.json
    - output/eval4/ext_q8/merge_s0_s2_manifest.json
    - output/eval4/ext_q8/q8_ensemble.json
    - scripts/eval4_ext_merge_seeds.py
    - scripts/eval4_ext_gguf_convert.sh
    - scripts/eval4_ext_q8_run.sh
    - scripts/eval4_ext_verdict.py
    - models/_gguf/wp-v4-judge-s0.Q8_0.gguf
    - models/_gguf/wp-v4-judge-s1.Q8_0.gguf
    - models/_gguf/wp-v4-judge-s2.Q8_0.gguf
  modified:
    - output/eval4/VERDICT-EVAL4.md

key-decisions:
  - "unequivocal_win = FALSE (rule_fired=paired_bootstrap): point 0.8067 > 0.8056 passes (a) by +0.0011, but paired delta +0.0010 [-0.0512, +0.0565] fails (b) -- v3 pair stays canonical, judge-only v4 ship NOT justified"
  - "Serving ceiling is engine-independent: v4 s1 Q8-llama.cpp 0.7877 ~= v4 s1 bf16-vLLM 0.7872 -- llama.cpp does NOT lift v4's numerics ceiling; capture-path gain (+0.0084) unrealized on every measured serving stack"
  - "GGUF block-count sanity must include MTP layers: expected = num_hidden_layers + mtp_num_hidden_layers (41 = 40+1 for this arch); initial 40-only assert was a checker bug, not a conversion defect"

patterns-established:
  - "Paired-comparison joins validated by recomputation: v3 ensemble recomputed from its raw per-item captures reproduces the shipped 0.8056 to 4 decimals before any delta is trusted"

requirements-completed: []

coverage:
  - id: D1
    description: "Pre-registration committed before any measurement (primary metric, win rule a+b, fallback, failure disposition)"
    requirement: "Phase-23-02-EXTENSION"
    verification:
      - kind: other
        ref: "git log: f1f74b1 (pre-registration) precedes dc18f3e/ca1d4ff/fb77e11 (merge/convert/eval commits)"
        status: pass
    human_judgment: false
  - id: D2
    description: "s0/s2 merges with 240/240 fused-expert guard; 3x Q8_0 GGUF with block-count sanity vs config (41=40+1 MTP)"
    requirement: "Phase-23-02-EXTENSION"
    verification:
      - kind: other
        ref: "output/eval4/ext_q8/merge_s0_s2_manifest.json (merge_ok both) + gguf_convert.log (3x 'block-count sanity: PASS')"
        status: pass
    human_judgment: false
  - id: D3
    description: "3-seed Q8 llama.cpp eval @8192/temp0 on the 121-item val set, 0 parse failures, median ensemble + pre-registered paired verdict"
    requirement: "Phase-23-02-EXTENSION"
    verification:
      - kind: other
        ref: "output/eval4/ext_q8_results.json: parse_fail [0,0,0], v4_ensemble.rho=0.8067, paired_bootstrap.ci_lower=-0.0512, unequivocal_win=false"
        status: pass
    human_judgment: false

duration: ~5h (wall clock incl. detached merges/conversions/serves)
completed: 2026-07-15
status: complete
---

# Phase 23 Plan 02: Shipped-Stack (llama.cpp Q8) Extension Summary

**v4's judge on v3's exact shipped stack is statistically indistinguishable from v3: Q8 3-seed ensemble 0.8067 vs 0.8056 (paired Δ +0.0010, CI [−0.051, +0.057] spans zero) — NOT an unequivocal win, so the v3 pair stays canonical and judge-only v4 shipping is not justified.**

## Performance

- **Duration:** ~5h wall clock (merges ~6.5 min/seed, conversions ~12 min/seed, serve+capture ~80 min/seed)
- **Completed:** 2026-07-15
- **Tasks:** 5 (pre-register, merge s0/s2, GGUF x3, eval x3 + ensemble, verdict + docs)

## Accomplishments

- Pre-registered the decision rule and committed it BEFORE any measurement (f1f74b1): win := point > 0.8056 AND paired-bootstrap CI-lower > 0.
- Merged s0/s2 promoted ep3 adapters via the proven `tinker_cookbook.build_hf_model` fused-expert path — 240/240 module guard on both, 66.99 GiB each.
- Converted all 3 merged seeds to Q8_0 GGUF with `convert_hf_to_gguf.py --outtype q8_0` (llama.cpp `8f114a9`, 774 commits past the b9180 arch floor); block-count sanity 41=40+1(MTP) on every seed; vision tower dropped (text pipeline, expected).
- Served each GGUF via v3's unmodified `_pkg_gguf_eval_run.sh` harness (llama-server, -ngl 999, --jinja, --parallel 4, real-generation warmup gate), captured the same 121 val prompts @8192/temp0 — **0 parse failures on all three seeds**.
- Median-ensembled and scored with the unmodified `eval_relabel_ensemble.py`: **0.8067 [0.7356, 0.8526]**.
- Applied the pre-registered rule mechanically via `scripts/eval4_ext_verdict.py`: (a) TRUE, (b) FALSE → **unequivocal_win = FALSE**. v3 recomputed from its own raw captures reproduces 0.8056 exactly, validating the paired join.
- Updated `VERDICT-EVAL4.md` with section 6 (shipped-stack comparison + judge-only-ship implication).

## The numbers

| Figure | v4 | v3 shipped |
|---|---|---|
| Q8 s0 / s1 / s2 | 0.7360 / 0.7877 / 0.7758 | 0.7744 / 0.7928 / 0.7894 |
| Q8 3-seed ensemble | **0.8067** [0.7356, 0.8526] | **0.8056** [0.7381, 0.8577] |
| Paired Δ (10k, seed 1337) | +0.0010 [−0.0512, +0.0565] | — |

Secondary read: v4 s1 Q8-llama.cpp (0.7877) ≈ v4 s1 bf16-vLLM served (0.7872) — the ~0.79 single-seed serving ceiling is engine-independent; llama.cpp does not unlock the capture-path 0.8358.

## Task Commits

1. **Pre-registration** — `f1f74b1` (docs)
2. **s0/s2 merges (240/240)** — `dc18f3e` (feat)
3. **GGUF x3 + harness scripts** — `ca1d4ff` (feat)
4. **Eval + verdict receipt** — `fb77e11` (feat)
5. **VERDICT section 6 + SUMMARY + STATE** — (this commit, docs)

## Deviations from Plan

- **[Rule 1 - Bug] Block-count assert missed MTP layers.** First conversion tripped `gguf=41 vs config=40`. Root cause: this arch has `mtp_num_hidden_layers=1` and b9180+ converters export the MTP layer as a block (bartowski ships likewise). Fixed the checker (`expected = num_hidden_layers + mtp_num_hidden_layers`); the conversion itself was never defective. Files: `scripts/eval4_ext_gguf_convert.sh`, commit `ca1d4ff`.
- **[Rule 3 - Blocking] TINKER_API_KEY not exported.** First merge launch failed on auth; key exists in `.env` per the established `set -a; source .env` convention (21-02 lesson). Re-launched with env sourced — not a human auth gate.
- **[Rule 3 - Blocking] `eval_relabel_ensemble.py` cannot be imported** (executes its body at module level). Inlined its `parse_capture` verbatim into `eval4_ext_verdict.py` instead of importing. Commit `fb77e11`.

## Issues Encountered

None beyond the deviations above. Teardown verified clean (no llama-server processes, port free) before verdict computation.

## Next Phase Readiness

- **Phases 24–26 (conditional gates) / Phase 27 (packaging):** the judge-role recommendation is unchanged and now measured on the shipping stack: **ship v3's judge pair** (v1.3 Q8 ensemble 0.8056); the v4 judge adds no serving-time value at +25% artifact size (37.8 vs 30.2 GiB).
- The capture-vs-served gap (~0.83–0.84 → ~0.79) is now confirmed engine-independent across vLLM-bf16 and llama.cpp-Q8 — any future attempt at the judge targets should attack the serving-numerics ceiling itself or evaluate on the capture path, per DIAGNOSTIC_SYNTHESIS.md.

---
*Phase: 23-final-evaluation*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 11 key files found on disk; all 4 task commit hashes (f1f74b1, dc18f3e, ca1d4ff, fb77e11) found in git log. Teardown verified: no llama-server running, GPU idle.
