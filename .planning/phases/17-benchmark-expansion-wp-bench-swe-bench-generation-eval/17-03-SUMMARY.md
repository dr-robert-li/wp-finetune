---
phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
plan: 03
subsystem: eval
tags: [swe-bench, arm64, vllm, generation-mode, model-card, benchmarks]

requires:
  - phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
    provides: "17-01 wp-bench full rerun receipt (0.4365); 17-02 arm64 wrapper + throughput probe + committed scope pre-registration (Lite-300 + PHP-43, oracle, <=20h)"
provides:
  - "output/bench17/swebench_predictions.jsonl + swebench_predictions_php.jsonl — generation-mode patches at the locked scope (schema: instance_id, model_name_or_path, model_patch)"
  - "output/bench17/swebench_eval_report.json — BENCH-02 receipt: Lite-300 5/300 resolved (1.67% full-scope; 3.82% on 131 evaluated), PHP-43 0/43; full disclosure by category"
  - "MODEL_CARD.md Benchmarks section: wp-bench 0.4365 + both SWE-bench numbers + explicit out-of-domain caveat (BENCH-03)"
affects: [18]

tech-stack:
  added: []
  patterns:
    - "Prompt construction via the official swebench.inference.make_datasets pipeline (style-2, file_source=oracle) for datasets without a pre-built *_oracle variant — keeps PHP-43 prompts oracle-equivalent without hand-rolling"
    - "Patch post-processing follows run_live.py order: extract_diff -> extract_minimal_patch (official utils, never hand-rolled)"
    - "Wrapper mirrors upstream get_dataset_from_preds: empty-patch predictions excluded from container runs but kept in the predictions dict so make_run_report classifies them; report-file existence check makes long eval runs resume-safe"

key-files:
  created:
    - scripts/bench17_swebench_generate_predictions.py
    - scripts/bench17_swebench_consolidate_report.py
    - output/bench17/swebench_predictions.jsonl
    - output/bench17/swebench_predictions_php.jsonl
    - output/bench17/swebench_generation_receipt.json
    - output/bench17/swebench_eval_report.json
    - output/bench17/swebench_harness_report_lite300_v1.json
    - output/bench17/swebench_harness_report_php43_v1.json
  modified:
    - scripts/swebench_arm64_eval.py
    - output/packaging/MODEL_CARD.md
    - JOURNAL.md
    - CHANGELOG.md
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Full-scope denominators (Lite n=300, PHP n=43) are the primary rates; every non-resolution (over-length, unparseable, apply-failed, arm64-env-failed) scored unresolved and disclosed by category — per the pre-registration, conservative against the model"
  - "29 Lite instances whose 2018-era Python envs cannot build on arm64/2026 toolchains (cdms2 no aarch64 build, py3.6 setuptools/scipy, pip removed --no-use-pep517, PEP-660 gaps, deleted sympy branch) scored unresolved rather than excluded — benchmark env specs were NOT modified"
  - "Stale Nov-2024 swebench Docker images (7 env + 17 instance) purged after they caused 15 /testbed-collision failures and 1 contaminated completed instance; all affected instances re-run, promoting requests-2674 to resolved"

requirements-completed: [BENCH-02, BENCH-03]

coverage:
  - id: D1
    description: "Generation-mode predictions produced for every scoped instance in exact swebench schema via local vLLM with real-generation warm-up"
    requirement: BENCH-02
    verification:
      - kind: other
        ref: "output/bench17/swebench_predictions.jsonl (300 rows) + swebench_predictions_php.jsonl (43 rows), schema-asserted; generation receipt output/bench17/swebench_generation_receipt.json"
        status: pass
    human_judgment: false
  - id: D2
    description: "Native arm64 containerized eval at the pre-registered scope with a consolidated resolved/unresolved report cross-referencing the pre-registration"
    requirement: BENCH-02
    verification:
      - kind: other
        ref: "output/bench17/swebench_eval_report.json (resolved rates both denominators, scope/variant/retrieval/arch/version/seed, disclosure by category); eval-report commit ae488fd postdates pre-registration commit 65116ed"
        status: pass
    human_judgment: false
  - id: D3
    description: "MODEL_CARD Benchmarks section with both numbers + out-of-domain caveat; JOURNAL/STATE/CHANGELOG/REQUIREMENTS/ROADMAP updated; committed + pushed as Dr. Robert Li, no AI trailer"
    requirement: BENCH-03
    verification:
      - kind: other
        ref: "output/packaging/MODEL_CARD.md '## Benchmarks'; commit d23fbf1 (author Dr. Robert Li, body clean of AI trailers); pushed to phase10-execution"
        status: pass
    human_judgment: false

duration: ~4.8h session (heavy path ~3.5h: generation 1.36h + eval ~2.2h) vs 16.93h projection
completed: 2026-07-11
status: complete
---

# Phase 17 Plan 03: SWE-bench Generation + arm64 Eval + MODEL_CARD Benchmarks Summary

**SWE-bench generation-mode at the locked pre-registered scope: Lite-300 resolves 5/300 (1.67% full-scope, 3.82% on the 131 container-evaluated) and PHP-43 resolves 0/43, all non-resolutions disclosed by category, both numbers folded into MODEL_CARD.md next to the fresh wp-bench 0.4365 with an explicit out-of-domain caveat.**

## Performance

- **Duration:** ~4.8h session; heavy path ~3.5h (generation 1.36h + Docker eval ~2.2h incl. fix pass) vs the 16.93h pre-registered projection. The 180s/instance Python Docker estimate was conservative; many failing instances die at patch-apply in seconds, and 4 workers ran in parallel.
- **Completed:** 2026-07-11
- **Tasks:** 3/3 auto
- **Files:** 8 created, 7 modified

## Accomplishments

- **Predictions at exact scope, official pipeline end to end:** Lite-300 prompts are the `SWE-bench_Lite_oracle` `text` field verbatim; PHP-43 prompts built with swebench's own `add_text_inputs(file_source="oracle", prompt_style="style-2")` (no hand-rolled template). Patches extracted with official `extract_diff` → `extract_minimal_patch`. vLLM bf16, max_model_len 24576, concurrency 2, temp 0.0, seed 0, enable_thinking=false, real-generation warm-up gate. Server stopped after generation.
- **Native arm64 eval, no host-side patch application:** 17-02's `swebench_arm64_eval.py` wrapper (make_test_spec arch="arm64", namespace=None) ran both legs through the swebench 4.1.0 containerized harness. Wrapper gained the upstream-mirror empty-patch exclusion and report-file resume (needed for the fix pass); FAIL_TO_PASS/PASS_TO_PASS bookkeeping is the harness's own per-instance report.json, not re-derived.
- **Honest accounting, full-scope denominators:** Lite 300 = 5 resolved + 126 unresolved-in-container + 80 over-length (never fit 24576−2048) + 1 unparseable + 59 apply-failed + 29 arm64-env-failed. PHP 43 = 20 unresolved-in-container + 17 over-length + 6 apply-failed. Sanity assertion in the consolidation script proves every scoped instance is accounted exactly once.
- **BENCH-03 docs:** MODEL_CARD gains a Benchmarks section (table + wp-bench reproduction note + "why the number is low" caveat: WordPress/PHP-specialized model, Python-repo patch generation, low number expected and published for positioning not vanity; PHP subset reported separately as the in-language data point). JOURNAL entry, CHANGELOG entry, STATE/ROADMAP/REQUIREMENTS closeout (BENCH-01..03 all checked; Phase 17 3/3 plans).

## Task Commits

1. **Task 1: generation-mode predictions** - `07275e2` (feat)
2. **Task 2: arm64 eval + consolidated report** - `ae488fd` (feat); raw harness reports archived in `7c93b9e` (chore)
3. **Task 3: MODEL_CARD Benchmarks + docs closeout** - `d23fbf1` (docs)

All authored Dr. Robert Li, no AI co-author trailer, pushed to `phase10-execution`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stale Nov-2024 swebench Docker images contaminated the first Lite pass**
- **Found during:** Task 2 (error triage: 98 errors, 44 not patch-apply failures)
- **Issue:** 7 env + 17 instance images from a 2024 swebench install on this host were silently reused by 4.1.0's name-based cache; the old layout bakes the repo into `/testbed`, colliding with the new instance build (15 instances errored) and one completed instance (pytest-7490) had run inside a stale image.
- **Fix:** purged all 2024-dated `sweb.*` images, deleted the contaminated instance's log dir, re-ran the affected instances (resume-safe wrapper). +10 evaluated, +1 resolved (requests-2674).
- **Files modified:** scripts/swebench_arm64_eval.py (resume + empty-patch filter)
- **Commit:** ae488fd

**2. [Rule 3 - Blocking] Wrapper would have wasted ~4h running 81 empty-patch containers**
- **Fix:** mirrored upstream `get_dataset_from_preds` semantics (empty-patch predictions excluded from container runs, still classified by make_run_report). Same commit.

### Known limitation (disclosed, not fixed by design)

29 Lite instances cannot be evaluated on this host: their 2018-era conda/pip environment specs do not build on linux-aarch64 with 2026 toolchains (cdms2 unavailable, py3.6 setuptools/scipy pins, `--no-use-pep517` removed from pip, PEP-660 editable gaps, sympy branch `1.7` deleted upstream). Modifying the benchmark's env specs would break comparability; the instances are scored unresolved and disclosed in the receipt (`harness_env_failed_ids`).

## Issues Encountered

- `output/` is gitignored; bench17 receipts and MODEL_CARD committed with `git add -f`, consistent with 17-01/17-02.
- The swebench harness writes its run report to CWD; archived under `output/bench17/swebench_harness_report_*.json` to keep the repo root clean (Phase 16 layout rule).

## User Setup Required

None.

## Next Phase Readiness

- Phase 17 complete (BENCH-01..03). Phase 18 (production sweep + HF publication) has both benchmark numbers in the card with receipts.
- Host clean: no vLLM containers, no stale swebench images; PHP/Python arm64 env images cached for any future rerun.

---
*Phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval*
*Completed: 2026-07-11*

## Self-Check: PASSED
All claimed files verified present on disk; all claimed commit hashes verified present in git log.
