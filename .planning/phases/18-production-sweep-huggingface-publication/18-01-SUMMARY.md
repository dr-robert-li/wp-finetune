---
phase: 18-production-sweep-huggingface-publication
plan: 01
subsystem: docs+repo-hygiene
tags: [production-sweep, publication-prep, archive-sweep, doc-currency]
dependency-graph:
  requires: [Phase 17 benchmark numbers, output/packaging/MODEL_CARD.md]
  provides: [current README/PROJECT/PIPELINE consistent with shipped v3.0/v3.1 reality, clean deprecated/ archive]
  affects: [18-02 (HF publication cards link back to this repo state)]
tech-stack:
  added: []
  patterns: [double-grep gate (import + string-literal) before any archive move]
key-files:
  created: []
  modified:
    - README.md
    - PROJECT.md
    - wp-moe.md
    - deprecated/README.md
    - "10 files: scripts/* -> deprecated/scripts/*"
    - "6 files: deprecated/scripts/* -> scripts/* (restored mis-archived live deps)"
decisions:
  - "wp-moe.md stays in place (README/PROJECT link to it, so it fails the double-grep dead-file test) — got a currency banner pointing to MODEL_CARD.md/PIPELINE.md instead of a full rewrite or move"
  - "PIPELINE.md needed no edits — verified its stage outcomes already match MODEL_CARD.md's lineage numbers exactly (no drift found)"
  - "Ran the double-grep gate against every existing file under deprecated/, not just Phase 18's new candidates — caught 6 files wrongly archived in an earlier phase and restored them"
metrics:
  duration: "~90 min"
  completed: "2026-07-11"
status: complete
---

# Phase 18 Plan 01: Production Sweep Summary

Brought README/PROJECT.md/wp-moe.md current with the shipped v3.0/v3.1 reality (RL rejected, MoE-Sieve no
compression, prune no winner, Q8 GGUF ship tier, Phase 17 benchmark numbers), then ran the mandatory
double-grep archive sweep and discovered — and fixed — 6 files from an earlier phase that were live
dependencies wrongly sitting in `deprecated/`.

## What shipped

**Task 1 (commit `5865ae9`):** README.md gained a two-model-pair framing (was incorrectly describing a
single routed model), a Project Status table running through Phase 18 (was stuck at "Next: Phase 9 GSPO"),
and a new Benchmarks section carrying the Phase 17 numbers verbatim from MODEL_CARD.md (wp-bench 0.4365
full / 0.4484 Gate-1, judge rho 0.8075 ensemble / 0.8017 single-seed, Q8 GGUF −47% lossless, SWE-bench
1.67%/0% with the out-of-domain caveat). PROJECT.md's Current Status checklist and Phase D/E sections now
state actual outcomes instead of "Planned"/"next step". wp-moe.md (the v1.0-era design doc, still linked
from both README and PROJECT.md) got a currency banner rather than a rewrite or move — it fails the
double-grep liveness test by definition since those two docs link to it.

**Task 2 (commit `3ad2513`):** Archived 10 double-grep-clean one-off drivers: the Phase 17 SWE-bench/
wp-bench-rerun family (5 files — Phase 17 numbers are final, no rerun planned) and the original Phase 1/2
judge-batch data-generation cluster (5 files — dataset frozen since v1.0). Then, running the same gate
against *every* file already under `deprecated/` (not just the new candidates) turned up 6 files wrongly
archived in an earlier phase that active tests/scripts still depend on — restored to `scripts/`:
`_p0_smoke_common.py` (imported by a test), `launch_validated_smoke.sh` (path-referenced by a test),
`reward_form_sweep.py` + `_probe_rl_reward.py` (a test imports the sweep, which imports the helper),
`_p0_revl01_preflight.py` (invoked via `subprocess.run`), `_sieve_vllm_patch/sitecustomize.py` (bind-mounted
into the vLLM container). This is exactly the class of break the plan's threat model called out (T-18-01-AV)
— caught before it shipped, not after.

## Verification

- Doc-currency gate (Task 1 automated check): PASS.
- Double-grep gate (Task 2 automated check): 4 remaining hits, all manually confirmed comment-only mentions
  (not runtime imports/subprocess/path calls) — documented in `deprecated/README.md` as known false
  positives of the blunt grep. No live reference to any archived file remains.
- Root tracked-file check: only `.env.example .gitignore .gitmodules CHANGELOG.md JOURNAL.md PIPELINE.md
  PROJECT.md README.md wp-moe.md` at root — no stray tracked artifacts.
- Restored files (`_p0_smoke_common.py`, `reward_form_sweep.py`, `_probe_rl_reward.py`,
  `_p0_revl01_preflight.py`) parse cleanly (`ast.parse`); `launch_validated_smoke.sh` and
  `_sieve_vllm_patch/sitecustomize.py` confirmed present at their expected live paths.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed a staged-commit mixing mistake**
- **Found during:** Task 1 commit.
- **Issue:** `git mv` for Task 2's archive moves had already staged those renames before Task 1's doc
  commit; a plain `git commit -m` (no pathspec) picked up all staged changes, merging both tasks into one
  commit.
- **Fix:** `git reset --soft HEAD~1`, unstaged the Task 2 renames, re-committed Task 1 alone, then staged
  and committed Task 2 separately.
- **Commit:** `5865ae9` (Task 1), `3ad2513` (Task 2).

**2. [Rule 1 - Bug] Restored 6 files wrongly archived to `deprecated/` in an earlier phase**
- **Found during:** Task 2, running the double-grep gate against the full `deprecated/` tree.
- **Issue:** `_p0_smoke_common.py`, `launch_validated_smoke.sh`, `reward_form_sweep.py`,
  `_probe_rl_reward.py`, `_p0_revl01_preflight.py`, `_sieve_vllm_patch/sitecustomize.py` were sitting in
  `deprecated/scripts/` but are live imports/subprocess targets/bind-mounts from active tests and scripts —
  the exact class of break the plan's threat model (T-18-01-AV) exists to catch.
- **Fix:** `git mv` each back to `scripts/`; documented the restore and the discovering grep hits in
  `deprecated/README.md`.
- **Files modified:** `deprecated/README.md`, plus the 6 restored files.
- **Commit:** `3ad2513`.

## Known Stubs

None.

## Threat Flags

None — Task 2's fix directly closes the T-18-01-AV gap the plan's threat model called out (mis-archived
live dependencies), rather than introducing new surface.

## Self-Check: PASSED

- `README.md`, `PROJECT.md`, `wp-moe.md`, `deprecated/README.md` — FOUND.
- `scripts/_p0_smoke_common.py` — FOUND (restored).
- `scripts/reward_form_sweep.py` — FOUND (restored).
- `deprecated/scripts/judge_batch.py` — FOUND (archived).
- Commit `5865ae9` — FOUND in `git log --oneline`.
- Commit `3ad2513` — FOUND in `git log --oneline`.
