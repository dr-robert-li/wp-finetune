---
phase: 11-compression-packaging
plan: 03
subsystem: sieve-routing-analysis
tags: [moe-routing, cross-seed-overlap, protected-mask, sieve]
requires:
  - models/_staging/qwen3-30b-wp-v1.3-s0-merged/ (from 11-02)
  - models/_staging/qwen3-30b-wp-v1.3-merged/ (s1, pre-existing)
  - models/_staging/qwen3-30b-wp-v1.3-s2-merged/ (from 11-02)
  - output/profiling/reasoning-merged-v4/ (Phase-7 gen profile + immutable mask)
  - scripts/profile_merged_model.py, scripts/compute_concentration.py
provides:
  - output/sieve/judge-s0|s1|s2/routing_report.jsonl + concentration_report.json (gitignored)
  - scripts/sieve_cross_seed_overlap.py
  - scripts/sieve_protected_retention.py
  - output/sieve/cross_seed_overlap.json (sieve_profile_mode=shared, gitignored)
  - output/sieve/protected_retention_check.json (per-k retention + stability notes, gitignored)
affects:
  - plan 11-04 (k-sweep masking design: ONE shared profile, protected experts force-retained at k<64)
  - plan 11-05 / Phase 13 (layer_stability_notes passthrough)
tech-stack:
  added: []
  patterns: [bounded-stimulus-seed-profiling, argsort-topk-jaccard-reuse, read-only-mask-verification]
key-files:
  created:
    - scripts/sieve_cross_seed_overlap.py
    - scripts/sieve_protected_retention.py
  modified:
    - scripts/profile_merged_model.py (--limit bounded stimulus, --model-tag)
decisions:
  - "Open Question 2 RESOLVED with a measured number: cross-seed mean Jaccard 0.9332 >= 0.90 -> sieve_profile_mode=shared (one masking profile covers all 3 judge seeds)"
  - "Protected mask fully inside shared top-64 hot set (0 at risk); at k=32/13 the k-sweep mask must force-retain 198/866 protected experts beyond pure top-k"
  - "Seed profiling used bounded 3000-example ratio_30_70 stimulus (not the full 34,855 pass) — mask is FROZEN from Phase 7's full pass; seeds only need relative cross-seed comparison"
  - "Host .venv-tinker python used for profiling instead of ngc-pytorch container — container script is interactive-only (-it, exec bash); .venv-tinker has working torch 2.12.0+cu130 + transformers 5.5.3 + CUDA"
metrics:
  duration: ~4h (dominated by 3 sequential ~70min GPU profiling passes)
  completed: 2026-07-09
status: complete
requirements: [SIEVE-01]
---

# Phase 11 Plan 03: Judge-Seed Routing Profiles + Cross-Seed Overlap Summary

Profiled all 3 judge-seed merged checkpoints on a bounded matched stimulus, measured
cross-seed routing overlap (mean Jaccard 0.9332 -> ONE shared Sieve profile covers all 3
seeds), and verified the immutable 1,480-expert protected mask is a subset of the shared
top-64 hot set — with per-k at-risk counts recorded for k=13/32.

## What was done

**Task 1 — Profile 3 judge seeds (gen profile reused).** Added `--limit` (bounded
full-pass stimulus) and `--model-tag` to `scripts/profile_merged_model.py`, then ran it
sequentially per seed (one ~60GB model resident at a time, GB10 memory wall respected)
with `--ratio ratio_30_70 --limit 3000`:

| Seed | Checkpoint | E_eff mean | records |
|------|-----------|-----------|---------|
| s0 | qwen3-30b-wp-v1.3-s0-merged | 72.80 | 48 layers |
| s1 | qwen3-30b-wp-v1.3-merged | 72.90 | 48 layers |
| s2 | qwen3-30b-wp-v1.3-s2-merged | 73.05 | 48 layers |

`compute_concentration.py` run per seed (base-model D-08 join intact). The gen model was
NOT re-profiled — Phase-7 `output/profiling/reasoning-merged-v4/` reused directly (its
routing_report.jsonl mtime unchanged, 2026-06-14).

Note: the PROF-03 jaccard CI gate printed FAIL for s1 (0.9258) and s2 (0.9352) vs the
0.94 bar. That gate's semantics (subsample-vs-FULL stability on the 34,855-example pass)
do not map to this bounded 3000-example stimulus, where the "subsample" (3,485 random
examples) is larger than the reference pass. Informational only — the plan's acceptance
criteria for seed profiling are per-layer E_eff for relative cross-seed comparison, and
the FROZEN Phase-7 mask (which DID pass the full-pass gate) is the pruning defense.

**Task 2 — Cross-seed overlap (Open Question 2).** `scripts/sieve_cross_seed_overlap.py`
computes per-layer top-32 expert sets per seed (same argsort convention as
`compute_jaccard_stability`) and pairwise Jaccard across s0/s1/s2:

- s0-s1: 0.9390 · s0-s2: 0.9335 · s1-s2: 0.9271
- **mean_overlap = 0.9332 >= 0.90 threshold -> `sieve_profile_mode = "shared"`**
- Threshold + observed value + decision rule all recorded in `cross_seed_overlap.json`

One masking profile covers all 3 judge seeds; plan 11-04's k-sweep does NOT need
per-seed union masks.

**Task 3 — Protected-mask subset verification.** `scripts/sieve_protected_retention.py`
(read-only w.r.t. the mask) asserts the Phase-7 mask is [48,128] bool sum=1480, builds
shared-profile top-k hot sets (summed seed counts, per the Task-2 decision), and checks
subset membership per candidate k:

| k | protected_retained | at risk |
|---|--------------------|---------|
| 13 | False | 866 |
| 32 | False | 198 |
| 64 | True | 0 |

The JSON records that k-sweep masks at k<64 MUST force-retain the at-risk protected
experts (HARD CONSTRAINT 1). `layer_stability_notes` carried forward verbatim
(verified equal to the Phase-7 mask JSON's value). Mask files byte-unchanged:
sha256 `659af6eb…` (.npy) / `ade549e0…` (.json) identical before/after.

## Verification

- `python3 -c ...routing_report.jsonl` → 48/48/48 records for s0/s1/s2
- `.venv-tinker/bin/python -m pytest tests/test_sieve_cross_seed_overlap.py tests/test_sieve_protected_retention.py -q` → **8 passed, 0 skipped** (Wave-0 skips now GREEN)
- `cross_seed_overlap.json` records sieve_profile_mode + 0.90 threshold + observed 0.9332
- `protected_expert_mask.{npy,json}` sha256-identical before/after Task 3 (T-11-06 mitigated)
- T-11-07 mitigated: decision derived from measured Jaccard vs recorded threshold, math test-asserted

## Deviations from Plan

**1. [Rule 3 - Blocking] Host miniconda python cannot load Qwen3Moe (torchvision::nms op error).**
- **Found during:** Task 1 (s0 launch)
- **Issue:** system `python3` (torch 2.12.1/torchvision 0.25.0 mismatch) fails
  `Qwen3MoeForCausalLM` import via a broken torchvision op registration.
- **Fix:** used `.venv-tinker/bin/python` (torch 2.12.0+cu130, transformers 5.5.3,
  CUDA available) for all profiling/analysis instead. No code change. The plan's
  ngc-pytorch container invocation was not usable non-interactively (`docker run -it
  … exec bash`), and `.venv-tinker` is the project's sanctioned working python.
- **Files modified:** none
- **Commit:** n/a (environment selection)

**2. [Rule 2 - Missing critical] `--limit` flag did not exist in profile_merged_model.py.**
- **Found during:** Task 1 read_first
- **Issue:** plan specifies `--limit` for the bounded stimulus; script only had
  fraction-based `full_subsample_frac` (not exposed on CLI).
- **Fix:** added `--limit` (hard example cap on the full reference pass, precedence
  over fraction) and `--model-tag` (JSONL model field, defaults to output-dir basename
  so seed reports are distinguishable).
- **Files modified:** scripts/profile_merged_model.py
- **Commit:** 355a538

## Known Stubs

None — all three scripts are wired to real data and produced real decision artifacts.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary schema changes.
T-11-06/T-11-07 mitigations verified above.

## Notes for plan 11-04

- `sieve_profile_mode = "shared"`: build ONE hot/cold classification from the summed
  seed counts (or s1 alone — overlap says any is representative), not 3.
- At k=13/32 the mask must be `top-k ∪ protected` — pure top-k drops 866/198 protected
  experts respectively. At k=64 pure top-k already contains the full protected set.
- Per-seed routing artifacts live under `output/sieve/judge-s{0,1,2}/` (gitignored).

## Self-Check: PASSED

- FOUND: output/sieve/judge-s0/routing_report.jsonl (48 records)
- FOUND: output/sieve/judge-s1/routing_report.jsonl (48 records)
- FOUND: output/sieve/judge-s2/routing_report.jsonl (48 records)
- FOUND: output/sieve/judge-s{0,1,2}/concentration_report.json
- FOUND: scripts/sieve_cross_seed_overlap.py, scripts/sieve_protected_retention.py
- FOUND: output/sieve/cross_seed_overlap.json, output/sieve/protected_retention_check.json
- FOUND commits: 355a538 (Task 1), 8df29c3 (Task 2), c1d08fc (Task 3)
