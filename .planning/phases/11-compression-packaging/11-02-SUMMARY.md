---
phase: 11-compression-packaging
plan: 02
subsystem: judge-ensemble-packaging
tags: [tinker-export, moe-merge, judge-seeds, sieve]
requires:
  - models/_staging/qwen3-30b-wp-v1.3-merged/ (s1 ground-truth shape)
  - output/tinker/wp-reasoning-relabel-v1-full-manifest.json (s0 sampler)
  - output/tinker/wp-reasoning-relabel-s2-manifest.json (s2 sampler)
  - scripts/tinker_export_checkpoint.py
  - scripts/merge_tinker_v3.py
provides:
  - models/_staging/qwen3-30b-wp-v1.3-s0-merged/ (13-shard bf16 judge seed s0)
  - models/_staging/qwen3-30b-wp-v1.3-s2-merged/ (13-shard bf16 judge seed s2)
  - output/sieve/merge_s0_report.json
  - output/sieve/merge_s2_report.json
affects:
  - plan 11-03 (routing profile of the 3-seed judge ensemble)
tech-stack:
  added: []
  patterns: [tinker-per-expert-moe-merge, retry-wrapper-for-archive-pack-timeout]
key-files:
  created:
    - scripts/_11_02_export_retry.sh
    - models/_staging/qwen3-30b-wp-v1.3-s0-merged/ (gitignored)
    - models/_staging/qwen3-30b-wp-v1.3-s2-merged/ (gitignored)
    - output/sieve/merge_s0_report.json (gitignored)
    - output/sieve/merge_s2_report.json (gitignored)
    - models/tinker_export/s0/ (gitignored)
    - models/tinker_export/s2/ (gitignored)
  modified: []
decisions:
  - "Reused scripts/merge_tinker_v3.py and tinker_export_checkpoint.py verbatim (path args only); no merge-algorithm change"
  - "Added a bash retry wrapper (not an SDK change) for the Tinker archive-pack client timeout; s2 succeeded on try 4"
metrics:
  duration: ~55min (dominated by s2 server-side archive packing, 4 export tries)
  completed: 2026-07-08
status: complete
requirements: [SIEVE-01]
---

# Phase 11 Plan 02: Export + Merge Judge Seeds s0/s2 Summary

Exported the two unmerged judge seeds (s0, s2) from Tinker and merged each into a full
13-shard bf16 checkpoint byte-identical in layout/config to the already-present s1 merge,
so all three ensemble members are locally servable for the routing profiling in plan 11-03.

## What was done

**Task 1 — Export s0/s2 sampler adapters.** Ran `scripts/tinker_export_checkpoint.py`
(via a new retry wrapper) for each seed. s0 was already present from a prior session;
s2 was re-launched and succeeded on export try 4 (the Tinker archive-pack job runs
server-side and times the client out mid-pack — a fresh request after backoff picks up
the by-then-ready signed URL). Both tars are 1849.7 MB with a valid `adapter_config.json`.
s1 was NOT re-exported.

- `models/tinker_export/s0/checkpoint.tar` — 3 members, tar OK
- `models/tinker_export/s2/checkpoint.tar` — 3 members, tar OK

**Task 2 — Merge to 13-shard bf16 + certify convention.** Ran `scripts/merge_tinker_v3.py`
once per seed with the same flags as the v1.3 s1 merge (per-expert MoE-only delta, no attn,
no lm_head — auto-detected `is_moe_only`). Sequential merges (never concurrent) to respect
the GB10 ~100 GiB peak; RAM was ~116-118 GiB free throughout.

- s0: 117.9s, 13 shards, differ w1=0.0159 / w3=0.0161 / w2=0.0061
- s2: 133.4s, 13 shards, differ w1=0.0145 / w3=0.0125 / w2=0.0066
- Both reports: `gate_up_touched=6144`, `down_touched=6144` (nonzero)
- Merge-convention test suite green (7 passed) — per-expert byte-exact extraction
  certified, so no broadcast-bug corruption entered the judge ensemble
- Both outputs: file listing + `config.json` byte-identical to the s1 ground truth
  (`models/_staging/qwen3-30b-wp-v1.3-merged/`)

## Verification

- `pytest tests/phase4_4/test_tinker_merge_convention.py -x` → 7 passed (T-11-03 mitigated)
- s0 and s2 each contain exactly 13 `model-*.safetensors` shards + `config.json`
- merge reports record nonzero gate_up/down touched counts
- Phase-7 `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` untouched
  (mtime unchanged, 2026-06-15) — T-11-05 mitigated
- export tars re-validated with `tarfile.is_tarfile` + `adapter_config.json` presence
  (T-11-04 mitigated); both at the full 1849.7 MB size (no truncated download)

## Deviations from Plan

**1. [Rule 3 - Blocking] Tinker archive-URL request times out mid server-side pack.**
- **Found during:** Task 1 (s2 export)
- **Issue:** `get_checkpoint_archive_url_from_tinker_path` raised `tinker.APITimeoutError`
  while the server was still packing the archive; the SDK's internal retry budget is
  shorter than the pack time for this checkpoint.
- **Fix:** Added `scripts/_11_02_export_retry.sh` — a bash retry loop (MAX_TRIES=10,
  120s backoff) around the unmodified export script. NOT an SDK/export-script code change.
  Precedent: `logs/relabel_sft/export_v13_retry.log`. s2 succeeded on try 4.
- **Files added:** scripts/_11_02_export_retry.sh
- **Commit:** Task-1 commit

No architectural changes; no new packages (T-11-SC held).

## Notes for plan 11-03

- The 3 judge seeds are now all merged 13-shard bf16 checkpoints under `models/_staging/`:
  s0 = `qwen3-30b-wp-v1.3-s0-merged`, s1 = `qwen3-30b-wp-v1.3-merged`,
  s2 = `qwen3-30b-wp-v1.3-s2-merged`. Serve SEQUENTIALLY (GB10 memory wall + Tinker MoE
  LoRA is not a runtime vLLM adapter), not concurrent multi-LoRA.
- All merge artifacts carry `status: staging_written_pending_anchor` — routing profiling
  can read them, but do not promote to canonical until anchors pass.

## Self-Check: PASSED

- FOUND: models/_staging/qwen3-30b-wp-v1.3-s0-merged/ (13 shards + config.json)
- FOUND: models/_staging/qwen3-30b-wp-v1.3-s2-merged/ (13 shards + config.json)
- FOUND: output/sieve/merge_s0_report.json
- FOUND: output/sieve/merge_s2_report.json
- FOUND: models/tinker_export/s0/checkpoint.tar, models/tinker_export/s2/checkpoint.tar
- FOUND commit (Task 1): export s0/s2 judge-seed adapters
- FOUND commit (Task 2): merge s0/s2 into 13-shard bf16 judge checkpoints
