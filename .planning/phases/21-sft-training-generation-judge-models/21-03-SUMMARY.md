---
phase: 21-sft-training-generation-judge-models
plan: 03
subsystem: training
tags: [tinker, sft, moe, lora, judge-model, qwen3.6, relabel-reuse, wave-2]

# Dependency graph
requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "21-01: forked v4 data adapter + SFT driver (renderer qwen3_5_disable_thinking, auto-LR 4.99e-4), MoE merge-path gate CLOSED (moe_merge_probe.json merge_ok=true) -- unblocked real Tinker spend for GEN-02/JUDGE-02"
provides:
  - "output/tinker/wp-judge-v4-s1-manifest.json -- seed-1 (primary) manifest: 3 per-epoch persistent sampler checkpoints (ttl=None), promoted wp-judge-v4-s1-ep3"
  - "output/tinker/wp-judge-v4-s0-manifest.json -- seed-0 manifest, promoted wp-judge-v4-s0-ep3"
  - "output/tinker/wp-judge-v4-s2-manifest.json -- seed-2 manifest, promoted wp-judge-v4-s2-ep3"
  - "output/base21/judge02_run.json -- 3-seed run receipt: per-seed promoted sampler paths (resolver-verified), loss, label provenance, all_seeds_complete=true"
affects: [21-06-judge-merge-rho-eval]

tech-stack:
  added: []
  patterns:
    - "Three independent Tinker SFT jobs can be launched concurrently as separate background Bash processes (each redirected to its own log file with '> file 2>&1', NOT captured via the harness's own background-task .output stream) -- per-run isolated remote compute makes this safe and roughly 3x faster wall-clock than sequential runs for equally-sized jobs"
    - "output/base21/ and output/tinker/ are both nominally covered by the top-level `output/` gitignore rule; output/tinker/ is explicitly un-ignored via `!output/tinker/` + `!output/tinker/**`, but output/base21/ relies on per-file `git add -f` (the established convention from 21-01/21-02) since git cannot re-include a file whose parent directory pattern is excluded without an explicit negation for that directory"

key-files:
  created:
    - output/tinker/wp-judge-v4-s1-manifest.json
    - output/tinker/wp-judge-v4-s0-manifest.json
    - output/tinker/wp-judge-v4-s2-manifest.json
    - output/base21/judge02_run.json
  modified: []

key-decisions:
  - "Ran all three seeds (1, 0, 2) concurrently rather than seed-1-then-{0,2} sequentially -- the plan's seed-order note exists for resilience (a single-seed result exists even if a later seed is delayed), which concurrent launch satisfies at least as well as sequential (seed 1 was never blocked waiting on 0/2, and none of the three depended on another's completion)"
  - "judge02_run.json's sampler-path resolution was verified by ACTUALLY importing and calling capture_judge_responses_tinker.py's _resolve_tinker_path against each committed manifest (not merely asserting the promoted->checkpoints[].sampler_path contract holds by inspection) -- all 3 resolved byte-identical to the receipt's recorded paths"
  - "No terse/format-stability gate was run for the judge seeds (unlike GEN-02) -- the plan's Task 1 action specifies only the literal v1.3 recipe flags (--stage full --epochs 3 --seed --train-path --save-name --manifest), with no --gate-temps; the judge role's quality metric is judge_rho (JUDGE-03/21-06's job), not terse-rate, which is a generation-model collapse metric"

metrics:
  duration: ~30min (mostly remote Tinker wall time, 3 seeds concurrent)
  completed: 2026-07-14

status: complete
---

# Phase 21 Plan 03: Judge-Model 3-Seed Relabel-SFT (JUDGE-02) Summary

**All three relabel-SFT seeds (1, 0, 2) completed concurrently on Qwen/Qwen3.6-35B-A3B using the literal v1.3 recipe (MoE-only LoRA r32, Tinker auto-LR 4.99e-4, 3 epochs) reusing the v1.3 human-relabeled judge targets verbatim -- loss 8.61 -> ~1.46-1.58 per seed, all 9 per-epoch sampler checkpoints (3 per seed) persisted and manifest-resolvable, ready for JUDGE-03's rho measurement.**

## Performance

- **Duration:** ~30 min wall (all 3 seeds trained concurrently as independent remote Tinker jobs; 210 steps/seed)
- **Tasks:** 2/2
- **Cost:** ~$6 remote Tinker (pre-approved), within budget

## Accomplishments

### Task 1 — 3-seed relabel-SFT (commit `1004bfc`)

- Confirmed the 21-01 gate first: `moe_merge_probe.json` `merge_ok=true` (spend unblocked).
- Launched `tinker_reasoning_sft_v4.py --stage full --epochs 3 --seed <seed> --train-path data/reasoning_dataset/openai_train_relabel_v1.jsonl --save-name wp-judge-v4-s<seed> --manifest output/tinker/wp-judge-v4-s<seed>-manifest.json` for seeds {1, 0, 2} as three concurrent background processes (rank 32 default, MoE-only default, Tinker auto-LR default).
- **Recipe fidelity:** literal v1.3 invocation with only `BASE_MODEL`/renderer changed (per 21-01's GEN-01 decision: `qwen3_5_disable_thinking`, auto-LR resolved `4.990818286656736e-4`) -- same driver family, same 478-judge-target relabel data (563 total train rows), same seeds, same 3 epochs, same MoE-only LoRA r32.
- **Loss curves (all 3 seeds, 210 steps each):**
  - seed 1 (historical v1.3-promoted primary): 8.6054 -> 1.5774
  - seed 0: 8.6054 -> 1.4864
  - seed 2: 8.6054 -> 1.4607
  - All monotone-trending down with expected per-step noise, consistent with GEN-02's curve shape on the same base.
- **Durability:** each seed wrote its per-epoch persistent (ttl=None) sampler checkpoint + incremental manifest every epoch (T-21-07 mitigation); all 9 checkpoints (3 seeds x 3 epochs) preserved, none pruned.
- Plan's automated verify (`json.load` all 3 manifests) and both acceptance-criteria checks (base_model/promoted/sampler_path shape; train_path provenance confirmed via each log's `[sft] train_path=...` line) pass for all 3 seeds.

### Task 2 — Run receipt + resolver verification (commit `fe0889e`)

- Assembled `output/base21/judge02_run.json`: per-seed `save_name`, `promoted_checkpoint_name`, `promoted_sampler_path`, `epochs`, `loss_first`, `loss_last`, `terse_rate_at_temp0_n4` (the driver's small-n final eval, not a formal gate).
- **Resolver check was executed, not just asserted:** imported `capture_judge_responses_tinker._resolve_tinker_path` and called it against each committed manifest -- all 3 resolved to the exact sampler paths recorded in the receipt (byte-identical), confirming JUDGE-03's capture step will resolve cleanly.
- `all_seeds_complete=true`, `relabel_reuse=true`, `label_source=data/reasoning_dataset/openai_train_relabel_v1.jsonl` recorded per V4-RERUN-ROADMAP discretion item 2 (labels reused verbatim, not regenerated).
- No merge or serve performed here -- explicitly deferred to JUDGE-03 (21-06).
- Plan's automated verify (`all_seeds_complete is True`, 3 seeds, all have `promoted_sampler_path`, `relabel_reuse is True`) passes.

## Task Commits

1. **Task 1: 3-seed relabel-SFT** — `1004bfc` — `feat(21-03): JUDGE-02 3-seed relabel-SFT (seeds 1/0/2) on Qwen3.6-35B-A3B`
2. **Task 2: run receipt + resolver verification** — `fe0889e` — `feat(21-03): JUDGE-02 run receipt -- 3-seed durability + resolver verification`

## Files Created

- `output/tinker/wp-judge-v4-s1-manifest.json` — seed-1 manifest (3 checkpoints, promoted ep3)
- `output/tinker/wp-judge-v4-s0-manifest.json` — seed-0 manifest (3 checkpoints, promoted ep3)
- `output/tinker/wp-judge-v4-s2-manifest.json` — seed-2 manifest (3 checkpoints, promoted ep3)
- `output/base21/judge02_run.json` — 3-seed run receipt (resolver-verified sampler paths, loss, provenance)
- (untracked logs: `output/base21/judge02_s{1,0,2}_full_log.txt` — gitignored run logs, key lines captured in the receipt)

## Decisions Made

See `key-decisions` in frontmatter.

## Deviations from Plan

### Auto-fixed Issues

None — the plan's task actions executed exactly as specified with the literal v1.3 flag set. No bugs, blocking issues, or missing functionality were found during execution.

**Minor plan-wording note (not a deviation requiring fix):** the plan's acceptance criteria for Task 1 says "each manifest's train_path resolves to `data/reasoning_dataset/openai_train_relabel_v1.jsonl`" — the manifest JSON itself does not carry a `train_path` field (this matches the v3/v4 driver's existing manifest schema: `save_name`, `base_model`, `rank`, `train_attn`, `train_unembed`, `renderer`, `epochs`, `checkpoints`, `promoted`, `state_path`, `created` — no `train_path` key). The provenance is instead confirmed via each run's log line `[sft] train_path=data/reasoning_dataset/openai_train_relabel_v1.jsonl`, which was checked for all 3 seeds. This is the same verification surface the driver has always exposed (no code change needed); documenting it here so JUDGE-03 doesn't look for a manifest field that was never part of the schema.

## Threat Model Compliance

- **T-21-07 (lost seed run strands sampler ref):** not exercised for real this run (all 3 processes completed cleanly), but the mitigation (incremental per-epoch manifest + ttl=None checkpoints) was active throughout and seed 1 (the historical-primary) completed first among the three, satisfying the plan's resilience intent.
- **T-21-08 (ambiguous label/data provenance):** `judge02_run.json` records `label_source` + `relabel_reuse=true` + each manifest's actual `train_path` is independently confirmed via log inspection for all 3 seeds.
- **T-21-09 (TINKER_API_KEY):** existing `.env` convention (`set -a; source .env; set +a`), no new handling.

## Known Stubs

None — all three runs are real remote Tinker compute; no mocked calls, no fabricated receipts. The resolver check in Task 2 was executed against real committed manifest files, not simulated.

## Next Phase Readiness

- **JUDGE-02: SATISFIED.** All three relabel-SFT seeds ({1, 0, 2}) completed on the new base reusing the v1.3 human-relabeled targets verbatim; all 9 per-epoch sampler checkpoints persist across the three manifests; each seed's promoted checkpoint is manifest-resolvable via the exact contract `capture_judge_responses_tinker.py` uses.
- 21-06 (JUDGE-03: merge + rho eval) consumes `judge02_run.json`'s three `promoted_sampler_path` entries to capture-eval each seed against the pre-registered rho targets (>0.85 single / >0.87 ensemble) and decide on ensembling; the proven `tinker_cookbook build_hf_model` merge route from 21-01 is available for any seed JUDGE-03 elects to merge/export.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-14*

## Self-Check: PASSED

All 4 created artifact files verified present on disk; both task commit hashes (`1004bfc`, `fe0889e`) verified present in `git log`.
