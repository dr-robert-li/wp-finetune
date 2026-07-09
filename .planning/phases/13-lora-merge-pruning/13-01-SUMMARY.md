---
phase: 13-lora-merge-pruning
plan: 01
subsystem: infra
tags: [moe, pruning, aimer, safetensors, qwen3-30b-a3b, transformers]

requires:
  - phase: 11-sieve-expert-masking
    provides: scripts/sieve_expert_mask_inference.py (build_ksweep_mask/apply_mask, reused unchanged downstream)
  - phase: 12-rl-training
    provides: RL-CLOSED decision (no RL LoRA produced, commit 8860e89) — informs MERGE-01 closure
provides:
  - MERGE-01 closed as a traceability record (no new merge code)
  - scripts/aimer_prune.py — AIMER weight-norm expert scorer (PRUNE-01), sharded per-expert safetensors read
  - output/prune/aimer_scores_gen.npy, output/prune/aimer_scores_judge.npy — [48,128] float32 score arrays
affects: [13-04-gate-before-remove, 13-05-selection, 13-06-physical-pruning]

tech-stack:
  added: []
  patterns:
    - "Streamed safetensors scoring: resolve each tensor key through model.safetensors.index.json weight_map, read one tensor at a time via safe_open, accumulate scalar (P,N,Q) running sums — avoids ever holding a full model in memory"
    - "Multi-checkpoint CLI --checkpoint nargs='+' with elementwise MEAN for a shared judge profile (mirrors scripts/sieve_expert_mask_inference.py's summed multi-report convention)"

key-files:
  created:
    - scripts/aimer_prune.py
    - tests/test_aimer_prune.py
    - output/prune/aimer_scores_gen.npy
    - output/prune/aimer_scores_judge.npy
    - .planning/phases/13-lora-merge-pruning/MERGE-01-TRACEABILITY.md
  modified: []

key-decisions:
  - "On-disk tensor layout differs from 13-RESEARCH's stacked-tensor skeleton: keys are per-expert UNSTACKED (model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight), sharded across 13 files — verified directly against model.safetensors.index.json before implementing, per plan's A2 caveat"
  - "Judge AIMER score = elementwise MEAN across the 3 judge seed checkpoints (not sum), matching the shared judge routing profile precedent (cross-seed Jaccard 0.933)"
  - "MERGE-01 closed via documentation only — no merge code executed, since Phase 12 rejected RL (no LoRA) and Phase 11's Sieve was training-free (no LoRA)"

patterns-established:
  - "Pattern: score any per-expert weight-based metric by streaming shard-by-shard through safetensors.safe_open with a key->shard lookup from the index, never materializing more than one tensor at a time — reusable for any future weight-only MoE analysis (e.g. PRUNE-02 REAP's calibration path still needs a forward pass, but non-calibration weight metrics should follow this streaming pattern)"

requirements-completed: [MERGE-01, PRUNE-01]

coverage:
  - id: D1
    description: "MERGE-01 closed as a traceability record — all 4 merged checkpoints (gen + 3 judge seeds) verified to load with num_local_experts=128, num_hidden_layers=48, and an index.json present; no unmerged adapters remain"
    requirement: "MERGE-01"
    verification:
      - kind: unit
        ref: ".venv-tinker/bin/python -c \"AutoConfig.from_pretrained(...)\" one-liner over all 4 checkpoint paths"
        status: pass
    human_judgment: false
  - id: D2
    description: "AIMER weight-norm scorer (score = P/sqrt(N*Q)) implemented as a streaming safetensors reader; produces real [48,128] score arrays for gen and the 3-seed judge mean"
    requirement: "PRUNE-01"
    verification:
      - kind: unit
        ref: "tests/test_aimer_prune.py#test_shape_and_finite, test_scale_invariance, test_determinism, test_bounded_in_unit_interval, test_missing_key_raises"
        status: pass
      - kind: unit
        ref: "scripts/aimer_prune.py --self-check"
        status: pass
      - kind: other
        ref: "output/prune/aimer_scores_{gen,judge}.npy shape (48,128), all finite (verified via numpy assert one-liner)"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-07-10
status: complete
---

# Phase 13 Plan 01: MERGE-01 Closure + AIMER Weight-Norm Scorer Summary

**AIMER weight-norm scorer streams sharded per-expert safetensors (13 shards, unstacked keys) to produce real [48,128] importance scores for gen and the 3-seed judge ensemble mean, with MERGE-01 closed as a zero-new-code traceability record.**

## Performance

- **Duration:** ~30 min (includes two full-checkpoint AIMER scoring runs, ~1 min for gen, ~3 min for the 3 judge seeds)
- **Completed:** 2026-07-10
- **Tasks:** 2/2
- **Files modified:** 5 (2 created code files, 2 generated score arrays, 1 doc)

## Accomplishments
- Verified all 4 Phase 13 checkpoints (gen v1.2 + 3 judge seeds) load correct MoE config (128 experts, 48 layers) and closed MERGE-01 with a traceability note — no merge code ran, since RL was rejected (no LoRA) and Sieve was training-free (no LoRA)
- Implemented `scripts/aimer_prune.py::compute_aimer_scores` — the AIMER formula (P/sqrt(N*Q)) computed by streaming each expert's gate/up/down tensors from its owning shard, one tensor at a time, never loading the ~60GB full model
- Discovered and corrected for a real on-disk layout mismatch vs. 13-RESEARCH's assumed skeleton: tensors are per-expert UNSTACKED (not stacked `gate_up_proj`/`down_proj`) and sharded across 13 files — resolved via `model.safetensors.index.json` weight_map
- Produced real `output/prune/aimer_scores_gen.npy` and `output/prune/aimer_scores_judge.npy` (elementwise mean of 3 judge seeds), both [48,128] float32, all finite — the score arrays every downstream gating/selection step (13-04) will consume

## Task Commits

Each task was committed atomically:

1. **Task 1: MERGE-01 traceability record** - `aef7086` (docs)
2. **Task 2: AIMER scorer tests** - `dd37261` (test — RED, module-level importorskip skips cleanly per repo convention until implementation lands)
3. **Task 2: AIMER scorer implementation** - `9495534` (feat — GREEN, 5/5 tests pass, self-check exits 0, real score arrays generated)

_TDD gate sequence for Task 2: test(dd37261) -> feat(9495534), confirmed in git log._

## Files Created/Modified
- `.planning/phases/13-lora-merge-pruning/MERGE-01-TRACEABILITY.md` - closes MERGE-01, names all 4 checkpoint paths, cites `scripts/merge_adapter.py` as gen's producing tool, records that no RL/Sieve LoRA exists to merge
- `scripts/aimer_prune.py` - AIMER weight-norm scorer: `compute_aimer_scores(checkpoint_dir)`, CLI (`--checkpoint` 1+, `--out`), `--self-check`
- `tests/test_aimer_prune.py` - synthetic 2-shard on-disk fixture; shape, finiteness, scale-invariance, determinism, [1/sqrt(N),1] bound, missing-key-raises (T-13-01)
- `output/prune/aimer_scores_gen.npy` - gen checkpoint AIMER scores, [48,128] float32, min=0.599 max=0.797 mean=0.791
- `output/prune/aimer_scores_judge.npy` - mean of 3 judge seed AIMER scores, [48,128] float32, min=0.598 max=0.797 mean=0.791 (correlation with gen scores: 0.9998)

## Decisions Made
- On-disk tensor layout verified directly (not assumed from 13-RESEARCH's skeleton): per-expert unstacked keys `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`, sharded across 13 files — implementation reads via `model.safetensors.index.json` weight_map resolution, per plan's explicit instruction (A2 caveat)
- Judge score = elementwise MEAN (not sum) across the 3 judge seed checkpoints, matching the `sieve_profile_mode=shared` precedent from Phase 11 (cross-seed Jaccard 0.933)
- `output/prune/*.npy` force-added to git (`git add -f`) despite the blanket `output/` `.gitignore` rule — matches established repo precedent (many other `output/*` artifacts, e.g. `output/sieve/prune_set_for_phase13.json`, are already tracked the same way) and fulfills the plan's `files_modified` contract for these two artifacts
- No REAP work in this plan — PRUNE-02 REAP scoring is a separate plan per 13-CONTEXT's recommendation to gate REAP on AIMER@25% passing first

## Deviations from Plan

None - plan executed exactly as written. The plan itself pre-registered the tensor-layout deviation from 13-RESEARCH's skeleton (A2 caveat) and instructed the corrected implementation directly; following that instruction is not a deviation.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Threat Flags

None - all file I/O stays within the existing checkpoint safetensors trust boundary (T-13-01 mitigation implemented: missing/renamed expert keys raise `KeyError`, never silently produce a zero score, verified by `test_missing_key_raises`).

## Next Phase Readiness

- `output/prune/aimer_scores_gen.npy` and `output/prune/aimer_scores_judge.npy` are ready for 13-04's `build_ksweep_mask()` reuse (ranking array swapped from routing counts to AIMER scores, k parameter unchanged as the per-layer keep-count for 25/50/75% compression)
- Protected mask (`output/sieve/prune_set_for_phase13.json`) verified byte-unchanged (sha `14e8f25366044fdc4fd3daa0dd549bf3d3022de0d8c5d4c7f7e8bbbd84ac07ef`) before and after this plan's execution — no writes touched it
- No blockers for 13-02/13-03 (REAP, overlap analysis) or 13-04 (gate-before-remove)

---
*Phase: 13-lora-merge-pruning*
*Completed: 2026-07-10*

## Self-Check: PASSED

All created files exist on disk; all 3 task commit hashes (`aef7086`, `dd37261`, `9495534`) found in git log.
