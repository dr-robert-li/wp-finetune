---
phase: 21-sft-training-generation-judge-models
plan: 02
subsystem: training
tags: [tinker, sft, moe, lora, gen-model, qwen3.6, terse-gate, wave-2]

# Dependency graph
requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "21-01: forked v4 data adapter + SFT driver (renderer qwen3_5_disable_thinking, auto-LR 4.99e-4), GEN-01 decision receipt, MoE merge-path gate CLOSED (moe_merge_probe.json merge_ok=true)"
provides:
  - "output/tinker/wp-gen-v4-manifest.json -- gen SFT manifest: 3 per-epoch persistent sampler checkpoints (ttl=None), promoted wp-gen-v4-ep3, fs_gate PASS (cot+ctf scope)"
  - "output/base21/gen02_run.json -- run receipt: smoke + full blocks, loss 7.97->1.46, terse gate both measurements"
  - "output/base21/gen02_fs_gate.json -- canonical REOPEN-scope terse gate receipt (temp0.0 0/120, temp0.7 3/360, PASS)"
affects: [21-05-gen-merge-wpbench-eval, 21-03-judge-sft]

tech-stack:
  added: []
  patterns:
    - "The v4 driver's in-driver terse gate scores ALL val rows including the 21 replay (code-gen) rows that legitimately carry no [/REASONING] -- a FAIL there is bounded by the replay fraction and must be re-scored with the canonical cot+ctf gate (scripts/tinker_fs_gate.py) before being treated as format collapse; the artifact is pre-documented in that script's docstring since the v1.2 run"
    - "tinker_fs_gate.py (v3-hardcoded imports) runs unmodified against the v4 base via sys.modules aliasing: sys.modules['tinker_reasoning_data'] = importlib.import_module('tinker_reasoning_data_v4')"
    - "TINKER_API_KEY is not auto-loaded from .env by the SFT drivers -- export it (set -a; source .env) before any tinker.ServiceClient() call"

key-files:
  created:
    - output/base21/gen02_run.json
    - output/base21/gen02_fs_gate.json
    - output/tinker/wp-gen-v4-manifest.json
    - output/tinker/wp-gen-v4-smoke-manifest.json
  modified: []

key-decisions:
  - "In-driver terse gate FAIL (20/141 @temp0) dispositioned as a KNOWN measurement artifact, not format collapse: verified all 21 no-[/REASONING] val targets are stream=replay; canonical REOPEN-scope (cot+ctf) gate re-scored standalone on the persisted ep3 checkpoint -> PASS both arms (0/120 @temp0.0, 3/360 @temp0.7). Both measurements recorded; nothing forced or silently retried."
  - "Epoch count 3 confirmed from the historical v1.2 gen manifests (wp-reasoning-v2/v3 both epochs=3), not hardcoded from the driver default"
  - "In-driver temp0.7 gate arm never ran (process killed mid-arm after training completed) -- no re-run of training needed: the per-epoch incremental manifest + ttl=None checkpoints (the P4 durability fix) meant nothing was stranded, and the standalone gate covered both temps on the persisted checkpoint"

metrics:
  duration: ~95min (mostly remote Tinker wall time)
  completed: 2026-07-14

status: complete
---

# Phase 21 Plan 02: Generation-Model SFT (GEN-02) Summary

**Full gen SFT completed on Qwen3.6-35B-A3B with the literal v1.2 recipe (MoE-only LoRA r32, frozen router, Tinker auto-LR 4.99e-4, 3 epochs on the reused reasoning-mix): loss 7.97 -> 1.46 monotone per-epoch, all 3 per-epoch sampler checkpoints persisted, and the terse format-stability gate PASSED on the canonical cot+ctf scope after the in-driver full-val FAIL was traced to the pre-documented replay-row measurement artifact.**

## Performance

- **Duration:** ~95 min wall (smoke ~4 min, full train ~37 min for 210 steps, gate arms ~50 min sampling)
- **Tasks:** 2/2
- **Cost:** ~$2 remote Tinker (pre-approved), within budget

## Accomplishments

### Task 1 — Smoke pre-flight on the REAL reasoning-mix (commit `6bfe1a3`)

- Confirmed the 21-01 gate first: `moe_merge_probe.json` `merge_ok=true` (spend unblocked).
- `tinker_reasoning_sft_v4.py --stage smoke --max-steps 4` against `data/reasoning_dataset/openai_train.jsonl` (real mix, not the 21-01 probe's toy example): 70 train batches tokenized/fed cleanly, losses 7.97/6.24/5.32/6.24 (trending down), persistent sampler checkpoint exported (`wp-gen-v4-smoke-ep1`).
- Receipt: `gen02_run.json` smoke block; plan's automated verify passed.

### Task 2 — Full gen SFT + terse gate + manifest (commit `6f708b2`)

- **Recipe fidelity:** 3 epochs confirmed against the historical v1.2 manifests (`wp-reasoning-v2/v3` both `epochs=3`); rank 32; MoE-only (`train_mlp=True`, `train_attn=False`, `train_unembed=False` — D-N1, frozen router by omission); auto-LR resolved 4.99e-4; renderer `qwen3_5_disable_thinking` (GEN-01 decision).
- **Loss curve:** 7.973 -> 1.458 over 210 steps, monotone per-epoch (ep1 end ~4.5, ep2 end ~2.5, ep3 end ~1.46).
- **Durability:** per-epoch persistent (ttl=None) sampler checkpoints + incremental manifest written every epoch — proven necessary this run: the driver process was killed mid-gate, yet nothing was stranded.
- **Terse gate (the REVL-05 collapse guard):**
  - In-driver full-val temp0.0 arm read **FAIL 20/141 (0.1418)**. Investigated before disposition: exactly 21/141 val targets carry no `[/REASONING]`, and **all 21 are `stream=replay`** (wp_gen code-generation rows whose targets are raw PHP by design). The 20-terse count is bounded by that set — the model reproducing replay-style output on replay prompts is trained behavior, not collapse. This exact artifact is pre-documented in `scripts/tinker_fs_gate.py`'s docstring ("the in-driver gate ... wrongly counts the 21 replay rows as terse — the REOPEN terse metric is defined on cot+ctf ONLY").
  - Canonical cot+ctf gate re-scored standalone on the persisted `wp-gen-v4-ep3`: **temp0.0 = 0/120 (rate 0.0000, Wilson-upper 0.031) PASS; temp0.7 = 3/360 (rate 0.0083, Wilson-upper 0.024) PASS.** Far inside the pre-registered thresholds (rate<=0.10, Wilson<=0.15).
  - Both measurements recorded verbatim in `gen02_run.json` and the manifest's `fs_gate` (canonical result + `in_driver_full_val_arm` artifact record). No silent retry, no forced pass.
- Plan's automated verifies (receipt fields, manifest shape, base_model, `loss_last <= loss_first`) all pass.

## Task Commits

1. **Task 1: smoke pre-flight** — `6bfe1a3` — `feat(21-02): GEN-02 smoke -- v4 driver validated on real gen reasoning-mix`
2. **Task 2: full SFT + terse gate + manifest** — `6f708b2` — `feat(21-02): GEN-02 full gen SFT on Qwen3.6-35B-A3B -- loss 7.97->1.46, terse gate PASS (cot+ctf)`

## Files Created

- `output/tinker/wp-gen-v4-manifest.json` — gen manifest: 3 per-epoch sampler paths, promoted `wp-gen-v4-ep3`, `fs_gate` (canonical PASS + in-driver artifact record)
- `output/tinker/wp-gen-v4-smoke-manifest.json` — smoke checkpoint manifest
- `output/base21/gen02_run.json` — run receipt (smoke + full blocks)
- `output/base21/gen02_fs_gate.json` — canonical gate receipt
- (untracked logs: `output/base21/gen02_full_log.txt`, `gen02_smoke_log.txt`, `gen02_fs_gate_log.txt` — gitignored run logs, key lines captured in the receipt)

## Decisions Made

See `key-decisions` in frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug/measurement artifact] In-driver terse gate FAIL re-scored with the canonical cot+ctf gate**
- **Found during:** Task 2 gate evaluation
- **Issue:** The plan specified the driver's `--gate-temps 0.0,0.7 --gate-n 300` gate, but the in-driver gate scores ALL 141 val rows, wrongly counting the 21 replay (code-gen) rows — whose targets legitimately contain no `[/REASONING]` — as "terse". temp0.0 arm read FAIL 20/141.
- **Fix:** Verified the artifact empirically (all 21 no-`[/REASONING]` val targets are `stream=replay`), then ran the project's pre-existing canonical gate `scripts/tinker_fs_gate.py` (cot+ctf scope, the REOPEN pre-registered metric definition) standalone against the persisted ep3 checkpoint via `sys.modules` aliasing to the v4 data module. PASS both arms. Recorded BOTH measurements; the plan's "FAIL is a recorded, valid outcome" clause was honored — the FAIL is recorded, with its root cause, alongside the correctly-scoped measurement.
- **Files modified:** none (no code changes; standalone re-score)
- **Commit:** `6f708b2`

**2. [Rule 3 - Blocking] In-driver temp0.7 gate arm lost to a process kill — recovered from persisted checkpoints**
- **Found during:** Task 2, post-training
- **Issue:** The detached driver process was killed after training + the temp0.0 gate arm completed but before the temp0.7 arm and the manifest `fs_gate` write.
- **Fix:** No training re-spend needed — the per-epoch incremental manifest + ttl=None sampler checkpoints (T-21-04's mitigation, already in the driver) preserved everything. The standalone gate run covered both temps on the persisted promoted checkpoint; the manifest `fs_gate` was patched from the canonical receipt.
- **Commit:** `6f708b2`

## Threat Model Compliance

- **T-21-04 (stranded sampler refs):** exercised for real — the process kill mid-gate stranded nothing; per-epoch incremental manifest + ttl=None checkpoints held.
- **T-21-05 (format collapse silently ships):** gate measured across temps 0.0/0.7 and recorded, including the raw in-driver FAIL and its disposition — nothing papered over.
- **T-21-06 (TINKER_API_KEY):** existing .env convention, no new handling.

## Known Stubs

None — all runs are real remote Tinker compute; no mocked calls, no fabricated receipts.

## Next Phase Readiness

- **GEN-02: SATISFIED.** Gen adapter trained on the new base with the v1.2 recipe; decreasing loss; all 3 per-epoch sampler checkpoints persist in `wp-gen-v4-manifest.json` for GEN-03 (21-05) to select among; terse gate measured and PASSED on the canonical scope.
- 21-05 (GEN-03 merge + wp-bench eval) consumes the manifest's `per_epoch_sampler_paths` and the proven `tinker_cookbook build_hf_model` merge route from 21-01.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-14*

## Self-Check: PASSED

All 4 created artifact files + SUMMARY verified present on disk; both task commit hashes (`6bfe1a3`, `6f708b2`) verified present in `git log`.
