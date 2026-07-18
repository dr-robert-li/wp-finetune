---
phase: 26-conditional-gate-c-merge-prune-re-test
plan: 02
subsystem: moe-prune
tags: [aimer, prune, physical-surgery, gate-before-remove, v4-judge, ensemble, ship-decision]
status: complete
requires:
  - output/prune-v4/gated/aimer_224_judge.json
  - output/prune-v4/gated/aimer_224_d2.json
  - output/prune-v4/masks/aimer_k224.npy
  - models/Qwen3.6-35B-A3B-judge-v4-s1-merged
provides:
  - scripts/prune_apply_physical_v4.py
  - models/Qwen3.6-35B-A3B-judge-v4-pruned-k224
  - output/prune-v4/gated/aimer_224_pruned_validation.json
  - output/prune-v4/gated/aimer_224_ensemble.json
  - output/prune-v4/selection_v4.json
affects: [phase-27]
tech_stack:
  added: []
  patterns: [stacked-tensor-axis0-slice, gate-before-remove-guard, serve-not-load, same-stack-tost]
key_files:
  created:
    - scripts/prune_apply_physical_v4.py
    - output/prune-v4/gated/aimer_224_pruned_validation.json
    - output/prune-v4/gated/aimer_224_ensemble.json
    - output/prune-v4/selection_v4.json
  modified:
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
key_decisions:
  - "Disposition: ship_pruned_v4 — the pruned k=224 v4 judge ships. Human sign-off granted (relayed via orchestrator)."
  - "Ship criterion is routing-(B) NON-INFERIORITY (ci_lower >= -2pp + D2_security + protected retained), NOT two-sided TOST equivalence. Locked at the 25-02 sign-off, reaffirmed at 26-02."
  - "Size-vs-v3 demoted from a ship gate to an informational note (2026-07-17 user directive): the newer Qwen3.6 base is preferred over v3's smaller artifact. Canonical flips v3 -> v4."
requirements-completed: [GATE4-04]
---

## Accomplishments

**Physical surgery (Task 1).** `scripts/prune_apply_physical_v4.py` — stacked-tensor axis-0 slice of the v4 judge's expert tensors from 256 → 224 per layer using the uniform k=224 AIMER mask. `shared_expert.*` / `shared_expert_gate.weight` / `mtp.layers.*` untouched; `text_config.num_experts` rewritten to 224 (NOT `num_local_experts` — the v3 key does not exist on this arch). Prefix resolved from `model.safetensors.index.json` weight_map rather than hardcoded. **Gate-before-remove is code-enforced**: `main()` hard-asserts the gate record before any `safe_open` write; the self-check proves the guard raises on a missing/false record. Output: `models/Qwen3.6-35B-A3B-judge-v4-pruned-k224` (60 GB bf16, from 67 GB). Commits `f4c9eca`, `53bd8d3`.

**Pruned-checkpoint validation (Task 2).** The physically-pruned checkpoint served **unmasked** (native 224 experts) through the same-stack patched vLLM: **pruned s1 rho 0.8134**, delta **−0.005** vs the masked-serve proxy (0.8184), `coherent: true`, parse_fail 1/120. The surgery removed the intended tensors — the pruned model reproduces the masked routing within noise and generates coherently.

**3-seed ensemble confirmation (Task 2).** s0/s1/s2 captured on the pruned checkpoint: **ensemble rho 0.8533** (s0 0.7653, s1 0.8184, s2 0.8264), n=121. Confirms the pruned checkpoint ensembles without collapse.

**Disposition + sign-off (Task 3).** `output/prune-v4/selection_v4.json` → **`ship_pruned_v4`**. Blocking human sign-off granted (approve surgery → ship pruned v4).

## Honest reading of the numbers — what is and is NOT claimed

- **The ship rests on the like-for-like s1-vs-s1 comparison**: pruned 0.8134 / masked 0.8184 vs the same-stack full arm 0.7935 → **+0.020, non-inferior, point-better**. Same seed, same stack, apples-to-apples.
- **The 0.8533 ensemble is CONFIRMATORY, not a +6pp pruning gain.** `ensemble_non_inferior` was computed against the **single-seed** full arm (0.7935); 3-seed ensembling lifts rho on its own, so the delta is **not attributable to pruning**. No pruning-attributable gain is claimed from it. Its job was to prove no collapse — it did.
- **The non-inferiority margin is THIN**: ci_lower −0.019 against the −0.020 bar → slack **0.001** at n=120. An unluckier bootstrap would have flipped it. Recorded, not smoothed. This thinness is the main caveat on the ship.
- **Two-sided TOST `equivalent: false` is kept as-measured** and was never flipped. It fails on the *upper* bound (the arm may be >2pp better), not because the arm is worse.
- **Stack caveat**: 0.8533 / 0.8134 are **bf16-vLLM**. v3's 0.8056 and v4's 0.8067 are **Q8 GGUF/llama.cpp**. Not comparable until Phase 27's Q8 conversion measures the pruned v4 on the shipped stack.
- **Size**: pruned v4 → ~33.6 GiB Q8 (projected, linear-scaling), still **~3.4 GiB larger than v3's 30.2 GiB**. Per the 2026-07-17 directive this is not a disqualifier — the newer base is preferred. Real Q8 size measured at Phase 27.

## Deviations from Plan

1. **Ship criterion = non-inferiority, not two-sided TOST** (relayed user directive). The plan gated on two-sided equivalence; the pre-registered routing-(B) bar is non-inferiority. Added a distinct code-computed `pass_ship` while preserving the measured `pass:false` / `equivalent:false`. The surgery guard asserts `pass_ship` + D2. Not a goalpost move — non-inferiority was the criterion chosen at the 25-02 sign-off.
2. **Ship target changed mid-execution** (2026-07-17 user directive): ship **v4**, not a revert to v3. "no_winner" would have shipped v4-unpruned, not v3. Moot here — the gate passed and the pruned v4 ships.
3. **REAP deferred** — per plan/research, gated on AIMER passing first; AIMER passed and shipped, so REAP was not needed (YAGNI, matching v3's PRUNE-02 precedent).
4. **Executor discontinuity**: the executing subagent's transcript was lost after the ensemble completed; the orchestrator wrote `selection_v4.json`, this SUMMARY, and the state updates directly from the on-disk receipts. All numbers here are read from committed artifacts, none reconstructed from memory.

## Verification

- `prune_apply_physical_v4 --self-check`: stacked axis-0 slice correct, shared_expert/mtp untouched, guard raises without a pass record — PASS.
- Gate-before-remove: pruned checkpoint exists ⟹ `pass_ship:true AND pass_d2_security:true` were recorded first (guard-enforced).
- Pruned checkpoint: `coherent:true`, rho 0.8134, delta −0.005 vs masked — surgery correct.
- Ensemble: 3/3 seeds captured, n=121, no collapse.
- D2_security: retention 6.326 ≥ baseline 6.115 — PASS (no security regression).

## Next

**Phase 27 — Packaging & Publication Refresh.** Package the **pruned v4** (`models/Qwen3.6-35B-A3B-judge-v4-pruned-k224`): Q8 GGUF conversion (measures the real size vs the 33.6 GiB projection), cascading compression gates, and the **operator-only HF model card** + v4 canonical lineage. Spec locked in `.planning/phases/27-packaging-publication-refresh/CONTEXT.md`.
