# Phase 26 — Conditional Gate C: Merge + Prune Re-Test (CONTEXT)

**Requirement:** GATE4-04. **Depends on:** Phase 25 (Gate B). **Disposition:** `no_winner` (ship merged-unpruned) is a valid, recorded outcome. Either disposition unblocks Phase 27.

## Where this comes from (routing from 25-02)

Phase 25 Gate B (MoE-Sieve k-sweep) closed **`no_winner`** (optimal_k=full) under the pre-registered two-sided CI-aware TOST (ε=2pp) vs the same-stack vLLM full arm (s1 rho 0.7935). The one non-inferior, point-better, fully-parseable arm was **k=224** (rho 0.8129, +0.0195, CI lower −0.018 clears −2pp, parse_fail 0/121). User signed off routing **(B):** Phase 26 SHOULD probe a prune anchored at k=224 rather than assume, confirming any candidate with the 3-seed ensemble before any publish call.

**Honest note (record in the plan):** k=224 keeps 224/256 experts = ~12.5% expert drop → ~33.6 GiB Q8, still above v3's 30.2 GiB. **This is no longer a ship disqualifier.**

## SHIP-DECISION POLICY UPDATE (user directive 2026-07-17) — ship v4, not v3

The canonical deliverable target is now **v4** (Qwen3.6-35B-A3B base), NOT a revert to v3. Rationale: the newer base is worth carrying even at a larger artifact size; v4 is at least a statistical tie with v3 on judge quality (0.8067 vs 0.8056). Therefore "beat v3's 30.2 GiB" is DEMOTED from a ship gate to an informational note.

Revised disposition for Gate C:
- **Prune PASSES the gate** (AIMER@k=224 non-inferior vs the same-stack full arm — CI lower clears −2pp — AND D2_security retained): **ship the PRUNED v4** (~33.6 GiB). Being above v3's 30.2 GiB is accepted.
- **Prune FAILS the gate**: **ship the merged-UNPRUNED v4** (37.8 GiB) — still v4, NOT a revert to v3. ("no_winner" now means "ship v4 unpruned," v3 is not the fallback.)

**UNCHANGED (hard bar, safety — no goalpost move):** gate-before-remove is still enforced in code; a pruned checkpoint is only produced if the gated eval records quality non-inferiority AND `pass_d2_security:true`. The machine TOST verdict is recorded as-measured. What changed is ONLY the ship target (v4 over v3) and the removal of the size-vs-v3 requirement — not any measurement or safety property.

Phase 27 packaging target flips to the chosen v4 artifact (pruned or unpruned); PROJECT.md/MODEL_CARD canonical model changes v3→v4 there (decided at Phase 27, recorded here).

## Locked scope

1. **Merge first.** Merge the v4 judge s1 LoRA into base weights before any pruning scoring (SC1). Checkpoint: `models/Qwen3.6-35B-A3B-judge-v4-s1-merged` (already the merged s1 — confirm it is the merge-of-record, not a re-merge, unless a fresh merge is required by the prune tooling's input contract).
2. **AIMER primary, REAP optional.** Score AIMER (primary) and optionally REAP (domain-aware comparison) at the k=224 compression point (and the protected mask from 25-01, `output/sieve-v4/protected_expert_mask.npy`), with **gate-before-remove**: NO physical weight removal until the gated eval passes. Per-dimension retention, **especially D2_security**, must be evaluated (SC2).
3. **3-seed ensemble confirmation.** Any prune candidate that passes the gate is confirmed with the s0/s1/s2 ensemble via the same-stack patched vLLM before a publish decision — mirrors the Gate B "ensemble reserved for the winner" discipline and the v3 Phase-11 precedent.
4. **Disposition (SC3):** a winning method+ratio ships pruned, OR `no_winner` ships the merged-unpruned model. Both valid; both unblock Phase 27.

## Reusable tooling (v3 precedent — adapt for 256 experts, do NOT re-derive)

v3 ran this exact gate at 128 experts and found `no_winner`. The scripts exist and are the starting point:
- `scripts/aimer_prune.py`, `scripts/reap_prune.py` — the scorers.
- `scripts/prune_gated_eval.py` — gate-before-remove eval driver.
- `scripts/prune_selection.py`, `scripts/prune_overlap.py`, `scripts/prune_apply_physical.py` — selection + physical surgery.
- v3 receipts for shape/format precedent: `output/prune/` (aimer_scores_judge.npy, selection.json, comparison_table.md, prune_methodology.md), `output/sieve/optimal_k.json`.
- Serving/eval: the same-stack patched vLLM (`scripts/_sieve_vllm_patch`) + the judge-eval harness used in 25-02 (`scripts/sieve_ksweep_v4_run.py` / eval_judge path). Same GB10 lesson: serve, don't load in-process (the 35B bf16 in-process load OOMs — `.planning/debug/resolved/v4-judge-load-oom-recurrence.md`).

## Threats to carry (from the milestone discipline)

- Gate-before-remove is load-bearing: never physically remove weights before D2_security + the gated eval pass (a false "prune succeeded" that drops security capability is the worst failure).
- Same-stack TOST reference (not llama.cpp Q8, not Tinker-capture) — the sanity_gate_recalibration lesson.
- No goalpost move: if a prune candidate is scored, the equivalence bar is the pre-registered one.
- Loader guard (T-25-01 carry): AutoModelForImageTextToText for any v4 load, 0 missing keys.

## Success criteria (from ROADMAP)

1. LoRA merged into base before pruning scoring.
2. AIMER (primary) + optionally REAP scored at the same compression discipline, per-dimension retention (esp. D2_security) via gate-before-remove.
3. Gate closes with a winning method+ratio (ships pruned) OR `no_winner` (ships merged-unpruned) — either unblocks Phase 27.
