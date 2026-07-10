# PIPELINE.md — the repeatable wp-finetune method

This is the frozen, end-to-end pipeline that produced the WordPress generation + review model pair from
Qwen3-30B-A3B. It is written so you can run the same method on a fresh same-architecture base (for example
a future Qwen3.6-30B-class MoE with task-token routing) and get a comparable or better result.

The pipeline has a spine (data → train → merge → eval) and three **conditional compression gates** (RL,
MoE-Sieve expert-drop, weight-level prune) that each returned *no improvement* on Qwen3-30B-A3B. They are
kept, not deleted. Each is a gate with a known result on this base and an open question on the next one. A
higher-rho, more concentrated base may flip any of them. Run the gate; don't assume the verdict.

Known results below are from the Qwen3-30B-A3B run (milestone v3.0, 2026-07). Replace them with your base's
numbers as you go.

---

## What it produces

A two-model pair sharing one MoE base, routed by task token:

- `<wp_gen>` — generates WPCS-compliant WordPress PHP.
- `<wp_judge>` — reviews code against a 9-dimension rubric and explains defects.

## Prerequisites

- A task-token MoE base (Qwen3-30B-A3B here: 128 experts, top-8, ~30.5B total / ~3.3B active).
- LoRA training backend. This project used Tinker (cloud LoRA); the trainer has no `load_state`, so each
  SFT stage is a fresh LoRA-from-base, not a weight-continuation.
- A GPU host for serving/eval. This project used a DGX Spark GB10 (121 GB unified memory) and served
  through vLLM containers via `scripts/dgx_toolbox.py` (`dgx.execute("vllm", ...)`), not local
  transformers, because a local 30B load hits the unified-memory wall.
- Judge labels: LLM-as-judge relabeling used in-session Claude Code agents (subscription-billed), never
  direct paid API loops. See `CLAUDE.md` billing policy.

---

## Stage 1 — Data pipeline (Phases 1-2)

Build the training corpus: curate WordPress repos, extract PHP functions, generate positive and negative
examples (real fails, programmatic mutations, synthetic contrastive pairs), judge them, add chain-of-thought,
and export train/val splits.

- Entrypoint: `wp-finetune:run-data-pipeline` skill (orchestrates the `scripts/` data stages).
- Output: `data/reasoning_dataset/` (SFT targets), `data/relabel_v1/` (human-relabeled judge labels).
- Gate: dataset assembly is a hard gate — examples whose reasoning contradicts their numeric score are
  rejected before the mix is built.
- Qwen3-30B-A3B result: 34,855 training examples; relabel campaign 603/603 items labeled, M=3 median
  aggregation, pilot QC gates passed (reliability 0.969, κ 0.623).

## Stage 2 — SFT: generation model (Phases 3, 4.3)

Fine-tune the base for generation with a reasoning mix (reasoning examples + 30% judge replay + 20% wp_gen
replay to prevent format collapse and codegen regression).

- Entrypoint: `wp-finetune:run-training` skill -> `scripts/tinker_reasoning_sft.py`.
- Config: MoE-only LoRA (rank 32), LR ≤ 2e-5 relative to base pretrain, router weights frozen (verify).
- Merge: `scripts/merge_adapter.py` (or the Unsloth-convention fused-MoE merge path for per-expert deltas).
- Gate: wp-bench codegen must meet/exceed the base acceptance bar.
- Qwen3-30B-A3B result: v1.2 reasoning-merged, wp-bench 0.4484 vLLM (bar 0.4286). Merged E2E validated
  (gen 10/10, judge 10/10, routing 20/20).

## Stage 3 — SFT: judge model (Phase 4.3 relabel)

Recalibrate the judge on human-relabeled scores. Judge-only relabel SFT (wp_gen data unchanged, so no
codegen regression path). Multi-seed for an ensemble.

- Entrypoint: `scripts/tinker_reasoning_sft.py --stage full --epochs 3 --seed {1,0,2}` on the relabel jsonl.
- Eval: `scripts/relabel/eval_relabel.py` (Spearman rho vs held-out relabeled val).
- Gate: judge rho must clear the recalibrated floor; ensemble median is the ship target, single-seed the
  fallback if 3× serve cost is unacceptable.
- Qwen3-30B-A3B result: v1.3, 3-seed median ensemble rho 0.8075 vLLM (floor 0.7554), single-seed s1 0.8017.
  Attenuation ceiling ~0.984; the residual ~0.16 is a genuine capability wall for SFT on this base. **This
  is the number a stronger base should move.**

## Stage 4 — Final eval (Phase 14)

A/B the shipping pair on wp-bench + the static suite; record size, speed, seed variance.

- Entrypoint: reuse the eval harness (`eval/`, wp-bench); consolidate into an EVAL3 report.
- Gate: wp-bench is the hard gate before packaging.
- Qwen3-30B-A3B result: `output/eval3/eval3_final_comparison.json`. Pair clears all bars; size flat
  (pruning gave nothing, see conditional gates).

---

## Conditional gate A — RL (GSPO) (Phases 8-10)

Try to push the judge past its SFT optimum with reinforcement learning on a calibration/defect reward.

- Entrypoint: `wp-finetune:run-rl-training` skill -> GSPO trainer, warm-started from the SFT judge.
- Gate: pre-registered kill criterion on validated teacher-Spearman improvement over warm-start noise, with
  a codegen trip-wire for Goodhart.
- Qwen3-30B-A3B result: **REJECTED.** Killed on 6/6 dead checkpoint reads; the reward signal was too weak
  to move a saturated judge. Retest on a new base only with a materially different reward family
  (execution-grounded / preference / multi-turn), not more steps of the same.

## Conditional gate B — MoE-Sieve expert-drop (Phases 11-12)

Profile routing and try to drop cold experts (training-free k-sweep with a TOST equivalence gate).

- Entrypoint: `wp-finetune:run-profiling` -> routing profile -> k-sweep + `eval_gate.py --tost --epsilon 2pp`.
- Protected expert mask: `output/.../protected_expert_mask.npy` (reasoning-critical experts, inviolable).
- Gate: a swept k must pass TOST equivalence at ε=2pp to ship.
- Qwen3-30B-A3B result: **no headroom.** E_eff ~88-99 live experts/layer of 128; wp-bench collapses
  0.4484(full) → 0.2275(k=64) → 0.0546(k=32). `optimal_k = full`. Retest on a base whose routing is more
  concentrated (lower E_eff) — that is exactly the condition that would make this gate pay.

## Conditional gate C — LoRA merge + weight-level prune (Phase 13)

Merge everything, then run AIMER (weight-norm, task-agnostic) and optionally REAP (calibration-based) at
25/50/75% with a gate-before-remove eval.

- Entrypoint: `scripts/` AIMER/REAP scorers + gate-before-remove driver (`output/prune/` artifacts).
- Gate: per-dimension retention (especially D2_security) within tolerance before any physical surgery.
- Qwen3-30B-A3B result: **no winner.** AIMER@25% (lightest ratio) collapses gen to 0.1577 and judge
  ensemble rho to 0.1651 (parse 44.6%); 50/75% skipped, REAP conditional-skipped. Ship unpruned. Methodology
  and both negative results: `output/prune/prune_methodology.md`. Retest on a higher-rho base with more
  redundant experts.

---

## Stage 5 — Packaging (Phase 15)

Cascading compression gates on the shipping pair. Quantization is the only size lever if the prune gates
return nothing.

- Entrypoint: `scripts/run_packaging_recipe.md` (GGUF via llama.cpp / AWQ via autoawq, served through the
  DGX container).
- Gate 1: record the bf16 baseline (size + quality) as the ±2pp reference.
- Gate 2: decide if quantization is warranted (deployment/memory constraint).
- Ladder: Q8 → Q6 → Q5 → Q4, ship the lowest tier within ±2pp. **Do not use uniform 4-bit nf4** on this
  architecture; it collapses the MoE router (measured). Use activation-aware methods below Q8.
- Model card: `output/packaging/MODEL_CARD.md` with the full lineage.
- Qwen3-30B-A3B result: bf16 57 GB/checkpoint; quantization warranted; Q4-nf4 dead; Q8 GGUF is the
  recommended ship tier (`output/packaging/`).

---

## Running it on the next base (Qwen3.6-class)

1. Swap the base in Stage 1-3 configs. Keep the task tokens and the frozen-router discipline.
2. Rerun the spine (Stages 1-4). The number to beat is judge rho 0.8075 and the attenuation ceiling ~0.98.
3. Rerun all three conditional gates. They returned nothing on Qwen3-30B-A3B; a higher-rho, more
   concentrated base is precisely where they might flip. Treat a `no_winner` as a valid, recorded outcome,
   not a failure to force.
4. Package with the same ladder. If the base is smaller or routing is prunable, the size story finally
   improves.

Deprecated one-off experiment drivers from the v3.0 run live in `deprecated/` with their own README. They
are not part of this pipeline; they are the archaeology of how the gates above were established.
