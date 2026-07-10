# Phase 14 — Final Comparative Evaluation Report

**Milestone:** v3.0 (MoE-Sieve, Pruning & Packaging)
**Date:** 2026-07-10
**Requirements:** EVAL3-01, EVAL3-02
**Machine-readable results:** `output/eval3/eval3_final_comparison.json`

## Summary

The v3.0 pipeline finishes with no pruned model and no shipped RL baseline. Both arms of the A/B the
ROADMAP planned for this phase are empty by prior decision, not by omission. What remains is a
confirmation pass: the shipping two-model pair, measured under the production vLLM stack, clears every
acceptance bar it is held to, and its size is unchanged because expert-level and weight-level pruning
both failed their gates in Phases 11 and 13. The only size lever left is quantization, which is Phase 15.

This report consolidates already-measured numbers rather than re-running the eval. Every figure is copied
from a named artifact captured on the same serving stack. Re-running the multi-hour vLLM wp-bench sweep
would reproduce `output/sieve/optimal_k.json` and change nothing.

## Why the planned A/B is vacuous

| Planned comparison arm | Status | Evidence |
|---|---|---|
| Pruned model | Does not exist | Phase 13 verdict `no_winner` -> ship unpruned, human sign-off 2026-07-10 (`output/prune/prune_methodology.md`, `output/selection.json`) |
| v2.0 RL baseline | Does not exist as a shipped model | RL rejected Phase 10, 6/6 dead checkpoint reads 2026-07-05; no RL checkpoint promoted |

With both candidates absent, EVAL3-01's "A/B eval of pruned model against v2.0 RL baseline" reduces to a
receipt: the shipping stack measured against its own acceptance bars.

## Shipping stack

- **Generation:** v1.2 reasoning-merged, 30/70 ratio — `models/qwen3-30b-wp-30_70-reasoning-merged-v4`
- **Judge:** v1.3 relabel-SFT, 3-seed median ensemble — `models/_staging/qwen3-30b-wp-v1.3-merged` (s1),
  `...-s0-merged`, `...-s2-merged`. Single-seed s1 is the pre-authorized fallback if 3x serve or GB10
  memory is unacceptable.

## Quality (measured, vLLM full arm)

Source: `output/sieve/optimal_k.json` (same vLLM stack both sides, per `sanity_gate_recalibration.json`).

| Metric | Measured | Bar | Pass |
|---|---|---|---|
| wp-bench overall (gen) | **0.4484** | 0.4286 (Phase 4.4 acceptance) | ✅ |
| judge rho, 3-seed ensemble | **0.8075** | 0.7554 (recalibrated) | ✅ |
| judge rho, single-seed s1 | **0.8017** | 0.7497 (fallback bar) | ✅ |
| seed noise floor | 0.0520 | — | — |

Note on the codegen number: the 0.4616 figure quoted through the campaign is the Tinker-runtime codegen
score. The vLLM-served figure is 0.4484. It clears the 0.4286 acceptance bar, which is the number that
governs the packaging gate. This is the honest shipping figure and it is the one recorded here.

**wp-bench HARD GATE (EVAL3-01): PASS** — 0.4484 ≥ 0.4286.

## Nine-dimension coverage

The 9-dimension rubric was exercised through the judge path across the campaign; the judge rho above is
the aggregate correlation against human-relabeled dimension scores. Per-dimension retention was the
decision axis in Phase 13's gate-before-remove eval (`output/prune/gated/`, D2_security tracked
explicitly at `output/prune/gated/aimer_25_d2.json`). Since no pruning ships, there is no per-dimension
degradation to report against a pruned variant. The shipping judge's dimension behavior is the same one
promoted as v1.3.

## Size and speed delta

| Property | Value |
|---|---|
| Architecture | qwen3_moe, 128 experts, top-8, 48 layers |
| Total params | ~30.5B |
| Active params / token | ~3.3B |
| bf16 size per checkpoint | **57 GB** |
| Size reduction vs base | **0%** |
| Inference speed vs base | **Unchanged** |

The v3.0 milestone was pitched on shrinking the model. It did not shrink. Expert-drop is dead (Phase 11:
E_eff ~88-99 live experts/layer, every masked-k budget cuts live capacity, wp-bench collapses
0.4484 -> 0.2275 at k=64). Weight-level AIMER is dead (Phase 13: 25% compression drops gen to 0.1577 and
judge ensemble rho to 0.1651 with 44.6% parse, both failing at the lightest ratio). No parameters were
removed, so latency and throughput are architecturally identical to base Qwen3-30B-A3B. There was no
speedup to measure because there was nothing to prune.

Served footprint of the pair: 114 GB single-seed (57 gen + 57 judge), 228 GB for the full ensemble
(57 gen + 3x57 judge served sequentially).

## Seed variance

Judge multi-seed spread was characterized during v1.3 promotion: three ep3 seeds 0.796 / 0.827 / 0.790
(Tinker-runtime), mean 0.804, all three above the v1.2 bar; the ensemble is the robustness play. The
vLLM seed noise floor recorded for gating is 0.0520 (`optimal_k.json`).

## Disposition

**RE-CONFIRMATION PASS.** The shipping stack clears every applicable bar on the measured vLLM full arm.
The size line is flat by two independent prior findings. Phase 14 is a receipt, and it reads clean.
Cleared to Phase 15 (packaging + quantization), where quantization is the only size lever left.
