---
phase: 12
slug: moe-sieve-comparative-evaluation
status: skipped
decided: 2026-07-10
decided_by: "Dr. Robert Li (goal directive: 'ship phase 12 (given it is a skip)')"
---

# Phase 12 — SKIPPED (nothing to A/B)

## Rationale

Phase 12's mandate (EVAL2-01/02) was to A/B each k-sweep MoE-Sieve adapter against the baseline.
Phase 11 closed with **optimal_k = "full", no_equivalent_k = true** (human sign-off 2026-07-10,
`output/sieve/optimal_k.json`): no swept expert budget survives TOST — k=64 loses 22pp wp-bench,
k=32 loses 39pp, and the judge collapses to 0/121 parseable outputs at k≤32. There are **zero sieve
variants to evaluate**; a comparative-evaluation phase over an empty variant set is vacuous.

## Why the substance is already delivered

The comparative evidence Phase 12 would have produced exists in Phase 11's artifacts, measured under
one harness (vLLM, sequential serving):
- `output/sieve/k_sweep_results.json` — per-arm wp-bench + judge ensemble rho (full/64/32/13)
- `output/sieve/optimal_k.json` — TOST verdicts per arm, regression bars, human sign-off
- 9-dimension per-variant comparison (EVAL2-02) is moot: every variant fails the coarse gates by
  an order of magnitude past epsilon before dimension-level analysis could matter.

## Requirements disposition

- **EVAL2-01: N/A — skipped.** No sieve adapters exist to A/B (optimal_k=full).
- **EVAL2-02: N/A — skipped.** No variant report possible; baseline-vs-baseline is not a comparison.

## Carried forward

- Phase 13 consumes `output/sieve/prune_set_for_phase13.json` directly (protected mask 1,480 +
  layer_stability_notes + no-expert-drop finding + vLLM shipping-rho ~0.81 note).
- Phase 14 (final comparative eval) remains the place where the pruned model is compared to the
  unpruned baseline on all 9 dimensions — that phase is NOT skipped.
