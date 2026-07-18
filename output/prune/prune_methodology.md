# Pruning Methodology + Compression Lineage (PRUNE-06)

**Outcome: SHIP UNPRUNED.** No physical surgery was performed. This is the documented
close of Phase 13's pruning track, executed per the `no_winner` verdict (13-06,
`output/prune/selection.json`) and the human sign-off approving `ship_unpruned`
(Dr. Robert Li, 2026-07-10).

## Method

Two expert-importance scoring methods were scoped for Phase 13:

- **AIMER** (primary): activation-informed expert relevance scoring, evaluated at
  all 3 ratios (25%/50%/75%, K=96/64/32 kept of 128 experts/layer).
- **REAP** (conditional): domain-specificity comparison against AIMER, gated on
  AIMER clearing its eligibility bars first (PRUNE-02 conditional rule).

Only AIMER@25 (K=96, the least aggressive ratio) was actually measured end to end
(gen + judge + D2, 13-04). It failed on three independent, vLLM-measured gates:

| Gate | Measured | Bar | Result |
|------|----------|-----|--------|
| gen wp_bench overall | 0.1577 | >= 0.4284 | FAIL (-27.1pp) |
| judge ensemble rho (3-seed) | 0.1651 | >= 0.7555 | FAIL (-59.0pp) |
| judge parse rate | 0.4463 | >= 0.95 | FAIL |

AIMER@50/75 and all 3 REAP ratios were not served — each is either
bounded-worse-by-monotonicity (Phase 11's own k-sweep already measured k=64/k=32
strictly below AIMER@25's failed numbers), physically infeasible (k=32 < 40, the
global max per-layer protected-expert floor), or a conditional-skip (REAP is moot
once AIMER@25 fails). Every unmeasured variant fails the eligibility gate closed
on missing fields — never a silent pass. Full per-variant detail:
`output/prune/selection.json`, `output/prune/comparison_table.md`.

## Verdict

**`no_winner`** (produced by `scripts/prune_selection.py`, not hand-declared).
No candidate ever reached the eligibility gate. Human sign-off approved
`ship_unpruned` (`selection.json.human_signoff`) at the 13-06 blocking
checkpoint. Per this plan's contract, the no-winner branch runs no surgery.

## Uniform-K mechanics (documented, NOT executed)

`scripts/prune_apply_physical.py` (built + self-tested in 13-03) implements the
physical-surgery mechanics that would have run had a winner been approved. Recorded
here for the model card / future re-evaluation, not because it ran:

1. **`build_uniform_keep_mask(scores, protected, K)`** — a `[48, 128]` boolean mask
   with exactly `K` True per layer. Every protected expert is kept; the remaining
   `K - protected_count[layer]` budget is filled by the highest-scoring
   non-protected experts. Raises if any layer's protected count exceeds `K`
   (the physical-feasibility floor).
2. **`apply_physical(checkpoint_dir, keep_mask, out_dir)`** — for each layer, drops
   the non-kept experts' `{gate,up,down}_proj` tensors, renumbers survivors to
   contiguous `0..K-1` (sorted by original index), slices the router
   (`model.layers.{L}.mlp.gate.weight`) to the kept rows in the same order (softmax
   renormalizes automatically over fewer rows), and rewrites
   `config.json.num_local_experts = K`.

**Physical-feasibility floor:** `K >= max_protected_per_layer` (40, driven by
layer 1). No ratio with `K < 40` can ever ship, independent of accuracy — this
ruled out K=32 (75%) outright. The layer-stability headroom obligation
(flagged layers `{9,13,14,31,35,36,45,46,47}`, max protected count 36 at layer
35) required `K >= 2 x 36 = 72` (conservative global variant: `K >= 80`); K=96
clears both, K=64/K=32 fail. Since `num_local_experts` is a scalar,
per-layer budgets are impossible — the obligation was to be enforced via the
protected mask, which every variant already respects. Headroom was never the
disqualifying factor for K=96: had AIMER@25 passed its accuracy bars, it would
also have cleared headroom (`selection.json.layer_stability_disposition.candidate_winner_check_reason`).

## Compression lineage

```
base (Qwen3-30B-A3B, 128 experts/layer, 48 layers)
  -> reasoning-merge (Tinker per-expert MoE-only LoRA merge, scripts/merge_adapter.py;
     attention + lm_head untouched; anchors_all_pass=true)
     gen: models/qwen3-30b-wp-30_70-reasoning-merged-v4
     judge seeds s0/s1/s2: models/_staging/qwen3-30b-wp-v1.3-{s0,,s2}-merged
  -> [no RL LoRA] (Phase 12 RL-CLOSED, commit 8860e89: ideal-conditions smoke
     killed 6/6; RL rejected as a training path, no adapter produced)
  -> [no Sieve LoRA] (Phase 11 Sieve is training-free routing-count masking at
     inference time, forward-hook -inf trick; no gradient step, no adapter weights)
  -> AIMER/REAP pruning evaluation (Phase 13): no_winner, ship unpruned
  -> SHIP: full 128-expert width per layer, unpruned
```

MERGE-01 established every checkpoint pruning evaluated is already a fully-merged
model (no adapters remained to merge — see
`.planning/phases/13-lora-merge-pruning/MERGE-01-TRACEABILITY.md`). Zero weight
was physically removed anywhere in this lineage.

## Relationship to Phase 11

This is the second independent negative pruning result for this model family:

- **Phase 11 (routing-cold / Sieve k-sweep):** training-free inference-time expert
  masking across k=13/32/64/full; `optimal_k=full` — no k below full width survived
  gen+judge bars without collapse. Routing is too distributed (E_eff ~88-99/128)
  for expert-subset compression to survive.
- **Phase 13 (weight-norm / AIMER-REAP physical pruning):** activation-informed
  expert-importance scoring at ratios 25/50/75%; `no_winner` — AIMER@25 (the least
  aggressive, most likely to survive) measured-failed 3 independent gates by wide
  margins (gen -27.1pp, judge rho -59.0pp).

Both findings point the same direction: this Qwen3-30B-A3B MoE's routing does not
concentrate enough per-expert specialization to support either inference-time
masking (Phase 11) or physical weight removal (Phase 13) at any measured ratio.
The model ships at full 128-expert width. Both negative results are recorded here
as the pruning-methodology finding for the model card and Phase 14 final
comparative eval.

## Consumers

- **Phase 14 final comparative eval:** input is the unpruned checkpoint
  (`models/qwen3-30b-wp-30_70-reasoning-merged-v4` for gen;
  `models/_staging/qwen3-30b-wp-v1.3-{s0,,s2}-merged` for judge seeds) — no
  pruned variant exists to compare.
- **Model card:** this file is the source for the pruning-methodology section;
  records both negative results (routing-cold Phase 11, weight-norm Phase 13).
