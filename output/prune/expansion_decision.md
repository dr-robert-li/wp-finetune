# Phase 13 Plan 05: Expansion Decision

## Branch taken: BOUNDED-WORSE-BY-MONOTONICITY (no GPU serving this plan)

## Input: 13-04 AIMER@25% gate result

AIMER@25% (keep=96/128 experts/layer, `output/prune/gated/aimer_25_gen.json` +
`aimer_25_judge.json`) is a decisive **MEASURED FAIL on both axes**:

| Axis | Metric | Measured | Bar | Pass |
|------|--------|----------|-----|------|
| gen | wp_bench overall | 0.1577 | >= 0.4284 | FAIL |
| judge | ensemble rho (3-seed) | 0.1651 | >= 0.7555 | FAIL |
| judge | parse rate | 0.4463 | >= 0.95 | FAIL |

Per-seed parse counts s0/s1/s2 = 26/34/5 of 121 -- the Phase-11 parse-collapse
mode reproduced exactly (13-RESEARCH Pitfall 2).

Per 13-CONTEXT / 13-RESEARCH Primary Recommendation, the branch rule is:
**IF AIMER@25 passed all bars on at least one axis -> expand to 50/75.
ELSE -> do not serve 50/75; record bounded-worse-by-monotonicity.**
AIMER@25 failed both axes decisively (gen -27.1pp below bar, judge rho -59.0pp
below bar) -> **this plan takes the ELSE branch.**

## Why 50%/75% cannot beat a failed 25%: the monotonicity argument

AIMER@50 keeps k=64/128 (fewer than AIMER@25's k=96); AIMER@75 keeps k=32/128
(fewer still). A uniform expert-keep policy is monotone in a strictly
narrowing sense: keeping a strict subset of an already-kept set cannot expose
more model capability. `output/sieve/optimal_k.json`'s own k-sweep (a
different ranking signal, routing-coldness, but the same keep-count axis)
demonstrates this monotonic collapse empirically at the exact same keep
counts AIMER@50/75 would use:

| k (keep-count) | wp_bench | delta vs full (0.4484) |
|----|----|----|
| 64 (= AIMER@50's k) | 0.2275 | -22.1pp |
| 32 (= AIMER@75's k) | 0.0546 | -39.4pp |

Both k=64 and k=32 are already below the 25%-level gen bar and collapse
further as k drops. AIMER@25 (k=96, the *least* aggressive of the three
expansion candidates) already measured 0.1577 -- below even the k=64 sweep
point above. There is no empirical or mechanistic basis on which a smaller
keep-count (AIMER@50 or AIMER@75) could recover a pass the larger keep-count
already failed by 27-59pp. Serving 50%/75% would spend ~12-16h of GB10
wall-clock (two more sequential vLLM gen arms + up to six more judge seed
captures) on cells with zero decision value (13-RESEARCH Common Pitfall 5).

## Physical-feasibility ceiling (independent of the gate result)

Even in a counterfactual world where AIMER@75% (k=32) passed every accuracy
gate, it would **still not be a shippable winner**: `protected_expert_mask.npy`
shows layer 1 alone carries 40 protected experts, and PRUNE-06 requires a
single UNIFORM per-layer keep-count K across all 48 layers. K=32 < 40 is
physically infeasible -- 8 of layer 1's protected experts could not be kept
without violating the protection constraint. This ceiling holds regardless of
measured accuracy and is encoded directly in `scripts/prune_selection.py`'s
`max_protected_per_layer` check (`k < max_protected` disqualifies any k=32
variant unconditionally). So 75% is at most informational even in the
expand branch; in this fail-branch it is not measured at all.

## Cells written this plan (documented, not silently skipped -- T-13-06)

- `output/prune/gated/aimer_50_gen.json`, `aimer_50_judge.json` -- measured=false,
  pass=false, bounded-worse-by-monotonicity rationale (this document).
- `output/prune/gated/aimer_75_gen.json`, `aimer_75_judge.json` -- measured=false,
  pass=false, bounded-worse-by-monotonicity rationale + physical-feasibility
  ceiling note.
- `output/prune/gated/reap_{25,50,75}_{gen,judge}.json` (6 files) -- all
  documented conditional-skip stubs (skipped=true): AIMER@25 did not pass, so
  REAP's domain-specificity comparison (PRUNE-02) is moot per 13-CONTEXT's
  conditional rule. No REAP calibration forward pass was run; no
  `reap_scores_gen.npy` / `reap_scores_judge.npy` were produced (there is
  nothing to score against -- `scripts/reap_prune.compute_reap_scores` remains
  unexecuted, exactly as 13-02 left it).
- `output/prune/aimer_reap_overlap_25.json` (PRUNE-04) -- documented
  conditional-skip: no REAP keep-mask exists to compare against AIMER's.

## Where this leaves the phase

The 6-variant table (2 methods x 3 ratios) is now complete: one cell measured
(AIMER@25, FAIL both axes), five cells bounded-worse/moot with explicit
evidence-backed dispositions. 13-06 selection will see zero eligible variants
on the same three independent gates AIMER@25 failed (gen bar, judge rho bar,
parse bar) -> expected verdict `no_winner`. The phase ships unpruned,
consistent with Phase 11's `optimal_k=full` sign-off (`output/sieve/optimal_k.json`,
`human_signoff`).

No GB10 wall-clock was spent on cells with zero decision value (success
criterion met).
