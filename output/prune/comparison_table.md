# Phase 13 Pruning: Comparison Table + Selection Verdict (PRUNE-05)

Source: `output/prune/gated/*.json` (13-04 measured + 13-05 documented dispositions),
`output/prune/expansion_decision.md` (branch rationale), `scripts/prune_selection.py`
(eligibility gate, run over all 6 variants), `output/prune/selection.json` (this
plan's output).

## Eligibility gate (from `scripts/prune_selection.py` / `output/sieve/prune_set_for_phase13.json`)

A variant is eligible only if ALL hold:
- `gen_wp_bench >= 0.4284` (vLLM-measured gen bar)
- `judge_ensemble_rho >= 0.7555` (vLLM-measured judge bar)
- `judge_parse_rate >= 0.95`
- `d2_security_retention` within 2pp of `d2_security_baseline` (never more than 2pp below)
- `protected_retained` is `true`
- physically feasible: `k >= max_protected_per_layer` (40, driven by layer 1's 40 protected experts — a uniform per-layer keep-count K is required by PRUNE-06, so no ratio with k<40 can ever ship, regardless of accuracy)

Among eligible variants: prefer smaller `k` (higher compression); ties broken by higher `d2_security_retention`.

## 6-variant x 2-axis comparison table

| Method | Ratio | K (kept/128) | Axis | Metric | Measured | Bar | Pass | Disposition |
|--------|-------|------|------|--------|----------|-----|------|-------------|
| AIMER | 25% | 96 | gen | wp_bench overall | **0.1577** | >= 0.4284 | **FAIL** | **MEASURED** (13-04) |
| AIMER | 25% | 96 | judge | ensemble rho (3-seed) | **0.1651** | >= 0.7555 | **FAIL** | **MEASURED** (13-04) |
| AIMER | 25% | 96 | judge | parse rate | **0.4463** | >= 0.95 | **FAIL** | **MEASURED** (13-04); per-seed parse 26/34/5 of 121 (parse-collapse) |
| AIMER | 25% | 96 | d2 | D2_security retention | 25.15 (baseline 6.98) | within 2pp of baseline | n/a | **MEASURED but UNRELIABLE** — parsed values ride corrupted scales under parse collapse (13-04); not treated as a real pass |
| AIMER | 50% | 64 | gen | wp_bench overall | null (not served) | >= 0.4284 | FAIL (assumed) | bounded-worse-by-monotonicity: k=64 wp_bench=0.2275 in Phase 11's own k-sweep (-22.1pp vs full), already below AIMER@25's failed 0.1577 |
| AIMER | 50% | 64 | judge | ensemble rho / parse rate | null (not served) | >= 0.7555 / >= 0.95 | FAIL (assumed) | bounded-worse-by-monotonicity: strict subset of AIMER@25's already-failed keep-set |
| AIMER | 75% | 32 | gen | wp_bench overall | null (not served) | >= 0.4284 | FAIL (assumed) | bounded-worse-by-monotonicity (k=32 wp_bench=0.0546, -39.4pp) **+ physically infeasible** (k=32 < 40 protected in layer 1) |
| AIMER | 75% | 32 | judge | ensemble rho / parse rate | null (not served) | >= 0.7555 / >= 0.95 | FAIL (assumed) | bounded-worse-by-monotonicity **+ physically infeasible** |
| REAP | 25% | 96 | gen + judge | — | null (skipped) | — | — | conditional-skip: AIMER@25 did not pass, REAP domain comparison moot per PRUNE-02 |
| REAP | 50% | 64 | gen + judge | — | null (skipped) | — | — | conditional-skip (moot) + independently bounded-collapse territory |
| REAP | 75% | 32 | gen + judge | — | null (skipped) | — | — | conditional-skip (moot) **+ physically infeasible** (k=32 < 40) |

`protected_retained`: `true` on both AIMER@25 records (the only measured variant); `null` (not evaluated) on every unmeasured/skipped variant — missing fields fail the eligibility gate closed, never a silent pass.

## AIMER-vs-REAP domain-specificity overlap (PRUNE-04)

`output/prune/aimer_reap_overlap_25.json`: **skipped** — no REAP keep-mask exists at k=96 to Jaccard against AIMER's `aimer_gen_k96.npy` / `aimer_judge_k96.npy`, since REAP calibration was never run (AIMER@25 failed, making the comparison moot per PRUNE-02's conditional rule).

## Layer-stability headroom obligation (13-CONTEXT hard constraint, layers {9,13,14,31,35,36,45,46,47})

Per-flagged-layer protected-expert counts (from `protected_expert_mask.npy`):

| Layer | 9 | 13 | 14 | 31 | 35 | 36 | 45 | 46 | 47 |
|-------|---|----|----|----|----|----|----|----|----|
| Protected count | 29 | 33 | 32 | 27 | **36** | 30 | 27 | 28 | 31 |

Max flagged-layer protected count = 36 (layer 35). Headroom check requires `K >= 2x36 = 72` (using the more conservative global max of 40 instead, the same check requires `K >= 80`). Result: K=96 (25%) clears both thresholds; K=64 (50%) fails both (64<72, 64<80); K=32 (75%) fails both plus the absolute physical-feasibility floor (32<40). Since PRUNE-06 requires one scalar `num_local_experts` across all layers, no per-layer budget is possible — the obligation is honored by keeping every flagged layer's protected experts inside the same protected mask that gates every variant. Full detail in `output/prune/selection.json`'s `layer_stability_disposition` block.

## Selection verdict

**`no_winner`** — every one of the 6 variants is either a measured FAIL (AIMER@25, on 3 independent gates: gen bar, judge rho bar, parse-rate bar) or bounded-worse-by-monotonicity / physically infeasible / conditional-skip. No candidate ever reaches the eligibility gate, so no `K >= 2x flagged-layer headroom` check needed to be asserted against an actual winner (see `candidate_winner_check_reason` in `selection.json`: had AIMER@25 passed its accuracy bars, it would also have cleared the headroom check — headroom was never the disqualifying factor).

This is a first-class, explicit outcome per 13-CONTEXT: **the phase ships unpruned**, consistent with Phase 11's `optimal_k=full` sign-off (`output/sieve/optimal_k.json`, `human_signoff`). Zero weight has been physically removed; nothing is gated on this decision except whether 13-07 (physical surgery) runs at all.

## What each approval option means for 13-07

- **"approved: prune METHOD@RATIO"** — Not available. No variant is eligible; there is no winner to authorize. (If a human insists on proceeding with an ineligible variant, that is an explicit override of the measured-fail evidence above and should be treated as a new decision requiring its own justification, not a same-plan approval.)
- **"approved: ship unpruned"** — Closes Phase 13 without physical surgery. 13-07 does not run. The MoE model ships at full 128-expert width per layer, consistent with Phase 11's `optimal_k=full` finding (routing is too distributed, E_eff ~88-99/128, for any expert-subset compression to survive gen+judge bars).
- **"request a different ratio / re-eval"** — No further GPU spend is warranted per the monotonicity argument (`output/prune/expansion_decision.md`): AIMER@25 (k=96, the least aggressive candidate) already failed decisively (gen -27.1pp, judge rho -59.0pp below bar), and k=64/k=32 are already measured worse in Phase 11's own k-sweep. A different ratio between 96 and 128 (i.e., <25% pruning) was not scoped by this phase's ratio set and would require a new plan.
