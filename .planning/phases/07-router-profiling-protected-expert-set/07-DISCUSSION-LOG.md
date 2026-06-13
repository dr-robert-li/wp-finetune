# Phase 7 Discussion Log

**Date:** 2026-06-14 · Human reference only (not consumed by downstream agents).

## Areas selected
Scope reduction · Protected-expert threshold (D-10) · Profiling stimulus set · Profile target + baseline (all 4).

## Q1 — Phase scope
Surfaced: ROADMAP §7 written for multiple surviving ratios + ratio-selection matrix (SC5), but Phase-4
triage gave NO_SURVIVORS except 30/70 (now the promoted v1.2 model).
**Decision:** Confirm scope reduction — profile the single promoted model + extract protected mask;
drop SC5. → D-01, D-02.

## Q2 — Protected-expert threshold (D-10)
Options: conservative co-activation (above per-layer mean both tasks) / statistical median / top-K
intersection.
**Decision:** **Conservative co-activation** — judge skill is the fragile axis; over-protection only
costs recoverable pruning headroom. Report mask-size sensitivity across thresholds. → D-03, D-04.

## Q3 — Bundled defaults (multi-select, all locked)
- Scope: single model, drop SC5 (E_eff kept informational + pruning baseline). → D-01/D-02
- Stimulus: reuse 4.4 captures (wp_gen + wp_judge val), balanced, 10% subsample Jaccard≥0.94. → D-05/D-06
- Target + baseline: profile merged reasoning-merged-v4, compare vs base_model_eeff.jsonl, reuse
  profile_base_model.py. → D-07/D-08
- Fold CI-aware gate todo (D-V4-10): gates report bootstrap CIs / CI-aware dispositions. → D-09

## Deferred
- Phase 8 judge-recalibration inheritance todo (resolves_phase 8) — reviewed, left for Phase 8.

## Claude's discretion
E_eff formula details, subsample/Jaccard mechanics, mask export format, telemetry embedding.
