# Phase 10: RL Comparative Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 10-rl-comparative-evaluation
**Areas discussed:** Regression-gate policy, Checkpoint selection, wp-bench hard-gate definition, RLEV-02 value-add bar

---

## Regression-gate policy (D-10-01)

| Option | Description | Selected |
|--------|-------------|----------|
| CI-aware per-dimension | Regression = real drop below baseline per bootstrap CI (D-09); within-noise dips pass; judge must improve beyond noise | ✓ |
| Strict zero-regression | Any point-estimate drop on any dim fails, no noise tolerance | |
| Aggregate-weighted only | Overall weighted score ≥ baseline gates; per-dim advisory | |

**User's choice:** CI-aware per-dimension
**Notes:** Reconciles ROADMAP "no dimension regression permitted" as "no statistically real regression." Per-dimension so a real gen drop can't hide behind a judge gain.

---

## Checkpoint selection (D-10-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Best-by-reward + final, head-to-head | Eval both, pick winner; robust to late divergence / reward overfit | ✓ |
| Single final checkpoint | Last checkpoint only; assumes monotonic improvement | |
| Best-by-reward only | Trust the reward curve | |

**User's choice:** Best-by-reward + final, head-to-head
**Notes:** Winner becomes the canonical RL model handed to Phase 11.

---

## wp-bench hard-gate definition (D-10-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Aggregate CI-aware + per-task floor | RL aggregate bootstrap lower bound ≥ baseline point + no per-task catastrophic regression | ✓ |
| Aggregate point estimate only | RL aggregate ≥ baseline aggregate; ignores per-task blowups/noise | |
| Per-task strict no-regression | Every task ≥ baseline; strictest, trips on noise | |

**User's choice:** Aggregate CI-aware + per-task floor
**Notes:** Balances overall meet-or-exceed with catastrophe protection. Floor value derived at planning.

---

## RLEV-02 value-add bar (D-10-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Five-part conjunctive + human sign-off | judge↑ + wp-bench pass + anti-hack ≥ Phase 8 + protected-expert retention ≥ Phase 7 + no router collapse; human table sign-off | ✓ |
| Core-three + advisory routing | judge + wp-bench + anti-hack gating; routing metrics advisory | |
| Holistic human judgment | Present table, human decides, no hard sub-gates | |

**User's choice:** Five-part conjunctive + human sign-off
**Notes:** Conjunctive + reproducible; human checkpoint presents full v1.2-vs-RL table before the v3.0 gate.

---

## Claude's Discretion

- Bootstrap method/N/CI level — reuse `eval/eval_gate.py`, don't invent.
- Per-task catastrophic-regression floor value (D-10-03) — derive from baseline spread.
- Eval-harness wiring, serving plumbing, report format.
- Whether the +3.58 judge-recalibration offset applies to the RL judge component.

## Deferred Ideas

- Fresh RL-policy routing re-profiling for sieve — Phase 11.
- MoE-Sieve / merge / pruning — Phases 11/13.
- Auto-retrain on regression — out of scope (surface + suggest fix, don't loop).
- Reviewed-not-folded todos: `phase7-8-ci-aware-noiseband-gates.md`, `phase8-inherit-judge-recalibration.md` (already satisfied in Phase 8).
