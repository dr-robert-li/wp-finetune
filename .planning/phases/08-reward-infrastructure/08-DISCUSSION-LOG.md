# Phase 8: Reward Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-19
**Phase:** 8-reward-infrastructure
**Areas discussed:** Signal sourcing, Recalibration apply-point, Anti-hack set, Reward contract

---

## Signal sourcing

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse eval/ harness | Import eval/rubric_scorer.py + llm_checks.py + eval_judge.py; inherit PHPCS standard + vLLM endpoint; single source of truth | ✓ |
| Hybrid wrapper | Reuse scanners behind a thin reward-signal adapter (decouple weighting from eval scoring) | |
| Build fresh | New self-contained scanners (risk of drift from eval signals) | |

**User's choice:** Reuse eval/ harness
**Notes:** Reward signals must match eval signals — no drift; inherit already-validated PHPCS standard + judge endpoint.

---

## Recalibration apply-point (D-V4-09 +3.58)

| Option | Description | Selected |
|--------|-------------|----------|
| Offset pre-norm, point | Add +3.58 to raw judge, clip, then MO-GRPO normalize; CI documented only | ✓ |
| CI as weight discount | Point offset + shrink 30% judge weight by SE/CI width | |
| Offset post-norm | Normalize first, then shift | |

**User's choice:** Offset pre-norm, point
**Notes:** Simplest correct application; rank-invariant per artifact. CI [1.24,6.09]/SE deferred as a possible future weight discount.

---

## Anti-hack set (D-11)

| Option | Description | Selected |
|--------|-------------|----------|
| Perturb real + CI threshold | Perturb real gen+judge outputs along 3 hack axes; Claude agents score; pass = CI-aware below clean baseline | ✓ |
| Synthesize fresh | Claude agents generate net-new adversarial cases; absolute reward cap | |
| Hand-curate gold set | Small manual gold set per hack type | |

**User's choice:** Perturb real + CI threshold
**Notes:** Grounds adversarial cases in real distribution; CI-aware pass criterion consistent with D-09.

---

## Reward contract

| Option | Description | Selected |
|--------|-------------|----------|
| Scalar + breakdown | Per-sample returns scalar + per-signal breakdown (pre/post-norm); accepts group for MO-GRPO/VeRPO; epsilon on zero-variance | ✓ |
| Bare scalar | Final scalar only | |
| Batch-only + breakdown | Scalar + breakdown but batch-only API | |

**User's choice:** Scalar + breakdown
**Notes:** Breakdown feeds RLEV-02 logging; per-sample entry + group input supports MO-GRPO variance and VeRPO difficulty.

---

## Claude's Discretion

None — all four areas decided explicitly by the user (each on the recommended option).

## Deferred Ideas

- CI-as-weight-discount for the judge component (revisit if RL shows judge signal noisy).
- Synthesize-fresh / hand-curated anti-hack augmentation (later hardening pass).
- Phase 9 items: router-shift stabilization, protected-expert routing regularizer, dual-mode judge RL rewards.
