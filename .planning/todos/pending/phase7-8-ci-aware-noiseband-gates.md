---
id: phase7-8-ci-aware-noiseband-gates
created: 2026-06-14
source_phase: "04.4"
resolves_phase: "7"
priority: medium
tags: [gate-design, statistics, D-V4-10]
---

# Adopt CI-aware disposition for noise-band gates (Phase 7/8 gate definitions)

**Decision:** D-V4-10 waiver hardening (`04.4-D-V4-10-WAIVER.md`).

Phase 04.4 was blocked by an absolute point-bar gate (REVL-01A Spearman ≥ 0.263) at n≈120, where the
Spearman standard error is ≈0.09. The bar sat *inside* the noise band: the merged candidate "failed"
(0.240, 95% CI [0.061, 0.410]) while the already-validated grid winner "passed" (0.294) — yet the
paired difference CI [−0.169, +0.052] **includes zero**. The gate cannot distinguish the two. The grid
winner itself would re-fail ~36% of the time on a redraw. This is the **second** judge "regression" on
this project to dissolve under scrutiny (RC-A unclosed-`<think>` harness bug was the first).

**Requirement:** For any judge/eval gate measured on small samples (n≲200), Phase 7/8 gate definitions
should use a **CI-aware disposition**: require the bootstrap lower bound to clear the bar, measured
identically on baseline and candidate — rather than comparing point estimates to an absolute bar. This
is NOT a post-hoc bar move (goalpost-moving); it is a consistent statistical disposition baked into the
gate definition *before* the next re-gate, so a point estimate crossing a line through its own noise
does not trigger a false block.

**Acceptance:** Phase 7/8 gate spec documents the CI-aware rule for noise-band metrics; where a metric
has SE comparable to its bar margin, the gate reports the bootstrap CI alongside the point estimate.
