# VERDICT-EVAL4 — v4.0 Milestone Final Evaluation

Source: `output/eval4/eval4_final_comparison.json` (all figures below are read from that file, not
re-derived). Comparability determination: `output/eval4/comparability_audit.json`.

## 1. Primary Verdict

**PRIMARY TARGET NOT MET.** The pre-registered judge quality target (rho > 0.85 single-seed OR
rho > 0.87 ensemble, CI-lower-aware) is **NOT MET**:

| Path | rho | CI-lower | Target | Met? |
|------|-----|----------|--------|------|
| served s1 (vLLM, pre-registered methodology) | 0.7872 | 0.7125 | > 0.85 | **NO** |
| capture ensemble (3-seed median, Tinker capture) | 0.8160 | 0.7563 | > 0.87 | **NO** |

`primary_judge_target_met = false`. This is recorded as the **valid, pre-registered failure
disposition** — "no_winner is a result" — not a forced pass. `disposition = valid_recorded_miss`.

## 2. Gen A/B (USER DIRECTIVE: dual candidates, gen-role winner picked)

| Candidate | overall | CI-lower | CI-upper | Clears 0.4286 floor (CI-aware) | Source |
|---|---|---|---|---|---|
| **Candidate A — raw base** (Qwen3.6-35B-A3B, no adapter) | **0.4897** | 0.3812 | 0.5983 | No | gen03_wpbench.json (fresh_new_base_anchor) |
| **Candidate B — ep1** (best trained gen variant) | **0.4381** | 0.3295 | 0.5504 | No | exp1_ep1_wpbench.json |
| reference: ep3 (shipped/promoted, overtrained) | 0.372 | 0.2847 | 0.4753 | No | gen03_wpbench.json |
| reference: v4b (rebuilt-mix, 2 epochs) | 0.4022 | 0.2924 | 0.5122 | No | exp4_bench.json |

vs v3.0 shipping gen (both OLD base, `cross_base_caveat=true`): Phase 17 fresh full re-run **0.4365**,
Gate-1 reference **0.4484**.

**Gen-role winner: RAW BASE.** Rationale: raw base dominates every trained variant on **both** point
estimate (0.4897 > ep1 0.4381 > v4b 0.4022 > ep3 0.372) **and** CI-lower (0.3812 > ep1 0.3295), and it
exceeds the v3.0 shipping gen figure (0.4365). The diagnostic's final verdict
(`output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md`) is that the v1.2 SFT-for-codegen recipe has
**negative headroom** on this stronger base — the model's raw coding ability exceeds anything the
current training corpus teaches it (regression-to-teacher, confirmed via exp5: training targets lose
to raw-base outputs on wiring/`<?php`/docblock axes). This winner call is robust regardless of whether
the raw-base CI-lower itself clears the floor, because it is a relative A/B and raw base wins both
metrics against every trained alternative.

## 3. Judge A/B

| Figure | rho | CI-lower | Note | Source |
|---|---|---|---|---|
| served s1 (vLLM, pre-registered methodology) | 0.7872 | 0.7125 | gating figure | judge03_rho.json |
| capture ensemble (3-seed median) | 0.8160 | 0.7563 | gating figure | judge03_rho.json / judge03_capture_rho.json |
| capture s1 (single seed) | 0.8358 | 0.7740 | NON-GATING promotion-path reference | judge03_rho.json |

vs v3.0 shipping judge (served, OLD base): ensemble **0.8075**, single **0.8017**. Ceiling reference:
**0.984**.

**capture-vs-capture:** new **0.8358** > old **0.8274** (+0.0084).

**Engine-numerics ceiling caveat:** the rebase **did** improve raw judge capability — capture-path rho
rose from 0.8274 (old base) to 0.8358 (new base). That gain is masked at the served-figure measurement
by a serving-stack numerics ceiling (Tinker-vs-vLLM, bf16-merge numerics + kernel differences flipping
~5/121 borderline greedy decodes) that DIAGNOSTIC_SYNTHESIS.md's exp3 confirms is common to both bases
(old-base served-equivalent recomputes to 0.7888, statistically indistinguishable from the new base's
0.7872). Served ~0.78–0.79 is a **serving-stack ceiling**, not a model or label deficiency.

## 4. Disposition + Next Levers (recorded, not triggered)

**Gen role.** The epoch-sweep lever is already realized (ep1 measured, exp1). The diagnostic's
recommended gen artifact for v4.0 is **raw-base-with-prompt-side-task-framing** — SFT-for-codegen is
not the productive lever on this base; SFT capacity is better spent on the judge role, where it
demonstrably works (+0.0084 capture rho).

**Judge role.** The relabel-campaign re-open condition (`V4-RERUN-ROADMAP.md` Claude's-Discretion item
2) is **UNMET**: `relabel_reopen_condition_met = false`. Both legs of the condition must hold — (a) the
new base's judge SFT saturates below target: **YES**, satisfied here; (b) a gap-closure diagnostic
(mirroring the 2026-07-08 investigation pattern — capacity, loss-shape, data-cleaning levers) must rule
out training-recipe causes **on this base**: **NOT YET RUN** — the existing gap-closure diagnostic ran
on Qwen3-30B-A3B, not Qwen3.6-35B-A3B. Because leg (b) is unmet, re-opening the relabel campaign is
**not** triggered here. If the judge targets are pursued further, that gap-closure diagnostic on this
base is the documented next step — before any relabel/data-quality work, since exp3 already shows the
served-figure gap is dominated by engine numerics rather than label quality.

## 5. Commit-Before-Decision (EVAL4-01 SC2)

These results are committed under `output/eval4/` **before** any packaging (Phase 27) or
conditional-gate (Phases 24–26) decision is made, satisfying EVAL4-01 SC2. No packaging or gate-flow
decision is made in this document — that call belongs to Phase 27 / the conditional-gate phases, which
consume this verdict as an input.

### Artifacts this phase produces

- `output/eval4/comparability_audit.json` — receipt-comparability determination + offline raw-base CI
  backfill.
- `output/eval4/eval4_final_comparison.json` — the machine-readable milestone verdict (this document's
  sole data source).
- `output/eval4/VERDICT-EVAL4.md` — this file.
- `scripts/build_eval4_comparison.py` — the synthesis script (`--emit audit`, `--emit verdict`).
