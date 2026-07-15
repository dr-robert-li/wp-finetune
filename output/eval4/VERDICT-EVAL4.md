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

## 6. Extension: shipped-stack (llama.cpp Q8) comparison (23-02)

Section 3's judge figures were measured on bf16 vLLM — a stack v3's judge never shipped on. This
extension (pre-registered **before** measurement in `output/eval4/ext_q8_preregistration.md`) re-ran
the v4 judge on **v3's exact shipped stack**: Q8_0 GGUF via llama.cpp (build `8f114a9`, 2026-07-10,
well past the b9180 architecture floor), 3-seed median ensemble, same 121 val items, 8192-token cap,
temp 0, the unmodified `eval_relabel*` scorers. Machine-readable receipt:
`output/eval4/ext_q8_results.json`.

| Figure | v4 (this run) | v3 shipped | Note |
|---|---|---|---|
| Q8 s0 / s1 / s2 | 0.7360 / 0.7877 / 0.7758 | 0.7744 / 0.7928 / 0.7894 | 0 parse failures, both milestones |
| **Q8 3-seed ensemble** | **0.8067** [0.7356, 0.8526] | **0.8056** [0.7381, 0.8577] | same harness, same items |
| Paired per-item bootstrap Δ(v4−v3) | +0.0010 [−0.0512, +0.0565] | — | 10k resamples, seed 1337 |

**Pre-registered rule applied (rule_fired = `paired_bootstrap`; no fallback needed — the 121-item
join is exact, and v3 recomputed from its raw captures reproduces 0.8056 to 4 decimals):**

- (a) v4 ensemble point 0.8067 > 0.8056: **TRUE** (by +0.0011)
- (b) paired bootstrap CI-lower > 0: **FALSE** (CI spans zero almost symmetrically)
- **UNEQUIVOCAL WIN = FALSE.** Per the pre-registered failure disposition, **the v3 pair stays
  canonical**; judge-only shipping of the v4 judge is **not** justified by this measurement.

**What the secondary reads say.** llama.cpp does NOT lift v4's serving ceiling: v4 s1 on Q8-llama.cpp
(0.7877) is statistically identical to v4 s1 on bf16-vLLM (0.7872) — the ~0.79 single-seed served
figure is engine-independent, confirming DIAGNOSTIC_SYNTHESIS.md's "engine numerics dominate" verdict
generalizes across serving stacks. The ensemble mechanism recovers the same ~+2pp on both milestones
(v4: 0.7877→0.8067; v3: 0.7928→0.8056). The capture-path gain (+0.0084) simply does not survive any
real serving stack measured so far: on the stack that matters for shipping, v4 ≈ v3 (Δ +0.10pp,
paired CI [−5.1pp, +5.7pp]).

**Implication for judge-only shipping.** The v4 judge offers no measurable serving-time advantage
over the already-shipped v3 judge, while requiring the larger 35B base (37.8 GiB Q8 vs 30.2 GiB).
A judge-only v4 ship would carry cost without measured benefit. The v4 judge's real, measured
improvement remains capture-path-only (0.8358 vs 0.8274), which no examined serving stack realizes.

Extension artifacts: `output/eval4/ext_q8_preregistration.md`, `output/eval4/ext_q8_results.json`,
`output/eval4/ext_q8/` (per-seed captures + scores + ensemble), `scripts/eval4_ext_{merge_seeds.py,
gguf_convert.sh,q8_run.sh,verdict.py}`, `models/_gguf/wp-v4-judge-s{0,1,2}.Q8_0.gguf`.
