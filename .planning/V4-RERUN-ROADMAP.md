# V4-RERUN-ROADMAP.md — Rerunning PIPELINE.md on Qwen3.6-35B-A3B

**Status: PLANNING ARTIFACT ONLY. This roadmap gates a FUTURE v4.0 milestone that starts only on explicit
human sign-off. No downloads, no training, no code changes happen as a result of this document. It exists
so that when sign-off is given, execution starts from an approved, costed plan instead of an open-ended
experiment.**

**Base locked:** `Qwen/Qwen3.6-35B-A3B` (35B total / 3B active, 256 experts top-8-routed + 1 shared, hybrid
Gated-DeltaNet/Gated-Attention, Apache-2.0). Full rationale + live verification:
`.planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md`. Every base reference below is
that same locked model — no other candidate is used in this roadmap.

**The number this rerun exists to move:** judge rho **0.8075** (3-seed median ensemble, vLLM), attenuation
ceiling ~0.984, residual gap **~0.157**. That gap was measured as a genuine SFT capability wall of
Qwen3-30B-A3B (`output/relabel/gap_closure_summary.json`) — three independent levers (capacity, loss-shape,
data-cleaning) all failed to close it on the current base. A stronger, more concentrated base is the
declared next lever.

---

## Two architecture-delta work items (schedule before their dependent stages)

These are LOCKED per Phase 19 context and must land as roadmap tasks, not be discovered mid-run:

1. **Sieve/protected-mask tooling adaptation — BEFORE Conditional Gate B.** The current MoE-Sieve profiler
   and `protected_expert_mask.npy` pipeline (`scripts/` router-profiling stack) assume pure top-8-of-128
   routing with uniform attention layers. Qwen3.6-35B-A3B has 256 experts, an always-on shared expert (not
   maskable the same way as routed experts), and mixed layer types (30 DeltaNet-MoE blocks + 10
   Gated-Attention-MoE blocks per the 10x(3x+1x) pattern). The profiler's per-layer E_eff computation and
   the k-sweep masking logic need to (a) treat DeltaNet-MoE and Attention-MoE layers as separate strata
   rather than one uniform stack, and (b) always exclude the shared expert from the sweepable/prunable set.
   This is a tooling-engineering task, not a research question — do it before Gate B runs, not during.

2. **eos/pad token-ID alignment — BEFORE any SFT stage (Stage 2 and Stage 3).** Known issue on the
   3.5/3.6 line (QwenLM/Qwen3.6 GitHub discussion #96): `model.config.eos_token_id` /
   `model.config.pad_token_id` can mismatch the tokenizer's actual special-token IDs out of the box. If
   unaligned, SFT loss masking and generation stopping behave incorrectly (train-time and inference-time
   silent corruption, hard to diagnose after the fact). A one-time config-alignment check-and-fix step
   must run and be verified before Stage 2 kicks off — this is a Stage 1.5 prerequisite gate, not optional
   cleanup.

---

## Stage-by-stage map

Order follows PIPELINE.md exactly: Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> [Gate A, Gate B, Gate C
conditional, order per v2.0 reorder: RL before Sieve] -> Stage 5. Costs are anchored to named v3.0/v3.1
actuals; where the new base's larger param count or 2x expert count plausibly changes wall-clock, that is
called out qualitatively (rough estimate, not a fabricated number) rather than invented.

### Stage 1 — Data pipeline

- **(a) Expected delta:** none required. The training corpus (`data/reasoning_dataset/`,
  `data/relabel_v1/`) is base-agnostic text — WPCS-compliant PHP examples, 9-dim rubric labels, CoT
  traces. No regeneration needed for a base swap alone.
- **(b) Carried-forward known result:** 34,855 training examples; relabel campaign 603/603 human-labeled,
  M=3 median aggregation, pilot QC (reliability 0.969, kappa 0.623).
- **(c) Re-test gate:** dataset assembly hard gate (reasoning-vs-score consistency) re-runs automatically
  as part of any re-assembly, but since no new data generation is planned, this gate does not need to
  re-fire — carried forward as-is unless discretion item 2 (relabel reuse) is overridden.
- **(d) Cost:** effectively $0 / near-zero wall-clock — reuse existing artifacts, no new generation.

### Stage 1.5 — eos/pad token-ID alignment (NEW prerequisite, see work item 2 above)

- **(a) Expected delta:** N/A (tooling correctness step, not a pipeline stage in the original PIPELINE.md).
- **(b) Carried-forward known result:** N/A — new for this base.
- **(c) Re-test gate:** verify `model.config.eos_token_id`/`pad_token_id` match the tokenizer's special
  tokens before any SFT run starts; block Stage 2 on failure.
- **(d) Cost:** trivial (config inspection + one smoke generate/stop-token check), rounds to the GB10
  smoke-test budget, well under an hour.

### Stage 2 — SFT: generation model

- **(a) Expected delta:** the reasoning-mix SFT recipe (reasoning + 30% judge replay + 20% wp_gen replay,
  MoE-only LoRA rank 32, LR <=2e-5, frozen router) should transfer directly. The one operational
  constraint found during Task 1 verification: Tinker's live model table caps `Qwen/Qwen3.6-35B-A3B`
  training context at **64K tokens** (below its native 262K) — irrelevant here since wp-gen/judge training
  examples are function-level PHP + rubric text, far under 64K.
- **(b) Carried-forward known result:** v1.2 reasoning-merged wp-bench 0.4484 vLLM (bar 0.4286); merged E2E
  validated (gen 10/10, judge 10/10, routing 20/20).
- **(c) Re-test gate:** wp-bench codegen must meet/exceed the acceptance bar on the new base, same 0.4286
  floor (or a freshly-derived floor if the new base's baseline coding ability materially shifts the noise
  band — measure, don't assume).
- **(d) Cost:** Tinker SFT run ~$2/run-class (anchored to v1.3's measured $1.83 actual,
  `output/tinker/PROMOTED_v1.3.json`); same per-unit Tinker price tier as the current base (Task 1
  verification: LoRA pricing $0.36/$0.89/$1.07 matches the 30B-A3B-Base row, no cost-class jump). GB10
  merge + serve smoke: hours, not the multi-hour profiling-scale cost seen elsewhere in the pipeline.

### Stage 3 — SFT: judge model (relabel)

- **(a) Expected delta:** this is the stage the whole rerun targets. Success criteria below.
- **(b) Carried-forward known result:** v1.3, 3-seed median ensemble rho **0.8075** vLLM (floor 0.7554),
  single-seed s1 **0.8017**. Attenuation ceiling ~0.984; residual ~0.157 is the wall this base must move.
- **(c) Re-test gate:** multi-seed (seeds {1,0,2}) relabel-SFT on the SAME relabeled data (see discretion
  item 2 below); `scripts/relabel/eval_relabel.py` Spearman rho vs held-out relabeled val; gate = clear the
  recalibrated floor, ship the ensemble median with single-seed as fallback per the existing v3.0 decision
  pattern.
- **(d) Cost:** ~$2/run-class per seed, 3 seeds ~= $6 Tinker spend (anchored to $1.83/run v1.3 actual,
  scaled by seed count, same per-unit price tier confirmed by Task 1).

### Stage 4 — Final eval

- **(a) Expected delta:** re-run the same A/B harness (wp-bench + static suite) on the new pair; expect a
  wp-bench number in the same ballpark or higher (coding-agent benchmarks published by the vendor for this
  base are strong — SWE-bench Verified 73.4, LiveCodeBench v6 80.4 — see selection doc axis 5), and a
  judge rho that is the actual test of this whole rerun.
- **(b) Carried-forward known result:** `output/eval3/eval3_final_comparison.json` — pair clears all bars;
  size flat (no pruning gain on the old base).
- **(c) Re-test gate:** wp-bench hard gate before packaging, same structure as v3.0.
- **(d) Cost:** wp-bench full run ~19 min (anchored to the Phase 17 fresh full-suite actual,
  `output/bench17/wpbench_full_gate_rerun.json`); judge ensemble eval ~2h/6 arms (anchored to the Q8
  ens8192 measurement, `output/packaging/pkg03_ens8192_results.json`).

### Conditional Gate A — RL (GSPO)

- **(a) Expected delta:** unknown by design — this is exactly the kind of gate a stronger, less-saturated
  judge might flip. A higher-rho SFT judge leaves more headroom for RL to move without hitting the
  saturated-judge wall that killed the last attempt.
- **(b) Carried-forward known result:** **REJECTED** on Qwen3-30B-A3B. Killed on 6/6 dead checkpoint reads
  (`logs/phase09_rerun/SMOKE_READS_TALLY.md`); reward signal too weak to move a saturated judge.
- **(c) Re-test gate:** per PIPELINE.md's own guidance, retest ONLY with a materially different reward
  family (execution-grounded / preference / multi-turn) — not more steps of the same calibration reward.
  Pre-registered kill criterion carries forward unchanged (validated teacher-Spearman improvement over
  warm-start noise, codegen trip-wire for Goodhart).
- **(d) Cost:** GB10 GSPO smoke-to-kill was a matter of hours in the last run (killed at step 50); budget a
  similar smoke-scale probe before committing to a full RL run, not a blind full-budget run.

### Conditional Gate B — MoE-Sieve expert-drop

- **(a) Expected delta:** this is the gate most likely to flip. 256 experts vs 128 means roughly 2x the
  raw expert count for the same task-routing job; if concentration (E_eff/expert-count ratio) is similar
  or better, absolute redundancy headroom is larger. Community REAP checkpoints report 20%-expert-drop as
  "competitive" on this architecture family (informal, not TOST-grade — noted in RESEARCH-BASESCAN).
- **(b) Carried-forward known result:** **no headroom** on Qwen3-30B-A3B. E_eff ~88-99/128 live
  experts/layer; wp-bench collapses 0.4484(full) -> 0.2275(k=64) -> 0.0546(k=32). `optimal_k = full`.
- **(c) Re-test gate:** same TOST equivalence gate at epsilon=2pp, but ONLY after work item 1 (Sieve
  tooling adaptation for mixed DeltaNet/Attention layers + shared-expert exclusion) lands. Do not run the
  k-sweep against unaudited tooling.
- **(d) Cost:** router profiling ~6h30m GB10 (anchored to the Phase 7 profiling actual on 34,855 examples /
  785.8M tokens); with 2x the expert count, budget qualitatively longer for the profiling pass and
  k-sweep grid — treat 6h30m as a floor, not a ceiling, for this stage on the new base.

### Conditional Gate C — LoRA merge + weight-level prune

- **(a) Expected delta:** also flip-candidate. A higher-rho base with more redundant experts (per Gate B's
  own finding, if positive) is exactly the condition PIPELINE.md flags as where weight-norm pruning might
  finally pay off.
- **(b) Carried-forward known result:** **no winner** on Qwen3-30B-A3B. AIMER@25% (lightest ratio)
  collapsed gen to wp-bench 0.1577 and judge ensemble rho to 0.1651 (parse 44.6%); 50/75% skipped, REAP
  conditional-skipped. Ship unpruned (`output/prune/prune_methodology.md`).
- **(c) Re-test gate:** same per-dimension retention gate (especially D2_security) before any physical
  surgery, same gate-before-remove eval structure.
- **(d) Cost:** merge is a one-time GB10 operation (hours); AIMER/REAP scoring + gate-before-remove eval at
  each ratio scales similarly to the wp-bench full-run cost (~19 min/eval arm) times the number of ratios
  tested.

### Stage 5 — Packaging

- **(a) Expected delta:** **memory-driven, not just quality-driven, this time.** Task 1 verification found
  that unlike Qwen3-30B-A3B (56.8 GiB/checkpoint bf16, pair fits 121 GB with ~7 GiB headroom),
  Qwen3.6-35B-A3B is 65.2 GiB/checkpoint bf16 — the gen+judge pair at bf16 is **130.4 GiB, which does NOT
  fit the GB10 121 GB host concurrently.** Quantization is therefore a hard prerequisite for
  concurrent-pair serving on this base, not an optional size lever. Scaling the v3.0 Q8 ratio (30.2/56.8 =
  53.2% of bf16) projects ~34.7 GiB/checkpoint at Q8, ~69.4 GiB for the pair — fits comfortably.
- **(b) Carried-forward known result:** bf16 57 GB/checkpoint (old base); Q4-nf4 uniform quantization is
  DEAD (measured router collapse, rho 0.165); Q8 GGUF is the recommended lossless ship tier (rho 0.8056
  ensemble vs bf16 0.8100, -0.4pp, 47% smaller, 0 parse failures at 8192-token cap).
- **(c) Re-test gate:** same cascading Gate 1 (bf16 baseline) / Gate 2 (quantization-warranted decision) /
  ladder (Q8 -> Q6 -> Q5 -> Q4, ship lowest tier within +-2pp) structure. Given the memory finding above,
  Gate 2's answer is pre-determined "yes, warranted" for concurrent serving (not just deployment-preference
  as it was on the old base) — still run the gate formally to confirm the quality side clears +-2pp. Do
  NOT use uniform 4-bit nf4 on this architecture either (same MoE-router-collapse mechanism applies to
  any MoE base with a router). Ecosystem check (Task 1): `bartowski` and `unsloth` GGUF conversions,
  NVIDIA official NVFP4, and QuantTrio AWQ builds already exist for this exact model at launch.
- **(d) Cost:** GGUF convert ~1h/model (anchored to the v3.0 actual); with 2 checkpoints (gen+judge) in the
  pair, budget ~2h for the Q8 conversion pass alone, more if the ladder descends further.

---

## Carry-forward lessons (all six LOCKED items)

1. **Truncation-aware evals (8192-token caps).** The Q8 single-seed judge read at max_tokens=2048 looked
   marginal (0.7239) purely from prose truncation; raising to 8192 fixed it (0.8056 ensemble, 0 parse
   fails). Any new-base eval harness MUST use a token cap generous enough to avoid false-negative parse
   failures before drawing a quality conclusion.
2. **Real-generation warm-up gating.** Serving infra (vLLM/GB10 containers) needs a real warm-up generation
   before timing/throughput numbers are trusted — cold-start latency previously corrupted throughput
   projections.
3. **`--parallel` context splitting.** Long-running eval/profiling passes use context-splitting
   parallelism to fit within session/tool budgets; carry the same pattern for the larger 256-expert
   profiling pass (Gate B), which will have a larger routing-statistics surface to process.
4. **CI-aware gates (bootstrap lower bound must clear the bar).** Point-estimate gates hide real failures
   (Phase 7's Jaccard gate: point estimate would have passed at 6/48 layers below threshold; CI-lower
   caught it, `jaccard_ci_lower=0.9426>=0.94`). Every re-test gate above (TOST, wp-bench, judge rho) must
   use bootstrap CI lower bounds, not raw point estimates.
5. **Benchmark pre-registration discipline.** Phase 17's SWE-bench scope was locked and committed
   (`output/bench17/swebench_scope_preregistration.md`) BEFORE any result existed. Apply the same
   discipline to this rerun: pre-register the wp-bench/judge-rho acceptance floors for the new base BEFORE
   Stage 2/3 results are read, not after (the pre-registered success criteria below are that
   pre-registration).
6. **Double-grep archive rule.** When archiving deprecated v3.0-specific scripts/configs during the v4.0
   transition, grep for BOTH the filename and its exported symbols/functions before moving anything to
   `deprecated/` — Phase 17-01 caught a case where `scripts/_wpbench_pth` and `scripts/_wpbench_shim` were
   mis-archived in Phase 16 cleanup despite being active runtime dependencies referenced only via string
   path construction (import-only grep missed them).

---

## Pre-registered success criteria

Registered NOW, before any v4.0 run exists, per carry-forward lesson 5:

- **Primary target:** judge rho **> 0.85** single-seed, OR ensemble **> 0.87** (3-seed median), measured
  the same way as the v3.0 shipping figure (vLLM-served, `scripts/relabel/eval_relabel.py` Spearman vs
  held-out relabeled val).
- **Framed against:** the 0.8075 ensemble / 0.8017 single-seed shipping figures and the ~0.157 residual
  gap to the ~0.984 attenuation ceiling. Clearing 0.85/0.87 would close roughly a third to a half of that
  gap — a real, non-trivial move, not noise.
- **wp-bench floor:** must clear the same 0.4286 acceptance bar (or a freshly-measured noise-adjusted
  floor if the new base's raw coding ability shifts the baseline materially — measure fresh, don't inherit
  blindly).
- **Failure disposition:** if the new base's judge SFT does NOT clear 0.85/0.87, that is itself a valid,
  recorded outcome (same "no_winner is a result, not a failure to force" discipline PIPELINE.md already
  applies to the three conditional gates) — it would mean the ceiling problem is deeper than base capacity
  alone, which is useful information for whatever comes after v4.0.

---

## Claude's-Discretion items — resolved with documented defaults

### 1. Proposed v4.0 milestone phase structure

Mirrors the PIPELINE.md stage list, informed by dependencies (Stage 1.5 token-alignment and the two
architecture-delta work items inserted where they gate, RL-before-Sieve per the existing v2.0 reorder
precedent):

| Phase | Content | Depends on |
|---|---|---|
| 20 | Base bring-up: download/load smoke test, eos/pad token-ID alignment (work item 2), DeltaNet-on-aarch64 op smoke check | Phase 19 sign-off |
| 21 | Stage 2 — SFT generation model (reasoning mix, reuse Stage 1 data) | 20 |
| 22 | Stage 3 — SFT judge model (relabel-SFT, reuse or re-run per discretion item 2 below) | 20 (parallel-safe with 21) |
| 23 | Stage 4 — Final eval (wp-bench + judge rho A/B vs v3.0 shipping figures; pre-registered criteria gate) | 21, 22 |
| 24 | Conditional Gate A — RL re-test (new reward family only, per re-test gate above) | 23 |
| 25 | Sieve/protected-mask tooling adaptation (work item 1) | 20 (can start early, independent of SFT) |
| 26 | Conditional Gate B — MoE-Sieve re-test | 24 (RL-before-Sieve precedent), 25 |
| 27 | Conditional Gate C — LoRA merge + weight-level prune re-test | 26 |
| 28 | Stage 5 — Packaging (Q8 GGUF mandatory for concurrent pair-serving per the memory finding; model card v2) | 23 (bf16 pair minimum), 26/27 if a compression winner emerges |
| 29 | Publication refresh (HF card update, benchmark deltas vs v3.0) | 28 |

This is a proposal for whoever plans v4.0, not a binding phase-numbering commitment — the real
`gsd-plan-phase` pass at execution time may split/merge these.

### 2. Relabel-campaign reuse vs re-run

**Recommendation: REUSE the v1.3 relabel labels** (603 human-relabeled items, M=3 median aggregation,
`data/relabel_v1/`) rather than re-running the human relabel campaign for v4.0. Rationale: the labels are
base-agnostic ground truth (human judgment of WordPress code quality against the frozen 9-dim rubric) —
they do not encode anything about Qwen3-30B-A3B specifically, so there is no reason a stronger base
invalidates them. Re-running a 603-item human-relabel campaign is expensive and the marginal value is
unclear until the new base's SFT is actually tried against the existing labels.

**Re-open condition (explicit, not silent):** re-open the relabel campaign ONLY IF the new base's judge
SFT, trained on the reused labels, saturates below the pre-registered rho target (0.85/0.87) above AND
diagnostic investigation (mirroring the 2026-07-08 gap-closure investigation pattern — capacity, loss-shape,
data-cleaning levers) rules out training-recipe causes. In that specific case, label quality/coverage
becomes the remaining suspect and a fresh or expanded relabel pass is warranted. Do not re-run the
campaign speculatively "just in case" — that is exactly the kind of ungated GPU/labor spend this whole
phase exists to prevent.

---

## Closing gate

**This roadmap is a plan, not an authorization.** Every phase in the proposed v4.0 structure above, and
every cost estimate in the stage-by-stage map, requires explicit human sign-off before any GPU spend,
Tinker spend, or weight download happens. The trigger for opening v4.0 is a human decision, not an
automated `gsd-progress` advance — this phase (19) ends with the roadmap written and pushed, and the next
action is a human choosing when (or whether) to greenlight v4.0.
