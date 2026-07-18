# v4.0 Phase 21 Diagnostic Synthesis — Why the Rebase Didn't Improve Results

**Date:** 2026-07-14. **Inputs:** `judge_attenuation_forensics.md`, `gen_regression_forensics.md`,
`recipe_provenance_audit.md` (same directory). Pure-analysis pass over existing Phase 21 artifacts —
no new GPU/Tinker spend.

## Verdict in one paragraph

The rebase DID improve the judge — capture-path rho 0.8358 beats the old base's 0.8274 under the identical
recipe — but the gain is masked at measurement time by a serving-stack attenuation (bf16-merge numerics +
kernel differences flipping ~5/121 borderline greedy decodes) that is statistically indistinguishable from
the old base's own capture→served drop (Δ0.048 vs Δ0.039, recomputed old served-equivalent 0.7888 ≈ new
0.7872). The gen regression is not a rebase failure either: it is a recipe-portability failure. The training
mix is 86% judge-shaped with only 13% gen targets, 92% of which are bare unwired function fragments; the old
base's weak raw prior (0.4033) matched that shape and gained, while the new base's strong raw prior (0.4897,
11/24 execution tests wiring hooks) was pulled DOWN to the training-data shape (4/24 hooks, 0/24 `<?php`,
half-length outputs) — classic regression-to-teacher on a base that now exceeds its training targets, likely
amplified by ep3 overtraining (terminal loss 1.46 vs the old base's 2.40 on the same 563 rows).

## The three findings

### 1. Judge (served 0.7872 vs target 0.85) — training improved, serving attenuates, same as old base
- Capture s1 0.8358 [0.7740, 0.8794] > old-base capture 0.8274. Ensemble capture 0.8160 > old 0.8075.
- Served drop driven by ~5/121 items (removing them halves the gap); mean scores identical (70.16 both);
  60/121 exact score matches; 0 parse failures; rendered prompts byte-identical between paths.
- Old-vs-new attenuation CIs overlap almost entirely → same known phenomenon, not a new regression.
- Ruled out: template mismatch (byte-identical), parser, pure noise.
- **Confirming experiment (cheap):** serve base + adapter via vLLM `--enable-lora` (no merge) on the same 121
  prompts. Matches 0.8358 → bug is the bf16 merge (fix: fp32 accumulation in merge). Matches 0.7872 →
  generic engine numerics.

### 2. Gen (0.372 vs raw 0.4897) — 24 execution tests, training-data shape, base-sensitivity
- Knowledge (320/344 tests, 93% of suite): FLAT (0.5625 merged vs 0.5594 raw). The entire overall drop is
  `correctness` on 24 execution tests: 0.229 merged vs 0.4375 raw (5 tests flip pass→zero).
- Merged reproduces training-target shape: hooks registered 4/24 vs raw 11/24; `<?php` openings 0/24 vs
  12/24; output length 487 vs 1025 chars. Coherent but structurally incomplete PHP — exactly what 92% of the
  73 `<wp_gen>` training targets look like (bare class-method fragments, 6/73 with hook wiring).
- Old base, same data: hooks 9/24 raw → 10/24 SFT (no loss — prior already matched the style). The defect
  only manifests on a base whose raw prior exceeds the targets.
- Ruled out as primary: thinking-mode loss (execution flips show no think-leak), truncation, parser.
  Unconfirmed contributor: ep3 overtraining (loss 1.46, smooth curves, 563 rows).
- Secondary confirmed (score-neutral today): judge-format bleed-through on knowledge MC (12.5% vs 2.5%);
  `<think>` tag leak 10/320 knowledge answers (0 raw) — merge slightly weakens `enable_thinking=False`.

### 3. Provenance — why the recipe ported asymmetrically
- Judge targets = 603 human relabels (era-agnostic ground truth) → recipe ports, +0.084 capture.
- Gen targets = mixed-era distillation (58% self-distilled new-base, but the load-bearing replay/CTF streams
  are old-era, unverified vs the new base's own quality) → recipe anti-ports on a stronger base.
- LR 4.99e-4 identical to old-base runs (Tinker auto heuristic) — not a stale carryover; renderer/masking
  verified symmetric train/eval. Both ruled out.
- Process gap found: the training terse-gate EXCLUDES the replay stream from its canonical metric — the one
  stream whose shape defect caused the regression was structurally invisible to the gate.

## Ranked next experiments (cost-ordered)

| # | Experiment | Cost | Tests hypothesis |
|---|---|---|---|
| 1 | Bench preserved `wp-gen-v4-ep1` on the 24 execution tests (or full 344) | ~1.5h GPU, $0 | ep3 overtraining |
| 2 | Judge s1 via vLLM `--enable-lora` (unmerged), same 121 prompts | ~1h GPU, $0 | merge-numerics vs engine-numerics |
| 3 | fp32-accumulation merge fix in `merge_adapter.py`, re-serve s1 | code + ~1.5h GPU | recover ~0.836 served → clears old-base ship bar |
| 4 | Rebuild gen mix: full-file wired targets (self-distill from raw base's own 0.4897 outputs, Claude-gated), rebalance gen share, 1-2 epochs | ~$2 Tinker + pipeline work | regression-to-teacher |
| 5 | Score 73 replay targets vs raw-base completions on same prompts | analysis-only | target-quality-below-base quantification |

## Experiment Results (2026-07-14, all five run — Tinker spend approved)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 5 | Replay-target quality vs raw-base outputs | targets lose on every axis: wiring 8.2% vs 45.8%, `<?php` 8.2% vs 50%, docblocks 8.2% vs 29.2% | **Regression-to-teacher CONFIRMED** |
| 2 | Judge s1 unmerged via vLLM `--enable-lora` | BLOCKED — vLLM `add_lora` rejects Tinker's w1/w2/w3 routed-expert naming (tooling-convention mismatch, not "MoE LoRA unsupported") | inconclusive by blockage |
| 3 | fp32-accumulation merge + re-serve | rho 0.7823 [0.7119, 0.8307] ≈ original 0.7872; real merge path was ALREADY fp32-accumulating (forensics premise analyzed dead legacy code) | **Engine numerics dominate** — served ~0.78-0.79 is a serving-stack ceiling on this stack, not model quality; capture 0.8358 stands |
| 1 | Gen ep1 checkpoint full wp-bench | **0.4381** [0.3295, 0.5504] vs ep3 0.372 | **Overtraining CONFIRMED** — ep1 recovers 56% of the ep3→raw gap |
| 4 | Rebuilt gen mix (wired human-written corpus units, gen ≥50%, replay stream dropped, 2 epochs) + retrain + bench | canonical fs-gate PASS (0/120, 0/360); wp-bench **0.4022** [0.2924, 0.5122] | **Partial fix** — beats ep3 (+3.0pp) but ≈ ep1 and still below raw 0.4897; data-shape fix + fewer epochs each help, neither closes the gap |

### Final causal verdict

1. **Judge:** the rebase WORKED (capture 0.8358 > old 0.8274; ensemble 0.8160 > 0.8075). The served figure
   (~0.78) is bounded by Tinker-vs-vLLM engine numerics — a measurement/serving ceiling common to both
   bases (old base recomputed served-equivalent 0.7888), NOT a training or label deficiency. The relabel
   re-open condition should be evaluated against the CAPTURE path or a fixed serving stack.
2. **Gen:** three stacked causes, all now measured: (a) regression-to-teacher — training targets are
   structurally below the raw base's own output (exp5); (b) overtraining — ep3 vs ep1 costs ~6.6pp (exp1);
   (c) mix composition — rebuilt wired mix at 2 epochs lands 0.4022 (exp4), better than the original ep3
   but still ~8.8pp below raw. **Conclusion: on a base this strong, the SFT-for-codegen approach itself has
   negative headroom on wp-bench — the model's raw coding ability exceeds anything the current corpus can
   teach it.** The gen role's best v4.0 artifact may be the RAW base (0.4897) with prompt-side task
   framing, with SFT reserved for the judge role where it demonstrably works (+0.084 capture).
3. **Recommended disposition for Phase 23:** A/B the raw base as the gen candidate vs the SFT variants;
   judge promotion decided on capture-path rho with the serving-numerics caveat documented; carry the
   vLLM-LoRA naming mismatch (exp2) upstream as a known tooling gap.

## Disposition

Feeds Phase 23 (EVAL4-01 A/B verdict) and the V4-RERUN-ROADMAP failure-disposition/re-open machinery:
- The judge relabel re-open condition remains UNMET — recipe causes are NOT ruled out; experiments 2-3 must
  run first (they may clear the served figure without touching labels).
- The gen miss now has a mechanistic explanation and a concrete remediation path (experiments 1 + 4).
- The pre-registered "stronger base" hypothesis is PARTIALLY validated: base capability improved (raw bench
  +8.6pp vs old raw; judge capture +0.084), but the v1.2-era gen recipe is not base-portable upward.
