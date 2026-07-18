---
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B
tags:
  - wordpress
  - php
  - code-generation
  - code-review
  - moe
  - qwen3
language:
  - en
pipeline_tag: text-generation
---

# wp-qwen3-moe — WordPress code generation + review (Qwen3-30B-A3B)

> **Superseded (2026-07-17):** the canonical judge deliverable is now the **v4 WP Judge**
> (`iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`, Qwen3.6-35B-A3B base) — see
> "[v4 outcome and lineage](#v40-outcome-qwen36-35b-a3b--why-v4-now-ships)" below. This v3.0 pair remains
> **published and untouched** on HuggingFace (`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`); the section
> below is its full, unedited lineage record, kept for provenance.

A two-model pair fine-tuned from Qwen3-30B-A3B for WordPress work: one model generates WPCS-compliant
PHP, the other reviews code against a 9-dimension rubric and explains what's wrong. Routing is by task
token: prepend `<wp_gen>` for generation, `<wp_judge>` for review.

This card documents the full v3.0 compression lineage, including the stages that returned nothing. Two of
them did, and that's recorded honestly rather than hidden.

## The pair

| Role | Token | Model | wp-bench / judge rho (vLLM) |
|---|---|---|---|
| Generation | `<wp_gen>` | v1.2 reasoning-merged (30/70 replay) | wp-bench **0.4484** |
| Review / judge | `<wp_judge>` | v1.3 relabel-SFT, 3-seed median ensemble | rho **0.8075** (single-seed s1 **0.8017**) |

- Base: Qwen3-30B-A3B (MoE, 128 experts, top-8, ~30.5B total / ~3.3B active, 48 layers).
- Size: 57 GB bf16 per checkpoint. See "Quantization" for the smaller serving tiers.

## Compression lineage (base -> RL -> MoE-Sieve -> merge -> prune -> quantize)

Each arrow is a real gate. Scores are vLLM-served unless noted.

1. **Base** — Qwen/Qwen3-30B-A3B.
2. **SFT (v1.2 / v1.3)** — reasoning SFT for generation (v1.2), relabel-SFT for the judge (v1.3, rho 0.827
   Tinker-runtime / 0.8017 vLLM single-seed). Judge trained on human-relabeled 9-dim scores.
3. **RL (GSPO)** — **REJECTED.** Warm-started from v1.3 with an oracle-passed calibration reward; killed on
   6/6 dead checkpoint reads (2026-07-05). No RL checkpoint was promoted. The reward signal was too weak to
   move the judge past its SFT local optimum.
4. **MoE-Sieve (expert drop)** — **no compression.** Routing profile shows ~88-99 effective experts/layer
   of 128; every masked-k budget cuts live capacity. wp-bench collapses 0.4484 (full) -> 0.2275 (k=64) ->
   0.0546 (k=32). `optimal_k = full`; nothing dropped.
5. **LoRA merge** — adapters merged into base weights; merged output matches adapter-on-base.
6. **AIMER / REAP pruning** — **no winner.** Weight-norm AIMER at the lightest ratio (25%) collapses
   generation to wp-bench 0.1577 and judge ensemble rho to 0.1651 (parse 44.6%). 50/75% skipped, REAP
   conditional-skipped per the pre-registered rule. Model ships unpruned at full 128-expert width.
7. **Quantization** — see below. Uniform 4-bit nf4 is a measured failure; Q8 is the recommended ship tier.

Net: the v3.0 pipeline confirmed the model **cannot be shrunk by expert-count or weight-norm methods** on
this workload. That's the finding. Quantization is the only size reduction available.

## Quantization

Gate 1 (bf16) is the quality baseline. Gate 2 decided quantization is warranted (the pair doesn't fit the
serving host with headroom at bf16). Ladder Q8 -> Q6 -> Q5 -> Q4, ship the lowest tier within ±2pp.

| Tier | Method | Size (judge s1) | Judge rho | Status |
|---|---|---|---|---|
| bf16 | — | 56.8 GiB | 0.8100 ens (llama.cpp@8192) / 0.8075 ens (vLLM) | baseline |
| **Q8** | **GGUF Q8_0** | **30.2 GiB (−47%)** | **0.8056 ens** | **SHIPPABLE — LOSSLESS** (Δ−0.4pp vs bf16, 0 parse fails) |
| Q6 / Q5 | GGUF Q6_K / Q5_K_M | ~24 / ~21 GiB | — | ladder candidates (pending) |
| Q4 | AWQ W4A16 (activation-aware) | ~16 GiB | — | high risk |
| Q4 | bitsandbytes nf4 (uniform) | ~16 GiB | 0.165 | **FAIL** — MoE router-quant collapse |

Q8 GGUF is lossless for the judge. Full 3-seed ensemble at max_tokens=8192 (llama.cpp CUDA), 0/121 parse
failures on every arm: Q8 ensemble rho 0.8056 vs bf16 ensemble 0.8100 (delta −0.4pp, clean ±2pp pass), at
47% smaller (30.2 vs 56.8 GiB per seed). The bf16 ensemble at 8192 (0.8100) matches the vLLM reference
(0.8075), validating the harness. An earlier single-seed read at max_tokens=2048 looked marginal (0.7239,
−4.6pp) but that was pure prose truncation — raising the cap to 8192 removed all parse failures and the gap.
Details: `output/packaging/pkg03_ens8192_results.json` (full 3-way with foundation in `pkg03_q8_results.json`).

Do not use uniform nf4 4-bit on this architecture. The router cannot tolerate uniform low-bit quantization;
an activation-aware method that protects router and attention weights is required below Q8. Foundation
check: the untrained Qwen3-30B-A3B base produced 0/121 parseable judge responses — the entire judge
capability comes from the fine-tune.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Generation
tok = AutoTokenizer.from_pretrained("wp-qwen3-moe-gen")
model = AutoModelForCausalLM.from_pretrained("wp-qwen3-moe-gen", torch_dtype="bfloat16", device_map="auto")
prompt = "<wp_gen>Write a WordPress function to safely query posts by meta value."
# ... generate; output is WPCS-compliant PHP

# Review
prompt = "<wp_judge>Review this code:\n<?php $wpdb->query(\"SELECT * FROM wp_posts WHERE ID=$id\"); ?>"
# ... generate; output is a 9-dim rubric critique (flags the unprepared SQL as a D2_security defect)
```

Serve `<wp_gen>` and `<wp_judge>` as the same base with the appropriate merged checkpoint, or as two
served checkpoints. The judge ensemble runs 3 seeds sequentially and takes the median; the single-seed s1
fallback trades ~0.006 rho for one-third the serve cost.

## Benchmarks

Numbers below are fresh, receipt-backed measurements on the shipping stack (vLLM bf16,
`models/qwen3-30b-wp-30_70-reasoning-merged-v4`, temperature 0.0), taken 2026-07-11; the base
anchor row serves the untrained `models/Qwen3-30B-A3B` on the same stack (2026-07-12).

| Benchmark | Score | Scope / config |
|---|---|---|
| **wp-bench** (in-domain) | **0.4365** overall | full 344-test wp-core-v1 suite, unlimited; knowledge 0.4906, correctness 0.3958 |
| **wp-bench, untrained base anchor** (Qwen3-30B-A3B) | **0.4033** overall | same suite/stack/seed, 2026-07-12; knowledge 0.4688, correctness 0.3542 |
| **SWE-bench Lite** (out-of-domain) | **1.67%** resolved (5/300) | generation-mode (non-agentic), oracle retrieval, 24k context, native arm64 local Docker eval |
| **SWE-bench-Multilingual PHP subset** (in-language, out-of-domain) | **0%** resolved (0/43) | same protocol; 4 PHP repos (phpspreadsheet, laravel, php-cs-fixer, carbon) |

**wp-bench.** The fresh full-suite score (0.4365) sits 1.19pp below the 0.4484 Gate-1 reference
figure, well inside the project's measured 5.20pp seed-noise floor, and clears the 0.4286 acceptance
bar. Same stack both sides (vLLM bf16, identical checkpoint and sampling); no regression signal.
Receipt: `output/bench17/wpbench_full_gate_rerun.json`.

**Base anchor.** The untrained foundation model (`models/Qwen3-30B-A3B`, no task-token embeddings,
`<wp_gen>` hitting it as plain text) scores 0.4033 on the identical harness, stack, and seed. The
fine-tune's lift is +3.32pp fresh-vs-fresh (0.4365 vs 0.4033), +4.51pp vs the Gate-1 figure — both
inside the 5.20pp seed-noise floor. Most of the base's score comes from the knowledge MCQ split
(0.4688) rather than execution (0.3542): Qwen3-30B already knows a lot of WordPress out of the box.
On this benchmark the gen fine-tune's measurable lift over base is modest; the training's clearest
demonstrated capability gain is the judge (untrained base: 0/121 parseable verdicts). Receipt:
`output/bench17/wpbench_base_anchor.json`.

**SWE-bench, and why the number is low.** SWE-bench Lite is Python-repository patch generation;
this model is WordPress/PHP-specialized. A low out-of-domain number is expected and is not a quality
defect: the model was never trained to patch Django or sympy, and the score is published for honest
positioning, not vanity. Protocol: one oracle-retrieval prompt in, one unified-diff patch out (no
agent scaffold, no retries), temperature 0.0, seed 0, max_model_len 24576, evaluated with the
official swebench 4.1.0 containerized harness built natively for arm64. The scope (Lite-300 + PHP-43)
was pre-registered and committed before any result existed
(`output/bench17/swebench_scope_preregistration.md`).

Full-scope denominators count everything against the model, disclosed in the receipt
(`output/bench17/swebench_eval_report.json`): of Lite's 300, 80 prompts exceeded the 24k serving
context (scored unresolved), 1 generation was unparseable, 59 patches failed to apply, and 29
instances could not be evaluated because their 2018-era Python environments cannot build on
arm64/2026 toolchains (also scored unresolved, conservative against the model). On the 131 instances
that ran to a test verdict the resolved rate is 3.82% (5/131). The PHP-Multilingual subset is
in-language but still out-of-domain (framework libraries, not WordPress): 0/43 full-scope, 0/20
evaluated; 17 over-length, 6 apply-failures.

## Evaluation

- **wp-bench** (WordPress code correctness/knowledge/quality): generation 0.4484 (bar 0.4286);
  fresh full-suite rerun on the shipping stack 0.4365 (within seed noise — see Benchmarks).
- **Judge rho** (Spearman vs human-relabeled 9-dim scores): ensemble 0.8075, single-seed 0.8017
  (recalibrated floors 0.7554 / 0.7497). Attenuation ceiling for the noisy val set is ~0.984; the ~0.16
  residual is a genuine capability wall for SFT on this base, not a bug.
- 9 rubric dimensions: WPCS, security, SQL safety, performance, WP API usage, i18n, accessibility, code
  quality, dependency integrity.

## Limitations

- Two-model pair, not a single unified model. Both share the base; route by task token.
- No RL enhancement (rejected). No pruning speedup (no viable prune). Size reduction depends entirely on
  quantization, and the naive 4-bit tier is unavailable.
- Judge rho plateaus around 0.80-0.83; a materially better judge needs a stronger base (Qwen3.6-class).
  That iteration was run — see "v4.0 outcome" below — and did not beat this checkpoint on any
  self-hostable serving stack.

## v4.0 outcome (Qwen3.6-35B-A3B) — why v4 now ships

The entire pipeline was rerun on `Qwen3.6-35B-A3B` to try to supersede this pair. First result,
receipt-backed (`output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md`, `output/eval4/VERDICT-EVAL4.md`):

- **Generation is retired as a deliverable.** The raw Qwen3.6 base scored wp-bench **0.4897**; every gen
  fine-tune regressed below it (best 0.4381) because the reasoning-mix targets are structurally weaker than
  what a modern base already writes. For generation, use a current base with prompt-side task framing — no
  fine-tune needed. This holds for v4 as well; v4 ships judge-only.
- **The judge improved but was, at that point, a shipped-stack tie.** The Qwen3.6 judge beat this one on
  the Tinker capture path (rho 0.8358 vs 0.8274), but a serving-numerics ceiling (~0.79, identical across
  vLLM-merged, llama.cpp-Q8-merged, and llama.cpp-unmerged runtime-LoRA) ate the gain. On the shipped Q8
  stack it read **0.8067 vs this model's 0.8056** — a statistical tie (paired bootstrap CI spans zero) at
  +25% size. Per the pre-registered rule, **v1.3 stayed canonical at that point.**

One v4 lever was still open at that point: the 256-expert Qwen3.6 judge had never been through MoE-Sieve
or weight-prune (v3.0 found no winner on 128 experts, but 256 + a shared expert is where the roadmap
predicted it might flip). Phases 22/25/26 (2026-07-16/17) resolved it:

- **Routing profile (Phase 25):** diffuse, no clean keep/drop cliff (E_eff mean 144.3/256) — the profile
  shape predicted the checkpoint would *resist* pruning.
- **AIMER weight-prune at k=224 (Phase 26) passed gate-before-remove anyway**, contradicting that
  prediction: pruned checkpoint rho **0.8134** (bf16-vLLM, single-seed s1) vs the same-stack full-width arm
  0.7935 — **+0.020, non-inferior, point-better** (ci_lower slack 0.001, thin but held). D2_security
  retained (6.326 ≥ 6.115 baseline). Surgery: stacked-tensor axis-0 slice 256→224 experts/layer
  (`shared_expert.*`/`mtp.*` untouched); `models/Qwen3.6-35B-A3B-judge-v4-pruned-k224`, 60 GB bf16.
- **GGUF conversion + quantization ladder (Phase 27):** the pruned checkpoint converted with `--no-mtp`
  (the MTP/nextn layer was left at 256 experts by the prune surgery; GGUF's `expert_count` metadata is a
  single global field, so the mixed-count checkpoint would not load — the pruned GGUF has no
  MTP/speculative-decoding head; see the unpruned variant below, which restores it). Full ladder measured
  on the shipped GGUF/llama.cpp stack, single-seed s1, n=121, gated against the frozen f16 floor:

  | Tier | Judge rho | Δ vs f16 floor | Parse fail | Size |
  |---|---|---|---|---|
  | f16 (floor) | 0.8002 | — | 0/121 | 57.10 GiB |
  | Q8_0 | 0.7851 | −1.51pp | 0/121 | 30.37 GiB |
  | **Q6_K (ships)** | **0.8063** | **+0.61pp** | **0/121** | **23.47 GiB** |
  | Q5_K_M | 0.8060 | +0.58pp | 1/121 | 20.36 GiB |

  All four rungs are statistically indistinguishable (95% CI half-widths ~7-8pp) — Q6_K scoring above its
  own f16 source is proof the rung-to-rung spread is single-seed sampling noise, not a real
  quantization-sensitivity signal (a lossy compression cannot legitimately exceed its source). Q6_K ships
  as the smallest of the two zero-parse-failure tiers, not as the highest-rho tier. Full derivation:
  `output/pkg-v4/pkg4_quantization_ladder.json` `noise_floor_finding` + `ship_rationale`.

- **Unpruned MTP variant (2026-07-18 addendum, human-directed):** the repo also ships
  `wp-judge-v4-unpruned.Q5_K_M.gguf` — the unpruned s1-merged checkpoint (256/256 experts, MTP head
  retained, block_count 41), quantized and gated on its **own** ladder (f16 floor rho 0.8081; Q8_0 0.7767;
  Q6_K 0.8074; Q5_K_M 0.8093 — non-monotonic again, reproducing the noise finding on a second independent
  checkpoint). Q5_K_M selected by the same rule (smallest in-band rung with 0/121 parse failures; unlike
  the pruned ladder, this Q5 ran clean). **MTP speculative decoding works on this file**: llama-server
  `--spec-type draft-mtp`, measured 56.4% draft acceptance (mean accepted length 2.69 tokens), judge rubric
  still parses under speculation. 23.61 GiB. Receipts: `output/pkg-v4/unpruned_quantization_ladder.json`,
  `output/pkg-v4/unpruned_mtp_smoke/mtp_smoke_receipt.json`, `output/pkg-v4/pub4_validation_receipt_unpruned.json`.

- **Ship policy (human-confirmed 2026-07-17; recommendation flipped 2026-07-18):** canonical flips
  **v3 → v4**. The v4 repo carries **two files**, and the **recommended file is the unpruned Q5_K_M**
  (23.61 GiB, 256/256 experts, MTP speculative decoding — `--spec-type draft-mtp`, 56% measured draft
  acceptance): same judge quality as the pruned file within measurement noise, faster serving. The
  **pruned Q6_K** (23.47 GiB, 224/256 experts) remains published as the **Gate C experiment artifact** —
  the AIMER k=224 prune passed gate-before-remove and proved 32 experts removable at tied quality, a real
  and receipted scientific result, but its 150 MB size advantage does not buy the operator anything the
  MTP head doesn't outweigh. It stays for provenance and for operators who want the smaller expert
  footprint. Both files are ~22-24% smaller than v3's 30.2 GiB — the size tradeoff v3.0 originally
  accepted (v4 "stays larger") is void; v4 is smaller, on the newer base, at tied quality.
  `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` (this pair) stays live, untouched, as the superseded
  prior artifact — it is not deprecated, deleted, or rewritten by the v4 publish.

Full v4 packaging receipts: `output/pkg-v4/` (`gate1_f16_baseline_v4.json`, `pkg4_quantization_ladder.json`,
`conversion_receipt_v4.json`); v4 HF card: `output/pkg-v4/hf_cards/judge_v4_README.md`; prune gate:
`output/prune-v4/selection_v4.json`.

## Provenance

Full training/eval history: `JOURNAL.md`, `.planning/ROADMAP.md`, `PIPELINE.md`. Pruning methodology and
both negative v3.0 pruning results: `output/prune/prune_methodology.md`. v3.0 final comparison:
`output/eval3/eval3_final_comparison.json`. v4 prune/pack lineage: `output/prune-v4/`, `output/pkg-v4/`.
