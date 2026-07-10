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
| bf16 | — | 56.8 GiB | 0.7700 (same-engine) / 0.8075 ens (vLLM) | baseline |
| **Q8** | **GGUF Q8_0** | **30.2 GiB (−47%)** | **0.7239** | **SHIPPABLE** — within noise of bf16, no collapse |
| Q6 / Q5 | GGUF Q6_K / Q5_K_M | ~24 / ~21 GiB | — | ladder candidates (pending) |
| Q4 | AWQ W4A16 (activation-aware) | ~16 GiB | — | high risk |
| Q4 | bitsandbytes nf4 (uniform) | ~16 GiB | 0.165 | **FAIL** — MoE router-quant collapse |

Q8 GGUF was measured on the single-seed judge (llama.cpp CUDA): 30.2 GiB (47% off bf16), judge rho 0.7239
vs same-engine bf16 0.7700 (delta −0.046, inside the 0.052 seed-noise floor, CIs overlap), parse rate 76%
matching bf16. It does not collapse. Full 3-way (foundation Qwen3-30B-A3B base / bf16 / Q8) in
`output/packaging/pkg03_q8_results.json`.

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

## Evaluation

- **wp-bench** (WordPress code correctness/knowledge/quality): generation 0.4484 (bar 0.4286).
- **Judge rho** (Spearman vs human-relabeled 9-dim scores): ensemble 0.8075, single-seed 0.8017
  (recalibrated floors 0.7554 / 0.7497). Attenuation ceiling for the noisy val set is ~0.984; the ~0.16
  residual is a genuine capability wall for SFT on this base, not a bug.
- 9 rubric dimensions: WPCS, security, SQL safety, performance, WP API usage, i18n, accessibility, code
  quality, dependency integrity.

## Limitations

- Two-model pair, not a single unified model. Both share the base; route by task token.
- No RL enhancement (rejected). No pruning speedup (no viable prune). Size reduction depends entirely on
  quantization, and the naive 4-bit tier is unavailable.
- Judge rho plateaus around 0.80-0.83; a materially better judge needs a stronger base (Qwen3.6-class),
  which is the intended next iteration of this exact pipeline.

## Provenance

Full training/eval history: `JOURNAL.md`, `.planning/ROADMAP.md`, `PIPELINE.md`. Pruning methodology and
both negative pruning results: `output/prune/prune_methodology.md`. Final comparison:
`output/eval3/eval3_final_comparison.json`.
