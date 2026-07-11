---
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - wordpress
  - php
  - code-review
  - moe
  - qwen3
  - gguf
---

# wp-qwen3-30b-a3b-wp-judge-v1.3-gguf — WordPress code review judge (Qwen3-30B-A3B, Q8_0 GGUF)

Review/judge half of a two-model WordPress pair fine-tuned from Qwen3-30B-A3B. This model reviews
PHP code against a 9-dimension rubric (WPCS, security, SQL safety, performance, WP API usage,
i18n, accessibility, code quality, dependency integrity) and explains what's wrong. The paired
generation model lives at
[iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2)
— route review requests here with `<wp_judge>` and generation requests there with `<wp_gen>`.

This repo ships the **Q8_0 GGUF** quantized tier — the only quantized tier that cleared the
lossless bar for this model. Three ensemble seeds are provided (`s0`, `s1`, `s2`); use all three
with median aggregation for the reported ensemble score, or `s1` alone as a leaner single-seed
fallback.

This card documents the full v3.0 compression lineage, including the stages that returned
nothing. Two of them did, and that's recorded honestly rather than hidden.

## The pair

| Role | Token | Model | wp-bench / judge rho (vLLM) |
|---|---|---|---|
| Generation | `<wp_gen>` | v1.2 reasoning-merged (30/70 replay) | wp-bench **0.4484** |
| **Review / judge (this repo)** | `<wp_judge>` | v1.3 relabel-SFT, 3-seed median ensemble | rho **0.8075** (single-seed s1 **0.8017**) |

- Base: Qwen3-30B-A3B (MoE, 128 experts, top-8, ~30.5B total / ~3.3B active, 48 layers).
- bf16 size: 57 GB per checkpoint. This repo ships the Q8_0 GGUF tier instead (see below).

## Compression lineage (base -> RL -> MoE-Sieve -> merge -> prune -> quantize)

Each arrow is a real gate, applied to the pair as a whole. Scores are vLLM-served unless noted.

1. **Base** — Qwen/Qwen3-30B-A3B.
2. **SFT (v1.2 / v1.3)** — reasoning SFT for generation (v1.2), relabel-SFT for the judge
   (v1.3, this repo, rho 0.827 Tinker-runtime / 0.8017 vLLM single-seed). Judge trained on
   human-relabeled 9-dim scores.
3. **RL (GSPO)** — **REJECTED.** Warm-started from v1.3 with an oracle-passed calibration reward;
   killed on 6/6 dead checkpoint reads (2026-07-05). No RL checkpoint was promoted. The reward
   signal was too weak to move the judge past its SFT local optimum.
4. **MoE-Sieve (expert drop)** — **no compression.** Routing profile shows ~88-99 effective
   experts/layer of 128; every masked-k budget cuts live capacity. wp-bench collapses
   0.4484 (full) -> 0.2275 (k=64) -> 0.0546 (k=32). `optimal_k = full`; nothing dropped.
5. **LoRA merge** — adapters merged into base weights; merged output matches adapter-on-base.
6. **AIMER / REAP pruning** — **no winner.** Weight-norm AIMER at the lightest ratio (25%)
   collapses generation to wp-bench 0.1577 and judge ensemble rho to 0.1651 (parse 44.6%).
   50/75% skipped, REAP conditional-skipped per the pre-registered rule. Ships unpruned at
   full 128-expert width.
7. **Quantization** — uniform 4-bit nf4 is a measured failure; Q8_0 GGUF (this repo) is the
   recommended ship tier.

Net: the v3.0 pipeline confirmed the model **cannot be shrunk by expert-count or weight-norm
methods** on this workload. That's the finding. Quantization is the only size reduction
available, and Q8_0 is where it lands losslessly.

## Quantization evidence

Gate 1 (bf16) is the quality baseline. Gate 2 decided quantization is warranted (the pair doesn't
fit the serving host with headroom at bf16). Ladder Q8 -> Q6 -> Q5 -> Q4, ship the lowest tier
within ±2pp.

| Tier | Method | Size (judge s1) | Judge rho | Status |
|---|---|---|---|---|
| bf16 | — | 56.8 GiB | 0.8100 ens (llama.cpp@8192) / 0.8075 ens (vLLM) | baseline |
| **Q8_0 (this repo)** | **GGUF** | **30.2 GiB (−47%)** | **0.8056 ens** | **SHIPPABLE — LOSSLESS** (Δ−0.4pp vs bf16, 0 parse fails) |
| Q6 / Q5 | GGUF Q6_K / Q5_K_M | ~24 / ~21 GiB | — | ladder candidates (not shipped) |
| Q4 | AWQ W4A16 (activation-aware) | ~16 GiB | — | high risk (not shipped) |
| Q4 | bitsandbytes nf4 (uniform) | ~16 GiB | 0.165 | **FAIL** — MoE router-quant collapse (not shipped) |

Q8_0 GGUF is lossless for the judge. Full 3-seed ensemble at max_tokens=8192 (llama.cpp CUDA),
0/121 parse failures on every arm: Q8 ensemble rho 0.8056 vs bf16 ensemble 0.8100 (delta −0.4pp,
clean ±2pp pass), at 47% smaller (30.2 vs 56.8 GiB per seed). The bf16 ensemble at 8192 (0.8100)
matches the vLLM reference (0.8075), validating the harness. An earlier single-seed read at
max_tokens=2048 looked marginal (0.7239, −4.6pp) but that was pure prose truncation — raising the
cap to 8192 removed all parse failures and the gap.

Do not use uniform nf4 4-bit on this architecture. The router cannot tolerate uniform low-bit
quantization; an activation-aware method that protects router and attention weights is required
below Q8. Foundation check: the untrained Qwen3-30B-A3B base produced 0/121 parseable judge
responses — the entire judge capability comes from the fine-tune.

The bf16 GGUF seeds and the base-model GGUF are **not published** in this repo (size, no
deployment need) — only the Q8_0 tier ships. A bf16 safetensors export of the judge for
Transformers/vLLM loading is also not shipped (the export was not produced complete); use this
GGUF tier for the judge, or the [paired gen repo](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2)'s
bf16 safetensors format if you need a Transformers-native checkpoint architecture reference.

## Usage (llama.cpp / Ollama)

3-seed median ensemble (recommended, matches the reported rho 0.8075/0.8056):

```bash
# serve each seed and take the median of the three overall scores per review
for s in wp-v1.3-judge-s0.Q8_0.gguf wp-v1.3-judge-s1.Q8_0.gguf wp-v1.3-judge-s2.Q8_0.gguf; do
  ~/llama.cpp/build/bin/llama-server -m "$s" -ngl 999 -c 8192 --jinja -a wp_judge --port 8091 &
done
```

Send one `<wp_judge>` prompt to each seed's `/v1/chat/completions`, parse each response's
`<judge_output>` JSON block (9 dimension scores + verdict), take the per-dimension or overall-score
median across the three responses.

Single-seed fallback (leaner, trades ~0.006 rho for one-third the serve cost — use `s1`):

```bash
~/llama.cpp/build/bin/llama-server -m wp-v1.3-judge-s1.Q8_0.gguf -ngl 999 -c 8192 --jinja -a wp_judge
```

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"wp_judge","messages":[{"role":"user","content":"<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction get_user($id){ global $wpdb; return $wpdb->get_row(\"SELECT * FROM wp_users WHERE ID=$id\"); }\n```"}],"max_tokens":2048}'
```

Output is a 9-dimension rubric critique (`<judge_output>` JSON with `wpcs_compliance`,
`sql_safety`, `security`, `performance`, `wp_api_usage`, `code_quality`, `dependency_integrity`,
`i18n`, `accessibility`, and `verdict`/`overall_score`) — e.g. the unprepared `$wpdb->get_row`
interpolation above is flagged as a security/SQL-safety defect.

Ollama: `ollama create wp-judge -f Modelfile` pointing `FROM ./wp-v1.3-judge-s1.Q8_0.gguf`, then
prompt with `<wp_judge> ...` the same way.

`<wp_judge>` is an ordinary prompt-prefix string, not a special tokenizer control token — the
GGUF's embedded chat template handles formatting when served with `--jinja`.

## Benchmarks

Full pair benchmarks (wp-bench, SWE-bench) are reported on the [gen repo](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2)
card. Judge-specific quality is the rho figures above.

## Evaluation

- **Judge rho** (Spearman vs human-relabeled 9-dim scores): ensemble 0.8075 (vLLM bf16 reference),
  Q8_0 GGUF ensemble 0.8056 (lossless, Δ−0.4pp), single-seed s1 0.8017. Recalibrated floors
  0.7554 / 0.7497. Attenuation ceiling for the noisy val set is ~0.984; the ~0.16 residual is a
  genuine capability wall for SFT on this base, not a bug.
- 9 rubric dimensions: WPCS, security, SQL safety, performance, WP API usage, i18n,
  accessibility, code quality, dependency integrity.

## Limitations

- Two-model pair, not a single unified model. Both share the base; route by task token.
- No RL enhancement (rejected). No pruning speedup (no viable prune). Size reduction depends
  entirely on quantization; Q8_0 is the floor that stays lossless — the naive 4-bit tier is
  unavailable (MoE router collapse).
- Judge rho plateaus around 0.80-0.83; a materially better judge needs a stronger base
  (Qwen3.6-class), which is the intended next iteration of this exact pipeline.

## Provenance

Full training/eval history and both negative pruning results are documented in the project
repository: `JOURNAL.md`, `PIPELINE.md`, `output/prune/prune_methodology.md`,
`output/eval3/eval3_final_comparison.json`.
