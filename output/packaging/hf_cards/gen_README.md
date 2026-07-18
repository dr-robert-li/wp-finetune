---
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B
pipeline_tag: text-generation
language:
  - en
tags:
  - wordpress
  - php
  - code-generation
  - moe
  - qwen3
---

# wp-qwen3-30b-a3b-wp-gen-v1.2 — WordPress code generation (Qwen3-30B-A3B)

Generation half of a two-model WordPress pair fine-tuned from Qwen3-30B-A3B. This model writes
WPCS-compliant PHP from a plain-English instruction. The paired review/judge model lives at
[iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf)
— route generation requests here with `<wp_gen>` and review requests there with `<wp_judge>`.

This card documents the full v3.0 compression lineage, including the stages that returned nothing.
Two of them did, and that's recorded honestly rather than hidden.

## The pair

| Role | Token | Model | wp-bench / judge rho (vLLM) |
|---|---|---|---|
| **Generation (this repo)** | `<wp_gen>` | v1.2 reasoning-merged (30/70 replay) | wp-bench **0.4484** |
| Review / judge | `<wp_judge>` | v1.3 relabel-SFT, 3-seed median ensemble | rho **0.8075** (single-seed s1 **0.8017**) |

- Base: Qwen3-30B-A3B (MoE, 128 experts, top-8, ~30.5B total / ~3.3B active, 48 layers).
- This repo ships the bf16 checkpoint: 57 GB, 13 safetensors shards, plus tokenizer
  (`tokenizer.json` carries the `<wp_gen>` / `<wp_judge>` task-token vocabulary).
- No quantized tier is shipped for the generation model — the Q8 evidence below applies to the
  judge model specifically (see the paired GGUF repo). bf16 is the only ship tier here.

## Compression lineage (base -> RL -> MoE-Sieve -> merge -> prune -> quantize)

Each arrow is a real gate, applied to the pair as a whole. Scores are vLLM-served unless noted.

1. **Base** — Qwen/Qwen3-30B-A3B.
2. **SFT (v1.2 / v1.3)** — reasoning SFT for generation (v1.2, this repo), relabel-SFT for the judge
   (v1.3, rho 0.827 Tinker-runtime / 0.8017 vLLM single-seed).
3. **RL (GSPO)** — **REJECTED.** Warm-started from v1.3 with an oracle-passed calibration reward;
   killed on 6/6 dead checkpoint reads (2026-07-05). No RL checkpoint was promoted.
4. **MoE-Sieve (expert drop)** — **no compression.** Routing profile shows ~88-99 effective
   experts/layer of 128; every masked-k budget cuts live capacity. wp-bench collapses
   0.4484 (full) -> 0.2275 (k=64) -> 0.0546 (k=32). `optimal_k = full`; nothing dropped.
5. **LoRA merge** — adapters merged into base weights; merged output matches adapter-on-base.
   This repo ships the merged checkpoint.
6. **AIMER / REAP pruning** — **no winner.** Weight-norm AIMER at the lightest ratio (25%)
   collapses generation to wp-bench 0.1577 and judge ensemble rho to 0.1651 (parse 44.6%).
   50/75% skipped, REAP conditional-skipped per the pre-registered rule. Ships unpruned at
   full 128-expert width.
7. **Quantization** — measured on the judge model (see the paired GGUF repo); uniform 4-bit nf4
   is a failure, Q8 is the recommended ship tier for the judge. This generation model ships bf16.

Net: the v3.0 pipeline confirmed the model **cannot be shrunk by expert-count or weight-norm
methods** on this workload. Quantization is the only size reduction available, and it was applied
to the judge (this generation repo ships full-precision bf16).

## Usage (vLLM)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2", trust_remote_code=True, dtype="bfloat16")
sp = SamplingParams(temperature=0.0, max_tokens=1024)

prompt = "<wp_gen> Write a WordPress function to safely query posts by meta value."
out = llm.generate([prompt], sp, chat_template_kwargs={"enable_thinking": False})
print(out[0].outputs[0].text)  # WPCS-compliant PHP
```

Or via `vllm serve iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2 --trust-remote-code` and the OpenAI-compatible
`/v1/chat/completions` endpoint, sending `chat_template_kwargs: {"enable_thinking": false}` per request.
`<wp_gen>` is an ordinary prompt-prefix string, not a special tokenizer control token — just prepend it
to the instruction; the shipped `tokenizer.json` / `chat_template.jinja` handle the rest.

For review/critique of existing code, send the same prompt shape with `<wp_judge>` to the
[paired judge repo](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf) instead.

## Benchmarks

Fresh, receipt-backed measurements on the shipping stack (vLLM bf16, this checkpoint,
temperature 0.0), taken 2026-07-11.

| Benchmark | Score | Scope / config |
|---|---|---|
| **wp-bench** (in-domain) | **0.4365** overall | full 344-test wp-core-v1 suite, unlimited; knowledge 0.4906, correctness 0.3958 |
| **SWE-bench Lite** (out-of-domain) | **1.67%** resolved (5/300) | generation-mode (non-agentic), oracle retrieval, 24k context, native arm64 local Docker eval |
| **SWE-bench-Multilingual PHP subset** (in-language, out-of-domain) | **0%** resolved (0/43) | same protocol; 4 PHP repos (phpspreadsheet, laravel, php-cs-fixer, carbon) |

**wp-bench.** The fresh full-suite score (0.4365) sits 1.19pp below the 0.4484 Gate-1 reference
figure, well inside the project's measured 5.20pp seed-noise floor, and clears the 0.4286
acceptance bar. Same stack both sides (vLLM bf16, identical checkpoint and sampling); no
regression signal.

**SWE-bench, and why the number is low.** SWE-bench Lite is Python-repository patch generation;
this model is WordPress/PHP-specialized. A low out-of-domain number is expected and is not a
quality defect: the model was never trained to patch Django or sympy, and the score is published
for honest positioning, not vanity. Protocol: one oracle-retrieval prompt in, one unified-diff
patch out (no agent scaffold, no retries), temperature 0.0, seed 0, max_model_len 24576,
official swebench 4.1.0 containerized harness built natively for arm64. The scope (Lite-300 +
PHP-43) was pre-registered before any result existed.

Full-scope denominators count everything against the model: of Lite's 300, 80 prompts exceeded
the 24k serving context (scored unresolved), 1 generation was unparseable, 59 patches failed to
apply, and 29 instances could not be evaluated because their 2018-era Python environments cannot
build on arm64/2026 toolchains (also scored unresolved, conservative against the model). On the
131 instances that ran to a test verdict the resolved rate is 3.82% (5/131). The PHP-Multilingual
subset is in-language but still out-of-domain (framework libraries, not WordPress): 0/43
full-scope, 0/20 evaluated; 17 over-length, 6 apply-failures.

## Evaluation

- **wp-bench** (WordPress code correctness/knowledge/quality): generation 0.4484 (bar 0.4286);
  fresh full-suite rerun on the shipping stack 0.4365 (within seed noise — see Benchmarks).
- The pair's judge rho (Spearman vs human-relabeled 9-dim scores) is measured on the paired judge
  repo: ensemble 0.8075, single-seed 0.8017.

## Limitations

- Two-model pair, not a single unified model. Both share the base; route by task token.
- No RL enhancement (rejected). No pruning speedup (no viable prune). This generation model ships
  bf16 only — no quantized tier was evaluated for generation.
- A materially better pair needs a stronger base (Qwen3.6-class), which is the intended next
  iteration of this exact pipeline.

## Provenance

Full training/eval history and both negative pruning results are documented in the project
repository: `JOURNAL.md`, `PIPELINE.md`, `output/prune/prune_methodology.md`,
`output/eval3/eval3_final_comparison.json`.
