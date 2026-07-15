# Qwen 3 WP Judge

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-wp--judge--v1.3--gguf-yellow.svg)](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf)
[![Base Model: Qwen3-30B-A3B](https://img.shields.io/badge/Base-Qwen3--30B--A3B-purple.svg)](https://huggingface.co/Qwen/Qwen3-30B-A3B)
[![Infrastructure: DGX Spark](https://img.shields.io/badge/Infrastructure-DGX_Spark-76b900.svg)](https://github.com/dr-robert-li/dgx-toolbox)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-orange.svg)](https://claude.com/claude-code)

**Author:** [Dr. Robert Li](https://github.com/dr-robert-li)

An open-weight WordPress code **reviewer**. Give it a PHP function, it returns a structured critique
scored across 9 WordPress-quality dimensions (WPCS compliance, SQL safety, security, performance, WP API
usage, code quality, dependencies, i18n, accessibility) and tells you what is wrong and why.

The model is `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` — a LoRA fine-tune of Qwen3-30B-A3B, shipped as
a lossless Q8_0 GGUF. It is self-hostable, needs no external API, and is deliberately opinionated: it
pushes back on unsafe SQL, missing nonce checks, and poor architecture instead of rubber-stamping.

> **Why a judge and not a generator?** Reviewing is the capability worth training. The untrained base
> produces **0 parseable rubric verdicts out of 121** — the judge is a capability created from nothing.
> Code *generation*, by contrast, is already solved by strong base models: in the v4.0 study a raw
> Qwen3.6-35B-A3B out-scored every fine-tuned generator we produced. So this project ships the judge and
> recommends a current base model for generation. See [The v4.0 finding](#the-v40-finding-qwen36).

## The model

| Property | Value |
|----------|-------|
| Repository | [`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf) |
| Base | Qwen3-30B-A3B (native MoE, 128 experts, top-8 routing, ~3B active) |
| Training | 3-epoch rank-32 MoE-only LoRA on 603 human-relabeled 9-dim scores |
| Ship tier | Q8_0 GGUF, 30.2 GiB/seed (−47% vs bf16, lossless) |
| Serving | llama.cpp / Ollama (GGUF), or vLLM (bf16) |
| Ensemble | 3 seeds (s0/s1/s2), median score; single-seed **s1** is the documented fallback |
| Model card | [output/packaging/MODEL_CARD.md](output/packaging/MODEL_CARD.md) |

## Quickstart

Download the ensemble and serve one seed with llama.cpp (single-seed `s1` is the cheapest usable config):

```bash
# 1. pull the GGUFs (three Q8_0 seeds, ~30 GiB each)
huggingface-cli download iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf \
  wp-v1.3-judge-s1.Q8_0.gguf --local-dir ./judge

# 2. serve (llama.cpp b9180+; -c 16384 so long critiques never truncate)
llama-server -m ./judge/wp-v1.3-judge-s1.Q8_0.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja
```

Judge a snippet — prepend the `<wp_judge>` task prefix and the model returns a 9-dimension rubric verdict:

```bash
curl -s http://127.0.0.1:8020/v1/chat/completions -d '{
  "messages": [{"role": "user",
    "content": "<wp_judge>Review this code:\n<?php $wpdb->query(\"SELECT * FROM wp_posts WHERE ID=$id\"); ?>"}],
  "temperature": 0.0, "max_tokens": 8192
}'
# → structured critique: flags the unprepared SQL as a D2_security FAIL, scores each dimension,
#   overall PASS/FAIL verdict. Parse the JSON block from the response.
```

For the full-fidelity number, run all three seeds and take the per-item median (see
[PIPELINE.md](PIPELINE.md) → *Stage 4 / packaging eval*). `<wp_judge>` is a plain-text prompt prefix, not
a special token — no tokenizer surgery is required to use the model.

## Benchmarks

Receipt-backed, shipping stack, 0/121 parse failures. Full detail and the out-of-domain protocol:
[MODEL_CARD.md](output/packaging/MODEL_CARD.md#benchmarks).

| Metric | Score | Notes |
|---|---|---|
| **Judge rho** — Q8 GGUF 3-seed ensemble | **0.8056** | shipping tier; lossless vs bf16 (0.8100), −47% size |
| Judge rho — bf16 ensemble (vLLM ref) | 0.8075 | single-seed s1 fallback 0.8017 |
| Judge rho — untrained base | **0/121 parseable** | the entire judge capability is the fine-tune |
| Attenuation ceiling (noisy val set) | ~0.984 | the ~0.16 residual is a genuine SFT wall on this base, not a bug |

Spearman rho is measured against human-relabeled 9-dimension scores on a held-out set of 121 items.

## How it was built (and how to recreate it)

The full, frozen method — every stage with its runnable entrypoint, pass/fail gate, and the known result —
is in **[PIPELINE.md](PIPELINE.md)**. In brief:

1. **Data** — clone 236 WordPress repos (top plugins/themes + deliberately poor-quality ones + WP Core),
   extract PHP functions, score each against the [9-dimension rubric](config/judge_system.md) with PHPCS
   pre-filtering and Claude Code agents. Poor-quality code becomes negative judge training data. See
   [docs/AGENT_PIPELINE.md](docs/AGENT_PIPELINE.md).
2. **Relabel** — 603 items re-scored by hand against the frozen rubric, rebuilt into judge SFT targets.
   Human ground truth is what makes the judge portable across base models.
3. **Fine-tune** — rank-32 MoE-only LoRA, frozen router, 3 epochs, 3 seeds, via [Tinker](https://thinkingmachines.ai).
4. **Merge + quantize** — merge each seed's adapter into the base, convert to Q8_0 GGUF (llama.cpp),
   verify rho is lossless vs bf16 at an 8192-token cap.

LLM-heavy steps run through **Claude Code agents** (subscription, no per-token API cost), not direct API
calls. Training, serving, and eval run on a single [DGX Spark (GB10)](https://github.com/dr-robert-li/dgx-toolbox)
via the DGX Toolbox execution engine. One-off experiment scaffolding from earlier campaigns lives in
[deprecated/](deprecated/), off the pipeline path.

## The v4.0 finding (Qwen3.6)

v4.0 reran the entire pipeline on the newer `Qwen3.6-35B-A3B` to try to beat the shipped judge. The
[full diagnostic](output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md) is receipt-backed; the short version:

- **Generation is a solved problem now.** The raw Qwen3.6 base scored wp-bench **0.4897**; every gen
  fine-tune we trained *regressed* below it (best 0.4381), because the training targets are structurally
  weaker than what a modern base already writes. So the project drops the gen model as a deliverable — for
  generation, use a strong current base with prompt-side task framing.
- **The judge did improve, but not enough to re-ship.** The Qwen3.6 judge beats the old base on the
  capture path (rho 0.8358 vs 0.8274), but a serving-stack numerics ceiling (~0.79, identical across
  vLLM-merged, llama.cpp-Q8-merged, and llama.cpp-unmerged-LoRA) eats the gain. On the shipped Q8 stack it
  scored **0.8067 vs v3's 0.8056** — a statistical tie (paired bootstrap CI spans zero) at +25% artifact
  size. Per the pre-registered rule, **v1.3 stays canonical.**

One lever is still in play. v4's judge is a tie on **128**-vs-**256** experts the compression pipeline has
never touched on a 256-expert base. v3.0 found no MoE-Sieve or prune winner on 128 experts; the v4 roadmap
flagged 256 experts + a shared expert as exactly the architecture where that might flip. If Sieve/prune
shrinks the v4 judge below v3's 30.2 GiB, it becomes unequivocally better (newer base, tied quality,
smaller) and gets published. That attempt is **in progress** (Phases 22/25/26). Until it resolves, v1.3
stays the canonical recommendation and the v4 judge stays on the bench. Either way, v4.0 already produced
durable knowledge: the capture-vs-served ceiling quantified across three engines, hardened routed-expert
merge tooling, and a fully recorded negative result.

## Repository layout

```
config/     rubric (judge_system.md), repos.yaml, training + benchmark configs
scripts/    data pipeline, training, merge, quantization, eval drivers
eval/       9-dimension rubric scorer + wp-bench harness
docs/       agent execution model + wp-finetune:* operator skills
output/     receipts (benchmarks, packaging, v4.0 diagnostic), MODEL_CARD.md
deprecated/ frozen one-off scaffolding + superseded specs (incl. wp-moe.md)
PIPELINE.md end-to-end method, gate by gate     JOURNAL.md  build log
PROJECT.md  project spec + status               CHANGELOG.md
```

## Requirements

- Python 3.10+, `pyyaml`, `python-dotenv`
- PHP CLI (`tokenizer` extension), PHP_CodeSniffer + WordPress-Coding-Standards (for the data pipeline)
- llama.cpp b9180+ or Ollama (to serve the GGUF), or vLLM (bf16)
- [Claude Code](https://claude.com/claude-code) — LLM pipeline steps
- [DGX Toolbox](https://github.com/dr-robert-li/dgx-toolbox) — training, eval, serving

## License

Apache 2.0. Base model Qwen3-30B-A3B is Apache 2.0.

**Building in public.** [JOURNAL.md](JOURNAL.md) is the unedited decision log — every tradeoff, dead end,
and recorded miss across four milestones.
