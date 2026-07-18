# Qwen 3 WP Judge

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-wp--judge--v4--gguf-yellow.svg)](https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf)
[![Base Model: Qwen3.6-35B-A3B](https://img.shields.io/badge/Base-Qwen3.6--35B--A3B-purple.svg)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
[![Infrastructure: DGX Spark](https://img.shields.io/badge/Infrastructure-DGX_Spark-76b900.svg)](https://github.com/dr-robert-li/dgx-toolbox)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-orange.svg)](https://claude.com/claude-code)

**Author:** [Dr. Robert Li](https://github.com/dr-robert-li)

An open-weight WordPress code **reviewer**. Give it a PHP function, it returns a structured critique
scored across 9 WordPress-quality dimensions (WPCS compliance, SQL safety, security, performance, WP API
usage, code quality, dependencies, i18n, accessibility) and tells you what is wrong and why.

The model is `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` — a fine-tune of Qwen3.6-35B-A3B, expert-pruned
to 224/256 experts, shipped as a Q6_K GGUF (23.47 GiB). It is self-hostable, needs no external API, and is
deliberately opinionated: it pushes back on unsafe SQL, missing nonce checks, and poor architecture instead
of rubber-stamping. Measured judge rho on the shipped stack (single-seed, Q6_K GGUF/llama.cpp): **0.8063**.

The prior `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` (Qwen3-30B-A3B base) remains available on
HuggingFace as the superseded prior artifact — it is not updated or removed by this release.

> **Why a judge and not a generator?** Reviewing is the capability worth training. The untrained base
> produces **0 parseable rubric verdicts out of 121** — the judge is a capability created from nothing.
> Code *generation*, by contrast, is already solved by strong base models: in the v4.0 study a raw
> Qwen3.6-35B-A3B out-scored every fine-tuned generator we produced. So this project ships the judge and
> recommends a current base model for generation. See [MODEL_CARD.md](output/packaging/MODEL_CARD.md).

## The model

| Property | Value |
|----------|-------|
| Repository | [`iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`](https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf) |
| Base | Qwen3.6-35B-A3B (native MoE, expert-pruned 256→224, top-8 routing) |
| Training | relabel-SFT judge fine-tune, AIMER weight-prune post-merge (k=224) |
| Ship tier | Q6_K GGUF, 23.47 GiB — smallest tier with zero parse failures |
| Variant | `wp-judge-v4-unpruned.Q5_K_M.gguf`, 23.61 GiB — unpruned 256/256 experts, **MTP speculative decoding** (`--spec-type draft-mtp`, 56% measured draft acceptance) |
| Serving | llama.cpp (GGUF) |
| Judge rho | **0.8063**, single-seed s1, shipped Q6_K/llama.cpp stack (unpruned variant: 0.8093 vs its own f16 floor — statistically tied) |
| Prior release | [`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf) — superseded, Qwen3-30B-A3B base, stays live untouched |
| Model card | [output/packaging/MODEL_CARD.md](output/packaging/MODEL_CARD.md) |

## Quickstart

```bash
# 1. pull the GGUF
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.Q6_K.gguf --local-dir ./judge

# 2. serve (llama.cpp b9180+; -c 16384 so long critiques never truncate)
llama-server -m ./judge/wp-judge-v4-pruned-k224.Q6_K.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja
```

Want speculative decoding? Pull the unpruned variant instead and add `--spec-type draft-mtp` — the model's
own MTP head drafts (56% measured acceptance), no separate draft model needed. Same judge quality, +150 MB:

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-unpruned.Q5_K_M.gguf --local-dir ./judge
llama-server -m ./judge/wp-judge-v4-unpruned.Q5_K_M.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja --spec-type draft-mtp
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

`<wp_judge>` is a plain-text prompt prefix, not a special token — no tokenizer surgery is required. v4
ships as a single-seed checkpoint (no 3-seed ensemble measured at this base).

## Benchmarks

Receipt-backed, shipping stack, 0/121 parse failures. Full detail: [MODEL_CARD.md](output/packaging/MODEL_CARD.md#benchmarks).

| Metric | Score | Notes |
|---|---|---|
| **Judge rho** — Q6_K GGUF, single-seed s1 (shipped) | **0.8063** | 0/121 parse failures |
| Judge rho — f16 GGUF, single-seed s1 (uncompressed floor) | 0.8002 | statistically tied with Q6_K |
| Judge rho — untrained base | **0/121 parseable** | the entire judge capability is the fine-tune |

All four GGUF rungs measured (f16/Q8_0/Q6_K/Q5_K_M) are statistically indistinguishable at n=121 (95% CI
half-widths ~7-8pp); Q6_K ships as the smallest zero-parse-failure tier, not the highest-scoring one. The
unpruned variant ran its own four-rung ladder against its own f16 floor (0.8081) with the same outcome —
statistically flat, tier selected on parse reliability + size (its Q5_K_M ran clean at 0/121, unlike the
pruned checkpoint's Q5). Spearman rho is measured against human-relabeled 9-dimension scores on a held-out
set of 121 items.

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
4. **Merge + quantize** — merge each seed's adapter into the base, convert to GGUF (llama.cpp) and walk a
   quantization ladder (Q8→Q6→Q5) against the checkpoint's own f16 floor; ship the smallest rung that
   stays in-band with zero parse failures. (On v3 that was Q8_0, measured lossless; on v4 it was Q6_K
   pruned / Q5_K_M unpruned — the rung is a measurement, not a constant.)

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
- **The judge did improve, but not enough to re-ship as-is.** The Qwen3.6 judge beats the old base on the
  capture path (rho 0.8358 vs 0.8274), but a serving-stack numerics ceiling (~0.79, identical across
  vLLM-merged, llama.cpp-Q8-merged, and llama.cpp-unmerged-LoRA) eats the gain. On the shipped Q8 stack it
  scored **0.8067 vs v3's 0.8056** — a statistical tie (paired bootstrap CI spans zero) at +25% artifact
  size. Per the pre-registered rule, **v1.3 stayed canonical at that point** — this was resolved by the
  prune below.

That lever resolved. v4's judge had a tie on **128**-vs-**256** experts the compression pipeline had never
touched on a 256-expert base; v3.0 found no MoE-Sieve or prune winner on 128 experts, and the v4 roadmap
flagged 256 experts + a shared expert as exactly the architecture where that might flip — it did. AIMER
weight-pruning at k=224 passed gate-before-remove (Phases 22/25/26), producing a 224/256-expert checkpoint
that ships at **Q6_K, 23.47 GiB** — smaller than v3's 30.2 GiB, on the newer base, at statistically tied
quality (judge rho 0.8063 single-seed vs v3's 0.8056 3-seed ensemble — different configurations, not a
clean delta). **Canonical flips to v4 (2026-07-17);** v1.3 remains published as the superseded prior
release. Either way, v4.0 already produced durable knowledge: the capture-vs-served ceiling quantified
across three engines, hardened routed-expert merge tooling, and a fully recorded negative result on the
un-pruned checkpoint.

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

Apache 2.0. Base models Qwen3.6-35B-A3B (v4, canonical) and Qwen3-30B-A3B (v1.3, superseded prior release)
are both Apache 2.0.

**Building in public.** [JOURNAL.md](JOURNAL.md) is the unedited decision log — every tradeoff, dead end,
and recorded miss across four milestones.
