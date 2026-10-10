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
to 224/256 experts and shipped as GGUF in three sizes: **IQ4_XS (15.44 GiB, default)**, a small mixed file
(11.99 GiB, experts at IQ3_XXS), and Q6_K (23.47 GiB, max fidelity). An unpruned Q5_K_M (23.61 GiB) keeps
the MTP head for speculative decoding. It is self-hostable, needs no external API, and is deliberately
opinionated: it pushes back on unsafe SQL, missing nonce checks, and poor architecture instead of
rubber-stamping. Measured judge rho (single-seed, llama.cpp): 0.8063 (Q6_K) / 0.8093 (unpruned Q5_K_M) —
statistically tied. The smaller files track the Q6_K scores at Spearman 0.935 (IQ4_XS) and 0.921 (small).

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
| Base | Qwen3.6-35B-A3B (native MoE, 256 experts, top-8 routing) |
| Training | relabel-SFT judge fine-tune |
| Default file | `wp-judge-v4-pruned-k224.IQ4_XS.gguf`, 15.44 GiB, ~16 GB GPU, 69 tok/s — Spearman 0.935 vs Q6_K scores, same PASS/FAIL on 99.2%, 1/121 repetition loop |
| Small file | `wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf`, 11.99 GiB, 12314 MiB GPU, 60 tok/s — Spearman 0.921, same PASS/FAIL on 96.7% (conditions below) |
| Max-fidelity file | `wp-judge-v4-pruned-k224.Q6_K.gguf`, 23.47 GiB, ~24 GB GPU, 55 tok/s — AIMER k=224 expert-prune (224/256 experts; no MTP); the reference the smaller files are measured against |
| MTP variant | `wp-judge-v4-unpruned.Q5_K_M.gguf`, 23.61 GiB — 256/256 experts, **MTP speculative decoding** (`--spec-type draft-mtp`, 56% measured draft acceptance) |
| Serving | llama.cpp (GGUF) |
| Judge rho | 0.8063 Q6_K / 0.8093 unpruned Q5_K_M — single-seed s1, each vs its own f16 floor, statistically tied |
| Prior release | [`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf) — superseded, Qwen3-30B-A3B base, stays live untouched |
| Model card | [output/packaging/MODEL_CARD.md](output/packaging/MODEL_CARD.md) |

## Quickstart

```bash
# 1. pull the GGUF (default file: IQ4_XS, 15.44 GiB)
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.IQ4_XS.gguf --local-dir ./judge

# 2. serve (llama.cpp b9180+; -c 16384 so long critiques never truncate)
llama-server -m ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja --chat-template-kwargs '{"enable_thinking": false}'
```

Other files, same command with a different `-m`:

- **Small** (`wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf`, 11.99 GiB). Use the default f16 context
  cache only; do not add `-ctk q8_0` / `-ctv q8_0` (that combination fails the agreement bar). Expect about
  4x more PASS/FAIL flips than IQ4_XS. Set `max_tokens` to 2048 and treat a response with no
  `<judge_output>` verdict as a failure (retry or flag).
- **Max fidelity** (`wp-judge-v4-pruned-k224.Q6_K.gguf`, 23.47 GiB). Low on GPU memory? Add
  `--n-cpu-moe 99` to keep the experts in system RAM: ~2.3 GB GPU + ~23 GB RAM, ~10 tok/s, same verdicts.
- **MTP variant** (`wp-judge-v4-unpruned.Q5_K_M.gguf`, 23.61 GiB). Add `--spec-type draft-mtp` for
  speculative decoding.

Judge a snippet — prepend the `<wp_judge>` task prefix and the model returns a 9-dimension rubric verdict:

```bash
curl -s http://127.0.0.1:8020/v1/chat/completions -d '{
  "messages": [{"role": "user",
    "content": "<wp_judge>Review this code:\n<?php $wpdb->query(\"SELECT * FROM wp_posts WHERE ID=$id\"); ?>"}],
  "temperature": 0.0, "max_tokens": 2048
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
| **Judge rho** — Q6_K GGUF, single-seed s1 (max-fidelity file) | **0.8063** | 0/121 parse failures |
| Judge rho — f16 GGUF, single-seed s1 (uncompressed floor) | 0.8002 | statistically tied with Q6_K |
| Judge rho — untrained base | **0/121 parseable** | the entire judge capability is the fine-tune |

All four GGUF rungs measured (f16/Q8_0/Q6_K/Q5_K_M) are statistically indistinguishable at n=121 (95% CI
half-widths ~7-8pp); Q6_K was picked as the smallest zero-parse-failure tier, not the highest-scoring one. The
unpruned variant ran its own four-rung ladder against its own f16 floor (0.8081) with the same outcome —
statistically flat, tier selected on parse reliability + size (its Q5_K_M ran clean at 0/121, unlike the
pruned checkpoint's Q5). Spearman rho is measured against human-relabeled 9-dimension scores on a held-out
set of 121 items.

The IQ4_XS and small files were measured differently: Spearman agreement with the Q6_K scores on the same
121 prompts (greedy, fixed seed), not against the human labels. The pass bar was a bootstrap 95% CI lower
bound of 0.85 or more on two perturbed runs; the noise ceiling (same weights, different numeric path) is
~0.94-0.96. IQ4_XS: 0.935 [0.897, 0.960]. Small: 0.921 [0.878, 0.951]. Full table:
[output/shrink/RESULTS.md](output/shrink/RESULTS.md).

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
   pruned / Q5_K_M unpruned — the rung is a measurement, not a constant.) A later size study rebuilt the
   f16 source and picked IQ4_XS (importance matrix) as the default file — see `output/shrink/`.

LLM-heavy steps run through **Claude Code agents** (subscription, no per-token API cost), not direct API
calls. Training, serving, and eval run on a single [DGX Spark (GB10)](https://github.com/dr-robert-li/dgx-toolbox)
via the DGX Toolbox execution engine. One-off experiment scaffolding from earlier campaigns lives in
[deprecated/](deprecated/), off the pipeline path.

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

## Scan a whole plugin or theme (harness)

[`harness/`](harness/README.md) packages the judge as `wpjudge`. It extracts every PHP function from a
plugin or theme (folder, `.zip` or `.tar.gz`), judges each one through llama-server, and writes one JSON
report: LOC-weighted overall score, a safety cap at 60 when `security` or `sql_safety` ≤ 4, coverage,
dimension stats, patterns and anti-patterns.

```bash
pip install -e harness
wpjudge scan my-plugin.zip -o report.json          # needs llama-server on 127.0.0.1:8080
cd harness && docker compose up -d                 # or: async-job API on 127.0.0.1:8000 + llama.cpp
```

Use it as a CLI, a Python library (`from wpjudge import scan`) or a container API. About 11 units (functions or templates) per
minute on a GB10 with IQ4_XS. Operator guide, API reference and caveats: [harness/README.md](harness/README.md).

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
