---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
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

# WP Judge v4 (Qwen3.6-35B-A3B)

An open-weight WordPress code **reviewer**. Give it a PHP function, it returns a structured critique
scored across 9 WordPress-quality dimensions (WPCS compliance, SQL safety, security, performance, WP API
usage, code quality, dependencies, i18n, accessibility) and tells you what is wrong and why.

**It judges, it does not generate.** For WordPress code generation, use the current base model directly —
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) with prompt-side task framing. No
generation fine-tune is shipped alongside this judge.

## Acquisition

Repo: [`iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`](https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf)

| File | Quant | Size | MTP / speculative decoding |
|---|---|---|---|
| `wp-judge-v4-unpruned.Q5_K_M.gguf` (recommended) | Q5_K_M | 23.61 GiB | yes |
| `wp-judge-v4-pruned-k224.Q6_K.gguf` | Q6_K | 23.47 GiB | no |

Both judge identically within measurement noise (see Performance). Take the recommended file —
it supports speculative decoding (faster serving) at the same download size to within 150 MB.
The pruned file is for operators who want the marginally smaller expert footprint (224/256
experts) and don't need speculation.

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-unpruned.Q5_K_M.gguf --local-dir ./judge
```

## Use

Serve with llama.cpp (b9180+):

```bash
llama-server -m ./judge/wp-judge-v4-unpruned.Q5_K_M.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja -a wp-judge-v4 --parallel 4 --spec-type draft-mtp
```

`--spec-type draft-mtp` enables speculative decoding via the model's own MTP head (56% measured
draft acceptance — no separate draft model). Omit it (or use the pruned file) if you don't want it.

`--jinja` is load-bearing — the chat template drives the rubric output format.

The pruned file ignores `--spec-type draft-mtp` (no MTP head) but is otherwise served identically.

**Ollama** (pulls the GGUF straight from this repo):

```bash
ollama run hf.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf:Q5_K_M
```

Two things matter on Ollama: set the context window — the default truncates long critiques mid-rubric
(`/set parameter num_ctx 16384` in the session, or `PARAMETER num_ctx 16384` in a Modelfile) — and note
that Ollama does not expose llama.cpp's `--spec-type` flag, so MTP speculative decoding is unavailable
there (both files serve at the same speed). The chat template is read from the GGUF metadata.

All published quality numbers were measured on llama.cpp; other engines (Ollama, LM Studio,
llama-cpp-python) run the same GGUF but their numerics are unverified here.

Judge a snippet — prepend the `<wp_judge>` task prefix and the model returns per-dimension reasoning
followed by a structured verdict:

```bash
curl -s http://127.0.0.1:8020/v1/chat/completions -d '{
  "model": "wp-judge-v4",
  "messages": [{"role": "user",
    "content": "<wp_judge> Evaluate this WordPress code:\n\n```php\n<?php $wpdb->query(\"SELECT * FROM wp_posts WHERE ID=$id\"); ?>\n```"}],
  "temperature": 0.0, "max_tokens": 2048
}'
# -> reasoning per dimension, then a <judge_output> JSON block:
# {"verdict": "FAIL", "wpcs_compliance": 9, "security": 2, "sql_safety": 1, ..., "overall_score": 41}
```

`<wp_judge>` is a plain-text prompt prefix, not a special token. Dimensions that don't apply to a given
snippet (e.g. accessibility on a REST handler) are omitted from the JSON rather than scored.

## Performance

Spearman rho vs. human-relabeled 9-dimension scores, n=121 held-out items, single-seed:

| Stack / config | rho | Notes |
|---|---|---|
| f16 GGUF, llama.cpp | 0.8002 | uncompressed floor |
| **Q6_K GGUF, llama.cpp (shipped)** | **0.8063** | 0 parse failures |
| Q8_0 GGUF, llama.cpp | 0.7851 | 0 parse failures |

All three rungs are statistically indistinguishable at this sample size (95% CI half-widths ~7-8pp). Q6_K
ships as the smallest tier with zero parse failures, not because it scored highest.

The unpruned variant measured rho 0.8093 (Q5_K_M, same protocol, 0 parse failures) against its own f16
floor of 0.8081 — statistically indistinguishable from the pruned file's numbers above. Its quant tier was
selected by the same rule: smallest rung within ±2pp of its own f16 floor with zero parse failures.

The prior WP Judge (`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`, Qwen3-30B-A3B base, 30.2 GiB) reports
0.8056 on a **3-seed ensemble** — a different stack and seed configuration than the single-seed number
above, and not directly comparable to it.

## Variants

| | `wp-judge-v4-unpruned.Q5_K_M.gguf` (recommended) | `wp-judge-v4-pruned-k224.Q6_K.gguf` |
|---|---|---|
| Experts | 256/256 (unpruned) | 224/256 (expert-pruned) |
| MTP / speculative decoding | **available** (`--spec-type draft-mtp`, 56% measured draft acceptance) | **not available** |
| Size | 23.61 GiB | 23.47 GiB |
| Judge quality | statistically tied (see Performance) | statistically tied (see Performance) |

Why the pruned file has no MTP head: expert-pruning left the MTP layer at a different expert count than
the trunk, and GGUF's expert-count metadata must be uniform, so the MTP tensors were dropped at its
conversion (`--no-mtp`). Standard generation and judging are unaffected in both files. The two files are
statistically indistinguishable on judge quality; the recommendation is purely serving speed.

## Links

- Base: fine-tuned from [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), then quantized to GGUF. The recommended file keeps all 256 experts (MTP intact); the pruned variant is additionally expert-pruned to 224/256.
- How this was built: [github.com/dr-robert-li/wp-finetune](https://github.com/dr-robert-li/wp-finetune) — see `PIPELINE.md`, `JOURNAL.md`, `CHANGELOG.md`.
- License: Apache 2.0.
