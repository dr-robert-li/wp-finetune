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

| File | Quant | Size |
|---|---|---|
| `wp-judge-v4-pruned-k224.Q6_K.gguf` | Q6_K | 23.47 GiB |

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.Q6_K.gguf --local-dir ./judge
```

## Use

Serve with llama.cpp (b9180+):

```bash
llama-server -m ./judge/wp-judge-v4-pruned-k224.Q6_K.gguf --host 127.0.0.1 --port 8020 \
  -ngl 999 -c 16384 --jinja -a wp-judge-v4 --parallel 4
```

`--jinja` is load-bearing — the chat template drives the rubric output format.

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

The prior WP Judge (`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`, Qwen3-30B-A3B base, 30.2 GiB) reports
0.8056 on a **3-seed ensemble** — a different stack and seed configuration than the single-seed number
above, and not directly comparable to it.

## Known limitation

This GGUF has **no MTP / speculative-decoding head**. The checkpoint's expert-pruning left the MTP layer
at a different expert count than the trunk, and GGUF's expert-count metadata must be uniform, so the MTP
tensors were dropped at conversion (`--no-mtp`). Speculative decoding is not available with this file;
standard generation and judging are unaffected.

## Links

- Base: fine-tuned and expert-pruned (224/256 experts) from [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), then quantized to GGUF.
- How this was built: [github.com/dr-robert-li/wp-finetune](https://github.com/dr-robert-li/wp-finetune) — see `PIPELINE.md`, `JOURNAL.md`, `CHANGELOG.md`.
- License: Apache 2.0.
