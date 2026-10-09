---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
library_name: gguf
quantized_by: iamchum
language:
  - en
tags:
  - wordpress
  - php
  - code-review
  - llm-judge
  - moe
  - qwen3
  - gguf
  - imatrix
  - iq4_xs
  - iq3_xxs
  - q6_k
  - q5_k_m
  - llama.cpp
---

# WP Judge v4 (Qwen3.6-35B-A3B)

WP Judge v4 is an open-weight reviewer for WordPress PHP code.
You give it one PHP function. It returns a critique and a score for each quality dimension.
The dimensions are WPCS compliance, SQL safety, security, performance, WP API usage, code quality, dependencies, i18n and accessibility.

**This model judges code. It does not write code.**
To generate WordPress code, use the base model [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) with a task prompt.
We do not ship a code-generation fine-tune with this judge.

## Files

Repo: [`iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`](https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf)

Use the first file unless you have a specific reason to use a different file.

| File | Quant | Size (GiB) | GPU memory (MiB) | tok/s | agree_rho vs Q6_K [95% CI] | Verdict agreement | Unreadable (of 121) | Use case |
|---|---|---|---|---|---|---|---|---|
| `wp-judge-v4-pruned-k224.IQ4_XS.gguf` **(default)** | IQ4_XS, imatrix | 15.44 | 15992 | 69.0 | 0.935 [0.897, 0.960] | 99.2% | 1 | Default. 34% smaller and 25% faster than Q6_K. |
| `wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf` | IQ3_XXS experts, Q6_K other tensors, imatrix | 11.99 | 12314 | 59.7 | 0.921 [0.878, 0.951] | 96.7% | 0 | Small GPUs (about 12 GB free). Read the small-file conditions below. |
| `wp-judge-v4-pruned-k224.Q6_K.gguf` | Q6_K | 23.47 | 24074 | 55.4 | 1.000 (reference) | 100% | 0 | Maximum fidelity. This file is the reference for all agreement numbers. |
| `wp-judge-v4-unpruned.Q5_K_M.gguf` | Q5_K_M | 23.61 | not measured | not measured | not measured | not measured | 0 (see Performance) | All 256 experts and the MTP head. Use it for MTP speculative decoding in llama.cpp. |

The three `pruned-k224` files have 224 of 256 experts.
They do not have an MTP head (see [Pruned files and MTP](#pruned-files-and-mtp)).
The IQ4_XS and small files were quantized from the f16 source with an importance matrix (imatrix).

Download the default file:

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.IQ4_XS.gguf --local-dir ./judge
```

### Low-VRAM option

Use this option if your GPU does not have space for the full model.
Add `--n-cpu-moe 99` to `llama-server`.
This option keeps the expert weights in system RAM.
We measured it with the Q6_K file:

| GPU memory | System RAM | tok/s | agree_rho vs Q6_K | Verdict agreement |
|---|---|---|---|---|
| 2342 MiB | about 23 GB | 9.9 | 0.957 | 100% |

The verdicts stayed the same (100% agreement). The speed decreases by approximately 80%.

## Conditions for the small file

The small file (`small-IQ3_XXS-experts`) has more risk than the default file. Obey these conditions:

1. **Do not use an 8-bit KV cache.** Do not set `-ctk q8_0` or `-ctv q8_0`. Keep the default f16 KV cache. With a q8_0 KV cache, agree_rho decreased to 0.858 [0.797, 0.903]. This value is below our bar.
2. **Expect more verdict flips.** The small file agrees with the Q6_K PASS/FAIL verdict on 95.9% to 96.7% of items. The default file agrees on 99.2%. Thus the small file flips approximately 4 times more verdicts.
3. **Use the repetition guard.** Set `max_tokens` to 2048. If the output has no `<judge_output>` block, record the result as a failure. Then retry or flag the item. Do not parse a score from a truncated output.

The repetition guard is also good practice for the default file. In our test, the default file had one repetition loop that stopped at the 2048-token limit.

## Prompt format

- Do not use a system prompt. Send one user message only.
- Start the user message with the text `<wp_judge>`. This prefix is plain text. It is not a special token.
- Put the PHP code in a fenced `php` code block after the prefix.
- Turn off thinking. Send `"chat_template_kwargs": {"enable_thinking": false}` in each request. If your engine cannot send this field, turn off thinking with the engine setting.
- Use `temperature` 0.0 for repeatable scores.
- Set `max_tokens` to 2048.
- Set the context to 16384 tokens. A smaller context can cut a long critique before the `<judge_output>` block.

Use this user message template:

````text
<wp_judge> Evaluate this WordPress code:

```php
<your PHP function>
```
````

The model writes a critique for each dimension. Then it writes a `<judge_output>` block that contains JSON.
The JSON block is the machine-readable result. Parse the JSON block. Do not parse the scores in the critique text, because they can be different from the JSON scores.
The JSON keys can change from one snippet to a different snippet. The model can omit a dimension that does not apply. Read each key by name.

## Use with llama.cpp

Use llama.cpp build b9180 or later. We tested build 9954.

### llama-server

1. Start the server:

   ```bash
   llama-server -m ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf \
     --host 127.0.0.1 --port 8020 -ngl 999 -c 16384 --jinja -a wp-judge-v4 \
     --chat-template-kwargs '{"enable_thinking": false}'
   ```

2. Send a request:

   ````bash
   curl -s http://127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "wp-judge-v4",
     "messages": [{"role": "user",
       "content": "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction wpj_get_post_title( $id ) {\n\tglobal $wpdb;\n\treturn $wpdb->get_var( \"SELECT post_title FROM {$wpdb->posts} WHERE ID = $id\" );\n}\n```"}],
     "temperature": 0.0,
     "max_tokens": 2048,
     "chat_template_kwargs": {"enable_thinking": false}
   }'
   ````

Notes:

- `--jinja` is necessary. The chat template controls the output format.
- `--chat-template-kwargs` sets `enable_thinking=false` for all requests. The request field does the same for one request. Use one or both.
- To serve more than one request at a time, add `--parallel 4`. All published numbers use `--parallel 1`.
- For the low-VRAM option, add `--n-cpu-moe 99`.
- For the small file, do not add `-ctk q8_0` or `-ctv q8_0`.
- For MTP speculative decoding, use the unpruned Q5_K_M file and add `--spec-type draft-mtp`. We measured 56% draft acceptance. The model uses its own MTP head, so you do not need a separate draft model.

### llama-cli

Run one judgment from the command line:

````bash
llama-cli -m ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf \
  -ngl 999 -c 16384 --jinja --temp 0 -n 2048 \
  --chat-template-kwargs '{"enable_thinking": false}' -st \
  -p $'<wp_judge> Evaluate this WordPress code:\n\n```php\n<your PHP function>\n```'
````

`-st` (single turn) makes `llama-cli` stop after one answer.

## Use with Ollama

We did not measure quality on Ollama. The commands below are the standard form.

1. Download the GGUF file (see [Files](#files)).
2. Write a file named `Modelfile`:

   ```text
   FROM ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf
   PARAMETER num_ctx 16384
   PARAMETER num_predict 2048
   PARAMETER temperature 0
   ```

3. Create the model:

   ```bash
   ollama create wp-judge-v4 -f Modelfile
   ```

4. Run the model with thinking off:

   ```bash
   ollama run wp-judge-v4 --think=false
   ```

You can also pull the file directly from this repo:

```bash
ollama run hf.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf:IQ4_XS
```

Notes:

- Ollama reads the chat template from the GGUF metadata. Do not add a `TEMPLATE` line unless the output format is wrong.
- Set `num_ctx`. The Ollama default context is small and can cut the critique before the `<judge_output>` block.
- In the Ollama API, send `"think": false` to turn off thinking.
- Ollama does not have the llama.cpp `--spec-type` flag. MTP speculative decoding is not available in Ollama.

## Use with LM Studio

We did not measure quality on LM Studio. The steps below are the standard form.

1. Search for `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` and download the IQ4_XS file.
2. Load the model. Set the context length to 16384.
3. Set the temperature to 0 and the maximum response length to 2048 tokens.
4. Turn off thinking in the model settings.
5. Send the prompt in the format from [Prompt format](#prompt-format). Do not use a system prompt.

LM Studio also has an OpenAI-compatible server. Use the Python example below with its base URL (the default is `http://localhost:1234/v1`).

## Use with Python (openai client)

This example works with `llama-server` and other OpenAI-compatible servers. Install the client with `pip install openai`.

```python
import json
import re

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8020/v1", api_key="none")

php = """function wpj_get_post_title( $id ) {
\tglobal $wpdb;
\treturn $wpdb->get_var( "SELECT post_title FROM {$wpdb->posts} WHERE ID = $id" );
}"""

resp = client.chat.completions.create(
    model="wp-judge-v4",
    messages=[{"role": "user",
               "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{php}\n```"}],
    temperature=0.0,
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
text = resp.choices[0].message.content or ""

# Repetition guard: no <judge_output> block means the judgment failed.
m = re.search(r"<judge_output>\s*(\{.*?\})\s*</judge_output>", text, re.S)
if m is None:
    raise RuntimeError("no <judge_output> block: retry or flag this item")
result = json.loads(m.group(1))
print(result["verdict"], result["overall_score"])
```

If your server rejects `chat_template_kwargs`, remove the `extra_body` line. Then turn off thinking on the server.

## Expected output

We sent the `llama-server` request above to the IQ4_XS file.
We used the same server flags as our measurements (`--parallel 1 --seed 1234`, greedy).
The hardware was an NVIDIA GB10 (unified memory) with llama.cpp build 9954.

Result: 71 prompt tokens, 1004 output tokens, 14.7 s wall time, 69.7 tok/s generation, `finish_reason: stop`.

This is the output. We removed some dimension lines (`[...]`). The `<judge_output>` block is complete.

```text
wpj_get_post_title constructs a raw SQL query with $id interpolated directly into the string without $wpdb->prepare(), creating a SQL injection vulnerability. [...]

SQL Safety: severity critical (2/10) — Issue: wpj_get_post_title interpolates $id directly into the SQL query string: "SELECT post_title FROM {$wpdb->posts} WHERE ID = $id". [...] Fix: Replace the raw query with $wpdb->get_var( $wpdb->prepare( 'SELECT post_title FROM %1$s WHERE ID = %d', $wpdb->posts, $id ) ) to ensure the ID is safely escaped.

Security: severity high (4/10) — Issue: wpj_get_post_title accepts an $id parameter and uses it in a database query without any type check. [...]

WP API Usage: severity medium (6/10) — Issue: wpj_get_post_title bypasses the WordPress post API (get_post()) in favor of a raw $wpdb query. [...]

[...]

Key observation: The critical SQL injection risk from $id interpolation is the primary defect; the secondary improvement is replacing the raw query with get_post() to use the WP object cache and follow the WP API convention.

[/REASONING]

<judge_output>
{
  "verdict": "FAIL",
  "wpcs_compliance": 4,
  "security": 4,
  "sql_safety": 4,
  "performance": 5,
  "wp_api_usage": 5,
  "code_quality": 6,
  "dependency_integrity": 8,
  "error_handling": 5,
  "overall_score": 46
}
</judge_output>
```

This sample shows two points. The text score for SQL safety is 2/10, but the JSON score is 4. The JSON also has an `error_handling` key and no `i18n` key. Thus, always read the JSON block by key name.

## Performance

### Quality against human labels

These are Spearman rho values against human-relabeled scores for 9 dimensions. We used 121 held-out items and one seed.

| Stack / config | rho | Notes |
|---|---|---|
| f16 GGUF, llama.cpp | 0.8002 | Uncompressed floor |
| Q6_K GGUF, llama.cpp | 0.8063 | 0 parse failures |
| Q8_0 GGUF, llama.cpp | 0.7851 | 0 parse failures |
| Unpruned Q5_K_M GGUF, llama.cpp | 0.8093 | 0 parse failures. Its own f16 floor is 0.8081. |

At this sample size, these values are statistically the same (95% CI half-width approximately 7 to 8 points).
We did not measure the IQ4_XS and small files against human labels with this protocol. We measured them against the Q6_K file (see below).

The prior WP Judge (`iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`, Qwen3-30B-A3B base, 30.2 GiB) reports 0.8056 on a 3-seed ensemble.
That stack and seed configuration are different. Do not compare that value directly with the values above.

### How we measured the smaller files

We ran the same 121 `wp_judge` validation prompts on each file.
We used greedy decoding, seed 1234, `--parallel 1` and no prompt cache.
**agree_rho** is the Spearman correlation between the scores of a file and the scores of the Q6_K file.
**Verdict agreement** is the share of items with the same PASS/FAIL verdict as Q6_K.
**Unreadable** is the number of outputs that we could not parse, including outputs with no valid JSON.
The 95% CI is a bootstrap interval.

The noise ceiling is approximately 0.94 to 0.96. Two runs of the Q6_K weights on a different numeric path gave 0.942 (q8_0 KV cache) and 0.957 (`--n-cpu-moe`).
Thus a value near 0.94 means "as close as the same weights on a different path".
To ship a file, we required a CI lower bound of 0.85 or more on two additional runs with changed settings.

Limits:

- The sample is small (n=121).
- agree_rho measures agreement with Q6_K. It is not a new skill score against human labels.
- We measured memory and speed on one NVIDIA GB10 GPU with unified memory. Your values on a discrete GPU can be different.
- We measured GPU memory with a 5120-token context and one short request. A 16384-token context uses more memory.

Stability runs (changed settings):

| File | Setting | agree_rho [95% CI] | Verdict agreement | Unreadable |
|---|---|---|---|---|
| IQ4_XS | `-ub 128` | 0.926 [0.891, 0.950] | 97.5% | 0 |
| IQ4_XS | q8_0 KV cache | 0.939 [0.908, 0.960] | 98.3% | 0 |
| small IQ3_XXS-experts | `-ub 128` | 0.930 [0.889, 0.957] | 96.7% | 1 (loop) |
| small IQ3_XXS-experts | `-ub 256` | 0.908 [0.863, 0.938] | 95.9% | 0 |
| small IQ3_XXS-experts | q8_0 KV cache | 0.858 [0.797, 0.903] | 95.0% | 1 (format break). Fails the bar. Do not use. |

### Tested options that we do not ship

| Option | Size (GiB) | agree_rho | Reason |
|---|---|---|---|
| Q5_K_M (from Q6_K / from f16) | 20.36 | 0.918 / 0.913 | Larger than IQ4_XS and lower agreement. |
| Q4_K_M (from Q6_K / from f16) | 17.43 | 0.876 / 0.916 | Larger than IQ4_XS and lower agreement. |
| Q3_K_M (from f16) | 13.82 | 0.839 | Below the bar. |
| IQ3_XXS, all tensors, imatrix | 11.24 | 0.835 | Below the bar. |
| Experts Q4_K_M, other tensors Q6_K | 16.70 | 0.867 | Lower agreement than IQ4_XS at a larger size. |
| Experts IQ4_XS, other tensors Q6_K | 15.88 | 0.926 | Larger and slower than IQ4_XS, with no gain in agreement. |
| Q6_K with q8_0 KV cache | 23.47 | 0.942 | Saves only 47 MiB of GPU memory. |
| Expert prune to 192 of 256 (Q6_K) | 20.38 | 0.879 | Lower agreement than IQ4_XS at a larger size. |
| Expert prune to 160 of 256 (Q6_K) | 17.30 | 0.880 | 19 unreadable outputs (word loops). |
| `-fa on` | | | No change. Flash attention is already on by default. |

## Pruned files and MTP

The `pruned-k224` files do not have an MTP head.
Expert pruning left the MTP layer with a different expert count than the main layers.
GGUF requires one expert count for all layers.
Thus we removed the MTP tensors at conversion (`--no-mtp`).

Effects:

- Normal generation and judging operate correctly in the pruned files.
- The pruned files ignore `--spec-type draft-mtp`. Do not use that flag with them.
- For MTP speculative decoding, use `wp-judge-v4-unpruned.Q5_K_M.gguf`.

## Other engines

We measured all published numbers on llama.cpp.
Other engines (Ollama, LM Studio, llama-cpp-python) can run the same GGUF files. We did not verify their numeric results.

## Links

- Base model: fine-tuned from [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), then quantized to GGUF. The unpruned file keeps all 256 experts and the MTP head. The pruned files have 224 of 256 experts.
- Build process: [github.com/dr-robert-li/wp-finetune](https://github.com/dr-robert-li/wp-finetune). See `PIPELINE.md`, `JOURNAL.md` and `CHANGELOG.md`.
- License: Apache 2.0.
