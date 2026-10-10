# Harness Component Inventory

## 1. Extraction

**`scripts/php_extract_functions.php`** (258 lines). Run as `php x.php <file>`; it prints a JSON array. It finds named functions and methods with `token_get_all` and skips closures.
- Fields: `function_name` (`Class::method`), `class_context`, `docblock`, `body`, `start_line`, `dependencies`, `sql_patterns`, `hooks_used`, `line_count`.
- Deps: php-cli 7.0+ (type hints) with the tokenizer and json extensions. Use 8.x. No network. PHP is **not installed on the host**.
- **Modify** to fix these bugs:
  - Strings containing `{$x}` emit T_CURLY_OPEN, which is not counted, but the closing `}` is, so functions end early.
  - Bodiless abstract and interface methods swallow the next function.
  - `new class` and braced `namespace {}` corrupt class context.
  - Add `end_line`.

**`scripts/phase1_extract.py`** (172 lines). Driver: runs the PHP script on each file found by `rglob("*.php")`, with a 30 s timeout. Errors are dropped silently. It adds `source_repo`, `source_file` and `quality_tier`.
- Skip logic: substring match on `vendor/`, `node_modules/`, `/tests/`, `/test/` and `.phpcs`, plus `repos.yaml` paths. **No minified skip.**
- Deps: pyyaml, `scripts.utils` checkpoints.
- **Modify:** keep `extract_functions_from_file` and the skip list. Drop yaml and checkpoints. Log failures.

## 2. Filters

**`scripts/judge_amp_wp.py`, `has_test_code`** (lines 18-36, stdlib). True when any of these hold:
- the path contains `/test`, `test-` or `-test.`, or ends with `Test.php` or `Tests.php`;
- the body contains `PHPUnit` or `extends WP_UnitTestCase|TestCase`;
- the path starts with `bin/` or `docs/`.

**Copy as is.**

**`scripts/judge_repos.py`** (752 lines). A regex rule-based judge with a hardcoded `/home/robert_li` path. **Do not copy** (R1.4). Only two rules are worth lifting: skip `line_count < 5`, and auto-pass core.

## 3. Prompt and rubric

- **No system prompt.** v4 was trained on user and assistant turns only (`data/reasoning_dataset/openai_val.jsonl`). `config/judge_system.md` is the old Claude labelling prompt with a different schema. Do not send it.
- **User message:** `"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"`.
- **Output:** prose lines `"<Label>: score N/10 — ..."` (N is 0-10 or `None` = not applicable), then `[/REASONING]`, then `<judge_output>{json}</judge_output>`.
- **The 9 dimensions** (0-10 each): WPCS Compliance, SQL Safety, Security, Performance, WP API Usage, Code Quality, Dependency Integrity, i18n, Accessibility.
- **overall_score** (0-100) and **verdict** come from the model.
  - Repo policy: PASS iff overall ≥ 70 (`_PASS_THRESHOLD`).
  - If overall is missing, it is derived as a weighted mean of the present dimensions (`eval/dim_map.json`: security 0.20, sql 0.15), capped at 69 on FAIL.
  - **Caution:** in the IQ4_XS capture, 32/120 model verdicts disagree with that rule (PASS at 56-68; every FAIL ≤ 54). Choose one rule.

## 4. Judge call and parsers

- **`eval/eval_judge.py` `_judge_create`** (lines 47-76, needs `openai`). Sends `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`. If the template rejects it, the kwarg is dropped with a warning. **Copy**, but pass `max_tokens=2048`.
- **`parse_judge_response`** (lines 310-383, stdlib). Strips `<think>`, then tries `<judge_output>`, raw JSON, fences, and the first `{...}`. **Copy as is.**
- **Optional:** the prose fallback in `judge_score_single`, using `_parse_prose_dim_scores`, `_derive_overall_from_dims` and their maps.
- **Do not import the module.** It pulls in scipy, `rubric_scorer` and `dgx_toolbox`. Copy the functions out.
- **`eval/output_parsers.py` `parse_judge_scores`** (199 lines; stdlib plus `dim_map.json`). Returns internal D1-D9 keys and drops `code_quality` and `dependency_integrity`. **Modify** to keep all 9 native names.

## 5. Response schema

Records in `output/shrink/iq4xs_im_f16/judge_responses.jsonl` have the keys `index` and `response`.

`<judge_output>` keys, with counts out of 120:

| Key | Count |
|---|---|
| `verdict` | 120 |
| `wpcs_compliance` | 120 |
| `code_quality` | 120 |
| `dependency_integrity` | 120 |
| `overall_score` | 120 |
| `performance` | 102 |
| `wp_api_usage` | 102 |
| `security` | 89 |
| `error_handling` | 47 |
| `i18n` | 26 |
| `accessibility` | 12 |
| `sql_safety` | 10 |

Dimensions the model rated not applicable are omitted.

## 6. Aggregation, reports, containers

- `scripts/shrink_score.py`, `aggregate_revl03.py` and `rlev02_report.py` serve other studies. None aggregate per plugin.
- `scripts/sieve_capture_judge_http.py` (119 lines) is a threaded client built on `_judge_create`.
- No Dockerfile, compose file, fastapi, uvicorn or flask exists.

## 7. llama-server (`scripts/shrink_eval.sh`, 56 lines)

```
-m $GGUF --host 127.0.0.1 --port 8097 -ngl 999 -c 5120 --jinja -a $NAME
--parallel 1 --seed 1234 --no-mmap --no-cache-prompt --cache-ram 0
```

- `-c 5120` = 2048 + 3072.
- For the container, prefer the README flags `-c 16384 --chat-template-kwargs '{"enable_thinking": false}'`.
- The small file requires an f16 KV cache: no `-ctk/-ctv q8_0`.

## Gaps (greenfield)

- Archive intake (traversal guard), minified detection, and scope policy (classes, templates, tiny functions).
- Prompt builder and long-function truncation.
- Retry and unjudged-function policy.
- Native 9-label parser with one PASS/FAIL rule.
- Aggregation (LOC-weighted mean, safety cap, per-dimension stats, worst functions) and pattern ranking.
- Report schema and versioning.
- Library API, CLI, FastAPI job API (disk store, auth, limits).
- Dockerfile with php-cli, compose, the all-in-one target, and the GGUF config.
- Tests.
