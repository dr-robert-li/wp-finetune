# v1.1 Decision 2: a retry that can recover a repetition loop

Builder notes. Work is on main, uncommitted. Evidence lives in `build_notes/e2e_work/v11/retry/`.

## Problem (recap)
`judge_one` retried a parse_failure with a byte-identical request (temperature 0, `cache_prompt: false`, serial server). Greedy decoding is deterministic, so a repetition loop to `max_tokens` (2048) came back byte-for-byte the same (EVAL_V11_WSE §4). Reproduced here: host `search.php` attempt-1 md5 `87ac1436…`, the same as the WS-E probe.

## Loop units
| server | unit | attempt 1 |
|---|---|---|
| host (llama.cpp 9954, `--parallel 1`) | twentytwelve 4.9 `search.php:<template>` | loops on "`esc_html` is not in core…", length 2048 |
| container b11515 (`--parallel 1 -c 8192`, the same llama-server flags as api/allinone) | fixture `includes/legacy.php:ser_search_registrations` | loops on "…unnecessary IOPerformance: severity medium (6/10)…", length 2048 |
| container b11515 | twentytwelve 4.9 `search.php:<template>` | **new**: loops like the host does, length 2048 |

To find more: I grepped every v11 report (`*.json`, `real/*.json`) for unjudged or attempts==2 rows. The only parse_failures are the two known ones. The `r_down.json` rows are server-down connection errors. Container `search.php` turned up because I ran both loop units on both servers. The full twentytwelve and classic-editor scans were not re-run on the container (too costly), so more container loops may exist there.

Setup: one server at a time, each under `flock /tmp/wpjudge-gpu.lock`, killed after use. The container was started with `sg docker`, image `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515`. Requests are the exact attempt-1 request (`judge.PROMPT`, `max_tokens` 2048, `logprobs`, the extra_body `enable_thinking: false` + `cache_prompt: false`) plus the candidate's change. Parsing and scoring go through `judge.to_judgment` and `judge.verdict_prob`. Probe script: `retry/retry_probe.py`. Every raw output is in `retry/<server>_<unit>__<candidate>.txt`, and every row is in `retry/results.jsonl`.

## Candidates
| id | change on attempt 2 | standard OpenAI param? |
|---|---|---|
| a1 | none (today's retry) | - |
| freq05 | `frequency_penalty` 0.5 | yes |
| pres05 | `presence_penalty` 0.5 | yes |
| rep11 | extra_body `repeat_penalty` 1.1 | no (llama.cpp) |
| t03s | `temperature` 0.3, `seed` 1234 | yes |
| **t05s** | **`temperature` 0.5, `seed` 1234** | **yes** |
| t07s | `temperature` 0.7, `seed` 1234 | yes |
| t07s42 | `temperature` 0.7, `seed` 42 (seed robustness) | yes |
| dry08 | extra_body `dry_multiplier` 0.8 (default breakers) | no (llama.cpp) |
| drynl | extra_body `dry_multiplier` 0.8, `dry_sequence_breakers` ["\n"] | no (llama.cpp) |
| freqall | `frequency_penalty` 0.5 + extra_body `repeat_last_n` -1 | no; b11515 rejects -1 (HTTP 400), so host only |

### Loop units: every candidate
| server | unit | cand | parseable | finish | tokens | overall | verdict | verdict_p | scores |
|---|---|---|---|---|---|---|---|---|---|
| host | `search.php::<template>` | a1 | no | length | 2048 | - | - | - | - |
| host | `search.php::<template>` | freq05 | no | length | 2048 | - | - | - | - |
| host | `search.php::<template>` | pres05 | no | length | 2048 | - | - | - | - |
| host | `search.php::<template>` | rep11 | no | length | 2048 | - | - | - | - |
| host | `search.php::<template>` | t03s | no | length | 2048 | - | - | - | - |
| host | `search.php::<template>` | t05s | yes | stop | 1023 | 68 | PASS | 0.997 | wpcs_compliance 6, security 6, performance 6, wp_api_usage 7, code_quality 7, dependency_integrity 7, i18n 8, accessibility 6, error_handling 5 |
| host | `search.php::<template>` | t07s | yes | stop | 1337 | 64 | PASS | 0.823 | wpcs_compliance 6, security 5, wp_api_usage 6, code_quality 6, i18n 6, accessibility 6, dependency_integrity 5, error_handling 5 |
| host | `search.php::<template>` | t07s42 | yes | stop | 1477 | 79 | PASS | 0.985 | wpcs_compliance 8, security 7, performance 8, wp_api_usage 8, code_quality 8, i18n 8, accessibility 7, dependency_integrity 8 |
| host | `search.php::<template>` | dry08 | yes | stop | 1274 | 68 | PASS | 0.847 | wpcs_compliance 8, security 6, performance 7, wp_api_usage 6, code_quality 7, dependency_integrity 6, i18n 8, accessibility 7, error_handling 5 |
| host | `search.php::<template>` | drynl | yes | stop | 1910 | 72 | PASS | 0.902 | wpcs_compliance 8, security 6, performance 7, wp_api_usage 7, code_quality 8, dependency_integrity 7, i18n 8, accessibility 7, error_handling 5 |
| host | `search.php::<template>` | freqall | no | stop | 786 | - | - | - | - |
| ctr | `ser_search_registrations` | a1 | no | length | 2048 | - | - | - | - |
| ctr | `ser_search_registrations` | freq05 | yes | stop | 902 | 38 | FAIL | 0.942 | wpcs_compliance 4, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 7, error_handling 5 |
| ctr | `ser_search_registrations` | pres05 | yes | stop | 928 | 38 | FAIL | 0.968 | wpcs_compliance 4, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 7, error_handling 5 |
| ctr | `ser_search_registrations` | rep11 | yes | stop | 817 | 30 | FAIL | 0.959 | wpcs_compliance 4, security 3, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 7, error_handling 4 |
| ctr | `ser_search_registrations` | t03s | yes | stop | 916 | 36 | FAIL | 0.956 | wpcs_compliance 3, security 4, sql_safety 2, performance 5, wp_api_usage 4, code_quality 3, dependency_integrity 6, error_handling 4 |
| ctr | `ser_search_registrations` | t05s | yes | stop | 1027 | 42 | FAIL | 0.963 | wpcs_compliance 4, security 4, sql_safety 2, performance 6, wp_api_usage 6, code_quality 5, dependency_integrity 8, error_handling 4 |
| ctr | `ser_search_registrations` | t07s | yes | stop | 883 | 42 | FAIL | 0.962 | wpcs_compliance 3, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 7, error_handling 4 |
| ctr | `ser_search_registrations` | t07s (repeat, new server) | yes | stop | 883 | 42 | FAIL | 0.962 | wpcs_compliance 3, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 7, error_handling 4 |
| ctr | `ser_search_registrations` | t07s42 | yes | stop | 1010 | 37 | FAIL | 0.944 | wpcs_compliance 3, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 8 |
| ctr | `ser_search_registrations` | dry08 | no | length | 2048 | - | - | - | - |
| ctr | `ser_search_registrations` | dry08 (repeat, new server) | no | length | 2048 | - | - | - | - |
| ctr | `ser_search_registrations` | drynl | yes | stop | 1019 | 36 | FAIL | 0.951 | wpcs_compliance 4, security 4, sql_safety 2, performance 6, wp_api_usage 5, code_quality 4, dependency_integrity 8, error_handling 5 |
| ctr | `search.php::<template>` | a1 | no | length | 2048 | - | - | - | - |
| ctr | `search.php::<template>` | freq05 | yes | stop | 788 | 79 | PASS | 0.999 | wpcs_compliance 8, security 7, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 7, error_handling 6 |
| ctr | `search.php::<template>` | pres05 | no | length | 2048 | - | - | - | - |
| ctr | `search.php::<template>` | rep11 | yes | stop | 1100 | 79 | PASS | 0.999 | wpcs_compliance 8, security 7, performance 8, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 7, error_handling 6 |
| ctr | `search.php::<template>` | t03s | yes | stop | 978 | 78 | PASS | 0.998 | wpcs_compliance 8, security 7, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 6, error_handling 6 |
| ctr | `search.php::<template>` | t05s | yes | stop | 907 | 80 | PASS | 0.999 | wpcs_compliance 8, security 8, performance 7, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 7, error_handling 6 |
| ctr | `search.php::<template>` | t07s | yes | stop | 834 | 82 | PASS | 0.999 | wpcs_compliance 8, security 8, performance 8, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 6, error_handling 6 |
| ctr | `search.php::<template>` | t07s (repeat, new server) | yes | stop | 834 | 82 | PASS | 0.999 | wpcs_compliance 8, security 8, performance 8, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 8, accessibility 6, error_handling 6 |
| ctr | `search.php::<template>` | t07s42 | yes | stop | 1104 | 78 | PASS | 0.99 | wpcs_compliance 8, security 8, performance 7, wp_api_usage 8, code_quality 8, dependency_integrity 8, i18n 6, accessibility 6 |
| ctr | `search.php::<template>` | dry08 | no | length | 2048 | - | - | - | - |
| ctr | `search.php::<template>` | dry08 (repeat, new server) | no | length | 2048 | - | - | - | - |
| ctr | `search.php::<template>` | drynl | no | length | 2048 | - | - | - | - |

Recovery count over the 3 loop cases (host search.php, ctr ssr, ctr search.php):

| cand | recovered |
|---|---|
| a1 | 0/3 |
| freq05 | 2/3 (host search.php still loops) |
| pres05 | 1/3 |
| rep11 | 2/3 (host search.php still loops) |
| t03s | 2/3 (host search.php still loops) |
| **t05s** | **3/3** |
| t07s | 3/3 |
| t07s42 | 3/3 |
| dry08 | 1/3 (both container loops still loop) |
| drynl | 2/3 (ctr search.php still loops) |
| freqall | 0/1 (host search.php ends at 786 tokens with no `<judge_output>`); rejected by b11515 |

Penalties at temp 0 do change the output: the md5s differ from a1. But the host search.php loop just changes shape ("`esc_html`? No." over and over). llama.cpp's frequency and presence penalties only look back over the last 64 tokens (`repeat_last_n`), and this loop has a longer period. DRY with its default sequence breakers (`\n : " *`) loses the IOPerformance loop: its repeats span `:` and `*`, which reset DRY's match.

### Units that parse on attempt 1: does the retry setting stay sane?
Each cell is overall plus verdict (F/P). Attempt-1 (a1) is the reference. Planted-bug units: ser_search_registrations, ser_delete_registration (SQLi) and ser_render_search_box (unescaped echo). Clean units: ser_get_registrations_for_event and ser_sanitize_settings. Real-world: twentytwelve_comment.


host:

| unit | a1 | t05s | t07s | dry08 | drynl | freq05 | pres05 | rep11 | t03s |
|---|---|---|---|---|---|---|---|---|---|
| ser_search_registrations | 36 F | 36 F | 39 F | 36 F | 36 F | 38 F | 36 F | 30 F | 36 F |
| ser_delete_registration | 46 F | 46 F | 40 F | 46 F | 36 F | 38 F | 38 F | 30 F | 44 F |
| ser_render_search_box | 14 F | 14 F | 18 F | 18 F | 14 F | 19 F | 19 F | 39 F | 42 F |
| ser_get_registrations_for_event | 80 P | 77 P | 79 P | 78 P | 78 P | 79 P | 80 P | 79 P | 78 P |
| ser_sanitize_settings | 84 P | 84 P | 88 P | 84 P | 82 P | 84 P | 84 P | 84 P | 84 P |
| twentytwelve_comment | 78 P | 76 P | 70 P | 78 P | 78 P | 79 P | 79 P | 79 P | 72 P |

container b11515:

| unit | a1 | t05s | t07s | dry08 | drynl | freq05 | pres05 | rep11 | t03s |
|---|---|---|---|---|---|---|---|---|---|
| ser_delete_registration | 42 F | 46 F | 15 F | 42 F | 32 F | n/r | n/r | n/r | n/r |
| ser_render_search_box | 42 F | 26 F | 38 F | 46 F | 14 F | n/r | n/r | n/r | n/r |
| ser_get_registrations_for_event | 80 P | 84 P | 84 P | 80 P | 80 P | n/r | n/r | n/r | n/r |
| ser_sanitize_settings | 82 P | 82 P | 86 P | 82 P | 82 P | n/r | n/r | n/r | n/r |
| twentytwelve_comment | 72 P | 74 P | 68 P | 72 P | 72 P | n/r | n/r | n/r | n/r |

(n/r = not run. ctr ser_search_registrations is a loop unit, see above.)

Max |Δ overall| against a1 on these units:

| cand | host (6 units) | container (5 units) | verdict flips |
|---|---|---|---|
| **t05s** | **3** | **16** (render_search_box 42→26; both FAIL, both capped) | 0 |
| t07s | 8 | 27 (delete_registration 42→15) | 0 |
| dry08 | 4 | 4 | 0 |
| drynl | 10 | 28 | 0 |
| rep11 | 25 | n/r | 0 |
| t03s | 28 | n/r | 0 |

No candidate flipped a verdict, and every planted unit stayed below every clean unit. On the container, render_search_box already sits 28 apart between host and container on attempt 1 (14 vs 42), so this unit is unstable on any numeric path. t05s lands between the two.

## Chosen option: t05s (`temperature` 0.5, `seed` 1234, attempt 2 only)
Why:
1. **It is the only family that recovered all 3 loop cases.** That is temperature ≥ 0.5 with a seed: t05s, t07s and t07s42 are all 3/3. DRY, the penalties and t03s each leave at least one loop.
2. **Smallest drift of the 3/3 options.** On parseable units t05s has max |Δ| 3 on the host and 16 on the container, against 8 and 27 for t07s. On the host it reproduces a1 exactly for all 3 planted fixture units.
3. **Most portable.** `temperature` and `seed` are standard OpenAI chat params. They go top-level, not in extra_body, so they survive the existing extra_body fallback path. A server that rejects `chat_template_kwargs` still gets a varied retry. DRY and `repeat_penalty` are llama.cpp extensions and would be dropped there.
4. **Deterministic.** A fixed seed on a serial server gives byte-identical output:
   - t07s run on 2 separate host server starts: md5 `35b8c9cb…` both times.
   - t07s on 2 separate container starts, for all 7 container units: identical md5s (see `ctr_*__t07s.txt` vs `ctr2_*__t07s.txt`).
   - t05s through the shipped `judge_one` (live run below) reproduced the probe's numbers exactly: host search.php 68, ctr search.php 80, ctr ssr 42.
   - Multi-slot servers are already flagged `reproducible: false` (T5). This changes nothing there.

Attempt 1 is untouched, so every unit that parses on attempt 1 keeps its exact score.

Not chosen: `max_tokens` up. A true loop just runs longer.

## Code diff summary
- `wpjudge/judge.py`
  - New `RETRY_SAMPLING = {"temperature": 0.5, "seed": 1234}`, with a comment that points here.
  - `_judge_create(..., temperature=0.0, **sampling)` passes `**sampling` top-level on both the extra_body path and the fallback path. The extra_body content and the rejection logic are unchanged.
  - `judge_one` passes `**(RETRY_SAMPLING if attempt == 2 else {})`. Attempt 1 sends exactly the same kwargs as before: no `seed`, temperature 0.
  - The retry also varies after a transport error on attempt 1. That keeps `attempts == 2` equal to "judged on the varied request", and it is still deterministic.
- `wpjudge/aggregate.py` `_row`: a judged row with `attempts == 2` gets `"retry"` in `confidence_reasons`, so `low_confidence` is true. This is a review lead only: score and cap are unchanged. Unjudged rows keep `[]`. No schema change: `schema_version` stays "1.0" and no new fields are added.
- `README.md` (Scoring, "Each unit is sent once and retried once"): one sentence on the varied, reproducible retry and the `retry` reason.
- Tests:
  - `tests/test_judge.py::test_retry_after_unparseable_then_ok` now also asserts:
    - attempt 1 has temperature 0 and no seed;
    - attempt 2 is exactly attempt 1 plus `temperature 0.5, seed 1234`.
  - New `test_retry_sampling_survives_extra_body_fallback`: attempt 2 hits a template rejection, and the fallback call still carries temperature and seed but no extra_body.
  - The existing `test_request_shape` still pins the attempt-1 kwargs.
  - New `tests/test_aggregate.py::test_retry_confidence`: attempts 2 judged gives `["retry"]` and low_confidence; unjudged gives none.

## Test results
- `.venv/bin/python -m pytest tests/ -q`: **72 passed**. The 1 warning (`FileNotFoundError: 'jobs'` in a test_config thread) was there before this change.
- `tests/e2e.sh cli`, live on the host, `--parallel 1`, under flock: **154 passed, 0 failed**.
  - Golden fixture unchanged: overall 60, **uncapped 68.8**, per-unit **36,46,46,14,46,80,80,80,84,84**. All 10 units have attempts 1, so the retry never fires on the fixture on the host.
  - dir, zip, tgz and w4 are exactly equal.
- Live `judge_one` + `aggregate._row` through the shipped code (`retry/live_j1.py`, outputs `retry/live_<server>_<unit>.json`):

| server | unit | status | attempts | overall | verdict | verdict_p | confidence_reasons |
|---|---|---|---|---|---|---|---|
| host | search.php `<template>` | judged | 2 | 68 | PASS | 0.997 | ["retry"] |
| host | ser_search_registrations | judged | 1 | 36 | FAIL | 0.971 | [] |
| container | search.php `<template>` | judged | 2 | 80 | PASS | 0.999 | ["retry"] |
| container | ser_search_registrations | judged | 2 | 42 | FAIL | 0.963 | ["retry"] |

Effect on the WS-E container finding: ser_search_registrations now has sql_safety 2, so it is capped. That makes 5/5 planted units capped on the b11515 serial run, with coverage back to 100%. I did not re-run the api/allinone e2e sections. The numbers come from the same image and flags, not from the compose stack.

nvidia-smi showed no compute apps and `docker ps` was empty at the end.

## Recovered critiques (attempt 2, t05s) for the evaluator

### host search.php <template>

Overall 68, verdict PASS, verdict_p 0.997. Scores: `{"wpcs_compliance":6,"security":6,"performance":6,"wp_api_usage":7,"code_quality":7,"dependency_integrity":7,"i18n":8,"accessibility":6,"error_handling":5}`. Raw: `retry/live_host_template.json` (field `raw`).

```text
WPCS Compliance: score 6/10 — The code follows standard WordPress template structure conventions. It uses `get_header()`, `get_sidebar()`, and `get_footer()` appropriately. The loop structure `while ( have_posts() ) : the_post(); ... endwhile;` is correct. However, it uses `get_post_format()` inside the loop without checking if a post has a format, which can return false or empty string for standard posts, potentially causing issues in template parts that expect a specific string. It lacks `wp_reset_postdata()` after the loop, which is a best practice to prevent global state pollution if the loop is modified or if other parts of the page rely on global `$post`.

SQL Safety: score None/10 — No direct SQL queries are executed in this template file. The search query is handled by the WordPress query engine (`$wp_query`), which is inherently safe against SQL injection. No custom SQL is present.

Security: score 6/10 — The search query is escaped using `esc_html( get_search_query( false ) )` inside the `printf` call. This is correct for outputting in an HTML context. The text strings are wrapped in `__()` and `_e()` with the correct text domain 'twentytwelve', which is good for translation safety. However, `get_search_query()` returns unescaped data by default, so the explicit `esc_html()` is crucial here. The code does not check for nonce or user capabilities, but search results are generally public and safe from CSRF. The main risk is XSS if the search query contains malicious scripts, which `esc_html` mitigates.

Performance: score 6/10 — The code uses standard WordPress functions. `get_post_format()` is called for every post in the loop. While not a heavy database query, it adds a slight overhead. `get_search_query()` is called inside the loop context for the title, which is fine. No obvious performance bottlenecks like N+1 queries or heavy loops are present.

WP API Usage: score 7/10 — The code uses `get_header()`, `get_sidebar()`, `get_footer()`, `have_posts()`, `the_post()`, `get_template_part()`, `get_search_query()`, `esc_html()`, `printf()`, `__()`, `_e()`, and `get_search_form()`. These are standard and correct usages of the WordPress API. The use of `get_post_format()` is valid but requires careful handling in the template part. `get_search_form()` is the correct API for displaying the search form.

Code Quality: score 7/10 — The code is readable and follows standard WordPress theme structure. The use of `printf` with `__()` for translatable strings with placeholders is correct. The structure is logical. The lack of `wp_reset_postdata()` after the loop is a minor code quality issue that can lead to subtle bugs in complex themes or plugins.

Dependency Integrity: score 7/10 — The code relies only on core WordPress functions and the theme's own template parts. No external dependencies or custom classes are used. The text domain 'twentytwelve' is consistent.

i18n: score 8/10 — The code correctly uses `__()` and `_e()` for translatable strings. The text domain 'twentytwelve' is used consistently. The placeholder `%s` is correctly placed in the translation string `'Search Results for: %s'` and passed as the second argument to `printf`. This allows translators to change the word order in their language. The search query is escaped with `esc_html()` before being passed to the translation function, which is the correct order (escape after translation).

Accessibility: score 6/10 — The code uses semantic HTML elements like `<section>`, `<header>`, `<h1>`, `<article>`, and `<p>`. The `role="main"` attribute on the content div is appropriate. The search form is included via `get_search_form()`, which typically provides a basic accessible form structure. However, the search results page does not explicitly use ARIA landmarks for the results list itself, though the `<section>` and `<article>` tags provide some semantic structure. The use of `<h1>` for the search title is correct.
```

### container search.php <template>

Overall 80, verdict PASS, verdict_p 0.999. Scores: `{"wpcs_compliance":8,"security":8,"performance":7,"wp_api_usage":8,"code_quality":8,"dependency_integrity":8,"i18n":8,"accessibility":7,"error_handling":6}`. Raw: `retry/live_ctr_template.json` (field `raw`).

```text
WPCS Compliance: score 8/10 — The code follows standard WordPress template structure. It uses `get_header()`, `get_sidebar()`, and `get_footer()` correctly. The loop structure `while ( have_posts() ) : the_post(); ... endwhile;` is correct. The use of `get_template_part()` is appropriate. There are no obvious WPCS violations such as missing semicolons or incorrect array syntax.

SQL Safety: score None/10 — No direct SQL queries are present in this template file. The search query is handled by the core WordPress loop (`have_posts()`), which internally uses `$wpdb->prepare()` for safety. The search query string is retrieved via `get_search_query()` and escaped via `esc_html()`, preventing XSS. There are no raw SQL strings or direct database interactions.

Security: score 8/10 — The search query is retrieved using `get_search_query( false )` and escaped using `esc_html()`. This is the correct pattern for displaying search terms in the context of HTML. The `printf` statement uses `%s` placeholder, which is safe when combined with `esc_html()`. The code does not output raw user input without escaping. No nonce checks are required in a public-facing search results template.

Performance: score 7/10 — The template uses standard WordPress functions. `get_search_query()` is lightweight. `get_template_part()` is efficient for modularity. The loop iterates through posts, which is expected. There no obvious performance bottlenecks like N+1 queries or excessive database calls within this template file.

WP API Usage: score 8/10 — Correct usage of `get_header()`, `get_sidebar()`, `get_footer()`, `get_search_query()`, `get_template_part()`, `have_posts()`, `the_post()`, `twentytwelve_content_nav()`. The use of `esc_html()` is the correct API for escaping HTML context. The use of `__()` and `_e()` for translation is correct.

Code Quality: score 8/10 — The code is clean, readable, and follows WordPress theme development best practices. It uses semantic HTML5 elements (`<section>`, `<header>`, `<article>`). The structure is logical. The only minor point is the use of `esc_html()` inside a `printf` string, which is technically correct but sometimes debated whether `esc_html()` should wrap the entire string or just the variable; however, since `printf` handles the concatenation, wrapping the variable is the precise way to escape the injected data.

Dependency Integrity: score 8/10 — The code relies on core WordPress functions and theme-specific functions like `twentytwelve_content_nav()`. There are no external libraries or plugins required. The dependencies are standard for a Twenty Twelve child theme or similar setup.

i18n: score 8/10 — The strings are correctly wrapped in `__()` or `_e()` with the text domain `'twentytwelve'`. The search query placeholder `%s` is correctly placed within the translatable string. The translators comment `/* translators: %s: Search query. */` is present, which is best practice for context.

Accessibility: score 7/10 — The HTML structure uses semantic elements. The search form is included via `get_search_form()`, which typically generates a form with a label (though the label text might be 'Search' or similar, it's standard). The page title uses `<h1>`, which is correct for the main heading. The 'Nothing Found' message is clear. The only potential minor issue is if the search form doesn't have a visible label, but `get_search_form()` usually handles this reasonably well or is a minor enhancement.
```

### container includes/legacy.php ser_search_registrations

Overall 42, verdict FAIL, verdict_p 0.963. Scores: `{"wpcs_compliance":4,"security":4,"sql_safety":2,"performance":6,"wp_api_usage":6,"code_quality":5,"dependency_integrity":8,"error_handling":4}`. Raw: `retry/live_ctr_ser_search_registrations.json` (field `raw`).

```text
The function executes a direct SQL query using $wpdb->get_results() with string concatenation of $_GET['q'] without any parameterization or escaping. An admin list table search handler performs no capability check before executing the database query, and the unescaped search string is interpolated directly into the SQL, creating a SQL injection vulnerability.

WPCS Compliance: score 4/10 — Issue: $_GET['q'] is read without sanitization and passed to $wpdb->get_results() without $wpdb->prepare(). WPCS requires all SQL queries to use parameterized statements via $wpdb->prepare() to prevent SQL injection.. Fix: Replace the raw query with $wpdb->prepare( "SELECT * FROM {$wpdb->prefix}ser_registrations WHERE email LIKE %%s ORDER BY id DESC", '%' . $search . '%' ) and pass the parameterized query to $wpdb->get_results().

SQL Safety: score 2/10 — Issue: The SQL query in ser_search_registrations() concatenates $_GET['q'] directly into the query string without $wpdb->prepare() or any escaping. An admin list table search handler must use parameterized queries to prevent SQL injection; the $wpdb->prepare() API is the required WordPress mechanism for this.. Fix: Use $wpdb->prepare( "SELECT * FROM {$wpdb->prefix}ser_registrations WHERE email LIKE %%s ORDER BY id DESC", '%' . $wpdb->esc_like( $search ) . '%' ) to safely parameterize the search query.

Security: score 4/10 — Issue: ser_search_registrations() performs a database query triggered by an admin list table search action without any capability check. Any user who can reach the admin list table endpoint (which may be broadened by missing capability checks in the listing plugin) can execute this search query. A search handler should verify current_user_can( 'edit_posts' ) or the specific capability before performing the database query.. Fix: Add current_user_can( 'edit_posts' ) before executing the query in ser_search_registrations(), or verify the specific ser_registrations capability before allowing the search to proceed.

Performance: score 6/10 — Issue: SELECT * retrieves all columns from ser_registrations when only a subset of columns may be needed for the admin list table display. Specifying only needed columns reduces IOP and memory overhead on large tables.. Fix: Replace SELECT * with SELECT id, email, created_at from {$wpdb->prefix}ser_registrations to limit the result set to only the columns displayed in the admin list table.

WP API Usage: score 6/10 — Issue: $wpdb->get_results() is the correct method for multi-row query results; the issue is not the choice of method but the lack of $wpdb->prepare() to parameterize the query. The corrected code uses $wpdb->prepare() with $wpdb->get_results() as the proper WP database API pattern.. Fix: No wp_api_usage change needed beyond the $wpdb->prepare() fix already described under sql_safety.

Code Quality: score 5/10 — Issue: No escaping of the $search string before use in the SQL query. $wpdb->esc_like() should be applied to the search term before interpolating it into the LIKE clause, even when using $wpdb->prepare().. Fix: Apply $wpdb->esc_like( $search ) to the search term before building the LIKE parameter for the prepared statement.

Dependency Integrity: score 8/10 — Issue: No external dependencies introduced; the function uses only $wpdb from the WP core global namespace.. Fix: No dependency_integrity changes needed.

i18n: score None/10 — Issue: ser_search_registrations() returns database rows, not UI strings; no i18n concern applies.. Fix: No i18n changes needed.

Accessibility: score None/10 — Issue: ser_search_registrations() is a data retrieval function with no HTML output; accessibility concerns do not apply at this layer.. Fix: No accessibility changes needed.

Key observation: The critical SQL injection vulnerability from unparameterized $_GET['q'] in the database query must be fixed with $wpdb->prepare() before any further consideration of code quality; the missing capability check is the secondary concern.
```

### Builder's read on defensibility (the code is quoted in `e2e_work/v11/retry/` raw files; sources: twentytwelve 4.9 `search.php`, fixture `includes/legacy.php:18`)
- **ser_search_registrations (container, 42 FAIL, sql_safety 2, security 4)**
  - Correct: `$_GET['q']` is concatenated raw into `get_results()`. That is SQLi, and the fix is `prepare()` + `esc_like`.
  - It agrees with the host's attempt-1 judgment (36, sql 2, sec 4).
  - Small error: the suggested fix writes `LIKE %%s`. In `prepare()` that is a literal `%s`, so it should be `%s`.
  - The cap fires, as it should for a planted SQLi.
- **search.php (container, 80 PASS)**
  - Accurate. Output is escaped via `esc_html( get_search_query( false ) )`, there is no SQL, and i18n uses a translators comment. Defensible.
  - It misses that the translated format string from `__()` is echoed through `printf` unescaped. WPCS EscapeOutput would flag this; it is a minor finding.
- **search.php (host, 68 PASS, security 6, wpcs 6)**
  - The scores are plausible for the same minor issue, and the unit is not capped.
  - The stated reasons are weak:
    - "missing `wp_reset_postdata()`" is wrong for the main query loop.
    - The `get_post_format()`-returns-false worry is wrong too: `get_template_part()` falls back to `content.php`.
  - It is defensible as a PASS. The 6s are under-justified.
  - This is the host/container numeric-path difference again (68 vs 80), not a retry artefact: the host attempt-1 path never produced a verdict to compare against.
- All three rows carry `low_confidence` with reason `retry`, so a reviewer sees they came from the sampled retry.

## Limits
- One fixed seed is not proof that every loop breaks. 3/3 known loops broke with seeds 1234 and 42 at temp ≥ 0.5. A loop that survives the retry still ends `unjudged`, exactly as today.
- Reproducibility of the retry holds only on a serial server, the same scope as T5.
- I tested container sanity units through the bare b11515 image with the api/allinone llama-server flags, not through the compose stack.

## Files
- Code: `wpjudge/judge.py`, `wpjudge/aggregate.py`, `README.md`, `tests/test_judge.py`, `tests/test_aggregate.py`
- Evidence:
  - `build_notes/e2e_work/v11/retry/`: 117 raw `.txt` outputs, `results.jsonl`, `live_*.json`, `retry_probe.py`, `live_j1.py`
  - `build_notes/e2e_work/v11/real/twentytwelve.4.9.zip`: sha256 `a151ec03…`, the same as the WS-E report
