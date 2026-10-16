# Evaluation: v1.1 Decision 2, varied retry (t05s)

Independent evaluator. Scope: main HEAD 1fbd472 plus the uncommitted diff (`wpjudge/judge.py`, `wpjudge/aggregate.py`, `README.md`, `tests/test_judge.py`, `tests/test_aggregate.py`). I edited nothing except this file.

User criterion: the retry must produce an attempt that parses AND is defensible, meaning a reasonable assessment of the actual code. I judged the three recovered units myself against the source (twentytwelve 4.9 `search.php`, fixture `includes/legacy.php:18-25`).

## 1. Recovered loop units (attempt 2, t05s)

### host `search.php:<template>`: 68 PASS, verdict_p 0.997. Parseable: yes. Defensible: yes, with reservations
The code is clean core-theme code. The one dynamic value, the search query, goes out as `esc_html( get_search_query( false ) )`. There is no SQL. Strings are translated with the right text domain, and the `%s` string has a translators comment. The only real lint finding is `printf( __( ... ) )`, which echoes the translated format string unescaped (WPCS EscapeOutput; `esc_html__` would fix it).
- **Verdict and cap are right.** PASS, with no cap dimension at or below 4 (security 6, sql_safety absent/None). The overall of 68 is low for this code, but it sits inside the model's own range for this unit (t07s 64, dry08 68, drynl 72, t07s42 79, container 80). It is not an outlier caused by the retry.
- **Correct claims:** the escaping is right for the HTML context; there is no SQL and WP_Query handles it; no nonce is needed on a public template; the i18n and translators-comment points; the semantic HTML points.
- **Wrong claims:**
  1. It says the template lacks `wp_reset_postdata()` (said twice, under WPCS and Code Quality). That function is for secondary `WP_Query` loops, not the main loop. Wrong.
  2. It says `get_post_format()` returning false "may cause issues in template parts". In fact `get_template_part('content', false)` falls back to `content.php`. Wrong.
  3. It says "`get_search_query()` returns unescaped data by default". The default is `$escaped = true`; the code passes `false` explicitly. Wrong (the conclusion still holds).
  4. It says `get_search_query()` "is called inside the loop context for the title". The call is on line 21, before the `while`. Wrong.
  5. It says the query is "escaped with `esc_html()` before being passed to the translation function... (escape after translation)". This contradicts itself: the escaped value is a `printf` argument, not input to `__()`. Muddled.
- **Missed:** the unescaped `__()` output in `printf`.
- **Summary:** the scores of 6 (wpcs, security, performance, accessibility) rest partly on false reasons. Still, the outcome (a clean template PASSes and nothing is capped) is a reasonable assessment, and the row carries `low_confidence: ["retry"]`.
- **Choice vs t07s:** on this same unit, t07s (`host_search.php_template__t07s.txt`) hallucinates that "`esc_html` is not a standard WordPress function", calls it a "critical XSS", and scores security 5. That is much worse than t05s. So t05s is the more defensible choice here, not just the lower-drift one.

### container `search.php:<template>`: 80 PASS, verdict_p 0.999. Parseable: yes. Defensible: yes
- **Verdict and cap are right.** Scores of 7-8 fit the code.
- **Correct claims:** escaping (`esc_html` around `get_search_query(false)`); no SQL, with WP_Query preparing it internally; no nonce needed; the translators comment is present; the list of APIs used; the semantic HTML.
- **Wrong claims:** none material. A trivial slip: it says the dependencies are "standard for a Twenty Twelve child theme", but this is the parent theme.
- **Missed:** the unescaped `__()` output in `printf`, same as the host. It is a minor lint item.

### container `ser_search_registrations`: 42 FAIL, sql_safety 2, security 4, verdict_p 0.963. Parseable: yes. Defensible: yes
- **The core finding is right.** Raw `$_GET['q']` is concatenated into `get_results()`, which is SQL injection. Hence sql_safety 2, the cap fires, and the verdict is FAIL. The `esc_like` advice and the missing capability check are right, and `SELECT *` is a fair performance note. It matches the host attempt-1 judgment (36, sql 2, sec 4).
- **Wrong claims:**
  1. The suggested fix writes `LIKE %%s` inside `$wpdb->prepare()`. `%%` is a literal `%`, so the placeholder never binds. It should be `LIKE %s`. Wrong (appears in both the WPCS and SQL fixes).
  2. The WPCS fix leaves out `esc_like()`, while the SQL fix includes it. Inconsistent, minor.
- **Summary:** security 4 is lenient for an injectable query, but the injection itself is scored under sql_safety 2, so the cap is unaffected. The prose header says security 2 against JSON 4. Both are at or below the threshold, so there is no `prose_disagrees`.
- **Missed:** no `isset` check on `$_GET['q']`, and the leading-wildcard LIKE cannot use an index. Both are minor.

## 2. Sanity units (parseable on attempt 1)
- I read `results.jsonl` for the t05s rows on both servers.
- No planted bug passes. Every planted unit stays FAIL with a cap dimension at or below 4:
  - host: ssr sql 2; delete sec 4 / sql 4; render sec 1
  - container: delete 4/4; render sec 2
- No clean unit is capped. `ser_get_registrations_for_event`, `ser_sanitize_settings` and `twentytwelve_comment` all have security ≥ 7 and sql ≥ 8, and all PASS.
- No verdict flips.
- **Container max |Δ| of 16 (`ser_render_search_box`, 42 → 26):**
  - I read both raw outputs.
  - The t05s output names all three reflected XSS sinks correctly: `$_GET['page']` and `$_GET['q']` in attributes, and `$_REQUEST['q']` in text. It sets JSON security to 2.
  - Attempt 1 said "critical (2/10)" in prose but put security 4 in the JSON.
  - The t05s score is more defensible than the a1 score for three unescaped echoes.
  - One wrong claim: it recommends `esc_text()`, which is not a WP function, though it hedges with "(or `esc_html()`)".
- Note that the retry only ever runs on units that fail attempt 1. These sanity units are a proxy for drift, not a path that ships.

## 3. Code review
- **Attempt 1 is unchanged.**
  - `judge_one` passes `**({})` on attempt 1, so `_judge_create` gets `temperature=0.0` and no `seed`. The extra_body path and the fallback path are otherwise the same as before.
  - `test_request_shape` pins the attempt-1 kwargs.
  - `test_retry_after_unparseable_then_ok` asserts `a1["temperature"] == 0 and "seed" not in a1`, and `a2 == {**a1, temperature 0.5, seed 1234}`. That is a strict equality, so no other kwarg drifted.
- **Attempt 2 is deterministic.** It always uses the module constant `RETRY_SAMPLING`. I confirmed this live (§4).
- **Fallback.**
  - `**sampling` is passed top-level on both branches. The rejection logic is unchanged.
  - `test_retry_sampling_survives_extra_body_fallback` drives attempt 2 into a template rejection and checks that the fallback call carries temperature and seed but no extra_body.
  - `monkeypatch` restores the global afterwards.
  - Edge: a server that rejects `seed` would raise a non-template error, and the unit would end `unjudged`, which is the same as today. Acceptable: `seed` is a standard OpenAI param.
- **`retry` reason.**
  - It is added only for judged rows with `attempts == 2`. Unjudged rows keep `[]`.
  - It also fires when attempt 1 was a transport error. That is consistent, because attempt 2 was still the sampled request.
  - It does not touch score or cap. No schema change.
  - `test_retry_confidence` covers both the judged and the unjudged case.
- **Tests are meaningful.** They would fail if attempt 1 gained a seed or temperature, if attempt 2 lost them, or if the fallback dropped them.
  - `.venv/bin/python -m pytest tests/ -q`: **72 passed**, 1 warning. The warning is the pre-existing `PytestUnhandledThreadExceptionWarning`.
  - `wpjudge.__file__` resolves inside this repo.
- **README sentence (L248) is accurate.**
  - Retry at temp 0.5 with a fixed seed: correct. The first try is greedy: correct. The identical greedy retry repeating a loop: verified (host a1 md5 matches WS-E). The `retry` reason: correct.
  - "so it stays reproducible" has no serial-server qualifier. L59 and L272 already say that only `--workers 1` is reproducible, so it is not misleading. Minor.

## 4. Live verification
- **`tests/e2e.sh api`.**
  - Setup: compose, b11515, llama-server `--parallel 1 -c 8192`, checked via `/proc/1/cmdline`. I confirmed the harness image contains `RETRY_SAMPLING`. Fresh WORK under my scratchpad.
  - Result: **52 passed, 0 failed, 0 WARN lines.**
  - `cap_reasons ids == the 5 planted bugs (extra [], missing [])` now prints **PASS**, so the soft-WARN is cleared.
  - Coverage is 100% (10/10 judged), overall 60, capped.
  - `ser_search_registrations` was judged on attempts 2: 42 FAIL, sql 2, sec 4, `["retry"]`. Its critique and scores are byte-identical to the builder's `live_ctr_ser_search_registrations.json`.
  - All other units were judged on attempt 1.
- **Host determinism.**
  - I judged host `search.php:<template>` through the shipped `judge_one` + `aggregate._row`, on two fresh `llama-server --parallel 1 -c 8192` starts (flags as in e2e.sh, port 8097).
  - Both runs: attempts 2, 68 PASS, vp 0.997, raw md5 `88c6aa9a6df9f20c9f0262d8937470f3`. The output files are identical (`cmp`), and the raw text is identical to the builder's `live_host_template.json`.
- **Cleanup.** After the run, `nvidia-smi` shows no compute apps, `docker ps` is empty, and no `llama-server` process is running.

## Findings
- **MINOR-1.** The host `search.php` retry critique contains five factually wrong or muddled claims (§1). The scores of 6 are under-justified. The verdict and cap are correct and the row is flagged `retry`. The builder disclosed two of the five.
- **MINOR-2.** The container `ser_search_registrations` fix suggests `LIKE %%s` in `prepare()`, which is broken advice, and the WPCS fix omits `esc_like`. The cap and verdict are unaffected.
- **MINOR-3.** Both `search.php` critiques miss the unescaped `printf( __() )` translation output.
- **MINOR-4.** The sanity set leaves out the two other planted units (`ser_ajax_export_registrations`, `ser_save_settings`). The container penalty candidates were not run on the sanity units. This does not affect the choice: t05s is the only low-drift 3/3 option.
- **MINOR-5.** The README's "stays reproducible" lacks the serial-server qualifier. It is covered elsewhere in the README.

No BLOCKER or MAJOR findings.

## Commands run
- `git diff -- wpjudge tests README.md`; read `V11_EVAL_BRIEF.md`, `V11_RETRY.md`, `results.jsonl`, `live_*.json`, and the raw `*.txt` files for t05s, t07s and a1.
- `unzip twentytwelve.4.9.zip` into scratchpad; `cat -n search.php` and `legacy.php`.
- `.venv/bin/python -m pytest tests/ -q`: 72 passed.
- `WORK=<scratch>/e2e_api GGUF=... flock /tmp/wpjudge-gpu.lock tests/e2e.sh api`: 52 passed, 0 failed.
- `docker exec wpjudge-e2e-llama-1 cat /proc/1/cmdline`; `grep RETRY_SAMPLING` inside the harness container.
- Under `flock`, twice: `llama-server -m $GGUF --host 127.0.0.1 --port 8097 -ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}' --parallel 1 -c 8192` + `<scratch>/det.py` (shipped `judge_one`), then kill. `cmp det1.json det2.json`: identical.
- `nvidia-smi --query-compute-apps`, `sg docker -c "docker ps -q"`, `pgrep llama-server`: all empty.

## Verdict
All three recovered loop units parse on the retry. Each gives a correct verdict and a correct cap decision for the actual code: the SQLi is capped and FAILs, and the clean template PASSes uncapped. Their critiques get the central security facts right. The errors are peripheral, and every such row is flagged `low_confidence: ["retry"]`. The fixture now caps all 5 planted bugs at 100% coverage on the compose b11515 serial stack. The retry is byte-deterministic across server restarts.

VERDICT: PASS
