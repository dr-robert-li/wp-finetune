VERDICT: PASS

# W2 judge: independent evaluation

Scope: `wpjudge/judge.py` (159 lines), `tests/test_judge.py` (9 tests), and `tests/fixtures/responses/` (5 files). They were checked against the PLAN Contracts, the W2 Locked plan, Binding changes 9 and 10 and the optional `submit`/`as_completed` item, `research/r2_judge.md`, and `eval/eval_judge.py`.

## Conformance (all pass)
- **`parse_judge_response`:** `diff <(sed -n 310,383p eval/eval_judge.py) <(sed -n 40,113p wpjudge/judge.py)` is empty, so the copy is verbatim.
- **`_judge_create`:** matches `eval_judge.py:47-76`, including the fallback that drops the kwarg when the template rejects it. The only changes are the default `max_tokens` (2048) and the warning text.
- **Prompt:** exactly `"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"` as one user message, with no system message. The test asserts the full messages list. Code containing `{}` is safe, because only the template is formatted.
- **Request settings:**
  - `temperature=0.0`, `max_tokens` 2048 by default, and the value is passed through (the test checks 123).
  - Thinking is turned off with `extra_body.chat_template_kwargs.enable_thinking=False`.
  - The client is `OpenAI(base_url, api_key="none", timeout=600, max_retries=0)`. A probe captured these kwargs, which satisfies Binding change 9.
- **Retry:**
  - An exception or an unusable parse leads to a second attempt, and then `unjudged` with `attempts: 2`.
  - `error` comes from the last attempt.
  - A success reports its attempt number (1 or 2).
- **Scores:**
  - Taken only from the parsed JSON, with native keys and `verdict`/`overall_score` excluded.
  - Floats like `7.0` are cast to int.
  - `7.5`, `"8"`, `true` and `null` are dropped (probed).
  - A string or bool `overall_score` is rejected and counts as `parse_failure`. A JSON list is rejected.
- **Critique:**
  - The text before `[/REASONING]`, falling back to the text before `<judge_output>` (a harmless superset of the contract).
  - The tests assert that neither marker leaks into it.
- **Judgment keys:** the Unit fields + `status, attempts, verdict, overall_score, scores, critique, raw, error`, on both the judged and unjudged paths. This matches the Contract.
- **`judge_units`:**
  - The signature matches the Contract, plus an extra `max_tokens` keyword (which `scan()` passes).
  - It uses `submit` + `as_completed`.
  - A 40-unit probe with `workers=4` confirmed that:
    - 4 requests ran at once (`peak 4`);
    - `on_result` was called 40 times, always on the calling thread;
    - the output is sorted by `id`.
  - Empty input returns `[]`.
- **Thread safety:**
  - One shared OpenAI client (httpx, which is thread-safe) is used, with no shared mutable state except `_thinking_kwarg_supported`.
  - The race on that flag is benign: at worst, one extra request that gets rejected.
- **Fixtures:**
  - All 5 files are byte-identical to the responses at indexes 0, 5, 7, 9 and 89 of `output/shrink/iq4xs_im_f16/judge_responses.jsonl`, so they are real captures.
  - Both prose formats are covered, and `broken_89` is the real unparseable repetition loop.
  - Over the whole capture, `to_judgment` parses 120/121, which matches r2.
- **Plan scope:** `DIM`, `TOPICS` and `rank` are absent, as Binding change 6 requires. There is no overflow-halving. Nothing is over-engineered.

## Defects
1. **minor:** a non-finite `overall_score` is misreported.
   - Where: `wpjudge/judge.py:119-126`.
   - What happens: if the JSON carries `NaN` or `Infinity` (which `json.loads` accepts), `int(round(...))` raises `ValueError`/`OverflowError` instead of returning None. `judge_one` catches it, so the unit is still retried and then marked unjudged. The only effect is that `error` reads `"ValueError: cannot convert float NaN to integer"` instead of `parse_failure`, and a direct call to `to_judgment` raises.
   - Fix: add `math.isfinite(j["overall_score"])` to the guard.
2. **minor:** an exception raised inside `on_result` aborts the whole run.
   - Where: `wpjudge/judge.py:155-157`.
   - What happens: the exception propagates out of `judge_units`. The pool waits for the in-flight futures, and then every result is lost.
   - Assessment: callers (`scan`, `api`) own their callbacks, so this is acceptable. If needed, wrap the call in try/except and log.
3. **minor:** `verdict` is passed through unvalidated.
   - Where: `wpjudge/judge.py:125`.
   - What happens: the Contract says `"PASS|FAIL|null"`, but any value from the model is stored. The plan explicitly says "verdict_counts taken from the model verdicts as given", so this is acceptable. Note it for W3.
4. **minor:** the template-rejection fallback is not covered by the test file.
   - My probe passed: the first call carries `extra_body`, it is rejected with a "chat_template" error, the retry without the kwarg is judged with `attempts` 1, and the flag is left False.
   - Fix (optional): add one stub test, resetting the module global afterwards.
5. **minor (acknowledged by the builder):** strategies 1-4 of the verbatim `parse_judge_response` can read JSON from outside `<judge_output>`.
   - This is still JSON, never prose, and it is required by the verbatim-copy rule.
   - The process-wide `_thinking_kwarg_supported` flag never resets in a long-lived API process. This mirrors the original.

## What I ran
- `.venv/bin/python -m pytest tests/test_judge.py -q`: 9 passed.
- The `diff` of the parser lines: empty.
- A scratchpad probe (`probe.py`, outside harness/) that checked:
  - fixtures byte-compared with the source jsonl;
  - the full-capture parse rate (120/121);
  - score and overall edge values;
  - `None` content;
  - NaN;
  - the template fallback;
  - concurrency and ordering, the thread `on_result` runs on, and the captured client kwargs;
  - empty input;
  - an `on_result` that raises.
- No server, GPU or network was used.
