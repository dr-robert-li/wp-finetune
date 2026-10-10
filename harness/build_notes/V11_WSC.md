# V11 WS-C build notes: T2b critique sync + T4 confidence

Branch `worktree-agent-a1ae90a2d758cc840`. Worktree `/home/rli/Desktop/projects/wp-finetune/.claude/worktrees/agent-a1ae90a2d758cc840`. Base: main 5e5ea1a (plan + research).

## What changed
- `wpjudge/judge.py`
  - `_judge_create`: both `create` calls send `logprobs=True` (no `top_logprobs`).
  - new `verdict_prob(choice)`: `getattr(choice, "logprobs", None).content`, joins the tokens, finds `<judge_output>`, then `"verdict"\s*:` after it. Returns `round(exp(logprob), 3)` of the first token whose part after the colon is non-blank and non-quote. Returns `None` when there are no logprobs, no tag or no verdict.
  - new `_sync_headers(critique, scores)`: `aggregate.DIM.sub`. It records `prose_scores[dim]` (first occurrence; `None` for "None/10"), then rewrites the header part (`score N/10` or `severity w (N/10)`) to `Label: score <scores.get(dim)>/10` and keeps the `— reason` tail.
  - `to_judgment(unit, raw, attempts, verdict_p=None)`: still cfg-free. Adds `verdict_p` and `prose_scores` and stores the synced critique. `raw` is unchanged.
  - `judge_one`: passes `verdict_prob(r.choices[0])` to `to_judgment`, so the value comes from the attempt that produced the judgment.
- `wpjudge/aggregate.py`
  - `_row(j, cfg)`: adds `prose_scores` (default `{}`) and `verdict_p` (forced to `None` on unjudged rows). It also adds `confidence_reasons` (`"verdict_p"` when p < `cfg.min_verdict_p`; `"prose_disagrees"` when, for some `cfg.cap_dims` dimension, prose and JSON sit on different sides of `cfg.cap_threshold`, with None, N/A or absent counted as above) and `low_confidence`.
  - `_skip_row`: `verdict_p: None, prose_scores: {}, confidence_reasons: [], low_confidence: False`.
  - `build_report`: builds the rows once and sets `summary.low_confidence` to the count of flagged rows.
- `wpjudge/config.py`: `min_verdict_p: float = 0.9` (`WPJUDGE_MIN_VERDICT_P`; 0 disables).
- `tests/test_judge.py`:
  - `test_critique_synced_to_json` (severity_format_9: prose sec 4 → header "score 6/10", SQL N/A → "None/10"; score_format_0 N/A dims);
  - `test_verdict_prob` (4 cases: the plain path, a split `"`/`FA`/`IL` token, a `"verdict"` before the tag that must be ignored, and no verdict);
  - `test_verdict_prob_absent`, `test_verdict_p_through_judge_one`, `test_template_rejected_path_sends_logprobs`;
  - `test_request_shape` asserts `logprobs is True` and no `top_logprobs`;
  - the Stub accepts `(content, logprobs)` tuples.
- `tests/test_aggregate.py`: `test_prose_disagrees` (the R4 cases, including `cap_threshold=5`), `test_cap_ignores_prose`, `test_verdict_p_confidence` (default, 0 disables, None, summary count, unjudged and skip rows). Also updated the exact skip-row dict in `test_worst_ties_and_rows`.
- `tests/e2e.sh` (chk.py):
  - new `confidence(r, tag, exp=None, soft=False)` checks four things:
    - every DIM header number equals `scores[d]` or `None`;
    - every judged row has `0 ≤ verdict_p ≤ 1` (non-null, which covers the api/aio non-null check);
    - `summary.low_confidence` equals the flagged rows;
    - re-aggregating via `config.load()` with env `WPJUDGE_MIN_VERDICT_P=0` leaves no `verdict_p` reason.
  - With `exp`, `cap_reasons` ids must be exactly the 5 planted ids. That is a hard check, and only a WARN for `api*`/`aio*` reports.
  - `cmd_fixture` calls it.
  - `cmd_real` calls it, plus a soft INFO/WARN for classic-editor `add_redirect_helper` `prose_disagrees` and an INFO line with `low_confidence/judged`.
  - `sec_cli` adds `chk fixture` on r_dir/r_zip/r_tgz (serial runs only; w4 excluded).

## Tests
`.venv/bin/python -m pytest tests/` → **46 passed** (35 before).

## Live run
`GGUF=… HOST_PARALLEL=1 WPJUDGE_WORKERS=1 WORK=<scratchpad>/e2e flock /tmp/wpjudge-gpu.lock tests/e2e.sh cli serverdown`, host llama-server build 9954, server killed at the end. Result: 144 passed, 24 failed.
- **All 24 FAILs are a worktree artefact, not WS-C.** `tests/fixtures/vuln-plugin/build/index.asset.php` is gitignored (`build/`), so it is missing from every git worktree. Every report then gets "skip rows/skip counts/files_scanned 6 == 7". Main's checkout has the file. The orchestrator or the evaluator should copy it in (`cp ../../../harness/tests/fixtures/vuln-plugin/build -r tests/fixtures/vuln-plugin/`) or force-add it to git.
- Every WS-C check passed on r_dir, r_zip and r_tgz:
  - headers synced (bad []);
  - verdict_p non-null on 10/10, min 0.887;
  - `summary.low_confidence` = 3/10;
  - the `MIN_VERDICT_P=0` check passed;
  - `cap_reasons` ids are exactly the 5 planted bugs, with no clean unit.
- Per-unit result (identical across dir/zip/tgz/w4; uncapped 68.8, the same as v1 r_dir):
  - `ser_ajax_export_registrations` p .887 and `ser_save_settings` p .891 get `verdict_p`. They are correct FAILs, matching R4's note that planted units sit at p ≈ .89.
  - `ser_handle_cancel_registration` gets `prose_disagrees`: prose sec 4, JSON 8. This is the R2 clean-unit case, which is now flagged and not capped.
  - Every other unit has p ≥ .917.
- Planted < clean holds: planted [36, 46, 46, 14, 46], clean [80, 80, 80, 84]. The SQLi units cap.
- serverdown: all WS-C-relevant checks pass (unjudged rows have `verdict_p` null and are not flagged).
- Not measured by the builder: the logprobs cost vs a no-logprobs baseline (the C evaluator's job per the plan). Last decode in the log: 13.45 ms/token at 1 slot.

## Deviations
- `prose_scores` keeps the first occurrence when a dimension header repeats; the rewrite still syncs every occurrence.
- `e2e.sh`: besides `cmd_fixture` (owned), I added one line to `sec_cli` and a `confidence`/classic-editor block to `cmd_real`. Both are needed for the "hard on host cli" and classic-editor soft checks. These lines may overlap with B (cmd_real) and D (sec_cli) at merge time.
- The PLAN line "Do not commit" conflicts with the builder brief; I followed the brief and committed on the worktree branch.
