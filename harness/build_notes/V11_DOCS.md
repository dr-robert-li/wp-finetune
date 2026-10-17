# v1.1 WS-F: docs

Work on main (HEAD 7b41194), uncommitted. I checked every default, env var and field against the code (`config.py`, `judge.py`, `aggregate.py`, `api.py`, `cli.py`, `__init__.py`, `docker/`, `docker-compose.yml`). Numbers come from `V11_E2E.md`, `EVAL_V11_WSE.md`, `V11_RETRY.md`, the WS-A to WS-D evaluator reports and the saved reports in `e2e_work/v11/`.

## harness/README.md
- Model server: `--parallel 1 -c 8192`. The old "keep `--workers` equal to `--parallel`" text is removed. Fast mode (`--parallel 4 --workers 4`) is opt-in and marked not reproducible.
- CLI: `--workers` default 1; ETA example 7.1 min at 5.5 units/min; the stderr warning when `reproducible` is false. The old "4 workers" text is removed.
- New `## Reproducibility` section: batch invariance, serial defaults, `cache_prompt: false` on every request (why, no measured cost, the fallback that drops it), `server_slots`, `reproducible`, the 1-slot / workers-1 scope, shared-server limit, the −46% cost.
- Compose and all-in-one: `LLAMA_PARALLEL` default 1. `WPJUDGE_WEBHOOK_ALLOW_NETS` named as a knob to add to the `harness` service.
- Configuration table: `WPJUDGE_WORKERS` 1, `WPJUDGE_CAP_THRESHOLD` note (bound for `low_rate` and anti-patterns; keep 4), new rows `WPJUDGE_PATTERN_MIN` 8, `WPJUDGE_MIN_VERDICT_P` 0.9, `WPJUDGE_WEBHOOK_ALLOW_NETS`; entrypoint `LLAMA_PARALLEL` 1.
- API: 400 row covers webhook resolve failures and blocked addresses. `/functions` rows carry `verdict_p` and `prose_scores`. New webhook SSRF guard text (blocked targets, unwrapped forms, submit and send checks, IP pinning, no redirects, `HTTP(S)_PROXY` not honoured, allowlist, known limits: IP echo in the 400 detail, no DNS timeout). The "not filtered by host" line is removed.
- Report: `model.workers/server_slots/reproducible`, `summary.low_confidence`, thresholds named by config, the four new row fields, skip-row defaults.
- Scoring rules: header rewrite and `prose_scores`, the tail-less header gap, the 27-topic lexicon, the retry sentence now says "on a serial server" and that a surviving loop ends unjudged.
- New confidence-flags list: `verdict_p`, the three reasons, `low_confidence`, the ~25% `prose_disagrees` rate, logprobs always sent (+6%, outright rejection = unjudged), no re-judge mode (D3), FN = systematic leniency.
- Limits table: new "Known false cap" row (classic-editor).
- Caveats rewritten: runaway retry, drift (dropped "host ≥ container on all 16", no longer true), prose vs scores, the classic-editor nonce false positive, false negatives, lexicon minors. "Parallel slots" removed (now in Reproducibility).
- Measured table: v1.1 serial numbers for the fixture, classic-editor, twentytwentyfour, twentytwelve; container 5/5; `low_confidence` counts; twentytwelve measured before the varied retry.

## harness/SPEC.md
- New "Amendments v1.1" table B1-B7 (T1, T2a, T2b/D1, T3, T4/D3, T5, Decision 2). `schema_version` stays 1.0.
- Schema snippet: `model` v1.1 fields, `summary.low_confidence`, the four row fields.
- "Next" points to HANDOVER.

## CHANGELOG.md (repo root, the only changelog)
- New `wpjudge` v1.1 entry under Unreleased/Added, above the 1.0.0 entry.

## harness/HANDOVER.md
- State: v1.1 done, pending commit and push. v1.1 summary, next steps (doc evaluator, `pyproject.toml` version still 1.0.0, commit, push, worktree cleanup), v1.2 candidates (nonce training data, D4 `rebuild_targets.py` gap with R2's counts, FN leniency), known v1.1 minors, v1 carry-over items (`/docs` auth, whole-number float knobs).

## Not changed
- `.planning/`, code, tests, the root README.
