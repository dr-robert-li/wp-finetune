# v1.1 WS-F doc evaluation

Scope: the uncommitted diff of `CHANGELOG.md`, `harness/README.md`, `harness/SPEC.md` and `harness/HANDOVER.md` on main 7b41194. I checked it against `wpjudge/*.py`, `docker/`, `docker-compose.yml`, `Dockerfile`, `tests/e2e.sh`, the `V11_*` / `EVAL_V11_*` notes, `research/v11_r*.md`, the worktree WS-C evaluator report, and the saved reports in `build_notes/e2e_work/v11/`. I edited nothing except this file.

## Commands
- `git diff -- CHANGELOG.md harness/README.md harness/SPEC.md harness/HANDOVER.md`
- `.venv/bin/python -m pytest tests/ -q` → **72 passed**, 1 warning (the known `jobs` thread warning).
- `python -c "from wpjudge.aggregate import TOPICS; print(len(TOPICS))"` → 27.
- Python over `e2e_work/v11/r_dir.json` and `real/*.json`: summary, `model`, `confidence_reasons` counts, `other` in the top 10.
- Greps for stale text (`4 workers`, `equal to --parallel`, `11 units`, `16-topic`, `not filtered`, `host ≥`) and for markdown links/anchors.

## Verified (matches the code or the evidence)
- Defaults: `config.workers` 1 (config.py:12); `LLAMA_PARALLEL` 1 in compose (llama and `WPJUDGE_WORKERS`) and `llama-entrypoint.sh`; `Dockerfile` sets no workers/parallel, so the all-in-one gets 1 and 1. `-c` = P × `LLAMA_SLOT_CTX` (8192).
- No stale "keep `--workers` equal to `--parallel`", "4 workers", "11 units/min" or "not filtered by host" text in the changed docs. SPEC R3.2 / A5 still state v1 values, but B4 / B6 amend them. This is the SPEC amendment pattern.
- `cache_prompt: False` in `extra_body` on every request (judge.py:30-33). The fallback drops both `chat_template_kwargs` and `cache_prompt` for the process and warns once (judge.py:34-44). e2e `_send` also sends it (e2e.sh:285).
- Retry: `RETRY_SAMPLING = {"temperature": 0.5, "seed": 1234}` on attempt 2 only (judge.py:22, 165). "Reproducible on a serial server" is in README:281, the SPEC B7 row and the caveat. `retry` reason is set when `attempts == 2` and judged (aggregate.py:118).
- ETA `total / 5.5` (cli.py:30): 39 → 7.1 min. The warning text prefix matches cli.py:39.
- `reproducible = workers == 1 or slots == 1`; `server_slots` from `/props` `total_slots`, `None` on error (judge.py:180-187, aggregate.py:157-163).
- Webhook guard (api.py:62-104): scheme check, `getaddrinfo`, `is_global`/`is_multicast`, ipv4_mapped plus NAT64 / ::/96 / SIIT unwrapping, `strict=False` allowlist ("host bits are ignored"), submit 400 before the job dir is made, send-time recheck with no retry on `ValueError`, IP pinned via `_create_connection`, `http.client` (no redirects, no proxy env). The 400 detail does echo `({ip})`. `getaddrinfo` has no timeout. All as documented.
- Env names `WPJUDGE_CAP_THRESHOLD` 4, `WPJUDGE_PATTERN_MIN` 8, `WPJUDGE_MIN_VERDICT_P` 0.9, `WPJUDGE_WEBHOOK_ALLOW_NETS` () all match config.py. `low_rate`, `rank(lo=cap_threshold, hi=pattern_min)` match aggregate.py:149, 152.
- Row fields `verdict_p`, `prose_scores`, `confidence_reasons`, `low_confidence`; unjudged rows get `verdict_p: null`, `prose_scores: {}`; skip-row defaults match `_skip_row` exactly; `summary.low_confidence` = row count. `prose_disagrees` semantics (cap dims, other side of `cap_threshold`, None/absent = above) match aggregate.py:115-117. `/functions` rows (judge output minus code/raw) carry `verdict_p` and `prose_scores`, not the confidence fields. This matches README:205-206.
- `schema_version` "1.0" (aggregate.py:165).
- 27 topics. `other` absent from the top 10 on all three real reports.
- Numbers: 426 s / 620 s / 675 s, 60 (uncapped 73.9), 81.2, 78.3, 54 units, 53 judged, fixture 60 / 68.8 / 8 reasons on 5 units, 6 of 8 reasons at exactly 4 (r_dir cap_reasons), byte-identical to v1, low_confidence 3/10, 9/39, 1/60, 7/53, verdict_p 0.918, parity max |Δ| 8 mean 1.5, 5/10 units and 14 → 42 at w4, 10.0 → 5.4 (−46%, ≈1.9×), +6.0% per token (worktree EVAL_V11_WSC:76), 1789 headers (EVAL_V11_WSC:21), 25% (R4/PLAN), +14 to 400% (R4/D3), 3/3 loops (V11_RETRY:133), 425 checks, merge range 66910b3..a8c457f.
- Classic-editor accepted limit: README:293 Limits row, README:308 caveat (request nonce vs CSP nonce, verdict_p 0.918, no flag), HANDOVER v1.2 item 1. D4: HANDOVER v1.2 item 2 names `scripts/relabel/rebuild_targets.py`, 29%, 67% (PLAN D1, R2 table row "570 (67%)").
- CHANGELOG: the repo-root `CHANGELOG.md` is the project's only changelog (the others are in `.claude/worktrees/` copies). The v1.0.0 entry is there too. The new entry sits under Unreleased/Added, above 1.0.0. Correct file.
- No markdown links or anchors in the changed text, so none can be broken. Cross-refs ("see Reproducibility", "see Cross-build drift") name existing headings and bullets.
- STE: mostly short active sentences.

## Findings

1. **MINOR: container 5/5 provenance is overstated.** README:333 says "On b11515 (compose and all-in-one, serial), all 5 planted bugs are capped". CHANGELOG.md:16 and HANDOVER.md:16 say "5/5 planted capped on host and container". The evidence differs. WS-E's compose/all-in-one runs gave **4/5** (ser_search_registrations unjudged, coverage 90%; V11_E2E.md "Fixture" section). The 5/5 comes from a `judge_one` probe on the bare b11515 image after the varied retry. V11_RETRY.md:176 says: "I did not re-run the api/allinone e2e sections". The inference is sound, because the retry only touches units whose attempt 1 failed. But the docs give a stack run that did not happen. Fix: README:333 → "On the b11515 image (serial, varied retry; single-unit probe, the compose and all-in-one sections were not re-run), all 5 planted bugs are capped …". CHANGELOG:16 / HANDOVER:16: "container (b11515 probe)". CHANGELOG:16 also reads "Measured (host build 9954, serial): … container", which mixes the two builds.
2. **MINOR: README:131 "`--workers 1` (which always sends `cache_prompt: false`)".** After the fallback (judge.py:34-44), every later request drops `cache_prompt`, and `reproducible` still reports true. README:128 states the fallback, so the "always" here contradicts it. Fix: "(which sends `cache_prompt: false` unless the server rejected it)".
3. **MINOR: CHANGELOG.md:8 "Fixes the five v1 follow-ups".** The v1 HANDOVER listed five follow-ups that included auth on `/docs` and did not include worker stability. `/docs` auth is still open (HANDOVER.md "Carry-over v1 open items"). v1.1 fixed the user's tasks T1-T5. Fix: "Fixes the five v1.1 tasks (T1-T5)" or "four v1 follow-ups plus worker-count stability".
4. **INFO (code comment, outside WS-F scope):** config.py:15-16 and PLAN_v1.1.md:94 say "7 of 8 planted-bug cap reasons sit at exactly 4". The v1.1 golden `r_dir.json` has 6 of 8 (scores 4,2,4,4,4,4,1,4). The README (166, 299) says 6 of 8, which matches the report. Correct the code comment when the code is next touched.
5. **INFO (STE):** README:281 has a 40-word sentence with a parenthetical. Split it: "The first try is greedy. A greedy retry would repeat a runaway byte for byte. So the retry samples at temperature 0.5 with seed 1234. It stays reproducible on a serial server."

No BLOCKER or MAJOR. Every default, env var, flag, field and v1.1 number checked matches the code or a recorded measurement. The three MINORs are wording fixes.

VERDICT: PASS
