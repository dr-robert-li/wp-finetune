VERDICT: PASS

Evaluator: independent (did not write the docs). Scope: harness/README.md, root README.md section "Scan a whole plugin or theme (harness)", root CHANGELOG.md entry (git diff), harness/.gitignore, writer notes build_notes/W7.md.

One major defect, so FAIL. Everything else checked is correct or minor. Fixing defect 1 alone gives PASS.

## Defects

1. **major** — harness/README.md:59 "With 4 workers, verdicts are stable but a unit's `overall_score` can move by up to ±6 between runs."
   - Wrong per-unit bound. The ±6 comes from research/r2_judge.md:22 (16 val prompts). Phase 5 measured far larger per-unit movement on the fixture.
   - Evidence (build_notes/e2e_work, same host build, workers=1 `r_dir.json` vs workers=4 `r_w4.json`): `ser_render_search_box` 14 -> 42 (Δ 28); `ser_delete_registration` and `ser_ajax_export_registrations` 46 -> 42. Container all-in-one `aio_report.json` vs r_dir: `ser_render_search_box` 14 -> 46 (Δ 32). Verdicts equal in all pairings (that part is true).
   - Side effect worth stating: per-dimension scores also move, so `cap_reasons` contents change. EVAL_PHASE5.md:39: "r_w4 and aio have only 1 entry below 4" (r_dir has 2: sql_safety 2, security 1).
   - Fix: replace "up to ±6" with the measured fact, e.g. "a unit's `overall_score` and dimension scores can move a lot (fixture: up to 28 points on one unit; the report's overall moved 0.3)". Keep "use `--workers 1` for a reproducible report" (true: dir/zip/tgz/library at workers=1 are identical, Δ 0).

2. **minor** — harness/README.md:111 "Extra container args pass through to `llama-server`."
   - True only for the compose `llama` service (entrypoint is llama-entrypoint.sh, `"$@"` at docker/llama-entrypoint.sh:17). False for the all-in-one image: ENTRYPOINT is allinone.sh, which runs `/bin/sh /llama-entrypoint.sh &` with no `"$@"` (docker/allinone.sh:4). `docker run wpjudge:allinone <args>` drops the args.
   - Fix: "Compose: extra `command:` args pass through to `llama-server`. All-in-one: they do not."

3. **minor** — harness/README.md:145-151 ("Docker entrypoint only" table) with :85-86.
   - docker-compose.yml passes only `WPJUDGE_MODEL_FILE` and `LLAMA_PARALLEL` to the `llama` service. `WPJUDGE_MODEL_REPO` and `LLAMA_SLOT_CTX` set in the shell or `.env` have no effect under compose. Line 86 says to add missing knobs to the `harness` service; these two belong in the `llama` service.
   - Fix: one line: "Compose: add `WPJUDGE_MODEL_REPO` / `LLAMA_SLOT_CTX` to the `llama` service `environment`."

4. **minor** — harness/README.md:139/141/142 (float knobs). Not a doc lie, a code quirk the table invites.
   - `retention_days`, `cap_value`, `coverage_threshold` are annotated float but default to int, and `_cast` uses `type(default)` (config.py:37). `WPJUDGE_RETENTION_DAYS=1.5` raises `ValueError: invalid literal for int()` at import (ran it). Docs could say "whole numbers", or the code should use float defaults (`7.0`, `60.0`, `90.0`). Not fixed here (no edits).

5. **minor** — harness/.gitignore has no `jobs/`. `wpjudge serve` run from harness/ writes `./jobs` (README:119, config default `jobs`). `.dockerignore` excludes `jobs`; `.gitignore` does not. Add `jobs/`.

6. **minor** — CHANGELOG entry: "50 MB upload limit". Code is MiB (`max_upload_mb * 2**20`, api.py:37); harness README correctly says MiB. Also root README "About 11 functions per minute": the unit is units (twentytwentyfour is 57 templates + 3 functions). Wording nits.

7. **minor** — harness/README.md:243 "Scores come only from the `<judge_output>` JSON". parse_judge_response falls back to raw JSON, fenced JSON, then the first embedded `{...}` (judge.py:187-222). Accurate statement: "from the JSON block (normally `<judge_output>`), never from the prose lines". Same nuance for :263 (a runaway with no tag is unjudged only if no fallback JSON parses).

## Verified correct (no defect)
- CLI: flags `path, -o/--output, --workers, --base-url, --model-file` match `scan --help` (binding change 16). `serve --help`: `--host 127.0.0.1`, `--port 8000`, uvicorn workers=1.
- Exit codes: dead base URL -> exit 0, 10/10 unjudged, `low_coverage: true`, overall null, error `APIConnectionError`. Missing path -> exit 2 "no such file or directory". Non-archive -> exit 2 "not a directory, zip or tar archive". ETA line `10 units, ETA ~0.9 min` (units ÷ 11, binding change 14).
- Library: unknown key -> `TypeError: unknown config field`; None ignored; on_result contract; Rejected is ValueError.
- Config table: all 17 fields of config.py with correct defaults; bool parsing (empty = false); tuple parsing; env overrides work (`WPJUDGE_TEMPLATES=0`, `WPJUDGE_CAP_DIMS=security`).
- API (TestClient, auth on, 1 MiB limit): POST no key 401; `/docs` and `/openapi.json` 200 with auth on; `.py` 415; `A.ZIP` 202 (any case); no `file` 422; `file://` webhook 400; `Content-Length: abc` 400; chunked 411; oversize without key 413 (size check before auth); GET bad id 404 with key, 401 without; bad archive job -> `failed`, `Rejected: not a directory, zip or tar archive: upload.tar.gz`, upload deleted, `/functions` returns `[]`.
- Restart: pre-seeded `running` job -> `failed` "interrupted by server restart; resubmit"; queued re-enqueued (api.py:129). Retention: 8-day-old job dir swept, 6-day and fresh kept. Sweep at start, after each job, hourly idle (api.py:105-113).
- Webhook: urllib, timeout 5 s, 2 attempts with 1 s sleep, silent; only on `done`.
- Queue/no-cancel note (optional item), template caveat with 81.6 vs 84.0 and `WPJUDGE_TEMPLATES=0` (binding change 15), restart and retention documented, host-9954-vs-container-b11515 provenance, runaway caveat — all present as PHASE5_RUN resolution requires.
- Numbers vs evidence: classic-editor 39 units / 221 s / 74.9 / not capped; twentytwentyfour 60 / 337 s / 81.7; fixture 10 units + 5 skips, 60 capped, uncapped 68.8, 8 cap reasons, 6 of 8 exactly 4 (r_dir.json); b11515 uncapped 67.7 / 69.7 same verdicts; parity max 8, mean 1.50, host ≥ container 16/16, verdicts 16/16; `add_redirect_helper` prose "(4/10)" vs stored security 5; CSRF-on-hidden-field critique present; 3 `other` findings in classic-editor patterns/anti-patterns. ~11 units/min (measured 10.6-10.7).
- Report schema, skip rows, cap, coverage, dimensions (`low_rate` ≤ 4), rank top 10, worst ties by id: match aggregate.py. Limits table matches extract.py (incl. bin/docs -> test, ratio only above 10 MiB, tar `filter="data"`, `__MACOSX` removal, single-folder unwrap).
- Docker: compose image pin, 127.0.0.1:8000, healthcheck start_period 40m, `WPJUDGE_WORKERS: ${LLAMA_PARALLEL}`; entrypoint URL, `curl -fL -C - --retry 5`, `.part`, name guard exit 2, `-c P*SLOT_CTX`, f16 KV; all-in-one exits if llama dies before /health; 15.44 GiB matches root README.
- .gitignore: `.venv/`, `.pytest_cache/`, `*.egg-info/`, `__pycache__/`, `build_notes/e2e_work/` all confirmed with `git check-ignore -v`.
- No secrets or personal absolute paths in harness/README.md (only the placeholder `WPJUDGE_API_KEY=secret`). Root README and CHANGELOG diffs touch only the new section/entry.
- Style: short active sentences, tables, no filler. CHANGELOG entry is one long paragraph but matches the neighbouring entries' format.
- `pytest harness/tests`: 35 passed (README Tests command works from repo root).

## What I ran
- `git diff README.md CHANGELOG.md`; read config.py, cli.py, api.py, extract.py, aggregate.py, judge.py, __init__.py, Dockerfile, docker-compose.yml, docker/*.sh, pyproject.toml, .dockerignore, PLAN.md binding changes, PHASE5_RUN.md, EVAL_PHASE5.md, research/r2_judge.md.
- `.venv/bin/wpjudge scan --help`, `serve --help`; scan of tests/fixtures/vuln-plugin against `http://127.0.0.1:9/v1`; scan of a missing path and a fake zip.
- Config env overrides via `wpjudge.config.load()` (found the float-cast crash).
- FastAPI TestClient probes for every status code, auth, /docs, bad-archive job, restart (pre-seeded job dirs) and retention (backdated mtimes), in scratch job dirs.
- Python comparison of per-unit scores across r_dir / r_w4 / r_zip / lib_dir / api_report / aio_report, and spot checks in real/classic-editor.json.
- `git check-ignore -v` on the five ignore targets; `pytest -q harness/tests`.
- No GPU servers, containers, edits to project files, or commits.

## Round 2

Independent re-check of defects 1-7 after doc fixes. No edits, no GPU, no commit. Result: all 7 fixed, no new errors. PASS.

1. Fixed. README:59 now says verdicts stayed stable, unit scores can move a lot, one fixture unit moved 28 points, report overall moved 0.3, `cap_reasons` changed. Checked r_dir.json vs r_w4.json: max per-unit |Δ overall_score| = 28; uncapped 68.8 -> 69.1 (0.3); verdict_counts 5/5 both; cap_reasons same 8 entries but scores differ (e.g. ser_render_search_box security 1 -> 4, ser_delete_registration sql_safety 4 -> 2). Accurate. `--workers 1` reproducible kept (README:59, :272).
2. Fixed. README:111 "Compose: extra `command:` args pass through to `llama-server`. All-in-one: they do not." Matches docker-compose.yml (entrypoint llama-entrypoint.sh) and docker/allinone.sh:4 (no `"$@"`).
3. Fixed. README:153 states compose passes only `WPJUDGE_MODEL_FILE` and `LLAMA_PARALLEL` to `llama`, add `WPJUDGE_MODEL_REPO` / `LLAMA_SLOT_CTX` to the `llama` service `environment`. Matches docker-compose.yml:6-8. README:85 "(both services)" for MODEL_FILE also correct.
4. Accepted as documented. README:155 "take whole numbers" for RETENTION_DAYS, CAP_VALUE, COVERAGE_THRESHOLD. config.py:16/17/28 int defaults + `_cast` type(default) confirm the int-only behaviour.
5. Fixed. harness/.gitignore now has `jobs/`.
6. Fixed. CHANGELOG "50 MiB upload limit"; root README "About 11 units (functions or templates) per minute". CHANGELOG measured line also says "units".
7. Fixed. README:247 "from the JSON block (normally `<judge_output>`), never from the prose lines"; README:267 "If the retry also runs away and no JSON block parses, the unit is unjudged". Matches judge.py fallback chain.

Nit (not a defect): README:269 still says "uses only the `<judge_output>` scores"; :247 already gives the precise rule and the point there (prose vs. scores) holds.

Diff scope: root README/CHANGELOG diffs touch only the new harness section/entry; no other harness README claims changed beyond the fixes.
