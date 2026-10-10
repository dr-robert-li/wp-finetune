# wpjudge: WP Judge harness

`wpjudge` judges every PHP function in a WordPress plugin or theme with the wp-judge v4 GGUF. It writes one JSON report.

- Input: a directory, `.zip` or `.tar.gz`.
- Extraction: tree-sitter-php. No PHP runtime.
- Judge: any OpenAI-compatible server, normally `llama-server`.
- Output: per-function verdicts and scores, a LOC-weighted overall score with a safety cap, dimension stats, patterns and anti-patterns.
- Use it as a CLI, a Python library, or an async-job HTTP API in a container.

## Install

Python 3.10 or newer.

```bash
python3 -m venv harness/.venv
harness/.venv/bin/pip install -e harness          # add '[test]' for pytest + httpx
```

This installs the `wpjudge` command.

## Run

### 1. Start a model server (host)

Download the default GGUF and start `llama-server` on port 8080, the default `WPJUDGE_BASE_URL`:

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.IQ4_XS.gguf --local-dir ./judge
llama-server -m ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf --host 127.0.0.1 --port 8080 \
  -ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}' \
  --parallel 4 -c 32768
```

- `-c` is `--parallel` × 8192. Keep `--workers` equal to `--parallel`.
- Small file (`wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf`): do not add `-ctk q8_0` / `-ctv q8_0`.

### 2. Scan (CLI)

```bash
wpjudge scan ./my-plugin -o report.json
wpjudge scan my-plugin.zip -o report.json
wpjudge scan my-theme.tar.gz --workers 1 --base-url http://127.0.0.1:8080/v1
```

| Flag | Default | Effect |
|---|---|---|
| `path` | required | Directory or archive. Archives are detected by content: zip, tar, tar.gz/tgz. |
| `-o`, `--output` | stdout | Report path. |
| `--workers` | `WPJUDGE_WORKERS` (4) | Concurrent requests. |
| `--base-url` | `WPJUDGE_BASE_URL` | OpenAI-compatible base URL. |
| `--model-file` | `WPJUDGE_MODEL_FILE` | Model name sent to the server and written to `report.model.file`. `llama-server` ignores it. |

- Stderr shows the unit count and an ETA: `39 units, ETA ~3.5 min` (units ÷ 11 per min).
- Archives extract into a temporary directory. It is removed after the scan.
- Exit 0: report written. Unjudged units (server down, bad output) do not change the exit code. Check `summary.coverage`.
- Exit 2: bad input (missing path, not an archive, unsafe archive). The reason goes to stderr.
- `--workers 1` gives a reproducible report. With 4 workers, verdicts stayed stable, but unit scores can move a lot between runs. On the fixture one unit moved 28 points, the report overall moved 0.3, and `cap_reasons` changed.

### 3. Library

```python
from wpjudge import scan

report = scan("my-plugin.zip", workers=1, base_url="http://127.0.0.1:8080/v1")
```

- `scan(path, on_result=None, **cfg)`. `cfg` keys are the config field names (lowercase, see Configuration). An unknown key raises `TypeError`. `None` values are ignored.
- Precedence: defaults, then `WPJUDGE_*` env, then keyword arguments.
- `on_result(judgment, done, total)` runs once with `(None, 0, total)` after extraction, then once per finished unit.
- Bad input raises `ValueError` (`wpjudge.extract.Rejected`).

### 4. Docker compose (API + llama.cpp)

Needs an NVIDIA GPU and the NVIDIA container toolkit. Run from `harness/`:

```bash
cd harness
docker compose up -d
```

- `llama`: `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515` (pinned; arm64 + sm_121). The GGUF lives in the `models` volume.
- `harness`: built from the `harness` target. It starts when `llama` is healthy. API on `127.0.0.1:8000`. Jobs in the `jobs` volume (`/data/jobs`).
- Compose variables: `WPJUDGE_MODEL_FILE` (both services), `LLAMA_PARALLEL` (default 4; also sets `WPJUDGE_WORKERS`), `WPJUDGE_API_KEY` (default empty = auth off).
- Other `WPJUDGE_*` knobs are not passed through. Add them to the `harness` service `environment`.

### 5. All-in-one image

One container runs `llama-server` and the API. The GGUF is mounted at `/models`, never baked in.

```bash
cd harness
docker build --target allinone -t wpjudge:allinone .
docker run -d --gpus all -v "$PWD/models:/models" -v wpjudge-jobs:/data/jobs \
  -p 127.0.0.1:8000:8000 wpjudge:allinone
```

- The API starts after `llama-server` answers `/health`. If `llama-server` exits first, the container exits.
- `LLAMA_PARALLEL` here does not change `WPJUDGE_WORKERS`. Set both if you change one.
- The harness image (`--target harness`) runs `wpjudge serve --host 0.0.0.0 --port 8000`.

### First-run model download

`docker/llama-entrypoint.sh` runs in both images. If `/models/$WPJUDGE_MODEL_FILE` is missing, it downloads it:

- URL: `https://huggingface.co/$WPJUDGE_MODEL_REPO/resolve/main/$WPJUDGE_MODEL_FILE`.
- `curl -fL -C -`, 5 retries. The download resumes from `<file>.part`. No token needed.
- The default file is 15.44 GiB. The healthcheck allows 40 min (`start_period`).
- The file name must not contain `/` or start with `..` (exit 2).
- Then: `llama-server -ngl 999 --jinja --parallel $LLAMA_PARALLEL -c $((LLAMA_PARALLEL * LLAMA_SLOT_CTX))`, thinking off, f16 KV cache. Compose: extra `command:` args pass through to `llama-server`. All-in-one: they do not.

### Serve without Docker

```bash
WPJUDGE_API_KEY=secret wpjudge serve --host 127.0.0.1 --port 8000
```

`serve` runs one uvicorn worker. Do not run more: the job queue is in memory. Jobs go to `./jobs` unless `WPJUDGE_JOBS_DIR` is set.

## Configuration

Every field in `wpjudge/config.py` reads `WPJUDGE_<FIELD>`. Booleans: `0`, `false`, `no`, `off` or empty mean false. Tuples: comma-separated.

| Env var | Default | Meaning |
|---|---|---|
| `WPJUDGE_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible server. Compose sets `http://llama:8080/v1`. |
| `WPJUDGE_MODEL_FILE` | `wp-judge-v4-pruned-k224.IQ4_XS.gguf` | Model name in requests and the report. The Docker entrypoint also uses it as the file to download and serve. |
| `WPJUDGE_MAX_TOKENS` | `2048` | Generation limit per request. |
| `WPJUDGE_WORKERS` | `4` | Concurrent requests. Match the server's `--parallel`. |
| `WPJUDGE_CAP_DIMS` | `security,sql_safety` | Dimensions that trigger the cap. |
| `WPJUDGE_CAP_THRESHOLD` | `4` | A cap dimension scored at or below this triggers the cap. |
| `WPJUDGE_CAP_VALUE` | `60` | Overall score ceiling when capped. |
| `WPJUDGE_COVERAGE_THRESHOLD` | `90` | Coverage % below this sets `low_coverage`. |
| `WPJUDGE_MAX_CHARS` | `15000` | Functions are truncated at this length. Larger templates are skipped (`too_large`). |
| `WPJUDGE_TEMPLATES` | `1` (on) | Judge function-less PHP files as one template unit. |
| `WPJUDGE_MAX_FILES` | `20000` | Archive member limit. |
| `WPJUDGE_MAX_TOTAL_MB` | `200` | Archive uncompressed size limit (MiB). |
| `WPJUDGE_MAX_RATIO` | `100` | Compression ratio limit. Applies above 10 MiB uncompressed. |
| `WPJUDGE_MAX_UPLOAD_MB` | `50` | API upload limit (MiB). |
| `WPJUDGE_JOBS_DIR` | `jobs` (images: `/data/jobs`) | API job store. |
| `WPJUDGE_RETENTION_DAYS` | `7` | Job directories older than this are deleted. |
| `WPJUDGE_API_KEY` | empty | Bearer key for the API. Empty means auth off. |

Docker entrypoint only (not in `config.py`):

| Env var | Default | Meaning |
|---|---|---|
| `WPJUDGE_MODEL_REPO` | `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | HF repo for the first-run download. |
| `LLAMA_PARALLEL` | `4` | `llama-server --parallel`. |
| `LLAMA_SLOT_CTX` | `8192` | Context per slot. |

Compose passes only `WPJUDGE_MODEL_FILE` and `LLAMA_PARALLEL` to the `llama` service. Add `WPJUDGE_MODEL_REPO` or `LLAMA_SLOT_CTX` to the `llama` service `environment`.

`WPJUDGE_RETENTION_DAYS`, `WPJUDGE_CAP_VALUE` and `WPJUDGE_COVERAGE_THRESHOLD` take whole numbers.

The API reads its config once, at start. Restart it after a change.

## API reference

Async jobs. One worker thread runs one job at a time. Results are stored on disk, one directory per job.

**Auth.** If `WPJUDGE_API_KEY` is set, every `/jobs` endpoint needs `Authorization: Bearer <key>`. Otherwise it returns 401. FastAPI's `/docs` and `/openapi.json` stay open.

### `POST /jobs`

Multipart form. Field `file`: the archive (name ends in `.zip`, `.tar.gz` or `.tgz`, any case). Optional field `webhook_url`.

```bash
curl -H "Authorization: Bearer $KEY" -F file=@my-plugin.zip \
  -F webhook_url=http://host:9000/hook http://127.0.0.1:8000/jobs
# 202 {"job_id": "<32 hex>"}
```

| Status | When |
|---|---|
| 202 | Accepted. Body: `{"job_id": ...}`. |
| 400 | `Content-Length` is not a number, or `webhook_url` is not an `http(s)` URL. |
| 401 | Bad or missing key (auth on). |
| 411 | No `Content-Length` (chunked upload). |
| 413 | `Content-Length` or the streamed body is over `WPJUDGE_MAX_UPLOAD_MB`. |
| 415 | File name has the wrong suffix. |
| 422 | No `file` field. |

The size check runs on every POST, before auth. Archive safety (paths, links, size, ratio) is checked when the job runs. A bad archive gives a `failed` job, not an HTTP error.

### `GET /jobs/{id}`

```json
{"id": "...", "status": "queued|running|done|failed", "progress": {"done": 3, "total": 39}, "error": null}
```

- `progress.total` is `null` until extraction ends.
- `status: "done"` adds `"report": {...}`.
- `status: "failed"` sets `error`, e.g. `"Rejected: unsafe path: ../x.php"`.
- 404: unknown id, or an id that is not 32 lowercase hex characters. 401: auth.

### `GET /jobs/{id}/functions`

Partial results while the job runs, and all rows after it ends: `{"functions": [...]}`.

- Rows arrive in completion order. Each row is a judgment: `id, file, line, end_line, name, kind, loc, truncated, status, attempts, verdict, overall_score, scores, critique, error`.
- Skip rows appear only in the final report.
- 404 and 401 as above.

### Webhook

When a job ends `done` and has `webhook_url`, the API POSTs the report as JSON to that URL. Timeout 5 s, one retry after 1 s, then it gives up silently. No webhook for a `failed` job.

### Behaviour

- **Queue.** Jobs run one at a time, in order. One large job blocks the jobs behind it. v1 has no cancel endpoint.
- **Restart.**
  - `done` and `failed` jobs survive. Their results are on disk.
  - `queued` jobs run again. Their upload is still on disk.
  - A `running` job becomes `failed` with `error: "interrupted by server restart; resubmit"`. Submit it again.
- **Retention.** A job directory is deleted when its last change is older than `WPJUDGE_RETENTION_DAYS`. The sweep runs at start, after every job, and every hour when idle. The upload is deleted when its job ends.
- **Webhook URLs** are not filtered by host. The API can reach anything its network can reach.

## Report (schema 1.0)

Top level: `schema_version`, `target`, `model`, `summary`, `dimensions`, `anti_patterns`, `patterns`, `worst_functions`, `functions`.

| Field | Meaning |
|---|---|
| `target` | `name`, `type` (`theme` if `style.css` has `Theme Name:`, else `plugin`), `files_scanned` (`.php` files walked, skipped included), `sha256` (archive; `""` for a directory). |
| `model` | `file`, `base_url`. |
| `summary.overall_score` | LOC-weighted mean of judged units' `overall_score`, 1 decimal place. Capped at `cap_value` (60) if any judged unit has `security` or `sql_safety` ≤ 4. `null` if nothing was judged. |
| `summary.overall_score_uncapped` | The mean before the cap. |
| `summary.capped`, `cap_reasons` | Cap flag and `[{id, dimension, score}]` for every triggering unit. |
| `summary.verdict_counts` | Model verdicts of judged units, unchanged. Always has `PASS` and `FAIL`. |
| `summary.coverage` | `units` = judged + unjudged; `judged`, `unjudged`, `skipped` (skip records), `pct` = judged / units × 100, `low_coverage` = pct < 90 or nothing judged. |
| `dimensions.<key>` | Per native dimension key: `mean`, `std` (population), `min`, `n_rated`, `low_rate` (share ≤ 4). |
| `anti_patterns` | Top 10 `{dimension, finding, count, examples}`. A finding is a topic from the critique line of a dimension scored ≤ 4. Counted once per function. One example, `"file:line name"`. |
| `patterns` | Top 10 `{dimension, finding, count}` for dimensions scored ≥ 8. |
| `worst_functions` | The 10 lowest judged units: `file, line, name, overall_score, verdict`. Ties break by id. |
| `functions` | Every unit, then every skip. |

Function rows:

- Judged or unjudged: `id` (`file:line:name`), `file`, `line`, `end_line`, `name` (`Class::method`, `function` or `<template>`), `kind` (`function|method|template`), `loc`, `truncated`, `status`, `attempts`, `verdict`, `overall_score`, `scores`, `critique`, `error`.
- Skipped: `file`, `line`, `name`, `loc: 0`, `truncated: false`, `status: "skipped"`, `reason`, `verdict: null`, `overall_score: null`, `scores: {}`, `critique: ""`. Per-file skips have `line: 0, name: ""`.

Scoring rules:

- The verdict is the model's own. The harness never recomputes it.
- Scores come only from the JSON block (normally `<judge_output>`), never from the prose lines.
- Each unit is sent once and retried once. If both fail (request error, or no JSON with a numeric `overall_score`), the unit is `unjudged`, with `error` set. It is excluded from the score.
- Dimension keys are the model's own (`wpcs_compliance`, `sql_safety`, `security`, `performance`, `wp_api_usage`, `code_quality`, `dependency_integrity`, `i18n`, `accessibility`, ...).

## Limits

| What | Rule |
|---|---|
| Archives | No absolute or `..` paths, no symlinks or hardlinks, ≤ 20000 members, ≤ 200 MiB uncompressed, ratio ≤ 100× (above 10 MiB). Tar uses `filter="data"`. A single top-level folder becomes the root. `__MACOSX/` is removed. |
| Upload (API) | ≤ 50 MiB, `Content-Length` required. |
| Skipped dirs | `vendor/` (vendor), `node_modules/` (node_modules), `tests/`, `test/`, `bin/`, `docs/` (test), `build/`, `dist/` (build). |
| Skipped files | `test-*`, `*-test.*`, `*Test.php`, `*Tests.php`, classes extending `WP_UnitTestCase`/`TestCase` (test); `*.asset.php` (build); `*.min.php` or average line > 200 chars in files > 2000 chars (minified); symlinked files (not read, not counted). |
| Skipped functions | Under 3 lines (tiny); syntax errors (parse_error). |
| Templates | A file with no functions or classes but top-level PHP is one unit. Under 3 lines: tiny. Over `MAX_CHARS`: too_large. |
| Truncation | Functions over 15000 chars are cut and marked `truncated: true`. |

## Known small-model caveats

- **Templates are out of distribution.** The judge was trained on functions. On twentytwentyfour, 57 template units averaged 81.6 and 3 functions averaged 84.0. Set `WPJUDGE_TEMPLATES=0` to judge functions only. The default is on.
- **The cap often hangs on scores of exactly 4.** On the test fixture, 6 of 8 cap reasons were exactly at the threshold. A ±1 change in the model output can turn the cap on or off. Read `cap_reasons` before you act on `capped`.
- **Runaway generations.** The model sometimes generates until `max_tokens` with no `<judge_output>`. The harness retries once. If the retry also runs away and no JSON block parses, the unit is unjudged and coverage drops.
- **Cross-build drift.** The same GGUF on two llama.cpp builds gave different scores. Host build 9954 vs container b11515, 16 prompts: overall |Δ| max 8, mean 1.5; host ≥ container on all 16; verdicts 16/16 equal. Compare reports only from the same server build.
- **Prose and scores can disagree.** The critique prose can name a different score from `<judge_output>`, or discuss a dimension that has no score. Example: in classic-editor `add_redirect_helper` the prose says "Security 4/10" but the stored score is 5. The harness uses only the `<judge_output>` scores for the cap and all statistics.
- **False positives.** On classic-editor the judge flagged a CSRF risk on a hidden empty form field. That is likely wrong. Treat findings as review leads, not audit results.
- **Generic pattern labels.** The keyword lexicon (`TOPICS` in `aggregate.py`) misses many phrasings, so some patterns show `finding: "other"`.
- **Parallel slots.** With 4 workers, output is not byte-deterministic. Use `--workers 1` for a reproducible report.

## Measured

Host build 9954, IQ4_XS, `--parallel 4`, CLI with 4 workers on the real targets. The shipped container uses b11515; scores there can differ (see above).

| Target | Units | Wall | Overall | Capped |
|---|---|---|---|---|
| Test fixture (`tests/fixtures/vuln-plugin`, `--workers 1`) | 10 (+5 skipped) | n/a | 60 (uncapped 68.8) | yes, 8 reasons |
| classic-editor (plugin) | 39 | 221 s | 74.9 | no |
| twentytwentyfour (theme) | 60 | 337 s | 81.7 | no |

Throughput: about 11 units per minute. On b11515 (compose and all-in-one), the fixture gave 60 capped, uncapped 67.7 and 69.7, with the same per-unit verdicts.

## Tests

```bash
harness/.venv/bin/pip install -e 'harness[test]'
harness/.venv/bin/pytest harness/tests          # unit tests, no server
harness/tests/e2e.sh all                        # live suite: GPU, host llama-server, Docker
```

`e2e.sh` uses `harness/.venv`. See its header for sections and env knobs.
