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
  --parallel 1 -c 8192
```

- `-c` is `--parallel` × 8192.
- Use `--parallel 1` for a reproducible report.
- Fast mode: `--parallel 4 -c 32768` with `--workers 4`. It is about 1.9× faster, but scores and `cap_reasons` can change between runs (see Reproducibility).
- Small file (`wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf`): do not add `-ctk q8_0` / `-ctv q8_0`.

### 2. Scan (CLI)

```bash
wpjudge scan ./my-plugin -o report.json
wpjudge scan my-plugin.zip -o report.json
wpjudge scan my-theme.tar.gz --base-url http://127.0.0.1:8080/v1
```

| Flag | Default | Effect |
|---|---|---|
| `path` | required | Directory or archive. Archives are detected by content: zip, tar, tar.gz/tgz. |
| `-o`, `--output` | stdout | Report path. |
| `--workers` | `WPJUDGE_WORKERS` (1) | Concurrent requests. More than 1 needs a server with `--parallel` > 1. |
| `--base-url` | `WPJUDGE_BASE_URL` | OpenAI-compatible base URL. |
| `--model-file` | `WPJUDGE_MODEL_FILE` | Model name sent to the server and written to `report.model.file`. `llama-server` ignores it. |

- Stderr shows the unit count and an ETA: `39 units, ETA ~7.1 min` (units ÷ 5.5 per min, the serial rate).
- Archives extract into a temporary directory. It is removed after the scan.
- Exit 0: report written. Unjudged units (server down, bad output) do not change the exit code. Check `summary.coverage`.
- Exit 2: bad input (missing path, not an archive, unsafe archive). The reason goes to stderr.
- When the report is not reproducible (`model.reproducible: false`), stderr shows one warning: `wpjudge: WARNING: workers=4 on a server with 4 slots decodes concurrently; ...`. The report is still written.

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
- Compose variables: `WPJUDGE_MODEL_FILE` (both services), `LLAMA_PARALLEL` (default 1; also sets `WPJUDGE_WORKERS`), `WPJUDGE_API_KEY` (default empty = auth off).
- Other `WPJUDGE_*` knobs are not passed through. Add them to the `harness` service `environment`. Example: `WPJUDGE_WEBHOOK_ALLOW_NETS`.

### 5. All-in-one image

One container runs `llama-server` and the API. The GGUF is mounted at `/models`, never baked in.

```bash
cd harness
docker build --target allinone -t wpjudge:allinone .
docker run -d --gpus all -v "$PWD/models:/models" -v wpjudge-jobs:/data/jobs \
  -p 127.0.0.1:8000:8000 wpjudge:allinone
```

- The API starts after `llama-server` answers `/health`. If `llama-server` exits first, the container exits.
- `LLAMA_PARALLEL` here does not change `WPJUDGE_WORKERS`. Both default to 1. Set both if you change one.
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

## Reproducibility

llama.cpp CUDA kernels are not batch-invariant. When the server decodes two or more requests at the same time, the reduction order changes, and so can the chosen token. Serial decoding is byte-identical across runs.

- The defaults are serial: `WPJUDGE_WORKERS` 1 and `LLAMA_PARALLEL` 1.
- Every request sends `cache_prompt: false`. Without it, a serial request can reuse a slot's prompt KV that a concurrent run computed, and its scores move. No throughput cost was measured: each unit prompt is distinct. If a server rejects `cache_prompt` or `chat_template_kwargs`, the harness drops both for the rest of the process and prints one warning.
- The report `model` block records `workers`, `server_slots` (from `GET /props` `total_slots`, `null` if the probe fails) and `reproducible`.
- `reproducible` is `workers == 1 or server_slots == 1`.
- A report is reproducible only on a 1-slot server, or with `--workers 1` (which sends `cache_prompt: false` unless the server rejected it).
- Concurrent traffic from other clients on a shared multi-slot server is not covered. Use a dedicated server.
- Reproducibility holds per llama.cpp build and per server config. Host and container builds still differ (see Cross-build drift).
- Fast mode cost: serial is about 46% slower (10.0 → 5.4 units/min). On the fixture, 4 workers on a `--parallel 4` server changed 5 of 10 units. One unit moved 14 → 42.
- Deterministic is not the same as accurate. See Known small-model caveats.

## Configuration

Every field in `wpjudge/config.py` reads `WPJUDGE_<FIELD>`. Booleans: `0`, `false`, `no`, `off` or empty mean false. Tuples: comma-separated.

| Env var | Default | Meaning |
|---|---|---|
| `WPJUDGE_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible server. Compose sets `http://llama:8080/v1`. |
| `WPJUDGE_MODEL_FILE` | `wp-judge-v4-pruned-k224.IQ4_XS.gguf` | Model name in requests and the report. The Docker entrypoint also uses it as the file to download and serve. |
| `WPJUDGE_MAX_TOKENS` | `2048` | Generation limit per request. |
| `WPJUDGE_WORKERS` | `1` | Concurrent requests. More than 1 needs server `--parallel` > 1 and makes the report not reproducible (see Reproducibility). |
| `WPJUDGE_CAP_DIMS` | `security,sql_safety` | Dimensions that trigger the cap. |
| `WPJUDGE_CAP_THRESHOLD` | `4` | A cap dimension scored at or below this triggers the cap. It is also the bound for `low_rate` and anti-patterns. Keep 4: most planted-bug cap reasons sit at exactly 4 (6 of 8 on the fixture), so 3 misses most of them. |
| `WPJUDGE_PATTERN_MIN` | `8` | A dimension scored at or above this counts as a pattern. |
| `WPJUDGE_CAP_VALUE` | `60` | Overall score ceiling when capped. |
| `WPJUDGE_COVERAGE_THRESHOLD` | `90` | Coverage % below this sets `low_coverage`. |
| `WPJUDGE_MIN_VERDICT_P` | `0.9` | A judged row with `verdict_p` below this gets the `verdict_p` confidence reason. `0` turns this check off. |
| `WPJUDGE_MAX_CHARS` | `15000` | Functions are truncated at this length. Larger templates are skipped (`too_large`). |
| `WPJUDGE_TEMPLATES` | `1` (on) | Judge function-less PHP files as one template unit. |
| `WPJUDGE_MAX_FILES` | `20000` | Archive member limit. |
| `WPJUDGE_MAX_TOTAL_MB` | `200` | Archive uncompressed size limit (MiB). |
| `WPJUDGE_MAX_RATIO` | `100` | Compression ratio limit. Applies above 10 MiB uncompressed. |
| `WPJUDGE_MAX_UPLOAD_MB` | `50` | API upload limit (MiB). |
| `WPJUDGE_JOBS_DIR` | `jobs` (images: `/data/jobs`) | API job store. |
| `WPJUDGE_RETENTION_DAYS` | `7` | Job directories older than this are deleted. |
| `WPJUDGE_API_KEY` | empty | Bearer key for the API. Empty means auth off. |
| `WPJUDGE_WEBHOOK_ALLOW_NETS` | empty | Comma-separated CIDRs that a webhook may reach even if they are not public, e.g. `127.0.0.0/8`. Host bits are ignored. |

Docker entrypoint only (not in `config.py`):

| Env var | Default | Meaning |
|---|---|---|
| `WPJUDGE_MODEL_REPO` | `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | HF repo for the first-run download. |
| `LLAMA_PARALLEL` | `1` | `llama-server --parallel`. More than 1 is faster but not reproducible. |
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
| 400 | `Content-Length` is not a number, or `webhook_url` is not an `http(s)` URL, does not resolve, or resolves to a blocked address. |
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

- Rows arrive in completion order. Each row is a judgment: `id, file, line, end_line, name, kind, loc, truncated, status, attempts, verdict, overall_score, scores, critique, error`. Judged rows also have `verdict_p` and `prose_scores`.
- `confidence_reasons` and `low_confidence` are set only in the final report.
- Skip rows appear only in the final report.
- 404 and 401 as above.

### Webhook

When a job ends `done` and has `webhook_url`, the API POSTs the report as JSON to that URL. Timeout 5 s, one retry after 1 s, then it gives up silently. No webhook for a `failed` job.

**SSRF guard.** The API resolves the webhook host at submit time and again at send time. Every resolved address must be public.

- Blocked: every address that is not global, plus multicast. This covers loopback, private ranges (10/8, 172.16/12, 192.168/16), link-local and cloud metadata (169.254.169.254), CGNAT, unspecified and reserved ranges, and the Docker networks (for example `http://llama:8080`).
- Encoded forms are unwrapped first: IPv4-mapped (`::ffff:a.b.c.d`), NAT64 (`64:ff9b::/96`), IPv4-compatible and SIIT addresses. Integer and short forms (`2130706433`, `127.1`) resolve to loopback and are blocked.
- A blocked URL at submit gives 400, and no job is created. A blocked URL at send time is dropped with no retry.
- The connection goes to the checked IP only, so DNS rebinding cannot redirect it. `Host`, TLS SNI and the certificate check still use the host name.
- Redirects are never followed.
- `HTTP_PROXY` and `HTTPS_PROXY` are no longer honoured. The API connects directly.
- To allow an internal target, set `WPJUDGE_WEBHOOK_ALLOW_NETS` (for example `127.0.0.0/8`). Addresses outside the list stay blocked, including metadata.
- Known limits: the 400 detail shows the resolved address, e.g. `(172.18.0.2)`, so a client can learn internal IPs. The DNS lookup has no timeout, so a slow DNS server can hold a request thread.

### Behaviour

- **Queue.** Jobs run one at a time, in order. One large job blocks the jobs behind it. v1 has no cancel endpoint.
- **Restart.**
  - `done` and `failed` jobs survive. Their results are on disk.
  - `queued` jobs run again. Their upload is still on disk.
  - A `running` job becomes `failed` with `error: "interrupted by server restart; resubmit"`. Submit it again.
- **Retention.** A job directory is deleted when its last change is older than `WPJUDGE_RETENTION_DAYS`. The sweep runs at start, after every job, and every hour when idle. The upload is deleted when its job ends.

## Report (schema 1.0)

Top level: `schema_version`, `target`, `model`, `summary`, `dimensions`, `anti_patterns`, `patterns`, `worst_functions`, `functions`.

| Field | Meaning |
|---|---|
| `target` | `name`, `type` (`theme` if `style.css` has `Theme Name:`, else `plugin`), `files_scanned` (`.php` files walked, skipped included), `sha256` (archive; `""` for a directory). |
| `model` | `file`, `base_url`, `workers`, `server_slots` (`/props` `total_slots`, or `null`), `reproducible` (`workers == 1 or server_slots == 1`). |
| `summary.overall_score` | LOC-weighted mean of judged units' `overall_score`, 1 decimal place. Capped at `cap_value` (60) if any judged unit has a `cap_dims` score (`security`, `sql_safety`) ≤ `cap_threshold` (4). `null` if nothing was judged. |
| `summary.overall_score_uncapped` | The mean before the cap. |
| `summary.capped`, `cap_reasons` | Cap flag and `[{id, dimension, score}]` for every triggering unit. |
| `summary.verdict_counts` | Model verdicts of judged units, unchanged. Always has `PASS` and `FAIL`. |
| `summary.coverage` | `units` = judged + unjudged; `judged`, `unjudged`, `skipped` (skip records), `pct` = judged / units × 100, `low_coverage` = pct < 90 or nothing judged. |
| `summary.low_confidence` | Count of rows with `low_confidence: true`. |
| `dimensions.<key>` | Per native dimension key: `mean`, `std` (population), `min`, `n_rated`, `low_rate` (share ≤ `cap_threshold`). |
| `anti_patterns` | Top 10 `{dimension, finding, count, examples}`. A finding is a topic from the critique line of a dimension scored ≤ `cap_threshold` (4). Counted once per function. One example, `"file:line name"`. |
| `patterns` | Top 10 `{dimension, finding, count}` for dimensions scored ≥ `pattern_min` (8). |
| `worst_functions` | The 10 lowest judged units: `file, line, name, overall_score, verdict`. Ties break by id. |
| `functions` | Every unit, then every skip. |

Function rows:

- Judged or unjudged: `id` (`file:line:name`), `file`, `line`, `end_line`, `name` (`Class::method`, `function` or `<template>`), `kind` (`function|method|template`), `loc`, `truncated`, `status`, `attempts`, `verdict`, `overall_score`, `scores`, `critique`, `error`, `verdict_p`, `prose_scores`, `confidence_reasons`, `low_confidence`.
- Skipped: `file`, `line`, `name`, `loc: 0`, `truncated: false`, `status: "skipped"`, `reason`, `verdict: null`, `overall_score: null`, `scores: {}`, `critique: ""`, `verdict_p: null`, `prose_scores: {}`, `confidence_reasons: []`, `low_confidence: false`. Per-file skips have `line: 0, name: ""`.

Scoring rules:

- The verdict is the model's own. The harness never recomputes it.
- Scores come only from the JSON block (normally `<judge_output>`), never from the prose lines.
- `critique` dimension headers are rewritten from the JSON: `Security: score 5/10 — ...`. A dimension with no JSON score shows `None/10`. The original prose numbers are kept in `prose_scores`. `raw` (dropped from reports) is not changed. A header without a `— reason` tail is not rewritten (none were seen in 1789 saved header lines).
- Anti-pattern and pattern topics come from a 27-topic lexicon (`TOPICS` in `aggregate.py`). A line is tagged with its own dimension's topics first. Only anti-pattern lines fall back to the whole lexicon. Sentences that say a dimension does not apply are dropped. A line with no topic counts as `other`.
- Each unit is sent once and retried once. The retry samples at temperature 0.5 with a fixed seed (the first try is greedy, and an identical greedy retry would repeat a runaway byte for byte), so it stays reproducible on a serial server. A unit judged on the retry gets `"retry"` in `confidence_reasons`. A loop that survives the retry still fails. If both fail (request error, or no JSON with a numeric `overall_score`), the unit is `unjudged`, with `error` set. It is excluded from the score.
- Dimension keys are the model's own (`wpcs_compliance`, `sql_safety`, `security`, `performance`, `wp_api_usage`, `code_quality`, `dependency_integrity`, `i18n`, `accessibility`, ...).

Confidence flags (review leads only; they never change a score, the verdict or the cap):

- `verdict_p`: the probability of the first verdict token inside `<judge_output>`, from the server's logprobs. `null` if the server returns no logprobs or there is no verdict.
- `confidence_reasons` can hold:
  - `verdict_p`: `verdict_p` is below `WPJUDGE_MIN_VERDICT_P` (0.9).
  - `prose_disagrees`: for a cap dimension, the prose score and the JSON score sit on different sides of `cap_threshold`. A missing or `None` score counts as above the threshold. The rule uses the config at aggregation time.
  - `retry`: the unit was judged on the varied retry.
- `low_confidence` is true when `confidence_reasons` is not empty.
- `prose_disagrees` is imprecise. It fires on about 25% of units on a presumed-clean plugin (classic-editor: `low_confidence` on 9 of 39).
- Every request sends `logprobs: true`. It costs about 6% per completion token on llama-server. The outputs do not change. A server that rejects `logprobs` outright makes every unit unjudged.
- There is no re-judge mode (n samples, a second pass, or a serial re-run). In tests they changed no errors at 14% to 400% more calls. The false negatives come from systematic model leniency, not from noise.

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
| Known false cap | classic-editor is capped at 60 (uncapped 73.9) by one false positive. See False positives below. Accepted for v1.1; a v1.2 training-data item. |

## Known small-model caveats

- **Templates are out of distribution.** The judge was trained on functions. On twentytwentyfour, 57 template units averaged 81.6 and 3 functions averaged 84.0. Set `WPJUDGE_TEMPLATES=0` to judge functions only. The default is on.
- **The cap often hangs on scores of exactly 4.** On the test fixture, 6 of 8 cap reasons were exactly at the threshold. A ±1 change in the model output can turn the cap on or off. Read `cap_reasons` before you act on `capped`.
- **Runaway generations.** The model sometimes repeats itself until `max_tokens` with no `<judge_output>`. The harness retries once with temperature 0.5 and seed 1234. This recovered all 3 known loops. If the retry also runs away, the unit is unjudged and coverage drops.
- **Cross-build drift.** The same GGUF on two llama.cpp builds gave different scores. Host build 9954 vs container b11515, 16 prompts: overall |Δ| max 8, mean 1.5; verdicts 16/16 equal. Compare reports only from the same server build.
- **Prose and scores can disagree.** The original critique prose can name a different score from `<judge_output>`. The prose numbers are stale severity buckets from the training data. The harness uses only the JSON scores, and it rewrites the critique headers to match. The old numbers stay in `prose_scores`, and `prose_disagrees` flags a disagreement across the cap threshold.
- **False positives.** classic-editor is capped by `Classic_Editor::add_edit_php_inline_style`: security 4 for "no nonce" on a function that only echoes a static `<style>` block. The model confuses the WordPress request nonce (CSRF, `check_admin_referer`) with the CSP `nonce=` attribute, like context-blind scanners do. The verdict is PASS with `verdict_p` 0.918, so no flag fires. Serial decoding makes this reproducible. Treat findings as review leads, not audit results.
- **False negatives.** The model is lenient on some real defects. This is systematic, so a retry or a second sample does not fix it.
- **Pattern labels.** The 27-topic lexicon left no `other` in the top-10 counts on classic-editor, twentytwentyfour and twentytwelve. Known errors: `docblock` also matches any "comment(s)", such as block-markup comments. The "does not apply" filter sometimes drops a real defect sentence from an anti-pattern line. Scores and the cap are not affected.

## Measured

v1.1, host llama.cpp build 9954, IQ4_XS, `--parallel 1`, CLI with 1 worker. The shipped container uses b11515; scores there can differ (see above).

| Target | Units | Wall | Overall | Capped |
|---|---|---|---|---|
| Test fixture (`tests/fixtures/vuln-plugin`) | 10 (+5 skipped) | about 110 s | 60 (uncapped 68.8) | yes, 8 reasons on the 5 planted bugs |
| classic-editor (plugin) | 39 | 426 s | 60 (uncapped 73.9) | yes, 1 false positive (see above) |
| twentytwentyfour (theme) | 60 | 620 s | 81.2 | no |
| twentytwelve 4.9 (theme) | 54 | 675 s | 78.3 | no |

- Throughput: about 5.5 units per minute, serial.
- The fixture per-unit scores are byte-identical to v1.
- On b11515 (compose, serial, with the varied retry), all 5 planted bugs are capped, and no clean unit is capped. The all-in-one section was not re-run after the retry change.
- twentytwelve was measured before the varied retry. Its `search.php` template was unjudged (53 of 54). The retry now judges it.
- `low_confidence`: fixture 3 of 10, classic-editor 9 of 39, twentytwentyfour 1 of 60, twentytwelve 7 of 53.

## Tests

```bash
harness/.venv/bin/pip install -e 'harness[test]'
harness/.venv/bin/pytest harness/tests          # unit tests, no server
harness/tests/e2e.sh all                        # live suite: GPU, host llama-server, Docker
```

`e2e.sh` uses `harness/.venv`. See its header for sections and env knobs.
