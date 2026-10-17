# wpjudge: WP Judge harness

`wpjudge` checks the PHP code in a WordPress plugin or theme. It sends each function to the WP Judge model. The model gives each function a verdict (PASS or FAIL) and scores out of 10 for security, SQL safety, coding standards and other areas. `wpjudge` then writes one JSON report for the whole plugin or theme.

Use the results as review leads. They are not an audit. Read the "Known limits" section before you act on a result.

You can use `wpjudge` in three ways:

- a command line tool (`wpjudge scan`);
- a Python library (`from wpjudge import scan`);
- a web API that runs in a container and handles jobs in the background.

## What you need

- Python 3.10 or newer.
- A model server. This is normally `llama-server` from llama.cpp, with the WP Judge GGUF file. Any server that speaks the OpenAI chat API also works.
- An NVIDIA GPU for the model server. The default model file is 15.44 GiB.
- For the container options: Docker and the NVIDIA container toolkit.

You do not need PHP. `wpjudge` reads PHP with its own parser.

## Quick start

### Step 1. Install

```bash
python3 -m venv harness/.venv
harness/.venv/bin/pip install -e harness
```

This gives you the `wpjudge` command. To run the tests too, install `'harness[test]'`.

### Step 2. Start the model server

Download the model file. Then start `llama-server` on port 8080. `wpjudge` uses this address by default.

```bash
hf download iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf \
  wp-judge-v4-pruned-k224.IQ4_XS.gguf --local-dir ./judge
llama-server -m ./judge/wp-judge-v4-pruned-k224.IQ4_XS.gguf --host 127.0.0.1 --port 8080 \
  -ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}' \
  --parallel 1 -c 8192
```

- Keep `--parallel 1`. Then the same input always gives the same report. See "Same input, same report".
- Set `-c` to `--parallel` × 8192.
- With the small model file (`wp-judge-v4-pruned-k224.small-IQ3_XXS-experts.gguf`), do not add `-ctk q8_0` or `-ctv q8_0`.

### Step 3. Scan a plugin or theme

```bash
wpjudge scan ./my-plugin -o report.json
```

The input can be a folder, a `.zip` file or a `.tar.gz` file. An archive is unpacked into a temporary folder, which is deleted after the scan. While it runs, the tool shows the number of functions and an estimated time, for example `39 units, ETA ~7.1 min`. It judges about 5.5 functions per minute.

### Step 4. Read the report

Open `report.json` and look at `summary` first:

| Field | What it tells you |
|---|---|
| `overall_score` | The score for the whole plugin or theme, 0 to 100. Large functions count more than small ones. |
| `capped` | `true` if at least one function scored 4 or less for `security` or `sql_safety`. Then `overall_score` is limited to 60. |
| `cap_reasons` | The functions that caused the cap. Look at these first. |
| `overall_score_uncapped` | The score without the cap. |
| `coverage` | How many functions the model judged. If `low_coverage` is `true`, less than 90% were judged, and the score is less reliable. |
| `low_confidence` | How many functions have a doubtful result. Check those by hand. |

Then look at:

- `worst_functions`: the 10 lowest-scored functions.
- `anti_patterns`: problems that occur again and again.
- `patterns`: things the code does well.
- `functions`: the full result for every function, with the model's written critique.

## Other ways to run

### Python library

```python
from wpjudge import scan

report = scan("my-plugin.zip", workers=1, base_url="http://127.0.0.1:8080/v1")
```

- `scan(path, on_result=None, **settings)` returns the report as a dict.
- Settings use the lowercase names from the "Settings" table, for example `cap_threshold=4`. An unknown name raises `TypeError`. A value of `None` is ignored.
- Order of priority: keyword arguments, then `WPJUDGE_*` environment variables, then defaults.
- `on_result(judgment, done, total)` is called once with `(None, 0, total)` after the functions are found. Then it is called once for each finished function.
- Bad input raises `ValueError`.

### Docker compose (API and model server)

Run from `harness/`:

```bash
cd harness
docker compose up -d
```

This starts two containers:

- `llama`: the model server, `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515`. It keeps the model file in the `models` volume.
- `harness`: the API, on `127.0.0.1:8000`. It starts when the model server is ready. It keeps jobs in the `jobs` volume.

On the first start, the model server downloads the model file (15.44 GiB). This can take up to 40 minutes. See "First start".

Compose passes these variables through: `WPJUDGE_MODEL_FILE`, `LLAMA_PARALLEL` (default 1) and `WPJUDGE_API_KEY` (default empty, which turns auth off). Compose also sets `WPJUDGE_WORKERS` to the value of `LLAMA_PARALLEL`, so the two always match. To set another `WPJUDGE_*` setting, add it to the `environment` of the `harness` service.

### All-in-one image

One container runs both the model server and the API. You mount the model folder at `/models`.

```bash
cd harness
docker build --target allinone -t wpjudge:allinone .
docker run -d --gpus all -v "$PWD/models:/models" -v wpjudge-jobs:/data/jobs \
  -p 127.0.0.1:8000:8000 wpjudge:allinone
```

- The API starts when the model server is ready. If the model server stops, the container stops.
- In this image, `LLAMA_PARALLEL` does not change `WPJUDGE_WORKERS`. If you change one, change the other too.

### First start

Both images use `docker/llama-entrypoint.sh`. If `/models/$WPJUDGE_MODEL_FILE` does not exist, the script downloads it from `https://huggingface.co/$WPJUDGE_MODEL_REPO/resolve/main/$WPJUDGE_MODEL_FILE`.

- It needs no token. If the download fails, it tries again up to 5 times, 10 seconds apart. If the download stops, it continues from `<file>.part`.
- The file name must not be empty, contain `/` or start with `..`. Otherwise the script stops with exit code 2.
- Then it starts `llama-server` with thinking off. The server handles `LLAMA_PARALLEL` requests at the same time, with `LLAMA_SLOT_CTX` tokens each. It uses the default f16 KV cache.
- In compose, extra `command:` arguments go to `llama-server`. The all-in-one image ignores them.

### API without Docker

```bash
WPJUDGE_API_KEY=secret wpjudge serve --host 127.0.0.1 --port 8000
```

Run one copy only. The job queue is in memory. Jobs are saved in `./jobs` unless you set `WPJUDGE_JOBS_DIR`.

## Command line reference

```bash
wpjudge scan PATH [-o report.json] [--workers N] [--base-url URL] [--model-file NAME]
```

| Option | Default | What it does |
|---|---|---|
| `PATH` | required | Folder or archive. The tool finds the archive type from the file content: zip, tar or tar.gz. |
| `-o`, `--output` | standard output | Where to write the report. Without it, the report goes to the screen, and you can pipe it. |
| `--workers` | `WPJUDGE_WORKERS` (1) | How many functions to send at the same time. More than 1 needs a server started with `--parallel` more than 1. It is faster, but the report can change between runs. |
| `--base-url` | `WPJUDGE_BASE_URL` | Address of the model server. |
| `--model-file` | `WPJUDGE_MODEL_FILE` | Model name. It is written into the report. `llama-server` ignores it. |

A flag you do not give takes its value from the environment variable, and then from the default in "Settings".

Exit codes:

- `0`: the report was written. Some functions can still be unjudged, for example if the server stopped. Check `summary.coverage`.
- `2`: bad input, for example a missing path or an unsafe archive. The reason is shown on the screen.

If the report is not reproducible, the tool shows one warning, for example `wpjudge: WARNING: workers=4 on a server with 4 slots decodes concurrently; ...`. It still writes the report.

## Same input, same report

On one server, the same input gives exactly the same report every time, if the server works on one request at a time. This is the default.

- When the server works on two or more requests at the same time, its calculations run in a different order. This can change a score. In one test, 4 at the same time changed 5 of 10 scores, and one score moved from 14 to 42.
- Defaults: `WPJUDGE_WORKERS` 1 and `LLAMA_PARALLEL` 1.
- Every request tells the server not to reuse saved results from earlier requests (`cache_prompt: false`). Saved results from a run with many requests at once can change a score. This costs no measurable time.
- The report shows this in `model`: `workers`, `server_slots` (from the server's `/props`; `null` if unknown) and `reproducible`.
- `reproducible` is `true` when `workers` is 1 or `server_slots` is 1.
- Faster mode: start the server with `--parallel 4 -c 32768` and use `--workers 4`. It is about 1.9 times faster. The report is then not reproducible.
- Other programs that use the same server can also change results. Use a server for `wpjudge` only.
- Two different llama.cpp versions can give different scores for the same input. Compare reports only from the same server version.
- If a server rejects `cache_prompt` or `chat_template_kwargs`, `wpjudge` stops sending both and shows one warning.
- The same report every time does not mean a correct report. See "Known limits".

## Settings

Set these as environment variables. Yes/no settings: `0`, `false`, `no`, `off` or empty mean no. Lists: separate the items with commas.

| Variable | Default | What it does |
|---|---|---|
| `WPJUDGE_BASE_URL` | `http://127.0.0.1:8080/v1` | Address of the model server. Compose sets `http://llama:8080/v1`. |
| `WPJUDGE_MODEL_FILE` | `wp-judge-v4-pruned-k224.IQ4_XS.gguf` | Model name in requests and in the report. The Docker images also use it as the file to download and load. |
| `WPJUDGE_MAX_TOKENS` | `2048` | Maximum length of one model answer. |
| `WPJUDGE_WORKERS` | `1` | Functions sent at the same time. See "Same input, same report". |
| `WPJUDGE_CAP_DIMS` | `security,sql_safety` | Score areas that can cause the cap. |
| `WPJUDGE_CAP_THRESHOLD` | `4` | A cap area scored at or below this causes the cap. It also sets the limit for `low_rate` and anti-patterns. Keep 4: most real bugs on the test plugin scored exactly 4. |
| `WPJUDGE_PATTERN_MIN` | `8` | A score at or above this counts as a good pattern. |
| `WPJUDGE_CAP_VALUE` | `60` | Highest overall score when capped. Whole numbers only. |
| `WPJUDGE_COVERAGE_THRESHOLD` | `90` | Coverage % below this sets `low_coverage`. Whole numbers only. |
| `WPJUDGE_MIN_VERDICT_P` | `0.9` | A verdict with a model certainty below this is marked low confidence. `0` turns this off. |
| `WPJUDGE_MAX_CHARS` | `15000` | Longer functions are cut to this length. Longer template files are skipped. |
| `WPJUDGE_TEMPLATES` | on | Judge PHP files that have no functions as one unit. |
| `WPJUDGE_MAX_FILES` | `20000` | Most files allowed in an archive. |
| `WPJUDGE_MAX_TOTAL_MB` | `200` | Largest unpacked archive size, in MiB. |
| `WPJUDGE_MAX_RATIO` | `100` | Largest allowed ratio of unpacked size to archive size. It applies only when the unpacked size is over 10 MiB. |
| `WPJUDGE_MAX_UPLOAD_MB` | `50` | Largest API upload, in MiB. |
| `WPJUDGE_JOBS_DIR` | `jobs` (images: `/data/jobs`) | Where the API saves jobs. |
| `WPJUDGE_RETENTION_DAYS` | `7` | The API deletes jobs older than this. Whole numbers only. |
| `WPJUDGE_API_KEY` | empty | Key for the API. Empty means no key is needed. |
| `WPJUDGE_WEBHOOK_ALLOW_NETS` | empty | Internal networks that a webhook may reach, for example `127.0.0.0/8`. Host bits are ignored. See "Webhook". |

Docker images only:

| Variable | Default | What it does |
|---|---|---|
| `WPJUDGE_MODEL_REPO` | `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` | Hugging Face repo for the first download. |
| `LLAMA_PARALLEL` | `1` | Requests the model server handles at the same time. More than 1 is faster but not reproducible. |
| `LLAMA_SLOT_CTX` | `8192` | Maximum tokens (prompt plus answer) for each request that the server handles at the same time. |

Compose passes only `WPJUDGE_MODEL_FILE` and `LLAMA_PARALLEL` to the `llama` service. To change `WPJUDGE_MODEL_REPO` or `LLAMA_SLOT_CTX`, add them to that service's `environment`.

The API reads its settings once, when it starts. Restart it after a change.

## API reference

You send an archive. The API gives you a job ID at once and does the work in the background. One job runs at a time, in the order received. Results are saved on disk.

**Key.** If `WPJUDGE_API_KEY` is set, every `/jobs` request needs the header `Authorization: Bearer <key>`. Without it, the API answers 401. The `/docs` and `/openapi.json` pages stay open.

### Send a job: `POST /jobs`

Send a form with the field `file` (a `.zip`, `.tar.gz` or `.tgz` file). You can add the field `webhook_url`.

```bash
curl -H "Authorization: Bearer $KEY" -F file=@my-plugin.zip \
  -F webhook_url=http://host:9000/hook http://127.0.0.1:8000/jobs
# 202 {"job_id": "<32 hex>"}
```

| Status | Meaning |
|---|---|
| 202 | Accepted. The answer contains `job_id`. |
| 400 | `Content-Length` is not a number, or `webhook_url` is not allowed (see "Webhook"). |
| 401 | Wrong or missing key. |
| 411 | No `Content-Length` header. |
| 413 | The upload is larger than `WPJUDGE_MAX_UPLOAD_MB`. |
| 415 | The file name does not end in `.zip`, `.tar.gz` or `.tgz`. |
| 422 | No `file` field. |

The API checks the upload size on every request, before it checks the key. So a request without `Content-Length` gets 411, not 401. The suffix check ignores upper and lower case.

The API checks the archive content when the job runs. A bad archive gives a `failed` job, not an HTTP error.

### Check a job: `GET /jobs/{id}`

```json
{"id": "...", "status": "queued|running|done|failed", "progress": {"done": 3, "total": 39}, "error": null}
```

- `progress.total` is `null` until the functions are found.
- When `status` is `done`, the answer also contains `report`.
- When `status` is `failed`, `error` says why, for example `"Rejected: unsafe path: ../x.php"`.
- 404: no job has this ID.

### See results so far: `GET /jobs/{id}/functions`

This gives `{"functions": [...]}`: the functions judged so far, in the order they finished. The `low_confidence` flags and skipped files appear only in the final report.

### Webhook

When a job is `done`, the API sends the report as JSON to `webhook_url`. It waits up to 5 seconds for each step of the connection. If the connection fails, it tries once more after 1 second. It does not retry an HTTP error answer. It sends nothing for a `failed` job.

The API refuses webhook addresses inside your network. This stops a user from using the webhook to reach internal systems.

- It looks up the host name when you send the job and again before it sends the report. Every address must be public.
- Refused: this machine (loopback, 127.x), private networks (10.x, 172.16-31.x, 192.168.x), cloud metadata addresses (169.254.x), Docker service names such as `http://llama:8080`, and other non-public addresses. Other ways to write these addresses, such as `2130706433` or `[::ffff:127.0.0.1]`, are also refused.
- A refused address at submit time gives 400, and no job is made. A refused address at send time is dropped.
- The API connects to the address it checked, and it does not follow redirects.
- The API ignores `HTTP_PROXY` and `HTTPS_PROXY`.
- To allow an internal target, set `WPJUDGE_WEBHOOK_ALLOW_NETS`, for example `127.0.0.0/8`. All other internal addresses stay refused.
- Known issues: the 400 message shows the address the name resolved to. A slow DNS server can delay the request.

### Queue, restart and clean-up

- One large job delays the jobs after it. There is no cancel.
- After a restart:
  - `done` and `failed` jobs stay.
  - `queued` jobs run again.
  - A `running` job becomes `failed`, with `error: "interrupted by server restart; resubmit"`. Send it again.
- The API deletes a job folder when its last change is older than `WPJUDGE_RETENTION_DAYS`. It checks at start, after every job, and every hour when idle. It deletes the upload when the job ends.

## What gets judged

`wpjudge` finds every PHP function and method. It also judges a PHP file with no functions as one "template" unit. It skips code that is not part of the product:

| Skipped | Rule |
|---|---|
| Folders | `vendor/`, `node_modules/`, `tests/`, `test/`, `bin/`, `docs/`, `build/`, `dist/` |
| Test files | `test-*`, `*-test.*`, `*Test.php`, `*Tests.php`, and classes that extend `WP_UnitTestCase` or `TestCase` |
| Generated files | `*.asset.php`, `*.min.php`, and long files with very long lines (minified) |
| Links | Symbolic links are not read. |
| Small or broken code | Functions under 3 lines, and code with syntax errors |
| Large templates | Template files longer than `WPJUDGE_MAX_CHARS` |

Functions longer than `WPJUDGE_MAX_CHARS` are cut and marked `truncated: true`.

Archive safety: `wpjudge` refuses archives with absolute paths, `..` paths, symbolic or hard links, more than 20000 files, more than 200 MiB unpacked, or an unpacked size more than 100 times the archive size (checked only above 10 MiB unpacked). If the archive has one top folder, that folder is the root. `__MACOSX/` is removed.

## How scoring works

- The verdict (PASS or FAIL) is the model's own. `wpjudge` does not change it.
- Scores come only from the JSON block in the model answer (`<judge_output>`).
- The model's written critique sometimes names a different score from the JSON block. `wpjudge` rewrites the score headings in `critique` to match the JSON. It keeps the old numbers in `prose_scores`.
- If the model's answer cannot be read, `wpjudge` asks again once. The first request has no randomness.
- The second request adds a little randomness (temperature 0.5) with a fixed seed (1234). On a server that works on one request at a time, it still gives the same result every run.
- A function judged on the second request is marked `retry`.
- If the second answer also cannot be read, the function is "unjudged" and does not count in the score.
- `overall_score` is the average of the function scores. Each function counts in proportion to its number of lines.
- Score area names come from the model, for example `security`, `sql_safety`, `wpcs_compliance`, `performance`, `wp_api_usage`, `code_quality`, `i18n`, `accessibility`.
- `patterns` and `anti_patterns` take topics from the critique lines, using a list of 27 topics. A line with no known topic counts as `other`.

### Low confidence

Some results are less certain. The report marks them so you can check them by hand. These marks never change a score, a verdict or the cap.

`confidence_reasons` can contain:

- `verdict_p`: the model was less than 90% certain of its verdict (`WPJUDGE_MIN_VERDICT_P`).
- `prose_disagrees`: for `security` or `sql_safety`, one of the two scores (written critique or JSON) is 4 or less and the other is above 4 (`WPJUDGE_CAP_THRESHOLD`).
- `retry`: the function was judged on the second try.

`low_confidence` is `true` when there is at least one reason. `prose_disagrees` is not precise: it marks about 25% of functions even on a clean plugin.

## Known limits

- **False positives.** The model can report a problem that is not there. Example: classic-editor is capped at 60 (73.9 without the cap). The model gives security 4 to `Classic_Editor::add_edit_php_inline_style`, which only prints a fixed `<style>` block. The model confuses two different things that are both called a nonce: the WordPress check against forged requests, and an HTML attribute on style tags. Automated scanners often make the same mistake. The model was 92% certain, so no low-confidence mark appears. This result repeats on every run.
- **False negatives.** The model misses some real problems. It is too lenient in some cases. A second try does not fix this.
- **Scores at the limit.** Most caps come from a score of exactly 4. A change of 1 point turns the cap on or off. Read `cap_reasons` before you act on `capped`.
- **Templates.** The model was trained on functions, not template files. To judge functions only, set `WPJUDGE_TEMPLATES=0`.
- **Answers that repeat.** Sometimes the model repeats one line until it reaches the length limit. The second try fixed all 3 known cases. If it fails, the function is unjudged and coverage drops.
- **Server versions.** The host server (llama.cpp build 9954) and the container server (b11515) gave scores up to 8 points apart on 16 test inputs. All 16 verdicts were the same.
- **Topic labels.** The `docblock` topic also matches any "comment" in the critique. The filter for "does not apply" sentences sometimes drops a real problem sentence. Scores and the cap are not affected.
- **Model certainty.** Every request asks the server for token probabilities. This costs about 6% more time. A server that refuses this request leaves every function unjudged.
- **No second opinion mode.** Asking the model several times did not find more errors in tests.

## Report reference (schema 1.0)

Top-level fields: `schema_version`, `target`, `model`, `summary`, `dimensions`, `anti_patterns`, `patterns`, `worst_functions`, `functions`.

| Field | Content |
|---|---|
| `target` | `name`; `type` (`theme` if `style.css` has `Theme Name:`, otherwise `plugin`); `files_scanned` (`.php` files found, skipped files included); `sha256` of the archive (`""` for a folder). |
| `model` | `file`, `base_url`, `workers`, `server_slots`, `reproducible`. |
| `summary.overall_score` | Line-weighted average of judged functions, 1 decimal place. Limited to `WPJUDGE_CAP_VALUE` when capped. `null` if nothing was judged. |
| `summary.overall_score_uncapped` | The average before the cap. |
| `summary.capped`, `summary.cap_reasons` | The cap flag, and `[{id, dimension, score}]` for each function that caused it. |
| `summary.verdict_counts` | Counts of PASS and FAIL. |
| `summary.coverage` | `units` (judged + unjudged), `judged`, `unjudged`, `skipped`, `pct` (judged ÷ units × 100), `low_coverage`. |
| `summary.low_confidence` | Number of functions marked low confidence. |
| `dimensions.<area>` | For each score area: `mean`, `std`, `min`, `n_rated`, `low_rate` (share scored at or below `WPJUDGE_CAP_THRESHOLD`). |
| `anti_patterns` | Top 10 `{dimension, finding, count, examples}` from areas scored at or below `WPJUDGE_CAP_THRESHOLD`. One example, as `"file:line name"`. |
| `patterns` | Top 10 `{dimension, finding, count}` from areas scored at or above `WPJUDGE_PATTERN_MIN`. |
| `worst_functions` | The 10 lowest-scored functions: `file`, `line`, `name`, `overall_score`, `verdict`. |
| `functions` | Every function, then every skipped item. |

Fields of a judged or unjudged function: `id` (`file:line:name`), `file`, `line`, `end_line`, `name` (`Class::method`, `function` or `<template>`), `kind` (`function`, `method` or `template`), `loc`, `truncated`, `status`, `attempts`, `verdict`, `overall_score`, `scores`, `critique`, `error`, `verdict_p`, `prose_scores`, `confidence_reasons`, `low_confidence`.

Fields of a skipped item: `file`, `line`, `name`, `loc: 0`, `truncated: false`, `status: "skipped"`, `reason`, `verdict: null`, `overall_score: null`, `scores: {}`, `critique: ""`, `verdict_p: null`, `prose_scores: {}`, `confidence_reasons: []`, `low_confidence: false`. A skipped file has `line: 0` and `name: ""`.

## Measured results

Host llama.cpp build 9954, IQ4_XS model, `--parallel 1`, 1 worker.

| Target | Functions | Time | Overall | Capped |
|---|---|---|---|---|
| Test plugin (`tests/fixtures/vuln-plugin`, 5 planted bugs) | 10 (+5 skipped) | about 110 s | 60 (68.8 without cap) | yes: the 5 planted bugs, no clean function |
| classic-editor (plugin) | 39 | 426 s | 60 (73.9 without cap) | yes: 1 false positive |
| twentytwentyfour (theme) | 60 | 620 s | 81.2 | no |
| twentytwelve 4.9 (theme) | 54 | 675 s | 78.3 | no |

- Speed: about 5.5 functions per minute.
- On the container server (b11515, compose), the 5 planted bugs are also capped, and no clean function is capped. The all-in-one image was not tested again after the second-request change.
- twentytwelve was measured before the second-try change. One template (`search.php`) was unjudged then. The second try now judges it.
- Low confidence: test plugin 3 of 10, classic-editor 9 of 39, twentytwentyfour 1 of 60, twentytwelve 7 of 53.

## Tests

```bash
harness/.venv/bin/pip install -e 'harness[test]'
harness/.venv/bin/pytest harness/tests          # unit tests, no server needed
harness/tests/e2e.sh all                        # live tests: GPU, host llama-server, Docker
```

`e2e.sh` uses `harness/.venv`. Its header lists the sections and settings.
