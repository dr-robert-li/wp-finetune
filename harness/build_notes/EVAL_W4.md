VERDICT: PASS

# W4 (api + cli + scan) evaluation

The workstream conforms to SPEC R1.2/R2.4, the r3 design B, the W4 locked plan and binding changes 4, 7, 8, 12, 14 and 16.
`scan()` calls the real W1, W2 and W3 signatures correctly:
- `load_target(path, tmp, cfg)` with a 3-key meta
- `collect_units(root, cfg)` returning a 3-tuple
- `judge_units(units, base_url, model, workers, on_result=, max_tokens=)` with a 1-arg callback, wrapped into `(j, done, total)`
- `build_report(..., files_scanned=)`

I found no blocker or major defects. All findings are minor.

## Defects

1. **minor, api.py:156-168 (create_job dest) + extract.py:87.** API reports of flat archives get `target.name` "upload". The CLI names the same archive from its file name, so the CLI and the API disagree for the same input. I reproduced it: a flat zip gives "flat" via the CLI and "upload" via the API. The impact is cosmetic. wordpress.org zips are always wrapped in a slug dir, so they are unaffected, and `sha256` still identifies the input. It is still worth fixing, because the API's report should match the CLI's.
   - **Minimal fix:** save the upload under a sanitised original name instead of the fixed one.
     ```python
     dest = d / re.sub(r"[^\w.-]", "_", Path(name).name)   # suffix already validated, so no clash with job.json/report.json
     ```
   - After this change, `upload_name` in job.json becomes redundant.
   - Keep the fixed-name file out of the report path. No traversal risk is introduced: `Path(name).name` plus the char filter means the name stays inside the job dir.
2. **minor, api.py:57-66 (webhook).** The webhook body is the bare report. The `job_id` is not in the body or in any header, and the client cannot know the id before it submits. So a receiver that handles several jobs cannot correlate a POST to its job, except by `sha256`.
   - **Fix:** one line. Add header `"X-Wpjudge-Job": jid`, and pass `jid` into `webhook()`. Alternatively, POST `{"job_id": jid, "report": report}`.
   - Document whichever you choose in the README (W7).
3. **minor, api.py:137-140 (Content-Length).** `cl.isdigit()` accepts Unicode digits such as "²", and `int()` then raises, which gives a 500. This cannot be reached in production: uvicorn/h11 itself rejects that header with 400 (verified with a raw socket). TestClient and httpx cannot send it either. A cleaner guard is `cl.isascii() and cl.isdigit()`. Optional.
4. **note (not a defect).** FastAPI parses the multipart body before it runs the `auth` dependency. An unauthenticated POST is therefore spooled by Starlette, up to the 50 MB header limit, and then gets 401. No job dir is created; I verified this. The middleware bounds the size, so this is acceptable for v1. The fix would be to check auth inside the `size_limit` middleware.
5. **note.** Failed jobs fire no webhook. SPEC says "on completion" and r3 does the same, so this is acceptable. The README should say so.

## Builder's "upload" note, judged
It is a real but minor inconsistency. The fix is the one-liner in defect 1. Injecting `upload_name` into the report after the scan would also work, but it adds code. Renaming the saved file is the smaller and root-cause fix.

## Checklist (all OK)
- **Auth:** `Depends(auth)` is on POST /jobs, GET /jobs/{id} and GET /jobs/{id}/functions. It uses `hmac.compare_digest` and is off when the key is empty. The tests cover /functions with a missing key and with a wrong key.
- **Limits:** 413 (header and streamed re-check), 411 (chunked), 400 (non-numeric or negative), 415 (suffix, case-insensitive). Rejected uploads leave no job dir. Against a real uvicorn I got:
  - chunked: 411
  - CL 99999999999: 413
  - CL "\xb2": 400 (from h11)
- **Job id:** `fullmatch([0-9a-f]{32})` runs before any path is built. All of these returned 404: `..%2f..`, `..%2F..%2Fetc`, `%2e%2e/functions`, uppercase, trailing `%0a`, and raw `../../etc/passwd`. A `../../evil.zip` upload filename is harmless, because the upload is stored under a fixed name.
- **Webhook:** scheme (http/https) and host are validated before mkdir, delivery uses stdlib urllib with a 5 s timeout, 1 retry, and never raises.
- **Retention:** keyed on dir mtime, which each atomic job.json replace bumps. It runs at startup, after each job and hourly. It only touches dirs with job-id names.
- **Restart:** running jobs become failed ("interrupted by server restart; resubmit"), queued jobs are re-enqueued in mtime order, and done and failed jobs are unchanged.
- **Partial results:** `functions.jsonl` rows drop `code`/`raw`, and any half-written last line is trimmed.
- **CLI:** `-o`, `--workers`, `--base-url`, `--model-file` all work, and `serve` runs with `workers=1`. On `ValueError` the CLI prints `wpjudge: msg` and exits 2, which is cleaner than a traceback.
- **TemporaryDirectory:** no `wpjudge-*` dirs are left in /tmp after zip, tar.gz or flat scans.
- **Over-engineering:** none worth flagging. api.py is 211 lines against the planned ~170, and the extra lines are guards and the restart-safe queue.

## What I ran
- `.venv/bin/python -m pytest tests/test_api.py -q`: 10 passed. Full `tests/`: 30 passed. The tests drive the real app, and removing auth from /functions or the id regex would make them fail.
- **CLI end to end on the dead port.** Command: `wpjudge scan tests/fixtures/vuln-plugin -o r.json --base-url http://127.0.0.1:9/v1 --workers 2 --model-file foo.gguf`.
  - Exit code 0.
  - stderr: `10 units, ETA ~0.9 min`.
  - The report has every v1 key, with `model={file: foo.gguf, base_url: ...:9/v1}` and `target.files_scanned=7`.
  - Coverage: 10 units, 0 judged, 10 unjudged, 5 skipped, pct 0.0, `low_coverage` true, `overall_score` null.
  - Each unjudged unit has the error `APIConnectionError`. No `code`/`raw` in rows.
  - The unit, skip and php-file counts match `vuln-plugin.expected.json` (10/5/7).
  - The .zip, .tar.gz and flat .zip give the same coverage.
  - A missing path and a non-archive each exit 2 with a one-line message.
- **API with the real `scan()` against the dead port (TestClient, auth on).** The .zip, flat .zip and .tar.gz jobs all reached `done` with progress 10/10, 10 rows in /functions, and the upload removed afterwards. A non-zip `.zip` failed with `Rejected: ...`. A missing file field returned 422.
- **`wpjudge serve --port 8797`** with raw-socket probes (results above). There were no tracebacks in the log.
- The probes are in the session scratchpad (`probe_api.py`). No harness files were edited.
