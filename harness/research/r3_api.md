# R3: API design (async jobs)

Covers SPEC R1.2, R2.3, R2.4 and R3.3. I built and ran a prototype: `api.py`, 171 lines (137 without comments or blank lines). It is in the session scratchpad `…/scratchpad/r3/` with its own venv: fastapi 0.143, starlette 1.7, uvicorn 0.54, python-multipart 0.0.32, httpx 0.28.

## Designs compared

| | A. `BackgroundTasks` | **B. One daemon thread + `queue.Queue`, started in lifespan** | C. `asyncio` task + `asyncio.Queue` |
|---|---|---|---|
| Runs jobs one at a time | No. Each request starts its own task in the threadpool, so 3 uploads give 3 scans in parallel on one GPU. It needs an extra lock. | Yes, by construction | Yes |
| Blocks the event loop | No | No. | The scan (blocking OpenAI calls and the PHP subprocess) needs `to_thread` everywhere |
| Re-queues after a restart | Hand-rolled | `Q.put(id)` in lifespan | Same as B |

**Recommendation: B.** `judge_units` is already threaded and blocking (W2 contract). A plain thread calls it directly. The queue gives the single-GPU ordering for free. A is fire-and-forget per request, the wrong tool here. C adds async plumbing that nothing needs.

## Endpoints

| Method | Path | Auth | Returns |
|---|---|---|---|
| POST | `/jobs` | Bearer | multipart `file` (.zip/.tar.gz/.tgz) + optional `webhook_url` form field → `202 {"job_id"}`. Errors: 401, 411 (no Content-Length), 413 (too large), 415 (wrong type) |
| GET | `/jobs/{id}` | Bearer | `{id, status: queued\|running\|done\|failed, progress: {done, total}, error, report?}`; `report` is present only when the status is `done` |
| GET | `/jobs/{id}/functions` | Bearer | `{"functions": [Judgment…]}` for the judgments finished so far |

Auth is a dependency that compares `Authorization: Bearer $WPJUDGE_API_KEY` with `hmac.compare_digest`. When the env var is empty, auth is off. The other env vars are `WPJUDGE_JOBS_DIR`, `WPJUDGE_MAX_UPLOAD_MB` (50) and `WPJUDGE_RETENTION_DAYS` (7). In the build they move into `config.py`.

## Disk store

```
jobs/<uuid4 hex>/job.json        status, progress, webhook_url, error (atomic: write tmp + os.replace)
                 functions.jsonl one Judgment per line, appended and flushed by the worker
                 report.json     written once on success
                 upload.zip|tar.gz  deleted after the scan
```
The job id must parse as a UUID before any path is built from it, which blocks traversal. The worker is the only writer. The readers (GET endpoints) are plain `def`, so they run in the threadpool. A reader drops any trailing partial line of functions.jsonl.

## Key code

```python
Q = queue.Queue()

@app.middleware("http")
async def size_limit(request, call_next):
    # Decide on the header, before any body byte is read. Chunked bodies would be spooled
    # whole by the multipart parser before the handler runs, so refuse them.
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is None:   return JSONResponse({"detail": "Content-Length required"}, 411)
        if int(cl) > MAX_BYTES: return JSONResponse({"detail": "upload too large"}, 413)
    return await call_next(request)

@app.post("/jobs", status_code=202, dependencies=[Depends(auth)])
async def create_job(file: UploadFile, webhook_url: str | None = Form(None)):
    ...suffix check (415), mkdir jobs/<uuid>...
    def save():                       # threadpool: the disk copy never blocks the loop
        n = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(2**20):
                n += len(chunk)
                if n > MAX_BYTES: return False   # second line of defence
                out.write(chunk)
        return True
    if not await run_in_threadpool(save): shutil.rmtree(d); raise HTTPException(413)
    write_json(d / "job.json", {...status "queued"...}); Q.put(jid); return {"job_id": jid}

def worker():
    while True:
        try: run_job(Q.get(timeout=3600))
        except queue.Empty: pass
        except Exception: pass   # ponytail: crash leaves job "running" until next restart
        sweep()                   # retention after every job and hourly when idle

def run_job(jid):   # status running → scan(on_result appends jsonl + updates progress)
                    # → report.json, done | failed+error → delete upload → webhook

def webhook(url, report):
    for _ in (1, 2):                                   # one retry, never raises
        try: httpx.post(url, json=report, timeout=5).raise_for_status(); return
        except Exception: time.sleep(1)

def sweep():   # rmtree job dirs whose mtime is older than RETENTION (mtime = last write)

@asynccontextmanager
async def lifespan(app):
    sweep()
    for f in sorted(JOBS.glob("*/job.json"), key=mtime):
        if status == "running": mark failed, "interrupted by server restart; resubmit"
        elif status == "queued": Q.put(id)            # its upload is still on disk
    threading.Thread(target=worker, daemon=True).start()
    yield
```
In the build, `fake_scan(path, on_result)` becomes `load_target` → `collect_units` → `judge_units(..., on_result=)` → `build_report`. The worker gets `total` from `len(units)` before judging starts.

## Restart behaviour (the honest minimum)
- **done/failed:** served from disk unchanged.
- **queued:** re-enqueued in mtime order, because the upload is on disk.
- **running:** marked `failed` with `error: "interrupted by server restart; resubmit"`. Partial `functions.jsonl` stays readable. Resuming the scan is not worth it for v1.

## Test evidence (real uvicorn on :8765, curl, a stdlib http.server webhook listener on :8766)
Settings for the run: `WPJUDGE_API_KEY=s3cret` and `MAX_UPLOAD_MB=1`. The fake scan emits 5 judgments at 0.4 s each.

| Check | Result |
|---|---|
| GET without a key / POST with the wrong key | `401` / `401` |
| 2 MB upload with a 1 MB limit | `413`, `size_upload 0B`: rejected on the header and the body was never sent |
| Chunked upload (no Content-Length) | `411`, and no job dir left behind |
| Lying `Content-Length: 100` on a 2 MB body | `422`: h11 reads exactly 100 bytes, so nothing is spooled |
| `.py` upload | `415` |
| Upload + poll at 1.1 s | `running, progress {done:2,total:5}` |
| Second job during that | `queued` (runs one at a time) |
| `/functions` mid-run | 2 judgments |
| Final poll | `done`, `{5,5}`, report embedded; `/functions` = 5 |
| Webhook | listener logged `POST /hook 200` with the report body |
| Restart with J3 running (4/5) and J4 queued | after restart, J3 is `failed` "interrupted by server restart; resubmit" and J4 runs to `done` |
| Retention | dir touched to 8 days ago is deleted at startup; dir touched to 6 days ago is kept |
| Event loop during a 40 MB upload (limit 50, curl `--limit-rate 20M`) | 3 concurrent GETs answered in 2–4 ms each |
| uvicorn log | no tracebacks; with no env var set, auth is off and uploads are accepted |

## Risks and notes
1. **The upload limit covers the compressed size only.** The zip-bomb and ratio guard belongs in `extract.load_target` (W1). The worker must call it inside the job dir.
2. **411 on chunked uploads** is a deliberate trade. Starlette's parser spools the whole file part to a temp file before the handler runs. `max_part_size` caps only non-file fields. curl, httpx and requests send Content-Length for files, so this costs nothing in practice.
3. **One process only.** The queue lives in memory, so `uvicorn --workers N` breaks the one-at-a-time ordering and splits the queue. The Dockerfile must run one worker. Document this.
4. **A worker bug** outside `run_job`'s try leaves a job `running` until the next restart marks it failed.
5. **Webhook** is unsigned and best effort, and runs on the worker thread. A dead endpoint delays the next job by about 11 s at most. The URL is unvalidated, which is an SSRF risk if callers are untrusted. The API key is the gate.
6. **Retention** uses dir mtime. Each job.json replace bumps it, so an active job is never swept.
7. Skipped: job listing, cancel and auth scopes.
