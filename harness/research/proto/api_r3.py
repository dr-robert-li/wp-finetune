"""Prototype: WP Judge async-job API. One worker thread, one dir per job."""
import hmac, json, os, queue, shutil, threading, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

JOBS = Path(os.environ.get("WPJUDGE_JOBS_DIR", "jobs"))
MAX_BYTES = int(os.environ.get("WPJUDGE_MAX_UPLOAD_MB", "50")) * 2**20
RETENTION_S = float(os.environ.get("WPJUDGE_RETENTION_DAYS", "7")) * 86400
API_KEY = os.environ.get("WPJUDGE_API_KEY", "")
SUFFIXES = (".zip", ".tar.gz", ".tgz")
Q: "queue.Queue[str]" = queue.Queue()


def write_json(p: Path, obj):  # atomic: readers never see half a file
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def read_job(jid: str) -> dict:
    try:
        uuid.UUID(hex=jid)  # also blocks path traversal
        return json.loads((JOBS / jid / "job.json").read_text())
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "no such job")


def fake_scan(archive: Path, on_result):  # stand-in for wpjudge.scan(..., on_result=)
    n = 5
    for i in range(n):
        time.sleep(0.4)
        on_result({"id": f"a.php:{i}:f{i}", "status": "judged", "verdict": "PASS", "overall_score": 80 + i}, i + 1, n)
    return {"schema_version": "1.0", "summary": {"overall_score": 82}, "target": {"name": archive.name}}


def webhook(url: str, report: dict):
    for attempt in (1, 2):  # best effort: one retry, never raises
        try:
            httpx.post(url, json=report, timeout=5).raise_for_status()
            return
        except Exception:
            time.sleep(1)


def sweep():
    cutoff = time.time() - RETENTION_S
    for d in JOBS.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def run_job(jid: str):
    d = JOBS / jid
    job = json.loads((d / "job.json").read_text())
    job["status"] = "running"
    write_json(d / "job.json", job)
    with open(d / "functions.jsonl", "a") as fl:
        def on_result(j, done, total):
            fl.write(json.dumps(j) + "\n"); fl.flush()
            job["progress"] = {"done": done, "total": total}
            write_json(d / "job.json", job)
        try:
            report = fake_scan(d / job["filename"], on_result)
            write_json(d / "report.json", report)
            job["status"] = "done"
        except Exception as e:
            job["status"], job["error"] = "failed", repr(e)
    (d / job["filename"]).unlink(missing_ok=True)  # input no longer needed
    write_json(d / "job.json", job)
    if job["status"] == "done" and job.get("webhook_url"):
        webhook(job["webhook_url"], report)


def worker():
    while True:
        try:
            run_job(Q.get(timeout=3600))
        except queue.Empty:
            pass
        except Exception:  # ponytail: a crashed job is left "running" until restart marks it failed
            pass
        sweep()  # after every job, and hourly when idle


@asynccontextmanager
async def lifespan(app):
    JOBS.mkdir(parents=True, exist_ok=True)
    sweep()
    for f in sorted(JOBS.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
        job = json.loads(f.read_text())
        if job["status"] == "running":  # interrupted mid-scan: say so, don't guess
            job.update(status="failed", error="interrupted by server restart; resubmit")
            write_json(f, job)
        elif job["status"] == "queued":  # input is on disk, so it can still run
            Q.put(job["id"])
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def size_limit(request: Request, call_next):
    # Decide on the header, before any body byte is read. Chunked bodies (no Content-Length)
    # would be spooled whole by the multipart parser before our handler runs, so refuse them.
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if int(cl) > MAX_BYTES:
            return JSONResponse({"detail": "upload too large"}, status_code=413)
    return await call_next(request)


def auth(request: Request):
    if API_KEY and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        raise HTTPException(401, "bad or missing API key", headers={"WWW-Authenticate": "Bearer"})


@app.post("/jobs", status_code=202, dependencies=[Depends(auth)])
async def create_job(file: UploadFile, webhook_url: str | None = Form(None)):
    name = file.filename or ""
    if not name.endswith(SUFFIXES):
        raise HTTPException(415, f"expected one of {SUFFIXES}")
    jid = uuid.uuid4().hex
    d = JOBS / jid
    d.mkdir()
    dest = d / ("upload.zip" if name.endswith(".zip") else "upload.tar.gz")

    def save():  # runs in a thread: disk copy never blocks the event loop
        n = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(2**20):
                n += len(chunk)
                if n > MAX_BYTES:
                    return False
                out.write(chunk)
        return True

    if not await run_in_threadpool(save):
        shutil.rmtree(d)
        raise HTTPException(413, "upload too large")
    job = {"id": jid, "status": "queued", "filename": dest.name, "upload_name": name,
           "webhook_url": webhook_url, "created": time.time(), "progress": {"done": 0, "total": None}}
    write_json(d / "job.json", job)
    Q.put(jid)
    return {"job_id": jid}


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):  # sync def: FastAPI runs it in the threadpool
    job = read_job(jid)
    out = {k: job.get(k) for k in ("id", "status", "progress", "error")}
    if job["status"] == "done":
        out["report"] = json.loads((JOBS / jid / "report.json").read_text())
    return out


@app.get("/jobs/{jid}/functions", dependencies=[Depends(auth)])
def get_functions(jid: str):
    read_job(jid)
    p = JOBS / jid / "functions.jsonl"
    text = p.read_text() if p.exists() else ""
    text = text[: text.rfind("\n") + 1]  # drop a line the worker is still writing
    return {"functions": [json.loads(l) for l in text.splitlines()]}
