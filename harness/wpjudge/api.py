"""Async-job API: one daemon worker thread + queue.Queue, one dir per job (r3 design B).

Run with ONE uvicorn worker: the queue lives in memory (`wpjudge serve` does this).
"""
import hmac
import http.client
import ipaddress
import json
import os
import queue
import re
import shutil
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from wpjudge import scan
from wpjudge.config import load

CFG = load()
SUFFIXES = (".zip", ".tar.gz", ".tgz")
JOB_ID = re.compile(r"[0-9a-f]{32}")
EMBED_V4 = [ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::/96", "::ffff:0:0:0/96")]  # NAT64, v4-compatible, SIIT
Q: "queue.Queue[str | None]" = queue.Queue()


def jobs_dir() -> Path:
    return Path(CFG.jobs_dir)


def max_bytes() -> int:
    return int(CFG.max_upload_mb * 2**20)


def write_json(p: Path, obj):  # atomic: readers never see half a file
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def job_path(jid: str) -> Path:
    if not JOB_ID.fullmatch(jid):  # checked before any path is built: blocks traversal
        raise HTTPException(404, "no such job")
    return jobs_dir() / jid


def read_job(jid: str) -> dict:
    try:
        return json.loads((job_path(jid) / "job.json").read_text())
    except FileNotFoundError:
        raise HTTPException(404, "no such job")


def webhook_target(url: str):
    """Resolve once; every address must be public (or in webhook_allow_nets). -> (parts, port, ip). ValueError if not."""
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("webhook_url must be an http(s) URL")
    port = u.port or (443 if u.scheme == "https" else 80)  # bad port -> ValueError
    try:
        infos = socket.getaddrinfo(u.hostname, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise ValueError(f"webhook host does not resolve: {e}")
    allow = [ipaddress.ip_network(n, strict=False) for n in CFG.webhook_allow_nets]
    ips = {ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos}
    for ip in ips:
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if any(ip in n for n in EMBED_V4):  # these forms reach the embedded a.b.c.d
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if (not ip.is_global or ip.is_multicast) and not any(ip in n for n in allow):
            raise ValueError(f"webhook_url resolves to a non-public address ({ip})")
    return u, port, str(min(ips, key=lambda a: a.version))


def webhook(url: str, report: dict):
    body = json.dumps(report).encode()
    for _ in (1, 2):  # best effort: one retry, never raises
        try:
            u, port, ip = webhook_target(url)  # re-checked at send time: DNS may have changed since submit
            C = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
            c = C(u.hostname, port, timeout=5)  # Host header, SNI and cert check use the name
            c._create_connection = lambda a, *r: socket.create_connection((ip, a[1]), *r)  # connect to the checked IP only
            try:
                c.request("POST", (u.path or "/") + (f"?{u.query}" if u.query else ""), body,
                          {"Content-Type": "application/json"})
                c.getresponse()  # http.client never follows redirects
            finally:
                c.close()
            return
        except ValueError:
            return  # blocked: retrying will not help
        except Exception:
            time.sleep(1)


def sweep():
    cutoff = time.time() - CFG.retention_days * 86400
    for d in jobs_dir().iterdir():  # only job-id dirs: a misconfigured jobs_dir loses nothing else
        if d.is_dir() and JOB_ID.fullmatch(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def run_job(jid: str):
    d = jobs_dir() / jid
    job = json.loads((d / "job.json").read_text())
    job["status"] = "running"
    write_json(d / "job.json", job)
    report = None
    with open(d / "functions.jsonl", "a") as fl:
        def on_result(j, done, total):
            if j is not None:
                fl.write(json.dumps({k: v for k, v in j.items() if k not in ("code", "raw")}) + "\n")
                fl.flush()
            job["progress"] = {"done": done, "total": total}
            write_json(d / "job.json", job)
        try:
            report = scan(str(d / job["filename"]), on_result=on_result)
            write_json(d / "report.json", report)
            job["status"] = "done"
        except Exception as e:
            job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
    (d / job["filename"]).unlink(missing_ok=True)  # input no longer needed
    write_json(d / "job.json", job)
    if job["status"] == "done" and job.get("webhook_url"):
        webhook(job["webhook_url"], report)


def worker(q):
    while True:
        try:
            jid = q.get(timeout=3600)
            if jid is None:  # shutdown
                return
            run_job(jid)
        except queue.Empty:
            pass
        except Exception:  # ponytail: a crash outside run_job's try leaves the job "running" until restart
            pass
        sweep()  # after every job, and hourly when idle


@asynccontextmanager
async def lifespan(app):
    global Q
    Q = queue.Queue()
    jobs_dir().mkdir(parents=True, exist_ok=True)
    sweep()
    for f in sorted(jobs_dir().glob("*/job.json"), key=lambda p: p.stat().st_mtime):
        if not JOB_ID.fullmatch(f.parent.name):
            continue
        job = json.loads(f.read_text())
        if job["status"] == "running":  # interrupted mid-scan: say so, don't guess
            job.update(status="failed", error="interrupted by server restart; resubmit")
            write_json(f, job)
        elif job["status"] == "queued":  # input is on disk, so it can still run
            Q.put(job["id"])
    threading.Thread(target=worker, args=(Q,), daemon=True).start()
    yield
    Q.put(None)


app = FastAPI(title="wpjudge", lifespan=lifespan)


@app.middleware("http")
async def size_limit(request: Request, call_next):
    # Decide on the header, before any body byte is read. Chunked bodies (no Content-Length)
    # would be spooled whole by the multipart parser before the handler runs, so refuse them.
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if not cl.isdigit():
            return JSONResponse({"detail": "bad Content-Length"}, status_code=400)
        if int(cl) > max_bytes():
            return JSONResponse({"detail": "upload too large"}, status_code=413)
    return await call_next(request)


def auth(request: Request):
    if CFG.api_key and not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                                               f"Bearer {CFG.api_key}".encode()):
        raise HTTPException(401, "bad or missing API key", headers={"WWW-Authenticate": "Bearer"})


@app.post("/jobs", status_code=202, dependencies=[Depends(auth)])
async def create_job(file: UploadFile, webhook_url: str | None = Form(None)):
    name = file.filename or ""
    if not name.lower().endswith(SUFFIXES):
        raise HTTPException(415, f"expected one of {SUFFIXES}")
    if webhook_url:
        try:
            await run_in_threadpool(webhook_target, webhook_url)  # getaddrinfo blocks: keep it off the event loop
        except ValueError as e:
            raise HTTPException(400, str(e))
    jid = uuid.uuid4().hex
    d = jobs_dir() / jid
    d.mkdir(parents=True)
    dest = d / ("upload.zip" if name.lower().endswith(".zip") else "upload.tar.gz")
    limit = max_bytes()

    def save():  # runs in a thread: the disk copy never blocks the event loop
        n = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(2**20):
                n += len(chunk)
                if n > limit:  # second line of defence behind the header check
                    return False
                out.write(chunk)
        return True

    if not await run_in_threadpool(save):
        shutil.rmtree(d)
        raise HTTPException(413, "upload too large")
    job = {"id": jid, "status": "queued", "filename": dest.name, "upload_name": name,
           "webhook_url": webhook_url, "created": time.time(), "error": None,
           "progress": {"done": 0, "total": None}}
    write_json(d / "job.json", job)
    Q.put(jid)
    return {"job_id": jid}


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):  # sync def: FastAPI runs it in the threadpool
    job = read_job(jid)
    out = {k: job.get(k) for k in ("id", "status", "progress", "error")}
    if job["status"] == "done":
        out["report"] = json.loads((job_path(jid) / "report.json").read_text())
    return out


@app.get("/jobs/{jid}/functions", dependencies=[Depends(auth)])
def get_functions(jid: str):
    read_job(jid)
    p = job_path(jid) / "functions.jsonl"
    text = p.read_text() if p.exists() else ""
    text = text[: text.rfind("\n") + 1]  # drop a line the worker is still writing
    return {"functions": [json.loads(line) for line in text.splitlines()]}
