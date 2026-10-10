"""Drive the real FastAPI app through TestClient, with scan() stubbed (no model server)."""
import http.server
import json
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from wpjudge import api
from wpjudge.config import load

KEY = {"Authorization": "Bearer s3cret"}
ZIP = {"file": ("plugin.zip", b"PK\x03\x04 not really", "application/zip")}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh config + a controllable stub scan. gate: each scan waits until released."""
    monkeypatch.setattr(api, "CFG", load(jobs_dir=str(tmp_path / "jobs"), api_key="", max_upload_mb=1))
    state = {"order": [], "gate": threading.Event(), "step": threading.Event()}
    state["gate"].set()

    def stub(path, on_result=None):
        assert os.path.exists(path)
        state["order"].append(path.split(os.sep)[-2])
        on_result(None, 0, 3)
        for i in range(3):
            if i == 1:  # pause mid-run so tests can see partial state
                state["step"].set()
                state["gate"].wait(10)
            on_result({"id": f"a.php:{i}:f{i}", "status": "judged", "code": "x", "raw": "y"}, i + 1, 3)
        return {"schema_version": "1.0", "summary": {"overall_score": 80}}

    monkeypatch.setattr(api, "scan", stub)
    return state


def submit(c, headers=None, **data):
    return c.post("/jobs", files=ZIP, data=data, headers=headers or {})


def wait(c, jid, status, headers=None):
    for _ in range(200):
        r = c.get(f"/jobs/{jid}", headers=headers or {}).json()
        if r["status"] == status:
            return r
        time.sleep(0.02)
    raise AssertionError(f"{jid} never reached {status}: {r}")


def test_auth_on_every_endpoint(env):
    api.CFG.api_key = "s3cret"
    with TestClient(api.app) as c:
        assert submit(c).status_code == 401
        assert submit(c, {"Authorization": "Bearer nope"}).status_code == 401
        jid = submit(c, KEY).json()["job_id"]
        assert c.get(f"/jobs/{jid}").status_code == 401
        assert c.get(f"/jobs/{jid}/functions").status_code == 401
        assert c.get(f"/jobs/{jid}/functions", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert wait(c, jid, "done", KEY)["report"]["summary"]["overall_score"] == 80
        assert len(c.get(f"/jobs/{jid}/functions", headers=KEY).json()["functions"]) == 3


def test_auth_off_when_key_empty(env):
    with TestClient(api.app) as c:
        jid = submit(c).json()["job_id"]
        assert wait(c, jid, "done")["progress"] == {"done": 3, "total": 3}
        fn = c.get(f"/jobs/{jid}/functions").json()["functions"]
        assert [f["id"] for f in fn] == ["a.php:0:f0", "a.php:1:f1", "a.php:2:f2"]
        assert "code" not in fn[0] and "raw" not in fn[0]
        assert not os.path.exists(os.path.join(api.CFG.jobs_dir, jid, "upload.zip"))


def test_upload_limits(env):
    jobs = api.CFG.jobs_dir
    with TestClient(api.app) as c:
        big = {"file": ("p.zip", b"0" * (2 * 2**20), "application/zip")}
        assert c.post("/jobs", files=big).status_code == 413
        r = c.post("/jobs", content=iter([b"abc"]), headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert r.status_code == 411
        r = c.post("/jobs", content=b"abc", headers={"Content-Length": "lots"})
        assert r.status_code == 400
        bad = {"file": ("evil.py", b"print(1)", "text/x-python")}
        assert c.post("/jobs", files=bad).status_code == 415
        assert os.listdir(jobs) == []  # no job dir left behind by a rejected upload
        for ok in ("p.tar.gz", "p.tgz", "P.ZIP"):
            r = c.post("/jobs", files={"file": (ok, b"x", "application/octet-stream")})
            assert r.status_code == 202
            wait(c, r.json()["job_id"], "done")  # don't leave the worker busy past teardown


def test_bad_job_ids(env):
    with TestClient(api.app) as c:
        for path in ("/jobs/x", "/jobs/..%2f..", "/jobs/..%2f../functions", "/jobs/..%2F..%2Fetc",
                     "/jobs/" + "A" * 32, "/jobs/" + "0" * 32, "/jobs/" + "0" * 32 + "/functions"):
            assert c.get(path).status_code == 404, path


def test_webhook_scheme_rejected(env):
    with TestClient(api.app) as c:
        for url in ("file:///etc/passwd", "ftp://h/x", "gopher://h", "http:///nohost", "javascript:alert(1)"):
            assert submit(c, webhook_url=url).status_code == 400, url
        assert os.listdir(api.CFG.jobs_dir) == []


def test_webhook_delivered(env):
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with TestClient(api.app) as c:
            jid = submit(c, webhook_url=f"http://127.0.0.1:{srv.server_port}/hook").json()["job_id"]
            wait(c, jid, "done")
            for _ in range(100):
                if got:
                    break
                time.sleep(0.02)
        assert got and got[0]["summary"]["overall_score"] == 80
    finally:
        srv.shutdown()


def test_queue_order_and_partial_results(env):
    env["gate"].clear()
    with TestClient(api.app) as c:
        j1 = submit(c).json()["job_id"]
        assert env["step"].wait(5)
        j2 = submit(c).json()["job_id"]
        j3 = submit(c).json()["job_id"]
        r1 = c.get(f"/jobs/{j1}").json()
        assert r1["status"] == "running" and r1["progress"] == {"done": 1, "total": 3} and "report" not in r1
        assert len(c.get(f"/jobs/{j1}/functions").json()["functions"]) == 1
        assert c.get(f"/jobs/{j2}").json()["status"] == "queued"  # one job at a time
        assert c.get(f"/jobs/{j2}/functions").json() == {"functions": []}
        env["gate"].set()
        for j in (j1, j2, j3):
            wait(c, j, "done")
    assert env["order"] == [j1, j2, j3]


def test_failed_scan(env, monkeypatch):
    def boom(path, on_result=None):
        raise ValueError("unsafe archive member")
    monkeypatch.setattr(api, "scan", boom)
    with TestClient(api.app) as c:
        jid = submit(c).json()["job_id"]
        r = wait(c, jid, "failed")
        assert "unsafe archive member" in r["error"] and "report" not in r


def make_job(jid, status, upload=True, age_days=0):
    d = os.path.join(api.CFG.jobs_dir, jid)
    os.makedirs(d)
    if upload:
        open(os.path.join(d, "upload.zip"), "wb").write(b"PK")
    with open(os.path.join(d, "job.json"), "w") as f:
        json.dump({"id": jid, "status": status, "filename": "upload.zip", "webhook_url": None,
                   "error": None, "progress": {"done": 1, "total": 3}}, f)
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(d, (t, t))
    return jid


def test_restart_behaviour(env):
    os.makedirs(api.CFG.jobs_dir)
    run = make_job("a" * 32, "running")
    q1 = make_job("b" * 32, "queued")
    time.sleep(0.02)
    q2 = make_job("c" * 32, "queued")
    with TestClient(api.app) as c:
        r = c.get(f"/jobs/{run}").json()
        assert r["status"] == "failed" and r["error"] == "interrupted by server restart; resubmit"
        wait(c, q1, "done")
        wait(c, q2, "done")
    assert env["order"] == [q1, q2]  # re-enqueued in mtime order


def test_retention(env):
    os.makedirs(api.CFG.jobs_dir)
    old = make_job("d" * 32, "failed", upload=False, age_days=8)
    keep = make_job("e" * 32, "failed", upload=False, age_days=6)
    other = os.path.join(api.CFG.jobs_dir, "not-a-job")
    os.makedirs(other)
    os.utime(other, (0, 0))
    with TestClient(api.app) as c:
        assert c.get(f"/jobs/{old}").status_code == 404
        assert c.get(f"/jobs/{keep}").json()["status"] == "failed"
    assert os.path.isdir(other)  # sweep only touches job-id dirs
