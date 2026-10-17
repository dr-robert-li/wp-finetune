"""Worker-count stability: serial defaults, reproducible flag, /props probe, client-side order independence."""
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpjudge import aggregate, judge
from wpjudge.config import Config, load

H = Path(__file__).resolve().parents[1]


def test_serial_defaults(monkeypatch):
    monkeypatch.delenv("WPJUDGE_WORKERS", raising=False)
    assert load().workers == 1
    assert "LLAMA_PARALLEL:-1" in (H / "docker/llama-entrypoint.sh").read_text()
    compose = (H / "docker-compose.yml").read_text()
    assert compose.count("LLAMA_PARALLEL:-1") == 2 and "LLAMA_PARALLEL:-4" not in compose


@pytest.mark.parametrize("workers,slots,want", [(1, 4, True), (4, 1, True), (1, None, True), (1, 1, True),
                                                (4, 4, False), (4, None, False)])
def test_reproducible_truth_table(workers, slots, want):
    m = aggregate.build_report({}, {"workers": workers, "server_slots": slots}, [], [], Config())["model"]
    assert (m["workers"], m["server_slots"], m["reproducible"]) == (workers, slots, want)


def _serve(body):
    class Hd(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/props" else 404)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass
    s = ThreadingHTTPServer(("127.0.0.1", 0), Hd)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


@pytest.mark.parametrize("body,want", [('{"total_slots": 4}', 4), ('{"total_slots": 1}', 1), ("{}", None),
                                       ('{"total_slots": true}', None), ("not json", None)])
def test_server_slots_probe(body, want):
    s, root = _serve(body)
    try:
        assert judge.server_slots(root + "/v1") == want
        assert judge.server_slots(root + "/v1/") == want
    finally:
        s.shutdown()


def test_server_slots_down():
    s, root = _serve("{}")
    s.shutdown()
    s.server_close()
    assert judge.server_slots(root + "/v1") is None


class Jitter:
    """Reply depends only on the prompt; random latency reorders completions across workers."""
    def __init__(self, **kw):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages, **kw):
        time.sleep(random.random() * 0.02)
        n = sum(map(ord, messages[0]["content"])) % 90 + 5
        out = json.dumps({"verdict": "PASS", "overall_score": n, "security": n % 10})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=f"crit\n<judge_output>{out}</judge_output>"))])


def test_judge_units_worker_invariant(monkeypatch):
    # Guards the client side only; server-side batching nondeterminism needs the live e2e determinism section.
    monkeypatch.setattr(judge, "OpenAI", Jitter)
    units = [{"id": f"u{i:02}", "code": f"<?php function f{i}() {{}}"} for i in range(20)]
    w1 = judge.judge_units(units, "http://x/v1", "m", workers=1)
    w4 = judge.judge_units(units, "http://x/v1", "m", workers=4)
    assert w1 == w4 and len({j["overall_score"] for j in w1}) > 1
