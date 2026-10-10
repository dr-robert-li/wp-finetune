"""wpjudge.judge on saved real IQ4_XS responses plus a stub client. No server, no network."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpjudge import judge

FIX = Path(__file__).parent / "fixtures" / "responses"
UNIT = {"id": "a.php:1:f", "file": "a.php", "line": 1, "end_line": 3, "name": "f",
        "kind": "function", "code": "<?php function f() {}", "loc": 3, "truncated": False}


def raw(name):
    return (FIX / f"{name}.txt").read_text()


@pytest.mark.parametrize("name,overall,scores", [
    ("score_format_0", 79, {"wpcs_compliance": 8, "security": 7, "performance": 8, "wp_api_usage": 8,
                            "code_quality": 8, "dependency_integrity": 7}),
    ("severity_format_7", 80, {"wpcs_compliance": 8, "wp_api_usage": 8, "code_quality": 8,
                               "dependency_integrity": 8}),
    ("severity_format_9", 72, {"wpcs_compliance": 7, "security": 6, "performance": 7, "wp_api_usage": 8,
                               "code_quality": 7, "dependency_integrity": 7, "accessibility": 8,
                               "error_handling": 6}),
])
def test_real_responses_parse(name, overall, scores):
    j = judge.to_judgment(UNIT, raw(name), 1)
    assert j["status"] == "judged" and j["verdict"] == "PASS" and j["error"] is None
    assert j["overall_score"] == overall and j["scores"] == scores  # JSON only, native keys
    assert j["critique"] and "[/REASONING]" not in j["critique"] and "<judge_output>" not in j["critique"]
    assert j["critique"].startswith(raw(name).strip()[:40]) and j["id"] == UNIT["id"]


def test_score_format_5_parses():
    assert judge.to_judgment(UNIT, raw("score_format_5"), 2)["attempts"] == 2


def test_broken_response_unparseable():
    assert judge.to_judgment(UNIT, raw("broken_89"), 1) is None


class Stub:
    """OpenAI-shaped client returning queued replies (str) or raising (Exception)."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def test_request_shape():
    c = Stub([raw("score_format_0")])
    j = judge.judge_one(c, "m", UNIT, max_tokens=123)
    kw = c.calls[0]
    assert j["attempts"] == 1 and kw["temperature"] == 0 and kw["max_tokens"] == 123 and kw["model"] == "m"
    assert kw["messages"] == [{"role": "user", "content":
                               "<wp_judge> Evaluate this WordPress code:\n\n```php\n<?php function f() {}\n```"}]
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_retry_after_unparseable_then_ok():
    c = Stub([raw("broken_89"), raw("severity_format_9")])
    j = judge.judge_one(c, "m", UNIT)
    assert j["status"] == "judged" and j["attempts"] == 2 and len(c.calls) == 2


def test_unjudged_after_two_failures():
    j = judge.judge_one(Stub([ConnectionError("down"), raw("broken_89")]), "m", UNIT)
    assert j["status"] == "unjudged" and j["attempts"] == 2 and j["error"] == "parse_failure"
    assert j["verdict"] is None and j["overall_score"] is None and j["scores"] == {}
    j = judge.judge_one(Stub([raw("broken_89"), ConnectionError("down")]), "m", UNIT)
    assert j["error"] == "ConnectionError: down"


def test_judge_units_sorted_and_callback(monkeypatch):
    units = [{**UNIT, "id": i} for i in ("c", "a", "b")]
    monkeypatch.setattr(judge, "OpenAI", lambda **kw: Stub([raw("score_format_0")] * 3))
    seen = []
    out = judge.judge_units(units, "http://x/v1", "m", workers=2, on_result=seen.append)
    assert [j["id"] for j in out] == ["a", "b", "c"] and len(seen) == 3
    assert all(j["status"] == "judged" for j in out)
