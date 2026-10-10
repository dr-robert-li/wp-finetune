"""wpjudge.judge on saved real IQ4_XS responses plus a stub client. No server, no network."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpjudge import judge
from wpjudge.aggregate import DIM

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
    assert j["status"] == "judged" and j["verdict"] == "PASS" and j["error"] is None and j["verdict_p"] is None
    assert j["overall_score"] == overall and j["scores"] == scores  # JSON only, native keys
    assert j["critique"] and "[/REASONING]" not in j["critique"] and "<judge_output>" not in j["critique"]
    assert j["critique"].startswith(raw(name).strip()[:40]) and j["id"] == UNIT["id"]


def test_score_format_5_parses():
    assert judge.to_judgment(UNIT, raw("score_format_5"), 2)["attempts"] == 2


def test_broken_response_unparseable():
    assert judge.to_judgment(UNIT, raw("broken_89"), 1) is None


def test_critique_synced_to_json():
    j = judge.to_judgment(UNIT, raw("severity_format_9"), 1)
    assert j["prose_scores"]["security"] == 4 and j["scores"]["security"] == 6
    assert j["prose_scores"]["sql_safety"] == 8 and "sql_safety" not in j["scores"]
    assert "Security: score 6/10 — Issue: $this->id is echoed" in j["critique"]
    assert "SQL Safety: score None/10 — Issue: No database" in j["critique"]
    assert "severity" not in j["critique"] and "severity high (4/10)" in j["raw"]
    for lab, a, b, _ in DIM.findall(j["critique"]):
        assert (a or b) == str(j["scores"].get(lab.lower().replace(" ", "_")))
    s0 = judge.to_judgment(UNIT, raw("score_format_0"), 1)   # N/A-in-JSON dims become None in score format too
    assert s0["prose_scores"]["sql_safety"] == 10 and "SQL Safety: score None/10 — No SQL" in s0["critique"]
    assert all(s0["prose_scores"][d] == v for d, v in s0["scores"].items() if d in s0["prose_scores"])


def lp(*pieces):
    """Fake choice.logprobs: (token, p) pairs."""
    return SimpleNamespace(content=[SimpleNamespace(token=t, logprob=math.log(p)) for t, p in pieces])


@pytest.mark.parametrize("pieces,want", [
    ([("[/REASONING]", 1), ("<judge_output>", 1), ("\n{\n", 1), (' "', 1), ("verdict", 1), ('":', 1),
      (' "', 0.9), ("FAIL", 0.5), ('",', 1)], 0.5),
    ([("<judge_output>", 1), ('{"verdict": ', 1), ('"', 0.7), ("FA", 0.25), ("IL", 0.99)], 0.25),  # split
    ([('"verdict": ', 1), ("PASS", 0.3), ("<judge_output>", 1), ('{"verdict":', 1), ('"PASS"', 0.8)], 0.8),
    ([("<judge_output>", 1), ('{"overall_score": 7}', 1)], None),
])
def test_verdict_prob(pieces, want):
    assert judge.verdict_prob(SimpleNamespace(logprobs=lp(*pieces))) == want


def test_verdict_prob_absent():
    assert judge.verdict_prob(SimpleNamespace()) is None
    assert judge.verdict_prob(SimpleNamespace(logprobs=None)) is None


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
        if isinstance(r, tuple):  # (content, logprobs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r[0]), logprobs=r[1])])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])


def test_request_shape():
    c = Stub([raw("score_format_0")])
    j = judge.judge_one(c, "m", UNIT, max_tokens=123)
    kw = c.calls[0]
    assert j["attempts"] == 1 and kw["temperature"] == 0 and kw["max_tokens"] == 123 and kw["model"] == "m"
    assert kw["messages"] == [{"role": "user", "content":
                               "<wp_judge> Evaluate this WordPress code:\n\n```php\n<?php function f() {}\n```"}]
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}} and kw["logprobs"] is True
    assert "top_logprobs" not in kw and j["verdict_p"] is None


def test_verdict_p_through_judge_one():
    c = Stub([(raw("score_format_0"), lp(("<judge_output>", 1), ('{"verdict": "', 1), ("PASS", 0.42)))])
    assert judge.judge_one(c, "m", UNIT)["verdict_p"] == 0.42


def test_template_rejected_path_sends_logprobs(monkeypatch):
    monkeypatch.setattr(judge, "_thinking_kwarg_supported", True)
    c = Stub([ValueError("chat_template kwargs not supported"), raw("score_format_0")])
    j = judge.judge_one(c, "m", UNIT)
    assert j["status"] == "judged" and len(c.calls) == 2
    assert c.calls[1]["logprobs"] is True and "extra_body" not in c.calls[1]


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
