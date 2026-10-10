from wpjudge.aggregate import build_report
from wpjudge.config import Config

CFG = Config()
META = {"name": "p", "type": "plugin", "sha256": "abc", "files_scanned": 3}
MODEL = {"file": "m.gguf", "base_url": "http://x/v1"}


def J(id_, score, loc=10, scores=None, verdict="PASS", critique="", status="judged"):
    f, line, name = id_.split(":")
    j = {"id": id_, "file": f, "line": int(line), "end_line": int(line) + loc, "name": name, "kind": "function",
         "code": "x", "loc": loc, "truncated": False, "status": status, "attempts": 1, "verdict": verdict,
         "overall_score": score, "scores": scores or {}, "critique": critique, "raw": "r", "error": None}
    if status == "unjudged":
        j.update(verdict=None, overall_score=None, scores={}, attempts=2, error="parse_failure")
    return j


def test_loc_weighting_no_cap():
    r = build_report(META, MODEL, [J("a.php:1:f", 90, loc=30, scores={"security": 5}),
                                   J("a.php:9:g", 50, loc=10, scores={"security": 9})], [], CFG)
    s = r["summary"]
    assert s["overall_score"] == s["overall_score_uncapped"] == 80.0   # (90*30+50*10)/40
    assert not s["capped"] and s["cap_reasons"] == []
    assert r["target"] == {"name": "p", "type": "plugin", "files_scanned": 3, "sha256": "abc"}
    assert r["model"] == MODEL and r["schema_version"] == "1.0"


def test_cap_and_files_scanned_arg():
    js = [J("a.php:1:f", 90, scores={"security": 9, "sql_safety": 4}, verdict="FAIL"),
          J("a.php:9:g", 80, scores={"security": 3}), J("a.php:20:h", 70, scores={"performance": 2})]
    r = build_report(META, MODEL, js, [], CFG, files_scanned=7)
    s = r["summary"]
    assert s["overall_score_uncapped"] == 80.0 and s["overall_score"] == 60 and s["capped"]
    assert s["cap_reasons"] == [{"id": "a.php:1:f", "dimension": "sql_safety", "score": 4},
                                {"id": "a.php:9:g", "dimension": "security", "score": 3}]
    assert s["verdict_counts"] == {"PASS": 2, "FAIL": 1}
    assert r["target"]["files_scanned"] == 7
    # cap never raises the score
    low = build_report(META, MODEL, [J("a.php:1:f", 40, scores={"security": 1})], [], CFG)["summary"]
    assert low["overall_score"] == 40.0 and low["capped"]


def test_coverage_and_zero_judged():
    js = [J(f"a.php:{i}:f{i}", 80) for i in range(1, 9)] + [J("a.php:99:u", None, status="unjudged")]
    c = build_report(META, MODEL, js, [{"file": "vendor/x.php", "line": 0, "name": "", "reason": "vendor"}],
                     CFG)["summary"]["coverage"]
    assert c == {"units": 9, "judged": 8, "unjudged": 1, "skipped": 1, "pct": 88.9, "low_coverage": True}
    r = build_report(META, MODEL, [J("a.php:1:u", None, status="unjudged")], [], CFG)
    s = r["summary"]
    assert s["overall_score"] is None and s["overall_score_uncapped"] is None and not s["capped"]
    assert s["coverage"]["pct"] == 0.0 and s["coverage"]["low_coverage"]
    assert r["dimensions"] == {} and r["worst_functions"] == [] and s["verdict_counts"] == {"PASS": 0, "FAIL": 0}
    e = build_report({}, MODEL, [], [], CFG)["summary"]["coverage"]
    assert e["units"] == 0 and e["pct"] == 0.0 and e["low_coverage"]


def test_dimensions():
    js = [J("a.php:1:f", 80, scores={"security": 2}), J("a.php:5:g", 80, scores={"security": 8, "i18n": 9}),
          J("a.php:9:h", None, status="unjudged")]
    d = build_report(META, MODEL, js, [], CFG)["dimensions"]
    assert d["security"] == {"mean": 5.0, "std": 3.0, "min": 2, "n_rated": 2, "low_rate": 0.5}
    assert d["i18n"]["n_rated"] == 1


def test_ranking():
    crit = ("Security: score 2/10 — Missing nonce verification and no current_user_can check; nonce absent\n"
            "WPCS Compliance: severity high (9/10) — Issue: camelCase method name. Fix: rename\n"
            "Performance: score 6/10 — fine\n"
            "SQL Safety: score 10/10 — no queries\n")
    js = [J("a.php:1:f", 50, scores={"security": 2, "wpcs_compliance": 3, "performance": 6}, critique=crit),
          J("b.php:7:g", 50, scores={"security": 3}, critique="Security: score 3/10 — nonce missing"),
          J("c.php:2:h", 90, scores={"wpcs_compliance": 9}, critique="WPCS Compliance: score 9/10 — good docblock"),
          J("d.php:1:u", None, status="unjudged", critique="Security: score 1/10 — nonce")]
    r = build_report(META, MODEL, js, [], CFG)
    anti = r["anti_patterns"]
    # nonce counted once per function despite appearing twice in f's line; unjudged ignored
    assert anti[0] == {"dimension": "security", "finding": "nonce", "count": 2, "examples": ["a.php:1 f"]}
    assert {"dimension": "security", "finding": "capability check", "count": 1, "examples": ["a.php:1 f"]} in anti
    # JSON score (3) wins over the prose score (9); sql_safety absent from JSON = N/A
    assert {"dimension": "wpcs_compliance", "finding": "naming", "count": 1, "examples": ["a.php:1 f"]} in anti
    assert not any(a["dimension"] in ("sql_safety", "performance") for a in anti)
    assert r["patterns"] == [{"dimension": "wpcs_compliance", "finding": "docblock", "count": 1}]


def test_worst_ties_and_rows():
    js = [J(f"a.php:{i}:f{i}", 70) for i in range(12, 0, -1)] + [J("z.php:1:low", 10)]
    r = build_report(META, MODEL, js, [{"file": "t/x.php", "line": 3, "name": "tiny", "reason": "tiny"}], CFG)
    w = r["worst_functions"]
    assert len(w) == 10 and w[0] == {"file": "z.php", "line": 1, "name": "low", "overall_score": 10, "verdict": "PASS"}
    assert [x["name"] for x in w[1:4]] == ["f10", "f11", "f12"]  # tie broken by id string ("a.php:10:" < "a.php:1:")
    f = r["functions"]
    assert len(f) == 14 and all("code" not in x and "raw" not in x for x in f)
    assert {"id", "kind", "end_line", "attempts", "error", "loc", "truncated", "critique"} <= set(f[0])
    assert f[-1] == {"file": "t/x.php", "line": 3, "name": "tiny", "loc": 0, "truncated": False, "status": "skipped",
                     "reason": "tiny", "verdict": None, "overall_score": None, "scores": {}, "critique": ""}
