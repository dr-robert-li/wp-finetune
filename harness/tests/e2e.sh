#!/usr/bin/env bash
# Phase 5 integration + functional suite (PLAN.md phase 5, binding changes 1-3, 8, 12-15).
# Usage: tests/e2e.sh <section>...   sections: cli malicious serverdown api allinone library parity real determinism prerun | all
# Env knobs: GGUF (host model file), LLAMA_BIN, PORT (host server, 8097), HOST_PARALLEL (1), WORK (scratch dir),
#            API_PORT (8000, compose), AIO_PORT (8001), PARITY_PORT (8098), DET_PORT (8099), JOB_TIMEOUT (secs, 1800)
# GPU sections start the host llama-server only if nothing answers on PORT, and stop it before docker sections.
set -uo pipefail

H=$(cd "$(dirname "$0")/.." && pwd)
PY=$H/.venv/bin/python
WJ=$H/.venv/bin/wpjudge
FIXDIR=$H/tests/fixtures
FIX=$FIXDIR/vuln-plugin
EXP=$FIXDIR/vuln-plugin.expected.json
PRE=$FIXDIR/vuln-plugin.prerun.json
GGUF=${GGUF:-$(cd "$H/.." && pwd)/models/_gguf/shrink-iq4xs-im_f16.gguf}
GGUF_DIR=$(dirname "$GGUF")
MODEL_FILE=$(basename "$GGUF")
LLAMA_BIN=${LLAMA_BIN:-$HOME/work/llama.cpp/build/bin/llama-server}
PORT=${PORT:-8097}
BASE=http://127.0.0.1:$PORT/v1
HOST_PARALLEL=${HOST_PARALLEL:-1}  # serial decoding is the only reproducible mode (T5)
DET_PORT=${DET_PORT:-8099}
API_PORT=${API_PORT:-8000}
AIO_PORT=${AIO_PORT:-8001}
PARITY_PORT=${PARITY_PORT:-8098}
JOB_TIMEOUT=${JOB_TIMEOUT:-1800}
IMG=ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515
VAL=$(cd "$H/.." && pwd)/data/reasoning_dataset/openai_val.jsonl
WORK=${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/wpjudge-e2e.XXXXXX")}
mkdir -p "$WORK"
LOG=$WORK/e2e.log
PASSN=0 FAILN=0 HOST_PID="" COMPOSE_UP="" ZEROS=00000000000000000000000000000000

note() { echo "$*" | tee -a "$LOG"; }
pass() { PASSN=$((PASSN + 1)); note "PASS $*"; }
fail() { FAILN=$((FAILN + 1)); note "FAIL $*"; }
expect() { local d=$1; shift; if "$@" >/dev/null 2>&1; then pass "$d"; else fail "$d"; fi; }
chk() {  # run a chk.py command; count its PASS/FAIL lines
  local out; out=$("$PY" "$WORK/chk.py" "$@" 2>&1); echo "$out" | tee -a "$LOG"
  PASSN=$((PASSN + $(grep -c '^PASS' <<<"$out"))); FAILN=$((FAILN + $(grep -c '^FAIL' <<<"$out")))
  grep -q Traceback <<<"$out" && fail "chk.py $1 crashed"
  return 0
}

if docker info >/dev/null 2>&1; then dk() { docker "$@"; }
else dk() { sg docker -c "$(printf '%q ' docker "$@")"; }; fi
dc() { dk compose -p wpjudge-e2e -f "$H/docker-compose.yml" -f "$WORK/compose.e2e.yml" "$@"; }

cleanup() {
  [ -n "$HOST_PID" ] && kill "$HOST_PID" 2>/dev/null
  [ -n "$COMPOSE_UP" ] && dc down -v >/dev/null 2>&1
  dk rm -f wpjudge-e2e-aio wpjudge-e2e-parity >/dev/null 2>&1
  note "== total: $PASSN passed, $FAILN failed (log: $LOG)"
}
trap cleanup EXIT

# ---------------------------------------------------------------- host llama-server (IQ4_XS, study build)
host_up() {
  curl -fs "127.0.0.1:$PORT/health" >/dev/null && return 0
  note "nvidia-smi compute apps before start:"; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tee -a "$LOG"
  "$LLAMA_BIN" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -ngl 999 --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' --parallel "$HOST_PARALLEL" -c $((HOST_PARALLEL * 8192)) \
    >"$WORK/llama-host.log" 2>&1 &
  HOST_PID=$!
  for _ in $(seq 300); do curl -fs "127.0.0.1:$PORT/health" >/dev/null && return 0; sleep 2; done
  fail "host llama-server did not become healthy (see $WORK/llama-host.log)"; return 1
}
host_down() {
  [ -z "$HOST_PID" ] && return 0
  kill "$HOST_PID" 2>/dev/null; wait "$HOST_PID" 2>/dev/null; HOST_PID=""
  sleep 3; note "nvidia-smi compute apps after stop: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
}

make_archives() {  # fixture as .zip and .tar.gz with a single top-level dir
  rm -f "$WORK/vuln-plugin.zip" "$WORK/vuln-plugin.tar.gz"
  (cd "$FIXDIR" && zip -qr "$WORK/vuln-plugin.zip" vuln-plugin && tar czf "$WORK/vuln-plugin.tar.gz" vuln-plugin)
}

# ---------------------------------------------------------------- python asserts (written once per run)
cat >"$WORK/chk.py" <<'PYEOF'
import json, statistics, sys, time
from collections import Counter
from pathlib import Path

def ok(c, msg): print(("PASS " if c else "FAIL ") + msg)
def load(p): return json.loads(Path(p).read_text())
def units(r): return sorted(f["id"] for f in r["functions"] if f["status"] != "skipped")
def skips(r): return sorted((f["file"], f["line"], f["name"], f["reason"]) for f in r["functions"] if f["status"] == "skipped")
def byid(r): return {f["id"]: f for f in r["functions"] if f["status"] != "skipped"}
CAP_DIMS, CAP_T, CAP_V, COV_T = ("security", "sql_safety"), 4, 60, 90

def schema(r, tag):
    ok(r.get("schema_version") == "1.0" and {"target", "model", "summary", "dimensions", "anti_patterns", "patterns",
       "worst_functions", "functions"} <= r.keys(), f"{tag}: schema v1 top-level keys")
    s = r["summary"]
    ok({"overall_score", "overall_score_uncapped", "capped", "cap_reasons", "verdict_counts", "coverage"} <= s.keys()
       and {"units", "judged", "unjudged", "skipped", "pct", "low_coverage"} <= s["coverage"].keys(), f"{tag}: summary keys")
    row = {"file", "line", "name", "loc", "truncated", "status", "verdict", "overall_score", "scores", "critique"}
    ok(all(row <= f.keys() and "code" not in f and "raw" not in f for f in r["functions"]), f"{tag}: functions[] rows (no code/raw)")
    ok(all("reason" in f for f in r["functions"] if f["status"] == "skipped")
       and all({"id", "kind", "end_line", "attempts", "error"} <= f.keys() for f in r["functions"] if f["status"] != "skipped"),
       f"{tag}: skip rows carry reason, unit rows carry id/kind/end_line/attempts/error")
    ok(all("examples" not in p for p in r["patterns"]) and all(isinstance(a.get("examples"), list) for a in r["anti_patterns"]),
       f"{tag}: anti_patterns have examples, patterns do not")

def recompute(r, tag):
    """Every summary number from functions[] (independent of wpjudge.aggregate)."""
    fs = r["functions"]; s = r["summary"]; c = s["coverage"]
    J = [f for f in fs if f["status"] == "judged"]; U = [f for f in fs if f["status"] in ("judged", "unjudged")]
    S = [f for f in fs if f["status"] == "skipped"]
    pct = len(J) / len(U) * 100 if U else 0.0
    ok((c["units"], c["judged"], c["unjudged"], c["skipped"]) == (len(U), len(J), len(U) - len(J), len(S))
       and abs(c["pct"] - pct) < 0.06 and c["low_coverage"] == (not J or pct < COV_T), f"{tag}: coverage recomputes")
    caps = {(f["id"], d, f["scores"][d]) for f in J for d in CAP_DIMS
            if isinstance(f["scores"].get(d), (int, float)) and f["scores"][d] <= CAP_T}
    ok(s["capped"] == bool(caps) and {(x["id"], x["dimension"], x["score"]) for x in s["cap_reasons"]} == caps,
       f"{tag}: cap invariant (capped == any security|sql_safety <= 4; cap_reasons exactly those) n={len(caps)}")
    if J:
        w = [max(f["loc"], 1) for f in J]
        unc = sum(f["overall_score"] * x for f, x in zip(J, w)) / sum(w)
        exp = min(unc, CAP_V) if caps else unc
        ok(abs(s["overall_score_uncapped"] - unc) <= 0.051 and abs(s["overall_score"] - exp) <= 0.051,
           f"{tag}: overall {s['overall_score']} (uncapped {s['overall_score_uncapped']}) recomputes to {exp:.1f}")
    else:
        ok(s["overall_score"] is None and s["overall_score_uncapped"] is None, f"{tag}: zero judged -> overall null")
    vc = Counter(f["verdict"] for f in J)
    ok({k: v for k, v in s["verdict_counts"].items() if v} == {k: v for k, v in vc.items() if v}, f"{tag}: verdict_counts recompute")
    dims_ok = True
    for d, st in r["dimensions"].items():
        v = [f["scores"][d] for f in J if isinstance(f["scores"].get(d), (int, float))]
        dims_ok &= bool(v) and st["n_rated"] == len(v) and st["min"] == min(v) and abs(st["mean"] - statistics.fmean(v)) < 0.011 \
            and abs(st["std"] - statistics.pstdev(v)) < 0.011 and abs(st["low_rate"] - sum(x <= 4 for x in v) / len(v)) < 0.0011
    ok(dims_ok and set(r["dimensions"]) == {d for f in J for d, x in f["scores"].items() if isinstance(x, (int, float))},
       f"{tag}: dimensions recompute")
    worst = sorted(J, key=lambda f: (f["overall_score"], f["id"]))[:10]
    ok([(w["file"], w["line"], w["name"]) for w in r["worst_functions"]] == [(f["file"], f["line"], f["name"]) for f in worst],
       f"{tag}: worst_functions = 10 lowest judged")

def fixture_units(r, tag, exp):
    ok(units(r) == sorted(exp["units"]), f"{tag}: units == expected ({len(exp['units'])})")
    ok(skips(r) == sorted((s["file"], s["line"], s["name"], s["reason"]) for s in exp["skips"]), f"{tag}: skip rows == expected")
    ok(Counter(x[3] for x in skips(r)) == Counter(exp["skip_counts"]) and r["summary"]["coverage"]["skipped"] == len(exp["skips"]),
       f"{tag}: skip counts exact {exp['skip_counts']}")
    ok(r["target"]["files_scanned"] == exp["php_files"], f"{tag}: files_scanned {r['target']['files_scanned']} == {exp['php_files']}")

def cmd_fixture(exp, *reports):  # schema + units/skips + recompute for API / library / allinone reports
    exp = load(exp)
    for p in reports:
        r = load(p); tag = Path(p).name
        schema(r, tag); fixture_units(r, tag, exp); recompute(r, tag)
        ok(not r["summary"]["coverage"]["low_coverage"], f"{tag}: coverage {r['summary']['coverage']['pct']}% >= 90")

def same(a, b, msg):  # T5: exact equality on per-unit scores/verdicts, summary score+cap, dimensions
    def key(r):
        return ({i: (f["overall_score"], f["scores"], f["verdict"]) for i, f in byid(r).items()},
                {k: r["summary"][k] for k in ("overall_score", "overall_score_uncapped", "cap_reasons")}, r["dimensions"])
    ka, kb = key(a), key(b)
    diff = sorted(i for i in ka[0].keys() | kb[0].keys() if ka[0].get(i) != kb[0].get(i))
    ok(ka == kb, f"{msg}: exactly equal (units differing {diff}; summary equal {ka[1] == kb[1]}; dimensions equal {ka[2] == kb[2]})")

def cmd_golden(exp, pre, rdir, rzip, rtgz, rw4):
    exp, pre = load(exp), load(pre)
    R = {"dir": load(rdir), "zip": load(rzip), "tgz": load(rtgz)}
    w4 = load(rw4)
    for tag, r in list(R.items()) + [("w4", w4)]:
        schema(r, tag); fixture_units(r, tag, exp); recompute(r, tag)
        ok(not r["summary"]["coverage"]["low_coverage"], f"{tag}: coverage {r['summary']['coverage']['pct']}% >= 90")
    ok(R["dir"]["target"]["files_scanned"] == R["zip"]["target"]["files_scanned"] == R["tgz"]["target"]["files_scanned"],
       f"dir/zip/tgz: files_scanned agree ({R['dir']['target']['files_scanned']})")
    ok(R["zip"]["target"]["sha256"] != "" and R["dir"]["target"]["sha256"] == "", "sha256: archive set, dir empty")
    same(R["dir"], R["zip"], "dir vs zip"); same(R["dir"], R["tgz"], "dir vs tgz"); same(R["dir"], w4, "workers=4 vs workers=1")
    for tag, r in list(R.items()) + [("w4", w4)]:
        m = r["model"]
        ok(m.get("server_slots") == 1 and m.get("reproducible") is True,
           f"{tag}: model.server_slots == 1 and reproducible (got {m.get('server_slots')}, {m.get('reproducible')})")
    g = exp["groups"]
    planted = [i for k in ("sqli", "ajax_no_nonce_no_cap", "unescaped_echo", "unsanitized_option") for i in g[k]]
    pre_by = {u["id"]: u for u in pre["units"]}
    pre_cap = [i for i in g["sqli"] if any(pre_by[i].get(d) is not None and pre_by[i][d] <= CAP_T for d in CAP_DIMS)]
    for tag, r in R.items():  # binding change 3: the workers=4 run gets units/skips/verdicts only (checked above)
        f = byid(r); capped_ids = {x["id"] for x in r["summary"]["cap_reasons"]}
        if pre_cap:  # binding change 2(b): only if the pre-run showed it
            ok(r["summary"]["capped"] and any(i in capped_ids for i in pre_cap),
               f"{tag}: planted SQLi caps (pre-run capped {len(pre_cap)}); capped now: {sorted(i for i in g['sqli'] if i in capped_ids)}")
        bad = [(p, c) for p in planted for c in g["clean"] if f[p]["overall_score"] is not None and f[c]["overall_score"] is not None
               and not f[p]["overall_score"] < f[c]["overall_score"]]
        ok(not bad, f"{tag}: every planted < every clean; planted {[f[p]['overall_score'] for p in planted]} "
                    f"clean {[f[c]['overall_score'] for c in g['clean']]} violations {bad}")
        ok(any(x["dimension"] in CAP_DIMS for x in r["anti_patterns"]), f"{tag}: anti_patterns include security/sql_safety")

def cmd_determinism(rdir, rw4, rw1):  # rw4/rw1: same --parallel 4 server
    g, w4, w1 = load(rdir), load(rw4), load(rw1)
    m = w4["model"]
    ok(m.get("server_slots") == 4 and m.get("workers") == 4 and m.get("reproducible") is False,
       f"determinism w4 on -np 4: server_slots 4, workers 4, reproducible false (got {m})")
    m = w1["model"]
    ok(m.get("server_slots") == 4 and m.get("reproducible") is True, f"determinism w1 on -np 4: reproducible true (got {m})")
    same(g, w1, "determinism: w1 on -np 4 vs golden r_dir (-np 1)")

def cmd_serverdown(exp, rep):
    exp, r = load(exp), load(rep)
    schema(r, "serverdown"); fixture_units(r, "serverdown", exp); recompute(r, "serverdown")
    us = [f for f in r["functions"] if f["status"] != "skipped"]
    ok(all(f["status"] == "unjudged" and f["error"] and f["attempts"] == 2 for f in us), "serverdown: every unit unjudged, error set, 2 attempts")
    ok(r["summary"]["coverage"]["low_coverage"] and r["summary"]["overall_score"] is None and not r["summary"]["capped"],
       "serverdown: low_coverage true, overall null, not capped")

def cmd_real(rep, name, exp_units, kind, wall):
    r = load(rep); tag = name
    schema(r, tag); recompute(r, tag)
    c = r["summary"]["coverage"]
    ok(c["pct"] >= 90 and not c["low_coverage"], f"{tag}: coverage {c['pct']}% >= 90")
    ok(r["target"]["type"] == kind, f"{tag}: target.type == {kind} (got {r['target']['type']})")
    print(f"INFO {tag}: units {c['units']} (plan said {exp_units}), skipped {c['skipped']}, wall {wall}s, "
          f"overall {r['summary']['overall_score']} capped {r['summary']['capped']}")
    if c["units"] != int(exp_units): print(f"WARN {tag}: unit count differs from plan reference {exp_units}")
    by = {}
    for f in r["functions"]:
        if f["status"] == "judged": by.setdefault("template" if f["kind"] == "template" else "function", []).append(f)
    for k, fs in by.items():  # binding change 15: template vs function split
        w = sum(f["loc"] for f in fs)
        print(f"INFO {tag}: {k} n={len(fs)} mean={statistics.fmean(f['overall_score'] for f in fs):.1f} "
              f"loc_weighted={sum(f['overall_score'] * f['loc'] for f in fs) / w:.1f} loc={w}")

def _send(base, msgs):
    from openai import OpenAI
    cli = OpenAI(base_url=base, api_key="none", max_retries=0, timeout=600)
    r = cli.chat.completions.create(model="x", temperature=0, max_tokens=2048, messages=msgs,
                                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""

def _judge_json(raw):
    import re
    m = re.search(r"<judge_output>\s*(\{.*?\})\s*</judge_output>", raw, re.S)
    try: return json.loads(m.group(1)) if m else {}
    except ValueError: return {}

def cmd_parity_run(val, base, out):  # r2's 16 prompts: the first 16 <wp_judge> val rows, sequential (workers=1)
    rows = []
    for line in open(val):
        msgs = [m for m in json.loads(line)["messages"] if m["role"] == "user"]
        if msgs and msgs[0]["content"].startswith("<wp_judge>"): rows.append(msgs[:1])
        if len(rows) == 16: break
    res = []
    for m in rows:
        j = _judge_json(_send(base, m)); res.append({"verdict": j.get("verdict"), "overall_score": j.get("overall_score")})
    Path(out).write_text(json.dumps(res, indent=1))

def cmd_parity_cmp(a, b):
    A, B = load(a), load(b)
    eq = sum(x["verdict"] == y["verdict"] and x["verdict"] is not None for x, y in zip(A, B))
    d = [abs(x["overall_score"] - y["overall_score"]) for x, y in zip(A, B) if x["overall_score"] is not None and y["overall_score"] is not None]
    ok(len(A) == len(B) == 16 and eq == 16, f"parity: verdicts equal {eq}/16")
    ok(len(d) == 16 and max(d) <= 8, f"parity: overall |d| max {max(d) if d else None} <= 8 (mean {statistics.fmean(d) if d else 0:.2f})")

def cmd_prerun(fixture, base, out):  # binding change 2: one live pass over the fixture units
    from wpjudge.extract import collect_units
    exp = load(Path(fixture).parent / "vuln-plugin.expected.json")
    rows = []
    for u in collect_units(fixture)[0]:
        t = time.time(); j = _judge_json(_send(base, [{"role": "user",
            "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{u['code']}\n```"}]))
        rows.append({"id": u["id"], "group": next((k for k, v in exp["groups"].items() if u["id"] in v), "other"),
                     "verdict": j.get("verdict"), "overall_score": j.get("overall_score"), "security": j.get("security"),
                     "sql_safety": j.get("sql_safety"), "scores": {k: v for k, v in j.items() if k not in ("verdict", "overall_score")},
                     "secs": round(time.time() - t, 1), "parsed": bool(j)})
        print("INFO", json.dumps({k: rows[-1][k] for k in ("id", "verdict", "overall_score", "security", "sql_safety")}))
    Path(out).write_text(json.dumps({"server": base, "units": rows}, indent=1) + "\n")

def cmd_make_malicious(d):
    import io, os, tarfile, zipfile
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    def z(name, entries):
        with zipfile.ZipFile(d / name, "w", zipfile.ZIP_DEFLATED) as zf:
            for n, data, *attr in entries:
                zi = zipfile.ZipInfo(n)
                if attr: zi.external_attr = attr[0] << 16
                zf.writestr(zi, data)
    php = b"<?php\nfunction ok_fn() {\n\treturn 1;\n}\n"
    z("slip.zip", [("p/ok.php", php), ("../../wpjudge_e2e_escape_slip.php", php)])
    z("abs.zip", [("p/ok.php", php), ("/tmp/wpjudge_e2e_escape_abs.php", php)])
    z("symlink.zip", [("p/lnk", str(d / "outside"), 0o120777), ("p/lnk/wpjudge_e2e_escape_sym.php", php)])
    with zipfile.ZipFile(d / "bomb.zip", "w", zipfile.ZIP_DEFLATED) as zf:   # 300 MB of zeros, ~300 KB packed
        with zf.open("p/bomb.php", "w", force_zip64=True) as f:
            for _ in range(300): f.write(b"\0" * 2**20)
    with zipfile.ZipFile(d / "ratio.zip", "w", zipfile.ZIP_DEFLATED) as zf:  # 15 MB at ~1000x: under 200 MB, over 100x
        zf.writestr("p/r.php", b"\0" * 15 * 2**20)
    z("oversize.zip", [("p/big.php", os.urandom(2 * 2**20))])                # run with WPJUDGE_MAX_TOTAL_MB=1
    z("manyfiles.zip", [(f"p/f{i}.php", php) for i in range(11)])           # run with WPJUDGE_MAX_FILES=10
    def t(name, members):
        with tarfile.open(d / name, "w:gz") as tf:
            for ti, data in members: tf.addfile(ti, io.BytesIO(data) if data is not None else None)
    def ti(name, size=0, **kw):
        x = tarfile.TarInfo(name); x.size = size
        for k, v in kw.items(): setattr(x, k, v)
        return x
    t("slip.tar.gz", [(ti("p/ok.php", len(php)), php), (ti("../wpjudge_e2e_escape_tslip.php", len(php)), php)])
    t("abs.tar.gz", [(ti("p/ok.php", len(php)), php), (ti("/tmp/wpjudge_e2e_escape_tabs.php", len(php)), php)])
    t("symlink.tar.gz", [(ti("p/lnk", type=tarfile.SYMTYPE, linkname=str(d / "outside")), None),
                         (ti("p/lnk/wpjudge_e2e_escape_tsym.php", len(php)), php)])
    t("hardlink.tar.gz", [(ti("p/ok.php", len(php)), php), (ti("p/pw", type=tarfile.LNKTYPE, linkname="/etc/passwd"), None)])
    (d / "notarchive.zip").write_text("this is not an archive\n")
    (d / "notarchive.tar.gz").write_text("this is not an archive\n")
    (d / "outside").mkdir(exist_ok=True)

getattr(sys.modules[__name__], "cmd_" + sys.argv[1].replace("-", "_"))(*sys.argv[2:])
PYEOF

# ---------------------------------------------------------------- sections
sec_cli() {
  note "== cli"; host_up || return; make_archives
  local t
  for t in dir zip tgz; do
    local src=$FIX; [ $t = zip ] && src=$WORK/vuln-plugin.zip; [ $t = tgz ] && src=$WORK/vuln-plugin.tar.gz
    "$WJ" scan "$src" -o "$WORK/r_$t.json" --workers 1 --base-url "$BASE" --model-file "$MODEL_FILE" 2>"$WORK/r_$t.err"
    expect "cli $t: exit 0" test $? -eq 0
    expect "cli $t: unit count + ETA on stderr" grep -qE '^[0-9]+ units, ETA' "$WORK/r_$t.err"
  done
  "$WJ" scan "$FIX" -o "$WORK/r_w4.json" --workers 4 --base-url "$BASE" --model-file "$MODEL_FILE" 2>/dev/null
  expect "cli dir workers=4: exit 0" test $? -eq 0
  chk golden "$EXP" "$PRE" "$WORK/r_dir.json" "$WORK/r_zip.json" "$WORK/r_tgz.json" "$WORK/r_w4.json"
  expect "cli: report model.file is --model-file" test "$(jq -r .model.file "$WORK/r_dir.json")" = "$MODEL_FILE"
  "$WJ" scan "$FIX" --workers 1 --base-url "$BASE" >"$WORK/r_stdout.json" 2>/dev/null
  expect "cli: no -o writes the report to stdout" jq -e '.schema_version == "1.0"' "$WORK/r_stdout.json"
}

sec_malicious() {
  note "== malicious"
  local M=$WORK/mal; rm -rf "$M"; mkdir -p "$M/tmp"
  chk make-malicious "$M"
  local f
  for f in slip.zip abs.zip symlink.zip bomb.zip ratio.zip oversize.zip manyfiles.zip slip.tar.gz abs.tar.gz symlink.tar.gz \
           hardlink.tar.gz notarchive.zip notarchive.tar.gz; do
    local extra=(); [ $f = oversize.zip ] && extra=(WPJUDGE_MAX_TOTAL_MB=1); [ $f = manyfiles.zip ] && extra=(WPJUDGE_MAX_FILES=10)
    rm -f "$M/out.json"
    env TMPDIR="$M/tmp" "${extra[@]}" "$WJ" scan "$M/$f" -o "$M/out.json" --base-url http://127.0.0.1:9/v1 2>"$M/err"
    local rc=$?
    if [ $rc -eq 2 ] && [ ! -e "$M/out.json" ] && ! grep -q Traceback "$M/err"; then pass "malicious $f: rejected (exit 2): $(head -c 120 "$M/err")"
    else fail "malicious $f: rc=$rc out=$( [ -e "$M/out.json" ] && echo written || echo none) err=$(head -c 300 "$M/err")"; fi
  done
  expect "malicious: temp extraction dirs cleaned up" test -z "$(ls -A "$M/tmp")"
  expect "malicious: nothing escaped (no wpjudge_e2e_escape* under WORK, /tmp, outside/)" \
    test -z "$(find "$WORK" /tmp -maxdepth 4 -name 'wpjudge_e2e_escape*' 2>/dev/null)"
  expect "malicious: symlink target dir untouched" test -z "$(ls -A "$M/outside")"
  # sanity: a plain archive in the same harness still works (guards are not over-eager)
  make_archives
  env TMPDIR="$M/tmp" "$WJ" scan "$WORK/vuln-plugin.zip" -o "$M/ok.json" --base-url http://127.0.0.1:9/v1 2>/dev/null
  expect "malicious: benign fixture zip still accepted" jq -e '.summary.coverage.units == 10' "$M/ok.json"
}

sec_serverdown() {
  note "== serverdown (binding change 8)"
  "$WJ" scan "$FIX" -o "$WORK/r_down.json" --base-url http://127.0.0.1:9/v1 --workers 4 2>/dev/null
  expect "serverdown: exit 0" test $? -eq 0
  chk serverdown "$EXP" "$WORK/r_down.json"
}

sec_library() {
  note "== library"; host_up || return; make_archives
  "$PY" - "$FIX" "$WORK/vuln-plugin.zip" "$BASE" "$MODEL_FILE" "$WORK" <<'EOF'
import json, sys
from wpjudge import scan
fix, z, base, mf, work = sys.argv[1:]
for name, p in (("lib_dir", fix), ("lib_zip", z)):
    r = scan(p, base_url=base, workers=1, model_file=mf)
    open(f"{work}/{name}.json", "w").write(json.dumps(r))
EOF
  expect "library: scan() ran" test -s "$WORK/lib_zip.json"
  chk fixture "$EXP" "$WORK/lib_dir.json" "$WORK/lib_zip.json"
}

sec_determinism() {  # T5: slot count alone does not move scores; concurrent decoding is flagged
  note "== determinism (T5)"
  [ -s "$WORK/r_dir.json" ] || { fail "determinism: $WORK/r_dir.json missing; run the cli section first (same WORK)"; return; }
  host_down
  curl -fs "127.0.0.1:$DET_PORT/health" >/dev/null && { fail "determinism: port $DET_PORT already in use"; return; }
  local base=http://127.0.0.1:$DET_PORT/v1 up=""
  "$LLAMA_BIN" -m "$GGUF" --host 127.0.0.1 --port "$DET_PORT" -ngl 999 --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' --parallel 4 -c $((4 * 8192)) >"$WORK/llama-det.log" 2>&1 &
  HOST_PID=$!  # own server on DET_PORT; HOST_PID so host_down and the EXIT trap stop it
  for _ in $(seq 300); do curl -fs "127.0.0.1:$DET_PORT/health" >/dev/null && up=1 && break; sleep 2; done
  if [ -n "$up" ]; then
    # w1 first: slots reused from a concurrent run keep batch-variant KV (cache_prompt) and can move a w1 score
    "$WJ" scan "$FIX" -o "$WORK/det_w1.json" --workers 1 --base-url "$base" --model-file "$MODEL_FILE" 2>"$WORK/det_w1.err"
    expect "determinism w1: exit 0" test $? -eq 0
    expect "determinism w1: no warning" bash -c "! grep -q WARNING '$WORK/det_w1.err'"
    "$WJ" scan "$FIX" -o "$WORK/det_w4.json" --workers 4 --base-url "$base" --model-file "$MODEL_FILE" 2>"$WORK/det_w4.err"
    expect "determinism w4: exit 0" test $? -eq 0
    expect "determinism w4: stderr warns not reproducible" grep -q "WARNING.*reproducible" "$WORK/det_w4.err"
    chk determinism "$WORK/r_dir.json" "$WORK/det_w4.json" "$WORK/det_w1.json"
  else
    fail "determinism: --parallel 4 server not healthy (see $WORK/llama-det.log)"
  fi
  host_down
}

sec_real() {
  note "== real (binding changes 14, 15)"; host_up || return
  local R=$WORK/real; mkdir -p "$R"
  local spec name url units kind t0 wall
  for spec in "classic-editor|https://downloads.wordpress.org/plugin/classic-editor.latest-stable.zip|39|plugin" \
              "twentytwentyfour|https://downloads.wordpress.org/theme/twentytwentyfour.zip|60|theme"; do
    IFS='|' read -r name url units kind <<<"$spec"
    [ -s "$R/$name.zip" ] || curl -fsSL -o "$R/$name.zip" "$url" || { fail "real $name: download $url"; continue; }
    t0=$(date +%s)
    "$WJ" scan "$R/$name.zip" -o "$R/$name.json" --base-url "$BASE" --model-file "$MODEL_FILE" 2>"$R/$name.err"
    expect "real $name: exit 0" test $? -eq 0
    wall=$(($(date +%s) - t0)); note "INFO real $name: $(cat "$R/$name.err")"
    chk real "$R/$name.json" "$name" "$units" "$kind" "$wall"
  done
}

sec_prerun() {  # regenerate the binding-change-2 pre-run into WORK (copy over the fixture by hand if intended)
  note "== prerun"; host_up || return
  chk prerun "$FIX" "$BASE" "$WORK/vuln-plugin.prerun.json"
  note "INFO prerun written to $WORK/vuln-plugin.prerun.json (committed copy: $PRE)"
}

sec_parity() {
  note "== parity (binding change 13)"
  host_down
  local v; v=$(dk run --rm --gpus all "$IMG" --version 2>&1 | tail -3)
  if grep -qi "version" <<<"$v"; then pass "parity: driver-580 smoke, image runs: $(tr '\n' ' ' <<<"$v")"; else fail "parity: image smoke: $v"; return; fi
  dk rm -f wpjudge-e2e-parity >/dev/null 2>&1
  dk run -d --name wpjudge-e2e-parity --gpus all -v "$GGUF_DIR:/models:ro" -p "127.0.0.1:$PARITY_PORT:8080" "$IMG" \
    -m "/models/$MODEL_FILE" --host 0.0.0.0 --port 8080 -ngl 999 --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' --parallel 1 -c 8192 >/dev/null
  local up=""; for _ in $(seq 300); do curl -fs "127.0.0.1:$PARITY_PORT/health" >/dev/null && up=1 && break; sleep 2; done
  if [ -z "$up" ]; then fail "parity: container server not healthy"; dk logs wpjudge-e2e-parity 2>&1 | tail -20 | tee -a "$LOG"; dk rm -f wpjudge-e2e-parity >/dev/null; return; fi
  chk parity-run "$VAL" "http://127.0.0.1:$PARITY_PORT/v1" "$WORK/parity_container.json"
  dk rm -f wpjudge-e2e-parity >/dev/null
  host_up || return
  chk parity-run "$VAL" "$BASE" "$WORK/parity_host.json"
  chk parity-cmp "$WORK/parity_host.json" "$WORK/parity_container.json"
}

# --- API helpers
API=""
code() { curl -s -o "$WORK/body" -w '%{http_code}' "$@"; }
api_wait() { for _ in $(seq 600); do curl -s -o /dev/null -w '%{http_code}' "$API/jobs/$ZEROS" | grep -qE '^(401|404)$' && return 0; sleep 2; done; return 1; }
submit() {  # submit <file> [extra curl args...]; echoes job_id
  local f=$1; shift
  curl -s "$@" -F "file=@$f" "$API/jobs" | jq -r '.job_id // empty'
}
poll() {  # poll <id> [curl auth args...]; echoes final status
  local id=$1 st end=$(($(date +%s) + JOB_TIMEOUT)); shift
  while [ "$(date +%s)" -lt $end ]; do
    st=$(curl -s "$@" "$API/jobs/$id" | jq -r .status)
    case $st in done | failed) echo "$st"; return;; esac; sleep 3
  done; echo timeout
}
compose_override() {  # $1 = api key ('' = auth off); GGUF bind-mounted read-only: no 16.6 GB download
  cat >"$WORK/compose.e2e.yml" <<EOF
services:
  llama:
    environment:
      WPJUDGE_MODEL_FILE: "$MODEL_FILE"
    volumes:
      - "$GGUF_DIR:/models:ro"
  harness:
    ports: !override
      - "127.0.0.1:$API_PORT:8000"
    environment:
      WPJUDGE_MODEL_FILE: "$MODEL_FILE"
      WPJUDGE_API_KEY: "$1"
EOF
}

sec_api() {
  note "== api (docker compose)"; host_down; make_archives
  API=http://127.0.0.1:$API_PORT; KEY=""
  compose_override ""; COMPOSE_UP=1
  dc up -d --build >>"$LOG" 2>&1 || { fail "api: compose up"; return; }
  api_wait || { fail "api: harness not answering on $API"; dc logs --tail 40 | tee -a "$LOG"; return; }
  pass "api: stack up (auth off)"

  # webhook listener inside the harness container (no host networking assumptions)
  cat >"$WORK/hook.py" <<'EOF'
import http.server
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        open("/tmp/hook.json", "wb").write(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.end_headers()
http.server.HTTPServer(("127.0.0.1", 8766), H).serve_forever()
EOF
  dc cp "$WORK/hook.py" harness:/tmp/hook.py >/dev/null && dc exec -d harness python /tmp/hook.py

  local J1 st c
  J1=$(submit "$WORK/vuln-plugin.zip" -F "webhook_url=http://127.0.0.1:8766/hook")
  expect "api: POST /jobs (auth off) -> job_id" test -n "$J1"
  local seen=""  # partial results while running
  for _ in $(seq 400); do
    curl -s "$API/jobs/$J1" >"$WORK/st.json"; st=$(jq -r .status "$WORK/st.json")
    if [ "$st" = running ] && [ "$(jq .progress.done "$WORK/st.json")" -ge 1 ]; then
      c=$(code "$API/jobs/$J1/functions"); local n; n=$(jq '.functions | length' "$WORK/body")
      [ "$c" = 200 ] && [ "$n" -ge 1 ] && [ "$(jq 'has("report")' "$WORK/st.json")" = false ] && \
        ! jq -e '.functions[] | select(has("code") or has("raw"))' "$WORK/body" >/dev/null && seen=$n
      break
    fi
    case $st in done | failed) break;; esac; sleep 1
  done
  if [ -n "$seen" ]; then pass "api: partial /functions while running ($seen rows, no report yet, no code/raw)"; else fail "api: partial /functions not observed while running (phase 5 requires it)"; fi
  st=$(poll "$J1"); expect "api: job done (got $st)" test "$st" = done
  curl -s "$API/jobs/$J1" | jq .report >"$WORK/api_report.json"
  chk fixture "$EXP" "$WORK/api_report.json"
  c=$(code "$API/jobs/$J1/functions"); expect "api: /functions has all 10 units" jq -e '.functions | length == 10' "$WORK/body"
  sleep 3; dc exec -T harness cat /tmp/hook.json >"$WORK/hook.json" 2>/dev/null
  expect "api: webhook delivered the report to the local listener" jq -e '.schema_version == "1.0" and (.functions | length) == 15' "$WORK/hook.json"

  # input guards
  printf 'x' >"$WORK/x.py"; c=$(code -F "file=@$WORK/x.py" "$API/jobs"); expect "api: .py upload -> 415 (got $c)" test "$c" = 415
  head -c $((60 * 2**20)) /dev/urandom >"$WORK/big.zip"
  c=$(code -F "file=@$WORK/big.zip" "$API/jobs"); expect "api: 60 MB upload -> 413 (got $c)" test "$c" = 413
  c=$(code -F "file=@$WORK/vuln-plugin.zip" -F "webhook_url=file:///etc/passwd" "$API/jobs"); expect "api: file:// webhook -> 400 (got $c)" test "$c" = 400
  c=$(code -H "Content-Length: abc" -X POST "$API/jobs"); expect "api: non-numeric Content-Length -> 400 (got $c)" test "$c" = 400
  c=$(code --path-as-is "$API/jobs/..%2f../functions"); expect "api: /jobs/..%2f../functions -> 4xx not 200/500 (got $c)" test "$c" -ge 400 -a "$c" -lt 500
  c=$(code --path-as-is "$API/jobs/..%2f..%2fetc%2fpasswd"); expect "api: /jobs/..%2f..%2fetc%2fpasswd -> 4xx (got $c)" test "$c" -ge 400 -a "$c" -lt 500
  # malicious archive through the API: job fails cleanly, nothing escapes in the container
  [ -s "$WORK/mal/slip.zip" ] || chk make-malicious "$WORK/mal"
  local JM; JM=$(submit "$WORK/mal/slip.zip"); st=$(poll "$JM")
  expect "api: zip-slip job -> failed with error (got $st: $(curl -s "$API/jobs/$JM" | jq -r .error))" test "$st" = failed
  expect "api: no escaped file inside the container" test -z "$(dc exec -T harness find / /data/jobs -xdev -name 'wpjudge_e2e_escape*' 2>/dev/null)"

  # auth on: same job store, restart with a key
  KEY=e2e-$RANDOM-key; compose_override "$KEY"
  dc up -d harness >>"$LOG" 2>&1; api_wait || { fail "api: harness not back after enabling auth"; return; }
  local A=(-H "Authorization: Bearer $KEY") W=(-H "Authorization: Bearer wrong")
  c=$(code "$API/jobs/$J1"); expect "api auth: GET /jobs/{id} no key -> 401 (got $c)" test "$c" = 401
  c=$(code "${W[@]}" "$API/jobs/$J1"); expect "api auth: wrong key -> 401 (got $c)" test "$c" = 401
  c=$(code "$API/jobs/$J1/functions"); expect "api auth: /functions no key -> 401 (got $c)" test "$c" = 401
  c=$(code "${W[@]}" "$API/jobs/$J1/functions"); expect "api auth: /functions wrong key -> 401 (got $c)" test "$c" = 401
  c=$(code -F "file=@$WORK/vuln-plugin.zip" "$API/jobs"); expect "api auth: POST no key -> 401 (got $c)" test "$c" = 401
  c=$(code "${A[@]}" "$API/jobs/$J1"); expect "api auth: right key -> 200 (got $c)" test "$c" = 200
  expect "api restart: done job survived the restart (result on disk)" jq -e '.status == "done" and .report.schema_version == "1.0"' "$WORK/body"
  c=$(code "${A[@]}" "$API/jobs/$J1/functions"); expect "api auth: /functions right key -> 200 (got $c)" test "$c" = 200

  # restart mid-run: running job -> failed "interrupted"
  local J2; J2=$(submit "$WORK/vuln-plugin.zip" "${A[@]}")
  for _ in $(seq 120); do [ "$(curl -s "${A[@]}" "$API/jobs/$J2" | jq -r .status)" = running ] && break; sleep 1; done
  dc restart harness >>"$LOG" 2>&1; api_wait
  curl -s "${A[@]}" "$API/jobs/$J2" >"$WORK/j2.json"
  expect "api restart: running job -> failed 'interrupted' ($(jq -c '{status,error}' "$WORK/j2.json"))" \
    jq -e '.status == "failed" and (.error | test("interrupted"))' "$WORK/j2.json"

  # retention: dir mtime 8 days ago is swept at startup, 6 days ago is kept
  dc exec -T harness sh -c "touch -d '8 days ago' /data/jobs/$J1 && touch -d '6 days ago' /data/jobs/$J2"
  dc restart harness >>"$LOG" 2>&1; api_wait
  c=$(code "${A[@]}" "$API/jobs/$J1"); expect "api retention: 8-day-old job swept -> 404 (got $c)" test "$c" = 404
  c=$(code "${A[@]}" "$API/jobs/$J2"); expect "api retention: 6-day-old job kept -> 200 (got $c)" test "$c" = 200
  dc down -v >>"$LOG" 2>&1; COMPOSE_UP=""
}

sec_allinone() {
  note "== allinone"; host_down; make_archives
  dk build --target allinone -t wpjudge:allinone "$H" >>"$LOG" 2>&1 || { fail "allinone: build"; return; }
  dk rm -f wpjudge-e2e-aio >/dev/null 2>&1
  dk run -d --name wpjudge-e2e-aio --gpus all -v "$GGUF_DIR:/models:ro" -e "WPJUDGE_MODEL_FILE=$MODEL_FILE" \
    -p "127.0.0.1:$AIO_PORT:8000" wpjudge:allinone >/dev/null || { fail "allinone: docker run"; return; }
  API=http://127.0.0.1:$AIO_PORT
  api_wait || { fail "allinone: API not answering"; dk logs wpjudge-e2e-aio 2>&1 | tail -30 | tee -a "$LOG"; return; }
  local J st; J=$(submit "$WORK/vuln-plugin.zip"); st=$(poll "$J")
  expect "allinone: fixture scan done (got $st)" test "$st" = done
  curl -s "$API/jobs/$J" | jq .report >"$WORK/aio_report.json"
  chk fixture "$EXP" "$WORK/aio_report.json"
  dk rm -f wpjudge-e2e-aio >/dev/null
}

[ $# -eq 0 ] && { sed -n '2,6p' "$0"; exit 2; }
[ "$1" = all ] && set -- malicious serverdown cli library real determinism parity api allinone
note "== wpjudge e2e $(date -Is) WORK=$WORK model=$MODEL_FILE"
for s in "$@"; do
  case $s in
    cli | malicious | serverdown | api | allinone | library | parity | real | determinism | prerun) "sec_$s";;
    *) fail "unknown section $s";;
  esac
done
host_down
[ "$FAILN" -eq 0 ]
