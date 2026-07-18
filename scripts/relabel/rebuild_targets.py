#!/usr/bin/env python3
"""Rebuild judge SFT targets from the relabel campaign (08.2-RELABEL, train side).

For each openai_train.jsonl wp_judge row whose relabel item exists:
  - replace the <judge_output> JSON with median-aggregated NEW scores
    (per-dim median from output/relabel/results/*, verdict = majority,
     overall_score = labels.json median — the tracked authority),
  - patch the inline "Header: score N/10" CoT numbers to the same new dims,
    so the reasoning prose stays consistent with the emitted JSON.
Rows without a relabel or without a <judge_output> block are passed through
unchanged. Writes data/reasoning_dataset/openai_train_relabel_v1.jsonl.

Self-check at the bottom: every rebuilt row's inline dim numbers equal its JSON,
and its JSON overall_score equals labels.json.
"""
import glob
import json
import re
from collections import Counter, defaultdict
from statistics import median

TRAIN = "data/reasoning_dataset/openai_train.jsonl"
OUT = "data/reasoning_dataset/openai_train_relabel_v1.jsonl"
LABELS = "data/relabel_v1/labels.json"

# inline CoT header -> judge_output JSON key
HEAD2KEY = {
    "WPCS Compliance": "wpcs_compliance",
    "SQL Safety": "sql_safety",
    "Security": "security",
    "Performance": "performance",
    "WP API Usage": "wp_api_usage",
    "Code Quality": "code_quality",
    "Dependency Integrity": "dependency_integrity",
    "i18n": "i18n",
    "Accessibility": "accessibility",
    "Error Handling": "error_handling",
}
DIM_KEYS = set(HEAD2KEY.values())
JO_RE = re.compile(r"<judge_output>\s*(\{.*?\})\s*</judge_output>", re.S)


def median_judge_from_results():
    """train:ridx -> {dim: median_int, ..., _verdict: majority}."""
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob("output/relabel/results/*.json")):
        try:
            rows = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for e in rows:
            iid = e.get("id", "")
            if not iid.startswith("train:"):
                continue
            j = e.get("judge") or {}
            for k, v in j.items():
                if k in DIM_KEYS and isinstance(v, (int, float)):
                    acc[iid][k].append(float(v))
                elif k == "verdict" and v in ("PASS", "FAIL"):
                    acc[iid]["_verdict"].append(v)
    out = {}
    for iid, dims in acc.items():
        verdicts = dims.pop("_verdict", [])
        m = {k: int(round(median(vs))) for k, vs in dims.items() if vs}
        if verdicts:
            m["_verdict"] = Counter(verdicts).most_common(1)[0][0]
        out[iid] = m
    return out


def rebuild():
    labels = json.load(open(LABELS))
    med = median_judge_from_results()
    rows = [json.loads(l) for l in open(TRAIN) if l.strip()]

    n_rebuilt = n_no_label = n_no_block = 0
    checks = []  # (iid, ok_inline, ok_overall)
    with open(OUT, "w") as fh:
        for ridx, r in enumerate(rows):
            iid = f"train:{ridx}"
            lab = labels.get(iid)
            content = r["messages"][1]["content"]
            m = JO_RE.search(content)
            if lab is None or iid not in med:
                n_no_label += 1
                fh.write(json.dumps(r) + "\n")
                continue
            if not m:
                n_no_block += 1
                fh.write(json.dumps(r) + "\n")
                continue

            new_dims = {k: v for k, v in med[iid].items() if k in DIM_KEYS}
            overall = int(round(lab["overall"]))
            verdict = med[iid].get("_verdict") or ("PASS" if overall >= 65 else "FAIL")

            # rebuild <judge_output> JSON (verdict first, dims, overall last — match style)
            jo = {"verdict": verdict, **new_dims, "overall_score": overall}
            jo_str = json.dumps(jo, indent=2)
            content = content[:m.start()] + f"<judge_output>\n{jo_str}\n</judge_output>" + content[m.end():]

            # patch inline "Header: score N/10" prose numbers to the new dims
            def _patch(mo):
                head = mo.group(1).strip()
                key = HEAD2KEY.get(head)
                if key in new_dims:
                    return f"{mo.group(1)}: score {new_dims[key]}/10"
                return mo.group(0)  # unknown/omitted dim -> leave as-is
            content = re.sub(r"(?m)^([A-Za-z0-9 /_]+?): score (\S+)/10", _patch, content)

            r["messages"][1]["content"] = content
            fh.write(json.dumps(r) + "\n")
            n_rebuilt += 1

            # collect self-check data
            inline = dict(re.findall(r"(?m)^([A-Za-z0-9 /_]+?): score (\S+)/10", content))
            ok_inline = all(
                str(new_dims[HEAD2KEY[h]]) == inline.get(h)
                for h in inline if HEAD2KEY.get(h) in new_dims
            )
            jo_back = json.loads(JO_RE.search(content).group(1))
            checks.append((iid, ok_inline, jo_back["overall_score"] == overall))

    print(f"rebuilt={n_rebuilt}  passthrough(no relabel)={n_no_label}  "
          f"relabeled-but-no-judge_output={n_no_block}  total={len(rows)}")
    bad_inline = [i for i, oi, _ in checks if not oi]
    bad_overall = [i for i, _, oo in checks if not oo]
    assert not bad_overall, f"overall mismatch: {bad_overall[:5]}"
    assert not bad_inline, f"inline/JSON dim mismatch: {bad_inline[:5]}"
    print(f"SELF-CHECK OK: {len(checks)} rebuilt rows internally consistent "
          f"(inline dims == JSON dims, JSON overall == labels.json)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    rebuild()
