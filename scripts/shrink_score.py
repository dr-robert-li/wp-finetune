"""Shrink study scorer: compare one variant's judge capture against the baseline capture.

Paired metrics (same 121 wp_judge prompts, greedy decode):
  agree_rho     Spearman of overall score, variant vs baseline (fidelity to the shipped model)
  mean_abs_diff mean |overall_variant - overall_baseline|
  verdict_agree share of items with the same PASS/FAIL verdict
  parse_fail    responses with no parseable judge output
  not_json      parsed only by the prose fallback (no <judge_output> JSON, e.g. truncated)
Reference metric:
  ref_rho       Spearman of overall score vs the assistant label stored in openai_val.jsonl,
                with a seeded 2000-sample bootstrap CI. (The relabel-v1 median labels behind the
                published 0.80 rho are not on this machine, so ref_rho is not comparable to it.)

Usage: python3 scripts/shrink_score.py <variant_capture.jsonl> <baseline_capture.jsonl> <out.json>
       python3 scripts/shrink_score.py --self-check
"""
import json
import re
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from eval.eval_judge import _derive_prose_overall  # noqa: E402
from eval.output_parsers import load_dim_map, parse_judge_scores  # noqa: E402

VAL = "data/reasoning_dataset/openai_val.jsonl"
_VERDICT = re.compile(r'"verdict"\s*:\s*"(PASS|FAIL)"')


def _weights():
    return {k: v for k, v in load_dim_map()["dimension_weights"].items() if not k.startswith("_")}


def load_capture(path, dw):
    """index -> (overall or None, verdict or None)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "index" not in r:
            continue
        p = parse_judge_scores(r["response"], "auto")
        ov, fmt = None, None
        if p and "overall" in p:
            ov, fmt = float(p["overall"]), p.get("_format")
        elif p and p.get("dimension_scores"):
            ov, fmt = _derive_prose_overall(p["dimension_scores"], dw), p.get("_format")
        m = _VERDICT.search(r["response"] or "")
        out[r["index"]] = (ov, m.group(1) if m else None, fmt)
    return out


def reference_labels(dw):
    rows = [json.loads(l) for l in open(VAL) if l.strip()]
    judge = [r for r in rows if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    ref = {}
    for i, r in enumerate(judge):
        p = parse_judge_scores(r["messages"][-1]["content"], "auto")
        if p and p.get("dimension_scores"):
            ref[i] = float(p["overall"]) if "overall" in p else _derive_prose_overall(p["dimension_scores"], dw)
    return ref


def boot_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    vals = [spearmanr(a[i], b[i]).statistic for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def score(var, base, ref):
    parse_fail = sum(1 for v in var.values() if v[0] is None)
    not_json = sum(1 for v in var.values() if v[0] is not None and len(v) > 2 and v[2] != "json")
    both = sorted(i for i in var if var[i][0] is not None and base.get(i, (None,))[0] is not None)
    v = [var[i][0] for i in both]
    b = [base[i][0] for i in both]
    vb = [i for i in var if var[i][1] and base.get(i, (None, None))[1]]
    rc = sorted(i for i in var if var[i][0] is not None and i in ref)
    rv, rr = [var[i][0] for i in rc], [ref[i] for i in rc]
    return {
        "n": len(var), "parse_fail": parse_fail, "not_json": not_json, "n_paired": len(both),
        "agree_rho": float(spearmanr(v, b).statistic) if len(both) > 2 else None,
        "mean_abs_diff": float(np.mean(np.abs(np.array(v) - np.array(b)))) if both else None,
        "exact_match": float(np.mean([x == y for x, y in zip(v, b)])) if both else None,
        "verdict_agree": float(np.mean([var[i][1] == base[i][1] for i in vb])) if vb else None,
        "ref_rho": float(spearmanr(rv, rr).statistic) if len(rc) > 2 else None,
        "ref_ci": boot_ci(rv, rr) if len(rc) > 2 else None, "n_ref": len(rc),
    }


def _self_check():
    base = {i: (float(i), "PASS") for i in range(10)}
    ref = {i: float(i) for i in range(10)}
    s = score(dict(base), base, ref)
    assert abs(s["agree_rho"] - 1) < 1e-9 and s["mean_abs_diff"] == 0 and s["verdict_agree"] == 1.0 and s["parse_fail"] == 0
    var = dict(base); var[3] = (None, None); var[4] = (99.0, "FAIL")
    s = score(var, base, ref)
    assert s["parse_fail"] == 1 and s["n_paired"] == 9 and s["verdict_agree"] < 1 and s["mean_abs_diff"] > 0
    print("self-check OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check(); sys.exit()
    dw = _weights()
    res = score(load_capture(sys.argv[1], dw), load_capture(sys.argv[2], dw), reference_labels(dw))
    res.update(capture=sys.argv[1], baseline=sys.argv[2])
    json.dump(res, open(sys.argv[3], "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("parse_fail", "not_json", "agree_rho", "mean_abs_diff", "verdict_agree", "ref_rho")}))
