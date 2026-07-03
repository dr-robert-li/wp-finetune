#!/usr/bin/env python3
"""A6 hold-out check — aggregate.

Median labels from the M=3 passes over 30 double-fresh items (never in v1.3's
training data, never seen by the rubric instrument), sentinel drift check, then
Spearman(v1.3 holdout capture overall, median holdout labels). Context bar:
v1.3 scored 0.827 on the relabel val set; a hold-out rho in the same region
(CI overlapping) says that number is not rubric-instrument overfit. n=30 →
wide CI; this is a sanity check, not a precision estimate.
"""
import json
import os
import sys
from statistics import median

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
import numpy as np
from scipy.stats import spearmanr

from eval.eval_judge import _derive_prose_overall
from eval.output_parsers import load_dim_map, parse_judge_scores

OUT = "output/relabel"

obs, sent = {}, {}
for p in (1, 2, 3):
    for e in json.load(open(f"{OUT}/results_holdout/pass_{p}.json")):
        iid = e["id"]
        o = (e.get("judge") or {}).get("overall_score")
        if not isinstance(o, (int, float)):
            continue
        (sent if iid.startswith("SENTINEL::") else obs).setdefault(iid, []).append(float(o))

labels = {i: median(v) for i, v in obs.items() if len(v) >= 2}
camp = json.load(open("data/relabel_v1/labels.json"))
drift = {s[10:]: (median(v), camp[s[10:]]["overall"]) for s, v in sent.items()}
print(f"holdout labels: {len(labels)} | sentinel (holdout vs campaign): "
      f"{ {k: v for k, v in drift.items()} }")

# inter-pass reliability on holdout
import itertools
pass_scores = [{e['id']: (e.get('judge') or {}).get('overall_score')
                for e in json.load(open(f'{OUT}/results_holdout/pass_{p}.json'))
                if not e['id'].startswith('SENTINEL')} for p in (1, 2, 3)]
prs = []
ids = sorted(set.intersection(*(set(p) for p in pass_scores)))
for a, b in itertools.combinations(range(3), 2):
    prs.append(spearmanr([pass_scores[a][i] for i in ids], [pass_scores[b][i] for i in ids]).statistic)
r1 = sum(prs) / len(prs)
print(f"holdout inter-pass r (mean pairwise) = {r1:.4f} (campaign was 0.9125)")

# v1.3 predictions
dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
pred = {}
for line in open(f"{OUT}/eval_holdout_v13/judge_responses.jsonl"):
    r = json.loads(line)
    if "index" not in r:
        continue
    p = parse_judge_scores(r["response"], "auto")
    if not p or not p.get("dimension_scores"):
        continue
    o = float(p["overall"]) if "overall" in p else _derive_prose_overall(p["dimension_scores"], dw)
    pred[f"holdout:{r['index']}"] = o

common = sorted(set(pred) & set(labels))
s = np.array([pred[i] for i in common])
L = np.array([labels[i] for i in common])
rho = spearmanr(s, L).statistic
rng = np.random.default_rng(29)
n = len(common)
boots = sorted(spearmanr(s[ix], L[ix]).statistic for ix in (rng.integers(0, n, n) for _ in range(2000)))
print(f"\nv1.3 vs HOLD-OUT labels: rho = {rho:.4f} (n={n})  CI [{boots[50]:.4f}, {boots[1949]:.4f}]")
print(f"context: v1.3 on relabel val = 0.827; overfit red-flag if holdout CI-upper << that")

json.dump({"n": n, "rho": rho, "ci": [boots[50], boots[1949]],
           "inter_pass_r": r1, "sentinel_drift": drift,
           "labels": labels},
          open(f"{OUT}/holdout_check.json", "w"), indent=2)
print(f"wrote {OUT}/holdout_check.json")
