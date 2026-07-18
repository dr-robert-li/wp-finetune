#!/usr/bin/env python3
"""G1 read — paired judge-rho delta of an RL checkpoint vs the v1.3 warmstart.

Paired item-level bootstrap (B3: point bars are noise; the gate is CI-based):
  PASS       : paired-delta CI-lower > 0 (checkpoint genuinely above warmstart)
  KILL-NEG   : paired-delta CI-upper < 0 (genuinely below)
  INCONCLUSIVE otherwise (expected mid-smoke; decide on trend + later reads)

Usage: g1_read.py <ckpt_capture.jsonl> <tag>
"""
import json
import os
import sys

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
import numpy as np
from scipy.stats import spearmanr

from eval.eval_judge import _derive_prose_overall
from eval.output_parsers import load_dim_map, parse_judge_scores

CAP = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "ckpt"
WARM = "output/relabel/eval_s1_ep3/judge_responses.jsonl"  # v1.3 (rho 0.827)

dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
wj = [i for i, r in enumerate(rows)
      if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
labels = {k: v for k, v in json.load(open("output/relabel/val_labels_v1.json")).items()
          if k.startswith("val:")}


def load(path):
    d, fail = {}, 0
    for line in open(path):
        r = json.loads(line)
        if "index" not in r:
            continue
        p = parse_judge_scores(r["response"], "auto")
        if not p or not p.get("dimension_scores"):
            fail += 1
            continue
        o = float(p["overall"]) if "overall" in p else _derive_prose_overall(p["dimension_scores"], dw)
        d[f"val:{wj[r['index']]}"] = o
    return d, fail


ck, ck_fail = load(CAP)
wm, _ = load(WARM)
common = sorted(set(ck) & set(wm) & set(labels))
n = len(common)
C = np.array([ck[i] for i in common])
W = np.array([wm[i] for i in common])
L = np.array([labels[i] for i in common])

r_ck = spearmanr(C, L).statistic
r_wm = spearmanr(W, L).statistic
rng = np.random.default_rng(31)
ds = sorted(spearmanr(C[ix], L[ix]).statistic - spearmanr(W[ix], L[ix]).statistic
            for ix in (rng.integers(0, n, n) for _ in range(2000)))
lo, hi = ds[50], ds[1949]
verdict = "PASS" if lo > 0 else ("KILL-NEG" if hi < 0 else "INCONCLUSIVE")

print(f"[G1:{TAG}] n={n} parse_fail={ck_fail}")
print(f"  ckpt rho={r_ck:.4f}  warmstart rho={r_wm:.4f}  Δ={r_ck - r_wm:+.4f}")
print(f"  paired Δ CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}")

out = f"output/rl_eval/g1_{TAG}.json"
json.dump({"tag": TAG, "capture": CAP, "n": n, "parse_fail": ck_fail,
           "rho_ckpt": r_ck, "rho_warmstart": r_wm, "delta": r_ck - r_wm,
           "delta_ci": [lo, hi], "verdict": verdict}, open(out, "w"), indent=2)
print(f"wrote {out}")
