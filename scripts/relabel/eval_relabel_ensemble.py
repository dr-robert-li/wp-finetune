#!/usr/bin/env python3
"""Ensemble eval: median of N seed captures per item vs NEW val labels.

Mirrors eval_relabel.py's parse + index join exactly, then medians the per-item
overall across the seed captures (require >=2 seeds parsed for an item) before
Spearman. Usage:
  python scripts/relabel/eval_relabel_ensemble.py OUT_JSON cap_s0.jsonl cap_s1.jsonl cap_s2.jsonl
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

OUT = sys.argv[1]
CAPS = sys.argv[2:]
assert CAPS, "need >=1 capture path"

dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}

rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
wj_rows = [i for i, r in enumerate(rows)
           if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]


def parse_capture(path):
    d = {}
    pf = 0
    for line in open(path):
        r = json.loads(line)
        if "index" not in r:
            continue
        parsed = parse_judge_scores(r["response"], "auto")
        if not parsed or not parsed.get("dimension_scores"):
            pf += 1
            continue
        o = float(parsed["overall"]) if "overall" in parsed else _derive_prose_overall(parsed["dimension_scores"], dw)
        d[f"val:{wj_rows[r['index']]}"] = o
    return d, pf


per_seed = []
pf_per_seed = []
for c in CAPS:
    d, pf = parse_capture(c)
    per_seed.append(d)
    pf_per_seed.append(pf)

new = {k: v for k, v in json.load(open("output/relabel/val_labels_v1.json")).items() if k.startswith("val:")}

# median ensemble: for each label key, gather seed overalls that parsed; require >=2
ens = {}
min_seeds = 2 if len(CAPS) >= 2 else 1
for k in new:
    vals = [d[k] for d in per_seed if k in d]
    if len(vals) >= min_seeds:
        ens[k] = float(np.median(vals))

common = sorted(set(ens) & set(new))
s = [ens[i] for i in common]
nl = [new[i] for i in common]
r_new = spearmanr(s, nl).statistic

rel3 = json.load(open("output/relabel/pilot_qc.json"))["rel_M3"]
ceil = rel3 ** 0.5
rng = np.random.default_rng(7)
n = len(common)
boots = sorted(spearmanr(np.array(s)[idx], np.array(nl)[idx]).statistic
               for idx in (rng.integers(0, n, n) for _ in range(2000)))
b = json.load(open("output/relabel/student_gap.json"))["rho_student_new"]

print(f"ENSEMBLE of {len(CAPS)} seeds: {CAPS}")
print(f"parse_fail per seed: {pf_per_seed}")
print(f"ensemble rho = {r_new:.4f} (n={n})  CI [{boots[50]:.4f}, {boots[1949]:.4f}]")
print(f"baseline v1.2 student = {b:.4f}   ceiling = {ceil:.4f}   Δ = {r_new - b:+.4f}")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump({"captures": CAPS, "ensemble_rho": r_new, "n": n, "ci": [boots[50], boots[1949]],
           "parse_fail_per_seed": pf_per_seed, "baseline_rho": b, "ceiling": ceil,
           "delta_vs_baseline": r_new - b, "min_seeds_required": min_seeds},
          open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
