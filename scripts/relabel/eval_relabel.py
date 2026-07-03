#!/usr/bin/env python3
"""Eval the relabel-SFT model vs NEW val labels (08.2-RELABEL, eval side).

Mirrors student_gap.py exactly (same index mapping, same parser, same ceiling)
but reads an arbitrary capture instead of the warmstart one, and prints the
delta against the pre-training student baseline (student_gap.json).

Usage:
  .venv-tinker/bin/python scripts/relabel/eval_relabel.py \
      output/relabel/eval_relabel_v1/judge_responses.jsonl
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

CAP = sys.argv[1] if len(sys.argv) > 1 else "output/relabel/eval_relabel_v1/judge_responses.jsonl"

dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}

rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
wj_rows = [i for i, r in enumerate(rows)
           if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]

model = {}
n_parse_fail = 0
for line in open(CAP):
    r = json.loads(line)
    if "index" not in r:
        continue
    parsed = parse_judge_scores(r["response"], "auto")
    if not parsed or not parsed.get("dimension_scores"):
        n_parse_fail += 1
        continue
    o = float(parsed["overall"]) if "overall" in parsed else _derive_prose_overall(parsed["dimension_scores"], dw)
    model[f"val:{wj_rows[r['index']]}"] = o

new = {k: v for k, v in json.load(open("output/relabel/val_labels_v1.json")).items() if k.startswith("val:")}
common = sorted(set(model) & set(new))
s = [model[i] for i in common]
nl = [new[i] for i in common]
r_new = spearmanr(s, nl).statistic

rel3 = json.load(open("output/relabel/pilot_qc.json"))["rel_M3"]
ceil = rel3 ** 0.5

rng = np.random.default_rng(7)
n = len(common)
boots = sorted(spearmanr(np.array(s)[idx], np.array(nl)[idx]).statistic
               for idx in (rng.integers(0, n, n) for _ in range(2000)))

base = json.load(open("output/relabel/student_gap.json"))
b = base["rho_student_new"]
print(f"capture: {CAP}  (parse_fail={n_parse_fail})")
print(f"relabel-SFT vs NEW labels: rho = {r_new:.4f} (n={n})  CI [{boots[50]:.4f}, {boots[1949]:.4f}]")
print(f"baseline v1.2 student   : rho = {b:.4f}")
print(f"ceiling = sqrt({rel3:.3f}) = {ceil:.4f}")
print(f"Δ vs baseline = {r_new - b:+.4f}   gap-to-ceiling now = {ceil - r_new:+.4f} (was {ceil - b:+.4f})")

json.dump({"capture": CAP, "rho_new": r_new, "n": n, "ci": [boots[50], boots[1949]],
           "baseline_rho": b, "delta_vs_baseline": r_new - b,
           "ceiling": ceil, "gap_to_ceiling": ceil - r_new, "parse_fail": n_parse_fail},
          open("output/relabel/eval_relabel_v1.json", "w"), indent=2)
print("wrote output/relabel/eval_relabel_v1.json")
