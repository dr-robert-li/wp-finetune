#!/usr/bin/env python3
"""Student TRUE-gap read: warmstart capture vs NEW val labels (relabel v1)."""
import json, os, sys
sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
from scipy.stats import spearmanr
import numpy as np
from eval.output_parsers import parse_judge_scores, load_dim_map
from eval.eval_judge import _derive_prose_overall

dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}

# Map: capture index k -> dataset row index of the k-th wp_judge row
rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
wj_rows = [i for i, r in enumerate(rows)
           if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]

student = {}
for line in open("output/rl_eval/warmstart/judge_responses.jsonl"):
    r = json.loads(line)
    if "index" not in r:
        continue
    parsed = parse_judge_scores(r["response"], "auto")
    if not parsed or not parsed.get("dimension_scores"):
        continue
    o = float(parsed["overall"]) if "overall" in parsed else _derive_prose_overall(parsed["dimension_scores"], dw)
    student[f"val:{wj_rows[r['index']]}"] = o

new = {k: v for k, v in json.load(open("output/relabel/val_labels_v1.json")).items() if k.startswith("val:")}
items = {x["id"]: x for x in json.load(open("output/relabel/items.json"))}

common = sorted(set(student) & set(new))
s = [student[i] for i in common]; nl = [new[i] for i in common]
r_new = spearmanr(s, nl).statistic
print(f"student vs NEW labels: rho = {r_new:.4f} (n={len(common)})")

old_ids = [i for i in common if items[i]["old_gt"] is not None]
r_old = spearmanr([student[i] for i in old_ids], [items[i]["old_gt"] for i in old_ids]).statistic
print(f"student vs OLD GT (sanity, expect ~0.62): rho = {r_old:.4f} (n={len(old_ids)})")

# ceiling under new labels: rel_M3 from pilot_qc
rel3 = json.load(open("output/relabel/pilot_qc.json"))["rel_M3"]
ceil = rel3 ** 0.5
print(f"new-label ceiling = sqrt({rel3:.3f}) = {ceil:.4f}")
print(f"TRUE GAP = ceiling - student = {ceil - r_new:+.4f}")

# bootstrap CI on student-vs-new rho
rng = np.random.default_rng(7); n = len(common); boots = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boots.append(spearmanr(np.array(s)[idx], np.array(nl)[idx]).statistic)
boots = sorted(boots)
print(f"student-vs-new rho CI: [{boots[50]:.4f}, {boots[1949]:.4f}]")
json.dump({"rho_student_new": r_new, "rho_student_old": r_old, "n": len(common),
           "rel_M3": rel3, "ceiling": ceil, "true_gap": ceil - r_new,
           "rho_ci": [boots[50], boots[1949]]},
          open("output/relabel/student_gap.json", "w"), indent=2)
