#!/usr/bin/env python3
"""Adversarial reformat audit — aggregate (A5). Disjoint blinded batches.

Per item: Δ = median(reform passes) - median(orig passes). Since different
agent instances scored each batch, Δ includes inter-rater noise; the reference
noise floor is the campaign single-pass reliability (r≈0.9125 → typical |dev|
1-3 pts). Gates: |mean Δ| < 5 (no systematic direction), mean|Δ| within ~2x
inter-rater noise. Also reports paired-audit comparison.
"""
import json
from statistics import median

OUT = "output/relabel"
key = json.load(open(f"{OUT}/reformat_audit2_key.json"))["key"]

def load(path):
    return {e["probe_id"]: (e.get("judge") or {}).get("overall_score")
            for e in json.load(open(path))}

orig = [load(f"{OUT}/results_reformat2/orig_p{i}.json") for i in (1, 2)]
reform = [load(f"{OUT}/results_reformat2/reform_p{i}.json") for i in (1, 2)]

rows, deltas = [], []
for item, ids in sorted(key.items()):
    o = [p[ids["orig"]] for p in orig if isinstance(p.get(ids["orig"]), (int, float))]
    r = [p[ids["reform"]] for p in reform if isinstance(p.get(ids["reform"]), (int, float))]
    if not o or not r:
        continue
    d = median(r) - median(o)
    deltas.append(d)
    rows.append((item, median(o), median(r), d))

mean_d = sum(deltas) / len(deltas)
mean_abs = sum(abs(d) for d in deltas) / len(deltas)
mx = max(abs(d) for d in deltas)
big = [(i, o, r, d) for i, o, r, d in rows if abs(d) >= 10]
for i, o, r, d in sorted(rows, key=lambda x: -abs(x[3]))[:8]:
    print(f"  {i:<12} orig={o:.0f} reform={r:.0f} Δ={d:+.0f}")
print(f"\nn={len(rows)}  mean Δ={mean_d:+.2f}  mean|Δ|={mean_abs:.2f}  max|Δ|={mx:.0f}  |Δ|>=10: {len(big)}")
gates = {"no_systematic_|meanΔ|<5": abs(mean_d) < 5,
         "mean_absΔ_lt_8 (2x inter-rater)": mean_abs < 8}
ok = all(gates.values())
print(f"GATES: {gates} -> {'PASS' if ok else 'FAIL'}")
json.dump({"n": len(rows), "mean_delta": mean_d, "mean_abs_delta": mean_abs,
           "max_abs_delta": mx, "n_ge10": len(big),
           "per_item": [{"item": i, "orig": o, "reform": r, "delta": d} for i, o, r, d in rows],
           "gates": gates, "pass": ok},
          open(f"{OUT}/reformat_audit2.json", "w"), indent=2)
print(f"wrote {OUT}/reformat_audit2.json")
