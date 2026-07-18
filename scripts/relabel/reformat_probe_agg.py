#!/usr/bin/env python3
"""Reformat-probe bias audit — aggregate + gate.

Unblinds the judged batch, pairs each item's original vs reformatted score
(median over the M judge passes), and reports the reformat Δ against the
campaign single-pass noise floor. Gate: no systematic direction
(|mean Δ| < 5) and no item drifts beyond ordinary judge jitter (max|Δ| < 15,
the campaign's own "still wide" threshold). Writes reformat_probe.json.
"""
import glob
import json
from collections import defaultdict
from statistics import median

OUT = "output/relabel"

key = json.load(open(f"{OUT}/reformat_probe_key.json"))
K, noise = key["key"], key["noise_floor"]

# median overall per probe_id across the M passes
by_pid = defaultdict(list)
passes = 0
for f in sorted(glob.glob(f"{OUT}/results_reformat/pass_*.json")):
    passes += 1
    for e in json.load(open(f)):
        o = (e.get("judge") or {}).get("overall_score")
        if isinstance(o, (int, float)):
            by_pid[e["probe_id"]].append(float(o))
score = {pid: median(v) for pid, v in by_pid.items() if v}

# pair per item
item = defaultdict(dict)
for pid, meta in K.items():
    if pid in score:
        item[meta["item"]][meta["variant"]] = score[pid]

rows, deltas = [], []
for iid, v in sorted(item.items()):
    if "orig" in v and "reformat" in v:
        d = v["reformat"] - v["orig"]
        deltas.append(d)
        rows.append({"item": iid, "orig": v["orig"], "reformat": v["reformat"],
                     "delta": d, "wide": iid in key["wide"]})

mean_d = sum(deltas) / len(deltas)
mean_abs = sum(abs(d) for d in deltas) / len(deltas)
max_abs = max(abs(d) for d in deltas)
gate = {"no_systematic_dir_|meanΔ|<5": abs(mean_d) < 5,
        "no_item_drift_max|Δ|<15": max_abs < 15}
ok = all(gate.values())

for r in rows:
    print(f"  {r['item']:<12} {'WIDE' if r['wide'] else 'rand'}  "
          f"orig={r['orig']:.0f} reformat={r['reformat']:.0f}  Δ={r['delta']:+.0f}")
print(f"\nM={passes} passes | n_items={len(rows)} | noise_floor(single-pass |dev| med)={noise:.1f}")
print(f"mean Δ = {mean_d:+.2f}  mean|Δ| = {mean_abs:.2f}  max|Δ| = {max_abs:.0f}")
print(f"GATES: {gate} -> {'PASS (no reformat bias)' if ok else 'FAIL (formatting-sensitive)'}")

json.dump({"n_passes": passes, "n_items": len(rows), "noise_floor": noise,
           "mean_delta": mean_d, "mean_abs_delta": mean_abs, "max_abs_delta": max_abs,
           "per_item": rows, "gates": gate, "pass": ok},
          open(f"{OUT}/reformat_probe.json", "w"), indent=2)
print(f"wrote {OUT}/reformat_probe.json")
