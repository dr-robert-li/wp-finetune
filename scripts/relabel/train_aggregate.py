#!/usr/bin/env python3
"""Train-wave aggregation + active-wave selection (08.2-RELABEL-PROTOCOL steps 3-4).

Aggregates the 2 train passes, reports pass agreement + sentinel drift, and
selects disagreement items (|p1-p2| > 15 OR verdict split OR only 1 pass) for
the active 3rd pass. Writes:
  output/relabel/train_labels_prelim.json  (median of available passes)
  output/relabel/active_items.json         (ids needing pass 3)
  output/relabel/batches/active_p3_*.json  (batch files for the active wave)
"""
import glob
import json
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, ".")
from scipy.stats import spearmanr

OUT = "output/relabel"
items = {x["id"]: x for x in json.load(open(f"{OUT}/items.json"))}
sent_ids = set(json.load(open(f"{OUT}/sentinels.json")))

passes: dict[str, dict[str, float]] = defaultdict(dict)
verdicts: dict[str, dict[str, str]] = defaultdict(dict)
sentinel_obs = []
bad = []
for f in sorted(glob.glob(f"{OUT}/results/train_p*_*.json")):
    pname = f.split("/")[-1].split("_")[1]
    try:
        rows = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        bad.append((f, str(e)))
        continue
    for e in rows:
        iid = e.get("id", "")
        j = e.get("judge") or {}
        o = j.get("overall_score")
        if not isinstance(o, (int, float)):
            continue
        if iid.startswith("SENTINEL::"):
            sentinel_obs.append((f.split("/")[-1], iid[10:], float(o)))
            continue
        passes[pname][iid] = float(o)
        v = j.get("verdict")
        if v in ("PASS", "FAIL"):
            verdicts[pname][iid] = v

print(f"bad files: {len(bad)} {bad[:3]}")
p1, p2 = passes["p1"], passes["p2"]
v1, v2 = verdicts["p1"], verdicts["p2"]
train_ids = [x for x in items if x.startswith("train")]
common = sorted(set(p1) & set(p2))
print(f"train items: {len(train_ids)}; p1={len(p1)} p2={len(p2)} common={len(common)}")
r12 = spearmanr([p1[i] for i in common], [p2[i] for i in common]).statistic
print(f"rho(p1,p2) = {r12:.4f}")

# sentinel drift per batch
camp = defaultdict(list)
for _, sid, o in sentinel_obs:
    camp[sid].append(o)
camp_med = {sid: median(v) for sid, v in camp.items()}
by_batch = defaultdict(dict)
for bf, sid, o in sentinel_obs:
    by_batch[bf][sid] = o
flagged = [bf for bf, obs in by_batch.items()
           if median([abs(obs[s] - camp_med[s]) for s in obs]) > 10]
print(f"sentinel medians: {camp_med}; flagged batches: {len(flagged)} {flagged[:5]}")

# active selection
active = []
for i in train_ids:
    if i in sent_ids:
        continue  # sentinels have many obs already
    a, b = p1.get(i), p2.get(i)
    va, vb = v1.get(i), v2.get(i)
    if a is None or b is None:
        active.append(i)
    elif abs(a - b) > 15 or (va and vb and va != vb):
        active.append(i)
print(f"active wave items: {len(active)} ({len(active)/len(train_ids):.1%})")

# prelim labels (median of available passes; sentinels get campaign median)
labels = {}
for i in train_ids:
    if i in sent_ids:
        labels[i] = camp_med.get(i)
        continue
    vals = [p[i] for p in (p1, p2) if i in p]
    if vals:
        labels[i] = median(vals)
json.dump(labels, open(f"{OUT}/train_labels_prelim.json", "w"))
json.dump(active, open(f"{OUT}/active_items.json", "w"))

# active batches (12 + 3 sentinels)
sentinels = [items[s] for s in sent_ids]
B = 12
n = 0
for bi in range(0, len(active), B):
    chunk = [items[i] for i in active[bi:bi + B]]
    payload = [{"id": x["id"], "prompt": x["prompt"]} for x in chunk]
    payload += [{"id": f"SENTINEL::{s['id']}", "prompt": s["prompt"]} for s in sentinels]
    json.dump(payload, open(f"{OUT}/batches/active_p3_{bi//B:03d}.json", "w"))
    n += 1
print(f"active batches written: {n}")
