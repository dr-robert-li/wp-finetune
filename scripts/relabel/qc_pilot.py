#!/usr/bin/env python3
"""Pilot QC gates (08.2-RELABEL-PROTOCOL step 1).

Aggregates the 3 val passes, computes reliability + bias audits, checks gates:
  G-A split-half reliability of aggregated labels >= 0.95 (Spearman-Brown projected)
  G-B kappa on PASS/FAIL verdicts across passes >= 0.6
  G-C verbosity: |Spearman(median_overall, code_length)| < 0.2
  G-D validity anchor: Spearman(median_overall, old GT) reported (context, not a gate)
  Sentinel drift: per-batch sentinel deviation report (>10 pts flagged)
Writes output/relabel/pilot_qc.json + val_labels_v1.json (median labels).
"""
import glob
import itertools
import json
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, ".")
from scipy.stats import spearmanr

OUT = "output/relabel"

items = {x["id"]: x for x in json.load(open(f"{OUT}/items.json"))}

# passes[pass][id] = overall ; verdicts[pass][id] = PASS/FAIL
passes: dict[str, dict[str, float]] = defaultdict(dict)
verdicts: dict[str, dict[str, str]] = defaultdict(dict)
sentinel_obs = []  # (batch_file, sent_id, overall)
n_bad = 0
for f in sorted(glob.glob(f"{OUT}/results/val_p*_*.json")):
    pname = f.split("/")[-1].split("_")[1]  # p1/p2/p3
    try:
        rows = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        print(f"UNPARSEABLE {f}: {e}")
        n_bad += 1
        continue
    for e in rows:
        iid = e.get("id", "")
        j = e.get("judge") or {}
        o = j.get("overall_score")
        if not isinstance(o, (int, float)):
            continue
        if iid.startswith("SENTINEL::"):
            sentinel_obs.append((f.split("/")[-1], iid[10:], float(o)))
        else:
            passes[pname][iid] = float(o)
            v = j.get("verdict")
            if v in ("PASS", "FAIL"):
                verdicts[pname][iid] = v

pn = sorted(passes)
common = sorted(set.intersection(*(set(passes[p]) for p in pn)))
print(f"passes: {pn}, per-pass n: {[len(passes[p]) for p in pn]}, common n={len(common)}, bad files={n_bad}")

# Pairwise pass correlations -> single-pass reliability r (mean of pairs)
pair_rs = []
for a, b in itertools.combinations(pn, 2):
    r = spearmanr([passes[a][i] for i in common], [passes[b][i] for i in common]).statistic
    pair_rs.append(r)
    print(f"rho({a},{b}) = {r:.4f}")
r1 = sum(pair_rs) / len(pair_rs)
rel3 = 3 * r1 / (1 + 2 * r1)  # Spearman-Brown for the 3-pass median/mean composite
print(f"single-pass r = {r1:.4f} -> Spearman-Brown rel(M=3) = {rel3:.4f}  [gate >= 0.95]")

# kappa on verdicts (pairwise mean Cohen's kappa)
def kappa(va, vb, ids):
    both = [(va[i], vb[i]) for i in ids if i in va and i in vb]
    if not both:
        return float("nan")
    n = len(both)
    po = sum(1 for x, y in both if x == y) / n
    pa = sum(1 for x, _ in both if x == "PASS") / n
    pb = sum(1 for _, y in both if y == "PASS") / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

kaps = [kappa(verdicts[a], verdicts[b], common) for a, b in itertools.combinations(pn, 2)]
kap = sum(kaps) / len(kaps)
print(f"verdict kappa (mean pairwise) = {kap:.4f}  [gate >= 0.6]  per-pair: {[round(k,3) for k in kaps]}")

# Median labels over items present in >= 2 passes (tolerates a dropped item)
all_ids = set().union(*(passes[p] for p in pn))
labels = {
    i: median([passes[p][i] for p in pn if i in passes[p]])
    for i in sorted(all_ids)
    if sum(i in passes[p] for p in pn) >= 2
}

# Verbosity bias: overall vs code length
lens = [len(items[i]["prompt"]) for i in common]
rv = spearmanr([labels[i] for i in common], lens).statistic
print(f"verbosity |rho(label, code_len)| = {abs(rv):.4f}  [gate < 0.2]")

# Validity anchor: vs old GT where present
gt_ids = [i for i in common if items[i]["old_gt"] is not None]
rg = spearmanr([labels[i] for i in gt_ids], [items[i]["old_gt"] for i in gt_ids]).statistic
print(f"vs old GT (n={len(gt_ids)}): rho = {rg:.4f}  [context]")

# Sentinel drift per batch
sent_med = defaultdict(list)
for _, sid, o in sentinel_obs:
    sent_med[sid].append(o)
camp_med = {sid: median(v) for sid, v in sent_med.items()}
flagged = []
by_batch = defaultdict(dict)
for bf, sid, o in sentinel_obs:
    by_batch[bf][sid] = o
for bf, obs in by_batch.items():
    devs = [abs(obs[s] - camp_med[s]) for s in obs]
    if devs and median(devs) > 10:
        flagged.append((bf, {s: obs[s] for s in obs}))
print(f"sentinel campaign medians: { {k: v for k, v in camp_med.items()} }")
print(f"flagged batches (median sentinel dev > 10): {len(flagged)}")
for bf, obs in flagged:
    print(f"  {bf}: {obs}")

gates = {
    "rel3_ge_0.95": bool(rel3 >= 0.95),
    "kappa_ge_0.6": bool(kap >= 0.6),
    "verbosity_lt_0.2": bool(abs(rv) < 0.2),
}
print("GATES:", gates, "-> ALL PASS" if all(gates.values()) else "-> FAIL (fix before train wave)")

json.dump(
    {
        "n_common": len(common), "pair_rhos": pair_rs, "single_pass_r": r1,
        "rel_M3": rel3, "verdict_kappa": kap, "verbosity_rho": rv,
        "rho_vs_old_gt": rg, "n_old_gt": len(gt_ids),
        "sentinel_medians": camp_med, "flagged_batches": [b for b, _ in flagged],
        "gates": gates,
    },
    open(f"{OUT}/pilot_qc.json", "w"), indent=2,
)
json.dump(labels, open(f"{OUT}/val_labels_v1.json", "w"), indent=2)
print(f"wrote {OUT}/pilot_qc.json + val_labels_v1.json ({len(labels)} labels)")
