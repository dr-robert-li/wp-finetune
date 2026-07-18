#!/usr/bin/env python3
"""B2 offline oracle gate — does reward-v2 track judge quality? (08.1 lesson:
never train on a reward that hasn't demonstrated it ranks checkpoints the same
way the target metric does.)

For every historical checkpoint capture: mean reward-v2 (blended + per-stream)
over val items, and measured judge-rho vs the relabel-v1 val labels. Oracle =
Spearman(mean_reward, judge_rho) across checkpoints, with a joint item-level
bootstrap CI (resample items -> recompute BOTH stats per checkpoint -> Spearman).
GATE: CI-lower > 0. Also reports per-stream oracles (calib-only / defect-only)
so a failing blend can be diagnosed.
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
from scripts.reward_v2 import combine, load_gt, score

CAPTURES = {
    "warmstart_v12": "output/rl_eval/warmstart/judge_responses.jsonl",
    "rl_step50": "output/rl_eval/step-50-seedA2/judge_responses.jsonl",
    "rl_step150": "output/rl_eval/step-150-seedA2/judge_responses.jsonl",
    "relabel_1ep": "output/relabel/eval_relabel_v1/judge_responses.jsonl",
    "oldctrl_1ep": "output/relabel/eval_oldctrl_1ep/judge_responses.jsonl",
    "full_ep1": "output/relabel/eval_full_ep1/judge_responses.jsonl",
    "full_ep2": "output/relabel/eval_full_ep2/judge_responses.jsonl",
    "full_ep3": "output/relabel/eval_full_ep3/judge_responses.jsonl",
    "s1_ep3_v13": "output/relabel/eval_s1_ep3/judge_responses.jsonl",
    "s2_ep3": "output/relabel/eval_s2_ep3/judge_responses.jsonl",
}

dm = load_dim_map()
dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
wj = [i for i, r in enumerate(rows)
      if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
labels = {k: v for k, v in json.load(open("output/relabel/val_labels_v1.json")).items()
          if k.startswith("val:")}
gt = load_gt("val")

# per checkpoint: {item_id: (model_overall_or_None, reward_streams)}
data = {}
for name, path in CAPTURES.items():
    if not os.path.exists(path):
        print(f"SKIP {name}: missing {path}")
        continue
    d = {}
    for line in open(path):
        r = json.loads(line)
        if "index" not in r:
            continue
        iid = f"val:{wj[r['index']]}"
        if iid not in gt or iid not in labels:
            continue
        st = score(r["response"], gt[iid])
        parsed = parse_judge_scores(r["response"], "auto")
        ov = None
        if parsed and parsed.get("dimension_scores"):
            ov = (float(parsed["overall"]) if "overall" in parsed
                  else _derive_prose_overall(parsed["dimension_scores"], dw))
        d[iid] = (ov, st)
    data[name] = d

common = sorted(set.intersection(*(set(v) for v in data.values())))
names = list(data)
print(f"checkpoints={len(names)}  common items={len(common)}\n")

L = np.array([labels[i] for i in common])


def ckpt_stats(name, idx):
    ovs = np.array([data[name][common[j]][0] if data[name][common[j]][0] is not None else np.nan
                    for j in idx], dtype=float)
    rew = np.array([combine(data[name][common[j]][1]) for j in idx])
    cal = np.array([data[name][common[j]][1]["calib"] for j in idx])
    dfc = np.array([data[name][common[j]][1]["defect"] for j in idx])
    lab = L[idx]
    ok = ~np.isnan(ovs)
    rho = spearmanr(ovs[ok], lab[ok]).statistic if ok.sum() > 5 else float("nan")
    return rho, rew.mean(), cal.mean(), dfc.mean()


full = list(range(len(common)))
table = {n: ckpt_stats(n, full) for n in names}
print(f"{'checkpoint':<15} {'judge_rho':>9} {'reward':>8} {'calib':>7} {'defect':>7}")
for n in names:
    rho, rw, ca, df = table[n]
    print(f"{n:<15} {rho:9.4f} {rw:8.4f} {ca:7.4f} {df:7.4f}")

rhos = [table[n][0] for n in names]
oracle = spearmanr([table[n][1] for n in names], rhos).statistic
oracle_cal = spearmanr([table[n][2] for n in names], rhos).statistic
oracle_dfc = spearmanr([table[n][3] for n in names], rhos).statistic
print(f"\nORACLE Spearman(mean_reward, judge_rho) across {len(names)} ckpts:")
print(f"  blended = {oracle:.4f}   calib-only = {oracle_cal:.4f}   defect-only = {oracle_dfc:.4f}")

# joint item bootstrap
rng = np.random.default_rng(13)
n = len(common)
boots = []
for _ in range(2000):
    idx = list(rng.integers(0, n, n))
    st = {m: ckpt_stats(m, idx) for m in names}
    boots.append(spearmanr([st[m][1] for m in names], [st[m][0] for m in names]).statistic)
boots = sorted(boots)
lo, hi = boots[50], boots[1949]
gate = lo > 0
print(f"  bootstrap CI [{lo:.4f}, {hi:.4f}]  ->  GATE CI-lower>0: {'PASS' if gate else 'FAIL'}")

json.dump({"n_ckpts": len(names), "n_items": len(common),
           "table": {m: {"judge_rho": table[m][0], "mean_reward": table[m][1],
                          "mean_calib": table[m][2], "mean_defect": table[m][3]} for m in names},
           "oracle_blended": oracle, "oracle_calib": oracle_cal, "oracle_defect": oracle_dfc,
           "ci": [lo, hi], "gate_pass": bool(gate)},
          open("output/relabel/oracle_gate_v2.json", "w"), indent=2)
print("wrote output/relabel/oracle_gate_v2.json")
