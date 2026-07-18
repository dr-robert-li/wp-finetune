#!/usr/bin/env python3
"""Finalize relabel campaign (08.2-RELABEL-PROTOCOL step 6, label side).

Merges train p1+p2+active-p3 and val p1+p2+p3 into final median labels, then:
  data/relabel_v1/labels.json           — {id: {overall, n_passes, old_gt}} (tracked)
  data/relabel_v1/judge_gt_sidecar_v2.jsonl — TRAIN rows {code_hash, teacher_overall,
      source:"train_relabel_v1"} via the canonical normalized_code_hash join
      (drop-in replacement shape for the reward-time GT sidecar; note the reward
      loader asserts source=="train" — pass sidecar_version explicitly when adopting)
Prints summary stats incl. dispersion of active items post-p3.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

OUT = "output/relabel"
items = {x["id"]: x for x in json.load(open(f"{OUT}/items.json"))}
sent_ids = set(json.load(open(f"{OUT}/sentinels.json")))

# gather ALL passes per item (train p1/p2/active_p3, val p1/p2/p3), sentinel obs pooled
obs: dict[str, list[float]] = defaultdict(list)
for f in sorted(glob.glob(f"{OUT}/results/*.json")):
    try:
        rows = json.load(open(f))
    except Exception:  # noqa: BLE001
        continue
    for e in rows:
        iid = e.get("id", "")
        j = e.get("judge") or {}
        o = j.get("overall_score")
        if not isinstance(o, (int, float)):
            continue
        if iid.startswith("SENTINEL::"):
            obs[iid[10:]].append(float(o))
        else:
            obs[iid].append(float(o))

labels = {}
for iid, it in items.items():
    vals = obs.get(iid, [])
    if not vals:
        continue
    labels[iid] = {
        "overall": median(vals),
        "n_passes": len(vals),
        "old_gt": it["old_gt"],
    }

n_train = sum(1 for i in labels if i.startswith("train"))
n_val = sum(1 for i in labels if i.startswith("val"))
missing = [i for i in items if i not in labels]
print(f"final labels: train={n_train}/482 val={n_val}/121 missing={len(missing)} {missing[:5]}")

os.makedirs("data/relabel_v1", exist_ok=True)
json.dump(labels, open("data/relabel_v1/labels.json", "w"), indent=1)

# sidecar v2 (train side) via canonical hash
from scripts.reward_calibration import normalized_code_hash  # noqa: E402
from scripts.rl_rollouts import judge_item_code_hash  # noqa: E402
from scripts.tinker_rl_data import load_rl_prompts  # noqa: E402
from eval.eval_judge import _extract_gt_from_assistant  # noqa: E402,F401

# hash per item from the original dataset user turns
rows_train = [json.loads(l) for l in open("data/reasoning_dataset/openai_train.jsonl") if l.strip()]
n_side = 0
with open("data/relabel_v1/judge_gt_sidecar_v2.jsonl", "w") as fh:
    for iid, lab in labels.items():
        if not iid.startswith("train"):
            continue
        ridx = int(iid.split(":")[1])
        u = next((m["content"] for m in rows_train[ridx]["messages"] if m["role"] == "user"), "")
        from eval.output_parsers import extract_php_code  # noqa: PLC0415
        h = normalized_code_hash(extract_php_code(u))
        if not h:
            continue
        fh.write(json.dumps({"code_hash": h, "teacher_overall": lab["overall"],
                             "source": "train_relabel_v1", "prompt_id": ridx,
                             "n_passes": lab["n_passes"]}) + "\n")
        n_side += 1
print(f"sidecar v2 rows: {n_side}")

# join coverage vs RL judge pool (informational)
pool = load_rl_prompts("judge")
side_hashes = set()
for line in open("data/relabel_v1/judge_gt_sidecar_v2.jsonl"):
    side_hashes.add(json.loads(line)["code_hash"])
hits = sum(1 for it in pool if judge_item_code_hash(it) in side_hashes)
print(f"RL judge pool coverage under sidecar v2: {hits}/{len(pool)} ({hits/len(pool):.1%})")

# dispersion check on active items post-p3
active = json.load(open(f"{OUT}/active_items.json"))
still_wide = [i for i in active if i in obs and len(obs[i]) >= 3
              and (max(obs[i]) - min(obs[i])) > 25]
print(f"active items still wide (range>25 after p3): {len(still_wide)}/{len(active)} {still_wide[:5]}")
