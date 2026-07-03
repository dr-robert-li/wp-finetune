#!/usr/bin/env python3
"""Relabel campaign prep (08.2-RELABEL-PROTOCOL step 0/1).

Dumps all wp_judge items (train+val) with old GT where present, picks 3 fixed
sentinel items (low/mid/high old-GT), and writes per-pass batch files of 9 items
+ 3 sentinels appended (drift monitor, fixed position LAST).
"""
import json, os, random, sys
sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
from eval.eval_judge import _extract_gt_from_assistant

OUT = "output/relabel"
os.makedirs(f"{OUT}/batches", exist_ok=True)

items = []
for split in ("train", "val"):
    rows = [json.loads(l) for l in open(f"data/reasoning_dataset/openai_{split}.jsonl") if l.strip()]
    for i, r in enumerate(rows):
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        if not u.startswith("<wp_judge>"):
            continue
        gt = _extract_gt_from_assistant(r["messages"])
        items.append({
            "id": f"{split}:{i}",
            "prompt": u,
            "old_gt": float(gt["overall"]) if gt else None,
        })
json.dump(items, open(f"{OUT}/items.json", "w"))
print(f"items: {len(items)} (train {sum(1 for x in items if x['id'].startswith('train'))}, "
      f"val {sum(1 for x in items if x['id'].startswith('val'))})")

# Sentinels: 3 train items nearest old-GT 30 / 65 / 90 (deterministic).
gt_items = [x for x in items if x["old_gt"] is not None and x["id"].startswith("train")]
sentinels = [min(gt_items, key=lambda x: abs(x["old_gt"] - t)) for t in (30, 65, 90)]
sent_ids = [s["id"] for s in sentinels]
json.dump(sent_ids, open(f"{OUT}/sentinels.json", "w"))
print("sentinels:", [(s['id'], s['old_gt']) for s in sentinels])

# Batches: per split+pass, shuffled with pass-specific seed (order differs per pass
# -> independent-ish passes, and no source-file ordering drift).
B = 9
def write_pass(split, pass_name, seed):
    pool = [x for x in items if x["id"].startswith(split) and x["id"] not in sent_ids]
    # sentinels are labeled via their own batch occurrences in train passes
    if split == "train":
        pool += [x for x in items if x["id"] in sent_ids]
    rng = random.Random(seed)
    pool = pool[:]; rng.shuffle(pool)
    n = 0
    for bi in range(0, len(pool), B):
        chunk = pool[bi:bi+B]
        payload = [{"id": x["id"], "prompt": x["prompt"]} for x in chunk]
        payload += [{"id": f"SENTINEL::{s['id']}", "prompt": s["prompt"]} for s in sentinels]
        json.dump(payload, open(f"{OUT}/batches/{split}_{pass_name}_{bi//B:03d}.json", "w"))
        n += 1
    print(f"{split} pass {pass_name}: {n} batches")
    return n

total = 0
for p, seed in (("p1", 101), ("p2", 202), ("p3", 303)):
    total += write_pass("val", p, seed)
print(f"val batch files: {total}")
