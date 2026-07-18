#!/usr/bin/env python3
"""A6 hold-out label check — prep.

Samples 30 functions from phase1 extraction that are DOUBLY fresh:
  (a) code hash not in openai_train or openai_val user turns (v1.3 never
      trained on or was evaluated against them), and
  (b) therefore also never seen by the relabel rubric agents.
Emits:
  output/relabel/holdout_items.json      — label batch for M=3 rubric agents
                                            (30 items + 3 campaign sentinels)
  data/reasoning_dataset/holdout_v1.jsonl — capture dataset for v1.3 sampling
  output/relabel/holdout_key.json        — id -> source bookkeeping
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
from eval.output_parsers import extract_php_code  # noqa: E402
from scripts.reward_calibration import normalized_code_hash  # noqa: E402

# hashes of every snippet the model/instrument has seen
seen = set()
for path in ("data/reasoning_dataset/openai_train.jsonl", "data/reasoning_dataset/openai_val.jsonl"):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        h = normalized_code_hash(extract_php_code(u))
        if h:
            seen.add(h)
print(f"seen hashes: {len(seen)}")

cands = []
for f in sorted(glob.glob("data/phase1_extraction/output/*/*.json")):
    if "backup" in f or "/extracted" in f:
        continue
    try:
        rows = json.load(open(f))
    except Exception:  # noqa: BLE001
        continue
    for e in rows if isinstance(rows, list) else []:
        body = e.get("body") or ""
        if not (120 <= len(body) <= 4000):
            continue
        code = (e.get("docblock") or "") + "\n" + body if e.get("docblock") else body
        h = normalized_code_hash(body)
        if not h or h in seen:
            continue
        cands.append({"src": f.split("/")[-1], "fn": e.get("function_name"), "code": code, "h": h})

# dedup by hash
uniq = {c["h"]: c for c in cands}
cands = list(uniq.values())
print(f"fresh candidates: {len(cands)}")

rng = random.Random(20260704)
picked = rng.sample(cands, 30)

# label batch (30 + 3 sentinels from campaign for drift anchoring)
items_all = {x["id"]: x for x in json.load(open("output/relabel/items.json"))}
sent_ids = json.load(open("output/relabel/sentinels.json"))
entries = [{"id": f"holdout:{n}",
            "prompt": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{c['code']}\n```"}
           for n, c in enumerate(picked)]
for s in sent_ids:
    entries.append({"id": f"SENTINEL::{s}", "prompt": items_all[s]["prompt"]})
rng.shuffle(entries)
json.dump(entries, open("output/relabel/holdout_items.json", "w"), indent=1)

# capture dataset for v1.3 (openai format, wp_judge user turns, holdout order 0..29)
with open("data/reasoning_dataset/holdout_v1.jsonl", "w") as fh:
    for n, c in enumerate(picked):
        fh.write(json.dumps({"messages": [
            {"role": "user", "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{c['code']}\n```"},
            {"role": "assistant", "content": ""}]}) + "\n")

json.dump({"picked": [{"id": f"holdout:{n}", "src": c["src"], "fn": c["fn"], "hash": c["h"]}
                      for n, c in enumerate(picked)]},
          open("output/relabel/holdout_key.json", "w"), indent=2)
print(f"wrote 30 holdout items (+{len(sent_ids)} sentinels) + capture dataset")
