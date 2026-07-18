#!/usr/bin/env python3
"""Adversarial reformat audit — prep (A5). Blinded DISJOINT batches.

Unlike reformat_probe_prep (paired: each judge saw orig+reformat in one batch,
so identical-substance pairs were recognizable), this splits variants across
batches: batch O = originals only, batch R = reformats only, shuffled
independently, opaque ids. A judge sees exactly one variant of any item and
cannot anchor on its twin. Bias = per-item cross-batch Δ vs the campaign
noise floor.

Items: the 10 from the paired probe (comparability) + 20 more seeded-random
labeled train items (no heredoc) = 30.
"""
import json
import random
import re

OUT = "output/relabel"
CODE_RE = re.compile(r"```php\n(.*?)\n```", re.S)


def reformat_php(code: str) -> str:
    lines = [ln.rstrip() for ln in code.replace("\t", "    ").split("\n")]
    out, blanks = [], 0
    for ln in lines:
        if ln == "":
            blanks += 1
            if blanks <= 1:
                out.append(ln)
        else:
            blanks = 0
            out.append(ln)
    return "\n".join(out)


items = {x["id"]: x for x in json.load(open(f"{OUT}/items.json"))}
labels = json.load(open("data/relabel_v1/labels.json"))
prev = json.load(open(f"{OUT}/reformat_probe_key.json"))["picked"]

rng = random.Random(20260704)
pool = [i for i in labels if i.startswith("train:") and i not in prev
        and "<<<" not in items[i]["prompt"]]
picked = prev + rng.sample(sorted(pool), 20)

orig_entries, reform_entries, key = [], [], {}
for n, i in enumerate(picked):
    code = CODE_RE.search(items[i]["prompt"]).group(1)
    oid, rid = f"O{n:02d}", f"R{n:02d}"
    orig_entries.append({"probe_id": oid,
                         "prompt": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"})
    reform_entries.append({"probe_id": rid,
                           "prompt": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{reformat_php(code)}\n```"})
    key[i] = {"orig": oid, "reform": rid}

rng.shuffle(orig_entries)
rng.shuffle(reform_entries)
json.dump(orig_entries, open(f"{OUT}/reformat_audit2_orig.json", "w"), indent=1)
json.dump(reform_entries, open(f"{OUT}/reformat_audit2_reform.json", "w"), indent=1)
json.dump({"key": key, "picked": picked}, open(f"{OUT}/reformat_audit2_key.json", "w"), indent=2)
print(f"wrote {len(orig_entries)} orig + {len(reform_entries)} reform entries (disjoint, blinded) + key")
