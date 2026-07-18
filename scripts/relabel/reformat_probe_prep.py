#!/usr/bin/env python3
"""Reformat-probe bias audit — prep (08.2-RELABEL belt-and-braces QC).

The rubric says "do not reward or punish formatting style". This probe tests
that: judge the SAME code twice, once as-is and once with a semantics-preserving
whitespace reformat. A calibrated judge gives ~equal scores; a formatting-biased
one drifts. We compare the reformat Δ against the campaign's own single-pass
noise floor (so we don't confuse ordinary judge jitter for bias).

Items = 5 widest-dispersion (from active_items.json, by overall pass-range) +
5 random (seeded, disjoint). Reformat = tabs->4 spaces, strip trailing ws,
collapse 3+ blank lines to 1 (whitespace only; heredoc items are excluded since
indentation there is significant). Emits a BLINDED batch (20 shuffled entries)
for the judge agents, plus a key file for unblinding.
"""
import json
import random
import re
from collections import defaultdict
from statistics import median

OUT = "output/relabel"
CODE_RE = re.compile(r"```php\n(.*?)\n```", re.S)


def pass_overalls():
    acc = defaultdict(list)
    import glob
    for f in sorted(glob.glob(f"{OUT}/results/*.json")):
        try:
            rows = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for e in rows:
            iid = e.get("id", "")
            o = (e.get("judge") or {}).get("overall_score")
            if iid.startswith("train:") and isinstance(o, (int, float)):
                acc[iid].append(float(o))
    return acc


def reformat_php(code: str) -> str:
    lines = code.replace("\t", "    ").split("\n")
    lines = [ln.rstrip() for ln in lines]
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


def main():
    items = {x["id"]: x for x in json.load(open(f"{OUT}/items.json"))}
    active = json.load(open(f"{OUT}/active_items.json"))
    labels = json.load(open("data/relabel_v1/labels.json"))
    ov = pass_overalls()

    def has_heredoc(iid):
        return "<<<" in items[iid]["prompt"]

    # 5 widest-dispersion active items (range of overall across passes), no heredoc
    ranked = sorted(
        (i for i in active if i in ov and len(ov[i]) >= 2 and not has_heredoc(i)),
        key=lambda i: max(ov[i]) - min(ov[i]), reverse=True,
    )
    wide = ranked[:5]

    # 5 random train items, seeded, disjoint from `wide`, no heredoc
    rng = random.Random(20260703)
    pool = [i for i in labels if i.startswith("train:") and i not in wide and not has_heredoc(i)]
    rnd = rng.sample(sorted(pool), 5)

    picked = wide + rnd
    print("wide (disp):", [(i, int(max(ov[i]) - min(ov[i]))) for i in wide])
    print("random:", rnd)

    # intrinsic single-pass noise floor: |pass - item median| over the campaign
    intrinsic = []
    for i in picked:
        if i in ov and len(ov[i]) >= 2:
            m = median(ov[i])
            intrinsic += [abs(x - m) for x in ov[i]]
    noise = median(intrinsic) if intrinsic else float("nan")
    print(f"campaign single-pass |dev| (median) for these items = {noise:.1f}")

    # blinded batch: orig + reformatted variant per item, opaque ids, shuffled
    entries, key = [], {}
    for i in picked:
        code = CODE_RE.search(items[i]["prompt"]).group(1)
        variants = {"orig": code, "reformat": reformat_php(code)}
        for tag, c in variants.items():
            pid = f"P{len(entries):02d}"
            entries.append({"probe_id": pid,
                            "prompt": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{c}\n```"})
            key[pid] = {"item": i, "variant": tag}
    rng.shuffle(entries)

    json.dump(entries, open(f"{OUT}/reformat_probe_batch.json", "w"), indent=1)
    json.dump({"key": key, "picked": picked, "wide": wide, "random": rnd,
               "noise_floor": noise},
              open(f"{OUT}/reformat_probe_key.json", "w"), indent=2)
    print(f"wrote {OUT}/reformat_probe_batch.json ({len(entries)} blinded entries) + key")


if __name__ == "__main__":
    main()
