#!/usr/bin/env python
"""Build the corrective augmented training set (recorded, reproducible).

augmented = openai_train.jsonl (unchanged teacher data) + negatives_train.jsonl
(invalid-PHP / fabricated-API should_fail negatives). We do NOT hand-edit or relabel
the teacher rows — the boundary (@51-57) miscalibration is handled by (a) the negatives
pulling raw scores on broken code into clear-FAIL territory and (b) a documented post-hoc
verdict policy (overall_score < 70 -> effective FAIL) applied at evaluation, NOT by
flipping teacher verdicts (which would contradict their PASS-leaning prose).

LEAKAGE GUARD: asserts no negative shares its code with the held-out sentinel
(invalid_php_sentinel.jsonl) or the val set — the sentinel must test generalization.

Usage: python scripts/build_augmented_train.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _codes(path):
    out = set()
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        out.add(u)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/reasoning_dataset/openai_train.jsonl")
    ap.add_argument("--negatives", default="data/reasoning_dataset/negatives_train.jsonl")
    ap.add_argument("--val", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--sentinel", default="data/reasoning_dataset/invalid_php_sentinel.jsonl")
    ap.add_argument("--out", default="data/reasoning_dataset/openai_train.augmented.jsonl")
    args = ap.parse_args()

    base = [l for l in open(args.base) if l.strip()]
    negs = [l for l in open(args.negatives) if l.strip()]

    neg_codes = _codes(args.negatives)
    leak_sent = neg_codes & _codes(args.sentinel)
    leak_val = neg_codes & _codes(args.val)
    if leak_sent:
        print(f"LEAKAGE: {len(leak_sent)} negatives share code with the sentinel", file=sys.stderr)
        return 1
    if leak_val:
        print(f"LEAKAGE: {len(leak_val)} negatives share code with val", file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        for l in base:
            f.write(l if l.endswith("\n") else l + "\n")
        for l in negs:
            f.write(l if l.endswith("\n") else l + "\n")

    # verdict balance report
    import re
    pas = fail = 0
    for l in (base + negs):
        a = next((m["content"] for m in json.loads(l)["messages"] if m["role"] == "assistant"), "")
        mv = re.search(r'"verdict"\s*:\s*"([A-Z]+)"', a)
        if mv:
            if mv.group(1) == "PASS":
                pas += 1
            elif mv.group(1) == "FAIL":
                fail += 1
    print(f"augmented: base={len(base)} + negatives={len(negs)} = {len(base) + len(negs)} rows")
    print(f"verdict balance (with-verdict rows): PASS={pas} FAIL={fail}")
    print(f"leakage: sentinel=0 val=0 (OK) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
