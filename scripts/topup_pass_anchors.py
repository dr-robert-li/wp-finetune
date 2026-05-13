"""Phase 1a step 1 top-up: append more PASS anchors to existing pool.

Reads existing anchors from output/diagnostic/pass_anchors_features.jsonl,
dedupes by (source_repo, source_file, function_name), then runs the same
4-tool rubric scorer (with LLM checks via vLLM when RUBRIC_USE_LLM_CHECKS=1
and LLM_BACKEND=vllm) on a fresh stratified sample until --target qualify.

Appends to the same output file. Idempotent if interrupted (skips already-
written anchors on relaunch).

Usage:
    LLM_BACKEND=vllm \\
    LLM_VLLM_BASE_URL=http://localhost:30000/v1 \\
    LLM_VLLM_MODEL=Qwen/Qwen3.6-35B-A3B \\
    RUBRIC_USE_LLM_CHECKS=1 \\
    python -m scripts.topup_pass_anchors --target 37 --sample-pool 2000 --seed 4242
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_pass_anchors import (
    iter_passed_functions,
    stratified_sample,
    is_deterministic_anchor,
)


def existing_keys(path: Path) -> set[tuple]:
    keys: set[tuple] = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r.get("source_repo"), r.get("source_file"), r.get("function_name")))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=37)
    p.add_argument("--sample-pool", type=int, default=2000)
    p.add_argument("--min-overall", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--output", default="output/diagnostic/pass_anchors_features.jsonl")
    args = p.parse_args()

    out_path = ROOT / args.output
    existing = existing_keys(out_path)
    print(f"Existing anchors: {len(existing)}")

    random.seed(args.seed)
    all_items = list(iter_passed_functions())
    print(f"Passed pool: {len(all_items)} functions")

    sampled = stratified_sample(all_items, n=args.sample_pool)
    print(f"Stratified-sampled {len(sampled)} candidates (seed={args.seed})")

    n_new = 0
    n_skipped_dup = 0
    n_rejected = 0
    reject_reasons: dict[str, int] = {}

    with out_path.open("a") as f:
        for i, item in enumerate(sampled):
            if n_new >= args.target:
                break
            key = (item.get("source_repo"), item.get("source_file"), item.get("function_name"))
            if key in existing:
                n_skipped_dup += 1
                continue
            code = item.get("body") or item.get("code")
            if not code:
                continue
            ok, diag = is_deterministic_anchor(code, min_overall=args.min_overall)
            if not ok:
                n_rejected += 1
                reason = diag.get("reject_reason", "unknown")
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            triggered_flat = sorted({
                cid for ids in diag["triggered_checks"].values() for cid in ids
            })
            anchor = {
                "function_name": item.get("function_name"),
                "source_repo": item.get("source_repo"),
                "source_file": item.get("source_file"),
                "training_tags": item.get("training_tags", []),
                "code": code,
                "rubric_overall": diag["overall"],
                "rubric_dim_scores": diag["dimension_scores"],
                "claude_assessment": item.get("assessment", {}),
                "triggered_checks_flat": triggered_flat,
                "dimension_na": list(diag["dimension_na"]),
                "floor_rules_applied": list(diag["floor_rules_applied"]),
                "rubric_triggered_check_count": diag["triggered_check_count"],
                "llm_checks_skipped": diag["llm_checks_skipped"],
            }
            f.write(json.dumps(anchor) + "\n")
            f.flush()
            existing.add(key)
            n_new += 1
            print(f"  [{i+1}/{len(sampled)}] +{n_new} anchor: {key[0]}::{key[2]} overall={diag['overall']:.2f} llm_skipped={diag['llm_checks_skipped']}")

    print()
    print(f"Top-up complete. Added {n_new} new anchors.")
    print(f"Total anchors now: {len(existing)}")
    print(f"Skipped dup keys: {n_skipped_dup}")
    print(f"Rejected: {n_rejected}")
    if reject_reasons:
        print("Top reject reasons:")
        for r, n in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {n:5d}  {r}")


if __name__ == "__main__":
    main()
