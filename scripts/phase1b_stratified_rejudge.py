"""Phase 1b: stratified re-judge of the phase1 function pool via the calibrated rubric.

Samples N functions from data/phase1_extraction/output/{passed,failed}/ stratified by
the prior Claude assessment overall mean (5 buckets covering 0-10) and re-scores each
through eval.rubric_scorer.score_code() — which now emits both the legacy rubric_overall
and the XGBoost calibrated_overall + calibrated_verdict.

Use LLM_BACKEND=vllm + RUBRIC_USE_LLM_CHECKS=1 to run the 41 LLM-assisted checks via
the local Qwen3.6 endpoint. The calibration was fit on LLM-on features, so inference
must match the training distribution.

Output JSONL per row:
    row_id, source_repo, source_file, function_name, bucket, claude_overall_mean,
    claude_verdict, rubric_overall, rubric_dim_scores, rubric_dim_na,
    triggered_checks_flat, rubric_llm_checks_skipped, calibrated_overall,
    calibrated_verdict

Usage (1K pilot, 8 workers):
    LLM_BACKEND=vllm \\
    LLM_VLLM_BASE_URL=http://localhost:30000/v1 \\
    LLM_VLLM_MODEL=Qwen/Qwen3.6-35B-A3B \\
    RUBRIC_USE_LLM_CHECKS=1 \\
    python -m scripts.phase1b_stratified_rejudge \\
        --target 1000 --workers 8 --seed 0 \\
        --output data/phase1b/rejudge_pilot_1k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.rubric_scorer import score_code

PASSED_DIR = ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = ROOT / "data" / "phase1_extraction" / "output" / "failed"

# (label, lo_inclusive, hi_exclusive)
BUCKETS = [
    ("0-4.99", 0.0, 5.0),
    ("5-6.99", 5.0, 7.0),
    ("7-7.99", 7.0, 8.0),
    ("8-8.99", 8.0, 9.0),
    ("9-10",   9.0, 10.01),
]


def iter_pool() -> Iterator[dict]:
    for d in (PASSED_DIR, FAILED_DIR):
        if not d.exists():
            continue
        for jf in sorted(d.glob("*.json")):
            try:
                items = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(items, dict):
                items = [items]
            for it in items:
                yield it


def claude_overall_mean(item: dict) -> float | None:
    a = item.get("assessment") or {}
    scores = a.get("scores") or {}
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_indexed_pool() -> list[dict]:
    pool: list[dict] = []
    for it in iter_pool():
        if not (it.get("body") or it.get("code")):
            continue
        mean = claude_overall_mean(it)
        if mean is None:
            continue
        it["_claude_overall_mean"] = mean
        a = it.get("assessment") or {}
        it["_claude_verdict"] = a.get("verdict")
        pool.append(it)
    return pool


def stratified_sample(pool: list[dict], n_total: int, seed: int = 0) -> list[dict]:
    """Stratified sample with non-uniform bucket handling.

    Buckets where eligible <= equal_share are taken in full (preserves rare
    classes — e.g., 0-4.99 has only 778 in 94k pool). Remaining target is
    distributed evenly across buckets that exceeded equal_share.
    """
    rng = random.Random(seed)
    eligible_per: dict[str, list[dict]] = {}
    for label, lo, hi in BUCKETS:
        eligible_per[label] = [p for p in pool if lo <= p["_claude_overall_mean"] < hi]

    equal_share = max(1, n_total // len(BUCKETS))
    out: list[dict] = []
    large_buckets: list[str] = []
    for label, _, _ in BUCKETS:
        elig = eligible_per[label]
        if len(elig) <= equal_share:
            for p in elig:
                p["_bucket"] = label
            out.extend(elig)
            note = " (100% — small bucket)" if elig else " (empty)"
            print(f"  bucket {label}: eligible={len(elig):5d}  sampled={len(elig)}{note}")
        else:
            large_buckets.append(label)

    remaining = n_total - len(out)
    if large_buckets and remaining > 0:
        per_large = remaining // len(large_buckets)
        for label in large_buckets:
            elig = list(eligible_per[label])
            rng.shuffle(elig)
            chunk = elig[:per_large]
            for p in chunk:
                p["_bucket"] = label
            out.extend(chunk)
            print(f"  bucket {label}: eligible={len(elig):5d}  sampled={len(chunk)}")

    rng.shuffle(out)  # interleave buckets for even progress
    return out


def score_one(item: dict) -> dict:
    code = item.get("body") or item.get("code") or ""
    sc = score_code(code)
    return {
        "row_id": f"{item.get('source_repo')}::{item.get('source_file')}::{item.get('function_name')}",
        "source_repo": item.get("source_repo"),
        "source_file": item.get("source_file"),
        "function_name": item.get("function_name"),
        "bucket": item["_bucket"],
        "claude_overall_mean": item["_claude_overall_mean"],
        "claude_verdict": item.get("_claude_verdict"),
        "rubric_overall": sc.overall,
        "rubric_dim_scores": sc.dimension_scores,
        "rubric_dim_na": list(sc.dimension_na),
        "triggered_checks_flat": sorted({
            cid for ids in sc.triggered_checks.values() for cid in ids
        }),
        "rubric_llm_checks_skipped": sc.llm_checks_skipped,
        "calibrated_overall": sc.calibrated_overall,
        "calibrated_verdict": sc.calibrated_verdict,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="data/phase1b/rejudge_pilot.jsonl")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already in --output (by row_id) and append. "
                        "Lets a long run survive crashes / restarts.")
    args = p.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading function pool from {PASSED_DIR.name}/ + {FAILED_DIR.name}/")
    pool = build_indexed_pool()
    print(f"  {len(pool)} functions with body + claude scores")
    if not pool:
        print("ERROR: empty pool")
        sys.exit(1)

    print(f"Stratified sample (target={args.target}, seed={args.seed}):")
    sample = stratified_sample(pool, args.target, seed=args.seed)
    print(f"  total sampled: {len(sample)}")

    already_scored: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_scored.add(json.loads(line)["row_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resume: {len(already_scored)} rows already in output, will skip")

    if already_scored:
        before = len(sample)
        def _row_id(item: dict) -> str:
            return f"{item.get('source_repo')}::{item.get('source_file')}::{item.get('function_name')}"
        sample = [s for s in sample if _row_id(s) not in already_scored]
        print(f"  filtered sample: {before} -> {len(sample)} (skipped {before - len(sample)} already-scored)")

    write_lock = threading.Lock()
    n_done = 0
    n_failed = 0
    start = time.time()

    open_mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(open_mode) as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            futs = {pool_exec.submit(score_one, item): item for item in sample}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as e:
                    n_failed += 1
                    print(f"  WARN: scoring failed: {e}")
                    continue
                with write_lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n_done += 1
                if n_done % args.progress_every == 0:
                    elapsed = time.time() - start
                    rate = n_done / max(1e-3, elapsed)
                    eta = (len(sample) - n_done) / max(1e-3, rate)
                    print(f"  [{n_done}/{len(sample)}] failed={n_failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone. wrote={n_done} failed={n_failed} elapsed={elapsed:.0f}s rate={n_done/max(1e-3, elapsed):.2f}/s")
    print(f"Output: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
