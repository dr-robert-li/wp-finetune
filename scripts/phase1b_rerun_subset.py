"""Phase 1b subset rerun — re-score a specific row_id list with patched scorer.

Built 2026-05-20 after Phase 1b 20K review (output/phase1b_disagreement_review.md)
flagged systematic SEC-N04 false positives on admin-context / REST-route code.

Reads row_ids from --row-ids-file (one row_id per line, format
"source_repo::source_file::function_name"), looks each up in the phase1
function pool, calls the patched score_code(), and writes one JSONL row per
result with the same schema as rejudge_full_20k.jsonl.

Usage:
    LLM_BACKEND=vllm \\
    LLM_VLLM_BASE_URL=http://192.168.1.61:30000/v1 \\
    LLM_VLLM_MODEL=Qwen/Qwen3.6-35B-A3B \\
    RUBRIC_USE_LLM_CHECKS=1 \\
    python -u -m scripts.phase1b_rerun_subset \\
        --row-ids-file /tmp/subset_row_ids.txt \\
        --workers 4 --resume \\
        --output data/phase1b/rerun_secn04_fix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.rubric_scorer import score_code

PASSED_DIR = ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = ROOT / "data" / "phase1_extraction" / "output" / "failed"


def build_pool_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
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
                rid = f"{it.get('source_repo')}::{it.get('source_file')}::{it.get('function_name')}"
                idx[rid] = it
    return idx


def claude_overall_mean(item: dict) -> float | None:
    a = item.get("assessment") or {}
    scores = a.get("scores") or {}
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def score_one(item: dict, row_id: str) -> dict:
    code = item.get("body") or item.get("code") or ""
    file_path = item.get("source_file") or "<generated>"
    sc = score_code(code, file_path=file_path)
    mean = claude_overall_mean(item)
    a = item.get("assessment") or {}
    return {
        "row_id": row_id,
        "source_repo": item.get("source_repo"),
        "source_file": item.get("source_file"),
        "function_name": item.get("function_name"),
        "claude_overall_mean": mean,
        "claude_verdict": a.get("verdict"),
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
    p.add_argument("--row-ids-file", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_ids = [l.strip() for l in Path(args.row_ids_file).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(row_ids)} row_ids from {args.row_ids_file}")

    print(f"Indexing function pool...")
    pool = build_pool_index()
    print(f"  {len(pool)} pool entries")

    missing = [r for r in row_ids if r not in pool]
    print(f"  missing in pool: {len(missing)}")
    items_to_score = [(rid, pool[rid]) for rid in row_ids if rid in pool]
    print(f"  to score: {len(items_to_score)}")

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
        print(f"Resume: {len(already_scored)} already scored, will skip")
        items_to_score = [(rid, it) for rid, it in items_to_score if rid not in already_scored]
        print(f"  remaining: {len(items_to_score)}")

    if not items_to_score:
        print("Nothing to do.")
        return

    write_lock = threading.Lock()
    n_done = 0
    n_failed = 0
    start = time.time()

    open_mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(open_mode) as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(score_one, it, rid): rid for rid, it in items_to_score}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as e:
                    n_failed += 1
                    print(f"  WARN: scoring failed for {futs[fut]}: {e}")
                    continue
                with write_lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    n_done += 1
                if n_done % args.progress_every == 0:
                    elapsed = time.time() - start
                    rate = n_done / max(1e-3, elapsed)
                    eta = (len(items_to_score) - n_done) / max(1e-3, rate)
                    print(f"  [{n_done}/{len(items_to_score)}] failed={n_failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone. wrote={n_done} failed={n_failed} elapsed={elapsed:.0f}s rate={n_done/max(1e-3, elapsed):.2f}/s")
    print(f"Output: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
