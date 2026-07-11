#!/usr/bin/env python3
"""Run a chunk of judge batch generation. Designed to be called repeatedly.

Usage: python scripts/run_judge_chunk.py --chunk-size 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_judge_batch import (
    load_source_samples,
    generate_judge_score,
    format_judge_training_example,
    validate_scores,
    JUDGE_OUTPUT,
)
from scripts.utils import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    batch = 5
    label = "high"
    expected_quality = "high"
    count = 3000
    output_file = JUDGE_OUTPUT / f"judge_training_calibrated_{label}_{batch}.json"
    checkpoint_key = f"judge_batch_{label}_{batch}"

    samples = load_source_samples("passed", count * 2, batch)

    checkpoint = load_checkpoint(checkpoint_key)
    completed_indices = set(checkpoint.get("completed", []))
    if "completed" not in checkpoint:
        checkpoint["completed"] = []

    training_examples = []
    if output_file.exists():
        try:
            training_examples = json.load(open(output_file))
        except Exception:
            pass

    start_count = len(training_examples)
    if start_count >= count:
        print(f"DONE: Already have {start_count}/{count} examples")
        _print_summary(training_examples, count)
        return

    errors = 0
    skipped = 0
    processed = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        if len(training_examples) >= count:
            break
        if processed >= args.chunk_size:
            break
        if str(i) in completed_indices:
            continue

        processed += 1
        try:
            scores = generate_judge_score(sample["code"])
        except Exception as e:
            errors += 1
            print(f"  ERR[{i}]: {e}", flush=True)
            checkpoint["completed"].append(str(i))
            continue

        if not validate_scores(scores, expected_quality):
            skipped += 1
            checkpoint["completed"].append(str(i))
            continue

        example = format_judge_training_example(
            sample["code"], scores, sample.get("source", "phase1_passed")
        )
        training_examples.append(example)
        checkpoint["completed"].append(str(i))

    # Save
    save_checkpoint(checkpoint_key, checkpoint)
    with open(output_file, "w") as f:
        json.dump(training_examples, f, indent=2)

    elapsed = time.time() - t0
    added = len(training_examples) - start_count
    print(
        f"CHUNK: +{added} examples ({start_count}->{len(training_examples)}/{count}) | "
        f"errors={errors} skip={skipped} | {elapsed:.0f}s",
        flush=True,
    )

    if len(training_examples) >= count:
        print("BATCH COMPLETE!")
        _print_summary(training_examples, count)


def _print_summary(training_examples, count):
    scores_list = [ex["metadata"]["overall_score"] for ex in training_examples]
    if not scores_list:
        return
    avg = sum(scores_list) / len(scores_list)
    lo, hi = min(scores_list), max(scores_list)
    buckets = {"<50": 0, "50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0}
    for s in scores_list:
        if s >= 90: buckets["90-100"] += 1
        elif s >= 80: buckets["80-89"] += 1
        elif s >= 70: buckets["70-79"] += 1
        elif s >= 60: buckets["60-69"] += 1
        elif s >= 50: buckets["50-59"] += 1
        else: buckets["<50"] += 1

    print(f"\n{'=' * 50}")
    print(f"FINAL: {len(training_examples)} examples")
    print(f"  Score range: {lo:.0f}-{hi:.0f} (avg {avg:.1f})")
    print(f"  Distribution:")
    for k, v in sorted(buckets.items()):
        pct = v / len(scores_list) * 100
        bar = "#" * int(pct / 2)
        print(f"    {k:>6}: {v:>5} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
