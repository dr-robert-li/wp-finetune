#!/usr/bin/env python3
"""Standalone runner for judge batch 5 - saves every 10 examples for safety."""

import json
import sys
import time
from pathlib import Path

# Add project root to path
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

batch = 5
label = "high"
expected_quality = "high"
count = 3000
output_file = JUDGE_OUTPUT / f"judge_training_calibrated_{label}_{batch}.json"
checkpoint_key = f"judge_batch_{label}_{batch}"

samples = load_source_samples("passed", count * 2, batch)
print(f"Judge Batch Generation: {label}_{batch}", flush=True)
print(f"Samples available: {len(samples)}", flush=True)
print(f"Target count: {count}", flush=True)
print(f"Output: {output_file}", flush=True)

checkpoint = load_checkpoint(checkpoint_key)
completed_indices = set(checkpoint.get("completed", []))
if "completed" not in checkpoint:
    checkpoint["completed"] = []

training_examples = []
if output_file.exists():
    try:
        training_examples = json.load(open(output_file))
        print(f"Resuming: {len(training_examples)} existing examples", flush=True)
    except Exception:
        pass

errors = 0
skipped_validation = 0
start_time = time.time()

for i, sample in enumerate(samples):
    if len(training_examples) >= count:
        break
    if str(i) in completed_indices:
        continue

    try:
        scores = generate_judge_score(sample["code"])
    except Exception as e:
        errors += 1
        print(f"  [{i}] Agent error: {e}", flush=True)
        if errors > 50:
            print("Too many errors, stopping.", flush=True)
            break
        continue

    if not validate_scores(scores, expected_quality):
        skipped_validation += 1
        checkpoint["completed"].append(str(i))
        continue

    example = format_judge_training_example(
        sample["code"], scores, sample.get("source", "phase1_passed")
    )
    training_examples.append(example)
    checkpoint["completed"].append(str(i))

    n = len(training_examples)

    # Save every 10 examples (more frequent for safety)
    if n % 10 == 0:
        save_checkpoint(checkpoint_key, checkpoint)
        with open(output_file, "w") as f:
            json.dump(training_examples, f, indent=2)

        elapsed = time.time() - start_time
        rate = n / elapsed * 3600 if elapsed > 0 else 0
        eta_h = (count - n) / (rate) if rate > 0 else 0
        print(
            f"  [{n}/{count}] saved | errors={errors} skip={skipped_validation} | "
            f"{rate:.0f}/hr | ETA {eta_h:.1f}h",
            flush=True,
        )

# Final save
save_checkpoint(checkpoint_key, checkpoint)
with open(output_file, "w") as f:
    json.dump(training_examples, f, indent=2)

scores_list = [ex["metadata"]["overall_score"] for ex in training_examples]
if scores_list:
    avg = sum(scores_list) / len(scores_list)
    lo, hi = min(scores_list), max(scores_list)
    # Distribution buckets
    buckets = {"50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0}
    for s in scores_list:
        if s >= 90: buckets["90-100"] += 1
        elif s >= 80: buckets["80-89"] += 1
        elif s >= 70: buckets["70-79"] += 1
        elif s >= 60: buckets["60-69"] += 1
        elif s >= 50: buckets["50-59"] += 1
else:
    avg, lo, hi = 0, 0, 0
    buckets = {}

elapsed = time.time() - start_time
print(f"\n{'=' * 50}", flush=True)
print(f"Batch Complete: {label}_{batch}", flush=True)
print(f"  Examples: {len(training_examples)}", flush=True)
print(f"  Errors: {errors}", flush=True)
print(f"  Validation skips: {skipped_validation}", flush=True)
print(f"  Score range: {lo:.0f}-{hi:.0f} (avg {avg:.1f})", flush=True)
print(f"  Duration: {elapsed/3600:.1f}h", flush=True)
if buckets:
    print(f"  Distribution:", flush=True)
    for k, v in sorted(buckets.items()):
        pct = v / len(scores_list) * 100
        print(f"    {k}: {v} ({pct:.1f}%)", flush=True)
print(f"  Saved to: {output_file}", flush=True)
