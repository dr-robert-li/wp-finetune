#!/usr/bin/env python3
"""Phase 2, Step 4: Generate judge training data for <wp_judge> mode.

Creates (code, rubric_scores) pairs for training the model's judge capability.
Sources:
1. Phase 1 passed code (high-scoring examples)
2. Phase 1 failed code (low-scoring examples with real defects)
3. Automated mutations from phase2_mutate.py (controlled-defect examples)
4. Claude-scored examples for silver-annotated ground truth

Outputs judge training data to phase2_synthetic/output/judge_training/
"""

import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic

from scripts.utils import (
    extract_json,
    call_with_backoff,
    load_checkpoint,
    save_checkpoint,
    batch_or_direct,
    make_batch_request,
    submit_batch,
    poll_batch,
    parse_batch_results,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASSED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "failed"
MUTATED_DIR = PROJECT_ROOT / "phase2_synthetic" / "output" / "mutated"
JUDGE_OUTPUT = PROJECT_ROOT / "phase2_synthetic" / "output" / "judge_training"

JUDGE_SCORER_SYSTEM = """You are a WordPress code quality scoring system. You produce structured
rubric scores for PHP code samples. Your scores will be used to train a judge model, so accuracy
and consistency are critical.

Score each dimension 0-100:
- wpcs_compliance: WordPress Coding Standards adherence
- security_score: Nonces, capabilities, escaping, sanitization, prepared statements
- performance_score: Query efficiency, caching, no N+1, targeted selects
- i18n_score: Translation function usage, text domain, pluralization
- accessibility_score: Semantic HTML, labels, ARIA, keyboard support
- documentation_score: PHPDoc coverage, @param/@return/@since

Also provide:
- overall_score: Weighted composite (security 30%, wpcs 20%, performance 20%, i18n 10%, a11y 10%, docs 10%)
- must_fix_issues: Array of critical problems (empty if none)
- suggested_improvements: Array of non-critical recommendations
- passes_threshold: true if overall_score >= 80 AND no critical security issues
- explanation: 2-3 sentence summary of the verdict

Return valid JSON only."""


def generate_judge_score(code: str, client: anthropic.Anthropic) -> dict:
    """Have Claude produce a detailed rubric score for a code sample."""
    prompt = f"""Score this WordPress PHP code:

```php
{code[:4000]}
```

Return your rubric scores as JSON matching the format in your instructions."""

    try:
        response = call_with_backoff(
            client,
            model="claude-sonnet-4-6-20250514",
            max_tokens=1024,
            system=JUDGE_SCORER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return extract_json(text)
    except anthropic.APIError:
        return None


def format_judge_training_example(code: str, scores: dict) -> dict:
    """Format as a <wp_judge> training example."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```",
            },
            {
                "role": "assistant",
                "content": json.dumps(scores, indent=2),
            },
        ],
        "metadata": {
            "task_type": "judge",
            "overall_score": scores.get("overall_score", 0),
            "passes_threshold": scores.get("passes_threshold", False),
        },
    }


def load_code_samples() -> list[dict]:
    """Load diverse code samples from all pipeline sources."""
    samples = []

    # High-quality passed code (expected score 80+).
    for f in PASSED_DIR.glob("*.json"):
        with open(f) as fh:
            functions = json.load(fh)
        for func in functions:
            body = func.get("body", "")
            docblock = func.get("docblock", "")
            code = f"{docblock}\n{body}" if docblock else body
            if len(code) > 50:
                samples.append({
                    "code": code,
                    "expected_quality": "high",
                    "source": "phase1_passed",
                })

    # Failed code (expected score < 80, real defects).
    for f in FAILED_DIR.glob("*.json"):
        with open(f) as fh:
            functions = json.load(fh)
        for func in functions:
            body = func.get("body", "")
            docblock = func.get("docblock", "")
            code = f"{docblock}\n{body}" if docblock else body
            if len(code) > 50:
                samples.append({
                    "code": code,
                    "expected_quality": "low",
                    "source": "phase1_failed",
                })

    # Mutated code (controlled defects, expected low score on specific dimensions).
    mutations_path = MUTATED_DIR / "contrastive_mutations.json"
    if mutations_path.exists():
        with open(mutations_path) as f:
            mutations = json.load(f)
        for m in mutations:
            # Score the BAD version — judge should catch the defects.
            samples.append({
                "code": m["bad_code"],
                "expected_quality": "low",
                "source": "automated_mutation",
                "mutation_type": m.get("mutation_type"),
            })
            # Also score the GOOD version for calibration.
            samples.append({
                "code": m["good_code"],
                "expected_quality": "high",
                "source": "mutation_original",
            })

    return samples


def _score_batch(samples: list[dict], client: anthropic.Anthropic,
                 judge_system: str) -> list[dict]:
    """Score a list of samples via batch API. Returns list of (sample, scores) tuples."""
    batch_requests = []
    for i, sample in enumerate(samples):
        prompt = f"""Score this WordPress PHP code:

```php
{sample['code'][:4000]}
```

Return your rubric scores as JSON matching the format in your instructions."""
        batch_requests.append(
            make_batch_request(
                custom_id=f"sample_{i}",
                system=judge_system,
                user_content=prompt,
                model="claude-sonnet-4-6-20250514",
                max_tokens=1024,
            )
        )

    print(f"  Submitting batch of {len(batch_requests)} scoring requests...")
    batch_id = submit_batch(client, batch_requests)

    # Save batch results immediately (they expire after 24h — Pitfall 3).
    results_cache_path = JUDGE_OUTPUT / f"batch_{batch_id}_raw.json"
    JUDGE_OUTPUT.mkdir(parents=True, exist_ok=True)
    results = poll_batch(client, batch_id)

    # Cache raw results to disk immediately.
    with open(results_cache_path, "w") as f:
        json.dump([r.model_dump() if hasattr(r, "model_dump") else str(r) for r in results], f, indent=2)

    successes, failures = parse_batch_results(results)
    result_map = {s["_custom_id"]: s for s in successes}

    scored = []
    for i, sample in enumerate(samples):
        custom_id = f"sample_{i}"
        scores = result_map.get(custom_id)
        if scores is not None:
            scores.pop("_custom_id", None)
            scored.append((sample, scores))

    return scored


def main():
    JUDGE_OUTPUT.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    samples = load_code_samples()

    if not samples:
        print("No code samples found. Run Phase 1 and phase2_mutate.py first.")
        sys.exit(1)

    print(f"Judge Dataset Generation")
    print(f"{'='*50}")
    print(f"Total code samples: {len(samples)}")
    print(f"  High quality (passed): {sum(1 for s in samples if s['expected_quality'] == 'high')}")
    print(f"  Low quality (failed/mutated): {sum(1 for s in samples if s['expected_quality'] == 'low')}")
    print()

    # Target: balanced dataset, cap per source.
    max_per_source = {
        "phase1_passed": 1500,
        "phase1_failed": 1000,
        "automated_mutation": 1000,
        "mutation_original": 500,
    }

    random.seed(42)
    random.shuffle(samples)

    # Load checkpoint for resume support.
    checkpoint = load_checkpoint("phase2_judge_dataset")
    completed_indices = set(checkpoint.get("completed", []))

    source_counts = {}
    training_examples = []
    skipped = 0

    # Filter samples to process (apply source caps, skip already done).
    pending_samples = []
    for i, sample in enumerate(samples):
        source = sample["source"]
        source_counts.setdefault(source, 0)

        if str(i) in completed_indices:
            skipped += 1
            continue

        if source_counts[source] >= max_per_source.get(source, 500):
            skipped += 1
            continue

        pending_samples.append((i, sample))
        source_counts[source] += 1

    print(f"Processing {len(pending_samples)} samples (skipped {skipped} capped/completed)")

    route = batch_or_direct(len(pending_samples))

    if route == "batch" and pending_samples:
        # Score all pending samples via batch API.
        just_samples = [s for _, s in pending_samples]
        scored = _score_batch(just_samples, client, JUDGE_SCORER_SYSTEM)

        for (i, sample), (_, scores) in zip(pending_samples, scored):
            if scores is None:
                continue

            # Sanity check: high-quality code should score high.
            overall = scores.get("overall_score", 0)
            if sample["expected_quality"] == "high" and overall < 50:
                continue
            if sample["expected_quality"] == "low" and overall > 95:
                continue

            example = format_judge_training_example(sample["code"], scores)
            example["metadata"]["source"] = sample["source"]
            training_examples.append(example)

            checkpoint["completed"].append(str(i))

        # Save checkpoint after batch.
        save_checkpoint("phase2_judge_dataset", checkpoint)

    else:
        # Direct mode: score one at a time with call_with_backoff.
        for batch_start in range(0, len(pending_samples), 100):
            batch_slice = pending_samples[batch_start:batch_start + 100]

            for i, sample in batch_slice:
                scores = generate_judge_score(sample["code"], client)
                if scores is None:
                    continue

                # Sanity check: high-quality code should score high.
                overall = scores.get("overall_score", 0)
                if sample["expected_quality"] == "high" and overall < 50:
                    continue
                if sample["expected_quality"] == "low" and overall > 95:
                    continue

                example = format_judge_training_example(sample["code"], scores)
                example["metadata"]["source"] = sample["source"]
                training_examples.append(example)

                checkpoint["completed"].append(str(i))

            # Save checkpoint every 100 examples.
            save_checkpoint("phase2_judge_dataset", checkpoint)
            print(f"  Checkpoint saved: {len(training_examples)} examples so far")

    # Save final output.
    output_path = JUDGE_OUTPUT / "judge_training.json"
    with open(output_path, "w") as f:
        json.dump(training_examples, f, indent=2)

    # Recount source distribution from actual examples.
    final_source_counts = {}
    for ex in training_examples:
        src = ex["metadata"].get("source", "unknown")
        final_source_counts[src] = final_source_counts.get(src, 0) + 1

    print(f"\n{'='*50}")
    print(f"Judge Training Dataset Complete")
    print(f"  Total examples: {len(training_examples)}")
    for source, count in sorted(final_source_counts.items()):
        print(f"    {source}: {count}")
    print(f"  Saved to: {output_path}")
    print(f"\nRun phase3_cot.py next.")


if __name__ == "__main__":
    main()
