#!/usr/bin/env python3
"""Generate judge training data in batches using Claude Code agents.

Reuses the scoring system from phase2_judge_dataset.py but supports
batch-parallel generation. Each invocation processes a named batch of
code samples and writes output to a separate file.

Usage:
    # Generate from passed functions (high-quality, expected 75-100)
    python -m scripts.generate_judge_batch --source passed --batch 4 --count 3000

    # Generate from failed functions (low-quality, expected 10-65)
    python -m scripts.generate_judge_batch --source failed --batch 4 --count 3000

    # Generate from mixed sources (dead-zone filling, 20-100)
    python -m scripts.generate_judge_batch --source mixed --batch 2 --count 1500
"""

import argparse
import json
import random
import sys
from pathlib import Path

from scripts.claude_agent import generate_json
from scripts.utils import load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
JUDGE_OUTPUT = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judge_training"

# Reuse the exact same system prompt from phase2_judge_dataset.py
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

REQUIRED_FIELDS = {
    "wpcs_compliance", "security_score", "performance_score",
    "i18n_score", "accessibility_score", "documentation_score",
    "overall_score", "must_fix_issues", "suggested_improvements",
    "passes_threshold",
}


def load_existing_codes() -> set:
    """Load code hashes from existing judge training files to avoid duplicates."""
    existing = set()
    for f in JUDGE_OUTPUT.glob("*.json"):
        try:
            data = json.load(open(f))
            for ex in data:
                msgs = ex.get("messages", [])
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                if user:
                    existing.add(hash(user))
        except Exception:
            continue
    return existing


def load_source_samples(source: str, count: int, batch: int) -> list[dict]:
    """Load code samples from the specified source, deterministic per batch."""
    source_dir = PASSED_DIR if source == "passed" else FAILED_DIR
    existing_hashes = load_existing_codes()

    all_samples = []
    for f in sorted(source_dir.glob("*.json")):
        try:
            functions = json.load(open(f))
        except Exception:
            continue
        for func in functions:
            body = func.get("body", "")
            docblock = func.get("docblock", "")
            code = f"{docblock}\n{body}" if docblock else body
            if len(code) < 50 or len(code) > 8000:
                continue
            # Skip if already in existing training data
            user_content = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
            if hash(user_content) in existing_hashes:
                continue
            all_samples.append({
                "code": code,
                "source": f"phase1_{source}",
                "repo": f.stem,
            })

    # Deterministic shuffle per batch number
    random.seed(42 + batch * 1000)
    random.shuffle(all_samples)

    # Take the slice for this batch
    start = 0
    return all_samples[:count]


def load_mixed_samples(count: int, batch: int) -> list[dict]:
    """Load a mix of passed and failed samples for dead-zone filling."""
    passed = load_source_samples("passed", count, batch)
    failed = load_source_samples("failed", count, batch)

    # Interleave: 60% failed (for borderline scores), 40% passed
    n_failed = min(int(count * 0.6), len(failed))
    n_passed = min(count - n_failed, len(passed))

    mixed = failed[:n_failed] + passed[:n_passed]
    random.seed(42 + batch * 2000)
    random.shuffle(mixed)
    return mixed[:count]


def generate_judge_score(code: str, model: str = "sonnet") -> dict | None:
    """Have Claude produce a detailed rubric score for a code sample."""
    prompt = f"""Score this WordPress PHP code:

```php
{code[:4000]}
```

Return your rubric scores as JSON matching the format in your instructions."""

    return generate_json(prompt, system=JUDGE_SCORER_SYSTEM, model=model)


def format_judge_training_example(code: str, scores: dict, source: str) -> dict:
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
            "source": source,
        },
    }


def validate_scores(scores: dict, expected_quality: str) -> bool:
    """Validate that scores have required fields and reasonable values."""
    if not scores:
        return False

    # Check required fields
    if not REQUIRED_FIELDS.issubset(scores.keys()):
        return False

    overall = scores.get("overall_score", 0)
    if not isinstance(overall, (int, float)):
        return False

    # Sanity: high-quality code shouldn't score below 50
    if expected_quality == "high" and overall < 50:
        return False
    # Sanity: low-quality code shouldn't score above 95
    if expected_quality == "low" and overall > 95:
        return False

    # All dimension scores should be 0-100
    for field in ["wpcs_compliance", "security_score", "performance_score",
                   "i18n_score", "accessibility_score", "documentation_score"]:
        val = scores.get(field)
        if not isinstance(val, (int, float)) or val < 0 or val > 100:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate judge training data batch")
    parser.add_argument("--source", choices=["passed", "failed", "mixed"],
                        required=True, help="Source: passed (high), failed (low), mixed")
    parser.add_argument("--batch", type=int, required=True,
                        help="Batch number (for deterministic sampling)")
    parser.add_argument("--count", type=int, default=3000,
                        help="Target number of examples")
    parser.add_argument("--model", choices=["sonnet", "opus", "haiku"],
                        default="sonnet",
                        help="Claude model to use for scoring (default: sonnet)")
    args = parser.parse_args()

    JUDGE_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Determine output filename and quality label
    if args.source == "passed":
        label = "high"
        expected_quality = "high"
    elif args.source == "failed":
        label = "low"
        expected_quality = "low"
    else:
        label = "synth"
        expected_quality = "mixed"

    # Append model suffix only when non-default to keep existing batch files unchanged
    model_suffix = f"_{args.model}" if args.model != "sonnet" else ""
    output_file = JUDGE_OUTPUT / f"judge_training_calibrated_{label}_{args.batch}{model_suffix}.json"
    checkpoint_key = f"judge_batch_{label}_{args.batch}{model_suffix}"

    # Load samples
    if args.source == "mixed":
        samples = load_mixed_samples(args.count * 2, args.batch)  # oversample for rejects
    else:
        samples = load_source_samples(args.source, args.count * 2, args.batch)

    print(f"Judge Batch Generation: {label}_{args.batch}")
    print(f"{'=' * 50}")
    print(f"Source: {args.source}")
    print(f"Samples available: {len(samples)}")
    print(f"Target count: {args.count}")
    print(f"Output: {output_file}")
    print()

    # Load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_key)
    completed_indices = set(checkpoint.get("completed", []))
    if "completed" not in checkpoint:
        checkpoint["completed"] = []

    # Load existing results if resuming
    training_examples = []
    if output_file.exists():
        try:
            training_examples = json.load(open(output_file))
            print(f"Resuming: {len(training_examples)} existing examples")
        except Exception:
            pass

    errors = 0
    skipped_validation = 0

    for i, sample in enumerate(samples):
        if len(training_examples) >= args.count:
            break

        if str(i) in completed_indices:
            continue

        # Score via Claude agent
        try:
            scores = generate_judge_score(sample["code"], model=args.model)
        except Exception as e:
            errors += 1
            print(f"  [{i}] Agent error: {e}", file=sys.stderr)
            if errors > 50:
                print("Too many errors, stopping.", file=sys.stderr)
                break
            continue

        eq = expected_quality if expected_quality != "mixed" else (
            "high" if sample.get("source", "").endswith("passed") else "low"
        )

        if not validate_scores(scores, eq):
            skipped_validation += 1
            checkpoint["completed"].append(str(i))
            continue

        example = format_judge_training_example(sample["code"], scores, sample.get("source", args.source))
        training_examples.append(example)
        checkpoint["completed"].append(str(i))

        # Progress + checkpoint every 10
        if len(training_examples) % 10 == 0:
            save_checkpoint(checkpoint_key, checkpoint)
            with open(output_file, "w") as f:
                json.dump(training_examples, f, indent=2)
            print(f"  [{len(training_examples)}/{args.count}] saved (errors={errors}, skipped={skipped_validation})")

    # Final save
    save_checkpoint(checkpoint_key, checkpoint)
    with open(output_file, "w") as f:
        json.dump(training_examples, f, indent=2)

    # Score distribution summary
    scores_list = [ex["metadata"]["overall_score"] for ex in training_examples]
    if scores_list:
        avg = sum(scores_list) / len(scores_list)
        lo, hi = min(scores_list), max(scores_list)
    else:
        avg, lo, hi = 0, 0, 0

    print(f"\n{'=' * 50}")
    print(f"Batch Complete: {label}_{args.batch}")
    print(f"  Examples: {len(training_examples)}")
    print(f"  Errors: {errors}")
    print(f"  Validation skips: {skipped_validation}")
    print(f"  Score range: {lo:.0f}-{hi:.0f} (avg {avg:.1f})")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()
