#!/usr/bin/env python3
"""Phase 2, Step 3: Judge synthetic examples with the same bar as Phase 1.

Uses the same judge_system.md criteria. Failed synthetic examples can be
revised once with feedback — if the revision also fails, it's discarded.
"""

import json
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
GENERATED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "generated"
JUDGED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judged"
JUDGE_SYSTEM_PATH = PROJECT_ROOT / "config" / "judge_system.md"


def load_judge_system() -> str:
    with open(JUDGE_SYSTEM_PATH) as f:
        return f.read()


def _apply_security_auto_fail(result: dict) -> dict:
    """Enforce security auto-FAIL: any security score < 5 forces verdict to FAIL."""
    if result is None:
        return result
    scores = result.get("scores", {})
    security_score = scores.get("security", 10)
    if security_score < 5:
        result["verdict"] = "FAIL"
        if "notes" in result:
            result["notes"] = f"[security auto-FAIL: score={security_score}] " + result["notes"]
        else:
            result["notes"] = f"security auto-FAIL: score={security_score}"
    return result


def judge_synthetic(code: str, gap_tag: str, client: anthropic.Anthropic, system: str) -> dict:
    """Judge a synthetic code example."""
    prompt = f"""Assess this synthetically generated WordPress code targeting the pattern: {gap_tag}

```php
{code}
```

Return your assessment as JSON matching the format in your instructions."""

    try:
        response = call_with_backoff(
            client,
            model="claude-sonnet-4-6-20250514",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        result = extract_json(text)
        if result is None:
            return None  # Caller handles None as parse failure
        result = _apply_security_auto_fail(result)
        return result
    except anthropic.APIError as e:
        print(f"    API error during judging: {e}")
        return None


def revise_with_feedback(original: dict, issues: list[str],
                         client: anthropic.Anthropic) -> str:
    """Attempt to revise a failed synthetic example using the judge's feedback."""
    prompt = f"""This WordPress code was assessed and found these issues:

Issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Original code:
```php
{original['body']}
```

Rewrite the code to fix ALL listed issues while maintaining the same functionality.
Follow WordPress Coding Standards strictly. Return only the corrected PHP code."""

    response = call_with_backoff(
        client,
        model="claude-sonnet-4-6-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    judge_system = load_judge_system()
    client = anthropic.Anthropic()

    generated_files = list(GENERATED_DIR.glob("*.json"))
    if not generated_files:
        print("No generated files found. Run phase2_generate.py first.")
        sys.exit(1)

    # Load checkpoint for resume support.
    checkpoint = load_checkpoint("phase2_judge")
    completed_files = set(checkpoint.get("completed", []))

    total_passed = 0
    total_failed = 0
    total_revised = 0

    for gen_file in generated_files:
        if gen_file.name in completed_files:
            print(f"\nSkipping {gen_file.name} (already judged)")
            continue

        print(f"\nJudging {gen_file.name}...")
        with open(gen_file) as f:
            examples = json.load(f)

        route = batch_or_direct(len(examples))

        if route == "batch":
            # Build batch judge requests.
            batch_requests = []
            for i, example in enumerate(examples):
                gap_tag = example.get("gap_tag", "unknown")
                code = example.get("body", "")
                prompt = f"""Assess this synthetically generated WordPress code targeting the pattern: {gap_tag}

```php
{code}
```

Return your assessment as JSON matching the format in your instructions."""
                batch_requests.append(
                    make_batch_request(
                        custom_id=f"{gen_file.stem}_{i}",
                        system=judge_system,
                        user_content=prompt,
                        model="claude-sonnet-4-6-20250514",
                        max_tokens=1024,
                    )
                )

            print(f"  Submitting batch of {len(batch_requests)} judge requests...")
            batch_id = submit_batch(client, batch_requests)

            # Save batch ID immediately (results expire after 24h).
            checkpoint["batch_job_ids"].append(batch_id)
            save_checkpoint("phase2_judge", checkpoint)

            results = poll_batch(client, batch_id)
            successes, failures = parse_batch_results(results)

            # Build lookup by custom_id.
            result_map = {s["_custom_id"]: s for s in successes}

            passed = []
            failed = []
            for i, example in enumerate(examples):
                custom_id = f"{gen_file.stem}_{i}"
                assessment = result_map.get(custom_id)
                if assessment is None:
                    failed.append(example)
                    total_failed += 1
                    continue
                assessment = _apply_security_auto_fail(assessment)
                example["assessment"] = assessment
                if assessment.get("verdict") == "PASS":
                    example["training_tags"] = assessment.get("training_tags", [example.get("gap_tag", "unknown")])
                    passed.append(example)
                    total_passed += 1
                else:
                    failed.append(example)
                    total_failed += 1

        else:
            passed = []
            failed = []

            for i, example in enumerate(examples):
                assessment = judge_synthetic(example.get("body", ""), example.get("gap_tag", "unknown"), client, judge_system)

                if assessment is None:
                    failed.append(example)
                    total_failed += 1
                    continue

                example["assessment"] = assessment

                if assessment.get("verdict") == "PASS":
                    example["training_tags"] = assessment.get("training_tags", [example.get("gap_tag", "unknown")])
                    passed.append(example)
                    total_passed += 1
                else:
                    # One revision attempt.
                    issues = assessment.get("critical_failures", []) + [assessment.get("notes", "")]
                    issues = [iss for iss in issues if iss]

                    if issues:
                        try:
                            revised_code = revise_with_feedback(example, issues, client)

                            # Re-judge the revision.
                            re_assessment = judge_synthetic(revised_code, example.get("gap_tag", "unknown"), client, judge_system)

                            if re_assessment is not None and re_assessment.get("verdict") == "PASS":
                                example["body"] = revised_code
                                example["assessment"] = re_assessment
                                example["revised"] = True
                                example["training_tags"] = re_assessment.get("training_tags", [example.get("gap_tag", "unknown")])
                                passed.append(example)
                                total_revised += 1
                                total_passed += 1
                            else:
                                failed.append(example)
                                total_failed += 1
                        except anthropic.APIError:
                            failed.append(example)
                            total_failed += 1
                    else:
                        failed.append(example)
                        total_failed += 1

                if (i + 1) % 10 == 0:
                    print(f"  Judged {i + 1}/{len(examples)} (passed: {len(passed)}, failed: {len(failed)})")

        # Save results.
        tag_name = gen_file.stem
        if passed:
            with open(JUDGED_DIR / f"{tag_name}_passed.json", "w") as f:
                json.dump(passed, f, indent=2)
        if failed:
            with open(JUDGED_DIR / f"{tag_name}_failed.json", "w") as f:
                json.dump(failed, f, indent=2)

        print(f"  [{tag_name}] {len(passed)} passed, {len(failed)} failed")

        # Checkpoint after each file.
        checkpoint["completed"].append(gen_file.name)
        save_checkpoint("phase2_judge", checkpoint)

    print(f"\n{'='*50}")
    print(f"Phase 2 Judging Complete")
    print(f"  Passed (first attempt): {total_passed - total_revised}")
    print(f"  Passed (after revision): {total_revised}")
    print(f"  Total passed: {total_passed}")
    print(f"  Total failed: {total_failed}")
    print(f"  Pass rate: {total_passed / max(total_passed + total_failed, 1):.1%}")
    print(f"\nRun phase3_cot.py next.")


if __name__ == "__main__":
    main()
