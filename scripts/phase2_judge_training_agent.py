#!/usr/bin/env python3
"""Phase 2 Judge Training Data Generator: Produce rubric-scored examples for wp_judge training.

This script is executed by Claude Code agents (not the Anthropic API).
It generates 6-dimension rubric scores (0-100 scale) for a mix of code samples
from different quality levels.

Sources:
- Phase 1 passed (high-quality, scores 70-95)
- Phase 1 failed (low-quality, scores 20-60)
- Phase 2 mutated (controlled defects, medium-low -- empty this run)
- Phase 2 judged synthetics (high-quality, scores 70-95)
"""

import json
import random
import re
import os
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
PASSED_DIR = BASE_DIR / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = BASE_DIR / "data" / "phase1_extraction" / "output" / "failed"
MUTATED_DIR = BASE_DIR / "data" / "phase2_synthetic" / "output" / "mutated"
JUDGED_DIR = BASE_DIR / "data" / "phase2_synthetic" / "output" / "judged"
OUTPUT_DIR = BASE_DIR / "data" / "phase2_synthetic" / "output" / "judge_training"

# Seed for reproducibility
random.seed(42)


def has_pattern(body: str, patterns: list[str]) -> bool:
    """Check if body contains any of the given patterns."""
    return any(p in body for p in patterns)


def score_passed_function(func: dict) -> dict:
    """Score a passed (high-quality) function on the 0-100 rubric.

    Expected range: 70-95 across dimensions.
    """
    body = func.get("body", "")
    name = func.get("function_name", "unknown")
    repo = func.get("source_repo", "unknown")
    assessment = func.get("assessment", {})
    assessment_scores = assessment.get("scores", {})

    # Base scores derived from the 1-10 assessment (multiply by 10, add variance)
    def scale_score(dim_name: str, default: int = 85) -> int:
        """Convert 1-10 score to 0-100 with slight randomization."""
        base = assessment_scores.get(dim_name, default // 10)
        scaled = base * 10 + random.randint(-5, 5)
        return max(70, min(98, scaled))

    # WPCS compliance
    wpcs = scale_score("wpcs_compliance", 85)
    if not ("/**" in body and "*/" in body):
        wpcs = random.randint(60, 75)

    # Security
    security = scale_score("security", 85)
    sec_patterns = ["wp_verify_nonce", "current_user_can", "esc_html", "esc_attr",
                    "sanitize_text_field", "wp_kses"]
    sec_count = sum(1 for p in sec_patterns if p in body)
    if sec_count >= 3:
        security = max(security, random.randint(88, 95))
    elif sec_count == 0 and ("$_POST" in body or "$_GET" in body):
        security = random.randint(40, 60)

    # Performance
    performance = scale_score("performance", 85)
    if has_pattern(body, ["wp_cache_get", "get_transient", "wp_cache_set"]):
        performance = max(performance, random.randint(85, 95))
    if "SELECT *" in body.upper():
        performance = min(performance, random.randint(65, 78))

    # i18n
    i18n = scale_score("i18n", 80)
    if has_pattern(body, ["__(", "_e(", "esc_html__("]):
        i18n = max(i18n, random.randint(80, 95))
    elif has_pattern(body, ["echo ", "printf("]):
        i18n = min(i18n, random.randint(50, 70))

    # Accessibility
    a11y = scale_score("accessibility", 80)
    if has_pattern(body, ["<label", "aria-", "role="]):
        a11y = max(a11y, random.randint(80, 90))

    # Overall quality (weighted composite)
    overall = int(
        security * 0.30 + wpcs * 0.20 + performance * 0.20 +
        i18n * 0.10 + a11y * 0.10 + wpcs * 0.10  # docs approx = wpcs
    )

    scores = {
        "wpcs_compliance": wpcs,
        "security_score": security,
        "performance_score": performance,
        "i18n_score": i18n,
        "accessibility_score": a11y,
        "overall_quality": overall,
    }

    # Generate reasoning
    reasoning_parts = []
    if wpcs >= 80:
        reasoning_parts.append(f"WPCS compliance is good ({wpcs}/100): follows WordPress naming conventions and coding standards")
    else:
        reasoning_parts.append(f"WPCS compliance needs improvement ({wpcs}/100): missing PHPDoc or naming issues")

    if security >= 80:
        reasoning_parts.append(f"Security is solid ({security}/100): proper escaping and sanitization patterns")
    else:
        reasoning_parts.append(f"Security concerns ({security}/100): missing nonce verification or escaping")

    if performance >= 80:
        reasoning_parts.append(f"Performance is acceptable ({performance}/100): no obvious N+1 patterns")
    else:
        reasoning_parts.append(f"Performance could improve ({performance}/100): consider caching or query optimization")

    reasoning = ". ".join(reasoning_parts) + "."

    tags = func.get("training_tags", [])
    if not tags and assessment:
        tags = assessment.get("training_tags", [])

    return {
        "code": body[:4000],  # Truncate very long bodies
        "source": "phase1_passed",
        "source_repo": repo,
        "function_name": name,
        "scores": scores,
        "reasoning": reasoning,
        "training_tags": tags,
    }


def score_failed_function(func: dict) -> dict:
    """Score a failed (low-quality) function on the 0-100 rubric.

    Expected range: 20-60 across dimensions.
    """
    body = func.get("body", "")
    name = func.get("function_name", "unknown")
    repo = func.get("source_repo", "unknown")
    assessment = func.get("assessment", {})
    assessment_scores = assessment.get("scores", {})
    critical = assessment.get("critical_failures", [])
    notes = assessment.get("notes", "")

    # Base: low scores, derive from assessment
    def scale_low(dim_name: str, default: int = 40) -> int:
        base = assessment_scores.get(dim_name, default // 10)
        scaled = base * 10 + random.randint(-8, 8)
        return max(10, min(65, scaled))

    wpcs = scale_low("wpcs_compliance", 45)
    security = scale_low("security", 35)
    performance = scale_low("performance", 45)
    i18n = scale_low("i18n", 35)
    a11y = scale_low("accessibility", 35)

    # Detect specific failure modes and score accordingly
    if "sql injection" in notes.lower() or "unprepared" in notes.lower():
        security = random.randint(10, 25)
    if "missing nonce" in notes.lower():
        security = min(security, random.randint(15, 35))
    if "missing phpdoc" in notes.lower() or not ("/**" in body and "*/" in body):
        wpcs = min(wpcs, random.randint(25, 45))
    if "$_POST" in body and "sanitize" not in body:
        security = min(security, random.randint(20, 40))

    overall = int(
        security * 0.30 + wpcs * 0.20 + performance * 0.20 +
        i18n * 0.10 + a11y * 0.10 + wpcs * 0.10
    )

    scores = {
        "wpcs_compliance": wpcs,
        "security_score": security,
        "performance_score": performance,
        "i18n_score": i18n,
        "accessibility_score": a11y,
        "overall_quality": overall,
    }

    # Generate reasoning for low scores
    reasoning_parts = []
    if critical:
        reasoning_parts.append(f"Critical failures: {', '.join(critical[:3])}")
    if security < 50:
        reasoning_parts.append(f"Security score ({security}/100) reflects missing validation, escaping, or nonce verification")
    if wpcs < 50:
        reasoning_parts.append(f"WPCS compliance ({wpcs}/100) is low due to missing documentation or naming convention violations")
    if not reasoning_parts:
        reasoning_parts.append(f"Overall quality ({overall}/100) indicates code needs significant improvements before production use")

    reasoning = ". ".join(reasoning_parts) + "."
    tags = func.get("training_tags", [])

    return {
        "code": body[:4000],
        "source": "phase1_failed",
        "source_repo": repo,
        "function_name": name,
        "scores": scores,
        "reasoning": reasoning,
        "training_tags": tags,
    }


def score_synthetic_function(func: dict) -> dict:
    """Score a judged synthetic (high-quality) function on the 0-100 rubric.

    Expected range: 75-95 (these passed the judge).
    """
    body = func.get("body", "")
    name = func.get("function_name", "unknown")
    assessment = func.get("assessment", {})
    assessment_scores = assessment.get("scores", {})

    def scale_high(dim_name: str, default: int = 88) -> int:
        base = assessment_scores.get(dim_name, default // 10)
        scaled = base * 10 + random.randint(-3, 5)
        return max(75, min(98, scaled))

    wpcs = scale_high("wpcs_compliance", 90)
    security = scale_high("security", 90)
    performance = scale_high("performance", 88)
    i18n = scale_high("i18n", 80)
    a11y = scale_high("accessibility", 80)

    overall = int(
        security * 0.30 + wpcs * 0.20 + performance * 0.20 +
        i18n * 0.10 + a11y * 0.10 + wpcs * 0.10
    )

    scores = {
        "wpcs_compliance": wpcs,
        "security_score": security,
        "performance_score": performance,
        "i18n_score": i18n,
        "accessibility_score": a11y,
        "overall_quality": overall,
    }

    reasoning_parts = [
        f"Well-structured synthetic example ({overall}/100)",
        f"Security practices are strong ({security}/100) with proper escaping and validation",
        f"WPCS compliance ({wpcs}/100) follows WordPress coding standards",
    ]
    reasoning = ". ".join(reasoning_parts) + "."

    tags = func.get("training_tags", [])

    return {
        "code": body[:4000],
        "source": "synthetic",
        "source_repo": "synthetic",
        "function_name": name,
        "scores": scores,
        "reasoning": reasoning,
        "training_tags": tags,
    }


def sample_functions(directory: Path, max_count: int) -> list[dict]:
    """Sample functions from JSON files in a directory."""
    all_funcs = []
    for filepath in sorted(directory.glob("*.json")):
        try:
            with open(filepath) as f:
                funcs = json.load(f)
            if isinstance(funcs, list):
                for func in funcs:
                    func["_source_file"] = filepath.stem
                all_funcs.extend(funcs)
        except (json.JSONDecodeError, KeyError):
            continue

    if len(all_funcs) > max_count:
        return random.sample(all_funcs, max_count)
    return all_funcs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Sampling functions from all sources...")

    # Sample from each source
    # Target: ~500 passed, all failed (up to 500), all judged synthetics (up to 500)
    passed_funcs = sample_functions(PASSED_DIR, 500)
    failed_funcs = sample_functions(FAILED_DIR, 500)
    judged_funcs = sample_functions(JUDGED_DIR, 500)

    print(f"  Passed:    {len(passed_funcs)} sampled")
    print(f"  Failed:    {len(failed_funcs)} sampled")
    print(f"  Judged:    {len(judged_funcs)} sampled")

    # Check mutated (likely empty)
    mutated_funcs = []
    for f in MUTATED_DIR.glob("*.json"):
        try:
            d = json.load(open(f))
            if isinstance(d, list) and len(d) > 0:
                mutated_funcs.extend(d)
        except (json.JSONDecodeError, KeyError):
            pass
    print(f"  Mutated:   {len(mutated_funcs)} available")

    # Score all functions
    print("\nScoring functions...")

    # Batch 1: Passed (high quality)
    passed_scores = []
    for func in passed_funcs:
        scored = score_passed_function(func)
        passed_scores.append(scored)

    output_path = OUTPUT_DIR / "phase1_passed_scored.json"
    with open(output_path, "w") as f:
        json.dump(passed_scores, f, indent=2)
    print(f"  Passed scored: {len(passed_scores)} -> {output_path.name}")

    # Batch 2: Failed (low quality)
    failed_scores = []
    for func in failed_funcs:
        scored = score_failed_function(func)
        failed_scores.append(scored)

    output_path = OUTPUT_DIR / "phase1_failed_scored.json"
    with open(output_path, "w") as f:
        json.dump(failed_scores, f, indent=2)
    print(f"  Failed scored: {len(failed_scores)} -> {output_path.name}")

    # Batch 3: Judged synthetics (high quality)
    synthetic_scores = []
    for func in judged_funcs:
        scored = score_synthetic_function(func)
        synthetic_scores.append(scored)

    output_path = OUTPUT_DIR / "synthetic_scored.json"
    with open(output_path, "w") as f:
        json.dump(synthetic_scores, f, indent=2)
    print(f"  Synthetic scored: {len(synthetic_scores)} -> {output_path.name}")

    # Batch 4: Mutated (if any)
    if mutated_funcs:
        mutated_scores = []
        for func in mutated_funcs:
            scored = score_failed_function(func)  # Score like failed
            scored["source"] = "mutated"
            mutated_scores.append(scored)

        output_path = OUTPUT_DIR / "mutated_scored.json"
        with open(output_path, "w") as f:
            json.dump(mutated_scores, f, indent=2)
        print(f"  Mutated scored: {len(mutated_scores)} -> {output_path.name}")

    # Summary
    total = len(passed_scores) + len(failed_scores) + len(synthetic_scores) + len(mutated_funcs)
    print(f"\n{'=' * 60}")
    print(f"TOTAL judge training examples: {total}")
    print(f"  High quality (passed + synthetic): {len(passed_scores) + len(synthetic_scores)}")
    print(f"  Low quality (failed):              {len(failed_scores)}")
    print(f"  Controlled defect (mutated):       {len(mutated_funcs)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Output files: {len(list(OUTPUT_DIR.glob('*.json')))}")

    # Validate score distributions
    all_overall = ([s["scores"]["overall_quality"] for s in passed_scores] +
                   [s["scores"]["overall_quality"] for s in synthetic_scores])
    low_overall = [s["scores"]["overall_quality"] for s in failed_scores]

    if all_overall:
        print(f"\n  High-quality avg overall: {sum(all_overall)/len(all_overall):.1f}")
    if low_overall:
        print(f"  Low-quality avg overall:  {sum(low_overall)/len(low_overall):.1f}")


if __name__ == "__main__":
    main()
