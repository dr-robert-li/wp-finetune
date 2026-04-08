#!/usr/bin/env python3
"""Phase 4.1, Plan 01 Task 2: Generate deep judge CoT training examples.

Generates reasoning-enriched judge examples from Phase 1 passed/failed functions
using golden seeds as few-shot exemplars. Each example includes dimension-by-dimension
analysis with WP-specific API citations verified against source code.

Usage:
    python scripts/generate_deep_judge_cot.py --pilot
    python scripts/generate_deep_judge_cot.py --target 200
    python scripts/generate_deep_judge_cot.py  # bulk mode
"""

import json
import os
import random
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
from scripts.utils import extract_json, call_with_backoff, load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase4_reasoning"
PILOT_DIR = OUTPUT_DIR / "pilot"
BULK_DIR = OUTPUT_DIR / "deep_judge_cot"

REQUIRED_DIMENSIONS = [
    "wpcs_compliance", "sql_safety", "security", "performance",
    "wp_api_usage", "code_quality", "dependency_integrity", "i18n", "accessibility"
]

WP_API_CITATIONS = [
    "$wpdb->prepare", "wp_verify_nonce", "check_ajax_referer",
    "esc_html", "esc_attr", "esc_url", "current_user_can", "wp_kses",
    "wp_nonce_field", "sanitize_text_field", "wp_die", "absint",
    "wp_safe_redirect", "wp_create_nonce", "wp_unslash"
]

# Threshold: if >50% of cited APIs don't appear in source code, reject as hallucinated
CITATION_HALLUCINATION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Seed loading and sampling
# ---------------------------------------------------------------------------

def load_seeds() -> list:
    """Load CoT seeds from both seed files, filtered to deep_judge_cot type."""
    seeds = []
    for seed_file in [SEEDS_DIR / "ugc_seeds.json", SEEDS_DIR / "ugc_boundary_seeds.json"]:
        if seed_file.exists():
            with open(seed_file) as f:
                data = json.load(f)
            seeds.extend(s for s in data if s.get("seed_type") == "deep_judge_cot")
    return seeds


def sample_seeds(seeds: list, n: int = 3) -> list:
    """Sample n seeds with boundary seeds weighted 2x higher."""
    if len(seeds) <= n:
        return seeds[:]
    weights = [2.0 if s.get("defect_subtlety") == "boundary" else 1.0 for s in seeds]
    return random.choices(seeds, weights=weights, k=n)


# ---------------------------------------------------------------------------
# Few-shot exemplar formatting
# ---------------------------------------------------------------------------

def format_seed_as_exemplar(seed: dict) -> str:
    """Format a deep_judge_cot seed as a few-shot text exemplar."""
    code = seed.get("code", "")[:1500]
    hr = seed.get("human_reasoning", {})
    verdict = hr.get("verdict", "UNKNOWN")
    overall = hr.get("overall_score", "N/A")
    key_obs = hr.get("key_observation", "")
    da = hr.get("dimension_analysis", {})

    lines = [
        "--- EXAMPLE ---",
        f"PHP Code:\n```php\n{code}\n```",
        "",
        "Reasoning:",
        f"Verdict: {verdict}",
        f"Overall Score: {overall}",
        "",
        "Dimension Analysis:",
    ]
    for dim, info in da.items():
        score = info.get("score", "N/A")
        analysis = info.get("analysis", "")
        lines.append(f"  {dim}: score={score}")
        lines.append(f"    {analysis}")
    lines.append(f"\nKey Observation: {key_obs}")
    lines.append("--- END EXAMPLE ---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Source function loading
# ---------------------------------------------------------------------------

def load_phase1_functions() -> list:
    """Load Phase 1 passed and failed functions, filtering out tiny functions."""
    functions = []
    for source_dir, label in [(PASSED_DIR, "passed"), (FAILED_DIR, "failed")]:
        if not source_dir.exists():
            continue
        for f in source_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    entries = json.load(fh)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    body = entry.get("body", "")
                    docblock = entry.get("docblock", "")
                    code = f"{docblock}\n{body}".strip() if docblock else body
                    if len(code) < 50:
                        continue
                    functions.append({
                        "code": code,
                        "source_file": f.name,
                        "source_dir": label,
                        "function_name": entry.get("function_name", "unknown"),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    return functions


# ---------------------------------------------------------------------------
# Citation accuracy verification (addresses review concern #1)
# ---------------------------------------------------------------------------

def verify_citation_accuracy(result: dict, source_code: str) -> dict:
    """Verify that cited WP API names are grounded in source code.

    Scans dimension_analysis analysis text for WP_API_CITATIONS strings.
    For each cited API, checks if it appears in source_code OR is cited as absent
    (negative citations like "missing $wpdb->prepare" are valid).

    Returns:
        dict with total_citations, grounded_citations, hallucinated_citations list,
        hallucination_ratio.
    """
    dimension_analysis = result.get("dimension_analysis", {})
    # Collect all analysis text
    all_analysis = " ".join(
        str(info.get("analysis", ""))
        for info in dimension_analysis.values()
        if isinstance(info, dict)
    )

    # Negative-citation markers -- if these appear near an API name, citation is grounded
    negative_markers = ["missing", "not used", "should be", "lacks", "absent",
                        "not present", "need", "without", "no ", "doesn't use",
                        "does not use", "isn't using", "fails to"]

    total_citations = 0
    grounded_citations = 0
    hallucinated = []

    for api in WP_API_CITATIONS:
        if api not in all_analysis:
            continue
        total_citations += 1

        # Check if API appears in source code
        if api in source_code:
            grounded_citations += 1
            continue

        # Check if it's cited as absent (negative citation)
        api_index = all_analysis.find(api)
        context_window = all_analysis[max(0, api_index - 60):api_index + len(api) + 60].lower()
        is_negative = any(marker in context_window for marker in negative_markers)

        if is_negative:
            grounded_citations += 1
        else:
            hallucinated.append(api)

    hallucination_ratio = len(hallucinated) / total_citations if total_citations > 0 else 0.0

    return {
        "total_citations": total_citations,
        "grounded_citations": grounded_citations,
        "hallucinated_citations": hallucinated,
        "hallucination_ratio": hallucination_ratio,
    }


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def passes_quality_gate(result: dict, source_code: str = None) -> bool:
    """Check if a generated result meets quality requirements.

    Args:
        result: Parsed LLM result dict
        source_code: Optional source code for citation accuracy check
    """
    if not isinstance(result, dict):
        return False

    # Verdict must be PASS or FAIL
    if result.get("verdict") not in ("PASS", "FAIL"):
        return False

    # All 9 dimensions must be present
    da = result.get("dimension_analysis", {})
    if not isinstance(da, dict):
        return False
    missing = [d for d in REQUIRED_DIMENSIONS if d not in da]
    if missing:
        return False

    # Each dimension must have score (1-10) and analysis
    for dim, info in da.items():
        if not isinstance(info, dict):
            return False
        score = info.get("score")
        if not isinstance(score, (int, float)) or not (1 <= score <= 10):
            return False
        analysis = info.get("analysis", "")
        if not analysis or not isinstance(analysis, str):
            return False

    # overall_score must be 0-100
    overall = result.get("overall_score")
    if not isinstance(overall, (int, float)) or not (0 <= overall <= 100):
        return False

    # Citation accuracy check (if source_code provided)
    if source_code is not None:
        ca = verify_citation_accuracy(result, source_code)
        if ca["hallucination_ratio"] > CITATION_HALLUCINATION_THRESHOLD:
            return False

    return True


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_deep_judge_cot(code: str, source_info: dict, seeds: list,
                             client: anthropic.Anthropic) -> dict:
    """Generate a deep judge CoT example for a PHP code snippet.

    Args:
        code: PHP source code to analyze
        source_info: Metadata about the code source
        seeds: List of CoT seeds for few-shot context
        client: Anthropic client

    Returns:
        Parsed result dict or None on failure
    """
    sampled = sample_seeds(seeds, 3)
    exemplars = "\n\n".join(format_seed_as_exemplar(s) for s in sampled)

    prompt = f"""You are a WordPress code quality assessor producing deep reasoning chains for training data. You MUST analyze ALL 9 dimensions regardless of how many the example seeds show. Seeds only show 2-3 dimensions as examples of analysis depth, not as the complete set.

Here are golden examples of deep reasoning:

{exemplars}

NOW ANALYZE the following WordPress PHP code. Return JSON with keys:
- verdict: "PASS" or "FAIL"
- dimension_analysis: object with ALL 9 dimensions: wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility -- each having score (integer 1-10) and analysis (string citing specific WordPress APIs by name)
- overall_score: integer 0-100
- key_observation: string summarizing the most important finding

When WordPress APIs appear in the code, name them explicitly: $wpdb->prepare(), wp_verify_nonce(), esc_html(), current_user_can(), check_ajax_referer(), esc_attr(), esc_url().
When WordPress APIs are MISSING from code that needs them, state explicitly what is missing and why.
Do not describe behavior abstractly without naming the specific API.
IMPORTANT: Only cite APIs that actually appear in the code or that are demonstrably missing. Do not invent citations.

Code to analyze:
```php
{code[:3000]}
```

Return valid JSON only."""

    try:
        resp = call_with_backoff(
            client,
            model="claude-sonnet-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        result = extract_json(resp.content[0].text)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Training example formatter
# ---------------------------------------------------------------------------

def format_training_example(source_info: dict, result: dict, source_code: str = None) -> dict:
    """Assemble the final training example dict with all metadata."""
    ca = verify_citation_accuracy(result, source_code) if source_code else {
        "total_citations": 0, "grounded_citations": 0,
        "hallucinated_citations": [], "hallucination_ratio": 0.0
    }
    return {
        "source_file": source_info.get("source_file", ""),
        "source_dir": source_info.get("source_dir", ""),
        "function_name": source_info.get("function_name", ""),
        "code": source_info.get("code", ""),
        "reasoning": result,
        "dimensions_addressed": REQUIRED_DIMENSIONS[:],
        "generation_method": "seed_few_shot_agent",
        "citation_accuracy": ca,
    }


# ---------------------------------------------------------------------------
# Pilot validation
# ---------------------------------------------------------------------------

def validate_pilot_batch(examples: list) -> dict:
    """Validate dimension coverage and API citation presence across pilot examples."""
    all_dims = set()
    all_api_citations = set()
    hallucination_ratios = []

    for ex in examples:
        reasoning = ex.get("reasoning", {})
        da = reasoning.get("dimension_analysis", {})
        all_dims.update(da.keys())

        # Collect API citations from analysis text
        all_text = " ".join(
            str(info.get("analysis", "")) for info in da.values() if isinstance(info, dict)
        )
        for api in WP_API_CITATIONS:
            if api in all_text:
                all_api_citations.add(api)

        # Track hallucination ratio
        ca = ex.get("citation_accuracy", {})
        hallucination_ratios.append(ca.get("hallucination_ratio", 0.0))

    missing_dimensions = [d for d in REQUIRED_DIMENSIONS if d not in all_dims]
    api_citations_found = list(all_api_citations)
    mean_hallucination = sum(hallucination_ratios) / len(hallucination_ratios) if hallucination_ratios else 0.0

    print(f"\n=== Pilot Batch Validation ===")
    print(f"Total examples: {len(examples)}")
    print(f"Dimensions covered: {sorted(all_dims)}")
    if missing_dimensions:
        print(f"MISSING dimensions: {missing_dimensions}")
    else:
        print("All 9 dimensions covered: OK")
    print(f"WP API citations found ({len(api_citations_found)}): {sorted(api_citations_found)}")
    if mean_hallucination > 0.3:
        print(f"WARNING: Mean citation hallucination rate: {mean_hallucination:.2f} (>0.3)")
    else:
        print(f"Citation hallucination rate: {mean_hallucination:.2f} (OK)")

    return {
        "missing_dimensions": missing_dimensions,
        "api_citations_found": api_citations_found,
        "mean_hallucination_ratio": mean_hallucination,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate deep judge CoT training examples")
    parser.add_argument("--pilot", action="store_true",
                        help="Generate 40 pilot examples for human review")
    parser.add_argument("--target", type=int, default=None,
                        help="Override bulk target count")
    args = parser.parse_args()

    # Create output directories
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    BULK_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    # Load seeds
    seeds = load_seeds()
    if not seeds:
        print("ERROR: No CoT seeds found. Run seed_import.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(seeds)} deep_judge_cot seeds")

    # Load functions
    all_functions = load_phase1_functions()
    if not all_functions:
        print("ERROR: No Phase 1 functions found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(all_functions)} Phase 1 functions")

    # Determine target and output path
    if args.pilot:
        target = int(os.environ.get("PILOT_TARGET", "40"))
        output_path = PILOT_DIR / "deep_judge_cot_pilot.json"
        checkpoint_key = "generate_deep_judge_cot_pilot"
    else:
        target = args.target if args.target is not None else int(len(all_functions) * 0.5)
        output_path = BULK_DIR / "deep_judge_cot.json"
        checkpoint_key = "generate_deep_judge_cot"

    print(f"Target: {target} examples")
    print(f"Output: {output_path}")

    # Load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_key)
    completed_ids = set(checkpoint.get("completed", []))

    # Shuffle deterministically based on target
    random.seed(42)
    random.shuffle(all_functions)

    examples = []
    parse_attempts = 0
    parse_failures = 0
    citation_rejections = 0

    for i, func in enumerate(all_functions):
        if len(examples) >= target:
            break

        func_id = f"{func['source_file']}::{func['function_name']}::{i}"
        if func_id in completed_ids:
            continue

        parse_attempts += 1
        result = generate_deep_judge_cot(func["code"], func, seeds, client)

        if result is None:
            parse_failures += 1
            checkpoint.setdefault("failed", []).append(func_id)
            continue

        # Citation accuracy check
        ca = verify_citation_accuracy(result, func["code"])
        if ca["hallucination_ratio"] > CITATION_HALLUCINATION_THRESHOLD:
            citation_rejections += 1
            print(f"  Citation rejection: hallucination_ratio={ca['hallucination_ratio']:.2f}")
            continue

        if not passes_quality_gate(result, func["code"]):
            parse_failures += 1
            continue

        example = format_training_example(func, result, func["code"])
        examples.append(example)
        checkpoint.setdefault("completed", []).append(func_id)

        if len(examples) % 5 == 0:
            print(f"  Generated {len(examples)}/{target} examples "
                  f"(parse_fail={parse_failures}, citation_rej={citation_rejections})")

        # Save checkpoint every 20 examples
        if len(examples) % 20 == 0:
            save_checkpoint(checkpoint_key, checkpoint)

    # Final checkpoint save
    save_checkpoint(checkpoint_key, checkpoint)

    # Save output
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Deep Judge CoT Generation Complete")
    print(f"  Total generated: {len(examples)}")
    print(f"  Parse attempts: {parse_attempts}")
    print(f"  Parse failures: {parse_failures} ({parse_failures/max(parse_attempts,1)*100:.1f}%)")
    print(f"  Citation rejections: {citation_rejections}")
    print(f"  Saved to: {output_path}")

    if args.pilot and examples:
        validation = validate_pilot_batch(examples)
        if validation["missing_dimensions"]:
            print(f"\nERROR: Missing dimensions: {validation['missing_dimensions']}")
            sys.exit(1)
        if len(validation["api_citations_found"]) < 3:
            print(f"\nERROR: Only {len(validation['api_citations_found'])} WP API citations found (need >= 3)")
            sys.exit(1)
        print("\nPilot validation PASSED")


if __name__ == "__main__":
    main()
