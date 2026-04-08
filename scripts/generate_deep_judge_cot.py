#!/usr/bin/env python3
"""Phase 4.1: Generate deep judge CoT training examples.

Uses golden seeds as few-shot exemplars to generate dimension-by-dimension
reasoning for Phase 1 passed/failed PHP functions. Each generated example
includes all 9 rubric dimensions with scores and WP API citations.

Usage:
    python scripts/generate_deep_judge_cot.py --pilot       # 40 examples
    python scripts/generate_deep_judge_cot.py --target 200  # custom bulk target
    python scripts/generate_deep_judge_cot.py               # default bulk (~50% of functions)

Output:
    data/phase4_reasoning/pilot/deep_judge_cot_pilot.json  (pilot mode)
    data/phase4_reasoning/deep_judge_cot/deep_judge_cot_bulk.json  (bulk mode)
"""
import argparse
import json
import random
import sys
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_DIMENSIONS = [
    "wpcs_compliance",
    "sql_safety",
    "security",
    "performance",
    "wp_api_usage",
    "code_quality",
    "dependency_integrity",
    "i18n",
    "accessibility",
]

WP_API_CITATIONS = [
    "$wpdb->prepare",
    "wp_verify_nonce",
    "check_ajax_referer",
    "esc_html",
    "esc_attr",
    "esc_url",
    "current_user_can",
    "wp_kses",
    "wp_nonce_field",
    "sanitize_text_field",
    "wp_die",
    "absint",
    "wp_safe_redirect",
    "wp_create_nonce",
    "wp_unslash",
]

# Threshold: if >50% of cited APIs don't appear in source code, reject as hallucinated
CITATION_HALLUCINATION_THRESHOLD = 0.5

CHECKPOINT_INTERVAL = 20  # Save checkpoint every N examples


# ---------------------------------------------------------------------------
# Seed loading and sampling
# ---------------------------------------------------------------------------


def load_seeds() -> list[dict]:
    """Load all deep_judge_cot seeds from both seed files.

    Returns a list of seed dicts, all with seed_type == 'deep_judge_cot'.
    """
    seeds = []
    for filename in ["ugc_seeds.json", "ugc_boundary_seeds.json"]:
        path = SEEDS_DIR / filename
        if not path.exists():
            print(f"WARNING: Seed file not found: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cot_seeds = [s for s in data if s.get("seed_type") == "deep_judge_cot"]
        seeds.extend(cot_seeds)
    return seeds


def sample_seeds(seeds: list[dict], n: int = 3) -> list[dict]:
    """Sample n seeds with boundary seeds weighted 2x higher.

    Uses random.choices with weights so boundary seeds appear more often
    as few-shot exemplars, improving coverage of subtle defects.
    """
    if not seeds:
        return []
    n = min(n, len(seeds))
    weights = [
        2.0 if s.get("defect_subtlety") == "boundary" else 1.0
        for s in seeds
    ]
    # random.choices can return duplicates; use a set trick to get unique samples
    seen_ids: set[str] = set()
    selected: list[dict] = []
    attempts = 0
    while len(selected) < n and attempts < n * 20:
        attempts += 1
        [candidate] = random.choices(seeds, weights=weights, k=1)
        sid = candidate.get("seed_id", id(candidate))
        if sid not in seen_ids:
            seen_ids.add(sid)
            selected.append(candidate)
    return selected


# ---------------------------------------------------------------------------
# Few-shot exemplar formatting
# ---------------------------------------------------------------------------


def format_seed_as_exemplar(seed: dict) -> str:
    """Format a deep_judge_cot seed as a text exemplar for the few-shot prompt.

    Shows code snippet (truncated to 1500 chars) followed by human reasoning:
    verdict, dimension_analysis, overall_score, key_observation.
    """
    code = seed.get("code", "")[:1500]
    hr = seed.get("human_reasoning", {})

    lines = [
        "--- EXEMPLAR ---",
        f"Code:\n```php\n{code}\n```",
        "",
        "Reasoning:",
        f"Verdict: {hr.get('verdict', 'UNKNOWN')}",
        "",
        "Dimension Analysis:",
    ]

    dim_analysis = hr.get("dimension_analysis", {})
    for dim, details in dim_analysis.items():
        if isinstance(details, dict):
            score = details.get("score", "?")
            analysis = details.get("analysis", "")[:300]
            lines.append(f"  {dim}: score={score}/10 — {analysis}")

    lines.extend([
        "",
        f"Overall Score: {hr.get('overall_score', '?')}/100",
        f"Key Observation: {hr.get('key_observation', '')}",
        "--- END EXEMPLAR ---",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1 function loading
# ---------------------------------------------------------------------------


def load_phase1_functions() -> list[dict]:
    """Load all individual functions from Phase 1 passed and failed directories.

    Each function becomes a dict with:
        code: str — the function body
        source_file: str — original PHP filename
        source_dir: str — "passed" or "failed"
        function_name: str — function name

    Filters out functions with code length < 50 chars (empty/minimal).
    """
    functions: list[dict] = []

    for source_dir, dir_path in [("passed", PASSED_DIR), ("failed", FAILED_DIR)]:
        if not dir_path.exists():
            print(f"WARNING: {source_dir} directory not found: {dir_path}", file=sys.stderr)
            continue

        for json_file in dir_path.glob("*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                # Get code from 'body' field (Phase 1 extraction format)
                code = entry.get("body", entry.get("code", ""))
                if len(code.strip()) < 50:
                    continue  # Skip empty/minimal functions (per research pitfall 6)
                functions.append({
                    "code": code,
                    "source_file": entry.get("source_file", str(json_file.name)),
                    "source_dir": source_dir,
                    "function_name": entry.get("function_name", "unknown"),
                })

    return functions


# ---------------------------------------------------------------------------
# Citation accuracy verification (addresses review concern #1)
# ---------------------------------------------------------------------------


def verify_citation_accuracy(result: dict, source_code: str) -> dict:
    """Verify that WP API citations in reasoning analysis are grounded in source code.

    Scans all dimension_analysis.*.analysis text fields for WP_API_CITATIONS.
    For each found citation, checks if:
    - The API string appears in source_code (positive citation, grounded), OR
    - The analysis text uses "missing", "not used", "should be", "lacks", "absent",
      "does not", "doesn't", "no " (negative citation — saying the API is absent
      is valid reasoning even if the API isn't in the code).

    Returns a dict with:
        total_citations: int — total WP API citations found in analysis text
        grounded_citations: int — citations where the API is in source_code OR cited as absent
        hallucinated_citations: list[str] — API names cited as present but NOT in source_code
        hallucination_ratio: float — len(hallucinated) / total_citations (0.0 if total == 0)
    """
    NEGATIVE_INDICATORS = [
        "missing", "not used", "should be", "lacks", "absent",
        "does not", "doesn't", "no ", "not present", "not found",
        "without ", "omit", "neglect", "fail to", "never ",
    ]

    dim_analysis = result.get("dimension_analysis", {})
    all_analysis_text = ""
    for dim_data in dim_analysis.values():
        if isinstance(dim_data, dict):
            all_analysis_text += " " + dim_data.get("analysis", "")

    found_citations: list[str] = []
    for api in WP_API_CITATIONS:
        if api in all_analysis_text:
            found_citations.append(api)

    total_citations = len(found_citations)
    hallucinated: list[str] = []
    grounded = 0

    for api in found_citations:
        # Find the analysis context around this citation
        # Look in each dimension's analysis for the citation context
        api_context = ""
        for dim_data in dim_analysis.values():
            if isinstance(dim_data, dict):
                analysis = dim_data.get("analysis", "")
                if api in analysis:
                    api_context += " " + analysis

        if api in source_code:
            # Positive citation is grounded — API actually appears in source
            grounded += 1
        else:
            # Check if it's cited as absent (negative citation)
            context_lower = api_context.lower()
            is_negative = any(indicator in context_lower for indicator in NEGATIVE_INDICATORS)
            if is_negative:
                grounded += 1
            else:
                hallucinated.append(api)

    hallucination_ratio = len(hallucinated) / total_citations if total_citations > 0 else 0.0

    return {
        "total_citations": total_citations,
        "grounded_citations": grounded,
        "hallucinated_citations": hallucinated,
        "hallucination_ratio": hallucination_ratio,
    }


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def passes_quality_gate(result: dict, source_code: str | None = None) -> bool:
    """Check whether a generated example passes all quality criteria.

    Checks:
    1. verdict is "PASS" or "FAIL"
    2. dimension_analysis has all 9 required dimensions
    3. Each dimension has score (int 1-10) and analysis (non-empty string)
    4. overall_score is int/float 0-100
    5. Citation accuracy: if source_code provided, hallucination_ratio must not
       exceed CITATION_HALLUCINATION_THRESHOLD

    Returns True if all checks pass, False otherwise.
    """
    if result is None or not isinstance(result, dict):
        return False

    # Check verdict
    if result.get("verdict") not in ("PASS", "FAIL"):
        return False

    # Check dimension_analysis
    dim_analysis = result.get("dimension_analysis")
    if not isinstance(dim_analysis, dict):
        return False

    for dim in REQUIRED_DIMENSIONS:
        if dim not in dim_analysis:
            return False
        dim_data = dim_analysis[dim]
        if not isinstance(dim_data, dict):
            return False
        score = dim_data.get("score")
        if not isinstance(score, (int, float)):
            return False
        if not (1 <= score <= 10):
            return False
        analysis = dim_data.get("analysis", "")
        if not isinstance(analysis, str) or not analysis.strip():
            return False

    # Check overall_score
    overall = result.get("overall_score")
    if not isinstance(overall, (int, float)):
        return False
    if not (0 <= overall <= 100):
        return False

    # Citation accuracy check (strengthened — addresses review concern #1)
    if source_code is not None:
        ca = verify_citation_accuracy(result, source_code)
        if ca["hallucination_ratio"] > CITATION_HALLUCINATION_THRESHOLD:
            return False

    return True


# ---------------------------------------------------------------------------
# Training example formatter
# ---------------------------------------------------------------------------


def format_training_example(
    source_info: dict,
    result: dict,
    citation_accuracy: dict | None = None,
) -> dict:
    """Assemble the final training example dict.

    Args:
        source_info: Dict with code, source_file, source_dir, function_name
        result: The quality-gate-passing generation result
        citation_accuracy: Optional dict from verify_citation_accuracy

    Returns a training example dict with all required fields.
    """
    return {
        "source_file": source_info.get("source_file", ""),
        "source_dir": source_info.get("source_dir", ""),
        "function_name": source_info.get("function_name", ""),
        "code": source_info.get("code", ""),
        "reasoning": result,
        "dimensions_addressed": list(REQUIRED_DIMENSIONS),
        "generation_method": "seed_few_shot_agent",
        "citation_accuracy": citation_accuracy or {},
    }


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def _build_prompt(code: str, seed_exemplars: list[str]) -> str:
    """Build the few-shot prompt for deep judge CoT generation."""
    exemplar_text = "\n\n".join(seed_exemplars)

    return f"""You are a WordPress code quality assessor producing deep reasoning chains for training data.
You MUST analyze ALL 9 dimensions regardless of how many the example seeds show.
Seeds only show 2-3 dimensions as examples of analysis depth, not as the complete set.

Here are example analyses from expert WordPress developers:

{exemplar_text}

NOW ANALYZE the following WordPress PHP code. You MUST produce reasoning for ALL 9 dimensions:
wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility

```php
{code[:3000]}
```

When WordPress APIs appear in the code, name them explicitly:
$wpdb->prepare(), wp_verify_nonce(), esc_html(), current_user_can(), check_ajax_referer(), esc_attr(), esc_url().
When WordPress APIs are MISSING from code that needs them, state explicitly what is missing and why.
Do not describe behavior abstractly without naming the specific API.
IMPORTANT: Only cite APIs that actually appear in the code or that are demonstrably missing.
Do not invent citations.

Return JSON with this exact structure:
{{
  "verdict": "PASS" or "FAIL",
  "dimension_analysis": {{
    "wpcs_compliance": {{"score": <1-10>, "analysis": "<specific text citing WP APIs by name>"}},
    "sql_safety": {{"score": <1-10>, "analysis": "<text>"}},
    "security": {{"score": <1-10>, "analysis": "<text>"}},
    "performance": {{"score": <1-10>, "analysis": "<text>"}},
    "wp_api_usage": {{"score": <1-10>, "analysis": "<text>"}},
    "code_quality": {{"score": <1-10>, "analysis": "<text>"}},
    "dependency_integrity": {{"score": <1-10>, "analysis": "<text>"}},
    "i18n": {{"score": <1-10>, "analysis": "<text>"}},
    "accessibility": {{"score": <1-10>, "analysis": "<text>"}}
  }},
  "overall_score": <0-100>,
  "key_observation": "<1-2 sentence summary of the main quality verdict>"
}}"""


def generate_deep_judge_cot(
    code: str,
    source_info: dict,
    seeds: list[dict],
    client: anthropic.Anthropic,
) -> dict | None:
    """Generate a deep judge CoT training example for a single PHP function.

    Args:
        code: PHP function body
        source_info: Metadata about the source function
        seeds: Pool of CoT seeds for few-shot sampling
        client: Anthropic client instance

    Returns a parsed result dict, or None if generation/parsing fails.
    """
    sampled = sample_seeds(seeds, n=3)
    exemplars = [format_seed_as_exemplar(s) for s in sampled]
    prompt = _build_prompt(code, exemplars)

    try:
        resp = call_with_backoff(
            client,
            model="claude-sonnet-4-6-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = resp.content[0].text
        return extract_json(raw_text)
    except Exception as e:
        print(f"WARNING: Generation failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Pilot batch validation
# ---------------------------------------------------------------------------


def validate_pilot_batch(examples: list[dict]) -> dict:
    """Validate a pilot batch of generated examples.

    Checks:
    1. All 9 dimensions appear across examples
    2. At least 3 distinct WP_API_CITATIONS appear in reasoning text
    3. Citation accuracy: mean hallucination_ratio, warn if > 0.3

    Returns a dict with:
        missing_dimensions: list[str]
        distinct_api_citations: int
        api_citations_found: list[str]
        mean_hallucination_ratio: float
        citation_accuracy_ok: bool
        dimension_score_distribution: dict[str, list[int]]
    """
    dimensions_seen: set[str] = set()
    api_citations_found: set[str] = set()
    hallucination_ratios: list[float] = []
    dim_scores: dict[str, list[int]] = {d: [] for d in REQUIRED_DIMENSIONS}

    for example in examples:
        reasoning = example.get("reasoning", {})
        dim_analysis = reasoning.get("dimension_analysis", {})

        for dim, details in dim_analysis.items():
            if dim in REQUIRED_DIMENSIONS:
                dimensions_seen.add(dim)
                if isinstance(details, dict):
                    score = details.get("score")
                    if isinstance(score, (int, float)):
                        dim_scores[dim].append(int(score))
                    analysis = details.get("analysis", "")
                    for api in WP_API_CITATIONS:
                        if api in analysis:
                            api_citations_found.add(api)

        ca = example.get("citation_accuracy", {})
        ratio = ca.get("hallucination_ratio", 0.0)
        hallucination_ratios.append(ratio)

    missing_dims = [d for d in REQUIRED_DIMENSIONS if d not in dimensions_seen]
    mean_ratio = sum(hallucination_ratios) / len(hallucination_ratios) if hallucination_ratios else 0.0

    if mean_ratio > 0.3:
        print(
            f"WARNING: Mean hallucination ratio is {mean_ratio:.2%} (threshold: 30%). "
            "Review API citation quality.",
            file=sys.stderr,
        )

    report = {
        "missing_dimensions": missing_dims,
        "distinct_api_citations": len(api_citations_found),
        "api_citations_found": sorted(api_citations_found),
        "mean_hallucination_ratio": mean_ratio,
        "citation_accuracy_ok": mean_ratio <= 0.3,
        "dimension_score_distribution": {
            d: scores for d, scores in dim_scores.items()
        },
    }

    print("\n--- Pilot Batch Validation ---")
    print(f"Examples validated: {len(examples)}")
    print(f"Missing dimensions: {missing_dims or 'none'}")
    print(f"Distinct WP API citations: {len(api_citations_found)}")
    print(f"APIs found: {sorted(api_citations_found)}")
    print(f"Mean hallucination ratio: {mean_ratio:.2%}")
    print(f"Citation accuracy OK: {report['citation_accuracy_ok']}")

    return report


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------


def _setup_output_dirs() -> None:
    """Create all required output directories."""
    for d in [OUTPUT_DIR, PILOT_DIR, BULK_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deep judge CoT training examples")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: generate 40 examples to data/phase4_reasoning/pilot/",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Override bulk target count (default: ~50% of available functions)",
    )
    args = parser.parse_args()

    _setup_output_dirs()

    # Load seeds and functions
    print("Loading seeds...")
    seeds = load_seeds()
    if not seeds:
        print("ERROR: No CoT seeds found. Run scripts/seed_import.py first.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(seeds)} deep_judge_cot seeds")

    print("Loading Phase 1 functions...")
    all_functions = load_phase1_functions()
    if not all_functions:
        print("ERROR: No Phase 1 functions found.", file=sys.stderr)
        return 1
    print(f"  Loaded {len(all_functions)} functions (passed + failed)")

    # Determine target and output path
    if args.pilot:
        target = 40
        output_path = PILOT_DIR / "deep_judge_cot_pilot.json"
        checkpoint_phase = "generate_deep_judge_cot_pilot"
        print(f"\nPilot mode: generating {target} examples")
    else:
        target = args.target or int(len(all_functions) * 0.5)
        output_path = BULK_DIR / "deep_judge_cot_bulk.json"
        checkpoint_phase = "generate_deep_judge_cot"
        print(f"\nBulk mode: generating {target} examples")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_phase)
    completed_ids: set[str] = set(checkpoint.get("completed", []))
    print(f"Resuming from checkpoint: {len(completed_ids)} already completed")

    # Initialize client
    client = anthropic.Anthropic()

    # Filter out already-processed functions
    remaining = [
        fn for fn in all_functions
        if f"{fn['source_dir']}:{fn['source_file']}:{fn['function_name']}" not in completed_ids
    ]
    random.shuffle(remaining)
    remaining = remaining[: target - len(completed_ids)]

    # Load existing results for checkpoint resume
    examples: list[dict] = []
    if output_path.exists() and completed_ids:
        try:
            with output_path.open("r", encoding="utf-8") as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} existing examples from checkpoint")
        except (json.JSONDecodeError, OSError):
            examples = []

    # Generation loop
    parse_attempts = 0
    parse_failures = 0
    citation_rejections = 0

    for fn in remaining:
        fn_id = f"{fn['source_dir']}:{fn['source_file']}:{fn['function_name']}"
        code = fn["code"]

        parse_attempts += 1
        result = generate_deep_judge_cot(code, fn, seeds, client)

        if result is None:
            parse_failures += 1
            print(f"  SKIP (parse fail): {fn_id}")
            continue

        # Citation accuracy check
        ca = verify_citation_accuracy(result, code)

        # Quality gate
        if not passes_quality_gate(result, source_code=code):
            if ca["hallucination_ratio"] > CITATION_HALLUCINATION_THRESHOLD:
                citation_rejections += 1
                print(f"  SKIP (hallucination {ca['hallucination_ratio']:.0%}): {fn_id}")
            else:
                parse_failures += 1
                print(f"  SKIP (quality gate): {fn_id}")
            continue

        example = format_training_example(fn, result, citation_accuracy=ca)
        examples.append(example)
        completed_ids.add(fn_id)

        print(f"  OK [{len(examples)}/{target}]: {fn_id} ({result['verdict']}, score={result['overall_score']})")

        # Save checkpoint
        if len(examples) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(checkpoint_phase, {"completed": list(completed_ids)})
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2)

        if len(examples) >= target:
            break

    # Final save
    save_checkpoint(checkpoint_phase, {"completed": list(completed_ids)})
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete")
    print(f"  Target:             {target}")
    print(f"  Generated:          {len(examples)}")
    print(f"  Parse attempts:     {parse_attempts}")
    print(f"  Parse failures:     {parse_failures} ({parse_failures/max(parse_attempts,1):.1%})")
    print(f"  Citation rejections:{citation_rejections}")
    print(f"  Output:             {output_path}")

    # Dimension coverage stats
    if examples:
        dim_coverage = {d: 0 for d in REQUIRED_DIMENSIONS}
        for ex in examples:
            dim_analysis = ex.get("reasoning", {}).get("dimension_analysis", {})
            for d in REQUIRED_DIMENSIONS:
                if d in dim_analysis:
                    dim_coverage[d] += 1
        print("\nDimension coverage:")
        for d, count in dim_coverage.items():
            print(f"  {d}: {count}/{len(examples)} ({count/len(examples):.0%})")

    # Pilot validation
    if args.pilot and examples:
        report = validate_pilot_batch(examples)
        if report["missing_dimensions"]:
            print(f"\nERROR: Missing dimensions in pilot batch: {report['missing_dimensions']}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
