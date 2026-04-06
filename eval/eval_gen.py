"""9-dimension rubric evaluation for generated WordPress PHP code.

Runs the served model in <wp_gen> mode on held-out test examples,
scores generated PHP through the full rubric scoring engine, and
computes per-dimension + overall metrics.

Usage:
    python -m eval.eval_gen [--limit N] [--output PATH]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Optional

import openai

from eval.rubric_definitions import CRITICAL_FLOOR_RULES, DIMENSION_WEIGHTS, GRADE_BANDS
from eval.rubric_scorer import RubricScore, score_code
from scripts.dgx_toolbox import get_toolbox

# Module-level toolbox singleton (lazy -- does not require DGX to be present at import time)
_dgx = None


def _get_dgx():
    global _dgx
    if _dgx is None:
        _dgx = get_toolbox()
    return _dgx


DEFAULT_MODEL = "openai/qwen3-wp"


def _detect_model(client: openai.OpenAI) -> str:
    """Auto-detect the served model name from /v1/models endpoint."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# PHP extraction helper (kept from original)
# ---------------------------------------------------------------------------


def _extract_php_code(text: str) -> str:
    """Extract PHP code from a model response.

    Handles fenced code blocks (```php ... ``` or ``` ... ```) or
    falls back to returning the full text.
    """
    # Try ```php fenced block
    match = re.search(r"```php\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is (assume entire response is PHP)
    return text.strip()


# ---------------------------------------------------------------------------
# Floor rule key mapping
# ---------------------------------------------------------------------------

# Build stable keys for floor rule tracking from CRITICAL_FLOOR_RULES tuples
_FLOOR_RULE_KEYS = []
for _rule in CRITICAL_FLOOR_RULES:
    _dim = _rule[0] if isinstance(_rule, (list, tuple)) else _rule.get("dimension", "")
    _FLOOR_RULE_KEYS.append(f"{_dim}_capped")


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

DIMENSION_KEYS = list(DIMENSION_WEIGHTS.keys())


def _compute_summary(
    rubric_scores: list[RubricScore],
) -> dict:
    """Aggregate per-example RubricScores into a summary dict."""
    n = len(rubric_scores)
    if n == 0:
        return {"total": 0}

    overall_scores = [r.overall for r in rubric_scores]
    overall_mean = statistics.mean(overall_scores)
    overall_median = statistics.median(overall_scores)

    # Grade distribution
    grade_names = [label for _, label in GRADE_BANDS]
    grade_dist = {label: 0 for label in grade_names}
    for r in rubric_scores:
        grade_dist[r.grade] = grade_dist.get(r.grade, 0) + 1

    # Per-dimension metrics
    per_dimension: dict[str, dict] = {}
    for dim_key in DIMENSION_KEYS:
        dim_vals = [
            r.dimension_scores[dim_key]
            for r in rubric_scores
            if r.dimension_scores.get(dim_key) is not None
        ]
        na_count = sum(
            1 for r in rubric_scores if r.dimension_scores.get(dim_key) is None
        )
        na_rate = na_count / n
        if dim_vals:
            dim_mean = statistics.mean(dim_vals)
            # pass_rate_8: among applicable examples only (excludes N/A)
            pass_rate_8 = sum(1 for v in dim_vals if v >= 8.0) / len(dim_vals)
            # pass_rate_8_inclusive: treats N/A as failing (denominator = total examples)
            pass_rate_8_inclusive = sum(1 for v in dim_vals if v >= 8.0) / n
        else:
            dim_mean = 0.0
            pass_rate_8 = 0.0
            pass_rate_8_inclusive = 0.0

        per_dimension[dim_key] = {
            "mean": round(dim_mean, 2),
            "pass_rate_8": round(pass_rate_8, 4),
            "pass_rate_8_inclusive": round(pass_rate_8_inclusive, 4),
            "na_count": na_count,
            "na_rate": round(na_rate, 4),
        }

    # Floor rule trigger rates
    floor_rules: dict[str, float] = {}
    for i, rule in enumerate(CRITICAL_FLOOR_RULES):
        rule_key = _FLOOR_RULE_KEYS[i]
        # Count how many examples had this floor rule fire
        if isinstance(rule, (list, tuple)):
            dim = rule[0]
        else:
            dim = rule.get("dimension", "")
        fired = sum(
            1 for r in rubric_scores
            if any(dim in applied for applied in r.floor_rules_applied)
        )
        floor_rules[rule_key] = round(fired / n, 4)

    # Backward compat metrics
    phpcs_pass_rate = sum(1 for r in rubric_scores if r.overall >= 80.0) / n
    security_vals = [
        r.dimension_scores.get("D2_security")
        for r in rubric_scores
        if r.dimension_scores.get("D2_security") is not None
    ]
    if security_vals:
        security_pass_rate: Optional[float] = sum(
            1 for v in security_vals if v >= 8.0
        ) / len(security_vals)
    else:
        # No security-applicable examples — report null rather than perfect 1.0
        security_pass_rate = None

    # phpcs_pass_rate: overall >= 80 proxy (all examples have an overall score)
    # Report null only if somehow no applicable examples (practically can't happen
    # since overall is always computed, but be consistent)
    phpcs_applicable = [r for r in rubric_scores if r.overall is not None]
    if phpcs_applicable:
        phpcs_pass_rate_val: Optional[float] = round(phpcs_pass_rate, 4)
    else:
        phpcs_pass_rate_val = None

    # n_applicable_dims: per-example average of how many dimensions were not N/A
    # Summarised as the mean count across examples for transparency
    n_applicable_dims_per_example = [
        len(DIMENSION_KEYS) - len(r.dimension_na)
        for r in rubric_scores
    ]
    n_applicable_dims_mean = round(statistics.mean(n_applicable_dims_per_example), 2)

    return {
        "total": n,
        "overall_mean": round(overall_mean, 2),
        "overall_median": round(overall_median, 2),
        "grade_distribution": grade_dist,
        "per_dimension": per_dimension,
        "floor_rules": floor_rules,
        "n_applicable_dims_mean": n_applicable_dims_mean,
        # Backward compat
        "phpcs_pass_rate": phpcs_pass_rate_val,
        "security_pass_rate": (
            round(security_pass_rate, 4) if security_pass_rate is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def _print_summary_table(summary: dict, file=sys.stderr) -> None:
    """Print a human-readable summary table to the given file."""
    print("\n" + "=" * 60, file=file)
    print("  RUBRIC EVALUATION SUMMARY", file=file)
    print("=" * 60, file=file)

    print(f"\n  Total examples:   {summary['total']}", file=file)
    print(f"  Overall mean:     {summary['overall_mean']:.1f} / 100", file=file)
    print(f"  Overall median:   {summary['overall_median']:.1f} / 100", file=file)

    # Grade distribution
    print("\n  Grade Distribution:", file=file)
    for grade, count in summary.get("grade_distribution", {}).items():
        bar = "#" * count
        print(f"    {grade:<12s} {count:>4d}  {bar}", file=file)

    # Per-dimension table
    print("\n  Per-Dimension Scores:", file=file)
    print(
        f"    {'Dimension':<16s} {'Mean':>6s} {'Pass@8':>7s} {'Incl@8':>7s} {'N/A':>5s} {'NA%':>6s}",
        file=file,
    )
    print(f"    {'-'*16} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*6}", file=file)
    for dim_key, metrics in summary.get("per_dimension", {}).items():
        print(
            f"    {dim_key:<16s} {metrics['mean']:>6.2f} {metrics['pass_rate_8']:>6.1%}"
            f" {metrics['pass_rate_8_inclusive']:>6.1%} {metrics['na_count']:>5d}"
            f" {metrics['na_rate']:>5.1%}",
            file=file,
        )

    # Floor rules
    print("\n  Floor Rule Trigger Rates:", file=file)
    for rule_key, rate in summary.get("floor_rules", {}).items():
        print(f"    {rule_key:<24s} {rate:>6.1%}", file=file)

    # Transparency metrics
    n_applicable = summary.get("n_applicable_dims_mean")
    if n_applicable is not None:
        print(f"\n  Avg applicable dims / example: {n_applicable:.1f} / {len(summary.get('per_dimension', {}))}", file=file)

    # Backward compat (None = not applicable)
    phpcs_rate = summary.get("phpcs_pass_rate")
    sec_rate = summary.get("security_pass_rate")
    phpcs_str = f"{phpcs_rate:.1%}" if phpcs_rate is not None else "N/A (no applicable examples)"
    sec_str = f"{sec_rate:.1%}" if sec_rate is not None else "N/A (no security-applicable examples)"
    print(f"\n  phpcs_pass_rate (overall>=80): {phpcs_str}", file=file)
    print(f"  security_pass_rate (D2>=8):    {sec_str}", file=file)
    print("=" * 60 + "\n", file=file)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    limit: Optional[int] = None,
    output_path: str = "output/eval_gen_results.json",
    model: Optional[str] = None,
) -> dict:
    """Run 9-dimension rubric evaluation on wp_gen examples.

    Loads wp_gen examples from the test dataset, queries the served model via
    vLLM endpoint (resolved from DGX Toolbox), and evaluates generated PHP
    through the rubric scoring engine.

    Args:
        dataset_path: Path to OpenAI-format JSONL test dataset.
        limit: Maximum number of examples to evaluate (None = all).
        output_path: Path to save JSON summary results.

    Returns:
        dict with overall_mean, overall_median, grade_distribution,
        per_dimension metrics, floor_rules, and backward-compat rates.
    """
    dgx = _get_dgx()
    client = openai.OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")
    resolved_model = model or _detect_model(client)

    # Load and filter wp_gen examples
    examples = []
    dataset_file = Path(dataset_path)
    with dataset_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            if "<wp_gen>" in user_msg:
                examples.append(example)

    if limit is not None:
        examples = examples[:limit]

    print(f"Evaluating {len(examples)} wp_gen examples via {dgx.vllm_endpoint()}", file=sys.stderr)

    # Per-example JSONL output path
    jsonl_path = Path(output_path).with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    rubric_scores: list[RubricScore] = []

    with jsonl_path.open("w") as jsonl_f:
        for i, example in enumerate(examples):
            messages = example["messages"]
            user_messages = [m for m in messages if m["role"] == "user"]

            # Query the model
            try:
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=user_messages,
                    max_tokens=2048,
                    temperature=0.0,
                )
                generated = response.choices[0].message.content or ""
            except Exception as e:
                print(f"  [{i}] Model error: {e}", file=sys.stderr)
                generated = ""

            # Extract and score PHP code
            php_code = _extract_php_code(generated)
            result = score_code(php_code)
            rubric_scores.append(result)

            # Write per-example detail (includes prompt + response for human review)
            detail = {
                "example_idx": i,
                "prompt": user_messages[0]["content"] if user_messages else "",
                "response": generated,
                "extracted_code": php_code,
                "overall": result.overall,
                "grade": result.grade,
                "dimension_scores": result.dimension_scores,
                "dimension_na": result.dimension_na,
                "floor_rules_applied": result.floor_rules_applied,
                "triggered_checks": result.triggered_checks,
            }
            jsonl_f.write(json.dumps(detail) + "\n")

            if (i + 1) % 50 == 0:
                current_scores = [r.overall for r in rubric_scores]
                current_mean = statistics.mean(current_scores)
                print(
                    f"  [{i+1}/{len(examples)}] overall_mean={current_mean:.1f}",
                    file=sys.stderr,
                )

    # Compute summary
    summary = _compute_summary(rubric_scores)

    # Save summary JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))

    print(f"Summary saved to {output_path}", file=sys.stderr)
    print(f"Per-example details saved to {jsonl_path}", file=sys.stderr)

    # Print human-readable summary to stderr
    _print_summary_table(summary)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WordPress PHP code quality with 9-dimension rubric scoring."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="output/eval_gen_results.json",
        help="Path to save results JSON (JSONL detail file saved alongside)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/final_dataset/openai_test.jsonl",
        help="Path to OpenAI-format JSONL test dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for vLLM (auto-detected from /v1/models if omitted)",
    )
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        output_path=args.output,
        model=args.model,
    )

    # Print JSON summary to stdout
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
