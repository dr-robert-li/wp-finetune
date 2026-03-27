"""Spearman correlation evaluation for judge mode.

Compares model's <wp_judge> scores against PHPCS error counts as ground truth.

Usage:
    python -m eval.eval_judge [--limit N] [--output PATH]
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import openai
from scipy.stats import spearmanr

from scripts.dgx_toolbox import get_toolbox

# Module-level toolbox singleton (lazy)
_dgx = None


def _get_dgx():
    global _dgx
    if _dgx is None:
        _dgx = get_toolbox()
    return _dgx


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def invert_phpcs_errors(error_count: int) -> float:
    """Convert a PHPCS error count to a quality score in [0, 100].

    Formula: max(0, 100 - error_count * 5)

    Args:
        error_count: Number of PHPCS errors (non-negative int).

    Returns:
        Score in [0.0, 100.0] where 0 errors → 100, 20+ errors → 0.
    """
    return float(max(0, 100 - error_count * 5))


def parse_judge_response(response: str) -> Optional[dict]:
    """Parse model judge response and extract score fields.

    Handles:
      - Raw JSON string
      - JSON in markdown code fences (```json ... ``` or ``` ... ```)
      - Missing keys → returns dict without those keys

    Args:
        response: Raw model response string.

    Returns:
        Parsed dict with judge fields (may include overall_score), or None
        if the response cannot be parsed as JSON.
    """
    text = response.strip()

    # Strategy 1: raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json fenced block
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: generic ``` fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 4: embedded JSON object (first { ... } match)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Cannot parse
    return None


def _extract_code_from_judge_prompt(user_message: str) -> str:
    """Extract the PHP code being judged from a <wp_judge> user message.

    The wp_judge format is:
        <wp_judge> Evaluate this WordPress code:\n\n<?php\n...code...

    Returns the code portion, or empty string if not found.
    """
    # Remove the <wp_judge> tag and "Evaluate this WordPress code:" prefix
    text = re.sub(r"<wp_judge>\s*", "", user_message, count=1)
    text = re.sub(r"Evaluate this WordPress code:\s*", "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def _run_phpcs_count(code: str) -> int:
    """Run phpcs on code and return the total error count."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["phpcs", "--standard=WordPress", "--report=json", tmp_path],
            capture_output=True,
            text=True,
        )
        raw = proc.stdout.strip()
        if not raw:
            return 0
        phpcs_output = json.loads(raw)
        return phpcs_output.get("totals", {}).get("errors", 0)
    except Exception:
        return 0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    limit: Optional[int] = None,
    output_path: str = "output/eval_judge_results.json",
) -> dict:
    """Compute Spearman correlation between model judge scores and PHPCS error counts.

    Loads wp_judge examples from the test dataset, queries the served model via
    vLLM endpoint (resolved from DGX Toolbox), parses overall_score from judge
    responses, runs PHPCS on the same code, and computes Spearman correlation.

    Args:
        dataset_path: Path to OpenAI-format JSONL test dataset.
        limit: Maximum number of examples to evaluate (None = all).
        output_path: Path to save JSON results.

    Returns:
        dict with spearman_corr, p_value, total_pairs.
    """
    dgx = _get_dgx()
    client = openai.OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")

    # Load and filter wp_judge examples
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
            if "<wp_judge>" in user_msg:
                examples.append(example)

    if limit is not None:
        examples = examples[:limit]

    print(f"Evaluating {len(examples)} wp_judge examples via {dgx.vllm_endpoint()}", file=sys.stderr)

    model_scores = []
    phpcs_scores = []
    skipped = 0

    for i, example in enumerate(examples):
        messages = example["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        user_msg_text = user_messages[0]["content"] if user_messages else ""

        # Extract code being judged
        code = _extract_code_from_judge_prompt(user_msg_text)

        # Query model in wp_judge mode
        try:
            response = client.chat.completions.create(
                model="openai/qwen3-wp",
                messages=user_messages,
                max_tokens=1024,
                temperature=0.0,
            )
            generated = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i}] Model error: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Parse overall_score from response
        parsed = parse_judge_response(generated)
        if parsed is None or "overall_score" not in parsed:
            skipped += 1
            continue

        overall_score = parsed["overall_score"]
        if not isinstance(overall_score, (int, float)):
            skipped += 1
            continue

        # Run PHPCS on same code for ground truth
        phpcs_errors = _run_phpcs_count(code)
        phpcs_score = invert_phpcs_errors(phpcs_errors)

        model_scores.append(float(overall_score))
        phpcs_scores.append(phpcs_score)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] pairs collected={len(model_scores)}", file=sys.stderr)

    # Compute Spearman correlation
    if len(model_scores) < 2:
        print("WARNING: Not enough valid pairs for Spearman correlation.", file=sys.stderr)
        spearman_corr = 0.0
        p_value = 1.0
    else:
        result = spearmanr(model_scores, phpcs_scores)
        spearman_corr = float(result.statistic)
        p_value = float(result.pvalue)

    summary = {
        "spearman_corr": spearman_corr,
        "p_value": p_value,
        "total_pairs": len(model_scores),
        "skipped": skipped,
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {output_path}", file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spearman correlation for wp_judge mode."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="output/eval_judge_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/final_dataset/openai_test.jsonl",
        help="Path to OpenAI-format JSONL test dataset",
    )
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        output_path=args.output,
    )

    print(json.dumps(summary, indent=2))

    print(f"\nSpearman correlation: {summary['spearman_corr']:.3f} (p={summary['p_value']:.4f})")
    print(f"Valid pairs:          {summary['total_pairs']}")
    print(f"Skipped:              {summary['skipped']}")


if __name__ == "__main__":
    main()
