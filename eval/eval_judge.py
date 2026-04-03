"""Per-dimension Spearman correlation evaluation for judge mode.

Compares model's <wp_judge> dimension scores against rubric_scorer ground truth.

Usage:
    python -m eval.eval_judge [--limit N] [--output PATH]
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import openai
from scipy.stats import spearmanr

from eval.rubric_definitions import DIM_NAME_MAP, DIMENSION_WEIGHTS
from eval.rubric_scorer import score_code
from scripts.dgx_toolbox import get_toolbox

# Module-level toolbox singleton (lazy)
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
# Model field -> dimension key mapping
# ---------------------------------------------------------------------------

# Model outputs these fields; map each to the rubric dimension key.
# Fields not in DIM_NAME_MAP (like overall_score) are handled separately.
_MODEL_FIELD_TO_DIM: dict[str, str] = {
    field: dim_key
    for field, dim_key in DIM_NAME_MAP.items()
    if not field.startswith("D")  # only field->dim direction
}


# ---------------------------------------------------------------------------
# Parse helpers (kept from original)
# ---------------------------------------------------------------------------


def parse_judge_response(response: str) -> Optional[dict]:
    """Parse model judge response and extract score fields.

    Handles:
      - Raw JSON string
      - JSON in markdown code fences (```json ... ``` or ``` ... ```)
      - Missing keys -> returns dict without those keys

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
    text = re.sub(
        r"Evaluate this WordPress code:\s*", "", text, count=1, flags=re.IGNORECASE
    )
    return text.strip()


# ---------------------------------------------------------------------------
# Spearman helper
# ---------------------------------------------------------------------------


def _safe_spearman(xs: list[float], ys: list[float]) -> dict:
    """Compute Spearman correlation, returning a standard dict.

    Returns {"corr": float, "p_value": float, "n_pairs": int}.
    If fewer than 2 pairs or all values identical, corr=0.0, p_value=1.0.
    """
    n = len(xs)
    if n < 2:
        return {"corr": 0.0, "p_value": 1.0, "n_pairs": n}
    # scipy warns / returns nan when all values are identical
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"corr": 0.0, "p_value": 1.0, "n_pairs": n}
    result = spearmanr(xs, ys)
    return {
        "corr": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    limit: Optional[int] = None,
    output_path: str = "output/eval_judge_results.json",
    model: Optional[str] = None,
) -> dict:
    """Compute per-dimension Spearman correlation between model judge scores
    and rubric_scorer ground truth.

    Loads wp_judge examples from the test dataset, queries the served model via
    vLLM endpoint (resolved from DGX Toolbox), parses dimension scores from
    judge responses, runs rubric_scorer.score_code() on the same code, and
    computes per-dimension + overall Spearman correlations.

    Args:
        dataset_path: Path to OpenAI-format JSONL test dataset.
        limit: Maximum number of examples to evaluate (None = all).
        output_path: Path to save JSON results.

    Returns:
        dict with overall_spearman, per_dimension, and backward-compat fields.
    """
    dgx = _get_dgx()
    client = openai.OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")
    resolved_model = model or _detect_model(client)

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
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            if "<wp_judge>" in user_msg:
                examples.append(example)

    if limit is not None:
        examples = examples[:limit]

    print(
        f"Evaluating {len(examples)} wp_judge examples via {dgx.vllm_endpoint()}",
        file=sys.stderr,
    )

    # Collectors: per-dimension pairs and overall pairs
    dim_model_scores: dict[str, list[float]] = {
        dim: [] for dim in DIMENSION_WEIGHTS
    }
    dim_gt_scores: dict[str, list[float]] = {dim: [] for dim in DIMENSION_WEIGHTS}
    overall_model: list[float] = []
    overall_gt: list[float] = []
    skipped = 0

    # Per-example debug records
    pair_records: list[dict] = []

    # JSONL output path (sibling of main output)
    pairs_path = Path(output_path).with_suffix(".pairs.jsonl")

    for i, example in enumerate(examples):
        messages = example["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        user_msg_text = user_messages[0]["content"] if user_messages else ""

        # Extract code being judged
        code = _extract_code_from_judge_prompt(user_msg_text)

        # Query model in wp_judge mode
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=user_messages,
                max_tokens=1024,
                temperature=0.0,
            )
            generated = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i}] Model error: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Parse dimension scores from response
        parsed = parse_judge_response(generated)
        if parsed is None or "overall_score" not in parsed:
            skipped += 1
            continue

        model_overall = parsed.get("overall_score")
        if not isinstance(model_overall, (int, float)):
            skipped += 1
            continue

        # Ground truth via rubric_scorer
        gt = score_code(code)

        # Record overall pair
        overall_model.append(float(model_overall))
        overall_gt.append(float(gt.overall))

        # Record per-dimension pairs
        pair_record: dict = {
            "index": i,
            "model_overall": float(model_overall),
            "gt_overall": float(gt.overall),
            "dimensions": {},
        }

        for model_field, dim_key in _MODEL_FIELD_TO_DIM.items():
            model_val = parsed.get(model_field)
            gt_val = gt.dimension_scores.get(dim_key)

            if (
                model_val is not None
                and isinstance(model_val, (int, float))
                and gt_val is not None
            ):
                dim_model_scores[dim_key].append(float(model_val))
                dim_gt_scores[dim_key].append(float(gt_val))
                pair_record["dimensions"][dim_key] = {
                    "model": float(model_val),
                    "gt": float(gt_val),
                }

        pair_records.append(pair_record)

        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1}/{len(examples)}] pairs collected={len(overall_model)}",
                file=sys.stderr,
            )

    # --- Compute correlations ---

    overall_result = _safe_spearman(overall_model, overall_gt)

    per_dimension: dict[str, dict] = {}
    for dim_key in DIMENSION_WEIGHTS:
        per_dimension[dim_key] = _safe_spearman(
            dim_model_scores[dim_key], dim_gt_scores[dim_key]
        )

    summary = {
        "overall_spearman": overall_result,
        "per_dimension": per_dimension,
        "skipped": skipped,
        "total": len(examples),
        # Backward compat
        "spearman_corr": overall_result["corr"],
        "p_value": overall_result["p_value"],
        "total_pairs": overall_result["n_pairs"],
    }

    # Save results JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {output_path}", file=sys.stderr)

    # Save per-example pairs JSONL for debugging
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w") as pf:
        for rec in pair_records:
            pf.write(json.dumps(rec) + "\n")
    print(f"Per-example pairs saved to {pairs_path}", file=sys.stderr)

    # Print human-readable correlation table to stderr
    _print_correlation_table(summary)

    return summary


def _print_correlation_table(summary: dict) -> None:
    """Print a human-readable correlation table to stderr."""
    print("\n" + "=" * 64, file=sys.stderr)
    print("  Per-Dimension Spearman Correlations", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(
        f"  {'Dimension':<16} {'Corr':>8} {'p-value':>10} {'N pairs':>8}",
        file=sys.stderr,
    )
    print("-" * 64, file=sys.stderr)

    for dim_key in DIMENSION_WEIGHTS:
        d = summary["per_dimension"].get(dim_key, {})
        corr = d.get("corr", 0.0)
        pval = d.get("p_value", 1.0)
        n = d.get("n_pairs", 0)
        sig = "*" if pval < 0.05 else " "
        print(
            f"  {dim_key:<16} {corr:>8.3f} {pval:>10.4f} {n:>7d} {sig}",
            file=sys.stderr,
        )

    print("-" * 64, file=sys.stderr)
    ov = summary["overall_spearman"]
    sig = "*" if ov["p_value"] < 0.05 else " "
    print(
        f"  {'OVERALL':<16} {ov['corr']:>8.3f} {ov['p_value']:>10.4f} {ov['n_pairs']:>7d} {sig}",
        file=sys.stderr,
    )
    print("=" * 64, file=sys.stderr)
    print("  (* = p < 0.05)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-dimension Spearman correlation for wp_judge mode."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max examples to evaluate"
    )
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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
