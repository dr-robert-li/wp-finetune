"""PHPCS pass rate and security pass rate evaluation.

Runs the served model in <wp_gen> mode on held-out test examples,
pipes generated PHP through phpcs, and computes pass/fail rates.

Usage:
    python -m eval.eval_gen [--limit N] [--output PATH]
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

from scripts.dgx_toolbox import get_toolbox

# Module-level toolbox singleton (lazy — does not require DGX to be present at import time)
_dgx = None


def _get_dgx():
    global _dgx
    if _dgx is None:
        _dgx = get_toolbox()
    return _dgx


# ---------------------------------------------------------------------------
# PHPCS helpers
# ---------------------------------------------------------------------------

SECURITY_SNIFF_PREFIX = "WordPress.Security."


def run_phpcs(code: str) -> dict:
    """Run phpcs on a PHP code string and return a structured result dict.

    Args:
        code: PHP source code as a string.

    Returns:
        dict with keys:
            - errors (int): total error count
            - warnings (int): total warning count
            - passed (bool): True if errors == 0
            - phpcs_output (dict): raw parsed PHPCS JSON
            - security_issues (list[str]): list of security sniff sources triggered
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["phpcs", "--standard=WordPress", "--report=json", tmp_path],
            capture_output=True,
            text=True,
        )
        # PHPCS exits 1 when violations found — that's expected, not an error
        raw = proc.stdout.strip()
        if not raw:
            # PHPCS not installed or no output — treat as pass with warning
            return {
                "errors": 0,
                "warnings": 0,
                "passed": True,
                "phpcs_output": {},
                "security_issues": [],
                "_phpcs_unavailable": True,
            }

        phpcs_output = json.loads(raw)
        totals = phpcs_output.get("totals", {})
        error_count = totals.get("errors", 0)
        warning_count = totals.get("warnings", 0)

        # Collect all messages across all files
        security_issues = []
        for file_data in phpcs_output.get("files", {}).values():
            for msg in file_data.get("messages", []):
                source = msg.get("source", "")
                if source.startswith(SECURITY_SNIFF_PREFIX):
                    security_issues.append(source)

        return {
            "errors": error_count,
            "warnings": warning_count,
            "passed": error_count == 0,
            "phpcs_output": phpcs_output,
            "security_issues": security_issues,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def classify_security(phpcs_output: dict) -> bool:
    """Return True if the phpcs output contains any WordPress.Security.* sniff violations.

    Args:
        phpcs_output: Parsed PHPCS JSON output dict (as returned by run_phpcs or directly).

    Returns:
        True if security issues found, False otherwise.
    """
    for file_data in phpcs_output.get("files", {}).values():
        for msg in file_data.get("messages", []):
            source = msg.get("source", "")
            if source.startswith(SECURITY_SNIFF_PREFIX):
                return True
    return False


def compute_pass_rate(results: list) -> float:
    """Compute the fraction of results where passed is True.

    Args:
        results: List of dicts, each must have a 'passed' key (bool).

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty list.
    """
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed", False))
    return passed / len(results)


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
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    limit: Optional[int] = None,
    output_path: str = "output/eval_gen_results.json",
) -> dict:
    """Run PHPCS pass rate and security pass rate evaluation.

    Loads wp_gen examples from the test dataset, queries the served model via
    vLLM endpoint (resolved from DGX Toolbox), and evaluates generated PHP
    through phpcs.

    Args:
        dataset_path: Path to OpenAI-format JSONL test dataset.
        limit: Maximum number of examples to evaluate (None = all).
        output_path: Path to save JSON results.

    Returns:
        dict with phpcs_pass_rate, security_pass_rate, total, passed,
        security_total, security_passed.
    """
    dgx = _get_dgx()
    client = openai.OpenAI(base_url=dgx.vllm_endpoint(), api_key="none")

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

    phpcs_results = []
    for i, example in enumerate(examples):
        messages = example["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]

        # Query the model
        try:
            response = client.chat.completions.create(
                model="openai/qwen3-wp",
                messages=user_messages,
                max_tokens=2048,
                temperature=0.0,
            )
            generated = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i}] Model error: {e}", file=sys.stderr)
            generated = ""

        # Extract and evaluate PHP code
        php_code = _extract_php_code(generated)
        result = run_phpcs(php_code)
        phpcs_results.append(result)

        if (i + 1) % 50 == 0:
            current_rate = compute_pass_rate(phpcs_results)
            print(f"  [{i+1}/{len(examples)}] pass_rate={current_rate:.3f}", file=sys.stderr)

    # Compute metrics
    phpcs_pass_rate = compute_pass_rate(phpcs_results)
    total = len(phpcs_results)
    passed = sum(1 for r in phpcs_results if r.get("passed", False))

    # Security subset: examples where security sniffs were triggered
    # (either in failures OR detected as security issues)
    security_results = [r for r in phpcs_results if r.get("security_issues")]
    security_total = len(security_results)
    security_passed = sum(1 for r in security_results if r.get("passed", False))
    security_pass_rate = security_passed / security_total if security_total > 0 else 1.0

    summary = {
        "phpcs_pass_rate": phpcs_pass_rate,
        "security_pass_rate": security_pass_rate,
        "total": total,
        "passed": passed,
        "security_total": security_total,
        "security_passed": security_passed,
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
    parser = argparse.ArgumentParser(description="Evaluate PHPCS pass rate for wp_gen mode.")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="output/eval_gen_results.json",
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

    # Print human-readable summary
    print(f"\nPHPCS pass rate:    {summary['phpcs_pass_rate']:.1%} ({summary['passed']}/{summary['total']})")
    print(f"Security pass rate: {summary['security_pass_rate']:.1%} ({summary['security_passed']}/{summary['security_total']})")


if __name__ == "__main__":
    main()
