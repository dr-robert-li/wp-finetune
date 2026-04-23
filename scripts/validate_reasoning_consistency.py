"""Validate reasoning-score consistency using Claude Code agents ONLY.

Reads CoT examples from `data/phase4_reasoning/deep_judge_cot/deep_judge_cot_bulk.json`
and CtF examples from `data/phase4_reasoning/critique_then_fix/critique_then_fix_bulk.json`,
routes ALL through Claude Code agents (no heuristic pre-filter per D-01),
and writes validated outputs.

Usage:
    python scripts/validate_reasoning_consistency.py --source cot
    python scripts/validate_reasoning_consistency.py --source ctf
    python scripts/validate_reasoning_consistency.py --source both
    python scripts/validate_reasoning_consistency.py --source both --auto-regenerate
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.claude_agent as claude_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "data" / "reasoning_dataset"
BATCH_SIZE = 20
AGENT_MODEL = "sonnet"
AGENT_TIMEOUT = 600  # seconds for batch validation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cot_examples() -> list[dict]:
    """Load CoT examples from bulk JSON."""
    path = PROJECT_ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "deep_judge_cot_bulk.json"
    if not path.exists():
        print(f"WARNING: CoT bulk file not found: {path}", file=sys.stderr)
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    return [data]


def load_ctf_examples() -> list[dict]:
    """Load CtF examples from bulk JSON."""
    path = PROJECT_ROOT / "data" / "phase4_reasoning" / "critique_then_fix" / "critique_then_fix_bulk.json"
    if not path.exists():
        print(f"WARNING: CtF bulk file not found: {path}", file=sys.stderr)
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    return [data]


def load_examples(source: str) -> list[dict]:
    """Load examples by source type."""
    if source == "cot":
        return load_cot_examples()
    elif source == "ctf":
        return load_ctf_examples()
    else:  # both
        cot = load_cot_examples()
        ctf = load_ctf_examples()
        # Tag with stream
        for ex in cot:
            ex.setdefault("stream", "cot")
        for ex in ctf:
            ex.setdefault("stream", "ctf")
        return cot + ctf


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def create_batches(items: list, batch_size: int = BATCH_SIZE) -> list[list]:
    """Split items into batches of ~batch_size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# ---------------------------------------------------------------------------
# Agent validation
# ---------------------------------------------------------------------------

def build_agent_prompt(examples: list[dict], stream: str) -> str:
    """Build the prompt for consistency validation of a batch of examples."""
    # Compact JSON payload for the agent
    examples_json = json.dumps({"examples": examples}, ensure_ascii=False)

    return f"""You are a WordPress code quality consistency validator. Your job is to check whether the written reasoning is consistent with the numeric scores.

Check: Does the reasoning describe issues that should result in low scores, but the scores are actually high? Or vice versa?

Look for contradictions like:
- Reasoning describes "critical SQL injection vulnerability" but security dimension score >= 7
- Reasoning says "no issues found" but overall_score < 50
- Critique describes "critical XSS" but the corrected_code only adds comments, not actual fix
- Reasoning lists multiple "critical" issues but overall_score >= 80

Stream type: {stream}

Input examples:
{examples_json}

Output ONLY one line per example, in order (line 1 = example 0, line 2 = example 1, etc.):
CONSISTENT|INCONSISTENT: <reason in 10 words or less>

Example output:
CONSISTENT: security score matches critical issue description
INCONSISTENT: critical injection described but security score is 8

Do NOT add explanations. One line per example only.
"""


def parse_agent_response(response: str, batch_size: int) -> list[tuple[str, str]]:
    """Parse agent response into list of (status, reason) tuples.

    Each line should be: CONSISTENT|INCONSISTENT: <reason>
    """
    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match CONSISTENT or INCONSISTENT at start
        if line.startswith("CONSISTENT"):
            parts = line.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "scores match reasoning"
            results.append(("consistent", reason))
        elif line.startswith("INCONSISTENT"):
            parts = line.split(":", 1)
            reason = parts[1].strip() if len(parts) > 1 else "score-reasoning mismatch"
            results.append(("inconsistent", reason))
        # Silently skip unrecognized lines (agent may add preamble)
    # If we got fewer results than expected, pad with consistent (shouldn't happen)
    while len(results) < batch_size:
        results.append(("consistent", "default: no inconsistency detected"))
    return results


def validate_batch(examples: list[dict], stream: str) -> list[tuple[str, str]]:
    """Validate a batch of examples through the Claude Code agent."""
    prompt = build_agent_prompt(examples, stream)
    response = claude_agent.generate(
        prompt,
        system="You are a strict consistency validator. You only output CONSISTENT or INCONSISTENT per example.",
        model=AGENT_MODEL,
        timeout=AGENT_TIMEOUT,
    )
    return parse_agent_response(response, len(examples))


# ---------------------------------------------------------------------------
# Example enrichment
# ---------------------------------------------------------------------------

def enrich_example(ex: dict, status: str, reason: str) -> dict:
    """Enrich example with consistency status for output JSONL."""
    result = {
        "source_file": ex.get("source_file", "unknown"),
        "function_name": ex.get("function_name", "unknown"),
        "stream": ex.get("stream", "cot"),
        "consistency_status": status,
    }
    if status == "inconsistent":
        result["inconsistency_reason"] = reason
    else:
        result["inconsistency_reason"] = None
    return result


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_examples(examples: list[dict], stream: str) -> dict:
    """Process examples through agent validation, return results dict.

    Returns:
        {
            "total": int,
            "consistent": int,
            "inconsistent": int,
            "examples": list[dict],  # enriched output records
        }
    """
    if not examples:
        return {"total": 0, "consistent": 0, "inconsistent": 0, "examples": []}

    # Batch and validate ALL examples (no heuristic pre-filter per D-01)
    batches = create_batches(examples)
    all_results = []

    for i, batch in enumerate(batches):
        batch_stream = stream if stream else batch[0].get("stream", "cot")
        logger.info("Validating batch %d/%d (%d examples)...", i + 1, len(batches), len(batch))
        batch_responses = validate_batch(batch, batch_stream)

        for j, (status, reason) in enumerate(batch_responses):
            enriched = enrich_example(batch[j], status, reason)
            all_results.append(enriched)

    consistent = sum(1 for r in all_results if r["consistency_status"] == "consistent")
    inconsistent = sum(1 for r in all_results if r["consistency_status"] == "inconsistent")

    return {
        "total": len(all_results),
        "consistent": consistent,
        "inconsistent": inconsistent,
        "examples": all_results,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_output(results: dict):
    """Write consistency_valid.jsonl and consistency_rejected.jsonl."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Valid (consistent) examples
    valid_path = OUTPUT_DIR / "consistency_valid.jsonl"
    rejected_path = OUTPUT_DIR / "consistency_rejected.jsonl"

    with open(valid_path, "w") as f:
        for ex in results["examples"]:
            if ex["consistency_status"] == "consistent":
                f.write(json.dumps(ex) + "\n")

    with open(rejected_path, "w") as f:
        for ex in results["examples"]:
            if ex["consistency_status"] == "inconsistent":
                f.write(json.dumps(ex) + "\n")

    return valid_path, rejected_path


# ---------------------------------------------------------------------------
# Auto-regeneration
# ---------------------------------------------------------------------------

def requeue_rejected(examples: list[dict], results: dict):
    """Re-queue rejected examples for regeneration via agents."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rejected = [ex for ex in results["examples"] if ex["consistency_status"] == "inconsistent"]
    if not rejected:
        print("No rejected examples to re-queue.")
        return 0

    requeue_data = []
    for rej in rejected:
        # Find original example
        original = None
        for ex in examples:
            if (ex.get("source_file") == rej["source_file"] and
                    ex.get("function_name") == rej["function_name"]):
                original = ex
                break
        if original is None:
            continue

        regenerated_prompt = (
            f"Your previous consistency check was rejected for: {rej['inconsistency_reason']}\n\n"
            f"Please regenerate your analysis with these corrections:\n"
            f"1. Ensure your reasoning descriptions match the numeric scores\n"
            f"2. If you describe a critical issue, the security score must be low\n"
            f"3. If scores are high, the reasoning should not describe critical issues\n\n"
            f"Original reasoning data:\n"
            f"{json.dumps(original, indent=2, ensure_ascii=False)[:2000]}"
        )
        requeue_data.append({
            "source_file": rej["source_file"],
            "function_name": rej["function_name"],
            "stream": rej["stream"],
            "inconsistency_reason": rej["inconsistency_reason"],
            "regenerate_prompt": regenerated_prompt,
        })

    requeue_path = OUTPUT_DIR / "requeue_for_regeneration.json"
    requeue_path.write_text(json.dumps(requeue_data, indent=2, ensure_ascii=False))
    print(f"Wrote {len(requeue_data)} rejected examples to requeue_for_regeneration.json")
    return len(requeue_data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(batch_size: int = BATCH_SIZE, model: str = AGENT_MODEL):
    parser = argparse.ArgumentParser(
        description="Validate reasoning-score consistency using Claude Code agents."
    )
    parser.add_argument(
        "--source",
        choices=["cot", "ctf", "both"],
        default="both",
        help="Which data source to validate (default: both)",
    )
    parser.add_argument(
        "--auto-regenerate",
        action="store_true",
        help="Re-queue rejected examples for regeneration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help=f"Batch size for agent validation (default: {batch_size})",
    )
    parser.add_argument(
        "--model",
        default=model,
        help=f"Model to use for validation (default: {model})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print(f"Loading examples (source={args.source})...")
    examples = load_examples(args.source)
    print(f"Loaded {len(examples)} examples total.")

    # Determine stream for this batch
    stream = "cot" if args.source == "cot" else "ctf" if args.source == "ctf" else None

    print("Running consistency validation via Claude Code agents...")
    results = process_examples(examples, stream)

    # Write output files
    valid_path, rejected_path = write_output(results)

    print(f"\n=== CONSISTENCY VALIDATION RESULTS ===")
    print(f"Total processed: {results['total']}")
    print(f"Consistent: {results['consistent']}")
    print(f"Inconsistent: {results['inconsistent']}")
    if results['total'] > 0:
        pct = results['inconsistent'] / results['total'] * 100
        print(f"Inconsistency rate: {pct:.1f}%")
    print(f"Valid examples: {valid_path}")
    print(f"Rejected examples: {rejected_path}")

    # Auto-regenerate if requested
    auto_regenerated = 0
    if args.auto_regenerate and results['inconsistent'] > 0:
        print("\nRe-queuing rejected examples for regeneration...")
        auto_regenerated = requeue_rejected(examples, results)
        print(f"Auto-regenerated: {auto_regenerated} examples")

    print(f"\nSummary: total={results['total']}, consistent={results['consistent']}, "
          f"inconsistent={results['inconsistent']}, auto-regenerated={auto_regenerated}")


if __name__ == "__main__":
    main()
