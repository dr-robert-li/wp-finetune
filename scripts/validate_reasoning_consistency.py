"""Consistency validation for Phase 4.1 reasoning examples.

Routes ALL examples through Claude Code agents (D-01: no deterministic rules).
Uses haiku for speed — one mega-prompt per source type with index numbering.
Output format: INDEX CONSISTENT/INCONSISTENT: reason

Usage:
    python scripts/validate_reasoning_consistency.py [--source cot|ctf|both] [--auto-regenerate] [--dry-run]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from scripts.claude_agent import generate

COT_PATH = PROJECT_ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "deep_judge_cot_bulk.json"
CTF_PATH = PROJECT_ROOT / "data" / "phase4_reasoning" / "critique_then_fix" / "critique_then_fix_bulk.json"
OUT_DIR = PROJECT_ROOT / "data" / "reasoning_dataset"
VALID_OUT = OUT_DIR / "consistency_valid.jsonl"
REJECTED_OUT = OUT_DIR / "consistency_rejected.jsonl"

MODEL = "haiku"
TIMEOUT = 300
BATCH_SIZE = 10
MAX_WORKERS = 3

SYSTEM_PROMPT = (
    "You are a WordPress code quality consistency validator. "
    "Check whether the written reasoning is consistent with the numeric scores. "
    "Look for contradictions: reasoning describes critical issues but scores are high, "
    "or reasoning says no issues but score is low. "
    "Each example has an INDEX number. Output exactly one line per example.\n"
    "Output format: INDEX CONSISTENT: reason  OR  INDEX INCONSISTENT: reason\n"
    "Example: 3 INCONSISTENT: critical injection described but security score is 8\n"
    "If consistent: INDEX CONSISTENT: null\n"
    "Do NOT skip any indices. Must output exactly N lines for N examples."
)

DIM_NAMES = ["wpcs_compliance", "sql_safety", "security", "performance",
             "wp_api_usage", "code_quality", "dependency_integrity", "i18n", "accessibility"]


def load_examples(source: str):
    path = COT_PATH if source == "cot" else CTF_PATH
    if not path.exists():
        logger.error("Missing %s", path)
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [(source, ex) for ex in data]
    key = "examples" if "examples" in data else "data"
    items = data.get(key, [])
    if not items and key == "data":
        items = [data]
    return [(source, ex) for ex in items]


def trim_cot(ex, idx):
    r = ex.get("reasoning", {})
    da = r.get("dimension_analysis", {})
    dim_strs = []
    for d in DIM_NAMES:
        info = da.get(d, {})
        score = info.get("score", "N/A")
        analysis = str(info.get("analysis", ""))[:350]
        dim_strs.append(f"    {d}: score={score}, analysis: {analysis[:280]}")
    return (
        f"INDEX {idx}:\n"
        f"  function={ex.get('function_name','?')}\n"
        f"  source={ex.get('source_file','?')}\n"
        f"  verdict={r.get('verdict','?')}\n"
        f"  overall_score={r.get('overall_score','?')}\n"
        f"  key_observation={str(r.get('key_observation',''))[:150]}\n"
        f"  dimensions:\n" + "\n".join(dim_strs)
    )


def trim_ctf(ex, idx):
    crit = ex.get("critique", {})
    dims = crit.get("dimensions", {})
    dim_strs = []
    for d in DIM_NAMES:
        info = dims.get(d, {})
        severity = info.get("severity", "?")
        issue = str(info.get("issue", ""))[:350]
        fix = str(info.get("fix", ""))[:150]
        dim_strs.append(f"    {d}: severity={severity}, issue: {issue[:280]}, fix: {fix[:120]}")
    return (
        f"INDEX {idx}:\n"
        f"  function={ex.get('function_name','?')}\n"
        f"  source={ex.get('source_file','?')}\n"
        f"  critique_summary={str(crit.get('summary',''))[:250]}\n"
        f"  key_observation={str(crit.get('key_observation',''))[:150]}\n"
        f"  dimensions:\n" + "\n".join(dim_strs)
    )


def build_mega_prompt(examples_by_type):
    """Build one mega-prompt per source type with indexed examples."""
    sections = []
    for stype, examples in examples_by_type.items():
        if not examples:
            continue
        trimmed = []
        for i, ex in enumerate(examples):
            if "reasoning" in ex or stype == "cot":
                trimmed.append(trim_cot(ex, i))
            else:
                trimmed.append(trim_ctf(ex, i))
        sections.append(f"=== {stype} stream ===\n" + "\n---\n".join(trimmed))

    return (
        "Validate reasoning-score consistency for the examples below.\n\n"
        + "\n\n".join(sections)
        + f"\n\nTotal examples to validate: {sum(len(v) for v in examples_by_type.values())}\n\n"
        "Output EXACTLY one line per example in INDEX order.\n"
        "Format: INDEX CONSISTENT: null  OR  INDEX INCONSISTENT: <reason>\n"
        "Do NOT skip indices. Do NOT add extra text or explanations."
    )


def parse_response(response, total_indices):
    """Parse agent response back to indexed results."""
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    results = {}
    for line in lines:
        # Parse: INDEX STATUS: reason
        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx > total_indices:
            continue
        if len(parts) < 2:
            results[idx] = ("unknown", f"parse_error: {line[:80]}")
            continue
        status = parts[1].rstrip(":").upper()
        reason = parts[2].strip() if len(parts) > 2 else None
        if reason and reason.lower() in ("null", "none", ""):
            reason = None
        if status == "CONSISTENT":
            results[idx] = ("consistent", reason)
        elif status == "INCONSISTENT":
            results[idx] = ("inconsistent", reason)
        else:
            results[idx] = ("unknown", f"parse_error: {line[:80]}")

    return results


def validate(source, auto_regenerate, limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_examples = []
    if source in ("cot", "both"):
        all_examples.extend(load_examples("cot"))
    if source in ("ctf", "both"):
        all_examples.extend(load_examples("ctf"))

    if limit:
        all_examples = all_examples[:limit]

    if not all_examples:
        VALID_OUT.write_text("")
        REJECTED_OUT.write_text("")
        print("No examples found.")
        return 0, 0

    # Split by type
    cot_ex = [ex for _, ex in all_examples if "reasoning" in ex]
    ctf_ex = [ex for _, ex in all_examples if "reasoning" not in ex]

    # Batched parallel validation (mega-prompts time out past ~10 examples — see
    # 04.2/.continue-here.md). Small batches + ThreadPoolExecutor keep each
    # claude --print call under the context/timeout ceiling. Backend stays haiku
    # per D-01 (no deterministic rules; every example still routed to an agent).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_stream(examples, stype):
        results = [("inconsistent", "missing_response")] * len(examples)
        batches = [(i, examples[i:i + BATCH_SIZE]) for i in range(0, len(examples), BATCH_SIZE)]

        def run_batch(start, batch):
            prompt = build_mega_prompt({stype: batch})
            try:
                resp = generate(prompt, system=SYSTEM_PROMPT, model=MODEL, timeout=TIMEOUT)
                parsed = parse_response(resp, len(batch))
            except Exception as e:
                logger.error("%s batch @%d failed: %s", stype, start, e)
                parsed = {i: ("inconsistent", "agent_error") for i in range(len(batch))}
            return start, parsed

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = [pool.submit(run_batch, start, batch) for start, batch in batches]
            done = 0
            for fut in as_completed(futs):
                start, parsed = fut.result()
                for local_i, sr in parsed.items():
                    if 0 <= start + local_i < len(examples):
                        results[start + local_i] = sr
                done += 1
                logger.info("%s: %d/%d batches done", stype, done, len(batches))
        return results

    all_results = []
    if cot_ex:
        logger.info("Validating %d CoT examples in batches of %d (workers=%d)...", len(cot_ex), BATCH_SIZE, MAX_WORKERS)
        for ex, (status, reason) in zip(cot_ex, _process_stream(cot_ex, "cot")):
            all_results.append((ex, status, reason))
    if ctf_ex:
        logger.info("Validating %d CtF examples in batches of %d (workers=%d)...", len(ctf_ex), BATCH_SIZE, MAX_WORKERS)
        for ex, (status, reason) in zip(ctf_ex, _process_stream(ctf_ex, "ctf")):
            all_results.append((ex, status, reason))

    valid = []
    rejected = []
    for ex, status, reason in all_results:
        entry = {
            "source_file": ex.get("source_file", "unknown"),
            "function_name": ex.get("function_name", "unknown"),
            "stream": "cot" if "reasoning" in ex else "ctf",
            "consistency_status": status,
            "inconsistency_reason": reason,
        }
        if status == "consistent":
            valid.append(entry)
        else:
            rejected.append(entry)

    VALID_OUT.write_text("\n".join(json.dumps(e) for e in valid) + ("\n" if valid else ""))
    REJECTED_OUT.write_text("\n".join(json.dumps(e) for e in rejected) + ("\n" if rejected else ""))

    if auto_regenerate and rejected:
        requeue_path = OUT_DIR / "requeue_for_regeneration.json"
        requeue_path.write_text(json.dumps({
            "rejected_count": len(rejected),
            "examples": rejected,
            "regenerate_prompt": "Regenerate reasoning. Previous attempt had consistency issues. "
                                 "Ensure reasoning text matches numeric scores.",
        }, indent=2, ensure_ascii=False))

    print(f"\nConsistency Validation Summary")
    print(f"  Total processed: {len(all_results)}")
    print(f"  Consistent:      {len(valid)}")
    print(f"  Inconsistent:    {len(rejected)} ({100*len(rejected)/len(all_results):.1f}%)")
    if rejected:
        print(f"\nRejected examples:")
        for r in rejected[:10]:
            print(f"  - {r['function_name']}: {r['inconsistency_reason']}")
        if len(rejected) > 10:
            print(f"  ... and {len(rejected) - 10} more")

    return len(valid), len(rejected)


def main():
    parser = argparse.ArgumentParser(description="Validate reasoning-score consistency via Claude Code agents")
    parser.add_argument("--source", choices=["cot", "ctf", "both"], default="both")
    parser.add_argument("--auto-regenerate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N examples (smoke test)")
    args = parser.parse_args()

    if args.dry_run:
        cot = load_examples("cot")
        ctf = load_examples("ctf")
        print(f"Cot: {len(cot)}, CtF: {len(ctf)}, Total: {len(cot) + len(ctf)}")
        return

    validate(args.source, args.auto_regenerate, limit=args.limit)


if __name__ == "__main__":
    main()
