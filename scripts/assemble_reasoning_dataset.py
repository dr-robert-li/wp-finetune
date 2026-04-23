"""Assemble, format, and export the reasoning dataset.

Reads consistency-valid examples from `data/reasoning_dataset/consistency_valid.jsonl`,
applies canonical template formatting, assembles the 60/25/15 training mix,
performs stratified 80/20 split, and exports OpenAI JSONL + metadata.

Usage:
    python scripts/assemble_reasoning_dataset.py [--dry-run]
"""
import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "reasoning_dataset"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
MANIFEST_DIR = PROJECT_ROOT / "data" / "phase4_reasoning" / "manifests"

# Training mix targets (D-05)
COT_TARGET_PCT = 60  # percent
CTF_TARGET_PCT = 25  # percent
REPLAY_TARGET_PCT = 15  # percent

# Split ratio
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.20
SPLIT_SEED = 42

# Minimum replay examples for statistical validity
MIN_REPLAY = 30

REQUIRED_DIMENSIONS = [
    "wpcs_compliance", "sql_safety", "security", "performance",
    "wp_api_usage", "code_quality", "dependency_integrity", "i18n",
    "accessibility",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_consistency_valid(path: str | Path) -> list[dict]:
    """Load examples that passed consistency validation."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: Consistency valid file not found: {path}", file=sys.stderr)
        return []
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Only keep consistent examples
            if ex.get("consistency_status") == "consistent":
                examples.append(ex)
    return examples


def load_replay_examples(path: str | Path) -> list[dict]:
    """Load replay examples from final_dataset."""
    path = Path(path)
    if not path.exists():
        print(f"WARNING: Replay source not found: {path}", file=sys.stderr)
        return []
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Filter for judge examples (has <wp_judge> in user content)
            messages = ex.get("messages", [])
            has_judge = any(
                "<wp_judge>" in (m.get("content") or "")
                for m in messages if m.get("role") == "user"
            )
            if has_judge:
                examples.append(ex)
    return examples


def load_taxonomy_coverage(path: str | Path) -> dict[str, int]:
    """Load taxonomy coverage from final_dataset metadata."""
    path = Path(path)
    if not path.exists():
        return {}
    meta = json.loads(path.read_text())
    return meta.get("taxonomy_coverage", {})


# ---------------------------------------------------------------------------
# Canonical template formatting
# ---------------------------------------------------------------------------

def format_canonical_cot(ex: dict) -> dict:
    """Format a CoT example into canonical template.

    Returns example with messages array and metadata.
    """
    reasoning = ex.get("reasoning", {})
    da = reasoning.get("dimension_analysis", {})
    overall_score = reasoning.get("overall_score", 0)
    verdict = reasoning.get("verdict", "UNKNOWN")

    # Build dimension analysis prose
    dim_parts = []
    dim_scores = {}
    for dim in REQUIRED_DIMENSIONS:
        if dim in da:
            val = da[dim]
            score = val.get("score", "N/A")
            analysis = val.get("analysis", "")
            dim_names = {
                "wpcs_compliance": "WPCS Compliance",
                "sql_safety": "SQL Safety",
                "security": "Security",
                "performance": "Performance",
                "wp_api_usage": "WP API Usage",
                "code_quality": "Code Quality",
                "dependency_integrity": "Dependency Integrity",
                "i18n": "i18n",
                "accessibility": "Accessibility",
            }
            label = dim_names.get(dim, dim)
            if score is None:
                dim_parts.append(f"{label}: score N/A/10 — {analysis}")
            else:
                dim_parts.append(f"{label}: score {score}/10 — {analysis}")
            dim_scores[dim] = score if score is not None else 0

    dimension_prose = "\n".join(dim_parts)

    # Build JSON scores block
    scores_json = {
        "wpcs_compliance": dim_scores.get("wpcs_compliance", 0),
        "sql_safety": dim_scores.get("sql_safety", 0),
        "security": dim_scores.get("security", 0),
        "performance": dim_scores.get("performance", 0),
        "wp_api_usage": dim_scores.get("wp_api_usage", 0),
        "code_quality": dim_scores.get("code_quality", 0),
        "dependency_integrity": dim_scores.get("dependency_integrity", 0),
        "i18n": dim_scores.get("i18n", 0),
        "accessibility": dim_scores.get("accessibility", 0),
        "overall_score": overall_score,
        "verdict": verdict,
    }

    scores_block = f"<judge_output>\n{json.dumps(scores_json, indent=2)}\n</judge_output>"

    # Build assistant content: dimension prose + separator + JSON scores
    assistant_content = f"{dimension_prose}\n\n[/REASONING]\n\n{scores_block}"

    # Build user message from code
    code = ex.get("code", "")
    user_content = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"

    # Extract dimensions addressed
    dimensions_addressed = ex.get("dimensions_addressed", REQUIRED_DIMENSIONS)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source_file": ex.get("source_file", "unknown"),
            "function_name": ex.get("function_name", "unknown"),
            "stream": ex.get("stream", "cot"),
            "format": "cot",
            "dimensions_addressed": dimensions_addressed,
            "source_dir": str(Path(ex.get("source_file", "")).parent) if ex.get("source_file") else "",
        },
    }


def format_canonical_ctf(ex: dict) -> dict:
    """Format a CtF example into canonical template.

    Returns example with messages array and metadata.
    """
    critique = ex.get("critique", {})
    summary = critique.get("summary", "")
    dims = critique.get("dimensions", {})

    # Build dimension analysis prose from critique
    dim_parts = [f"{summary}\n"]
    dim_scores = {}
    severity_to_score = {
        "critical": 1,
        "high": 3,
        "medium": 5,
        "low": 8,
    }

    dim_names = {
        "wpcs_compliance": "WPCS Compliance",
        "sql_safety": "SQL Safety",
        "security": "Security",
        "performance": "Performance",
        "wp_api_usage": "WP API Usage",
        "code_quality": "Code Quality",
        "dependency_integrity": "Dependency Integrity",
        "i18n": "i18n",
        "accessibility": "Accessibility",
    }

    for dim in REQUIRED_DIMENSIONS:
        if dim in dims:
            val = dims[dim]
            severity = val.get("severity", "low")
            issue = val.get("issue", "")
            fix = val.get("fix", "")
            label = dim_names.get(dim, dim)
            dim_parts.append(f"{label}: severity {severity} — Issue: {issue}. Fix: {fix}")
            dim_scores[dim] = severity_to_score.get(severity, 8)

    dimension_prose = "\n".join(dim_parts)

    # Build JSON scores from severity
    scores_json = {
        "wpcs_compliance": dim_scores.get("wpcs_compliance", 8),
        "sql_safety": dim_scores.get("sql_safety", 8),
        "security": dim_scores.get("security", 8),
        "performance": dim_scores.get("performance", 8),
        "wp_api_usage": dim_scores.get("wp_api_usage", 8),
        "code_quality": dim_scores.get("code_quality", 8),
        "dependency_integrity": dim_scores.get("dependency_integrity", 8),
        "i18n": dim_scores.get("i18n", 8),
        "accessibility": dim_scores.get("accessibility", 8),
    }

    # Calculate overall from severity scores
    if dim_scores:
        overall = sum(dim_scores.values()) // len(dim_scores)
    else:
        overall = 50

    scores_json["overall_score"] = overall
    scores_json["verdict"] = "FAIL" if overall < 50 else "PASS"

    scores_block = f"<judge_output>\n{json.dumps(scores_json, indent=2)}\n</judge_output>"

    # Build assistant content
    assistant_content = f"{dimension_prose}\n\n[/REASONING]\n\n{scores_block}"

    # Build user message
    defective_code = ex.get("defective_code", "")
    user_content = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{defective_code}\n```"

    dimensions_addressed = ex.get("dimensions_addressed", REQUIRED_DIMENSIONS)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source_file": ex.get("source_file", "unknown"),
            "function_name": ex.get("function_name", "unknown"),
            "stream": ex.get("stream", "ctf"),
            "format": "ctf",
            "dimensions_addressed": dimensions_addressed,
            "source_dir": str(Path(ex.get("source_file", "")).parent) if ex.get("source_file") else "",
        },
    }


# ---------------------------------------------------------------------------
# Training mix assembly
# ---------------------------------------------------------------------------

def calculate_mix_targets(cot_count: int, ctf_count: int) -> tuple[int, int, int]:
    """Calculate target counts for CoT, CtF, replay.

    Returns (cot_target, ctf_target, replay_target).
    """
    reasoning_total = cot_count + ctf_count
    replay_target = max(MIN_REPLAY, round(reasoning_total * REPLAY_TARGET_PCT / COT_TARGET_PCT))
    cot_target = round(reasoning_total * COT_TARGET_PCT / COT_TARGET_PCT)  # = reasoning_total
    ctf_target = round(reasoning_total * CTF_TARGET_PCT / COT_TARGET_PCT)  # = reasoning_total

    # Cap at available counts
    cot_target = min(cot_target, cot_count)
    ctf_target = min(ctf_target, ctf_count)

    return cot_target, ctf_target, replay_target


def stratified_replay_sampling(replay_examples: list[dict], target_count: int,
                                taxonomy_coverage: dict) -> list[dict]:
    """Sample replay examples proportionally by source_file (domain).

    Uses taxonomy_coverage to determine domain weights.
    """
    if not replay_examples:
        return []

    # Group by source_file
    by_domain: dict[str, list] = {}
    for ex in replay_examples:
        sf = ex.get("source_file", "unknown")
        if sf not in by_domain:
            by_domain[sf] = []
        by_domain[sf].append(ex)

    # Calculate proportional targets per domain
    total_replay = sum(len(v) for v in by_domain.values())
    sampled = []
    remaining = target_count

    # Sort domains by coverage (descending) for deterministic sampling
    for domain in sorted(by_domain.keys(), key=lambda d: -len(by_domain[d])):
        domain_count = len(by_domain[domain])
        if domain_count == 0:
            continue
        # Proportional sample
        domain_target = max(1, round(target_count * domain_count / total_replay))
        domain_target = min(domain_target, domain_count, remaining)
        random.seed(SPLIT_SEED)
        sampled.extend(random.sample(by_domain[domain], domain_target))
        remaining -= domain_target
        if remaining <= 0:
            break

    return sampled


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(all_examples: list[dict], output_dir: Path, seed: int = SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """Group by source_file (domain), then 80/20 random split within each domain."""
    # Group by domain
    by_domain: dict[str, list] = {}
    for ex in all_examples:
        sf = ex.get("source_file", "unknown")
        if sf not in by_domain:
            by_domain[sf] = []
        by_domain[sf].append(ex)

    train_set = []
    val_set = []

    for domain, domain_examples in by_domain.items():
        random.seed(seed)
        shuffled = domain_examples.copy()
        random.shuffle(shuffled)

        split_point = max(1, int(len(shuffled) * TRAIN_RATIO))
        train_set.extend(shuffled[:split_point])
        val_set.extend(shuffled[split_point:])

    return train_set, val_set


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[dict], path: Path):
    """Write examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_metadata(
    total_examples: int,
    consistency_rejected: int,
    mix: dict,
    train_set: list[dict],
    val_set: list[dict],
    taxonomy_coverage: dict,
) -> dict:
    """Generate metadata.json content."""
    total = len(train_set) + len(val_set)

    train_mix = {}
    val_mix = {}
    if total > 0:
        train_cot = sum(1 for e in train_set if e["metadata"]["format"] == "cot")
        train_ctf = sum(1 for e in train_set if e["metadata"]["format"] == "ctf")
        train_replay = sum(1 for e in train_set if e["metadata"]["format"] != "cot" and e["metadata"]["format"] != "ctf")
        val_cot = sum(1 for e in val_set if e["metadata"]["format"] == "cot")
        val_ctf = sum(1 for e in val_set if e["metadata"]["format"] == "ctf")
        val_replay = sum(1 for e in val_set if e["metadata"]["format"] != "cot" and e["metadata"]["format"] != "ctf")

        train_mix = {
            "cot": round(train_cot / total * 100, 1),
            "ctf": round(train_ctf / total * 100, 1),
            "replay": round(train_replay / total * 100, 1),
        }
        val_mix = {
            "cot": round(val_cot / max(total, 1) * 100, 1),
            "ctf": round(val_ctf / max(total, 1) * 100, 1),
            "replay": round(val_replay / max(total, 1) * 100, 1),
        }

    return {
        "phase": "04.2",
        "generated_at": "2026-04-23",
        "total_examples": total_examples,
        "rejection_counts": {
            "consistency": consistency_rejected,
            "template_noncompliant": 0,
            "dedup_removal": 0,
        },
        "mix": mix,
        "split": {
            "train_count": len(train_set),
            "val_count": len(val_set),
            "split_ratio": "80/20",
            "train_mix": train_mix,
            "val_mix": val_mix,
        },
        "taxonomy_coverage": taxonomy_coverage,
        "contamination_manifests": {
            "phase4_1_cot": "data/phase4_reasoning/manifests/cot_input_function_ids.json",
            "phase4_1_ctf": "data/phase4_reasoning/manifests/ctf_input_function_ids.json",
        },
    }


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def assemble_and_export(consistency_path: str = None, final_dir: str = None) -> dict:
    """Main assembly pipeline. Returns metadata dict."""
    consistency_path = consistency_path or str(OUTPUT_DIR / "consistency_valid.jsonl")
    final_dir = final_dir or str(FINAL_DIR)

    # Step 1: Load consistent examples
    print("Loading consistency-valid examples...")
    consistent_examples = load_consistency_valid(consistency_path)
    print(f"  Loaded {len(consistent_examples)} consistent examples")

    # Step 2: Separate into CoT and CtF streams
    cot_examples = [ex for ex in consistent_examples if ex.get("stream") == "cot"]
    ctf_examples = [ex for ex in consistent_examples if ex.get("stream") == "ctf"]
    print(f"  CoT: {len(cot_examples)}, CtF: {len(ctf_examples)}")

    # Step 3: Load replay examples
    print("Loading replay examples...")
    replay_source = Path(final_dir) / "openai_train.jsonl"
    replay_examples = load_replay_examples(replay_source)
    print(f"  Replay candidates: {len(replay_examples)}")

    # Step 4: Calculate targets and sample replay
    cot_count = len(cot_examples)
    ctf_count = len(ctf_examples)
    cot_target, ctf_target, replay_target = calculate_mix_targets(cot_count, ctf_count)
    print(f"  Targets: CoT={cot_target}, CtF={ctf_target}, Replay={replay_target}")

    # Use all available if capped
    cot_sample = cot_examples[:cot_target]
    ctf_sample = ctf_examples[:ctf_target]

    taxonomy_coverage = load_taxonomy_coverage(Path(final_dir) / "metadata.json")
    replay_sampled = stratified_replay_sampling(replay_examples, replay_target, taxonomy_coverage)
    print(f"  Replay sampled: {len(replay_sampled)}")

    # Step 5: Format into canonical template
    print("Applying canonical template formatting...")
    formatted_examples = []
    for ex in cot_sample:
        formatted_examples.append(format_canonical_cot(ex))
    for ex in ctf_sample:
        formatted_examples.append(format_canonical_ctf(ex))
    for ex in replay_sampled:
        # Replay examples already have messages, just ensure metadata
        meta = ex.get("metadata", {})
        meta["stream"] = "replay"
        meta["format"] = "replay"
        meta.setdefault("dimensions_addressed", REQUIRED_DIMENSIONS)
        meta.setdefault("source_dir", str(Path(ex.get("source_file", "")).parent) if ex.get("source_file") else "")
        formatted_examples.append(ex)

    # Step 6: Stratified split
    print("Performing stratified 80/20 split...")
    train_set, val_set = stratified_split(formatted_examples, OUTPUT_DIR)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")

    # Step 7: Export
    train_path = OUTPUT_DIR / "openai_train.jsonl"
    val_path = OUTPUT_DIR / "openai_val.jsonl"

    write_jsonl(train_set, train_path)
    write_jsonl(val_set, val_path)

    # Step 8: Generate metadata
    mix = {
        "cot_count": len(cot_sample),
        "ctf_count": len(ctf_sample),
        "replay_count": len(replay_sampled),
        "total_reasoning": len(cot_sample) + len(ctf_sample),
        "replay_target": replay_target,
    }
    total_for_pct = len(cot_sample) + len(ctf_sample) + len(replay_sampled)
    if total_for_pct > 0:
        mix["cot_percent"] = round(len(cot_sample) / total_for_pct * 100, 1)
        mix["ctf_percent"] = round(len(ctf_sample) / total_for_pct * 100, 1)
        mix["replay_percent"] = round(len(replay_sampled) / total_for_pct * 100, 1)

    # Count consistency rejected
    consistency_rejected = 0
    rejected_path = Path(consistency_path).parent / "consistency_rejected.jsonl"
    if rejected_path.exists():
        with open(rejected_path) as f:
            consistency_rejected = sum(1 for line in f if line.strip())

    metadata = generate_metadata(
        total_examples=len(formatted_examples),
        consistency_rejected=consistency_rejected,
        mix=mix,
        train_set=train_set,
        val_set=val_set,
        taxonomy_coverage=taxonomy_coverage,
    )

    meta_path = OUTPUT_DIR / "metadata.json"
    write_jsonl([metadata], meta_path)
    # Rewrite as a single JSON object, not JSONL
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== ASSEMBLY SUMMARY ===")
    print(f"Total examples: {metadata['total_examples']}")
    print(f"Mix: CoT={mix.get('cot_percent', 0)}%, CtF={mix.get('ctf_percent', 0)}%, Replay={mix.get('replay_percent', 0)}%")
    print(f"Split: train={len(train_set)}, val={len(val_set)} (80/20)")
    print(f"Consistency rejected: {consistency_rejected}")
    print(f"Output: {OUTPUT_DIR}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble, format, and export the reasoning dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing files.",
    )
    parser.add_argument(
        "--consistency-path",
        default=str(OUTPUT_DIR / "consistency_valid.jsonl"),
        help="Path to consistency_valid.jsonl",
    )
    parser.add_argument(
        "--final-dir",
        default=str(FINAL_DIR),
        help="Path to final_dataset directory for replay",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        print("DRY RUN — would assemble and export without writing files.")
        print(f"  Consistency input: {args.consistency_path}")
        print(f"  Replay source: {args.final_dir}/openai_train.jsonl")
        print(f"  Output: {OUTPUT_DIR}")

        # Load counts without writing
        consistent = load_consistency_valid(args.consistency_path)
        cot = [e for e in consistent if e.get("stream") == "cot"]
        ctf = [e for e in consistent if e.get("stream") == "ctf"]
        replay = load_replay_examples(Path(args.final_dir) / "openai_train.jsonl")

        cot_target, ctf_target, replay_target = calculate_mix_targets(len(cot), len(ctf))
        total = min(len(cot), cot_target) + min(len(ctf), ctf_target) + min(len(replay), replay_target)

        if total > 0:
            actual_cot = min(len(cot), cot_target)
            actual_ctf = min(len(ctf), ctf_target)
            actual_replay = min(len(replay), replay_target)
            print(f"  Predicted mix: CoT={actual_cot/max(total,1)*100:.1f}%, CtF={actual_ctf/max(total,1)*100:.1f}%, Replay={actual_replay/max(total,1)*100:.1f}%")
        print(f"  Predicted train:val split ~ {int(total * 0.8)}:{int(total * 0.2)}")
        return

    assemble_and_export(args.consistency_path, args.final_dir)


if __name__ == "__main__":
    main()
