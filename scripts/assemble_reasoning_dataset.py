"""Assemble reasoning dataset: template enforcement, training mix, stratified export.

Reads consistency_valid.jsonl, applies canonical template, assembles 60/25/15 mix,
writes stratified 80/20 split to data/reasoning_dataset/.

Usage:
    python scripts/assemble_reasoning_dataset.py [--dry-run]
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

VALID_PATH = PROJECT_ROOT / "data" / "reasoning_dataset" / "consistency_valid.jsonl"
REJECTED_PATH = PROJECT_ROOT / "data" / "reasoning_dataset" / "consistency_rejected.jsonl"
COT_PATH = PROJECT_ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "deep_judge_cot_bulk.json"
CTF_PATH = PROJECT_ROOT / "data" / "phase4_reasoning" / "critique_then_fix" / "critique_then_fix_bulk.json"
REPLAY_PATH = PROJECT_ROOT / "data" / "final_dataset" / "openai_train.jsonl"
METADATA_PATH = PROJECT_ROOT / "data" / "final_dataset" / "metadata.json"
OUT_DIR = PROJECT_ROOT / "data" / "reasoning_dataset"

DIM_NAMES = ["wpcs_compliance", "sql_safety", "security", "performance", "wp_api_usage",
             "code_quality", "dependency_integrity", "i18n", "accessibility"]
DIM_LABELS = {
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
SEVERITY_TO_SCORE = {"critical": 2, "high": 4, "medium": 6, "low": 8, "none": 10}

# Auto-generated vendor/SDK code (protobuf accessors, etc.) is not first-party
# WordPress and pollutes WPCS/security supervision — reject it. Match distinctive
# tokens only (NOT bare "plugnmeet", which appears in legit WP settings code).
VENDOR_RE = re.compile(
    r"Generated from protobuf field|GPBUtil|PlugnmeetProto|Mynaparrot|\bLivekit\b",
    re.IGNORECASE,
)


def filter_reason(code):
    """Return a rejection reason for unusable judged code, else None."""
    text = code or ""
    if VENDOR_RE.search(text):
        return "vendor"
    if abs(text.count("{") - text.count("}")) > 2:
        return "truncated"
    return None


def load_consistent():
    """Load consistency-valid examples."""
    if not VALID_PATH.exists():
        logger.error("Missing %s. Run validate_reasoning_consistency.py first.", VALID_PATH)
        return []
    examples = []
    for line in VALID_PATH.read_text().strip().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("consistency_status") == "consistent":
            examples.append(e)
    return examples


def load_rejection_count():
    if not REJECTED_PATH.exists():
        return 0
    return sum(1 for l in REJECTED_PATH.read_text().strip().splitlines() if l.strip())


def load_cot_examples():
    data = json.loads(COT_PATH.read_text())
    if isinstance(data, list):
        return data
    return data.get("examples", data.get("data", [data]))


def load_ctf_examples():
    data = json.loads(CTF_PATH.read_text())
    if isinstance(data, list):
        return data
    return data.get("examples", data.get("data", [data]))


def severity_to_num(s):
    return SEVERITY_TO_SCORE.get(s.lower().strip(), 5)


def build_cot_canonical(ex):
    """Build canonical assistant content from CoT example."""
    r = ex.get("reasoning", {})
    da = r.get("dimension_analysis", {})
    parts = []
    scores = {}
    for dim in DIM_NAMES:
        info = da.get(dim)
        if not info:
            continue
        score = info.get("score", "N/A")
        analysis = info.get("analysis", "")
        label = DIM_LABELS.get(dim, dim)
        parts.append(f"{label}: score {score}/10 — {analysis}")
        scores[dim] = score if isinstance(score, (int, float)) else None

    overall = r.get("overall_score")
    verdict = r.get("verdict")

    # Build scores JSON
    scores_json = {"verdict": verdict}
    for dim in DIM_NAMES:
        v = scores.get(dim)
        if v is not None:
            scores_json[dim] = v
    if overall is not None:
        scores_json["overall_score"] = overall

    dims_addressed = [d for d in DIM_NAMES if d in da]
    return {
        "messages": [
            {"role": "user", "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{ex.get('code', '')}\n```"},
            {"role": "assistant", "content": "\n\n".join(parts) + f"\n\n[/REASONING]\n\n<judge_output>\n{json.dumps(scores_json, indent=2)}\n</judge_output>"},
        ],
        "metadata": {
            "source_file": ex.get("source_file", "unknown"),
            "function_name": ex.get("function_name", "unknown"),
            "stream": "cot",
            "format": "cot",
            "dimensions_addressed": dims_addressed,
            "source_dir": ex.get("source_dir", "unknown"),
        },
    }, dims_addressed


def build_ctf_canonical(ex):
    """Build canonical assistant content from CtF example."""
    crit = ex.get("critique", {})
    dims = crit.get("dimensions", {})
    parts = []
    scores = {}

    for dim in DIM_NAMES:
        info = dims.get(dim)
        if not info:
            continue
        severity = info.get("severity", "low")
        issue = info.get("issue", "")
        fix = info.get("fix", "")
        num_score = severity_to_num(severity)
        label = DIM_LABELS.get(dim, dim)
        parts.append(f"{label}: severity {severity} ({num_score}/10) — Issue: {issue}. Fix: {fix}")
        scores[dim] = num_score

    summary = crit.get("summary", "")
    key_obs = crit.get("key_observation", "")
    if summary:
        parts.insert(0, summary)
    if key_obs:
        parts.append(f"Key observation: {key_obs}")

    corrected = ex.get("corrected_code", "")

    scores_json = {}
    for dim in DIM_NAMES:
        if dim in scores:
            scores_json[dim] = scores[dim]
    scores_json["verdict"] = "FAIL" if any(severity_to_num(dims.get(d, {}).get("severity", "low")) < 6 for d in DIM_NAMES if d in dims) else "PASS"

    dims_addressed = [d for d in DIM_NAMES if d in dims]
    defective = ex.get("defective_code", "")
    return {
        "messages": [
            {"role": "user", "content": f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{defective}\n```"},
            {"role": "assistant", "content": "\n\n".join(parts) + f"\n\n[/REASONING]\n\n<judge_output>\n{json.dumps(scores_json, indent=2)}\n</judge_output>"},
        ],
        "metadata": {
            "source_file": ex.get("source_file", "unknown"),
            "function_name": ex.get("function_name", "unknown"),
            "stream": "ctf",
            "format": "ctf",
            "dimensions_addressed": dims_addressed,
            "source_dir": "critique_then_fix",
        },
    }, dims_addressed


def load_replay_examples(target_n):
    """Load replay examples stratified by source_file (domain) from final dataset."""
    meta = json.loads(METADATA_PATH.read_text())
    taxonomy = meta.get("taxonomy_coverage", {})

    # Read replay JSONL and group by domain (source_file)
    domain_examples = defaultdict(list)
    domain_count = 0
    with open(REPLAY_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get("metadata", {})
            domain = meta.get("source_file", meta.get("domain", "unknown"))
            domain_examples[domain].append(ex)
            domain_count += 1
            if domain_count >= target_n * 3:
                break

    # Stratified sampling: proportional from each domain
    total_domains = sum(len(v) for v in domain_examples.values())
    sampled = []
    remaining = target_n
    for domain, items in sorted(domain_examples.items()):
        if remaining <= 0:
            break
        share = max(1, round(len(items) / total_domains * target_n * 2))  # oversample to ensure diversity
        share = min(share, remaining, len(items))
        sampled.extend(random.sample(items, share))
        remaining -= share

    if remaining > 0:
        # Fill from remaining domains
        all_remaining = []
        for items in domain_examples.values():
            all_remaining.extend(items)
        random.shuffle(all_remaining)
        sampled.extend(all_remaining[:remaining])

    out = sampled[:target_n]
    for ex in out:
        md = ex.setdefault("metadata", {})
        md.setdefault("stream", "replay")
        md.setdefault("format", "replay")
    return out


def stratified_split(examples, train_ratio=0.8, seed=42):
    """Split stratified by stream (cot/ctf/replay) so train and val preserve the
    aggregate stream mix. Domain-stratifying instead forced every singleton-domain
    example into train (max(1, round(0.8*N))), starving val of the most
    domain-fragmented stream (CoT). Random within stream spreads domains."""
    random.seed(seed)
    by_stream = defaultdict(list)
    for ex in examples:
        stream = (ex.get("metadata") or {}).get("stream") or "unknown"
        by_stream[stream].append(ex)

    train = []
    val = []
    for stream, items in sorted(by_stream.items()):
        random.shuffle(items)
        split = max(1, round(len(items) * train_ratio))
        train.extend(items[:split])
        val.extend(items[split:])

    return train, val


def compute_mix_pct(examples, stream):
    """Compute percentage of examples with given stream/format."""
    if not examples:
        return 0.0
    count = sum(1 for e in examples if e.get("metadata", {}).get("stream") == stream)
    return round(100 * count / len(examples), 1)


def assemble(dry_run=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load consistent examples
    consistent = load_consistent()
    if not consistent:
        logger.error("No consistent examples found.")
        return

    # 2. Build canonical examples from CoT and CtF
    cot_raw = load_cot_examples()
    ctf_raw = load_ctf_examples()

    cot_canonical = []
    ctf_canonical = []
    filter_counts = {"vendor": 0, "truncated": 0}

    # Match consistent examples with raw data by function_name + source_file
    consistent_map = {(e.get("source_file"), e.get("function_name")): e for e in consistent}

    for ex in cot_raw:
        key = (ex.get("source_file"), ex.get("function_name"))
        if key in consistent_map and consistent_map[key].get("consistency_status") == "consistent":
            reason = filter_reason(ex.get("code", ""))
            if reason:
                filter_counts[reason] += 1
                continue
            canonical, _ = build_cot_canonical(ex)
            cot_canonical.append(canonical)

    for ex in ctf_raw:
        key = (ex.get("source_file"), ex.get("function_name"))
        if key in consistent_map and consistent_map[key].get("consistency_status") == "consistent":
            reason = filter_reason(ex.get("defective_code", ""))
            if reason:
                filter_counts[reason] += 1
                continue
            canonical, _ = build_ctf_canonical(ex)
            ctf_canonical.append(canonical)

    reasoning_total = len(cot_canonical) + len(ctf_canonical)
    if reasoning_total == 0:
        logger.error("No canonical examples built. Check consistency_valid.jsonl matching.")
        return

    # 3. Training mix assembly.
    # Mix policy (2026-05-21): keep ALL consistent reasoning examples (CoT supply-
    # caps below the 60% target, so the strict 60/25 split would discard valid CtF).
    # Replay fills to 15% of the final total. Natural CoT:CtF ratio retained.
    replay_target = max(30, round(reasoning_total * 15 / 85))

    actual_cot = cot_canonical
    actual_ctf = ctf_canonical

    logger.info("Mix (all-reasoning policy): CoT=%d, CtF=%d, Replay target=%d",
                len(actual_cot), len(actual_ctf), replay_target)

    random.seed(42)
    replay_examples = load_replay_examples(replay_target)
    clean_replay = []
    for ex in replay_examples:
        user_text = " ".join(
            m.get("content", "") for m in ex.get("messages", []) if m.get("role") == "user"
        )
        if VENDOR_RE.search(user_text):
            filter_counts["vendor"] += 1
            continue
        clean_replay.append(ex)
    replay_examples = clean_replay
    logger.info("Loaded %d replay examples (after vendor filter)", len(replay_examples))

    # 4. Combine all
    all_examples = actual_cot + actual_ctf + replay_examples
    random.shuffle(all_examples)

    # 5. Stratified split
    train, val = stratified_split(all_examples)

    if dry_run:
        print(f"DRY RUN:")
        print(f"  CoT: {len(actual_cot)}, CtF: {len(actual_ctf)}, Replay: {len(replay_examples)}")
        print(f"  Total: {len(all_examples)}, Train: {len(train)}, Val: {len(val)}")
        print(f"  Filtered: vendor={filter_counts['vendor']}, truncated={filter_counts['truncated']}")
        return

    # 6. Write output
    def write_jsonl(path, examples):
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n")

    train_path = OUT_DIR / "openai_train.jsonl"
    val_path = OUT_DIR / "openai_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    # 7. Metadata
    rejection_count = load_rejection_count()
    by_domain = defaultdict(int)
    for ex in all_examples:
        by_domain[(ex.get("metadata") or {}).get("source_file") or "__replay__"] += 1

    def split_mix(split_examples):
        return {
            "cot_pct": compute_mix_pct(split_examples, "cot"),
            "ctf_pct": compute_mix_pct(split_examples, "ctf"),
            "replay_pct": 100 - compute_mix_pct(split_examples, "cot") - compute_mix_pct(split_examples, "ctf"),
        }

    metadata = {
        "phase": "04.2",
        "generated_at": "2026-04-23",
        "total_examples": len(all_examples),
        "rejection_counts": {
            "consistency": rejection_count,
            "template_noncompliant": 0,
            "dedup_removal": 0,
            "vendor_contamination": filter_counts["vendor"],
            "truncated_invalid": filter_counts["truncated"],
        },
        "mix": {
            "cot_count": len(actual_cot),
            "ctf_count": len(actual_ctf),
            "replay_count": len(replay_examples),
            "cot_percent": round(100 * len(actual_cot) / len(all_examples), 1),
            "ctf_percent": round(100 * len(actual_ctf) / len(all_examples), 1),
            "replay_percent": round(100 * len(replay_examples) / len(all_examples), 1),
        },
        "split": {
            "train_count": len(train),
            "val_count": len(val),
            "split_ratio": "80/20",
            "train_mix": split_mix(train),
            "val_mix": split_mix(val),
        },
        "taxonomy_coverage": dict(sorted(by_domain.items(), key=lambda x: -x[1])),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_meta = OUT_DIR / "metadata.json.tmp"
    tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    tmp_meta.rename(OUT_DIR / "metadata.json")

    # 8. Summary
    print(f"\n{'='*50}")
    print(f"Reasoning Dataset Assembly Complete")
    print(f"{'='*50}")
    print(f"  Consistent examples: {reasoning_total}")
    print(f"  CoT: {len(actual_cot)}, CtF: {len(actual_ctf)}, Replay: {len(replay_examples)}")
    print(f"  Total: {len(all_examples)}")
    print(f"  Train: {len(train)}, Val: {len(val)} ({100*len(val)/len(all_examples):.1f}%)")
    print(f"  Mix: CoT={metadata['mix']['cot_percent']}%, CtF={metadata['mix']['ctf_percent']}%, Replay={metadata['mix']['replay_percent']}%")
    print(f"  Train mix: CoT={split_mix(train)['cot_pct']}%, CtF={split_mix(train)['ctf_pct']}%")
    print(f"  Val mix: CoT={split_mix(val)['cot_pct']}%, CtF={split_mix(val)['ctf_pct']}%")
    print(f"  Rejected: consistency={rejection_count}, vendor={filter_counts['vendor']}, truncated={filter_counts['truncated']}")
    print(f"\nOutput: {OUT_DIR}")
    print(f"  openai_train.jsonl: {train_path.stat().st_size:,} bytes")
    print(f"  openai_val.jsonl: {val_path.stat().st_size:,} bytes")
    print(f"  metadata.json: written")


def main():
    parser = argparse.ArgumentParser(description="Assemble reasoning dataset with canonical template, mix, and split")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assemble(args.dry_run)


if __name__ == "__main__":
    main()
