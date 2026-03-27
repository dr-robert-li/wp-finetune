#!/usr/bin/env python3
"""Export final dataset in formats for different finetuning platforms.

Reads from data/final_dataset/wordpress_finetune.jsonl and exports:
- MoE format with <wp_gen>/<wp_judge> task tokens (Alpaca for Llama-MoE)
- OpenAI format (messages array with system/user/assistant)
- Split into train/validation/test sets (80/10/10)
"""

import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
SOURCE_PATH = FINAL_DIR / "wordpress_finetune.jsonl"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

# 40/60 gen/judge ratio target (locked decision).
GEN_TARGET_RATIO = 0.40
JUDGE_TARGET_RATIO = 0.60


def load_dataset() -> list[dict]:
    examples = []
    with open(SOURCE_PATH) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def infer_task_type(example: dict) -> str:
    """Determine whether this is a generation or judging example."""
    metadata = example.get("metadata", {})

    # Explicit judge examples.
    if metadata.get("task_type") == "judge":
        return "judge"

    # Check if user message already has task token.
    messages = example.get("messages", [])
    for msg in messages:
        if msg["role"] == "user":
            if "<wp_judge>" in msg["content"]:
                return "judge"
            if "<wp_gen>" in msg["content"]:
                return "gen"

    # Default: generation.
    return "gen"


def add_task_token(example: dict) -> dict:
    """Add <wp_gen> or <wp_judge> task token to the user message if missing."""
    task_type = infer_task_type(example)
    token = "<wp_judge>" if task_type == "judge" else "<wp_gen>"

    messages = []
    for msg in example["messages"]:
        if msg["role"] == "user":
            content = msg["content"]
            # Only add if not already present.
            if "<wp_gen>" not in content and "<wp_judge>" not in content:
                content = f"{token} {content}"
            messages.append({"role": msg["role"], "content": content})
        else:
            messages.append(msg)

    result = dict(example)
    result["messages"] = messages
    return result


def to_openai_format(example: dict) -> dict:
    """OpenAI finetuning format — messages array only, no metadata."""
    tokenized = add_task_token(example)
    return {"messages": tokenized["messages"]}


def to_alpaca_format(example: dict) -> dict:
    """Alpaca/Llama-MoE format with task tokens in instruction."""
    tokenized = add_task_token(example)
    messages = tokenized["messages"]
    system = ""
    instruction = ""
    output = ""

    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            instruction = msg["content"]
        elif msg["role"] == "assistant":
            output = msg["content"]

    return {
        "instruction": f"{system}\n\n{instruction}" if system else instruction,
        "input": "",
        "output": output,
    }


def to_raw_format(example: dict) -> dict:
    """Raw format preserving all metadata for analysis."""
    tokenized = add_task_token(example)
    return {
        "messages": tokenized["messages"],
        "metadata": example.get("metadata", {}),
        "task_type": infer_task_type(example),
    }


def enforce_ratio(dataset: list[dict]) -> list[dict]:
    """Enforce 40/60 gen/judge ratio by capping the majority class."""
    gen_examples = [ex for ex in dataset if infer_task_type(ex) == "gen"]
    judge_examples = [ex for ex in dataset if infer_task_type(ex) == "judge"]
    gen_count = len(gen_examples)
    judge_count = len(judge_examples)
    if gen_count == 0 or judge_count == 0:
        return dataset
    ideal_judge = round(gen_count * (JUDGE_TARGET_RATIO / GEN_TARGET_RATIO))
    if judge_count > ideal_judge:
        random.seed(42)
        judge_examples = random.sample(judge_examples, ideal_judge)
    else:
        ideal_gen = round(judge_count * (GEN_TARGET_RATIO / JUDGE_TARGET_RATIO))
        if gen_count > ideal_gen:
            random.seed(42)
            gen_examples = random.sample(gen_examples, ideal_gen)
    return gen_examples + judge_examples


def add_sample_weight(example: dict) -> dict:
    """Add sample_weight metadata for training loss weighting.

    Contrastive (mutated) and low-score examples get weight 1.5.
    All others get weight 1.0.
    """
    result = dict(example)
    meta = dict(result.get("metadata", {}))
    source = meta.get("source", "")
    is_contrastive = source in ("mutated", "contrastive")
    is_low_score = meta.get("overall_score", 10) < 7
    meta["sample_weight"] = 1.5 if (is_contrastive or is_low_score) else 1.0
    result["metadata"] = meta
    return result


def deduplicate(dataset: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate examples based on assistant message content hash."""
    seen: set[str] = set()
    unique = []
    dupes = 0
    for ex in dataset:
        msgs = ex.get("messages", [])
        assistant_text = "".join(m["content"] for m in msgs if m["role"] == "assistant")
        h = hashlib.sha256(assistant_text.encode()).hexdigest()
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        unique.append(ex)
    return unique, dupes


def validate_php_sample(dataset: list[dict], sample_size: int = 50) -> int:
    """Run php -l on a sample of assistant responses. Returns failure count."""
    sample = random.sample(dataset, min(sample_size, len(dataset)))
    failures = 0
    for ex in sample:
        msgs = ex.get("messages", [])
        code = "".join(m["content"] for m in msgs if m["role"] == "assistant")
        if not code.strip():
            continue
        # Wrap in <?php if needed.
        if not code.strip().startswith("<?php"):
            code = f"<?php\n{code}"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            result = subprocess.run(
                ["php", "-l", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                failures += 1
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # php not available, skip lint.
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
    return failures


def generate_metadata(
    dataset: list[dict],
    train_set: list[dict],
    val_set: list[dict],
    test_set: list[dict],
    php_lint_failures: int,
    dupes_removed: int,
) -> dict:
    """Generate full stats report for data/final_dataset/metadata.json."""
    gen_count = sum(1 for ex in dataset if infer_task_type(ex) == "gen")
    judge_count = sum(1 for ex in dataset if infer_task_type(ex) == "judge")
    total = max(len(dataset), 1)

    # Taxonomy coverage.
    tag_counts: dict[str, int] = {}
    for ex in dataset:
        tags = ex.get("metadata", {}).get("training_tags", [])
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    rejection_count = sum(
        1 for ex in dataset
        if any("rejection:" in str(t) for t in ex.get("metadata", {}).get("training_tags", []))
    )

    return {
        "total_examples": len(dataset),
        "gen_count": gen_count,
        "judge_count": judge_count,
        "gen_judge_ratio": f"{gen_count}/{judge_count}",
        "gen_ratio_actual": round(gen_count / total, 3),
        "judge_ratio_actual": round(judge_count / total, 3),
        "gen_ratio_target": GEN_TARGET_RATIO,
        "judge_ratio_target": JUDGE_TARGET_RATIO,
        "rejection_examples": rejection_count,
        "taxonomy_coverage": tag_counts,
        "taxonomy_gaps_remaining": [tag for tag, count in tag_counts.items() if count < 20],
        "train_val_test_counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "php_lint_failures": php_lint_failures,
        "duplicates_removed": dupes_removed,
        "sample_weighted_count": sum(
            1 for ex in dataset if ex.get("metadata", {}).get("sample_weight", 1.0) > 1.0
        ),
        "spot_check_required": True,
        "phase2_complete": False,
    }


def main():
    if not SOURCE_PATH.exists():
        print("No final dataset found. Run phase3_cot.py first.")
        sys.exit(1)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} examples")

    # Deduplicate.
    dataset, dupes_removed = deduplicate(dataset)
    print(f"Deduplicated: {dupes_removed} duplicates removed, {len(dataset)} unique examples remain")

    # Enforce 40/60 gen/judge ratio.
    dataset = enforce_ratio(dataset)
    gen_count = sum(1 for ex in dataset if infer_task_type(ex) == "gen")
    judge_count = sum(1 for ex in dataset if infer_task_type(ex) == "judge")
    print(f"After ratio enforcement: {gen_count} gen ({gen_count/max(len(dataset),1):.1%}), "
          f"{judge_count} judge ({judge_count/max(len(dataset),1):.1%})")

    # Add sample weights for contrastive/low-score examples.
    dataset = [add_sample_weight(ex) for ex in dataset]
    weighted_count = sum(1 for ex in dataset if ex.get("metadata", {}).get("sample_weight", 1.0) > 1.0)
    print(f"Sample weights: {weighted_count} examples with weight > 1.0")

    # Shuffle and split.
    random.seed(42)
    random.shuffle(dataset)

    n = len(dataset)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]

    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    # PHP lint validation on sample.
    php_lint_failures = validate_php_sample(dataset)
    print(f"PHP lint failures (sample): {php_lint_failures}")

    # Generate and save metadata.json.
    metadata = generate_metadata(dataset, train_set, val_set, test_set, php_lint_failures, dupes_removed)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {FINAL_DIR / 'metadata.json'}")

    # Export all formats.
    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        # OpenAI format.
        path = FINAL_DIR / f"openai_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(to_openai_format(ex)) + "\n")
        print(f"  OpenAI {split_name}: {path}")

        # Alpaca/Llama-MoE format.
        path = FINAL_DIR / f"alpaca_{split_name}.json"
        with open(path, "w") as f:
            json.dump([to_alpaca_format(ex) for ex in split_data], f, indent=2)
        print(f"  Alpaca {split_name}: {path}")

        # Raw format with metadata.
        path = FINAL_DIR / f"raw_{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(to_raw_format(ex)) + "\n")
        print(f"  Raw {split_name}: {path}")

    # Dataset stats.
    print(f"\nDataset Stats:")
    print(f"  Total: {len(dataset)}")
    print(f"  <wp_gen> examples: {gen_count}")
    print(f"  <wp_judge> examples: {judge_count}")
    print(f"  Gen ratio actual: {metadata['gen_ratio_actual']:.3f} (target: {GEN_TARGET_RATIO})")
    print(f"  With CoT: {sum(1 for ex in dataset if ex.get('metadata', {}).get('has_cot'))}")

    sources: dict[str, int] = {}
    for ex in dataset:
        src = ex.get("metadata", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"  By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")

    print(f"\nExport complete. Files in {FINAL_DIR}/")


if __name__ == "__main__":
    main()
