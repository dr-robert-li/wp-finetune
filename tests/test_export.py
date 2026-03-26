"""Tests for export_dataset.py — ratio enforcement, metadata, validation, dedup, sample_weight."""

import hashlib
import sys
from pathlib import Path

import pytest

# Ensure project root is on path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_dataset import (
    add_sample_weight,
    add_task_token,
    deduplicate,
    enforce_ratio,
    generate_metadata,
    infer_task_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen_example(idx: int) -> dict:
    """Create a minimal gen-type training example."""
    return {
        "messages": [
            {"role": "system", "content": "You are a WordPress expert."},
            {"role": "user", "content": f"Write a WordPress function #{idx}."},
            {"role": "assistant", "content": f"<?php function example_{idx}() {{ return {idx}; }}"},
        ],
        "metadata": {"source": "phase1_real", "task_type": "gen", "training_tags": []},
    }


def _make_judge_example(idx: int) -> dict:
    """Create a minimal judge-type training example."""
    return {
        "messages": [
            {"role": "system", "content": "You are a WordPress expert."},
            {"role": "user", "content": f"Review this WordPress code #{idx}."},
            {"role": "assistant", "content": f'{{"overall_score": 8, "verdict": "PASS"}}'},
        ],
        "metadata": {"source": "phase2_judge", "task_type": "judge", "training_tags": []},
    }


# ---------------------------------------------------------------------------
# Task 2.1: enforce_ratio — standard 100 gen + 200 judge -> ~40/60
# ---------------------------------------------------------------------------

def test_gen_judge_ratio():
    """100 gen + 200 judge examples should yield ~40/60 split."""
    dataset = [_make_gen_example(i) for i in range(100)]
    dataset += [_make_judge_example(i) for i in range(200)]

    result = enforce_ratio(dataset)

    gen_count = sum(1 for ex in result if infer_task_type(ex) == "gen")
    judge_count = sum(1 for ex in result if infer_task_type(ex) == "judge")
    total = len(result)
    assert total > 0
    gen_ratio = gen_count / total
    judge_ratio = judge_count / total

    assert abs(gen_ratio - 0.40) < 0.05, f"Gen ratio {gen_ratio:.3f} not close to 0.40"
    assert abs(judge_ratio - 0.60) < 0.05, f"Judge ratio {judge_ratio:.3f} not close to 0.60"


# ---------------------------------------------------------------------------
# Task 2.2: enforce_ratio — gen-limited case (20 gen + 200 judge -> 20 gen + 30 judge)
# ---------------------------------------------------------------------------

def test_ratio_gen_limited():
    """When gen count is the bottleneck, judge should be capped at gen * (60/40)."""
    dataset = [_make_gen_example(i) for i in range(20)]
    dataset += [_make_judge_example(i) for i in range(200)]

    result = enforce_ratio(dataset)

    gen_count = sum(1 for ex in result if infer_task_type(ex) == "gen")
    judge_count = sum(1 for ex in result if infer_task_type(ex) == "judge")

    assert gen_count == 20, f"Expected 20 gen, got {gen_count}"
    assert judge_count == 30, f"Expected 30 judge (20 * 60/40), got {judge_count}"


# ---------------------------------------------------------------------------
# Task 2.3: generate_metadata — required keys present
# ---------------------------------------------------------------------------

def test_metadata_fields():
    """generate_metadata must return all required keys."""
    dataset = [_make_gen_example(i) for i in range(10)]
    dataset += [_make_judge_example(i) for i in range(15)]
    train_set = dataset[:20]
    val_set = dataset[20:22]
    test_set = dataset[22:]

    metadata = generate_metadata(dataset, train_set, val_set, test_set, php_lint_failures=0, dupes_removed=0)

    required_keys = [
        "total_examples",
        "gen_judge_ratio",
        "gen_ratio_actual",
        "judge_ratio_actual",
        "rejection_examples",
        "taxonomy_coverage",
        "train_val_test_counts",
        "php_lint_failures",
    ]
    for key in required_keys:
        assert key in metadata, f"Required key '{key}' missing from metadata"

    assert metadata["total_examples"] == len(dataset)
    assert isinstance(metadata["train_val_test_counts"], dict)
    assert "train" in metadata["train_val_test_counts"]
    assert "val" in metadata["train_val_test_counts"]
    assert "test" in metadata["train_val_test_counts"]


# ---------------------------------------------------------------------------
# Task 2.4: add_task_token — every user message gets <wp_gen> or <wp_judge>
# ---------------------------------------------------------------------------

def test_task_tokens_present():
    """After add_task_token, every user message must contain <wp_gen> or <wp_judge>."""
    dataset = [_make_gen_example(i) for i in range(5)]
    dataset += [_make_judge_example(i) for i in range(5)]

    for example in dataset:
        tokenized = add_task_token(example)
        user_messages = [m for m in tokenized["messages"] if m["role"] == "user"]
        assert len(user_messages) > 0
        for msg in user_messages:
            has_token = "<wp_gen>" in msg["content"] or "<wp_judge>" in msg["content"]
            assert has_token, f"User message missing task token: {msg['content'][:80]}"


# ---------------------------------------------------------------------------
# Task 2.5: deduplicate — 2 identical examples -> 1 unique, 1 dupe
# ---------------------------------------------------------------------------

def test_dedup_detection():
    """Two identical examples should result in 1 unique and 1 dupe detected."""
    ex = _make_gen_example(1)
    dataset = [ex, dict(ex)]  # Two copies of the same example.

    unique, dupes = deduplicate(dataset)

    assert len(unique) == 1, f"Expected 1 unique example, got {len(unique)}"
    assert dupes == 1, f"Expected 1 duplicate detected, got {dupes}"


# ---------------------------------------------------------------------------
# Task 2.6: add_sample_weight — contrastive (mutated) examples get weight 1.5
# ---------------------------------------------------------------------------

def test_sample_weight_contrastive():
    """Examples with metadata.source='mutated' should get sample_weight = 1.5."""
    example = _make_gen_example(1)
    example["metadata"]["source"] = "mutated"

    result = add_sample_weight(example)

    assert result["metadata"]["sample_weight"] == 1.5, (
        f"Expected sample_weight=1.5 for mutated source, got {result['metadata']['sample_weight']}"
    )


# ---------------------------------------------------------------------------
# Task 2.7: add_sample_weight — normal examples get weight 1.0
# ---------------------------------------------------------------------------

def test_sample_weight_normal():
    """Examples with metadata.source='extracted' should get sample_weight = 1.0."""
    example = _make_gen_example(1)
    example["metadata"]["source"] = "extracted"

    result = add_sample_weight(example)

    assert result["metadata"]["sample_weight"] == 1.0, (
        f"Expected sample_weight=1.0 for extracted source, got {result['metadata']['sample_weight']}"
    )
