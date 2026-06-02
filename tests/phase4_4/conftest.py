"""Shared fixtures for Phase 4.4 REVL-03 tests.

The synthetic `response` template mirrors the REAL model/dataset format:
reasoning prose followed by a [/REASONING] CLOSE tag (NO [REASONING] open tag —
0 occurrences across all 478 cot+ctf rows) then a <judge_output>{...}</judge_output>
block. Do NOT add a paired [REASONING] open tag: the live REASONING_RE makes the
open tag optional, and a paired fixture would pass while the real close-only format
extracts nothing.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _judge_response(idx: int, with_fix: bool = False) -> str:
    """Build a close-only judge response: prose + [/REASONING] + <judge_output>."""
    prose = (
        f"WPCS Compliance: the function follows snake_case and has a docblock (sample {idx}).\n"
        "Security: input is sanitized and output escaped; nonce check present.\n"
        "Performance: a single indexed query, no N+1 pattern observed.\n"
        "WP API Usage: uses rest_ensure_response idiomatically.\n"
        "Code Quality: single responsibility, readable.\n"
        "Dependency Integrity: no unvendored third-party calls.\n"
    )
    judge = json.dumps({
        "verdict": "PASS",
        "wpcs_compliance": 9, "security": 8, "performance": 7,
        "wp_api_usage": 9, "code_quality": 8, "dependency_integrity": 8,
        "i18n": 3, "overall_score": 74,
    })
    fix = ""
    if with_fix:
        fix = ("\n<corrected_code>\n<?php\nfunction wp_example_fixed() {\n"
               "    return rest_ensure_response( array( 'ok' => true ) );\n}\n</corrected_code>")
    return f"{prose}[/REASONING]\n<judge_output>\n{judge}\n</judge_output>{fix}"


# --- session-scoped path fixtures ---

@pytest.fixture(scope="session")
def reasoning_merged_dir():
    return PROJECT_ROOT / "models" / "qwen3-30b-wp-30_70-reasoning-merged"


@pytest.fixture(scope="session")
def val_dataset():
    return PROJECT_ROOT / "data" / "reasoning_dataset" / "openai_val.jsonl"


@pytest.fixture(scope="session")
def captured_path():
    return (PROJECT_ROOT / "output" / "eval_reasoning" / "reasoning_merged"
            / "captured_responses.jsonl")


# --- function-scoped tmp fixtures (captured-response schema) ---

@pytest.fixture
def tmp_pairs_jsonl(tmp_path):
    """20 captured-response rows in the close-only format, no <corrected_code>."""
    p = tmp_path / "captured_responses.jsonl"
    with open(p, "w") as fh:
        for i in range(20):
            task_type = "cot" if i % 2 == 0 else "ctf"
            row = {
                "example_idx": i,
                "task_type": task_type,
                "prompt": f"<wp_judge> Evaluate this WordPress code (sample {i}):\n```php\n<?php function f_{i}() {{}}\n```",
                "response": _judge_response(i),
                "model_scores": {f"d{d}": (8 if d % 2 else 7) for d in range(1, 10)},
            }
            fh.write(json.dumps(row) + "\n")
    return p


@pytest.fixture
def tmp_pairs_jsonl_with_fix(tmp_path):
    """5 captured-response rows whose response also carries a <corrected_code> block."""
    p = tmp_path / "captured_responses_with_fix.jsonl"
    with open(p, "w") as fh:
        for i in range(5):
            row = {
                "example_idx": i,
                "task_type": "ctf",
                "prompt": f"<wp_judge> Evaluate this WordPress code (sample {i}).",
                "response": _judge_response(i, with_fix=True),
                "model_scores": None,
            }
            fh.write(json.dumps(row) + "\n")
    return p
