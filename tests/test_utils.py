"""Unit tests for scripts/utils.py.

Tests checkpoint persistence and JSON extraction utilities.
Batch API helpers and call_with_backoff were removed — all LLM work
now goes through scripts/claude_agent.py (Claude Code CLI).
"""
import json
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import (
    extract_json,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# extract_json tests
# ---------------------------------------------------------------------------

def test_extract_json_raw():
    """Strategy 1: raw JSON string."""
    result = extract_json('{"verdict":"PASS"}')
    assert result == {"verdict": "PASS"}


def test_extract_json_fenced_json():
    """Strategy 2: ```json fenced block."""
    result = extract_json('```json\n{"verdict":"PASS"}\n```')
    assert result == {"verdict": "PASS"}


def test_extract_json_fenced_plain():
    """Strategy 3: plain ``` fenced block."""
    result = extract_json('```\n{"verdict":"PASS"}\n```')
    assert result == {"verdict": "PASS"}


def test_extract_json_embedded():
    """Strategy 4: JSON embedded in surrounding text."""
    result = extract_json('Some text {"verdict":"PASS"} more text')
    assert result == {"verdict": "PASS"}


def test_extract_json_failure_no_json():
    """Returns None when no JSON is present."""
    result = extract_json('no json here at all')
    assert result is None


def test_extract_json_failure_empty():
    """Returns None for empty string."""
    result = extract_json('')
    assert result is None


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    """save_checkpoint then load_checkpoint returns identical state."""
    state = {"completed": ["a", "b"], "failed": ["c"], "batch_job_ids": []}
    save_checkpoint("test", state, checkpoint_dir=tmp_path)
    loaded = load_checkpoint("test", checkpoint_dir=tmp_path)
    assert loaded["completed"] == ["a", "b"]
    assert loaded["failed"] == ["c"]


def test_checkpoint_atomic(tmp_path):
    """Checkpoint file is written atomically; no .tmp file remains."""
    state = {"completed": ["x"], "failed": [], "batch_job_ids": []}
    save_checkpoint("test", state, checkpoint_dir=tmp_path)
    assert (tmp_path / "test_checkpoint.json").exists()
    assert not (tmp_path / "test_checkpoint.tmp").exists()


def test_checkpoint_missing_returns_empty(tmp_path):
    """load_checkpoint for nonexistent phase returns empty state."""
    loaded = load_checkpoint("nonexistent", checkpoint_dir=tmp_path)
    assert loaded["completed"] == []
    assert loaded["failed"] == []
