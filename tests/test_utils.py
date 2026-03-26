"""Unit tests for scripts/utils.py — RED phase stubs.

All tests should FAIL with ImportError until scripts/utils.py is implemented.
"""
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import (
    extract_json,
    call_with_backoff,
    load_checkpoint,
    save_checkpoint,
    batch_or_direct,
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


# ---------------------------------------------------------------------------
# call_with_backoff tests
# ---------------------------------------------------------------------------

def test_backoff_retries():
    """Retries on RateLimitError up to max_retries, then raises."""
    import anthropic

    mock_client = MagicMock()
    rate_limit_error = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    success_response = MagicMock()
    mock_client.messages.create.side_effect = [
        rate_limit_error,
        rate_limit_error,
        rate_limit_error,
        success_response,
    ]

    with patch("time.sleep"):
        result = call_with_backoff(
            mock_client,
            max_retries=5,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert mock_client.messages.create.call_count == 4
    assert result is success_response


def test_backoff_retry_after():
    """Uses retry_after attribute from RateLimitError when present."""
    import anthropic

    mock_client = MagicMock()

    # Create error with retry_after attribute
    error = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    error.retry_after = 2.5

    success_response = MagicMock()
    mock_client.messages.create.side_effect = [error, success_response]

    sleep_calls = []
    with patch("time.sleep", side_effect=lambda t: sleep_calls.append(t)):
        call_with_backoff(
            mock_client,
            max_retries=3,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert len(sleep_calls) >= 1
    # Should sleep for at least retry_after (2.5) and at most retry_after + 10% jitter
    assert 2.5 <= sleep_calls[0] <= 2.75


# ---------------------------------------------------------------------------
# batch_or_direct tests
# ---------------------------------------------------------------------------

def test_routing_threshold():
    """batch_or_direct threshold is at 50 items."""
    assert batch_or_direct(0) == "direct"
    assert batch_or_direct(49) == "direct"
    assert batch_or_direct(50) == "batch"
    assert batch_or_direct(51) == "batch"
