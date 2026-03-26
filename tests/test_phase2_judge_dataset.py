"""Tests for phase2_judge_dataset.py — rate limiting and utils.py integration.

These tests verify that phase2_judge_dataset.py uses the correct hardened
patterns from utils.py instead of brittle direct API calls and time.sleep.

Behavior-level checks (import inspection) that fail fast if the module
reverts to the old pattern.
"""
import sys
import inspect
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.phase2_judge_dataset as jd


def test_rate_limiting_uses_backoff():
    """call_with_backoff must be imported (not direct client.messages.create + sleep)."""
    from scripts.utils import call_with_backoff
    # The module should reference call_with_backoff in its source
    source = inspect.getsource(jd)
    assert "call_with_backoff" in source, (
        "phase2_judge_dataset.py must use call_with_backoff for rate limiting "
        "(PIPE-03 fix from CONCERNS.md)"
    )


def test_uses_extract_json():
    """extract_json must be imported for robust JSON parsing."""
    source = inspect.getsource(jd)
    assert "extract_json" in source, (
        "phase2_judge_dataset.py must use extract_json instead of brittle split-based parsing"
    )


def test_has_checkpoint():
    """load_checkpoint and save_checkpoint must be imported for resume support."""
    source = inspect.getsource(jd)
    assert "load_checkpoint" in source, "phase2_judge_dataset.py must import load_checkpoint"
    assert "save_checkpoint" in source, "phase2_judge_dataset.py must import save_checkpoint"


def test_no_time_sleep_request_interval_pattern():
    """time.sleep(REQUEST_INTERVAL) pattern must be absent — replaced by call_with_backoff."""
    source = inspect.getsource(jd)
    assert "REQUEST_INTERVAL" not in source, (
        "REQUEST_INTERVAL constant must be removed — call_with_backoff handles rate limiting"
    )
    assert "time.sleep(REQUEST_INTERVAL)" not in source, (
        "time.sleep(REQUEST_INTERVAL) must be removed (PIPE-03 fix)"
    )
