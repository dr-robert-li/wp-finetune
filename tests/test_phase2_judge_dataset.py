"""Tests for phase2_judge_dataset.py — Claude Code agent integration.

These tests verify that phase2_judge_dataset.py uses Claude Code agents
(via claude_agent.py) instead of direct Anthropic API calls.
"""
import sys
import inspect
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.phase2_judge_dataset as jd


def test_uses_claude_agent():
    """generate_json from claude_agent must be used for LLM calls."""
    source = inspect.getsource(jd)
    assert "generate_json" in source, (
        "phase2_judge_dataset.py must use generate_json from claude_agent "
        "for LLM-based scoring"
    )


def test_no_anthropic_import():
    """anthropic must NOT be imported — all LLM work via Claude Code agents."""
    source = inspect.getsource(jd)
    assert "import anthropic" not in source, (
        "phase2_judge_dataset.py must not import anthropic — use claude_agent instead"
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
    """time.sleep(REQUEST_INTERVAL) pattern must be absent."""
    source = inspect.getsource(jd)
    assert "REQUEST_INTERVAL" not in source, (
        "REQUEST_INTERVAL constant must be removed"
    )
    assert "time.sleep(REQUEST_INTERVAL)" not in source, (
        "time.sleep(REQUEST_INTERVAL) must be removed"
    )
