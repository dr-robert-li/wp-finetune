"""GPU-free unit tests for scripts/capture_reasoning_responses.py.

Exercises classify_task_type + REASONING_RE / extract_reasoning only. The vLLM
boot path (main()) is execution-tested separately (W4-cap-02), not here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

cap = pytest.importorskip("scripts.capture_reasoning_responses")


class TestClassifyTaskType:
    def test_cot(self):
        assert cap.classify_task_type({"metadata": {"stream": "cot"}}) == "cot"

    def test_ctf(self):
        assert cap.classify_task_type({"metadata": {"stream": "ctf"}}) == "ctf"

    def test_replay(self):
        assert cap.classify_task_type({"metadata": {"stream": "replay"}}) == "replay"


class TestExtractReasoning:
    def test_close_only_format(self):
        resp = "some reasoning prose here[/REASONING]\n<judge_output>{}</judge_output>"
        out = cap.extract_reasoning(resp)
        assert "some reasoning prose here" in out
        assert "[/REASONING]" not in out
        assert "judge_output" not in out

    def test_strips_think(self):
        resp = "<think>\nscratch\n</think>\nreal prose[/REASONING]<judge_output>{}</judge_output>"
        out = cap.extract_reasoning(resp)
        assert "scratch" not in out
        assert "real prose" in out

    def test_fallback_no_close_tag(self):
        resp = "just whole text, no tags at all"
        assert cap.extract_reasoning(resp).strip() == "just whole text, no tags at all"

    def test_optional_open_tag_regex(self):
        # REASONING_RE must make the open tag optional (close-only safe)
        assert cap.REASONING_RE.search("prose[/REASONING]") is not None


class TestModuleSurface:
    def test_exposes_symbols(self):
        assert hasattr(cap, "classify_task_type")
        assert hasattr(cap, "extract_reasoning")
        assert hasattr(cap, "REASONING_RE")
        assert hasattr(cap, "main")
