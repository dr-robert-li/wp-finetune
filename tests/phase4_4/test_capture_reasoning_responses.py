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


class TestModelDirPlumbing:
    """RTRN-05 (04.3-02 Task 2): --model-dir must plumb through to the vLLM boot AND
    appear (FULL resolved path, not the served-name alias) in the output provenance header.
    GPU-free: boot/serve/stop are monkeypatched to no-ops; the OpenAI client is faked."""

    def _run(self, tmp_path, monkeypatch, model_dir, served_name="wp-30_70"):
        import json as _json

        # 1-row GPU-free dataset (cot stream so it survives the include-streams filter).
        ds = tmp_path / "mini.jsonl"
        row = {"metadata": {"stream": "cot"},
               "messages": [{"role": "user", "content": "judge this"}]}
        ds.write_text(_json.dumps(row) + "\n")
        out = tmp_path / "cap.jsonl"

        boot_calls = []
        monkeypatch.setattr(cap, "boot_vllm",
                            lambda md, name, port, mem: boot_calls.append((md, name, port)))
        monkeypatch.setattr(cap, "wait_healthy", lambda port, name: None)
        monkeypatch.setattr(cap, "stop_vllm", lambda name: None)

        class _FakeMsg:
            content = ""  # empty -> model_scores None, no parse_judge_response dependency

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResp:
            choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **kw):
                # Capture the served alias the client was asked to use.
                served_seen.append(kw.get("model"))
                return _FakeResp()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        served_seen = []
        import openai
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: _FakeClient())

        argv = ["capture_reasoning_responses",
                "--dataset", str(ds), "--out", str(out),
                "--include-streams", "cot", "--model-dir", model_dir,
                "--served-name", served_name, "--min-parseable-rate", "0.0"]
        monkeypatch.setattr(sys, "argv", argv)
        rc = cap.main()
        return rc, boot_calls, served_seen, out

    def test_model_dir_reaches_boot_and_header(self, tmp_path, monkeypatch):
        import json as _json
        model_dir = "models/_staging/qwen3-30b-wp-30_70-reasoning-ckpt50-merged"
        rc, boot_calls, served_seen, out = self._run(tmp_path, monkeypatch, model_dir)

        assert rc == 0
        # (a) the resolved model-dir reached boot_vllm (NOT the served-name alias)
        assert boot_calls and boot_calls[0][0] == model_dir
        # (b) the client was asked to serve under the served-name alias, not the path
        assert served_seen and served_seen[0] == "wp-30_70"
        # (c) FIRST line of the output is a provenance header carrying the FULL resolved path
        first = out.read_text().splitlines()[0]
        hdr = _json.loads(first)
        assert hdr.get("__provenance__") == model_dir
        assert model_dir in first
        assert "ckpt50" in first  # the substring the Task-4 verify gate matches

    def test_help_lists_model_dir(self, capsys):
        import subprocess
        r = subprocess.run([sys.executable, "scripts/capture_reasoning_responses.py", "--help"],
                           capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[2]))
        assert "--model-dir" in r.stdout
