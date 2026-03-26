"""Unit tests for scripts/preflight.py — RED phase stubs.

All tests should FAIL with ImportError until scripts/preflight.py is implemented.
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.preflight import run_preflight


def _make_run(returncode=0, stdout=""):
    """Helper to create a mock subprocess.CompletedProcess."""
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    return result


def test_missing_phpcs():
    """Exits with code 1 when phpcs --version fails."""
    def side_effect(cmd, **kwargs):
        if "phpcs" in cmd:
            return _make_run(returncode=1)
        return _make_run(returncode=0)

    with patch("subprocess.run", side_effect=side_effect):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight()
    assert exc_info.value.code == 1


def test_missing_api_key():
    """Exits with code 1 when ANTHROPIC_API_KEY is not set."""
    env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            run_preflight()
    assert exc_info.value.code == 1


def test_missing_wp_standards():
    """Exits with code 1 when phpcs -i output lacks WordPress-Extra."""
    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "phpcs" in cmd_str and "-i" in cmd_str:
            return _make_run(returncode=0, stdout="Installed coding standards: PSR2")
        return _make_run(returncode=0, stdout="PHP 8.1 / phpcs 3.8")

    with patch("subprocess.run", side_effect=side_effect):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight()
    assert exc_info.value.code == 1


def test_preflight_pass(capsys):
    """Does NOT raise SystemExit when all checks pass."""
    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        if "phpcs" in cmd_str and "-i" in cmd_str:
            return _make_run(returncode=0, stdout="Installed coding standards: WordPress-Extra, WordPress")
        return _make_run(returncode=0, stdout="PHP 8.1 / phpcs 3.8")

    with patch("subprocess.run", side_effect=side_effect):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            run_preflight()  # Should not raise

    captured = capsys.readouterr()
    assert "all checks passed" in captured.out.lower()
