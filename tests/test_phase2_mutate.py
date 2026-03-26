"""Tests for phase2_mutate.py PHPCS hard-fail guard.

These tests verify:
1. _require_phpcs() exits with SystemExit when phpcs is not found
2. _require_phpcs() passes silently when phpcs is available
3. verify_mutation_detectable() does NOT silently return True on FileNotFoundError
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase2_mutate import _require_phpcs, verify_mutation_detectable


def test_phpcs_required_exits():
    """_require_phpcs exits with SystemExit when phpcs is not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError("phpcs not found")):
        with pytest.raises(SystemExit):
            _require_phpcs()


def test_phpcs_required_passes():
    """_require_phpcs passes silently when phpcs returns returncode=0."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        # Should not raise any exception
        _require_phpcs()


def test_verify_mutation_no_silent_fallback():
    """verify_mutation_detectable does NOT return True when PHPCS is unavailable.

    The original code had `except FileNotFoundError: return True` which allowed
    corrupted mutation data to silently pass through. After the fix, it must
    call sys.exit(1) instead.
    """
    bad_code = "<?php echo $var;"
    with patch("subprocess.run", side_effect=FileNotFoundError("phpcs not found")):
        with pytest.raises(SystemExit):
            verify_mutation_detectable(bad_code)
