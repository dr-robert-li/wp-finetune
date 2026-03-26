"""Tests for config file correctness — Wave 0 scaffolds for DATA-01/DATA-02."""
import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_SYSTEM_PATH = PROJECT_ROOT / "config" / "judge_system.md"
SYNTHETIC_PROMPTS_PATH = PROJECT_ROOT / "config" / "synthetic_prompts.yaml"


def test_judge_threshold_v2():
    """Judge threshold must be >= 8 (not >= 7)."""
    text = JUDGE_SYSTEM_PATH.read_text()
    # Must contain ">= 8" as the threshold
    assert ">= 8" in text, "judge_system.md must require ALL dimensions >= 8"
    # The old ">= 7" threshold line must be gone
    # Allow ">= 7" ONLY inside quoted/example contexts, not as the actual threshold line
    lines = text.splitlines()
    threshold_lines = [
        ln for ln in lines
        if ">= 7" in ln and "threshold" not in ln.lower() and "pass" in ln.lower()
    ]
    assert len(threshold_lines) == 0, (
        f"Found '>= 7' in a PASS requirement line: {threshold_lines}"
    )


def test_security_auto_fail():
    """Judge config must have security dimension auto-FAIL rule."""
    text = JUDGE_SYSTEM_PATH.read_text()
    assert "SECURITY AUTO-FAIL" in text, (
        "judge_system.md must contain 'SECURITY AUTO-FAIL' rule"
    )
    assert "< 5" in text, (
        "judge_system.md must specify '< 5' as the security auto-FAIL threshold"
    )


def test_na_scoring_deflated():
    """N/A dimensions must score 7 (not 10) to prevent inflation."""
    text = JUDGE_SYSTEM_PATH.read_text()
    # Must NOT have "Score N/A (10)" anywhere
    count_ten = text.count("Score N/A (10)")
    assert count_ten == 0, (
        f"Found {count_ten} occurrence(s) of 'Score N/A (10)' — must be 0"
    )
    # Must have exactly 2 "Score N/A (7)" entries (i18n + accessibility)
    count_seven = text.count("Score N/A (7)")
    assert count_seven >= 2, (
        f"Expected >= 2 occurrences of 'Score N/A (7)', got {count_seven}"
    )


def test_rejection_templates_exist():
    """synthetic_prompts.yaml must have rejection_templates with all 3 sub-keys."""
    data = yaml.safe_load(SYNTHETIC_PROMPTS_PATH.read_text())
    assert "rejection_templates" in data, (
        "synthetic_prompts.yaml must have 'rejection_templates' key"
    )
    rt = data["rejection_templates"]
    required_keys = ["proactive_nonce", "proactive_capability", "proactive_escaping"]
    for key in required_keys:
        assert key in rt, f"rejection_templates must have sub-key '{key}'"
        assert len(rt[key]) >= 1, f"rejection_templates.{key} must have >= 1 template"
