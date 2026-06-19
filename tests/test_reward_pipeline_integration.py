"""Integration tests for reward_pipeline.py — 50-case known-good/bad suite.

Tests exercise compute_group_rewards() end-to-end with fixture PHP files.
External services (vLLM judge endpoint) mocked where unavailable in CI.
PHPCS is mocked for general integration tests but REQUIRED (not mocked) for
the SC2 security-gate test — a silently-passing SC2 is itself a fail-open.

SC2 DESIGN NOTE
---------------
The SC2 (secure-fail-but-high-quality) test exercises the security terminal
override using a fixture file that:
  - Scores well on other rubric checks (75+ overall)
  - Triggers SEC-N20 (regex: preg_replace /e modifier) in D2_security.triggered_checks
  - Uses a DETERMINISTIC trigger (regex-based, no PHPCS required for the SEC-N20 check)
  - Result: compute_group_rewards returns scalar == 0.0 despite high non-security signals

WAVE 0 NOTES (preserved for context)
--------------------------------------
All tests WERE STUBBED with pytest.skip until 08-02 populates the fixture
directory and implements compute_group_rewards(). Un-stubbed in 08-03.
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixture directory constants
# ---------------------------------------------------------------------------

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "reward_integration_cases"
KNOWN_GOOD_DIR = FIXTURE_DIR / "known_good_php"
KNOWN_BAD_DIR = FIXTURE_DIR / "known_bad_php"
SC2_FILE = FIXTURE_DIR / "secure_fail_high_quality.php"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_phpcs_available() -> bool:
    """Return True if phpcs is available in PATH."""
    return shutil.which("phpcs") is not None


def _load_php_files(directory: Path) -> list[str]:
    """Load all PHP file contents from a directory, sorted by name."""
    return [f.read_text() for f in sorted(directory.glob("*.php"))]


# ---------------------------------------------------------------------------
# Integration test class — 50-case harness
# ---------------------------------------------------------------------------


class TestRewardPipelineIntegration:
    """End-to-end integration harness for compute_group_rewards().

    Judge is mocked so the suite is CI-runnable without a live vLLM endpoint.
    score_code() is run for real on the fixture PHP files (PHPCS + regex).
    """

    def test_known_good_group_above_floor(self):
        """Known-good PHP group: all rewards exist and breakdown fields are populated."""
        from scripts.reward_pipeline import compute_group_rewards

        if not KNOWN_GOOD_DIR.exists():
            pytest.skip(f"Fixture directory missing: {KNOWN_GOOD_DIR}")

        php_codes = _load_php_files(KNOWN_GOOD_DIR)
        assert len(php_codes) == 25, f"Expected 25 known_good files, got {len(php_codes)}"

        with patch("scripts.reward_pipeline.judge_score_single", return_value=75.0):
            results = compute_group_rewards(php_codes, MagicMock(), "test-model")

        assert len(results) == 25, "Must return one RewardResult per input"
        # All members of a clean group must have security_fail=False
        for i, r in enumerate(results):
            assert r.breakdown.security_fail is False, (
                f"known_good file {i} must not trigger security gate"
            )
        # Group should have variation (non-trivial phpcs_raw signals)
        phpcs_raws = [r.breakdown.phpcs_raw for r in results]
        assert max(phpcs_raws) > 0, "At least one known_good file should have phpcs_raw > 0"

    def test_known_bad_group_below_ceiling(self):
        """Known-bad PHP group: rewards are lower than known-good group mean.

        Compares mean composite_pre_gate (pre-security-gate, pre-normalization within group).
        Since both groups are scored independently, this compares phpcs_raw as the
        discriminating signal (verpo and judge are the same between groups for simplicity).
        """
        from scripts.reward_pipeline import compute_group_rewards

        if not KNOWN_GOOD_DIR.exists() or not KNOWN_BAD_DIR.exists():
            pytest.skip("Fixture directories missing")

        good_codes = _load_php_files(KNOWN_GOOD_DIR)
        bad_codes = _load_php_files(KNOWN_BAD_DIR)

        assert len(good_codes) == 25
        assert len(bad_codes) == 24

        with patch("scripts.reward_pipeline.judge_score_single", return_value=70.0):
            good_results = compute_group_rewards(good_codes, MagicMock(), "test-model")
            bad_results = compute_group_rewards(bad_codes, MagicMock(), "test-model")

        good_phpcs_mean = sum(r.breakdown.phpcs_raw for r in good_results) / len(good_results)
        bad_phpcs_mean = sum(r.breakdown.phpcs_raw for r in bad_results) / len(bad_results)

        # Known-good files use proper WP APIs — they should score at least as well
        # (this is a soft check: we verify good >= bad on phpcs_raw)
        assert good_phpcs_mean >= bad_phpcs_mean, (
            f"Known-good mean phpcs_raw ({good_phpcs_mean:.2f}) should be >= "
            f"known-bad mean phpcs_raw ({bad_phpcs_mean:.2f})"
        )

    def test_group_size_matches_input(self):
        """compute_group_rewards returns exactly len(php_codes) RewardResult items."""
        from scripts.reward_pipeline import compute_group_rewards

        if not KNOWN_GOOD_DIR.exists():
            pytest.skip(f"Fixture directory missing: {KNOWN_GOOD_DIR}")

        php_codes = _load_php_files(KNOWN_GOOD_DIR)

        with patch("scripts.reward_pipeline.judge_score_single", return_value=70.0):
            results = compute_group_rewards(php_codes, MagicMock(), "test-model")

        assert len(results) == len(php_codes), (
            f"Expected {len(php_codes)} results, got {len(results)}"
        )
        for i, r in enumerate(results):
            assert r.breakdown.group_size == len(php_codes), (
                f"breakdown.group_size must equal {len(php_codes)}, got {r.breakdown.group_size}"
            )

    def test_breakdown_dict_keys_present(self):
        """Every RewardResult.breakdown has required RLEV-02 fields and to_dict() works."""
        from scripts.reward_pipeline import compute_group_rewards

        if not KNOWN_GOOD_DIR.exists():
            pytest.skip(f"Fixture directory missing: {KNOWN_GOOD_DIR}")

        php_codes = _load_php_files(KNOWN_GOOD_DIR)[:4]  # Use subset for speed

        with patch("scripts.reward_pipeline.judge_score_single", return_value=70.0):
            results = compute_group_rewards(php_codes, MagicMock(), "test-model")

        required_keys = {
            "phpcs_raw", "verpo_raw", "judge_raw", "judge_offset_applied",
            "security_fail", "phpcs_norm", "verpo_norm", "judge_norm",
            "composite_pre_gate", "check_pass_rates", "check_difficulties",
            "group_size", "group_phpcs_mean", "group_phpcs_std",
            "group_judge_mean", "group_judge_std",
            "judge_parse_failure", "judge_imputed_from_group",
        }

        for i, r in enumerate(results):
            breakdown_dict = r.breakdown.to_dict()
            missing = required_keys - set(breakdown_dict.keys())
            assert not missing, (
                f"breakdown.to_dict() missing keys for result {i}: {missing}"
            )
            # Verify JSON-serializable (no numpy floats, etc.)
            import json
            json.dumps(breakdown_dict)  # raises if not serializable


# ---------------------------------------------------------------------------
# SC2 security-gate integration test
# ---------------------------------------------------------------------------


def test_sc2_security_fail_scores_zero():
    """SC2: high-quality but security-failing code -> reward_result.scalar == 0.0.

    This test uses the SC2 fixture PHP file that is INTENTIONALLY crafted to:
    - Score well on other rubric checks (phpcs_raw ~75, overall quality good)
    - Trigger a D2_security CRITICAL_FLOOR_RULE hit via SEC-N20 (preg_replace /e modifier)
      which is a DETERMINISTIC trigger (regex-method, fires without phpcs)

    With the judge mocked at 95.0 (very high), the security gate must still
    override composite to exactly 0.0 (D-08-05 / GRPO-02).

    PHPCS REQUIREMENT: this test uses REAL score_code() (not mocked).
    SEC-N20 is regex-based so does NOT require phpcs — but we still check
    phpcs is available to ensure the rubric scorer can run fully.
    If phpcs is absent, this test SKIPS LOUD (never silently passes —
    a silently-passing SC2 is itself a fail-open in the test suite).
    """
    if not _check_phpcs_available():
        pytest.skip(
            "LOUD SKIP: phpcs not available in PATH. "
            "The SC2 security-gate integration test REQUIRES phpcs to be installed. "
            "Install with: composer global require squizlabs/php_codesniffer. "
            "A silently-passing SC2 test is a fail-open; this skip is intentional (D-08-05)."
        )

    if not SC2_FILE.exists():
        pytest.skip(
            f"SC2 fixture missing: {SC2_FILE}. "
            "Create tests/fixtures/reward_integration_cases/secure_fail_high_quality.php"
        )

    from scripts.reward_pipeline import compute_group_rewards, _REWARD_SEC_TRIGGERS

    php_code = SC2_FILE.read_text()

    # Use 4-member group (all identical SC2 code) so normalization is well-defined.
    # All 4 trigger security gate -> all 4 scalars == 0.0.
    php_codes = [php_code] * 4

    with patch("scripts.reward_pipeline.judge_score_single", return_value=95.0):
        results = compute_group_rewards(php_codes, MagicMock(), "test-model")

    assert len(results) == 4

    for i, result in enumerate(results):
        assert result.scalar == 0.0, (
            f"SC2 member {i}: security failure must override reward to exactly 0.0, "
            f"got scalar={result.scalar}. Judge was 95.0 (high), should not matter."
        )
        assert result.breakdown.security_fail is True, (
            f"SC2 member {i}: breakdown.security_fail must be True"
        )
        # composite_pre_gate must be non-zero (real composite, not pre-gated to 0)
        # (all members identical -> zero-variance -> composite ~0 from MO-GRPO, but
        # that is expected; the test focuses on scalar==0 driven by security gate)
        assert result.breakdown.phpcs_raw > 0, (
            f"SC2 fixture should have non-trivial phpcs_raw; got {result.breakdown.phpcs_raw}. "
            "This confirms the fixture is high-quality (not just bad code)."
        )

    # Verify the trigger was SEC-N20 (or another deterministic trigger, never SEC-N04)
    from eval.rubric_scorer import score_code as real_score_code
    import os
    os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)  # ensure deterministic

    rubric = real_score_code(php_code)
    all_triggered = {cid for ids in rubric.triggered_checks.values() for cid in ids}
    triggered_sec = all_triggered & _REWARD_SEC_TRIGGERS
    assert triggered_sec, (
        f"SC2 fixture must trigger at least one _REWARD_SEC_TRIGGERS id "
        f"({{SEC-N01,N03,N06,N08,N19,N20}}); triggered: {all_triggered}"
    )
    assert "SEC-N04" not in triggered_sec, (
        "SC2 must trigger a DETERMINISTIC (non-llm) check, not SEC-N04. "
        "SEC-N04 is the llm-method check excluded by design (D-08)."
    )
