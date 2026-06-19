"""Integration tests for reward_pipeline.py — 50-case known-good/bad suite.

Tests exercise compute_group_rewards() end-to-end with fixture PHP files.
External services (vLLM judge endpoint, PHPCS) mocked where unavailable in CI.

WAVE 0 NOTES
------------
All tests are STUBBED with pytest.skip until 08-02 populates the fixture
directory and implements compute_group_rewards().  The test shape and fixture
constants are present so 08-02 can un-skip without restructuring.

The SC2 (secure-fail-but-high-quality) test IS the priority integration gate:
a PHP file that passes quality rubric checks but triggers D2_security
CRITICAL_FLOOR_RULE → reward must be exactly 0.0 regardless of other signals.
"""
from __future__ import annotations

import sys
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
# Integration test class — 50-case harness
# ---------------------------------------------------------------------------


class TestRewardPipelineIntegration:
    """End-to-end integration harness for compute_group_rewards().

    Populated in 08-02/08-03.  Tests here serve as the structural contract so
    downstream plan tasks know exactly what to un-skip.
    """

    def test_known_good_group_above_floor(self):
        """Known-good PHP group: all rewards > 0 and composite is positive."""
        pytest.skip("implemented in 08-02")

    def test_known_bad_group_below_ceiling(self):
        """Known-bad PHP group: rewards are lower than known-good group mean."""
        pytest.skip("implemented in 08-02")

    def test_group_size_matches_input(self):
        """compute_group_rewards returns exactly len(php_codes) RewardResult items."""
        pytest.skip("implemented in 08-02")

    def test_breakdown_dict_keys_present(self):
        """Every RewardResult.breakdown has required RLEV-02 fields."""
        pytest.skip("implemented in 08-02")


# ---------------------------------------------------------------------------
# SC2 security-gate integration test
# ---------------------------------------------------------------------------


def test_sc2_security_fail_scores_zero():
    """SC2: high-quality but security-failing code -> reward_result.scalar == 0.0.

    This test uses the SC2 fixture PHP file that is INTENTIONALLY crafted to:
    - Score well on PHPCS / WP-standards (high quality signals)
    - Trigger a D2_security CRITICAL_FLOOR_RULE hit (a genuine security failure)

    With the judge mocked at 95.0 (very high), the security gate must still
    override composite to exactly 0.0 (D-08-05).
    """
    pytest.skip("implemented in 08-03 (requires SC2 fixture + compute_group_rewards)")
    # When un-skipped the test body will look like:
    #
    #   from scripts.reward_pipeline import compute_group_rewards
    #   php_code = SC2_FILE.read_text()
    #   with patch("scripts.reward_pipeline.judge_score_single", return_value=95.0):
    #       results = compute_group_rewards([php_code] * 4, MagicMock(), "test-model")
    #   assert results[0].scalar == 0.0, "SC2 security failure must override to 0.0"
    #   assert results[0].breakdown.security_fail is True
