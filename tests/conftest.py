"""Shared fixtures for Phase 8 reward pipeline tests.

All four fixtures are session-scoped where possible to avoid repeated
tmp-dir creation. mock_judge_client and sample_rollout_group are function-
scoped so individual tests can mutate them safely.

No project-module imports at conftest top — all imports lazy inside fixtures
to prevent import-collection failures when reward_pipeline.py is absent
(which is true during Wave 0 test scaffolding, before Task 3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def recalib_json(tmp_path_factory):
    """Synthetic judge_recalibration.json for tests that call _load_score_offset().

    Writes a JSON file with score_offset=3.58 into a temporary directory so
    tests can pass an injectable path without touching the real artifact.
    """
    d = tmp_path_factory.mktemp("recalib")
    p = d / "judge_recalibration.json"
    p.write_text(
        json.dumps(
            {
                "score_offset": 3.58,
                "ci_95": [1.24, 6.09],
                "rank_invariant": True,
            }
        )
    )
    return p


@pytest.fixture(scope="session")
def php_fixture_dir():
    """Path to the reward integration test fixtures directory.

    The directory may not exist until 08-02 populates it; fixture returns
    the path regardless so tests can skip meaningfully.
    """
    return PROJECT_ROOT / "tests" / "fixtures" / "reward_integration_cases"


@pytest.fixture
def mock_judge_client():
    """MagicMock openai.OpenAI client that returns a parseable JSON overall_score.

    Imported lazily so conftest collects even when openai is unavailable.
    """
    from unittest.mock import MagicMock

    client = MagicMock()
    # Default: returns a response with overall_score 75
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"overall_score": 75})
    client.chat.completions.create.return_value = mock_resp
    return client


@pytest.fixture
def sample_rollout_group():
    """List of simple PHP strings for rollout-group tests."""
    return [
        "<?php\nfunction wp_hello() {\n    return 'hello';\n}\n",
        "<?php\nfunction wp_world() {\n    return 'world';\n}\n",
        "<?php\nfunction wp_greet( $name ) {\n    return 'hi ' . esc_html( $name );\n}\n",
        "<?php\nfunction wp_noop() {}\n",
    ]
