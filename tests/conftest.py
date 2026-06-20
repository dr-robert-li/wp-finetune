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


@pytest.fixture
def mock_tinker_client():
    """Mock Tinker training client for GSPO/GRPO unit tests (GRPO-05/06/07/08).

    Mocks BOTH forward_backward (GRPO token-level fallback) AND forward_backward_custom
    (GSPO sequence-level default path — D-09-03 locked). Both return the same
    ForwardBackwardOutput mock so tests can assert which path was selected.

    MoE routing keys are set to below-threshold values (e_frac_with_tokens:mean=0.6)
    so routing-autohalt tests can exercise the halt branch without needing live infra.
    """
    from unittest.mock import MagicMock

    tc = MagicMock()
    fb_out = MagicMock()
    fb_out.metrics = {
        "e_frac_with_tokens:mean": 0.6,
        "e_max_violation:mean": 0.002,
        "e_max_violation:max": 0.008,
    }
    fb_out.training_logprobs = []
    # GRPO token-level fallback path (--grpo-fallback / --no-gspo flag)
    tc.forward_backward.return_value = fb_out
    # GSPO sequence-level default path via forward_backward_custom (D-09-03).
    # Explicit mock required — auto-mock returns a new MagicMock, not fb_out,
    # so .metrics / .training_logprobs would be inaccessible.
    tc.forward_backward_custom.return_value = fb_out
    tc.optim_step.return_value = None
    tc.save_weights_for_sampler.return_value.path = "/fake/sampler"
    return tc
