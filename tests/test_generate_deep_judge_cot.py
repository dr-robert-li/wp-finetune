"""Wave 0 test stubs for scripts/generate_deep_judge_cot.py.

Tests FAIL with ImportError until the production script exists (RED phase).
Once scripts/generate_deep_judge_cot.py is created in Plan 01 Task 2,
these tests will pass for all non-API functions.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_deep_judge_cot import (
    load_seeds,
    sample_seeds,
    passes_quality_gate,
    format_seed_as_exemplar,
    format_training_example,
    validate_pilot_batch,
    verify_citation_accuracy,
    REQUIRED_DIMENSIONS,
    CITATION_HALLUCINATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_good_result():
    """Return a complete valid deep_judge_cot result with all 9 dimensions."""
    return {
        "verdict": "PASS",
        "overall_score": 85,
        "key_observation": "Code follows WordPress coding standards well.",
        "dimension_analysis": {
            d: {"score": 8, "analysis": f"Uses esc_html correctly for {d}."}
            for d in REQUIRED_DIMENSIONS
        },
    }


# ---------------------------------------------------------------------------
# Task 1: load_seeds / sample_seeds
# ---------------------------------------------------------------------------


def test_load_seeds_returns_cot_only():
    """load_seeds() must return only seed_type == 'deep_judge_cot' entries, count >= 59."""
    seeds = load_seeds()
    assert all(s["seed_type"] == "deep_judge_cot" for s in seeds), (
        "load_seeds() returned non-CoT seeds"
    )
    assert len(seeds) >= 59, f"Expected >= 59 CoT seeds, got {len(seeds)}"


def test_sample_seeds_boundary_weighting():
    """sample_seeds(seeds, n) produces boundary seeds more often than uniform random.

    Over 200 samples of n=3 from the full seed pool, boundary seeds should appear
    more than 10% of sampled slots (expected ~20-30% due to 2x weight).
    """
    seeds = load_seeds()
    if not seeds:
        pytest.skip("No seeds available")

    total_samples = 0
    boundary_count = 0
    for _ in range(200):
        batch = sample_seeds(seeds, n=3)
        total_samples += len(batch)
        boundary_count += sum(1 for s in batch if s.get("defect_subtlety") == "boundary")

    boundary_fraction = boundary_count / total_samples if total_samples > 0 else 0
    assert boundary_fraction > 0.10, (
        f"Boundary weighting too low: {boundary_fraction:.2%} (expected > 10%)"
    )


# ---------------------------------------------------------------------------
# Task 2–3: passes_quality_gate
# ---------------------------------------------------------------------------


def test_passes_quality_gate_valid():
    """A complete result with all 9 dimensions and valid scores passes the gate."""
    result = _make_good_result()
    assert passes_quality_gate(result) is True


def test_passes_quality_gate_missing_dimensions():
    """A result with only 5 dimensions fails the quality gate."""
    result = {
        "verdict": "PASS",
        "overall_score": 85,
        "key_observation": "ok",
        "dimension_analysis": {
            d: {"score": 8, "analysis": "ok"}
            for d in REQUIRED_DIMENSIONS[:5]  # Only 5 of 9
        },
    }
    assert passes_quality_gate(result) is False


def test_passes_quality_gate_bad_score_range():
    """A result with a score of 15 (out of 1-10 range) fails the quality gate."""
    result = _make_good_result()
    result["dimension_analysis"]["security"]["score"] = 15
    assert passes_quality_gate(result) is False


# ---------------------------------------------------------------------------
# Task 4: validate_pilot_batch
# ---------------------------------------------------------------------------


def test_pilot_dimension_coverage():
    """validate_pilot_batch() reports 0 missing dimensions for a complete pilot batch."""
    pilot_examples = [
        {
            "reasoning": {
                "verdict": "PASS",
                "overall_score": 80,
                "key_observation": "ok",
                "dimension_analysis": {
                    d: {"score": 7, "analysis": f"Uses esc_html for {d}."}
                    for d in REQUIRED_DIMENSIONS
                },
            },
            "dimensions_addressed": list(REQUIRED_DIMENSIONS),
            "citation_accuracy": {"total_citations": 1, "grounded_citations": 1, "hallucinated_citations": [], "hallucination_ratio": 0.0},
        }
        for _ in range(20)
    ]
    report = validate_pilot_batch(pilot_examples)
    assert report["missing_dimensions"] == [], (
        f"Expected 0 missing dimensions, got {report['missing_dimensions']}"
    )


def test_pilot_api_citation_check():
    """validate_pilot_batch() finds >= 3 distinct WP API citations across pilot examples."""
    api_names = ["esc_html", "wp_verify_nonce", "esc_attr", "current_user_can"]
    pilot_examples = []
    for i, api in enumerate(api_names):
        pilot_examples.append({
            "reasoning": {
                "verdict": "PASS",
                "overall_score": 80,
                "key_observation": "ok",
                "dimension_analysis": {
                    d: {"score": 7, "analysis": f"Uses {api} correctly."}
                    for d in REQUIRED_DIMENSIONS
                },
            },
            "dimensions_addressed": list(REQUIRED_DIMENSIONS),
            "citation_accuracy": {"total_citations": 1, "grounded_citations": 1, "hallucinated_citations": [], "hallucination_ratio": 0.0},
        })
    report = validate_pilot_batch(pilot_examples)
    assert report["distinct_api_citations"] >= 3, (
        f"Expected >= 3 API citations, got {report['distinct_api_citations']}"
    )


# ---------------------------------------------------------------------------
# Task 5: format_training_example schema
# ---------------------------------------------------------------------------


def test_format_training_example_schema():
    """format_training_example() output dict has all required keys."""
    source_info = {
        "source_file": "wp-content/plugins/test/functions.php",
        "source_dir": "passed",
        "function_name": "my_function",
        "code": "<?php function my_function() { return esc_html($x); } ?>",
    }
    result = _make_good_result()
    ca = {"total_citations": 1, "grounded_citations": 1, "hallucinated_citations": [], "hallucination_ratio": 0.0}
    example = format_training_example(source_info, result, citation_accuracy=ca)
    required_keys = [
        "source_file", "source_dir", "function_name", "code",
        "reasoning", "dimensions_addressed", "generation_method", "citation_accuracy",
    ]
    for key in required_keys:
        assert key in example, f"Missing required key: {key}"


# ---------------------------------------------------------------------------
# Task 6–7: Citation accuracy (addresses review concern #1)
# ---------------------------------------------------------------------------


def test_citation_accuracy_rejects_hallucinated_apis():
    """verify_citation_accuracy() flags APIs cited in analysis but absent from source code.

    Specifically: if analysis text mentions '$wpdb->prepare' but the source code
    does NOT contain '$wpdb->prepare', verify_citation_accuracy() must return at
    least one hallucinated citation.

    Furthermore, passes_quality_gate() must return False when hallucination_ratio
    exceeds CITATION_HALLUCINATION_THRESHOLD.
    """
    # Source code has no $wpdb->prepare
    source_code = "<?php function foo() { global $wpdb; $wpdb->get_results('SELECT * FROM wp_posts'); } ?>"
    # Analysis falsely claims $wpdb->prepare is used
    result = {
        "verdict": "FAIL",
        "overall_score": 40,
        "key_observation": "Unsafe query",
        "dimension_analysis": {
            "sql_safety": {
                "score": 2,
                "analysis": "Uses $wpdb->prepare and check_ajax_referer correctly — actually no, it doesn't prepare the query at all.",
            },
            **{
                d: {"score": 7, "analysis": "ok"}
                for d in REQUIRED_DIMENSIONS
                if d != "sql_safety"
            },
        },
    }
    # Contradiction: analysis says "$wpdb->prepare" is used but it's in the analysis text
    # as a positive citation while not appearing in source. Build a cleaner hallucination:
    result_hallucinating = {
        "verdict": "PASS",
        "overall_score": 85,
        "key_observation": "Secure code",
        "dimension_analysis": {
            "security": {
                "score": 9,
                "analysis": "Uses wp_verify_nonce and check_ajax_referer for CSRF protection",
            },
            **{
                d: {"score": 8, "analysis": "ok"}
                for d in REQUIRED_DIMENSIONS
                if d != "security"
            },
        },
    }
    source_no_apis = "<?php function foo() { return 1; } ?>"
    ca = verify_citation_accuracy(result_hallucinating, source_no_apis)
    assert len(ca["hallucinated_citations"]) > 0, (
        f"Expected hallucinated citations, got {ca}"
    )
    assert ca["hallucination_ratio"] > 0, (
        f"Expected nonzero hallucination ratio, got {ca['hallucination_ratio']}"
    )
    # When hallucination_ratio > threshold, passes_quality_gate must return False
    if ca["hallucination_ratio"] > CITATION_HALLUCINATION_THRESHOLD:
        assert passes_quality_gate(result_hallucinating, source_code=source_no_apis) is False


def test_citation_accuracy_accepts_grounded_apis():
    """verify_citation_accuracy() returns 0 hallucinated citations when API is in source.

    Analysis mentions 'esc_html' and source code DOES contain 'esc_html'.
    """
    source_code = "<?php echo esc_html($var); ?>"
    result = {
        "verdict": "PASS",
        "overall_score": 85,
        "key_observation": "Secure output escaping",
        "dimension_analysis": {
            "wp_api_usage": {
                "score": 9,
                "analysis": "Correctly uses esc_html for output escaping.",
            },
            **{
                d: {"score": 8, "analysis": "ok"}
                for d in REQUIRED_DIMENSIONS
                if d != "wp_api_usage"
            },
        },
    }
    ca = verify_citation_accuracy(result, source_code)
    assert ca["hallucination_ratio"] == 0.0, (
        f"Expected 0.0 hallucination ratio for grounded citation, got {ca}"
    )
    assert ca["hallucinated_citations"] == [], (
        f"Expected no hallucinated citations, got {ca['hallucinated_citations']}"
    )
