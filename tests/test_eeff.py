"""Unit tests for E_eff computation and JSONL output in profile_base_model.py.

Tests are GPU-free: they use mock data and direct function calls.
"""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniform_counts(n_experts: int = 128, total_tokens: int = 128000) -> dict:
    """Return uniform expert counts (each expert gets total_tokens // n_experts)."""
    count = total_tokens // n_experts
    return {i: count for i in range(n_experts)}


def _make_single_expert_counts(expert_id: int = 0, n_experts: int = 128, total: int = 10000) -> dict:
    """Return expert counts concentrated on a single expert."""
    counts = {i: 0 for i in range(n_experts)}
    counts[expert_id] = total
    return counts


def _make_k_uniform_counts(k: int = 8, n_experts: int = 128, total: int = 8000) -> dict:
    """Return counts uniform over k experts, zero elsewhere."""
    count = total // k
    return {i: count for i in range(k)}


# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------

from scripts.profile_base_model import (
    RoutingCollector,
    WP_GEN_ID,
    WP_JUDGE_ID,
    compute_eeff,
    has_downward_eeff_trend,
    write_profiling_jsonl,
    write_summary_md,
)


# ---------------------------------------------------------------------------
# Tests: compute_eeff
# ---------------------------------------------------------------------------


class TestComputeEeff:
    def test_uniform_128_experts(self):
        """Uniform distribution over 128 experts -> E_eff ~= 128."""
        counts = _make_uniform_counts(n_experts=128)
        result = compute_eeff(counts)
        assert abs(result - 128.0) < 0.5, f"Expected ~128.0, got {result}"

    def test_single_expert_all_counts(self):
        """All counts on one expert -> E_eff ~= 1.0."""
        counts = _make_single_expert_counts()
        result = compute_eeff(counts)
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0, got {result}"

    def test_eight_experts_equal(self):
        """8 experts with equal counts -> E_eff ~= 8.0."""
        counts = _make_k_uniform_counts(k=8)
        result = compute_eeff(counts)
        assert abs(result - 8.0) < 0.5, f"Expected ~8.0, got {result}"

    def test_zero_total_counts_returns_nan(self):
        """Zero total counts must return float('nan'), NOT n_experts."""
        result = compute_eeff({})
        assert math.isnan(result), f"Expected NaN for zero counts, got {result}"

    def test_zero_total_counts_math_isnan(self):
        """math.isnan() must be True for zero-count E_eff."""
        result = compute_eeff({i: 0 for i in range(128)})
        assert math.isnan(result), f"Expected NaN for all-zero counts, got {result}"


# ---------------------------------------------------------------------------
# Tests: RoutingCollector.set_token_types / tag_token_types
# ---------------------------------------------------------------------------

import torch


class TestTagTokenTypes:
    """Tests for token-type tagging logic via RoutingCollector.set_token_types()."""

    def _collector(self):
        return RoutingCollector(n_layers=4, n_experts=8, top_k=2, pad_token_id=151643)

    def test_wp_gen_tag(self):
        """Tokens after <wp_gen> are tagged 'wp_gen'."""
        collector = self._collector()
        ids = torch.tensor([100, WP_GEN_ID, 200, 300])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert types[2] == "wp_gen"
        assert types[3] == "wp_gen"

    def test_wp_judge_tag(self):
        """Tokens after <wp_judge> are tagged 'wp_judge'."""
        collector = self._collector()
        ids = torch.tensor([100, WP_JUDGE_ID, 200, 300])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert types[2] == "wp_judge"
        assert types[3] == "wp_judge"

    def test_no_task_tokens_all_other(self):
        """With no task tokens, all positions return 'other'."""
        collector = self._collector()
        ids = torch.tensor([100, 200, 300])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert all(t == "other" for t in types)

    def test_padding_tokens_tagged_pad(self):
        """Padding tokens (pad_token_id) are tagged 'pad' regardless of context."""
        collector = self._collector()
        pad_id = 151643
        # After wp_gen, padding should still be 'pad' not 'wp_gen'
        ids = torch.tensor([WP_GEN_ID, 200, pad_id, 300])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert types[2] == "pad", f"Expected 'pad', got '{types[2]}'"

    def test_truncated_sequence_missing_task_token_is_other(self):
        """Truncated sequence without task token returns 'other' (not crash)."""
        collector = self._collector()
        # Simulate truncated sequence - just regular tokens, no task token
        ids = torch.tensor([500, 600, 700, 800])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert all(t == "other" for t in types), f"Expected all 'other', got {types}"

    def test_padding_after_task_token_still_pad(self):
        """Padding tokens after task token are tagged 'pad', not the task type."""
        collector = self._collector()
        pad_id = 151643
        ids = torch.tensor([WP_GEN_ID, 200, pad_id, pad_id])
        collector.set_token_types(ids)
        types = collector._current_token_types
        assert types[2] == "pad"
        assert types[3] == "pad"


# ---------------------------------------------------------------------------
# Tests: JSONL output schema
# ---------------------------------------------------------------------------


class TestJsonlSchema:
    def _make_collector_with_data(self):
        """Create a RoutingCollector populated with simple test data."""
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        # Manually set some counts
        collector._counts_total = [
            {0: 100, 1: 80, 2: 60},  # layer 0
            {0: 50, 1: 50},           # layer 1
        ]
        collector._counts_wp_gen = [
            {0: 60, 1: 40},
            {0: 30},
        ]
        collector._counts_wp_judge = [
            {0: 40, 1: 40},
            {0: 20, 1: 50},
        ]
        collector._n_tokens_total = [200, 100]
        collector._n_tokens_wp_gen = [100, 30]
        collector._n_tokens_wp_judge = [80, 70]
        return collector

    def test_required_fields_present(self):
        """JSONL output record has all required fields."""
        collector = self._make_collector_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(collector, ratio="30_70", subsample_n=100, out_path=str(out_path))
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            required = [
                "ratio", "layer_idx", "n_tokens_total", "n_tokens_wp_gen", "n_tokens_wp_judge",
                "expert_counts_total", "expert_counts_wp_gen", "expert_counts_wp_judge",
                "eeff_total", "eeff_wp_gen", "eeff_wp_judge", "subsample_n", "model"
            ]
            for rec in records:
                for field in required:
                    assert field in rec, f"Field '{field}' missing from JSONL record: {rec.keys()}"

    def test_model_field_is_base(self):
        """JSONL 'model' field must be 'base'."""
        collector = self._make_collector_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(collector, ratio="30_70", subsample_n=100, out_path=str(out_path))
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            for rec in records:
                assert rec["model"] == "base"

    def test_zero_wp_gen_tokens_eeff_is_null(self):
        """When wp_gen token count is 0 for a layer, eeff_wp_gen is null in JSON."""
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        # Layer 0 has no wp_gen tokens
        collector._counts_total = [{0: 100, 1: 80}, {0: 50}]
        collector._counts_wp_gen = [{}, {}]  # no wp_gen tokens
        collector._counts_wp_judge = [{0: 40}, {0: 30}]
        collector._n_tokens_total = [200, 50]
        collector._n_tokens_wp_gen = [0, 0]
        collector._n_tokens_wp_judge = [80, 30]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(collector, ratio="30_70", subsample_n=50, out_path=str(out_path))
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            # NaN serialized as null in JSON
            assert records[0]["eeff_wp_gen"] is None, (
                f"Expected null for zero wp_gen tokens, got {records[0]['eeff_wp_gen']}"
            )


# ---------------------------------------------------------------------------
# Tests: has_downward_eeff_trend
# ---------------------------------------------------------------------------


class TestDownwardEeffTrend:
    def test_decreasing_trend_returns_true(self):
        """E_eff decreasing as gen% increases -> True."""
        # 30/70=90, 40/60=80, 50/50=70: downward trend
        eeffs = [90.0, 80.0, 70.0, 60.0, 50.0]
        assert has_downward_eeff_trend(eeffs) is True

    def test_flat_returns_false(self):
        """E_eff flat (no decrease) -> False."""
        eeffs = [70.0, 70.0, 70.0, 70.0, 70.0]
        assert has_downward_eeff_trend(eeffs) is False

    def test_increasing_returns_false(self):
        """E_eff increasing -> False."""
        eeffs = [50.0, 60.0, 70.0, 80.0, 90.0]
        assert has_downward_eeff_trend(eeffs) is False

    def test_nan_values_skipped(self):
        """NaN values should be skipped in comparison."""
        # non-NaN pairs show a downward trend
        eeffs = [90.0, float("nan"), 70.0, float("nan"), 50.0]
        assert has_downward_eeff_trend(eeffs) is True

    def test_all_nan_returns_false(self):
        """All NaN values -> no comparison possible -> False."""
        eeffs = [float("nan"), float("nan"), float("nan")]
        assert has_downward_eeff_trend(eeffs) is False


# ---------------------------------------------------------------------------
# Tests: RoutingCollector.reset()
# ---------------------------------------------------------------------------


class TestRoutingCollectorReset:
    def test_reset_clears_all_counts(self):
        """reset() clears all accumulated counts."""
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        # Manually add some data
        collector._counts_total[0][0] = 100
        collector._n_tokens_total[0] = 200
        collector._current_token_types = ["wp_gen", "other"]
        collector.reset()
        assert collector._n_tokens_total[0] == 0
        assert len(collector._counts_total[0]) == 0
        assert collector._current_token_types == []


# ---------------------------------------------------------------------------
# Tests: write_summary_md NaN-safe aggregation
# ---------------------------------------------------------------------------


class TestWriteSummaryMd:
    def test_excludes_nan_from_aggregation(self):
        """write_summary_md excludes NaN values from mean/max/variance."""
        # Provide all_ratio_eeffs where some values are NaN
        all_ratio_eeffs = {
            "30_70": {
                "eeff_total": [40.0, float("nan"), 50.0],
                "eeff_wp_gen": [35.0, float("nan"), 45.0],
                "eeff_wp_judge": [float("nan"), float("nan"), float("nan")],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.md"
            write_summary_md(all_ratio_eeffs, str(out_path))
            content = out_path.read_text()
        # Should produce valid output without crashing
        assert "30_70" in content
        # All-NaN column should display "N/A"
        assert "N/A" in content


# ---------------------------------------------------------------------------
# Tests: RoutingCollector.make_hook padding exclusion
# ---------------------------------------------------------------------------


class TestMakeHookPaddingExclusion:
    def test_make_hook_ignores_padding_positions(self):
        """make_hook should NOT count routing for padding token positions."""
        collector = RoutingCollector(n_layers=1, n_experts=8, top_k=2, pad_token_id=151643)
        # Set token types: positions [wp_gen, wp_gen, pad, pad]
        collector._current_token_types = ["wp_gen", "wp_gen", "pad", "pad"]

        # Create fake hook outputs: router_indices shape [4, 2] (4 tokens, top_k=2)
        router_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        # Simulate hook call: (module, inputs, outputs)
        # outputs = (router_logits, router_scores, router_indices)
        mock_outputs = (None, None, router_indices)
        hook_fn = collector.make_hook(layer_idx=0)
        hook_fn(None, None, mock_outputs)

        # Experts 4, 5, 6, 7 (from padding positions) must NOT be in counts
        counts = collector._counts_total[0]
        for expert_id in [4, 5, 6, 7]:
            assert expert_id not in counts or counts.get(expert_id, 0) == 0, (
                f"Expert {expert_id} from padding position should not be counted"
            )
        # Experts 0, 1, 2, 3 (from wp_gen positions) should be counted
        assert counts.get(0, 0) > 0 or counts.get(1, 0) > 0, "wp_gen tokens should be counted"
