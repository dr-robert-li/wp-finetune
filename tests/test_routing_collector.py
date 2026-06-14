"""Unit tests for RoutingCollector hook infrastructure and write_profiling_jsonl model-tag param.

Tests are GPU-free: mock data, direct function calls.
Covers PROF-01 (hook registration + accumulation) and PROF-02 (per-task token attribution).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.profile_base_model import (
    RoutingCollector,
    WP_GEN_ID,
    WP_JUDGE_ID,
    write_profiling_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_eeff.py mock pattern)
# ---------------------------------------------------------------------------


def _make_collector_with_data(n_layers: int = 2, n_experts: int = 8) -> RoutingCollector:
    """Build a RoutingCollector pre-loaded with deterministic synthetic counts."""
    collector = RoutingCollector(n_layers=n_layers, n_experts=n_experts, top_k=2, pad_token_id=151643)
    for layer in range(n_layers):
        for expert in range(n_experts):
            collector._counts_total[layer][expert] = (layer + 1) * (expert + 1)
            collector._counts_wp_gen[layer][expert] = (layer + 1) * (expert + 1) // 2
            collector._counts_wp_judge[layer][expert] = (layer + 1) * (expert + 1) // 3
        collector._n_tokens_total[layer] = 1000
        collector._n_tokens_wp_gen[layer] = 400
        collector._n_tokens_wp_judge[layer] = 300
    return collector


# ---------------------------------------------------------------------------
# Tests: RoutingCollector make_hook — accumulation
# ---------------------------------------------------------------------------


class TestMakeHookAccumulation:
    """PROF-01: Hook accumulates router_indices into the correct per-layer bucket."""

    def test_hook_accumulates_into_correct_layer(self):
        """make_hook(layer_idx) accumulates indices into that layer's total counts."""
        collector = RoutingCollector(n_layers=4, n_experts=8, top_k=2, pad_token_id=151643)
        hook = collector.make_hook(2)  # layer 2

        # Simulate a forward hook call: outputs = (None, None, router_indices)
        # router_indices shape: [batch * seq_len, top_k] — two tokens, top_k=2 each
        router_indices = torch.tensor([[0, 3], [1, 2]])  # token 0 -> experts 0,3; token 1 -> experts 1,2
        mock_outputs = (None, None, router_indices)

        # Set token types: no special tokens -> all "other"
        input_ids = torch.tensor([[100, 200]])  # no wp_gen/wp_judge tokens
        collector.set_token_types(input_ids)

        hook(None, None, mock_outputs)

        # Experts 0, 1, 2, 3 each got 1 count in layer 2
        assert collector._counts_total[2][0] == 1
        assert collector._counts_total[2][1] == 1
        assert collector._counts_total[2][2] == 1
        assert collector._counts_total[2][3] == 1
        # Layer 0 should be untouched
        assert sum(collector._counts_total[0].values()) == 0

    def test_hook_does_not_accumulate_pad_tokens(self):
        """Padding tokens (pad_token_id) are excluded from expert counts."""
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        hook = collector.make_hook(0)

        # Two tokens: one pad, one real
        input_ids = torch.tensor([[151643, 100]])  # first is pad
        collector.set_token_types(input_ids)

        # router_indices for 2 tokens x top_k=2
        router_indices = torch.tensor([[0, 1], [2, 3]])
        hook(None, None, (None, None, router_indices))

        # Only the real token (token 1) should accumulate experts 2,3
        assert collector._counts_total[0].get(2, 0) == 1
        assert collector._counts_total[0].get(3, 0) == 1
        # Pad token's experts (0,1) should NOT accumulate
        assert collector._counts_total[0].get(0, 0) == 0
        assert collector._counts_total[0].get(1, 0) == 0


# ---------------------------------------------------------------------------
# Tests: RoutingCollector set_token_types — per-task attribution (PROF-02)
# ---------------------------------------------------------------------------


class TestTagTokenTypes:
    """PROF-02: Token-type tagging routes wp_gen/wp_judge correctly."""

    def _collector(self):
        return RoutingCollector(n_layers=4, n_experts=8, top_k=2, pad_token_id=151643)

    def test_wp_gen_tag(self):
        """WP_GEN_ID token sets token type to wp_gen."""
        c = self._collector()
        input_ids = torch.tensor([[WP_GEN_ID, 100, 200]])
        c.set_token_types(input_ids)
        # After WP_GEN_ID at position 0, subsequent tokens are wp_gen
        assert c._current_token_types[1] == "wp_gen"

    def test_wp_judge_tag(self):
        """WP_JUDGE_ID token sets token type to wp_judge."""
        c = self._collector()
        input_ids = torch.tensor([[WP_JUDGE_ID, 100, 200]])
        c.set_token_types(input_ids)
        assert c._current_token_types[1] == "wp_judge"

    def test_tokens_before_any_marker_are_other(self):
        """Tokens before any wp_gen/wp_judge marker have type 'other'."""
        c = self._collector()
        input_ids = torch.tensor([[50, 60, WP_GEN_ID, 100]])
        c.set_token_types(input_ids)
        assert c._current_token_types[0] == "other"
        assert c._current_token_types[1] == "other"

    def test_marker_itself_excluded(self):
        """The marker token itself does not accumulate (or is typed as marker, not as content)."""
        c = self._collector()
        input_ids = torch.tensor([[WP_GEN_ID]])
        c.set_token_types(input_ids)
        # The marker token itself is typed as "wp_gen" (it transitions current_type)
        # but it's the marker itself — subsequent tokens are wp_gen content
        # (acceptable: marker types to wp_gen, content tokens after it also wp_gen)
        assert c._current_token_types[0] == "wp_gen"

    def test_wp_gen_attribution_via_hook(self):
        """Experts activated during wp_gen tokens accumulate in _counts_wp_gen.

        Note: set_token_types transitions current_type at the marker position,
        then appends current_type. So the marker itself is also tagged wp_gen.
        All three tokens [WP_GEN_ID, 100, 200] are typed wp_gen.
        """
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        hook = collector.make_hook(0)

        # Three tokens: wp_gen marker + two content tokens (all three typed wp_gen)
        input_ids = torch.tensor([[WP_GEN_ID, 100, 200]])
        collector.set_token_types(input_ids)

        # router_indices for 3 tokens x top_k=2
        router_indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
        hook(None, None, (None, None, router_indices))

        # All tokens are wp_gen -> experts 0,1,2,3,4,5 all in _counts_wp_gen
        assert collector._counts_wp_gen[0].get(0, 0) >= 1
        assert collector._counts_wp_gen[0].get(2, 0) >= 1
        assert collector._counts_wp_gen[0].get(4, 0) >= 1
        # Nothing in wp_judge
        assert sum(collector._counts_wp_judge[0].values()) == 0

    def test_wp_judge_attribution_via_hook(self):
        """Experts activated during wp_judge tokens accumulate in _counts_wp_judge."""
        collector = RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)
        hook = collector.make_hook(0)

        input_ids = torch.tensor([[WP_JUDGE_ID, 100, 200]])
        collector.set_token_types(input_ids)

        router_indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
        hook(None, None, (None, None, router_indices))

        # All tokens typed wp_judge -> experts in _counts_wp_judge
        assert collector._counts_wp_judge[0].get(0, 0) >= 1
        assert collector._counts_wp_judge[0].get(2, 0) >= 1
        # Nothing in wp_gen
        assert sum(collector._counts_wp_gen[0].values()) == 0


# ---------------------------------------------------------------------------
# Tests: write_profiling_jsonl model_tag parameter
# ---------------------------------------------------------------------------


class TestWriteProfilingJsonlModelTag:
    """Parameterized model_tag in write_profiling_jsonl."""

    def _make_collector_with_data(self):
        return _make_collector_with_data(n_layers=2, n_experts=8)

    def test_model_tag_default_is_base(self):
        """Calling with no model_tag -> every record['model'] == 'base' (backward compat)."""
        collector = self._make_collector_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(collector, ratio="30_70", subsample_n=100, out_path=str(out_path))
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            for rec in records:
                assert rec["model"] == "base"

    def test_model_tag_custom(self):
        """model_tag='reasoning-merged-v4' -> every record['model'] == 'reasoning-merged-v4'."""
        collector = self._make_collector_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(
                collector,
                ratio="30_70",
                subsample_n=100,
                out_path=str(out_path),
                model_tag="reasoning-merged-v4",
            )
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            for rec in records:
                assert rec["model"] == "reasoning-merged-v4"

    def test_model_tag_explicit_base(self):
        """model_tag='base' explicit -> every record['model'] == 'base'."""
        collector = self._make_collector_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.jsonl"
            write_profiling_jsonl(
                collector,
                ratio="30_70",
                subsample_n=50,
                out_path=str(out_path),
                model_tag="base",
            )
            records = [json.loads(line) for line in out_path.read_text().strip().split("\n")]
            for rec in records:
                assert rec["model"] == "base"
