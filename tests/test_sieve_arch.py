"""Unit tests for scripts.sieve_arch (GATE4-02 arch-awareness helper).

Tests are GPU-free: synthetic dict-based fake configs and stub model/tokenizer
objects, no model load, no torch import required at collection time.
"""
from __future__ import annotations

import pytest

from scripts.sieve_arch import (
    ATTENTION_STRATUM,
    DELTANET_STRATUM,
    arch_dims,
    infer_dims_from_records,
    layer_strata,
    resolve_moe_layers,
    resolve_task_token_ids,
)


# ---------------------------------------------------------------------------
# arch_dims
# ---------------------------------------------------------------------------


class TestArchDims:
    def test_v4_composite_text_config(self):
        config = {"text_config": {"num_hidden_layers": 40, "num_experts": 256}}
        assert arch_dims(config) == (40, 256)

    def test_v3_plain_config(self):
        config = {"num_hidden_layers": 48, "num_experts": 128}
        assert arch_dims(config) == (48, 128)


# ---------------------------------------------------------------------------
# layer_strata
# ---------------------------------------------------------------------------


class TestLayerStrata:
    def test_real_v4_layer_types_pattern(self):
        """40-entry v4 layer_types: 10x(3 linear_attention + 1 full_attention)."""
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 10
        config = {"text_config": {"num_hidden_layers": 40, "layer_types": layer_types}}
        strata = layer_strata(config)
        assert len(strata) == 40
        assert strata.count(ATTENTION_STRATUM) == 10
        assert strata.count(DELTANET_STRATUM) == 30
        attn_idx = [i for i, s in enumerate(strata) if s == ATTENTION_STRATUM]
        assert attn_idx == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    def test_full_attention_interval_fallback(self):
        config = {"num_hidden_layers": 8, "full_attention_interval": 4}
        strata = layer_strata(config)
        assert strata == [
            DELTANET_STRATUM, DELTANET_STRATUM, DELTANET_STRATUM, ATTENTION_STRATUM,
            DELTANET_STRATUM, DELTANET_STRATUM, DELTANET_STRATUM, ATTENTION_STRATUM,
        ]

    def test_v3_uniform_fallback_never_raises(self):
        config = {"num_hidden_layers": 48, "num_experts": 128}
        strata = layer_strata(config)
        assert strata == [DELTANET_STRATUM] * 48


# ---------------------------------------------------------------------------
# resolve_moe_layers
# ---------------------------------------------------------------------------


class _StubMLP:
    def __init__(self):
        self.gate = object()


class _StubLayer:
    def __init__(self):
        self.mlp = _StubMLP()


class _StubInner:
    def __init__(self, n):
        self.layers = [_StubLayer() for _ in range(n)]


class TestResolveMoeLayers:
    def test_flat_root_resolves(self):
        class StubModel:
            def __init__(self, n):
                self.model = _StubInner(n)

        result = resolve_moe_layers(StubModel(40))
        assert len(result) == 40
        assert [idx for idx, _ in result] == list(range(40))

    def test_nested_language_model_root_resolves(self):
        class StubLanguageModelWrapper:
            def __init__(self, n):
                self.language_model = _StubInner(n)

        class StubModel:
            def __init__(self, n):
                self.model = StubLanguageModelWrapper(n)

        result = resolve_moe_layers(StubModel(40))
        assert len(result) == 40

    def test_raises_on_empty_tree(self):
        class EmptyModel:
            pass

        with pytest.raises(RuntimeError):
            resolve_moe_layers(EmptyModel())

    def test_unwraps_peft_model(self):
        class StubModel:
            def __init__(self, n):
                self.model = _StubInner(n)

        class StubPeftModel:
            def __init__(self, n):
                self._base = StubModel(n)

            def get_base_model(self):
                return self._base

        result = resolve_moe_layers(StubPeftModel(40))
        assert len(result) == 40


# ---------------------------------------------------------------------------
# infer_dims_from_records
# ---------------------------------------------------------------------------


class TestInferDimsFromRecords:
    def test_tiny_two_record_fixture(self):
        records = [
            {"layer_idx": 0, "expert_counts_total": {"0": 5, "255": 2}},
            {"layer_idx": 39, "expert_counts_total": {"1": 3}},
        ]
        assert infer_dims_from_records(records) == (40, 256)

    def test_empty_records(self):
        assert infer_dims_from_records([]) == (0, 0)


# ---------------------------------------------------------------------------
# resolve_task_token_ids
# ---------------------------------------------------------------------------


class TestResolveTaskTokenIds:
    def test_returns_none_none_when_vocab_lacks_task_tokens(self):
        class StubTok:
            def get_vocab(self):
                return {"a": 0, "b": 1}

        assert resolve_task_token_ids(StubTok(), 151669, 151670) == (None, None)

    def test_returns_defaults_when_both_task_tokens_present(self):
        class StubTok:
            def get_vocab(self):
                return {"<wp_gen>": 151669, "<wp_judge>": 151670}

        assert resolve_task_token_ids(StubTok(), 151669, 151670) == (151669, 151670)
