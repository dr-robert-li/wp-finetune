"""Unit tests for scripts._sieve_vllm_patch.sitecustomize._resolve_moe_block_class
(GATE4-02 T-22-02, SIEVE-04 router-mask patch).

Tests are GPU-free and vLLM-free: fake stub modules stand in for the real
vllm.model_executor.models.* modules, injected via sys.modules so the resolver's
importlib.import_module() calls succeed without vLLM installed (vLLM only lives
in the GPU serving container -- not importable on this host).
"""
from __future__ import annotations

import sys
import types

import pytest

from scripts._sieve_vllm_patch.sitecustomize import (
    _MOE_BLOCK_CANDIDATES,
    _resolve_moe_block_class,
)


class DummySparseMoeBlock:
    pass


def test_picks_first_present_class_by_exact_name(monkeypatch):
    """A missing module is skipped; the next candidate with an exact class-name
    match is selected."""
    fake_mod = types.ModuleType("fake_vllm_qwen3_5_moe")
    fake_mod.Qwen3_5MoeSparseMoeBlock = DummySparseMoeBlock
    monkeypatch.setitem(sys.modules, "fake_vllm_qwen3_5_moe", fake_mod)

    candidates = [
        ("fake_vllm_missing_module", "SomeClass"),
        ("fake_vllm_qwen3_5_moe", "Qwen3_5MoeSparseMoeBlock"),
        ("fake_vllm_unused_fallback", "Qwen3MoeSparseMoeBlock"),
    ]
    resolved = _resolve_moe_block_class(candidates)
    assert resolved is DummySparseMoeBlock


def test_tolerates_unknown_class_name_via_single_sparse_moe_block_scan(monkeypatch):
    """If the exact class_name doesn't match but the module exposes exactly one
    *SparseMoeBlock class, the resolver falls back to that scanned class."""
    class ActualClassName:
        pass

    fake_mod = types.ModuleType("fake_vllm_qwen3_next")
    fake_mod.SomeOtherSparseMoeBlock = ActualClassName
    monkeypatch.setitem(sys.modules, "fake_vllm_qwen3_next", fake_mod)

    candidates = [("fake_vllm_qwen3_next", "Qwen3NextSparseMoeBlock")]
    resolved = _resolve_moe_block_class(candidates)
    assert resolved is ActualClassName


def test_raises_when_no_candidate_resolves():
    candidates = [
        ("fake_vllm_missing_a", "A"),
        ("fake_vllm_missing_b", "B"),
    ]
    with pytest.raises(RuntimeError, match="no MoE-block class resolved"):
        _resolve_moe_block_class(candidates)


def test_ambiguous_scan_does_not_resolve(monkeypatch):
    """Two *SparseMoeBlock classes in one module is ambiguous -- do NOT guess,
    fall through to the next candidate (or raise if exhausted)."""
    class FirstSparseMoeBlock:
        pass

    class SecondSparseMoeBlock:
        pass

    fake_mod = types.ModuleType("fake_vllm_ambiguous")
    fake_mod.FirstSparseMoeBlock = FirstSparseMoeBlock
    fake_mod.SecondSparseMoeBlock = SecondSparseMoeBlock
    monkeypatch.setitem(sys.modules, "fake_vllm_ambiguous", fake_mod)

    candidates = [("fake_vllm_ambiguous", "NoSuchExactName")]
    with pytest.raises(RuntimeError):
        _resolve_moe_block_class(candidates)


def test_default_candidate_ordering_is_v4_first():
    """qwen3_5_moe/qwen3_next candidates are ordered before the qwen3_moe (v3)
    fallback."""
    modules_in_order = [module_path for module_path, _ in _MOE_BLOCK_CANDIDATES]
    qwen3_moe_idx = modules_in_order.index("vllm.model_executor.models.qwen3_moe")
    for v4_module in (
        "vllm.model_executor.models.qwen3_5_moe",
        "vllm.model_executor.models.qwen3_next",
    ):
        assert modules_in_order.index(v4_module) < qwen3_moe_idx
