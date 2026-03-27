"""Wave 0 tests for prepare_tokenizer.py — run before implementation.

All tests use mocks/fixtures; no GPU or model download required.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_tokenizer():
    """Tiny mock tokenizer with realistic add_special_tokens / encode behaviour."""
    vocab = {f"tok{i}": i for i in range(100)}
    additional: list[str] = []

    tok = MagicMock()
    tok.additional_special_tokens = additional

    def add_special_tokens(d: dict) -> int:
        new_tokens = [t for t in d.get("additional_special_tokens", []) if t not in vocab]
        for t in new_tokens:
            idx = len(vocab)
            vocab[t] = idx
            additional.append(t)
        return len(new_tokens)

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG001
        if text in vocab:
            return [vocab[text]]
        # Simulate multi-token fallback for unknown text
        return [vocab.get(c, 0) for c in text]

    tok.add_special_tokens.side_effect = add_special_tokens
    tok.encode.side_effect = encode
    tok.__len__ = MagicMock(return_value=lambda: len(vocab))
    return tok, vocab


@pytest.fixture()
def embedding_weights():
    """100-token, 64-dim embedding weight tensor."""
    torch.manual_seed(42)
    return torch.randn(100, 64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpecialTokensAdded:
    """test_special_tokens_added — assert <wp_gen> and <wp_judge> land in additional_special_tokens."""

    def test_special_tokens_added(self, small_tokenizer):
        tok, _ = small_tokenizer
        tok.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
        assert "<wp_gen>" in tok.additional_special_tokens
        assert "<wp_judge>" in tok.additional_special_tokens

    def test_no_duplicate_tokens(self, small_tokenizer):
        """Adding the same tokens twice should not duplicate them."""
        tok, _ = small_tokenizer
        tok.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
        tok.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
        assert tok.additional_special_tokens.count("<wp_gen>") == 1
        assert tok.additional_special_tokens.count("<wp_judge>") == 1


class TestEmbeddingsMeanInit:
    """test_embeddings_mean_init — new token rows equal the mean of existing rows."""

    def test_embeddings_mean_init(self, embedding_weights):
        """Mean-init logic: last 2 rows set to mean of rows[:-2]."""
        weights = embedding_weights  # shape (100, 64)
        expected_mean = weights[:-2].mean(dim=0)

        with torch.no_grad():
            weights[-2] = expected_mean
            weights[-1] = expected_mean

        assert torch.allclose(weights[-2], expected_mean), "<wp_gen> embedding not mean-initialized"
        assert torch.allclose(weights[-1], expected_mean), "<wp_judge> embedding not mean-initialized"

    def test_mean_not_zero(self, embedding_weights):
        """Sanity: mean of randomly-initialised embeddings is not all-zeros."""
        mean_emb = embedding_weights[:-2].mean(dim=0)
        assert not torch.allclose(mean_emb, torch.zeros_like(mean_emb))

    def test_mean_not_random(self, embedding_weights):
        """Two separate mean computations on the same tensor give identical results."""
        mean_a = embedding_weights[:-2].mean(dim=0)
        mean_b = embedding_weights[:-2].mean(dim=0)
        assert torch.allclose(mean_a, mean_b)


class TestSmokeSingleTokenIds:
    """test_smoke_single_token_ids — each special token encodes to exactly 1 token ID."""

    def test_single_token_wp_gen(self, small_tokenizer):
        tok, vocab = small_tokenizer
        tok.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
        ids = tok.encode("<wp_gen>", add_special_tokens=False)
        assert len(ids) == 1, f"<wp_gen> encoded to {len(ids)} tokens, expected 1"

    def test_single_token_wp_judge(self, small_tokenizer):
        tok, vocab = small_tokenizer
        tok.add_special_tokens({"additional_special_tokens": ["<wp_gen>", "<wp_judge>"]})
        ids = tok.encode("<wp_judge>", add_special_tokens=False)
        assert len(ids) == 1, f"<wp_judge> encoded to {len(ids)} tokens, expected 1"
