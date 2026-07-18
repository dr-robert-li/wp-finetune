"""Wave 0 tests for check_token_alignment.py — run before implementation.

All tests use mocks/fixtures; no GPU or model download required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.check_token_alignment import (
    align_and_check,
    build_receipt,
    classify_stopped_naturally,
)


# ---------------------------------------------------------------------------
# align_and_check
# ---------------------------------------------------------------------------


class TestAlignAndCheck:
    """align_and_check mutates model_config in place to match tokenizer."""

    def test_mismatched_pair_gets_aligned(self):
        """The documented real-world mismatch: config eos=248044, tokenizer eos=248046, config pad=None."""
        config = MagicMock()
        config.eos_token_id = 248044
        config.pad_token_id = None
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 248046
        tokenizer.pad_token_id = 248044

        align_and_check(config, tokenizer)

        assert config.eos_token_id == 248046
        assert config.pad_token_id == 248044
        assert config.pad_token_id is not None

    def test_mismatched_pair_pad_falls_back_to_eos_when_tokenizer_pad_none(self):
        """When tokenizer.pad_token_id is also None, fall back to tokenizer.eos_token_id."""
        config = MagicMock()
        config.eos_token_id = 248044
        config.pad_token_id = None
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 248046
        tokenizer.pad_token_id = None

        align_and_check(config, tokenizer)

        assert config.eos_token_id == 248046
        assert config.pad_token_id == 248046  # fell back to tokenizer.eos_token_id
        assert config.pad_token_id is not None

    def test_already_aligned_pair_is_a_noop(self):
        """Given an already-matched pair, align_and_check still passes (no-op change)."""
        config = MagicMock()
        config.eos_token_id = 248046
        config.pad_token_id = 248044
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 248046
        tokenizer.pad_token_id = 248044

        align_and_check(config, tokenizer)

        assert config.eos_token_id == 248046
        assert config.pad_token_id == 248044

    def test_postfix_invariant_always_holds(self):
        """Post-fix: config.eos_token_id == tokenizer.eos_token_id and pad is not None, for any input."""
        config = MagicMock()
        config.eos_token_id = 1
        config.pad_token_id = None
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = None

        align_and_check(config, tokenizer)

        assert config.eos_token_id == tokenizer.eos_token_id
        assert config.pad_token_id is not None


# ---------------------------------------------------------------------------
# classify_stopped_naturally
# ---------------------------------------------------------------------------


class TestClassifyStoppedNaturally:
    """Distinguishes a natural stop (ends on eos, strictly shorter than budget) from run-to-length."""

    def test_run_to_length_is_not_natural(self):
        """Output length == max_tokens budget => False (ran to length, did not stop)."""
        output_ids = [1, 2, 3, 4]
        assert classify_stopped_naturally(output_ids, max_tokens=4, eos_token_ids=[4]) is False

    def test_short_output_ending_on_eos_is_natural(self):
        """Output strictly shorter than budget and ends on an eos id => True."""
        output_ids = [1, 2, 248046]
        assert classify_stopped_naturally(output_ids, max_tokens=64, eos_token_ids=[248046, 248044]) is True

    def test_short_output_not_ending_on_eos_is_not_natural(self):
        """Output shorter than budget but does NOT end on eos => False (truncated some other way)."""
        output_ids = [1, 2, 3]
        assert classify_stopped_naturally(output_ids, max_tokens=64, eos_token_ids=[248046]) is False

    def test_single_eos_id_accepted_not_just_list(self):
        """eos_token_ids may be a bare int, not only a list."""
        output_ids = [1, 248046]
        assert classify_stopped_naturally(output_ids, max_tokens=64, eos_token_ids=248046) is True

    def test_empty_output_is_not_natural(self):
        assert classify_stopped_naturally([], max_tokens=64, eos_token_ids=[248046]) is False


# ---------------------------------------------------------------------------
# build_receipt
# ---------------------------------------------------------------------------


class TestBuildReceipt:
    """The receipt-builder emits a flat dict with the required gate fields."""

    def test_receipt_has_required_fields(self):
        receipt = build_receipt(
            status="pass",
            orig_eos_id=248044,
            aligned_eos_id=248046,
            orig_pad_id=None,
            aligned_pad_id=248044,
            tokenizer_eos_id=248046,
            stopped_naturally=True,
        )
        for field in (
            "status",
            "orig_eos_id",
            "aligned_eos_id",
            "orig_pad_id",
            "aligned_pad_id",
            "tokenizer_eos_id",
            "stopped_naturally",
        ):
            assert field in receipt, f"missing required field: {field}"

        assert receipt["status"] == "pass"
        assert receipt["stopped_naturally"] is True

    def test_receipt_accepts_extra_fields(self):
        receipt = build_receipt(
            status="fail",
            orig_eos_id=248044,
            aligned_eos_id=248046,
            orig_pad_id=None,
            aligned_pad_id=248044,
            tokenizer_eos_id=248046,
            stopped_naturally=False,
            stop_gen_len=64,
            max_tokens_budget=64,
        )
        assert receipt["stop_gen_len"] == 64
        assert receipt["max_tokens_budget"] == 64
