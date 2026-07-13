"""Wave 0 test for the v4 (Qwen3.6-35B-A3B) reasoning data adapter (GEN-01).

Requires the Tinker venv (tinker_cookbook + a live tokenizer resolve for the new
base) -- run via `.venv-tinker/bin/python -m pytest tests/test_tinker_reasoning_data_v4.py -x -q`.
Not mock-only: this is an integration test against the real cookbook renderer
registry and the real reasoning dataset, matching 21-RESEARCH.md's discipline of
measuring GEN-01's format decision rather than assuming it.
"""
from __future__ import annotations

from scripts.tinker_reasoning_data_v4 import (
    BASE_MODEL,
    RENDERER_NAME,
    RENDERER_SOURCE,
    build_datasets,
    _max_tokenized_len_and_think_check,
)


def test_base_model_is_v4():
    assert BASE_MODEL == "Qwen/Qwen3.6-35B-A3B"


def test_renderer_resolved_from_registry():
    assert RENDERER_NAME is not None
    assert RENDERER_SOURCE == "registry"


def test_build_succeeds_with_nonzero_batches():
    train_ds, val_ds = build_datasets(batch_size=8)
    assert len(train_ds) > 0
    assert len(val_ds) > 0


def test_max_length_under_cap_and_no_empty_think_in_loss_target():
    train_ds, val_ds = build_datasets(batch_size=8)
    max_len, empty_think_injected = _max_tokenized_len_and_think_check(train_ds, val_ds)
    assert max_len < 64_000, f"max tokenized length {max_len} exceeds Tinker's 64K cap"
    assert empty_think_injected is False, (
        "a spurious empty <think></think> pair leaked into the loss-weighted "
        "target span (QwenLM #131) -- see tinker_reasoning_data_v4 docstring"
    )
