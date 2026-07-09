"""Unit tests for scripts.aimer_prune (PRUNE-01: AIMER weight-norm expert scorer).

Tests are GPU-free: tiny synthetic on-disk safetensors fixture (2 layers x 3
experts, 2 shards), exercising the real index.json -> safe_open resolution
path with no full-model load. Module-level importorskip so this file SKIPs
cleanly while scripts/aimer_prune.py is absent, mirroring
tests/test_sieve_ksweep_mask.py's convention.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

aimer_prune = pytest.importorskip("scripts.aimer_prune")

N_LAYERS = 2
N_EXPERTS = 3
PROJS = ("gate_proj", "up_proj", "down_proj")


def _write_fixture_checkpoint(ckpt_dir, scale: float = 1.0, seed: int = 0):
    """Write a tiny 2-shard checkpoint (one shard per layer) with synthetic expert weights."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for layer in range(N_LAYERS):
        shard = f"model-{layer:05d}.safetensors"
        for expert in range(N_EXPERTS):
            for proj in PROJS:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                w = (torch.randn(4, 5, generator=gen) * scale).float()
                weight_map[key] = shard
                shard_tensors.setdefault(shard, {})[key] = w

    for shard, tensors in shard_tensors.items():
        save_file(tensors, ckpt_dir / shard)
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return ckpt_dir


def test_shape_and_finite(tmp_path):
    ckpt = _write_fixture_checkpoint(tmp_path / "ckpt")
    scores = aimer_prune.compute_aimer_scores(ckpt, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    assert scores.shape == (N_LAYERS, N_EXPERTS)
    assert np.isfinite(scores).all()


def test_scale_invariance(tmp_path):
    ckpt_a = _write_fixture_checkpoint(tmp_path / "a", scale=1.0, seed=42)
    ckpt_b = _write_fixture_checkpoint(tmp_path / "b", scale=7.0, seed=42)
    scores_a = aimer_prune.compute_aimer_scores(ckpt_a, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    scores_b = aimer_prune.compute_aimer_scores(ckpt_b, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    np.testing.assert_allclose(scores_a, scores_b, rtol=1e-4)


def test_determinism(tmp_path):
    ckpt = _write_fixture_checkpoint(tmp_path / "ckpt", seed=7)
    scores_1 = aimer_prune.compute_aimer_scores(ckpt, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    scores_2 = aimer_prune.compute_aimer_scores(ckpt, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    np.testing.assert_array_equal(scores_1, scores_2)


def test_bounded_in_unit_interval(tmp_path):
    ckpt = _write_fixture_checkpoint(tmp_path / "ckpt", seed=3)
    scores = aimer_prune.compute_aimer_scores(ckpt, n_layers=N_LAYERS, n_experts=N_EXPERTS)
    n = 4 * 5 * len(PROJS)
    lower = 1.0 / np.sqrt(n)
    assert (scores >= lower - 1e-6).all()
    assert (scores <= 1.0 + 1e-6).all()


def test_missing_key_raises(tmp_path):
    """T-13-01: a missing/renamed expert key must raise, never silently produce a zero score."""
    ckpt = _write_fixture_checkpoint(tmp_path / "ckpt")
    idx_path = ckpt / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    del idx["weight_map"]["model.layers.0.mlp.experts.0.down_proj.weight"]
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(KeyError):
        aimer_prune.compute_aimer_scores(ckpt, n_layers=N_LAYERS, n_experts=N_EXPERTS)
