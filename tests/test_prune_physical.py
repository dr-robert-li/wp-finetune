"""Unit tests for scripts.prune_apply_physical (PRUNE-06: physical expert-removal
surgery).

Tests are GPU-free: a tiny synthetic on-disk checkpoint (2 layers, 8 experts,
small tensors), no 60GB model ever loaded. Module-level importorskip so this
file SKIPs cleanly while scripts/prune_apply_physical.py is absent, mirroring
tests/test_aimer_prune.py's convention.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

prune_apply_physical = pytest.importorskip("scripts.prune_apply_physical")

N_LAYERS = 2
N_EXPERTS = 8
HIDDEN = 6
MOE_DIM = 4
K = 5


def _write_fixture_checkpoint(ckpt_dir, seed=0):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    weight_map = {}
    shard = "model.safetensors"
    for layer in range(N_LAYERS):
        for expert in range(N_EXPERTS):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"] = torch.randn(
                MOE_DIM, HIDDEN, generator=gen
            )
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = torch.randn(
                MOE_DIM, HIDDEN, generator=gen
            )
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = torch.randn(
                HIDDEN, MOE_DIM, generator=gen
            )
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(
            N_EXPERTS, HIDDEN, generator=gen
        )
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.randn(HIDDEN, generator=gen)

    for key in tensors:
        weight_map[key] = shard
    save_file(tensors, ckpt_dir / shard)
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (ckpt_dir / "config.json").write_text(
        json.dumps({"num_local_experts": N_EXPERTS, "num_hidden_layers": N_LAYERS})
    )
    return tensors


def _protected_mask():
    protected = np.zeros((N_LAYERS, N_EXPERTS), dtype=bool)
    protected[1, 7] = True  # coldest-scored expert in layer 1, protected -> must survive
    return protected


def _scores():
    # Deterministic scores: expert index itself (higher index = higher score),
    # except we want expert 7 (the protected one) to be the LOWEST-scored so
    # keeping it proves the protected-union logic, not just top-k luck.
    scores = np.zeros((N_LAYERS, N_EXPERTS))
    for layer in range(N_LAYERS):
        scores[layer] = np.arange(N_EXPERTS, dtype=float)
    scores[:, 7] = -1.0  # lowest score everywhere
    return scores


def test_uniform_mask_exactly_k_per_layer():
    keep_mask = prune_apply_physical.build_uniform_keep_mask(_scores(), _protected_mask(), K)
    assert (keep_mask.sum(axis=1) == K).all()
    assert keep_mask[1, 7]  # protected expert always kept despite lowest score


def test_infeasible_k_raises():
    protected = _protected_mask()
    protected[1, :6] = True  # 6 protected experts in layer 1 now (0..5 plus already-set 7)
    with pytest.raises(ValueError):
        prune_apply_physical.build_uniform_keep_mask(_scores(), protected, k=5)


def test_apply_physical_shapes_and_renumbering(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"
    _write_fixture_checkpoint(ckpt_dir)

    keep_mask = prune_apply_physical.build_uniform_keep_mask(_scores(), _protected_mask(), K)
    result = prune_apply_physical.apply_physical(ckpt_dir, keep_mask, out_dir)
    assert result["k"] == K

    config = json.loads((out_dir / "config.json").read_text())
    assert config["num_local_experts"] == K

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    for layer in range(N_LAYERS):
        expert_keys = [
            key for key in weight_map
            if prune_apply_physical.EXPERT_KEY_RE.match(key)
            and int(prune_apply_physical.EXPERT_KEY_RE.match(key).group("layer")) == layer
        ]
        assert len(expert_keys) == K * 3
        new_experts = sorted(
            {int(prune_apply_physical.EXPERT_KEY_RE.match(key).group("expert")) for key in expert_keys}
        )
        assert new_experts == list(range(K))

        gate_key = f"model.layers.{layer}.mlp.gate.weight"
        with safe_open(out_dir / weight_map[gate_key], framework="pt") as f:
            router = f.get_tensor(gate_key)
        assert router.shape == (K, HIDDEN)

        # unrelated non-expert tensor survives unchanged.
        assert f"model.layers.{layer}.input_layernorm.weight" in weight_map


def test_protected_expert_weight_survives_unmodified(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    out_dir = tmp_path / "out"
    orig_tensors = _write_fixture_checkpoint(ckpt_dir)

    keep_mask = prune_apply_physical.build_uniform_keep_mask(_scores(), _protected_mask(), K)
    prune_apply_physical.apply_physical(ckpt_dir, keep_mask, out_dir)

    kept_sorted = sorted(np.where(keep_mask[1])[0].tolist())
    new_idx = kept_sorted.index(7)
    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    new_key = f"model.layers.1.mlp.experts.{new_idx}.gate_proj.weight"
    with safe_open(out_dir / weight_map[new_key], framework="pt") as f:
        surv = f.get_tensor(new_key)
    assert torch.equal(orig_tensors["model.layers.1.mlp.experts.7.gate_proj.weight"], surv)
