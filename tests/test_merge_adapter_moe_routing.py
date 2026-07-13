"""21-01 gap-closure: routed-MoE-expert adapter detection + merge-path routing.

The T-21-01 gap: merge_adapter.py's PEFT target_modules path silently drops
Tinker's train_mlp=True routed-expert tensors (mlp.experts.{w1,w2,w3}).
The fix routes such adapters through tinker_cookbook's build_hf_model,
whose vendor-maintained w1/w3/w2 -> gate/up/down mapping matches the
convention scripts/merge_tinker_v3.py independently shipped v1.2/v1.3 on.

These tests are cheap (no model load): they pin (1) the detector that picks
the merge path, and (2) the vendor mapping + fused layout the fix relies on,
so a tinker_cookbook upgrade that silently changes either fails loudly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_adapter import _adapter_has_routed_expert_params  # noqa: E402

safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_adapter(tmp_path: Path, keys: list[str]) -> str:
    tensors = {k: torch.zeros(2, 2) for k in keys}
    safetensors_torch.save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    return str(tmp_path)


def test_detects_routed_expert_adapter(tmp_path):
    adapter_dir = _write_adapter(tmp_path, [
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight",
        "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
    ])
    assert _adapter_has_routed_expert_params(adapter_dir) is True


def test_attention_only_adapter_not_routed(tmp_path):
    adapter_dir = _write_adapter(tmp_path, [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_B.weight",
    ])
    assert _adapter_has_routed_expert_params(adapter_dir) is False


def test_vendor_w1_w3_w2_mapping_is_gate_up_down():
    """Pin the semantic mapping the whole gap-closure rests on: the installed
    tinker_cookbook (same vendor as the exporter) maps w1->gate, w3->up,
    w2->down -- identical to merge_tinker_v3.py's empirically-shipped
    old-base convention."""
    tinker_cookbook = pytest.importorskip("tinker_cookbook")  # noqa: F841
    from tinker_cookbook.weights._merge import MergeProfile

    assert MergeProfile().expert_key_remaps == (
        (".w1.weight", ".gate_proj.weight"),
        (".w3.weight", ".up_proj.weight"),
        (".w2.weight", ".down_proj.weight"),
    )


def test_vendor_fused_layout_gate_first():
    """Pin fused_concatenated = [gate | up] with gate at fused_proj_idx 0
    (matches modeling_qwen3_5_moe.py's `gate, up = ...chunk(2)` and
    merge_tinker_v3.py's gate-first concat)."""
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.weights._merge_utils import plan_expert_ops
    from tinker_cookbook.weights._merge import MergeProfile

    profile = MergeProfile(expert_layout="fused_concatenated")
    ops: dict = {}
    n_exp, rank, in_dim, out_dim = 4, 2, 8, 6
    lora_A = torch.randn(1, rank, in_dim)        # shared A (w1/w3 convention)
    lora_B = torch.randn(n_exp, out_dim, rank)   # per-expert B
    model_keys = {"model.layers.0.mlp.experts.gate_up_proj"}
    plan_expert_ops(
        "model.layers.0.mlp.experts.w1.weight", lora_A, lora_B,
        "adapter.w1", profile, model_keys, ops,
    )
    (op,) = ops["model.layers.0.mlp.experts.gate_up_proj"]
    assert op.is_expert_3d is True
    assert op.fused_proj_idx == 0  # w1 -> gate -> FIRST half of gate_up_proj

    ops2: dict = {}
    plan_expert_ops(
        "model.layers.0.mlp.experts.w3.weight", lora_A, lora_B,
        "adapter.w3", profile, model_keys, ops2,
    )
    (op3,) = ops2["model.layers.0.mlp.experts.gate_up_proj"]
    assert op3.fused_proj_idx == 1  # w3 -> up -> SECOND half
