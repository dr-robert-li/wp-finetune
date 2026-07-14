"""Phase 21 diagnostic Experiment 3: fp32-accumulation merge fix.

Cheap, no-model-load tests (mirrors tests/test_merge_adapter_moe_routing.py's
convention) pinning the two upcast helpers merge_adapter.py uses to close the
bf16-delta-computation gap in both merge paths:

  - _fp32_upcast_adapter_copy: routed-MoE-expert (tinker_cookbook) path --
    upcasts a saved adapter's safetensors to float32 on disk.
  - _upcast_lora_layers_to_fp32: legacy PEFT path -- upcasts a live
    PeftModel's LoRA lora_A/lora_B weight tensors to float32 in-place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_adapter import _fp32_upcast_adapter_copy, _upcast_lora_layers_to_fp32  # noqa: E402

safetensors_torch = pytest.importorskip("safetensors.torch")
peft = pytest.importorskip("peft")


def test_fp32_upcast_adapter_copy_upcasts_tensors_and_preserves_config(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tensors = {
        "base_model.model.model.layers.0.mlp.experts.w1.lora_A.weight": torch.randn(2, 3).bfloat16(),
        "base_model.model.model.layers.0.mlp.experts.w1.lora_B.weight": torch.randn(3, 2).bfloat16(),
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text('{"r": 32, "lora_alpha": 32}')

    fp32_dir = _fp32_upcast_adapter_copy(str(adapter_dir))

    assert (Path(fp32_dir) / "adapter_config.json").read_text() == '{"r": 32, "lora_alpha": 32}'
    with safe_open(str(Path(fp32_dir) / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            assert t.dtype == torch.float32, f"{k} not upcast: {t.dtype}"
            # Values preserved (bf16->fp32 upcast, not corrupted/zeroed)
            assert torch.allclose(t, tensors[k].float(), atol=1e-2)


def test_upcast_lora_layers_to_fp32_flips_dtype_in_place():
    from peft import LoraConfig, get_peft_model

    base = torch.nn.Linear(8, 8, bias=False).bfloat16()
    wrapper = torch.nn.Sequential(base)
    peft_model = get_peft_model(wrapper, LoraConfig(r=4, lora_alpha=4, target_modules=["0"]))

    # Simulate a loaded checkpoint's LoRA tensors being bf16 (PEFT's own
    # get_peft_model() default-inits new LoRA layers in fp32 regardless of
    # base dtype -- a real Tinker-exported adapter loaded via
    # load_adapter()/PeftModel.load_adapter() is bf16, matching the
    # merge_adapter.py code path this test exercises).
    lora_module = next(m for m in peft_model.modules()
                        if isinstance(getattr(m, "lora_A", None), torch.nn.ModuleDict) and "default" in m.lora_A)
    lora_module.lora_A["default"].weight.data = lora_module.lora_A["default"].weight.data.bfloat16()
    lora_module.lora_B["default"].weight.data = lora_module.lora_B["default"].weight.data.bfloat16()
    assert lora_module.lora_A["default"].weight.dtype == torch.bfloat16

    n = _upcast_lora_layers_to_fp32(peft_model, "default")

    assert n == 1
    assert lora_module.lora_A["default"].weight.dtype == torch.float32
    assert lora_module.lora_B["default"].weight.dtype == torch.float32


def test_upcast_lora_layers_to_fp32_returns_zero_for_unknown_adapter_name():
    from peft import LoraConfig, get_peft_model

    base = torch.nn.Linear(4, 4, bias=False).bfloat16()
    wrapper = torch.nn.Sequential(base)
    peft_model = get_peft_model(wrapper, LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))

    n = _upcast_lora_layers_to_fp32(peft_model, "does-not-exist")

    assert n == 0
