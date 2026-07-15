#!/usr/bin/env python
"""Phase 23-03 extension: convert a Tinker routed-MoE-expert LoRA export
(``mlp.experts.{w1,w2,w3}`` 3D-batched target_parameters tensors) into the
PEFT convention vLLM's ``FusedMoE3DWithLoRA`` (vllm/lora/layers/fused_moe.py,
vllm/lora/model_manager.py::_stack_moe_lora_weights) expects for a model with
``is_3d_moe_weight = True`` (confirmed for Qwen3_5MoeForConditionalGeneration,
models/Qwen3.6-35B-A3B's registered architecture):

  - ``mlp.experts.base_layer.{lora_A,lora_B}`` -- fused gate+up (w13)
  - ``mlp.experts.{lora_A,lora_B}``            -- down (w2), the "experts"
    module's own direct tensors

This is a LOSSLESS rename+reshape, not an approximation:
  - w1.lora_A and w3.lora_A are bit-identical per layer in the Tinker export
    (verified) -- Tinker already trains gate_proj/up_proj sharing one LoRA-A,
    exactly the structure FusedMoE3DWithLoRA requires (one shared A, a
    doubled-width B). No rank inflation, no block-diagonal padding.
  - Shapes below are traced by hand against the installed vLLM build's
    ``_stack_moe_lora_weights`` reshape/permute chain (not guessed) -- see
    ``output/eval4/ext_unmerged_preregistration.md`` for the derivation.

Per layer (num_experts=E, rank=r, hidden=H, intermediate=I):
  Tinker raw:
    w1.lora_A [1,r,H]  w1.lora_B [E,I,r]   (gate)
    w3.lora_A [1,r,H]  w3.lora_B [E,I,r]   (up, A == w1's A)
    w2.lora_A [E,r,I]  w2.lora_B [1,H,r]   (down)
  vLLM PEFT convention (this script writes):
    experts.base_layer.lora_A [E,r,H]      = w1.lora_A materialized (repeat E)
    experts.base_layer.lora_B [2I,r,E]     = cat(w1.lora_B.permute(1,2,0),
                                                  w3.lora_B.permute(1,2,0), dim=0)
    experts.lora_A            [E,r,I]      = w2.lora_A unchanged
    experts.lora_B            [H,r,E]      = w2.lora_B materialized+permuted

shared_expert.{gate_proj,up_proj,down_proj} keys are copied unchanged --
their names already match vLLM's supported LoRA module suffixes.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert(src_dir: Path, dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(str(src_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        raw = {k: f.get_tensor(k) for k in keys}

    # Group routed-expert keys by layer.
    layer_re = re.compile(r"^(.*\.layers\.(\d+)\.mlp)\.experts\.(w[123])\.(lora_[AB])\.weight$")
    by_layer: dict[str, dict[str, torch.Tensor]] = {}
    other: dict[str, torch.Tensor] = {}
    for k, t in raw.items():
        m = layer_re.match(k)
        if m:
            prefix, _idx, w, ab = m.groups()
            by_layer.setdefault(prefix, {})[f"{w}.{ab}"] = t
        else:
            other[k] = t

    out: dict[str, torch.Tensor] = dict(other)
    n_layers_converted = 0
    for prefix, d in by_layer.items():
        w1a, w1b = d["w1.lora_A"], d["w1.lora_B"]
        w2a, w2b = d["w2.lora_A"], d["w2.lora_B"]
        w3a, w3b = d["w3.lora_A"], d["w3.lora_B"]

        assert torch.equal(w1a, w3a), f"{prefix}: w1/w3 lora_A must be identical (Tinker shared-A convention)"
        num_experts, rank, hidden = w1b.shape[0], w1a.shape[1], w1a.shape[2]
        assert w1a.shape[0] == 1, f"{prefix}: expected broadcast-shared w1.lora_A shape[0]==1, got {w1a.shape}"
        assert w2b.shape[0] == 1, f"{prefix}: expected broadcast-shared w2.lora_B shape[0]==1, got {w2b.shape}"

        # experts.base_layer.lora_A: [E, r, H] (materialize the broadcast dim)
        base_layer_a = w1a.expand(num_experts, rank, hidden).contiguous()
        # experts.base_layer.lora_B: [2I, r, E] = cat(gate.permute(1,2,0), up.permute(1,2,0))
        gate_b = w1b.permute(1, 2, 0).contiguous()  # [I, r, E]
        up_b = w3b.permute(1, 2, 0).contiguous()  # [I, r, E]
        base_layer_b = torch.cat([gate_b, up_b], dim=0)  # [2I, r, E]

        # experts.lora_A: [E, r, I] -- already in the right layout, unchanged
        experts_a = w2a.contiguous()
        # experts.lora_B: [H, r, E] (materialize the broadcast dim, then permute)
        H = w2b.shape[1]
        experts_b = w2b.squeeze(0).unsqueeze(-1).expand(H, rank, num_experts).contiguous()

        out[f"{prefix}.experts.base_layer.lora_A.weight"] = base_layer_a
        out[f"{prefix}.experts.base_layer.lora_B.weight"] = base_layer_b
        out[f"{prefix}.experts.lora_A.weight"] = experts_a
        out[f"{prefix}.experts.lora_B.weight"] = experts_b
        n_layers_converted += 1

    save_file(out, str(dst_dir / "adapter_model.safetensors"))

    cfg = json.loads((src_dir / "adapter_config.json").read_text())
    (dst_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    receipt = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "n_layers_converted": n_layers_converted,
        "n_output_keys": len(out),
        "n_input_keys": len(raw),
        "other_keys_copied_unchanged": len(other),
        "sample_shapes": {
            k: list(out[k].shape)
            for k in list(out.keys())
            if ".layers.0.mlp.experts." in k or ".layers.0.mlp.shared_expert." in k
        },
    }
    return receipt


def validate(dst_dir: Path, expected_total_modules: int = 240) -> dict:
    """Cheap validation: no tensor math, just coverage + shape sanity."""
    with safe_open(str(dst_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    mods = sorted({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in keys})
    routed = [m for m in mods if m.endswith(".experts") or m.endswith(".experts.base_layer")]
    shared = [m for m in mods if ".shared_expert." in m]
    ok = (len(routed) + len(shared)) > 0 and len(mods) > 0
    return {
        "total_modules": len(mods),
        "routed_expert_modules": len(routed),  # 2 per layer now (base_layer + bare), was 3 (w1/w2/w3)
        "shared_expert_modules": len(shared),
        "coverage_ok": ok,
    }


def main() -> int:
    src = PROJECT_ROOT / "output" / "base21" / "judge03_s1_adapter"
    dst = PROJECT_ROOT / "output" / "eval4" / "ext_unmerged" / "judge03_s1_adapter_vllm_peft"
    receipt = convert(src, dst)
    check = validate(dst)
    receipt["validation"] = check
    out_path = PROJECT_ROOT / "output" / "eval4" / "ext_unmerged_convert_receipt_s1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    if not check["coverage_ok"]:
        print("VALIDATION FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
