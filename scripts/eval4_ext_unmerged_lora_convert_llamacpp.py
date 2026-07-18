#!/usr/bin/env python
"""Phase 23-03 extension, llama.cpp fallback: convert a Tinker routed-MoE
LoRA export (mlp.experts.{w1,w2,w3}) into the PEFT convention
llama.cpp/convert_lora_to_gguf.py + conversion/qwen.py's Qwen3.5-MoE
modify_tensors expects for FUSED-checkpoint architectures:

  - mlp.experts.gate_up_proj.{lora_A,lora_B}  -- fused gate(w1)+up(w3)
  - mlp.experts.down_proj.{lora_A,lora_B}     -- down (w2)

This mirrors the exact naming the base model's own safetensors index uses
(models/Qwen3.6-35B-A3B/model.safetensors.index.json:
"mlp.experts.gate_up_proj" / "mlp.experts.down_proj", confirmed by direct
inspection) -- conversion/qwen.py's modify_tensors() chunks gate_up_proj on
dim=-2 into gate_proj/up_proj at conversion time (same logic used
successfully to build models/_gguf/wp-v4-judge-s*.Q8_0.gguf from merged
checkpoints).

Traced against convert_lora_to_gguf.py's LoraTorchTensor class (the
"__getitem__" mod-broadcast-index trick specifically supports a
broadcast-shared A with a materialized-per-expert B, or vice versa -- see
output/eval4/ext_unmerged_preregistration.md for the derivation). Per this
tracing:
  - gate_up_proj.lora_A can stay broadcast-shared [1,r,H] (w1's A ==
    w3's A, verified bit-identical) -- the class's __getitem__ mod-indexes
    a size-1 leading dim safely.
  - gate_up_proj.lora_B = cat([w1.lora_B, w3.lora_B], dim=1) -> [E,2I,r]
    (real per-expert data, chunked back apart at conversion time by
    modify_tensors's data_torch[..., :n_ff, :] / [..., n_ff:, :] slicing).
  - down_proj passes straight through (no chunking) to the generic
    tensor-name mapper, which uses .shape purely via LoraTorchTensor's
    property `(*B.shape[:-1], A.shape[-1])` -- since down_proj's raw A is
    per-expert [E,r,I] but its raw B is broadcast-shared [1,H,r], leaving
    B unmaterialized would make .shape report expert-dim=1 (WRONG, B's
    leading dim drives .shape, not A's). So down_proj.lora_B is
    materialized [1,H,r] -> [E,H,r] here (small tensor, cheap) to keep
    .shape correct regardless of what any downstream logic inspects.
"""
from __future__ import annotations

import json
import re
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

        assert torch.equal(w1a, w3a), f"{prefix}: w1/w3 lora_A must be identical"
        num_experts = w1b.shape[0]
        rank, hidden = w1a.shape[1], w1a.shape[2]
        H = w2b.shape[1]  # w2.lora_B: [1, H, rank]

        # gate_up_proj: A materialized to [E,r,H] -- llama.cpp's build_lora_mm_id applies A via
        # ggml_mul_mat_id(ctx, lw->a, cur, ids), which INDEXES A by real per-token expert ids
        # (a gather, not a numpy-style broadcast batch matmul like vLLM/PyTorch tolerate). A
        # broadcast-1 leading dim would need ids to stay in [0,1) to be safe; real routing
        # produces ids in [0,256), which would be an out-of-bounds gather on the ggml tensor.
        # Materializing avoids relying on undocumented/unverified ggml broadcast behavior.
        gate_up_a = w1a.expand(num_experts, rank, hidden).contiguous()  # [E, r, H]
        gate_up_b = torch.cat([w1b, w3b], dim=1)  # [E, 2I, r]

        # down_proj: A unchanged (already per-expert [E,r,I]); B materialized [E,H,r]
        down_a = w2a
        down_b = w2b.expand(num_experts, H, w2b.shape[-1]).contiguous()

        out[f"{prefix}.experts.gate_up_proj.lora_A.weight"] = gate_up_a
        out[f"{prefix}.experts.gate_up_proj.lora_B.weight"] = gate_up_b
        out[f"{prefix}.experts.down_proj.lora_A.weight"] = down_a
        out[f"{prefix}.experts.down_proj.lora_B.weight"] = down_b
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


def main() -> int:
    src = PROJECT_ROOT / "output" / "base21" / "judge03_s1_adapter"
    dst = PROJECT_ROOT / "output" / "eval4" / "ext_unmerged" / "judge03_s1_adapter_llamacpp_peft"
    receipt = convert(src, dst)
    out_path = PROJECT_ROOT / "output" / "eval4" / "ext_unmerged_convert_receipt_s1_llamacpp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
