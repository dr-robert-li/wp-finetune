"""Diagnostic: is the forward-anchor FAIL bf16 storage rounding or a real merge bug?

Isolates the question by working at the WEIGHT level in fp32:
  true_merged_fp32[e] = base_v3_weight[e].float() + delta_e.float()
  candidate_stored[e] = merged candidate weight[e] (bf16 → float())

If (candidate_stored - true_merged_fp32) ~ bf16 epsilon for the weight magnitude,
the merge math is CORRECT and the forward-anchor FAIL is purely bf16 weight-storage
precision (delta baked into bf16). If the diff is structured/large, real bug.

Also reports the bf16 quantization floor for the weight magnitude as the reference scale.
CPU-only, loads base v3 + candidate sequentially (one at a time).
"""

from __future__ import annotations

import gc
import json
import math
import sys

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE_MODEL_PATH = "models/qwen3-30b-wp-30_70-merged-v2"
CANDIDATE_PATH  = "models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate"
ADAPTER_SF      = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
ADAPTER_CFG     = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_config.json"

TEST = [(0, 0), (0, 1), (23, 63), (47, 127)]  # (layer, expert)


def bf16_round(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def main() -> int:
    cfg = json.load(open(ADAPTER_CFG))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = (alpha / math.sqrt(r)) if cfg.get("use_rslora", False) else (alpha / r)
    print(f"r={r} scale={scale}\n")

    # Load adapter LoRA for needed layers
    layers = sorted({L for L, _ in TEST})
    lora = {}
    with safe_open(ADAPTER_SF, framework="pt", device="cpu") as f:
        for L in layers:
            lora[L] = {
                "gu_A": f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.base_layer.lora_A.weight").float(),
                "gu_B": f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.base_layer.lora_B.weight").float(),
                "dn_A": f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.lora_A.weight").float(),
                "dn_B": f.get_tensor(f"base_model.model.model.layers.{L}.mlp.experts.lora_B.weight").float(),
            }

    # Capture base v3 expert weights for the test (layer, expert) pairs
    print("[1] Loading base v3 ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map={"": "cpu"}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base_w = {}
    for (L, e) in TEST:
        base_w[(L, e, "gu")] = base.model.layers[L].mlp.experts.gate_up_proj.data[e].float().clone()
        base_w[(L, e, "dn")] = base.model.layers[L].mlp.experts.down_proj.data[e].float().clone()
    del base
    gc.collect()
    print("[1] base captured + freed")

    # Capture candidate stored weights
    print("[2] Loading candidate ...")
    cand = AutoModelForCausalLM.from_pretrained(
        CANDIDATE_PATH, device_map={"": "cpu"}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    cand_w = {}
    for (L, e) in TEST:
        cand_w[(L, e, "gu")] = cand.model.layers[L].mlp.experts.gate_up_proj.data[e].float().clone()
        cand_w[(L, e, "dn")] = cand.model.layers[L].mlp.experts.down_proj.data[e].float().clone()
    del cand
    gc.collect()
    print("[2] candidate captured + freed\n")

    print(f"{'layer/exp/proj':>16} {'weight_rms':>11} {'bf16_floor':>11} {'cand-true_max':>14} {'cand-true_rms':>14} {'verdict':>10}")
    all_ok = True
    for (L, e) in TEST:
        for proj, suf in [("gu", "gate_up"), ("dn", "down")]:
            bw = base_w[(L, e, proj)]
            cw = cand_w[(L, e, proj)]
            if proj == "gu":
                A = lora[L]["gu_A"][e * r:(e + 1) * r, :]
                B = lora[L]["gu_B"][:, e * r:(e + 1) * r]
            else:
                A = lora[L]["dn_A"][e * r:(e + 1) * r, :]
                B = lora[L]["dn_B"][:, e * r:(e + 1) * r]
            delta = (B @ A) * scale                      # fp32
            true_merged_fp32 = bw + delta                # fp32 ground truth
            # What bf16 storage of the true merge would give:
            true_merged_bf16 = bf16_round(true_merged_fp32)

            diff_cand_vs_true = (cw - true_merged_fp32).abs()
            diff_cand_vs_bf16round = (cw - true_merged_bf16).abs()

            weight_rms = true_merged_fp32.pow(2).mean().sqrt().item()
            # bf16 has 8 mantissa bits → relative step ~2^-8; absolute floor ~ rms * 2^-8
            bf16_floor = weight_rms * (2 ** -8)
            max_d = diff_cand_vs_true.max().item()
            rms_d = diff_cand_vs_true.pow(2).mean().sqrt().item()

            # Verdict: candidate should match bf16-rounded true merge near-exactly
            # (proves merge math correct; residual is just bf16 storage)
            ok = diff_cand_vs_bf16round.max().item() < 1e-4
            all_ok = all_ok and ok
            verdict = "bf16-noise" if (rms_d < 3 * bf16_floor) else "REAL-BUG?"
            tag = "OK" if ok else "MISMATCH"
            print(f"  L{L}e{e}/{suf:>7} {weight_rms:>11.6f} {bf16_floor:>11.6f} "
                  f"{max_d:>14.6f} {rms_d:>14.6f} {verdict:>10} [{tag}]")

    print()
    print("Interpretation:")
    print("  cand-vs-bf16round MISMATCH everywhere → real merge bug")
    print("  cand ≈ bf16round(true) AND cand-true_rms ~ bf16_floor → merge correct,")
    print("    forward-anchor FAIL was bf16 weight-storage precision (thresholds too tight)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
