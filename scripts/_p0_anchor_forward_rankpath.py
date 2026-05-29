"""Stage-2 forward anchor: CPU rank-path vs materialized-weight-path equivalence.

Council-certified Stage-2 gate (cpu_rank_path_vs_materialized_weight_path).

Reference (rank-path): base v3 weights + LoRA applied in ACTIVATION space during a
  full MoE-block forward — gate+up LoRA → SwiGLU → down LoRA (down sees the
  LoRA-adapted intermediate). Uses the Stage-1-certified contiguous-block extraction.
Candidate (weight-path): plain forward through merged weights (delta baked into params).

If the two agree within bf16 tolerance, the `param.data[e] += delta` bake faithfully
reproduces activation-space LoRA application. Router is untouched by the adapter
(no router LoRA keys), so selected_experts must be identical — asserted as invariance.

Sequential load (one model at a time) → peak RAM ~one model. CPU-only, no GPU, no Unsloth.
HF+PEFT fallback is NOT used (semantically wrong for this adapter) — this is the only ref.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE_MODEL_PATH = "models/qwen3-30b-wp-30_70-merged-v2"
CANDIDATE_PATH  = "models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate"
ADAPTER_SF      = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
ADAPTER_CFG     = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_config.json"
REPORT_PATH     = f"{CANDIDATE_PATH}/merge_report.json"

TEST_LAYERS = [0, 23, 47]
SEEDS       = [42, 137, 999]
HIDDEN_DIM  = 2048
DTYPE       = torch.bfloat16

# bf16-calibrated thresholds (council A+B, 2026-05-29). Old fp32-grade thresholds
# (cos>=0.99999, rel_l2<=1e-3) failed AS EXPECTED — bf16-stored 30B weights cannot
# meet fp32 equivalence. Primary Stage-2 certifier is the fp32 weight-control
# (_p0_anchor_fp32_control.py: cand == bf16(true merge), rms sub-floor). This forward
# anchor is bf16-calibrated CORROBORATION + hard router-invariance.
THRESHOLDS = {"mean_abs": 2e-3, "relative_l2": 1e-2, "cosine": 0.99990, "max_abs_hard": 1e-1}


def make_hidden(seed: int, batch: int = 4, seq: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # small-magnitude realistic post-norm hidden states
    return torch.randn(batch, seq, HIDDEN_DIM, generator=g, dtype=torch.float32).to(DTYPE)


def load_layer_adapter(layer: int, r: int) -> dict:
    """Load the 4 fused-MoE LoRA tensors for one layer."""
    out = {}
    with safe_open(ADAPTER_SF, framework="pt", device="cpu") as f:
        out["gu_A"] = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.base_layer.lora_A.weight").float()
        out["gu_B"] = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.base_layer.lora_B.weight").float()
        out["dn_A"] = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.lora_A.weight").float()
        out["dn_B"] = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.lora_B.weight").float()
    return out


def make_rankpath_experts_forward(orig_module, lora: dict, r: int, scale: float):
    """Build a patched experts.forward that adds activation-space LoRA per expert.

    Mirrors Qwen3MoeExperts.forward (transformers 5.3) but injects:
      gate_up += (x @ A_gu_e.T) @ B_gu_e.T * scale     (before chunk/SwiGLU)
      down    += (h @ A_dn_e.T) @ B_dn_e.T * scale     (h = SwiGLU-adapted intermediate)
    """
    gu_A, gu_B = lora["gu_A"], lora["gu_B"]   # (E*R, hidden), (2*inter, E*R)
    dn_A, dn_B = lora["dn_A"], lora["dn_B"]   # (E*R, inter), (hidden, E*R)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]   # (n, hidden)

            # gate_up: base + activation-space LoRA
            gu_base = F.linear(current_state, self.gate_up_proj[e])          # (n, 2*inter)
            A_e = gu_A[e * r:(e + 1) * r, :]                                 # (R, hidden)
            B_e = gu_B[:, e * r:(e + 1) * r]                                 # (2*inter, R)
            gu_lora = ((current_state.float() @ A_e.T) @ B_e.T) * scale      # (n, 2*inter)
            gate_up = gu_base.float() + gu_lora
            gate, up = gate_up.chunk(2, dim=-1)
            h = self.act_fn(gate) * up                                       # (n, inter)
            h = h.to(DTYPE)

            # down: base + activation-space LoRA (sees adapted intermediate h)
            dn_base = F.linear(h, self.down_proj[e])                         # (n, hidden)
            Ad_e = dn_A[e * r:(e + 1) * r, :]                               # (R, inter)
            Bd_e = dn_B[:, e * r:(e + 1) * r]                               # (hidden, R)
            dn_lora = ((h.float() @ Ad_e.T) @ Bd_e.T) * scale               # (n, hidden)
            current_hidden_states = (dn_base.float() + dn_lora).to(DTYPE)

            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    return forward


def capture_ref(layers: list, seeds: list, r: int, scale: float) -> dict:
    """Load base v3, patch experts.forward to rank-path, capture MoE-block outputs + routing."""
    print("[REF] Loading base v3 (CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, device_map={"": "cpu"}, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    )
    model.eval()
    out = {}
    for L in layers:
        lora = load_layer_adapter(L, r)
        block = model.model.layers[L].mlp
        block.experts.forward = types.MethodType(
            make_rankpath_experts_forward(block.experts, lora, r, scale), block.experts
        )
        for seed in seeds:
            hidden = make_hidden(seed)
            with torch.no_grad():
                hr = hidden.view(-1, HIDDEN_DIM)
                _, rw, sel = block.gate(hr)
                moe_out = block.experts(hr, sel, rw).view(hidden.shape)
            out[(L, seed)] = {"out": moe_out.float().clone(), "sel": sel.clone()}
        del lora
    del model
    gc.collect()
    print("[REF] captured + freed")
    return out


def capture_cand(layers: list, seeds: list) -> dict:
    print("[CAND] Loading merged candidate (CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        CANDIDATE_PATH, device_map={"": "cpu"}, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    )
    model.eval()
    out = {}
    for L in layers:
        block = model.model.layers[L].mlp
        for seed in seeds:
            hidden = make_hidden(seed)
            with torch.no_grad():
                hr = hidden.view(-1, HIDDEN_DIM)
                _, rw, sel = block.gate(hr)
                moe_out = block.experts(hr, sel, rw).view(hidden.shape)
            out[(L, seed)] = {"out": moe_out.float().clone(), "sel": sel.clone()}
    del model
    gc.collect()
    print("[CAND] captured + freed")
    return out


def compare(ref: torch.Tensor, cand: torch.Tensor, sel_ref, sel_cand, label: str) -> dict:
    router_match = bool(torch.equal(sel_ref, sel_cand))
    diff = (cand - ref).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    rel_l2 = (diff.norm() / (ref.norm() + 1e-8)).item()
    cos = F.cosine_similarity(ref.flatten().unsqueeze(0), cand.flatten().unsqueeze(0)).item()
    r = {
        "label": label, "router_match": router_match,
        "mean_abs": round(mean_abs, 8), "max_abs": round(max_abs, 8),
        "relative_l2": round(rel_l2, 8), "cosine": round(cos, 8),
        "pass_mean": mean_abs <= THRESHOLDS["mean_abs"],
        "pass_rel_l2": rel_l2 <= THRESHOLDS["relative_l2"],
        "pass_cosine": cos >= THRESHOLDS["cosine"],
        "fail_hard": max_abs >= THRESHOLDS["max_abs_hard"],
    }
    r["overall_pass"] = (
        router_match and r["pass_mean"] and r["pass_rel_l2"] and r["pass_cosine"] and not r["fail_hard"]
    )
    status = "PASS" if r["overall_pass"] else "FAIL"
    print(f"  {label}: router={'OK' if router_match else 'MISMATCH'} "
          f"mean={mean_abs:.2e} max={max_abs:.2e} rel_l2={rel_l2:.2e} cos={cos:.8f} [{status}]")
    return r


def main() -> int:
    cfg = json.load(open(ADAPTER_CFG))
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scale = (alpha / math.sqrt(r)) if cfg.get("use_rslora", False) else (alpha / r)
    print(f"r={r} alpha={alpha} scale={scale}  layers={TEST_LAYERS} seeds={SEEDS}\n")

    ref = capture_ref(TEST_LAYERS, SEEDS, r, scale)
    cand = capture_cand(TEST_LAYERS, SEEDS)

    print("\n[COMPARE]")
    results = []
    any_fail = False
    for L in TEST_LAYERS:
        for seed in SEEDS:
            rr = ref[(L, seed)]
            cc = cand[(L, seed)]
            res = compare(rr["out"], cc["out"], rr["sel"], cc["sel"], f"L{L}_seed{seed}")
            results.append(res)
            if not res["overall_pass"]:
                any_fail = True

    anchor_pass = not any_fail
    router_all_ok = all(x["router_match"] for x in results)
    print(f"\n{'=' * 60}")
    print(f"FORWARD ANCHOR (rank-path vs weight-path): {'PASS' if anchor_pass else 'FAIL'}")
    print(f"  Tests: {len(results)}  Pass: {sum(x['overall_pass'] for x in results)}  "
          f"Router-invariant: {router_all_ok}")
    print(f"{'=' * 60}")

    report = json.load(open(REPORT_PATH))
    report["forward_anchor"] = "pass" if anchor_pass else "fail"
    report["forward_anchor_type"] = "cpu_rank_path_vs_materialized_weight_path"
    report["forward_anchor_router_invariant"] = router_all_ok
    report["forward_anchor_results"] = results
    report["forward_anchor_promotable"] = anchor_pass  # certifying ref by construction
    if anchor_pass and report.get("tensor_anchor") == "pass":
        report["status"] = "candidate_promotable"
        print("  PROMOTABLE — tensor + forward anchors both PASS")
    elif anchor_pass:
        report["status"] = "candidate_forward_passed_tensor_pending"
    else:
        report["status"] = "candidate_anchor_failed"
        print("  DO NOT PROMOTE — weight bake does not match activation-space LoRA")
    json.dump(report, open(REPORT_PATH, "w"), indent=2)
    print(f"  Report updated: {REPORT_PATH}")
    return 0 if anchor_pass else 1


if __name__ == "__main__":
    sys.exit(main())
