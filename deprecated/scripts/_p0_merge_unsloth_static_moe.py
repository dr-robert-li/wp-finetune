"""Unsloth-static fused-MoE candidate merge for ckpt-72 → HF safetensors.

CORRECTED from council artifact (2026-05-29): math uses PER-EXPERT extraction,
not broadcast. Forensic basis: scripts/_p0_extraction_probe.py demonstrated
cos_sim(broadcast, per_expert) ≈ 0.08 — broadcast interpretation is orthogonal
to trained signal. Per-expert reshape matches Unsloth's `_extract_lora_from_wrapper`
(moe_utils.py:421-426) exactly (max_diff < 1e-6 sanity check).

Merge strategy:
  - Attention Q/K/V/O: standard PEFT merge via attention-only temp adapter
  - MoE gate_up (base_layer): per-expert delta_e = B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] * scale
                                split along output dim into gate (first 768) + up (last 768)
                                apply to per-expert HF gate_proj/up_proj weights
  - MoE down_proj (direct):    per-expert delta_e = B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] * scale
                                apply directly to per-expert HF down_proj weights

Writes candidate only. Does not promote. Forward-pass anchor must pass before
promotion to canonical `models/qwen3-30b-wp-30_70-reasoning-merged/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = "models/qwen3-30b-wp-30_70-merged-v2"
ADAPTER_PATH    = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72"
CANDIDATE_PATH  = "models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate"
REPORT_PATH     = f"{CANDIDATE_PATH}/merge_report.json"
TMP_SUFFIX      = ".tmp_merge"

NUM_LAYERS  = 48
NUM_EXPERTS = 128
ATTN_PROJS  = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_scaling(cfg: dict) -> float:
    r = cfg["r"]
    alpha = cfg.get("lora_alpha", r)
    if cfg.get("use_rslora", False):
        scale = alpha / math.sqrt(r)
        print(f"  rsLoRA: scale = {alpha} / sqrt({r}) = {scale:.6f}")
    else:
        scale = alpha / r
        print(f"  Standard LoRA: scale = {alpha} / {r} = {scale:.6f}")
    return scale


def load_adapter_tensors(adapter_path: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    single = Path(adapter_path) / "adapter_model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    else:
        for sf in sorted(Path(adapter_path).glob("adapter_model*.safetensors")):
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    print(f"  Loaded {len(tensors)} adapter tensor keys")
    return tensors


def build_attention_only_adapter(adapter_path: str, tmp_dir: str) -> str:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    all_tensors = load_adapter_tensors(adapter_path)
    attn_tensors = {
        k: v for k, v in all_tensors.items()
        if any(f".{p}." in k for p in ATTN_PROJS)
    }
    print(f"  Attention-only adapter: {len(attn_tensors)} keys")
    assert len(attn_tensors) > 0, "No attention LoRA tensors found"
    save_file(attn_tensors, f"{tmp_dir}/adapter_model.safetensors")

    cfg = json.load(open(f"{adapter_path}/adapter_config.json"))
    cfg["target_modules"] = ATTN_PROJS
    cfg["target_parameters"] = []
    cfg["modules_to_save"] = None
    json.dump(cfg, open(f"{tmp_dir}/adapter_config.json", "w"), indent=2)
    return tmp_dir


def peft_merge_attention(base_path: str, attn_adapter_dir: str):
    print("\n[STEP 1] PEFT attention merge ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map={"": "cpu"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(model, attn_adapter_dir)
    merged = peft_model.merge_and_unload()
    print("  Attention merge complete")
    return merged


def apply_moe_delta_per_expert(model, adapter_tensors: dict, scale: float, r: int, report: dict):
    """Per-expert Unsloth-convention delta application.

    transformers 5.3.0 stores Qwen3MoeExperts as FUSED stacked 3D params:
      gate_up_proj: (E, 2*intermediate, hidden) = (128, 1536, 2048)
      down_proj:    (E, hidden, intermediate)   = (128, 2048, 768)
    No per-submodule indexing, no gate/up split — gate_up stays fused.
    delta_e = (B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :]) * scale  added to param.data[e].
    """
    print("\n[STEP 2] Custom per-expert Unsloth-static MoE merge (fused 3D params) ...")
    stats = {"gate_up_proj": [], "down_proj": []}
    gate_up_touched = down_touched = 0
    l0_down_delta_e0 = None
    l0_down_delta_e1 = None

    for L in range(NUM_LAYERS):
        experts = model.model.layers[L].mlp.experts
        gate_up_param = experts.gate_up_proj   # (E, 1536, 2048)
        down_param = experts.down_proj         # (E, 2048, 768)

        # gate_up_proj fused (adapter base_layer.*)
        key_A_gu = f"base_model.model.model.layers.{L}.mlp.experts.base_layer.lora_A.weight"
        key_B_gu = f"base_model.model.model.layers.{L}.mlp.experts.base_layer.lora_B.weight"
        if key_A_gu in adapter_tensors and key_B_gu in adapter_tensors:
            A_gu = adapter_tensors[key_A_gu].float()   # (E*R, hidden) = (4096, 2048)
            B_gu = adapter_tensors[key_B_gu].float()   # (out, E*R) = (1536, 4096)
            assert A_gu.shape == (NUM_EXPERTS * r, 2048), f"L{L} A_gu shape {A_gu.shape}"
            assert B_gu.shape == (1536, NUM_EXPERTS * r), f"L{L} B_gu shape {B_gu.shape}"
            assert tuple(gate_up_param.shape) == (NUM_EXPERTS, 1536, 2048), f"L{L} base gate_up {tuple(gate_up_param.shape)}"

            for e in range(NUM_EXPERTS):
                A_e = A_gu[e * r:(e + 1) * r, :]       # (R, hidden)
                B_e = B_gu[:, e * r:(e + 1) * r]       # (out, R)
                delta_e = (B_e @ A_e) * scale          # (1536, 2048) = (out, in) matches param[e]
                gate_up_param.data[e] += delta_e.to(torch.bfloat16)
                gate_up_touched += 1
                if e == 0:
                    stats["gate_up_proj"].append(delta_e.abs().max().item())
        else:
            print(f"  WARNING: missing gate_up keys at layer {L}")

        # down_proj direct (no base_layer prefix)
        key_A_d = f"base_model.model.model.layers.{L}.mlp.experts.lora_A.weight"
        key_B_d = f"base_model.model.model.layers.{L}.mlp.experts.lora_B.weight"
        if key_A_d in adapter_tensors and key_B_d in adapter_tensors:
            A_d = adapter_tensors[key_A_d].float()    # (E*R, intermediate) = (4096, 768)
            B_d = adapter_tensors[key_B_d].float()    # (hidden, E*R) = (2048, 4096)
            assert A_d.shape == (NUM_EXPERTS * r, 768), f"L{L} A_d shape {A_d.shape}"
            assert B_d.shape == (2048, NUM_EXPERTS * r), f"L{L} B_d shape {B_d.shape}"
            assert tuple(down_param.shape) == (NUM_EXPERTS, 2048, 768), f"L{L} base down {tuple(down_param.shape)}"

            for e in range(NUM_EXPERTS):
                A_e = A_d[e * r:(e + 1) * r, :]       # (R, intermediate)
                B_e = B_d[:, e * r:(e + 1) * r]       # (hidden, R)
                delta_e = (B_e @ A_e) * scale         # (2048, 768) = (out, in) matches param[e]
                down_param.data[e] += delta_e.to(torch.bfloat16)
                down_touched += 1
                if e == 0:
                    stats["down_proj"].append(delta_e.abs().max().item())
                if L == 0 and e == 0:
                    l0_down_delta_e0 = delta_e.clone()
                if L == 0 and e == 1:
                    l0_down_delta_e1 = delta_e.clone()
        else:
            print(f"  WARNING: missing down_proj keys at layer {L}")

        if (L + 1) % 8 == 0:
            print(f"  layers {L - 7}..{L} done")

    expected = NUM_LAYERS * NUM_EXPERTS
    assert gate_up_touched == expected, f"gate_up touched {gate_up_touched}/{expected}"
    assert down_touched == expected, f"down_proj touched {down_touched}/{expected}"

    print(f"\n  gate_up_proj: {gate_up_touched} experts touched, max_delta(L0e0) {max(stats['gate_up_proj']):.6f}")
    print(f"  down_proj:    {down_touched} experts touched, max_delta(L0e0) {max(stats['down_proj']):.6f}")

    # GPT/Claude council safeguard: per-expert DELTAS must differ (not final weights —
    # base weights already differ per expert). Compares captured L0 down_proj deltas for
    # experts 0,1. If identical, merge silently broadcast (regression to rejected math).
    assert l0_down_delta_e0 is not None and l0_down_delta_e1 is not None, "delta capture failed"
    expert_delta_diff = (l0_down_delta_e0.float() - l0_down_delta_e1.float()).abs().max().item()
    assert expert_delta_diff > 1e-5, (
        f"REGRESSION: L0 down_proj deltas for experts 0,1 identical (max_diff={expert_delta_diff:.2e}). "
        "Merge silently broadcast — per-expert math failed."
    )
    print(f"  per-expert-differ check: L0 down_proj delta e0 vs e1 max_diff={expert_delta_diff:.6f} (>1e-5 OK)")
    report["moe_merge"]["gate_up_proj_touched"] = gate_up_touched
    report["moe_merge"]["down_proj_touched"] = down_touched
    report["moe_merge"]["gate_up_max_delta_L0e0"] = round(max(stats["gate_up_proj"]), 6)
    report["moe_merge"]["down_max_delta_L0e0"] = round(max(stats["down_proj"]), 6)
    report["moe_merge"]["per_expert_delta_differ_check"] = round(expert_delta_diff, 6)
    report["moe_merge"]["fused_layout"] = "transformers_5.3_stacked_3d (gate_up fused, no split)"
    return model


def save_candidate(model, tokenizer, candidate_path: str) -> int:
    print(f"\n[STEP 3] Saving candidate to {candidate_path} ...")
    tmp = candidate_path + TMP_SUFFIX
    if Path(tmp).exists():
        shutil.rmtree(tmp)
    Path(tmp).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(tmp)
    if Path(candidate_path).exists():
        shutil.rmtree(candidate_path)
    Path(tmp).rename(candidate_path)
    shards = list(Path(candidate_path).glob("model-*-of-*.safetensors"))
    print(f"  {len(shards)} shards written")
    return len(shards)


def main() -> int:
    import argparse
    # --- RTRN-05 parameterization (04.3-02 Task 2): adapter + out-dir are CLI args. ---
    # ADAPTER_PATH/CANDIDATE_PATH/REPORT_PATH remain ONLY as overridable argparse DEFAULTS;
    # main() reads the resolved adapter_path/out_dir locals below, never the ckpt-72 module
    # constant directly, so no ckpt-72 path is a code-read constant (contamination blocker,
    # T-0432-01). Merge math + the 5 promotion gates are untouched.
    ap = argparse.ArgumentParser(description="Unsloth-static fused-MoE candidate merge (parameterized)")
    ap.add_argument("--adapter-path", default=ADAPTER_PATH,
                    help="LoRA adapter/checkpoint dir to merge (full path). Overrides the legacy "
                         "ADAPTER_PATH default; the resolved value is echoed + written to merge_report.json.")
    ap.add_argument("--out-dir", default=CANDIDATE_PATH,
                    help="Candidate output dir (staging). Overrides the legacy CANDIDATE_PATH default.")
    args = ap.parse_args()

    adapter_path = args.adapter_path
    out_dir = args.out_dir
    report_path = f"{out_dir}/merge_report.json"
    # Echo the resolved adapter + out-dir BEFORE any work (provenance / T-0432-04).
    print(f"[merge] resolved adapter-path: {adapter_path}")
    print(f"[merge] resolved out-dir:      {out_dir}")

    t0 = time.time()
    cfg = json.load(open(f"{adapter_path}/adapter_config.json"))
    scale = get_scaling(cfg)
    r = cfg["r"]

    report = {
        "status":        "candidate_pending_anchor",
        "merge_type":    "unsloth_static_moe_per_expert_plus_peft_attention",
        "base_model":    BASE_MODEL_PATH,
        "adapter":       adapter_path,
        "resolved_adapter_path": adapter_path,
        "out_dir":       out_dir,
        "scale":         scale,
        "r":             r,
        "lora_alpha":    cfg.get("lora_alpha"),
        "use_rslora":    cfg.get("use_rslora", False),
        "created_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "math_convention": "per_expert_contiguous_block (Unsloth moe_utils.py:421-426)",
        "preflight_probe": "scripts/_p0_extraction_probe.py — broadcast cos_sim 0.08, per_expert raw-vs-permute exact match",
        "attention_merge": {
            "method": "peft_attention_only_adapter",
            "projs":  ATTN_PROJS,
            "layers": NUM_LAYERS,
        },
        "moe_merge": {
            "method":      "unsloth_static_per_expert",
            "num_experts": NUM_EXPERTS,
            "num_layers":  NUM_LAYERS,
        },
        "tensor_anchor":  "pending",
        "forward_anchor": "pending",
    }

    adapter_tensors = load_adapter_tensors(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # Per-run tmp adapter dir derived from out_dir so a ckpt-50 run never collides with /
    # overwrites a ckpt-72 run's scratch dir.
    tmp_attn_dir = f"/tmp/attn_only_adapter_{Path(out_dir).name}"

    build_attention_only_adapter(adapter_path, tmp_attn_dir)
    model = peft_merge_attention(BASE_MODEL_PATH, tmp_attn_dir)
    model = apply_moe_delta_per_expert(model, adapter_tensors, scale, r, report)

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    n_shards = save_candidate(model, tokenizer, out_dir)
    report["shard_count"] = n_shards

    report["wall_clock_sec"] = round(time.time() - t0, 1)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(report_path, "w"), indent=2)
    print(f"\n[DONE] Candidate written in {report['wall_clock_sec']}s")
    print(f"  Report: {report_path}")
    print(f"  STATUS: candidate_pending_anchor — DO NOT PROMOTE until anchors pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
