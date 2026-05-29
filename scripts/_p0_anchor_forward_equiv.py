"""Forward-pass equivalence anchor for the unsloth-static MoE candidate merge.

Per-layer MoE block forward: compares Unsloth-adapter-attached reference vs
candidate-merged HF model. Layers {0, 23, 47} × seeds {42, 137, 999}.

PASS criteria (all must hold per test):
    mean_abs_diff  <= 1e-3
    relative_l2    <= 1e-3
    cosine_sim     >= 0.99999
    max_abs_diff   <  1e-1   (hard fail)

Updates merge_report.json with verdict. Run inside Unsloth container if available;
falls back to HF+PEFT reference if not (anchor weakened — flag in report).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ADAPTER_PATH    = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72"
BASE_MODEL_PATH = "models/qwen3-30b-wp-30_70-merged-v2"
CANDIDATE_PATH  = "models/qwen3-30b-wp-30_70-reasoning-merged-unsloth-static-candidate"
REPORT_PATH     = f"{CANDIDATE_PATH}/merge_report.json"

TEST_LAYERS = [0, 23, 47]
SEEDS       = [42, 137, 999]
HIDDEN_DIM  = 2048
DTYPE       = torch.bfloat16

THRESHOLDS = {
    "mean_abs":     1e-3,
    "relative_l2":  1e-3,
    "cosine":       0.99999,
    "max_abs_hard": 1e-1,
}


def compare_outputs(out_ref: torch.Tensor, out_cand: torch.Tensor, label: str) -> dict:
    diff = (out_cand - out_ref).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    rel_l2 = (diff.norm() / (out_ref.norm() + 1e-8)).item()
    cos = F.cosine_similarity(
        out_ref.flatten().unsqueeze(0),
        out_cand.flatten().unsqueeze(0),
    ).item()
    result = {
        "label":       label,
        "mean_abs":    round(mean_abs, 8),
        "max_abs":     round(max_abs, 8),
        "relative_l2": round(rel_l2, 8),
        "cosine":      round(cos, 8),
        "pass_mean":   mean_abs <= THRESHOLDS["mean_abs"],
        "pass_rel_l2": rel_l2 <= THRESHOLDS["relative_l2"],
        "pass_cosine": cos >= THRESHOLDS["cosine"],
        "fail_hard":   max_abs >= THRESHOLDS["max_abs_hard"],
    }
    result["overall_pass"] = (
        result["pass_mean"]
        and result["pass_rel_l2"]
        and result["pass_cosine"]
        and not result["fail_hard"]
    )
    status = "PASS" if result["overall_pass"] else "FAIL"
    print(
        f"  {label}: mean={mean_abs:.2e} max={max_abs:.2e} "
        f"rel_l2={rel_l2:.2e} cos={cos:.8f}  [{status}]"
    )
    return result


def make_hidden(seed: int, batch: int = 4, seq: int = 16) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq, HIDDEN_DIM, dtype=DTYPE)


def load_ref_model() -> tuple:
    """Load Unsloth-attached reference if available; else HF+PEFT fallback."""
    try:
        from unsloth import FastLanguageModel
        model, _ = FastLanguageModel.from_pretrained(
            BASE_MODEL_PATH, dtype=DTYPE, load_in_4bit=False,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.eval()
        print("  Reference: Unsloth + PeftModel loaded")
        return model, "unsloth_peft"
    except (ImportError, RuntimeError) as exc:
        print(f"  WARNING: Unsloth unavailable ({exc}); HF+PEFT fallback (WEAKER ANCHOR)")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map={"": "cpu"},
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        return model, "hf_peft_fallback"


def load_cand_model():
    model = AutoModelForCausalLM.from_pretrained(
        CANDIDATE_PATH,
        device_map={"": "cpu"},
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("  Candidate: HF merged loaded")
    return model


def forward_moe_block(model, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    """Run a single MoE block isolated from attention/positional effects."""
    # PEFT-wrapped models: traverse via .base_model or .model
    if hasattr(model, "base_model"):
        layers = model.base_model.model.model.layers
    else:
        layers = model.model.layers
    layer = layers[layer_idx]
    with torch.no_grad():
        normed = layer.post_attention_layernorm(hidden)
        out = layer.mlp(normed)
    if isinstance(out, tuple):
        out = out[0]
    return out


def main() -> int:
    print("[ANCHOR] Loading reference model ...")
    ref_model, ref_type = load_ref_model()
    print("[ANCHOR] Loading candidate model ...")
    cand_model = load_cand_model()

    all_results = []
    any_fail = False
    t0 = time.time()

    for layer_idx in TEST_LAYERS:
        print(f"\n[Layer {layer_idx}]")
        for seed in SEEDS:
            hidden = make_hidden(seed)
            ref_out = forward_moe_block(ref_model, layer_idx, hidden)
            cand_out = forward_moe_block(cand_model, layer_idx, hidden)
            label = f"L{layer_idx}_seed{seed}"
            r = compare_outputs(ref_out, cand_out, label)
            all_results.append(r)
            if not r["overall_pass"]:
                any_fail = True

    anchor_pass = not any_fail
    anchor_status = "pass" if anchor_pass else "fail"
    # Council gate: promotable ONLY if numeric pass AND reference was the real Unsloth
    # runtime. HF+PEFT fallback is diagnostic only (it shares the same suspect merge
    # path, so it cannot certify correctness).
    promotable = anchor_pass and (ref_type == "unsloth_peft")

    print(f"\n{'=' * 60}")
    print(f"ANCHOR RESULT: {'PASS' if anchor_pass else 'FAIL'}  (ref_type={ref_type})")
    print(
        f"  Tests: {len(all_results)}  |  "
        f"Pass: {sum(r['overall_pass'] for r in all_results)}  |  "
        f"Fail: {sum(not r['overall_pass'] for r in all_results)}"
    )
    print(f"  PROMOTABLE: {promotable}")
    print(f"{'=' * 60}")
    if promotable:
        print("  Candidate may be promoted to canonical reasoning merge")
        print("  Proceed to W0-03 smoke gate")
    elif anchor_pass and ref_type != "unsloth_peft":
        print("  NUMERIC PASS but ref was HF fallback — DIAGNOSTIC ONLY, NOT PROMOTABLE")
        print("  Re-run inside Unsloth container for a certifying reference")
    else:
        print("  DO NOT PROMOTE — semantics do not match Unsloth reference")
        print("  Re-evaluate merge math or escalate to GPU Unsloth merge")

    if Path(REPORT_PATH).exists():
        report = json.load(open(REPORT_PATH))
    else:
        report = {}
    report["forward_anchor"] = anchor_status
    report["forward_anchor_ref_type"] = ref_type
    report["forward_anchor_promotable"] = promotable
    report["forward_anchor_results"] = all_results
    report["forward_anchor_elapsed"] = round(time.time() - t0, 1)
    if promotable:
        report["status"] = "candidate_promotable"
    elif anchor_pass:
        report["status"] = "candidate_anchor_passed_diagnostic_only"
    else:
        report["status"] = "candidate_anchor_failed"
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(REPORT_PATH, "w"), indent=2)
    print(f"\n  Report updated: {REPORT_PATH}")
    return 0 if promotable else 1


if __name__ == "__main__":
    sys.exit(main())
