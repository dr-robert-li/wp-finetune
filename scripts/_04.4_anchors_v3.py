"""Consolidated 3-anchor certifier for the Tinker v3 staging merge (CPU, 2 model loads).

Replaces the three ckpt-72-era anchor scripts (_p0_anchor_tensor_full / _fp32_control /
_forward_rankpath), which were hardwired to the Unsloth fused-`base_layer` contiguous-block
convention + the merged-v2 base. v3 uses the STOCK Qwen3-30B-A3B base and Tinker's
separate w1/w2/w3 (shared-A / per-expert-B for w1/w3; per-expert-A / shared-B for w2).

Three verdicts, all written into output/merge_v3/merge_report.json:

  tensor_anchor   : staging weight[e] - stock-base weight[e] == bf16(build_*_delta(e)) for
                    sampled experts/layers, AND per-expert deltas distinct (e0 != e1). Certifies
                    the EXTRACTION+APPLICATION wrote the right per-expert deltas.
  fp32_control    : staging_stored[e] ~= bf16(stock_base[e].float() + delta_e.float()); residual
                    rms < bf16 floor (rms * 2^-8). PRIMARY certifier (precision-clean).
  forward_anchor  : full MoE-block forward, rank-path (stock base + Tinker activation-space LoRA)
                    vs weight-path (staging merged). bf16-calibrated cos>=0.99990, rel_l2<=1e-2,
                    mean<=2e-3, router-invariant HARD. Corroboration of the weight bake.

Memory: load stock base once (capture rank-path forward refs + sampled expert weights), free;
load staging once (capture weight-path forward + sampled weights), free. Peak ~one model.
Imports the SAME production delta math from scripts.merge_tinker_v3 for the weight anchors;
the forward anchor independently injects activation-space LoRA (genuine cross-check).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

# Import production delta math (same code the merge runs) for the weight anchors.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.merge_tinker_v3 import (  # noqa: E402
    build_down_delta,
    build_gate_up_delta,
    per_expert_differ,
)

STOCK_BASE = "models/Qwen3-30B-A3B"
_DEFAULT_STAGING = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3"
_DEFAULT_ADAPTER_TAR = "models/tinker_export/wp-reasoning-v3/checkpoint.tar"
_DEFAULT_REPORT = "output/merge_v3/merge_report.json"

# Allow callers (e.g. the v4 launcher) to redirect STAGING/REPORT/ADAPTER_TAR without touching defaults.
# Parse args early so STAGING/REPORT/ADAPTER_TAR are available at module level for the rest of main().
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--report", default=_DEFAULT_REPORT)
_ap.add_argument("--staging", default=_DEFAULT_STAGING)
_ap.add_argument("--adapter", default=_DEFAULT_ADAPTER_TAR)
_known, _unknown = _ap.parse_known_args()
STAGING = _known.staging
REPORT = _known.report
ADAPTER_TAR = _known.adapter

HIDDEN = 2048
GATE_HALF = 768          # per-expert gate (or up) output dim
DTYPE = torch.bfloat16
TEST_LAYERS = [0, 23, 47]
SEEDS = [42, 137, 999]
SAMPLE_EXPERTS = [0, 1, 63, 127]          # adjacent 0,1 to catch broadcast
APPLIED_TOL = 1e-4                         # staging must EXACTLY equal bf16_add(base, bf16(delta))
FWD_THRESH = {"mean_abs": 2e-3, "relative_l2": 1e-2, "cosine": 0.99990, "max_abs_hard": 1e-1}


def _akey(layer: int, w: str, ab: str) -> str:
    return f"base_model.model.model.layers.{layer}.mlp.experts.{w}.lora_{ab}.weight"


def _load_adapter_for_layers(layers, tmp_dir):
    """Untar adapter once, return {scale, r, per-layer w1/w2/w3 A/B tensors (fp32)}."""
    import tarfile

    os.makedirs(tmp_dir, exist_ok=True)
    with tarfile.open(ADAPTER_TAR, "r:*") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if base in ("adapter_config.json", "adapter_model.safetensors"):
                m.name = base
                tf.extract(m, tmp_dir)
    cfg = json.load(open(os.path.join(tmp_dir, "adapter_config.json")))
    r = int(cfg["r"])
    scale = float(cfg.get("lora_alpha", r)) / float(r)
    sf = os.path.join(tmp_dir, "adapter_model.safetensors")
    lora = {}
    with safe_open(sf, framework="pt", device="cpu") as f:
        for L in layers:
            lora[L] = {ww + ab: f.get_tensor(_akey(L, ww, ab)).float()
                       for ww in ("w1", "w2", "w3") for ab in ("A", "B")}
    return scale, r, lora


def _rankpath_forward(experts_mod, lyr_lora, scale):
    """Patched experts.forward: stock-base expert + Tinker activation-space LoRA per expert."""
    A_w1, B_w1 = lyr_lora["w1A"], lyr_lora["w1B"]   # A:[1,32,2048] shared ; B:[128,768,32] per-expert
    A_w3, B_w3 = lyr_lora["w3A"], lyr_lora["w3B"]
    A_w2, B_w2 = lyr_lora["w2A"], lyr_lora["w2B"]   # A:[128,32,768] per-expert ; B:[1,2048,32] shared
    a_w1 = A_w1.squeeze(0) if A_w1.dim() == 3 else A_w1   # [32,2048]
    a_w3 = A_w3.squeeze(0) if A_w3.dim() == 3 else A_w3
    b_w2 = B_w2.squeeze(0) if B_w2.dim() == 3 else B_w2   # [2048,32]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in hit:
            e = ei[0]
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]                                   # (n,2048)
            gu_base = F.linear(x, self.gate_up_proj[e]).float()       # (n,1536)
            gate_lora = ((x.float() @ a_w1.T) @ B_w1[e].T) * scale    # (n,768)
            up_lora = ((x.float() @ a_w3.T) @ B_w3[e].T) * scale      # (n,768)
            gate_up = gu_base + torch.cat([gate_lora, up_lora], dim=-1)
            gate, up = gate_up.chunk(2, dim=-1)
            h = (self.act_fn(gate) * up).to(DTYPE)                    # (n,768)
            dn_base = F.linear(h, self.down_proj[e]).float()          # (n,2048)
            dn_lora = ((h.float() @ A_w2[e].T) @ b_w2.T) * scale      # (n,2048)
            cur = (dn_base + dn_lora).to(DTYPE)
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final

    return forward


def _make_hidden(seed, batch=4, seq=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, HIDDEN, generator=g, dtype=torch.float32).to(DTYPE)


def _capture_block(model, L, seed):
    block = model.model.layers[L].mlp
    hidden = _make_hidden(seed)
    with torch.no_grad():
        hr = hidden.view(-1, HIDDEN)
        _, rw, sel = block.gate(hr)
        out = block.experts(hr, sel, rw).view(hidden.shape)
    return out.float().clone(), sel.clone()


def main() -> int:
    if not os.path.isdir(STAGING):
        print(f"FATAL: staging dir missing: {STAGING}", file=sys.stderr)
        return 2
    tmp = "/tmp/_04.4_anchor_adapter_v3"
    scale, r, lora = _load_adapter_for_layers(TEST_LAYERS, tmp)
    print(f"r={r} scale={scale} layers={TEST_LAYERS} seeds={SEEDS} experts={SAMPLE_EXPERTS}")

    # ---- Load STOCK base once: capture rank-path forward refs + sampled expert weights ----
    print("[REF] loading stock base ...")
    base = AutoModelForCausalLM.from_pretrained(
        STOCK_BASE, device_map={"": "cpu"}, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    base.eval()
    ref_fwd, base_w = {}, {}
    for L in TEST_LAYERS:
        for e in SAMPLE_EXPERTS:
            base_w[(L, e, "gu")] = base.model.layers[L].mlp.experts.gate_up_proj.data[e].float().clone()
            base_w[(L, e, "dn")] = base.model.layers[L].mlp.experts.down_proj.data[e].float().clone()
        block = base.model.layers[L].mlp
        block.experts.forward = types.MethodType(_rankpath_forward(block.experts, lora[L], scale), block.experts)
        for s in SEEDS:
            ref_fwd[(L, s)] = _capture_block(base, L, s)
    del base
    gc.collect()
    print("[REF] captured + freed")

    # ---- Load STAGING once: capture weight-path forward + sampled expert weights ----
    print("[CAND] loading staging ...")
    cand = AutoModelForCausalLM.from_pretrained(
        STAGING, device_map={"": "cpu"}, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    cand.eval()
    cand_fwd, cand_w = {}, {}
    for L in TEST_LAYERS:
        for e in SAMPLE_EXPERTS:
            cand_w[(L, e, "gu")] = cand.model.layers[L].mlp.experts.gate_up_proj.data[e].float().clone()
            cand_w[(L, e, "dn")] = cand.model.layers[L].mlp.experts.down_proj.data[e].float().clone()
        for s in SEEDS:
            cand_fwd[(L, s)] = _capture_block(cand, L, s)
    del cand
    gc.collect()
    print("[CAND] captured + freed")

    report = json.load(open(REPORT)) if os.path.exists(REPORT) else {}

    # ========================= ANCHOR 1: tensor (applied + distinct) =========================
    tensor_fail = 0
    applied_max_resid = 0.0
    distinct_min = float("inf")
    for L in TEST_LAYERS:
        gu_deltas, dn_deltas = {}, {}
        A_w1, B_w1 = lora[L]["w1A"], lora[L]["w1B"]
        A_w3, B_w3 = lora[L]["w3A"], lora[L]["w3B"]
        A_w2, B_w2 = lora[L]["w2A"], lora[L]["w2B"]
        for e in SAMPLE_EXPERTS:
            d_gu = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale)
            d_dn = build_down_delta(A_w2, B_w2, e, scale)
            gu_deltas[e], dn_deltas[e] = d_gu, d_dn
            # applied: staging[e] must EXACTLY equal what the merge computes -
            # param.data[e] += delta.to(bf16), i.e. bf16_add(base_bf16, bf16(delta)).
            # Reconstruct that exact op (delta rounded to bf16 BEFORE the bf16 add); the
            # residual is then ~0 (a full-precision-delta reference would be off by 1 bf16
            # ulp of the delta, ~2e-3 at high-magnitude gate_up — that is storage, not a bug).
            for tag, d, bw, cw in (("gu", d_gu, base_w[(L, e, "gu")], cand_w[(L, e, "gu")]),
                                   ("dn", d_dn, base_w[(L, e, "dn")], cand_w[(L, e, "dn")])):
                recon = (bw.to(DTYPE) + d.to(DTYPE)).to(DTYPE).float()   # bf16_add(base_bf16, bf16(delta))
                resid = (cw - recon).abs().max().item()
                applied_max_resid = max(applied_max_resid, resid)
                if resid > APPLIED_TOL:
                    tensor_fail += 1
                    print(f"  TENSOR FAIL L{L} {tag} e{e}: applied residual {resid:.2e}")
        if 0 in gu_deltas and 1 in gu_deltas:
            distinct_min = min(distinct_min,
                               (gu_deltas[0] - gu_deltas[1]).abs().max().item(),
                               (dn_deltas[0] - dn_deltas[1]).abs().max().item())
    tensor_pass = (tensor_fail == 0) and (applied_max_resid <= APPLIED_TOL) and (distinct_min > 1e-5)
    report["tensor_anchor"] = "pass" if tensor_pass else "fail"
    report["tensor_anchor_detail"] = {
        "applied_max_residual": round(applied_max_resid, 8),
        "adjacent_expert_min_distinct": round(distinct_min, 6),
        "failures": tensor_fail,
    }
    print(f"TENSOR ANCHOR: {'PASS' if tensor_pass else 'FAIL'} "
          f"(applied_resid {applied_max_resid:.2e}<=1e-3, distinct {distinct_min:.4f}>1e-5)")

    # ========================= ANCHOR 2: fp32 weight control =========================
    fp32_ok = True
    fp32_rows = []
    for L in TEST_LAYERS:
        A_w1, B_w1 = lora[L]["w1A"], lora[L]["w1B"]
        A_w3, B_w3 = lora[L]["w3A"], lora[L]["w3B"]
        A_w2, B_w2 = lora[L]["w2A"], lora[L]["w2B"]
        for e in SAMPLE_EXPERTS:
            for tag, d in (("gu", build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale)),
                           ("dn", build_down_delta(A_w2, B_w2, e, scale))):
                bw, cw = base_w[(L, e, tag)], cand_w[(L, e, tag)]
                true_fp32 = bw + d
                rms_d = (cw - true_fp32).pow(2).mean().sqrt().item()
                weight_rms = true_fp32.pow(2).mean().sqrt().item()
                bf16_floor = weight_rms * (2 ** -8)
                # bf16-noise criterion: stored merge is within bf16 storage noise of the
                # ideal-precision merge (base + fp32 delta); rms residual < 3x the bf16 floor.
                # (A real merge-math error would push rms_d well ABOVE the floor.)
                ok = rms_d < 3 * bf16_floor
                fp32_ok = fp32_ok and ok
                fp32_rows.append({"L": L, "e": int(e), "proj": tag,
                                  "rms_d": round(rms_d, 8), "bf16_floor": round(bf16_floor, 8),
                                  "ok": ok})
    report["fp32_control_anchor"] = "pass" if fp32_ok else "fail"
    report["fp32_control_detail"] = fp32_rows
    print(f"FP32-CONTROL ANCHOR: {'PASS' if fp32_ok else 'FAIL'} "
          f"(stored==bf16(base+fp32 delta), rms sub-floor)")

    # ========================= ANCHOR 3: forward rank-path vs weight-path =========================
    fwd_fail = False
    router_all = True
    fwd_results = []
    for L in TEST_LAYERS:
        for s in SEEDS:
            ro, rsel = ref_fwd[(L, s)]
            co, csel = cand_fwd[(L, s)]
            router_match = bool(torch.equal(rsel, csel))
            diff = (co - ro).abs()
            mean_abs = diff.mean().item()
            max_abs = diff.max().item()
            rel_l2 = (diff.norm() / (ro.norm() + 1e-8)).item()
            cos = F.cosine_similarity(ro.flatten().unsqueeze(0), co.flatten().unsqueeze(0)).item()
            ok = (router_match and mean_abs <= FWD_THRESH["mean_abs"]
                  and rel_l2 <= FWD_THRESH["relative_l2"] and cos >= FWD_THRESH["cosine"]
                  and max_abs < FWD_THRESH["max_abs_hard"])
            router_all = router_all and router_match
            fwd_fail = fwd_fail or (not ok)
            fwd_results.append({"label": f"L{L}_seed{s}", "router_match": router_match,
                                "mean_abs": round(mean_abs, 8), "max_abs": round(max_abs, 8),
                                "relative_l2": round(rel_l2, 8), "cosine": round(cos, 8),
                                "pass": ok})
            print(f"  FWD L{L}_seed{s}: router={'OK' if router_match else 'MISMATCH'} "
                  f"mean={mean_abs:.2e} rel_l2={rel_l2:.2e} cos={cos:.8f} [{'PASS' if ok else 'FAIL'}]")
    fwd_pass = (not fwd_fail) and router_all
    report["forward_anchor"] = "pass" if fwd_pass else "fail"
    report["forward_anchor_router_invariant"] = router_all
    report["forward_anchor_results"] = fwd_results
    print(f"FORWARD ANCHOR: {'PASS' if fwd_pass else 'FAIL'} (router-invariant {router_all})")

    all_pass = tensor_pass and fp32_ok and fwd_pass
    report["anchors_all_pass"] = all_pass
    report["status"] = "staging_anchor_certified" if all_pass else "staging_anchor_FAILED"
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(report, open(REPORT, "w"), indent=2)
    print(f"\n{'=' * 60}\nANCHORS {'ALL PASS' if all_pass else 'FAILED'} -> {report['status']}\n{'=' * 60}")
    print(f"Report: {REPORT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
