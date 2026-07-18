#!/usr/bin/env python3
"""D-IT-02 cheap static check: per-expert LoRA delta-norm concentration (no GPU, minutes).

WHY (advisor reframe): the Phase-4.3 router was FROZEN, so during judge SFT only the top-k
experts a judge token routed to received gradient. Therefore the adapter's per-expert delta
MAGNITUDE already encodes judge-routing — high-delta experts ARE the judge-heavy experts. No
30B forward needed.

DECISION LOGIC:
- If per-expert deltas are ~UNIFORM across the 128 experts -> no judge-specific subset exists ->
  EVERY expert is perturbed -> codegen (whatever experts it uses) cannot avoid perturbed experts
  -> MoE-subset ablation salvage is DEAD. Remaining RC-B arms: attention-only probe + retrain/accept.
- If CONCENTRATED -> a judge-expert subset is identified -> codegen MIGHT avoid it -> MoE salvage
  is plausible -> proceed to build-and-probe (attention-only + MoE-only behavioral censuses).
- AMBIGUOUS -> skip further proxy analysis, go straight to build-and-probe.

Reuses merge_tinker_v3's exact per-expert extraction convention (Unsloth contiguous-block).
scale (lora_alpha/r) is a constant multiplier -> irrelevant to the cross-expert distribution
SHAPE, so we use scale=1.0 and analyse relative concentration only.

PRE-COMMITTED THRESHOLDS (set before looking at any number):
  UNIFORM (salvage DEAD):   median norm-entropy >= 0.97 AND median top16_mass <= 0.20
  CONCENTRATED (salvage OK): median norm-entropy <= 0.90 OR  median top16_mass >= 0.35
  else AMBIGUOUS -> build-and-probe
(uniform baseline for 128 experts: norm-entropy = 1.0, top16_mass = 16/128 = 0.125)
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.merge_tinker_v3 import (  # noqa: E402
    DEFAULT_ADAPTER_TAR, _untar_adapter, _load_adapter_tensors, _k,
    build_gate_up_delta, build_down_delta,
)

N_EXPERTS = 128
N_LAYERS = 48
TOPK = 16
OUT = ROOT / "output" / "eval_reasoning_v3" / "dit02_expert_delta_norms.json"

# Pre-committed thresholds
UNIFORM_ENTROPY_MIN = 0.97
UNIFORM_TOP16_MAX = 0.20
CONC_ENTROPY_MAX = 0.90
CONC_TOP16_MIN = 0.35


def _layer_expert_norms(t: dict, L: int) -> list[float]:
    A_w1, B_w1 = t[_k(L, "w1", "A")], t[_k(L, "w1", "B")]
    A_w3, B_w3 = t[_k(L, "w3", "A")], t[_k(L, "w3", "B")]
    A_w2, B_w2 = t[_k(L, "w2", "A")], t[_k(L, "w2", "B")]
    norms = []
    for e in range(N_EXPERTS):
        gu = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale=1.0)
        dn = build_down_delta(A_w2, B_w2, e, scale=1.0)
        norms.append(math.sqrt(float(gu.pow(2).sum()) + float(dn.pow(2).sum())))
    return norms


def _entropy_norm(norms: list[float]) -> float:
    s = sum(norms)
    if s <= 0:
        return 1.0
    ps = [n / s for n in norms if n > 0]
    H = -sum(p * math.log(p) for p in ps)
    return H / math.log(N_EXPERTS)  # normalized to [0,1]; 1.0 = uniform


def _topk_mass(norms: list[float], k: int) -> float:
    s = sum(norms)
    if s <= 0:
        return 0.0
    return sum(sorted(norms, reverse=True)[:k]) / s


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dit02_delta_norms_")
    adapter_dir = _untar_adapter(DEFAULT_ADAPTER_TAR, tmp)
    t = _load_adapter_tensors(adapter_dir)

    per_layer = []
    for L in range(N_LAYERS):
        try:
            norms = _layer_expert_norms(t, L)
        except KeyError as e:
            print(f"layer {L}: missing key {e}; skipping", file=sys.stderr)
            continue
        mean = statistics.fmean(norms)
        std = statistics.pstdev(norms)
        ent = _entropy_norm(norms)
        t16 = _topk_mass(norms, TOPK)
        per_layer.append({
            "layer": L, "mean": mean, "std": std,
            "cv": (std / mean if mean else 0.0),
            "max_over_mean": (max(norms) / mean if mean else 0.0),
            "norm_entropy": ent, "top16_mass": t16,
        })
        print(f"L{L:02d}: entropy={ent:.4f} top16={t16:.4f} cv={std/mean if mean else 0:.3f} "
              f"max/mean={max(norms)/mean if mean else 0:.2f}", flush=True)

    med_ent = statistics.median(p["norm_entropy"] for p in per_layer)
    med_t16 = statistics.median(p["top16_mass"] for p in per_layer)
    med_cv = statistics.median(p["cv"] for p in per_layer)

    if med_ent >= UNIFORM_ENTROPY_MIN and med_t16 <= UNIFORM_TOP16_MAX:
        verdict = "UNIFORM"
        decision = ("MoE-subset salvage DEAD — every expert perturbed, codegen cannot avoid "
                    "perturbed experts. RC-B arms left: attention-only probe + retrain/accept.")
    elif med_ent <= CONC_ENTROPY_MAX or med_t16 >= CONC_TOP16_MIN:
        verdict = "CONCENTRATED"
        decision = ("Judge-expert subset exists — codegen MIGHT avoid it. MoE-subset salvage "
                    "PLAUSIBLE. Proceed to build-and-probe (attention-only + MoE-only censuses).")
    else:
        verdict = "AMBIGUOUS"
        decision = ("Proxy inconclusive — skip further proxy analysis, go straight to "
                    "build-and-probe (behavioral truth on both arms).")

    result = {
        "analysis": "per_expert_lora_delta_norm_concentration",
        "adapter": DEFAULT_ADAPTER_TAR,
        "router_frozen": True,
        "n_experts": N_EXPERTS, "n_layers_analyzed": len(per_layer), "topk": TOPK,
        "uniform_baseline": {"norm_entropy": 1.0, "top16_mass": TOPK / N_EXPERTS},
        "thresholds": {
            "uniform_entropy_min": UNIFORM_ENTROPY_MIN, "uniform_top16_max": UNIFORM_TOP16_MAX,
            "conc_entropy_max": CONC_ENTROPY_MAX, "conc_top16_min": CONC_TOP16_MIN,
        },
        "median_norm_entropy": med_ent,
        "median_top16_mass": med_t16,
        "median_cv": med_cv,
        "verdict": verdict,
        "decision": decision,
        "per_layer": per_layer,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"median norm-entropy={med_ent:.4f} (uniform=1.0) | median top16_mass={med_t16:.4f} "
          f"(uniform={TOPK/N_EXPERTS:.4f}) | median cv={med_cv:.3f}")
    print(decision)
    print(f"Wrote: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
