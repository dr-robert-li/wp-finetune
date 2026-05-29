"""Council-mandated disambiguation test v2 — CORRECTED indexing per PEFT source.

PEFT 0.18.1 ParamWrapper.get_delta_weight (peft/tuners/lora/layer.py:2074-2090):
    weight_A.reshape(E, r, in_features)      # outer=expert, inner=rank
    weight_B.reshape(out_features, r, E)     # outer=rank,   inner=expert
    delta = einsum("o r e, e r i -> e i o", B, A) * scaling
    # result: (E, in, out)

Earlier test slicing convention (WRONG for B):
    A[e*r:(e+1)*r, :]   ✓
    B[:, e*r:(e+1)*r]   ✗   ← should be B[:, e::E]

This v2 uses correct PEFT indexing. Block-diagonal test:
- diag norm:   ||B[:, e::E] @ A[e*r:(e+1)*r, :]||_F
- off-diag:    ||B[:, e1::E] @ A[e2*r:(e2+1)*r, :]||_F  for e1 != e2

If off-diag << diag → block-diagonal → PEFT einsum computes per-expert deltas correctly.
"""

from __future__ import annotations

import sys

import torch
from safetensors import safe_open

ADAPTER = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
NUM_EXPERTS = 128
LORA_RANK = 32
SAMPLE_LAYERS = [0, 12, 24, 36, 47]


def load_pair(f, layer: int, suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
    A = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.{suffix}lora_A.weight").float()
    B = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.{suffix}lora_B.weight").float()
    return A, B


def block_diag_stats_peft(
    A: torch.Tensor, B: torch.Tensor, E: int, r: int, off_diag_sample_n: int = 256
) -> dict:
    """PEFT-correct slicing:
        A: (E*r, in)    →  expert e block is A[e*r:(e+1)*r, :]
        B: (out, r*E)   →  expert e block is B[:, e::E]
    """
    assert A.shape[0] == E * r
    assert B.shape[1] == E * r
    in_features = A.shape[1]
    out_features = B.shape[0]

    def A_e(e: int) -> torch.Tensor:
        return A[e * r:(e + 1) * r, :]     # (r, in)

    def B_e(e: int) -> torch.Tensor:
        return B[:, e::E]                  # (out, r)

    diag_norms = torch.zeros(E)
    for e in range(E):
        delta_e = B_e(e) @ A_e(e)          # (out, in)
        diag_norms[e] = delta_e.norm().item()

    torch.manual_seed(42)
    pairs = []
    while len(pairs) < off_diag_sample_n:
        e1 = int(torch.randint(0, E, (1,)).item())
        e2 = int(torch.randint(0, E, (1,)).item())
        if e1 != e2:
            pairs.append((e1, e2))
    off_norms = torch.zeros(len(pairs))
    for i, (e1, e2) in enumerate(pairs):
        d = B_e(e1) @ A_e(e2)
        off_norms[i] = d.norm().item()

    # Sanity: einsum sum-of-diagonals should equal sum_e B_e(e) @ A_e(e)
    A_r = A.view(E, r, in_features)
    B_r = B.view(out_features, r, E)
    delta_einsum = torch.einsum("o r e, e r i -> e i o", B_r, A_r)  # (E, in, out)
    delta_loop = torch.stack([B_e(e) @ A_e(e) for e in range(E)], dim=0)  # (E, out, in)
    # Verify sanity: einsum output transposed should equal loop
    einsum_vs_loop_diff = (delta_einsum.transpose(1, 2) - delta_loop).norm().item()

    return {
        "in_features": in_features,
        "out_features": out_features,
        "diag_mean": diag_norms.mean().item(),
        "diag_std":  diag_norms.std().item(),
        "diag_min":  diag_norms.min().item(),
        "diag_max":  diag_norms.max().item(),
        "off_mean":  off_norms.mean().item(),
        "off_std":   off_norms.std().item(),
        "off_min":   off_norms.min().item(),
        "off_max":   off_norms.max().item(),
        "ratio":     off_norms.mean().item() / max(diag_norms.mean().item(), 1e-12),
        "einsum_vs_loop_diff": einsum_vs_loop_diff,
        "delta_einsum_total_norm": delta_einsum.norm().item(),
    }


def verdict(ratio: float) -> str:
    if ratio < 0.05:
        return "BLOCK-DIAGONAL — PEFT einsum extracts clean per-expert deltas"
    if ratio > 0.5:
        return "DENSE CROSS-MIX — still mixed even under PEFT indexing"
    return f"INTERMEDIATE (ratio={ratio:.3f})"


def main() -> int:
    print(f"Adapter: {ADAPTER}")
    print(f"Layers sampled: {SAMPLE_LAYERS}")
    print(f"E={NUM_EXPERTS}, r={LORA_RANK}, off-diag sample = 256 pairs/layer")
    print(f"Indexing: A[e*r:(e+1)*r, :], B[:, e::E]  (per PEFT 0.18.1 ParamWrapper)")
    print()

    suffixes = [
        ("base_layer.", "base_layer (fused gate_up_proj, in=2048 out=1536)"),
        ("",            "direct (fused down_proj, in=768 out=2048)"),
    ]

    with safe_open(ADAPTER, framework="pt", device="cpu") as f:
        for suffix, label in suffixes:
            print(f"=== {label} ===")
            ratios = []
            for layer in SAMPLE_LAYERS:
                A, B = load_pair(f, layer, suffix)
                s = block_diag_stats_peft(A, B, NUM_EXPERTS, LORA_RANK)
                ratios.append(s["ratio"])
                print(
                    f"  layer {layer:>2}: in={s['in_features']:>4} out={s['out_features']:>4}  "
                    f"diag(mean±std)={s['diag_mean']:.4f}±{s['diag_std']:.4f}  "
                    f"off(mean±std)={s['off_mean']:.4f}±{s['off_std']:.4f}  "
                    f"ratio={s['ratio']:.4f}  "
                    f"einsum-vs-loop={s['einsum_vs_loop_diff']:.2e}"
                )
            avg = sum(ratios) / len(ratios)
            print(f"  → AVG ratio: {avg:.4f}")
            print(f"  → VERDICT: {verdict(avg)}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
