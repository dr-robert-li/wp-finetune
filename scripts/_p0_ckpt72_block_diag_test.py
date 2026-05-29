"""Council-mandated disambiguation test for ckpt-72 reasoning adapter.

Hypothesis under test:
    Unsloth fused-experts MoE LoRA represents per-expert deltas via
    rank-block-diagonal structure in (lora_A, lora_B). Specifically:
      lora_A.shape = (r*E, D_in)  reshaped as (E, r, D_in)
      lora_B.shape = (D_out, r*E) reshaped as (D_out, E, r)
    where E = num_experts = 128, r = lora_rank = 32.

    For each expert e, the per-expert delta is:
      delta[e] = lora_B[:, e, :] @ lora_A[e, :, :]   shape: (D_out, D_in)

    The naive 2D product `lora_B @ lora_A` equals
      sum_{e1,e2} lora_B[:, e1, :] @ lora_A[e2, :, :]
    Cross terms (e1 != e2) measure "cross-expert leakage".

Lossless Path (I) expansion requires cross terms ~ 0 (block-diagonal).
Dense cross-mixing (cross terms ~ diagonal magnitude) means the rank
dimension does NOT correspond to per-expert indexing — Path (I) invalid.

Decision rule:
    cross_to_diag_ratio = mean_off_diag_norm / mean_diag_norm
    if ratio < 0.05:   BLOCK-DIAGONAL — Path (I) lossless
    if ratio > 0.5:    DENSE CROSS-MIX — Path (I) invalid
    else:              INTERMEDIATE — additional probes needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open

ADAPTER = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72/adapter_model.safetensors"
NUM_EXPERTS = 128
LORA_RANK = 32
SAMPLE_LAYERS = [0, 12, 24, 36, 47]


def load_pair(f, layer: int, suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (lora_A, lora_B) pair for one layer + suffix."""
    A = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.{suffix}lora_A.weight").float()
    B = f.get_tensor(f"base_model.model.model.layers.{layer}.mlp.experts.{suffix}lora_B.weight").float()
    return A, B


def block_diag_stats(
    A: torch.Tensor, B: torch.Tensor, E: int, r: int, off_diag_sample_n: int = 256
) -> dict:
    """Compute diagonal vs off-diagonal block norms.

    A shape: (E*r, D_in) → reshape (E, r, D_in)
    B shape: (D_out, E*r) → reshape (D_out, E, r)
    Diagonal: B[:, e, :] @ A[e, :, :]
    Off-diag: B[:, e1, :] @ A[e2, :, :] for e1 != e2
    """
    assert A.shape[0] == E * r, f"A rows {A.shape[0]} != E*r {E*r}"
    assert B.shape[1] == E * r, f"B cols {B.shape[1]} != E*r {E*r}"
    D_in = A.shape[1]
    D_out = B.shape[0]

    A_r = A.view(E, r, D_in)       # (E, r, D_in)
    B_r = B.view(D_out, E, r)      # (D_out, E, r)

    # Diagonal block norms (per expert)
    diag_norms = torch.zeros(E)
    for e in range(E):
        d = B_r[:, e, :] @ A_r[e, :, :]   # (D_out, D_in)
        diag_norms[e] = d.norm().item()

    # Sample off-diagonal cross-term norms
    torch.manual_seed(42)
    pairs = []
    while len(pairs) < off_diag_sample_n:
        e1 = int(torch.randint(0, E, (1,)).item())
        e2 = int(torch.randint(0, E, (1,)).item())
        if e1 != e2:
            pairs.append((e1, e2))
    off_norms = torch.zeros(len(pairs))
    for i, (e1, e2) in enumerate(pairs):
        d = B_r[:, e1, :] @ A_r[e2, :, :]
        off_norms[i] = d.norm().item()

    # Full naive product norm (sanity check)
    full = B @ A
    full_norm = full.norm().item()

    # Sum of diagonals
    diag_sum = torch.zeros(D_out, D_in)
    for e in range(E):
        diag_sum += B_r[:, e, :] @ A_r[e, :, :]
    diag_sum_norm = diag_sum.norm().item()

    return {
        "D_in": D_in,
        "D_out": D_out,
        "diag_mean": diag_norms.mean().item(),
        "diag_std": diag_norms.std().item(),
        "diag_min": diag_norms.min().item(),
        "diag_max": diag_norms.max().item(),
        "off_mean": off_norms.mean().item(),
        "off_std": off_norms.std().item(),
        "off_min": off_norms.min().item(),
        "off_max": off_norms.max().item(),
        "ratio_offmean_over_diagmean": off_norms.mean().item() / max(diag_norms.mean().item(), 1e-12),
        "full_norm_full_product": full_norm,
        "diag_sum_norm": diag_sum_norm,
        "full_vs_diagsum_diff": (full - diag_sum).norm().item(),
    }


def verdict(ratio: float) -> str:
    if ratio < 0.05:
        return "BLOCK-DIAGONAL — Path (I) lossless expansion valid"
    if ratio > 0.5:
        return "DENSE CROSS-MIX — Path (I) invalid; fall back to GPU Unsloth or PEFT patch"
    return f"INTERMEDIATE (ratio={ratio:.3f}) — additional probes required"


def main() -> int:
    print(f"Adapter: {ADAPTER}")
    print(f"Layers sampled: {SAMPLE_LAYERS}")
    print(f"E={NUM_EXPERTS}, r={LORA_RANK}, off-diag sample = 256 pairs/layer")
    print()

    suffixes = [
        ("base_layer.", "base_layer (fused gate_up_proj)"),
        ("",            "direct (fused down_proj)"),
    ]

    with safe_open(ADAPTER, framework="pt", device="cpu") as f:
        for suffix, label in suffixes:
            print(f"=== {label} ===")
            ratios = []
            for layer in SAMPLE_LAYERS:
                A, B = load_pair(f, layer, suffix)
                s = block_diag_stats(A, B, NUM_EXPERTS, LORA_RANK)
                ratios.append(s["ratio_offmean_over_diagmean"])
                print(
                    f"  layer {layer:>2}: D_in={s['D_in']:>5} D_out={s['D_out']:>5}  "
                    f"diag(mean±std)={s['diag_mean']:.4f}±{s['diag_std']:.4f}  "
                    f"off(mean±std)={s['off_mean']:.4f}±{s['off_std']:.4f}  "
                    f"ratio={s['ratio_offmean_over_diagmean']:.4f}  "
                    f"full-vs-Σdiag={s['full_vs_diagsum_diff']:.6f}"
                )
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  → AVG ratio across {len(SAMPLE_LAYERS)} layers: {avg_ratio:.4f}")
            print(f"  → VERDICT: {verdict(avg_ratio)}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
