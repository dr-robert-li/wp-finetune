"""Wave-0 unit test: Tinker per-expert MoE delta math (locks the merge convention).

Imports the EXACT production functions from scripts.merge_tinker_v3 (no copy-pasted
math). Asserts per-expert deltas are DISTINCT across experts for w1/w2/w3 -- a broadcast
merge (the cos_sim-0.08 ckpt-72-era bug) makes them equal and MUST fail these asserts.

NOTE (2026-06-07): the obsolete 04.4-RESEARCH claim "w1/w3 shared-A => same delta per
expert" is FALSE -- only lora_A is shared; lora_B is per-expert, so delta_e differs.
An across-expert equality assertion (e0 equal to e1) would pass a broadcast bug;
we assert the deltas DIFFER.
"""
import torch

from scripts.merge_tinker_v3 import (
    build_down_delta,
    build_gate_up_delta,
    per_expert_differ,
)

R = 32
HIDDEN = 2048
MLP = 768          # per-expert gate/up/down inner dim
GATE_UP_OUT = 1536  # 2 * MLP
E = 4              # sample a handful of experts (real model has 128)


def _w1w3_factors():
    torch.manual_seed(0)
    A_w1 = torch.randn(1, R, HIDDEN)      # SHARED gate lora_A
    A_w3 = torch.randn(1, R, HIDDEN)      # SHARED up lora_A
    B_w1 = torch.randn(E, MLP, R)         # PER-EXPERT gate lora_B
    B_w3 = torch.randn(E, MLP, R)         # PER-EXPERT up lora_B
    return A_w1, B_w1, A_w3, B_w3


def _w2_factors():
    torch.manual_seed(1)
    A_w2 = torch.randn(E, R, MLP)         # PER-EXPERT down lora_A
    B_w2 = torch.randn(1, HIDDEN, R)      # SHARED down lora_B
    return A_w2, B_w2


def test_gate_up_delta_shape_is_1536x2048():
    A_w1, B_w1, A_w3, B_w3 = _w1w3_factors()
    d = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e=0)
    assert tuple(d.shape) == (GATE_UP_OUT, HIDDEN), tuple(d.shape)


def test_gate_up_is_gate_first_concat():
    """Top 768 rows == gate delta (B_w1[e]@A_w1); bottom 768 == up delta (B_w3[e]@A_w3)."""
    A_w1, B_w1, A_w3, B_w3 = _w1w3_factors()
    e = 2
    out = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e=e)
    A1 = A_w1.squeeze(0).float()
    A3 = A_w3.squeeze(0).float()
    delta_gate = B_w1[e].float() @ A1
    delta_up = B_w3[e].float() @ A3
    assert torch.allclose(out[:MLP], delta_gate, atol=1e-5)
    assert torch.allclose(out[MLP:], delta_up, atol=1e-5)


def test_gate_up_delta_differs_per_expert():
    """w1/w3: per-expert lora_B => delta_e0 != delta_e1 (broadcast would make them equal)."""
    A_w1, B_w1, A_w3, B_w3 = _w1w3_factors()
    d0 = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e=0)
    d1 = build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e=1)
    assert (d0 - d1).abs().max().item() > 1e-5


def test_down_delta_shape_and_differs_per_expert():
    """w2: per-expert lora_A => delta_e0 != delta_e1; shape [2048,768]."""
    A_w2, B_w2 = _w2_factors()
    d0 = build_down_delta(A_w2, B_w2, e=0)
    d1 = build_down_delta(A_w2, B_w2, e=1)
    assert tuple(d0.shape) == (HIDDEN, MLP), tuple(d0.shape)
    assert (d0 - d1).abs().max().item() > 1e-5


def test_per_expert_differ_accepts_distinct_rejects_broadcast():
    A_w1, B_w1, A_w3, B_w3 = _w1w3_factors()
    distinct = [build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e=e) for e in range(E)]
    assert per_expert_differ(distinct) > 1e-5

    # Broadcast bug: same delta replicated across experts -> differ ~ 0.
    broadcast = [distinct[0].clone() for _ in range(E)]
    assert per_expert_differ(broadcast) <= 1e-6


def test_scale_is_applied_linearly():
    A_w2, B_w2 = _w2_factors()
    d1 = build_down_delta(A_w2, B_w2, e=0, scale=1.0)
    d2 = build_down_delta(A_w2, B_w2, e=0, scale=2.0)
    assert torch.allclose(d2, 2.0 * d1, atol=1e-5)
