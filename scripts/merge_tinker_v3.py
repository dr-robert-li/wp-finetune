"""Tinker-convention MoE LoRA merge for wp-reasoning-v3 -> stock Qwen3-30B-A3B.

This module implements Tinker's THIRD distinct MoE tensor convention (distinct from
PEFT strided `B[:, e::E]` and from Unsloth contiguous-block `B[:, e*R:(e+1)*R]`).

Tinker layout (VERIFIED by tensor inspection of
models/tinker_export/wp-reasoning-v3/checkpoint.tar, 2026-06-07):

  w1 (gate_proj): lora_A [1,32,2048] SHARED      ; lora_B [128,768,32] PER-EXPERT
  w2 (down_proj): lora_A [128,32,768] PER-EXPERT ; lora_B [1,2048,32]  SHARED
  w3 (up_proj):   lora_A [1,32,2048] SHARED      ; lora_B [128,768,32] PER-EXPERT
  unembed:        lora_A [32,2048]               ; lora_B [151936,32]  (STOCK vocab)

Per-expert delta math (scale = lora_alpha / r):
  delta_gate_e = B_w1[e] @ A_w1            -> [768,2048]
  delta_up_e   = B_w3[e] @ A_w3            -> [768,2048]
  gate_up[e]  += cat([delta_gate_e, delta_up_e], dim=0)  -> [1536,2048]  (gate FIRST)
  delta_down_e = B_w2 @ A_w2[e]            -> [2048,768]
  down[e]     += delta_down_e
  lm_head     += (B_unembed @ A_unembed)   -> [151936,2048]

EMPIRICALLY VERIFIED per-expert distinctness (this plan's research, 2026-06-07):
  w1.lora_B[0] vs [1] max_diff = 0.049 ; w2.lora_A[0] vs [1] max_diff = 0.049.
  => ALL of w1/w2/w3 produce PER-EXPERT-DISTINCT deltas. A test asserting
  delta_e0 == delta_e1 (the obsolete 04.4-RESEARCH "shared-A => same delta" claim)
  would PASS a broadcast bug and FAIL a correct merge -- do NOT use it.

NO model load and NO file IO happen at import: the delta builders + fidelity helpers
are pure tensor/logic functions so tests/phase4_4/test_tinker_merge_convention.py and
tests/phase4_4/test_fidelity_protocol.py import the exact production code. The CLI merge
(stock base load, per-expert apply, manual lm_head, staging save, merge_report) lives
under `if __name__ == "__main__"` and is added in Task 2.

Plan 02 import path (fidelity gate consumes these):
  from scripts.merge_tinker_v3 import sentinel_agreement, spearman_agree
"""

from __future__ import annotations

from typing import Sequence

import torch


# --------------------------------------------------------------------------- #
# Per-expert LoRA delta builders (pure functions; no IO, no model load).
# --------------------------------------------------------------------------- #

def _squeeze_shared(t: torch.Tensor) -> torch.Tensor:
    """Squeeze a leading singleton dim off a SHARED factor: [1,R,in] -> [R,in]."""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def build_gate_up_delta(
    A_w1: torch.Tensor,
    B_w1: torch.Tensor,
    A_w3: torch.Tensor,
    B_w3: torch.Tensor,
    e: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused gate_up delta for expert `e` -> [1536, 2048] (gate rows first, up rows second).

    A_w1/A_w3 are the SHARED gate/up lora_A ([1,32,2048] or [32,2048]).
    B_w1/B_w3 are the PER-EXPERT lora_B ([128,768,32]); expert slice is B[e] -> [768,32].
    delta_gate = B_w1[e] @ A_w1 ; delta_up = B_w3[e] @ A_w3 ; cat on dim 0, gate first.
    """
    A1 = _squeeze_shared(A_w1).float()          # [32,2048]
    A3 = _squeeze_shared(A_w3).float()          # [32,2048]
    Bg = B_w1[e].float()                        # [768,32]
    Bu = B_w3[e].float()                        # [768,32]
    delta_gate = (Bg @ A1) * scale              # [768,2048]
    delta_up = (Bu @ A3) * scale                # [768,2048]
    return torch.cat([delta_gate, delta_up], dim=0)   # [1536,2048] gate-first


def build_down_delta(
    A_w2: torch.Tensor,
    B_w2: torch.Tensor,
    e: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """down_proj delta for expert `e` -> [2048, 768].

    A_w2 is the PER-EXPERT lora_A ([128,32,768]); expert slice A_w2[e] -> [32,768].
    B_w2 is the SHARED lora_B ([1,2048,32] or [2048,32]).
    delta_down = B_w2 @ A_w2[e].
    """
    B2 = _squeeze_shared(B_w2).float()          # [2048,32]
    A2_e = A_w2[e].float()                       # [32,768]
    return (B2 @ A2_e) * scale                    # [2048,768]


def build_lm_head_delta(
    A_un: torch.Tensor,
    B_un: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """unembed_tokens -> lm_head delta -> [151936, 2048], computed in float32.

    A_un [32,2048], B_un [151936,32]. delta = B_un @ A_un.
    """
    return (B_un.float() @ A_un.float()) * scale  # [151936,2048] float32


def per_expert_differ(deltas: Sequence[torch.Tensor]) -> float:
    """Min pairwise max-abs-difference across a list of per-expert deltas.

    Returns ~0 for a broadcast merge (all experts identical) and > 1e-5 for genuine
    per-expert deltas. The merge guard aborts when this is <= 1e-5. Requires >= 2
    deltas; with fewer there is no differentiation evidence so 0.0 is returned
    (which deliberately TRIPS the guard rather than vacuously passing it).
    """
    n = len(deltas)
    if n < 2:
        return 0.0
    worst = float("inf")
    for i in range(n):
        di = deltas[i].float()
        for j in range(i + 1, n):
            d = (di - deltas[j].float()).abs().max().item()
            if d < worst:
                worst = d
    return worst


# --------------------------------------------------------------------------- #
# Fidelity agreement helpers (pure logic; consumed by plan 02's fidelity gate).
# --------------------------------------------------------------------------- #

def sentinel_agreement(
    tinker_verdicts: Sequence,
    merged_verdicts: Sequence,
) -> int:
    """Count index-aligned verdict matches between Tinker-sampled and merged-served.

    Lengths MUST match (each sentinel prompt scored on both sides) -- a length
    mismatch is a wiring bug, not a partial result, so it raises instead of
    silently truncating.
    """
    if len(tinker_verdicts) != len(merged_verdicts):
        raise ValueError(
            f"verdict length mismatch: tinker={len(tinker_verdicts)} "
            f"merged={len(merged_verdicts)}"
        )
    return sum(1 for a, b in zip(tinker_verdicts, merged_verdicts) if a == b)


def _avg_rank(values: Sequence[float]) -> "list[float]":
    """Tie-aware average ranks (1-based), pure-python (no scipy dependency)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # mean of 1-based positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (tie-aware), no scipy dependency."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return 0.0
    rx = _avg_rank(list(x))
    ry = _avg_rank(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = (vx * vy) ** 0.5
    if denom == 0.0:
        return 0.0
    return cov / denom


def spearman_agree(
    tinker_scores: Sequence[float],
    merged_scores: Sequence[float],
    thresh: float = 0.95,
) -> bool:
    """True iff Spearman(tinker, merged) >= thresh. Identical -> 1.0 -> True;
    shuffled/uncorrelated -> below thresh -> False."""
    return spearman_rho(tinker_scores, merged_scores) >= thresh


if __name__ == "__main__":  # pragma: no cover
    # CLI merge (stock base load + per-expert apply + manual lm_head + staging save +
    # merge_report) is implemented in Task 2. Task 1 ships the importable math above.
    import sys
    print(
        "merge_tinker_v3: delta/fidelity functions importable. "
        "Run the merge via the Task-2 CLI (added next).",
        file=sys.stderr,
    )
    sys.exit(0)
