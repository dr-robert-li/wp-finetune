"""REAP calibration-saliency expert scorer (PRUNE-02, arxiv 2510.13999).

Domain-aware, calibration-based per-expert importance score (wp-moe.md):

    S_j = mean_{x in active(j)}( g_j(x) * ||f_j(x)||_2 )

Scores each expert by the mean product of its router softmax gate weight
and its output L2 norm, over only the tokens routed to it. Captures both
routing frequency (an inactive expert scores 0) and per-token impact
(magnitude of what the expert actually contributes).

REAPCollector extends the RoutingCollector forward-hook pattern in
scripts/profile_base_model.py: instead of counting routed tokens, it
accumulates sum(gate_weight * expert_output_L2_norm) and an activation
count per (layer, expert), so score = sum / max(count, 1) never divides
by zero for an expert that was never selected.

compute_reap_scores() is the calibration entry point (loads the merged
checkpoint, runs a forward pass over calibration data, returns a [48,128]
array) -- per 13-02-PLAN this function is provided but NOT executed in
this plan (GPU + hours of calibration; deferred to 13-05, conditional on
AIMER@25% passing gates). Forward passes MUST run on the merged checkpoint
being masked (never base/unmerged), per 13-RESEARCH Anti-Patterns.

Calibration composition (13-RESEARCH): judge axis uses the 141-item
data/reasoning_dataset/openai_val.jsonl; gen axis uses a 500-2000 subsample
of data/final_dataset/ratio_30_70/openai_train.jsonl.

Usage:
    python -m scripts.reap_prune --self-check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_LAYERS = 48
N_EXPERTS = 128


class REAPCollector:
    """Accumulates per-expert sum(gate_weight * output_L2_norm) and an activation count.

    scores() returns sum/count as a [n_layers, n_experts] float array; an
    expert never activated yields a defined 0.0, not a divide-by-zero.
    """

    def __init__(self, n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self._sum = np.zeros((n_layers, n_experts), dtype=np.float64)
        self._count = np.zeros((n_layers, n_experts), dtype=np.int64)

    def reset(self) -> None:
        self._sum[:] = 0.0
        self._count[:] = 0

    def record(self, layer_idx: int, expert_idx: int, gate_weight: float,
               expert_output_norm: float) -> None:
        """Accumulate one activation event: gate_weight * ||expert_output||_2."""
        self._sum[layer_idx, expert_idx] += gate_weight * expert_output_norm
        self._count[layer_idx, expert_idx] += 1

    def scores(self) -> np.ndarray:
        out = np.zeros_like(self._sum)
        active = self._count > 0
        out[active] = self._sum[active] / self._count[active]
        return out.astype(np.float32)

    def make_gate_hook(self, layer_idx: int, top_k: int = 8):
        """Forward hook for the router (mlp.gate): captures per-token softmax
        gate weight for each of the top-k selected experts at this layer, for
        use by the paired expert-output hooks (make_expert_hook) in the same
        forward pass. Mirrors scripts.profile_base_model.RoutingCollector's
        router-hook pattern (outputs = raw router logits tensor).
        """
        def hook(module, inputs, outputs):
            import torch
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            weights = torch.softmax(logits, dim=-1)
            topk = torch.topk(weights, k=top_k, dim=-1)
            self._current_gate = {
                "layer_idx": layer_idx,
                "indices": topk.indices,   # [n_tokens, top_k]
                "weights": topk.values,    # [n_tokens, top_k]
            }
        return hook

    def make_expert_hook(self, layer_idx: int, expert_idx: int):
        """Forward hook for one expert submodule (mlp.experts[expert_idx]):
        pairs this expert's output L2 norm (per selected token) with the
        gate weight captured by make_gate_hook for the same forward pass.

        # ponytail: exact HF Qwen3MoE per-expert input/output token-subset
        # wiring is verified against the real model at 13-05 calibration
        # time (not exercised here -- no GPU/model in this plan). The
        # accumulation contract (REAPCollector.record) is what's tested.
        """
        def hook(module, inputs, outputs):
            gate = getattr(self, "_current_gate", None)
            if gate is None or gate["layer_idx"] != layer_idx:
                return
            import torch
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            sel = (gate["indices"] == expert_idx)
            if not sel.any():
                return
            tok_rows, k_cols = torch.where(sel)
            norms = out.norm(dim=-1)
            for row, tok_idx in enumerate(tok_rows.tolist()):
                w = gate["weights"][tok_idx, k_cols[row]].item()
                n = norms[row].item() if norms.shape[0] > row else norms[-1].item()
                self.record(layer_idx, expert_idx, w, n)
        return hook


def compute_reap_scores(checkpoint_dir: str | Path, calibration_jsonl_paths: list[str],
                         sample_count: int) -> np.ndarray:
    """Run the REAP calibration forward pass and return a [48,128] score array.

    NOT executed in plan 13-02 (GPU/hours). Deferred to 13-05, conditional on
    AIMER@25% passing gates (13-CONTEXT recommendation). Forward passes MUST
    run on the merged checkpoint being masked, never base/unmerged.
    """
    raise NotImplementedError(
        "compute_reap_scores is the 13-05 calibration entry point -- deliberately "
        "not run in plan 13-02 (module-only; gated on AIMER@25% passing first)."
    )


def _self_check() -> None:
    """Assert-based self-check on a tiny synthetic hook-event fixture (no GPU, no model)."""
    c = REAPCollector(n_layers=4, n_experts=8)
    events = [
        (0, 1, 0.2, 2.0),   # product 0.4
        (0, 1, 0.3, 4.0),   # product 1.2
        (0, 1, 0.4, 2.0),   # product 0.8
        (2, 5, 1.0, 0.5),   # single activation
    ]
    for layer, expert, gate, norm in events:
        c.record(layer_idx=layer, expert_idx=expert, gate_weight=gate, expert_output_norm=norm)

    scores = c.scores()
    assert scores.shape == (4, 8)
    assert np.isfinite(scores).all(), "no expert may score nan/inf"
    assert np.isclose(scores[0, 1], np.mean([0.4, 1.2, 0.8]))
    assert np.isclose(scores[2, 5], 0.5)
    assert scores[1, 0] == 0.0, "never-activated expert must score a defined 0.0"

    c.reset()
    assert np.all(c.scores() == 0.0)

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        print(__doc__)
