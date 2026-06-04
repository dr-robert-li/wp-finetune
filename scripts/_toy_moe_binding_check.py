#!/usr/bin/env python
"""04.3-03 parallel de-risk: validate probe_moe_binding's PRIMARY path (the forward-ACTIVATION
delta on the experts submodule) on a TINY CPU MoE with a real PEFT LoRA — no 30B load needed.

The 9 GPU-free unit tests only exercise the representation-agnostic STRUCTURAL-name fallback.
The real verdict path (capture experts input -> rerun adapter-on vs disable_adapter()-off ->
nonzero output delta == expert-LoRA runtime-active) has never been run end-to-end. This does
that on CPU so the guard's correctness is confirmed regardless of how the 30B memory fight ends.

CASE A: LoRA INSIDE experts  -> expect BOUND, activation_delta > EPS.
CASE B: LoRA on attention only -> expect BINDING_FAILED, delta ~ 0, structural_attn populated.
"""
import sys
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import scripts.checkpoint_parse_check as cpc


class _BatchEncoding(dict):
    def to(self, _device):
        return self


class _Tok:
    """Minimal tokenizer: probe calls tokenizer(text, return_tensors=...).to(device)."""
    def __call__(self, _text, return_tensors=None):
        return _BatchEncoding(input_ids=torch.randint(0, 32, (1, 6)))


class Experts(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_up_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.gate_up_proj(x)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.experts = Experts(d)

    def forward(self, x):
        return self.experts(x)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.q_proj(x)


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = Attn(d)
        self.mlp = MLP(d)

    def forward(self, x):
        return self.mlp(x + self.self_attn(x))


class TinyMoE(nn.Module):
    def __init__(self, d=16, vocab=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Layer(d)])
        self.head = nn.Linear(d, vocab, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, **_):
        x = self.embed(input_ids)
        for ly in self.layers:
            x = ly(x)
        return self.head(x)


def _build(target_modules):
    torch.manual_seed(0)
    base = TinyMoE()
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, init_lora_weights=False)
    return get_peft_model(base, cfg)


def _check(label, target_modules, want_verdict):
    model = _build(target_modules)
    model.eval()
    rep = cpc.probe_moe_binding(model, _Tok())
    got = rep["verdict"]
    ok = got == want_verdict
    print(f"[{label}] target={target_modules}")
    print(f"    experts_module      = {rep.get('experts_module')}")
    print(f"    activation_delta_max= {rep.get('activation_delta_max')}")
    print(f"    activation_error    = {rep.get('activation_delta_error')}")
    print(f"    structural_expert   = {rep.get('structural_expert_lora')}")
    print(f"    structural_attn     = {rep.get('structural_attn_lora')}")
    print(f"    verdict             = {got}  (want {want_verdict})  basis: {rep.get('verdict_basis')}")
    print(f"    => {'PASS' if ok else 'FAIL'}\n")
    return ok, rep


def main():
    ok_a, rep_a = _check("A expert-LoRA", ["gate_up_proj"], "BOUND")
    ok_b, rep_b = _check("B attn-only", ["q_proj"], "BINDING_FAILED")

    # Stronger assertions on the PRIMARY (activation-delta) path, not just the verdict label.
    extra = []
    da = rep_a.get("activation_delta_max")
    if not (isinstance(da, float) and da > 1e-6):
        extra.append("A: activation_delta path did NOT produce a >EPS delta (primary path unproven)")
    if rep_a.get("activation_delta_error") is not None:
        extra.append(f"A: activation path errored: {rep_a['activation_delta_error']}")
    db = rep_b.get("activation_delta_max")
    # B should run the activation path too and find ~0 (experts unaffected).
    if db is not None and db > 1e-6:
        extra.append(f"B: expected ~0 expert delta but got {db}")
    for e in extra:
        print("ASSERT-FAIL:", e)

    passed = ok_a and ok_b and not extra
    print("=== TOY MoE BINDING GUARD:", "ALL PASS" if passed else "FAIL", "===")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
