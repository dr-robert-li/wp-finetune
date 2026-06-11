"""Wave-0 unit test: Tinker per-expert MoE delta math (locks the merge convention).

Imports the EXACT production functions from scripts.merge_tinker_v3 (no copy-pasted
math). Asserts per-expert deltas are DISTINCT across experts for w1/w2/w3 -- a broadcast
merge (the cos_sim-0.08 ckpt-72-era bug) makes them equal and MUST fail these asserts.

NOTE (2026-06-07): the obsolete 04.4-RESEARCH claim "w1/w3 shared-A => same delta per
expert" is FALSE -- only lora_A is shared; lora_B is per-expert, so delta_e differs.
An across-expert equality assertion (e0 equal to e1) would pass a broadcast bug;
we assert the deltas DIFFER.
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import transformers

# Force-resolve transformers lazy attributes once at module import time.
# transformers uses _LazyModule whose __getattr__ always re-derives the class from its
# internal mapping, bypassing __dict__ writes — so patch('transformers.AutoModelForCausalLM')
# and patch.object(transformers, ...) both fail to intercept `from transformers import ...`
# inside _run_merge.  The only reliable approach: resolve the real class objects HERE,
# then patch their `from_pretrained` classmethods directly (works because the lazy module
# always returns the same class object, so patching that object's method is visible
# from any subsequent `from transformers import` call).
_AMFCLM = transformers.AutoModelForCausalLM   # resolves lazy attr; same object every call
_ATok = transformers.AutoTokenizer            # same

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.merge_tinker_v3 import (
    STOCK_TOK_LEN,
    STOCK_VOCAB,
    _k,
    _run_merge,
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


# ---------------------------------------------------------------------------
# is_moe_only=True merge path — synthetic adapter with NO attn/unembed keys
# ---------------------------------------------------------------------------

def _build_moe_only_adapter_tar(tmp_dir: str, num_layers: int, num_experts: int) -> str:
    """Write a minimal Tinker-convention MoE-only adapter tar.

    Contains ONLY w1/w2/w3 lora_A/lora_B keys for `num_layers` layers and
    `num_experts` experts. No self_attn.{q,k,v,o}_proj keys; no unembed_tokens keys.
    The differ guard requires >=4 experts to produce distinct deltas — caller must
    pass num_experts >= 4.
    """
    from safetensors.torch import save_file

    torch.manual_seed(42)
    tensors = {}
    for L in range(num_layers):
        # w1 (gate_proj): shared A [1,R,HIDDEN], per-expert B [E,MLP,R]
        tensors[_k(L, "w1", "A")] = torch.randn(1, R, HIDDEN)
        tensors[_k(L, "w1", "B")] = torch.randn(num_experts, MLP, R)
        # w3 (up_proj): shared A [1,R,HIDDEN], per-expert B [E,MLP,R]
        tensors[_k(L, "w3", "A")] = torch.randn(1, R, HIDDEN)
        tensors[_k(L, "w3", "B")] = torch.randn(num_experts, MLP, R)
        # w2 (down_proj): per-expert A [E,R,MLP], shared B [1,HIDDEN,R]
        tensors[_k(L, "w2", "A")] = torch.randn(num_experts, R, MLP)
        tensors[_k(L, "w2", "B")] = torch.randn(1, HIDDEN, R)

    safetensors_path = os.path.join(tmp_dir, "adapter_model.safetensors")
    save_file(tensors, safetensors_path)

    adapter_cfg = {"r": R, "lora_alpha": R, "target_modules": ["w1", "w2", "w3"]}
    cfg_path = os.path.join(tmp_dir, "adapter_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(adapter_cfg, fh)

    tar_path = os.path.join(tmp_dir, "checkpoint.tar")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(safetensors_path, arcname="adapter_model.safetensors")
        tf.add(cfg_path, arcname="adapter_config.json")
    return tar_path


def _make_fake_model(num_layers: int, num_experts: int) -> MagicMock:
    """Build a minimal fake model that satisfies all _run_merge assertions.

    Required by _run_merge:
      - model.config.tie_word_embeddings == False
      - model.config.vocab_size == STOCK_VOCAB (151936)
      - model.config.hidden_size == HIDDEN (2048)
      - model.lm_head.weight.shape == (STOCK_VOCAB, HIDDEN)
      - model.model.layers[L].mlp.experts.gate_up_proj.data[e] += delta  (in-place add)
      - model.model.layers[L].mlp.experts.down_proj.data[e] += delta      (in-place add)
      - model.save_pretrained(path, ...) -> no-op (writes nothing; step 6)
    """
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.vocab_size = STOCK_VOCAB
    model.config.hidden_size = HIDDEN

    # lm_head.weight: real tensor so shape assertion passes; data is never mutated
    # (is_moe_only skips Step 5, so lm_head_excluded=True and the delta is NOT added).
    lm_head_weight = torch.zeros(STOCK_VOCAB, HIDDEN, dtype=torch.bfloat16)
    model.lm_head.weight = lm_head_weight

    # model.model.layers[L].mlp.experts with real gate_up_proj / down_proj tensors
    # so the in-place += operations in Step 3 succeed on real data.
    layers = []
    for _ in range(num_layers):
        gate_up = torch.zeros(num_experts, GATE_UP_OUT, HIDDEN, dtype=torch.bfloat16)
        down = torch.zeros(num_experts, HIDDEN, MLP, dtype=torch.bfloat16)
        experts = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
        layer = SimpleNamespace(mlp=SimpleNamespace(experts=experts))
        layers.append(layer)
    model.model.layers = layers

    # save_pretrained: no-op — the output dir already exists (created by _run_merge)
    model.save_pretrained = MagicMock(return_value=None)
    return model


def _make_fake_tokenizer() -> MagicMock:
    """Build a minimal fake tokenizer that satisfies the three stock asserts in Step 6.

      1. len(tok.encode("<wp_judge>", add_special_tokens=False)) > 1  (stock text routing)
      2. len(tok) == STOCK_TOK_LEN  (151669)
      3. max(tok.get_vocab().values()) < STOCK_VOCAB  (151936)
    """
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]           # len 3 > 1: stock multi-piece tokenisation
    tok.__len__ = MagicMock(return_value=STOCK_TOK_LEN)
    tok.get_vocab.return_value = {"a": 0, "b": STOCK_TOK_LEN - 1}  # max < STOCK_VOCAB
    tok.save_pretrained = MagicMock(return_value=None)
    return tok


def test_moe_only_merge_path():
    """MoE-only adapter (no attn/unembed keys) merges without AssertionError or KeyError.

    Asserts all 7 required report conditions (PATTERNS lines 381-388):
      1. report["is_moe_only_adapter"] == True
      2. report["attention_skipped"] == True
      3. report["attention_q_proj_changed"] == False
      4. report["lm_head_excluded"] == True
      5. report["lm_head_applied"] == False
      6. MoE per-expert deltas ARE applied (gate_up_touched + down_touched > 0)
      7. per_expert_delta_differ_check all > 1e-5 (differ guard still runs on MoE-only)

    Uses synthetic adapter and a mocked base model/tokenizer — no 57 GiB base load.
    NUM_LAYERS and NUM_EXPERTS are patched to 2/4 for speed; the differ guard uses >=4
    experts so num_experts=4 is the floor.
    """
    import scripts.merge_tinker_v3 as mv3

    TEST_NUM_LAYERS = 2
    TEST_NUM_EXPERTS = 4   # differ guard hardcodes range(4); must be >= 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = _build_moe_only_adapter_tar(tmp_dir, TEST_NUM_LAYERS, TEST_NUM_EXPERTS)
        output_dir = os.path.join(tmp_dir, "_staging", "moe_only_test")
        report_path = os.path.join(tmp_dir, "merge_report.json")

        args = argparse.Namespace(
            adapter_tar=tar_path,
            base="/fake/base",
            output_dir=output_dir,
            report=report_path,
            exclude_lm_head=False,   # not set; is_moe_only takes over
        )

        fake_model = _make_fake_model(TEST_NUM_LAYERS, TEST_NUM_EXPERTS)
        fake_tok = _make_fake_tokenizer()

        # Patch NUM_LAYERS and NUM_EXPERTS on the module so the MoE loop uses our tiny dims.
        # Patch AutoModelForCausalLM and AutoTokenizer at the transformers module level
        # (they are imported via `from transformers import ...` inside _run_merge).
        with (
            patch.object(mv3, "NUM_LAYERS", TEST_NUM_LAYERS),
            patch.object(mv3, "NUM_EXPERTS", TEST_NUM_EXPERTS),
            patch.object(_AMFCLM, "from_pretrained", return_value=fake_model),
            patch.object(_ATok, "from_pretrained", return_value=fake_tok),
        ):
            ret = _run_merge(args)

        assert ret == 0, f"_run_merge returned {ret} (expected 0)"

        # Load the written report for assertions
        with open(report_path) as fh:
            report = json.load(fh)

        # 1. is_moe_only detected correctly
        assert report["is_moe_only_adapter"] is True, \
            f"is_moe_only_adapter not True: {report.get('is_moe_only_adapter')}"

        # 2. Attention stage skipped
        assert report["attention_skipped"] is True, \
            f"attention_skipped not True: {report.get('attention_skipped')}"

        # 3. q_proj not changed (attention stage did not run)
        assert report["attention_q_proj_changed"] is False, \
            f"attention_q_proj_changed not False: {report.get('attention_q_proj_changed')}"

        # 4. lm_head excluded
        assert report["lm_head_excluded"] is True, \
            f"lm_head_excluded not True: {report.get('lm_head_excluded')}"

        # 5. lm_head not applied
        assert report["lm_head_applied"] is False, \
            f"lm_head_applied not False: {report.get('lm_head_applied')}"

        # 6. MoE per-expert deltas were applied (Step 3 ran for both layers × all experts)
        expected_touched = TEST_NUM_LAYERS * TEST_NUM_EXPERTS
        assert report["gate_up_touched"] == expected_touched, \
            f"gate_up_touched={report['gate_up_touched']} expected {expected_touched}"
        assert report["down_touched"] == expected_touched, \
            f"down_touched={report['down_touched']} expected {expected_touched}"

        # 7. per_expert_differ > 1e-5 for w1/w2/w3 (broadcast-merge guard still runs)
        differ = report["per_expert_delta_differ_check"]
        for w in ("w1", "w2", "w3"):
            assert differ[w] > 1e-5, \
                f"per_expert_differ[{w}]={differ[w]} <= 1e-5 (broadcast merge guard failed)"
