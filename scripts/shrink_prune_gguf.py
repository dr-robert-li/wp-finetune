"""Shrink study: drop more routed MoE experts from an already-pruned (k224) GGUF.

Only the Q6_K GGUF of the k224 judge exists, so experts are removed directly in GGUF
space. GGUF slot j of layer L is original expert sorted(where(k224_mask[L]))[j]
(prune_apply_physical_v4 keeps experts in ascending original order). Per layer we keep
the top-K AIMER scores (higher = more important, same rule as build_uniform_keep_mask)
among the current 224, never dropping a protected expert; survivors keep their order.

Sliced along the expert axis (outermost in GGUF, i.e. reader data axis 0):
    blk.N.ffn_{gate,up,down,gate_up}_exps.weight   (quantized: raw byte blocks)
    blk.N.ffn_gate_inp.weight                       (router rows)
    blk.N.exp_probs_b(.bias)                        (if present)
Everything else (shared expert, attention, norms, tokenizer, chat template) is copied
byte-identical; only {arch}.expert_count is rewritten.

Usage:
    python scripts/shrink_prune_gguf.py --in models/_gguf/wp-judge-v4-pruned-k224.Q6_K.gguf \
        --out models/_gguf/shrink-k192.Q6_K.gguf --keep 192
    python scripts/shrink_prune_gguf.py --self-check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
_LLAMA_GGUF_PY = Path.home() / "work/llama.cpp/gguf-py"
if (_LLAMA_GGUF_PY / "gguf").is_dir():
    sys.path.insert(0, str(_LLAMA_GGUF_PY))  # match the llama.cpp build when available

import gguf  # noqa: E402

from scripts.prune_apply_physical import build_uniform_keep_mask  # noqa: E402

MASK_NPY = "output/prune-v4/masks/aimer_k224.npy"
SCORES_NPY = "output/prune-v4/aimer_scores_judge_v4.npy"
PROTECTED_NPY = "output/sieve-v4/protected_expert_mask.npy"  # per protected_manifest_v4.json

_EXPERT_TENSOR = re.compile(
    r"^blk\.(\d+)\.(ffn_(gate|up|down|gate_up)_exps|ffn_gate_inp|exp_probs_b)(\.weight|\.bias)?$")


def select_slots(prev_mask: np.ndarray, scores: np.ndarray, protected: np.ndarray, k: int) -> np.ndarray:
    """[n_layers, k] GGUF slot indices (into the current prev_mask.sum(1) experts) to keep."""
    assert prev_mask.shape == scores.shape == protected.shape
    assert not (protected & ~prev_mask).any(), "a protected expert is already missing from the input"
    new = build_uniform_keep_mask(np.where(prev_mask, scores, -np.inf), protected, k)
    assert not (new & ~prev_mask).any()
    return np.stack([np.where(new[L][prev_mask[L]])[0] for L in range(len(prev_mask))])


def shrink_gguf(src: str | Path, dst: str | Path, slots: np.ndarray) -> dict:
    reader = gguf.GGUFReader(str(src))
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_exp = reader.fields[f"{arch}.expert_count"].contents()
    n_layers, k = slots.shape
    assert n_layers == reader.fields[f"{arch}.block_count"].contents()

    writer = gguf.GGUFWriter(str(dst), arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == f"{arch}.expert_count":
            val = k
        elif field.name == f"{arch}.expert_used_count":
            assert val <= k, f"expert_used_count {val} > K {k}"
        writer.add_key_value(field.name, val, vtype, sub_type=sub)

    out, sliced = [], []
    for t in reader.tensors:
        data, m = t.data, _EXPERT_TENSOR.match(t.name)
        if m:
            layer = int(m.group(1))
            assert layer < n_layers, f"{t.name}: no selection for layer {layer}"
            # experts are the outermost GGUF dim == reader axis 0 -> contiguous byte blocks
            assert data.shape[0] == n_exp and data.nbytes % n_exp == 0, f"{t.name} {data.shape}"
            data = np.ascontiguousarray(data[slots[layer]])
            assert data.nbytes == t.data.nbytes // n_exp * k
            sliced.append(t.name)
        out.append((t.name, data, t.tensor_type))
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in out:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    return {"arch": arch, "from": n_exp, "k": k, "sliced": sliced}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--mask", default=MASK_NPY)
    ap.add_argument("--scores", default=SCORES_NPY)
    ap.add_argument("--protected", default=PROTECTED_NPY)
    args = ap.parse_args()

    slots = select_slots(np.load(args.mask), np.load(args.scores), np.load(args.protected), args.keep)
    r = shrink_gguf(args.src, args.out, slots)
    kinds = sorted({n.split(".", 2)[2] for n in r["sliced"]})
    print(f"{r['arch']}: {r['from']} -> {r['k']} experts/layer; sliced {len(r['sliced'])} tensors "
          f"({', '.join(kinds)}) -> {args.out}", flush=True)
    return 0


def _self_check() -> None:
    """Synthetic 2-layer MoE GGUF (Q8_0 experts, F32 router), prune 6 -> 4, assert everything."""
    import tempfile

    Q = gguf.GGMLQuantizationType
    n_layers, n_orig, d, ff = 2, 8, 32, 64
    rng = np.random.RandomState(0)
    prev = np.zeros((n_layers, n_orig), bool)
    prev[0, [0, 1, 2, 4, 5, 7]] = prev[1, [1, 2, 3, 5, 6, 7]] = True  # k=6 already
    scores = rng.rand(n_layers, n_orig).astype(np.float32)
    protected = np.zeros_like(prev)
    lowest = np.where(prev[1])[0][np.argmin(scores[1][prev[1]])]
    protected[1, lowest] = True  # coldest current expert in layer 1 must survive
    k = 4
    slots = select_slots(prev, scores, protected, k)
    cur1 = np.where(prev[1])[0]
    assert lowest in cur1[slots[1]] and (np.diff(slots, axis=1) > 0).all()
    exp0 = np.argsort(-scores[0][prev[0]])[:k]
    assert sorted(exp0.tolist()) == slots[0].tolist(), "layer 0 must keep top-k scores"

    n = int(prev[0].sum())
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.gguf", Path(tmp) / "out.gguf"
        w = gguf.GGUFWriter(str(src), arch="qwen35moe")
        w.add_block_count(n_layers)
        w.add_expert_count(n)
        w.add_expert_used_count(2)
        w.add_token_list(["a", "b", "<|im_end|>"])
        w.add_chat_template("{{ messages }}")
        orig = {}
        for L in range(n_layers):
            for name, arr, qt in (
                (f"blk.{L}.ffn_gate_exps.weight", rng.randn(n, ff, d).astype(np.float32), Q.Q8_0),
                (f"blk.{L}.ffn_up_exps.weight", rng.randn(n, ff, d).astype(np.float32), Q.Q8_0),
                (f"blk.{L}.ffn_down_exps.weight", rng.randn(n, d, ff).astype(np.float32), Q.Q8_0),
                (f"blk.{L}.ffn_gate_inp.weight", rng.randn(n, d).astype(np.float32), Q.F32),
                (f"blk.{L}.ffn_gate_inp_shexp.weight", rng.randn(d).astype(np.float32), Q.F32),
                (f"blk.{L}.ffn_up_shexp.weight", rng.randn(ff, d).astype(np.float32), Q.Q8_0),
            ):
                data = gguf.quants.quantize(arr, qt) if qt != Q.F32 else arr
                orig[name] = data
                w.add_tensor(name, data, raw_dtype=qt if qt != Q.F32 else None)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

        r = shrink_gguf(src, dst, slots)
        assert len(r["sliced"]) == 4 * n_layers, r["sliced"]
        rd = gguf.GGUFReader(str(dst))
        f = rd.fields
        assert f["qwen35moe.expert_count"].contents() == k
        assert f["qwen35moe.expert_used_count"].contents() == 2
        assert f["tokenizer.ggml.tokens"].contents() == ["a", "b", "<|im_end|>"]
        assert f["tokenizer.chat_template"].contents() == "{{ messages }}"
        for t in rd.tensors:
            o = orig[t.name]
            if "shexp" in t.name:
                assert t.data.tobytes() == o.tobytes(), f"{t.name} touched"
            else:
                L = int(t.name.split(".")[1])
                assert t.data.shape[0] == k and t.data.tobytes() == o[slots[L]].tobytes(), t.name
                assert t.shape[-1] == k  # GGUF ne: expert dim is last (slowest)
            assert t.tensor_type == (Q.F32 if o.dtype == np.float32 else Q.Q8_0)
        # protected expert survives byte-identical
        j = list(cur1[slots[1]]).index(lowest)
        src_slot = list(cur1).index(lowest)
        t = next(t for t in rd.tensors if t.name == "blk.1.ffn_down_exps.weight")
        assert t.data[j].tobytes() == orig["blk.1.ffn_down_exps.weight"][src_slot].tobytes()
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
