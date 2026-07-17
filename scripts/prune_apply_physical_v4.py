"""Stacked-tensor physical expert-removal surgery for the v4 judge (GATE4-04, Plan 26-02).

Re-targets scripts/prune_apply_physical.py at the v4 merged checkpoint's STACKED
layout. The surgery is SIMPLER than v3: the stacked tensor's row order IS the
expert index, so t[sorted(kept_idx)] both drops removed experts AND renumbers
survivors 0..k-1 in one op — no separate rename step.

Per model.language_model.layers.{L}:
    experts.gate_up_proj [256,a,b] -> [k,a,b]   (slice dim 0 to sorted kept idx)
    experts.down_proj    [256,b,c] -> [k,b,c]
    mlp.gate.weight      [256,h]   -> [k,h]      (router rows, same kept idx)
Everything else is copied BYTE-IDENTICAL:
    shared_expert.{gate,up,down}_proj.weight, shared_expert_gate.weight, and the
    ENTIRE mtp.layers.* block (its own stacked experts are NOT part of the routed set).
Config: text_config.num_experts = k  (NOT num_local_experts — writing the wrong key
silently no-ops and ships a 256-claiming config over k-row tensors, a next-load crash).

GATE-BEFORE-REMOVE (SC2, Pitfall 3, T-26-01): main() hard-asserts
output/prune-v4/gated/aimer_224_judge.json records pass:true AND pass_d2_security:true
BEFORE any safe_open. A pruned checkpoint can never exist without a pass record.

Usage:
    .venv-tinker/bin/python -m scripts.prune_apply_physical_v4 --self-check   # CPU, no GPU
    .venv-tinker/bin/python -m scripts.prune_apply_physical_v4                # real surgery (guarded)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aimer_prune_v4 import derive_prefix  # noqa: E402  (reuse the same prefix logic)

N_LAYERS = 40
N_EXPERTS = 256
K_ANCHOR = 224

CHECKPOINT = "models/Qwen3.6-35B-A3B-judge-v4-s1-merged"
MASK_NPY = "output/prune-v4/masks/aimer_k224.npy"
OUT_DIR = "models/Qwen3.6-35B-A3B-judge-v4-pruned-k224"
GATE_JUDGE_JSON = PROJECT_ROOT / "output/prune-v4/gated/aimer_224_judge.json"


def assert_gate_passed(judge_json_path: str | Path = GATE_JUDGE_JSON) -> dict:
    """Gate-before-remove guard (T-26-01): refuse to run without a recorded SHIP pass.

    Raises unless the gate result exists AND records pass_ship:true AND pass_d2_security:true.
    The ship criterion is the pre-registered routing-(B) NON-INFERIORITY bar (ci_lower >= -2pp
    AND D2 retained AND protected retained), NOT the stricter two-sided `pass` (equivalent) — a
    point-better arm is non-inferior yet fails two-sided TOST on the upper bound. This is still
    fully code-enforced; it just gates on the correct pre-registered bar.
    """
    judge_json_path = Path(judge_json_path)
    if not judge_json_path.exists():
        raise RuntimeError(
            f"gate-before-remove VIOLATION: {judge_json_path} missing — run the gate "
            "(scripts.prune_gate_v4) and record a ship pass before any physical surgery"
        )
    rec = json.loads(judge_json_path.read_text())
    if not (rec.get("pass_ship") is True and rec.get("pass_d2_security") is True):
        raise RuntimeError(
            f"gate-before-remove VIOLATION: {judge_json_path.name} does not record "
            f"pass_ship:true AND pass_d2_security:true (pass_ship={rec.get('pass_ship')}, "
            f"pass_d2_security={rec.get('pass_d2_security')}) — no weight may be removed"
        )
    return rec


def _slice_key_map(prefix: str, n_layers: int) -> dict[str, int]:
    """key -> layer for every routed tensor that must be sliced along dim 0."""
    out: dict[str, int] = {}
    for layer in range(n_layers):
        out[f"{prefix}.layers.{layer}.mlp.experts.gate_up_proj"] = layer
        out[f"{prefix}.layers.{layer}.mlp.experts.down_proj"] = layer
        out[f"{prefix}.layers.{layer}.mlp.gate.weight"] = layer
    return out


def apply_physical_v4(checkpoint_dir: str | Path, keep_mask: np.ndarray,
                      out_dir: str | Path) -> dict:
    """Slice the routed experts + router to the kept rows; copy everything else.

    keep_mask must be UNIFORM (same kept-count per layer) so the output has a single
    num_experts. Returns {"k", "n_layers"}.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers, n_experts = keep_mask.shape
    per_layer_k = keep_mask.sum(axis=1)
    assert (per_layer_k == per_layer_k[0]).all(), "keep_mask must be UNIFORM per-layer"
    k = int(per_layer_k[0])
    kept_idx = {layer: sorted(np.where(keep_mask[layer])[0].tolist()) for layer in range(n_layers)}

    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = derive_prefix(weight_map)
    slice_keys = _slice_key_map(prefix, n_layers)
    config = json.loads((checkpoint_dir / "config.json").read_text())

    new_weight_map: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors_out = {}
        with safe_open(checkpoint_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                layer = slice_keys.get(key)
                if layer is not None:
                    t = f.get_tensor(key)
                    assert t.shape[0] == n_experts, f"{key} axis-0 {t.shape[0]} != {n_experts}"
                    tensors_out[key] = t[kept_idx[layer]].contiguous()  # drops + renumbers in one op
                else:
                    tensors_out[key] = f.get_tensor(key)  # shared_expert / mtp / norms: byte-identical
                new_weight_map[key] = shard_name
        if tensors_out:
            save_file(tensors_out, out_dir / shard_name)

    total_size = sum((out_dir / s).stat().st_size for s in set(new_weight_map.values()))
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {**index.get("metadata", {}), "total_size": total_size},
                    "weight_map": new_weight_map}, indent=2)
    )

    config["text_config"]["num_experts"] = k  # NOT num_local_experts (Pitfall: silent no-op)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Copy the non-weight assets a loader needs (tokenizer, chat template, etc.).
    for asset in checkpoint_dir.iterdir():
        if asset.suffix == ".safetensors" or asset.name in ("config.json", "model.safetensors.index.json"):
            continue
        if asset.is_file():
            (out_dir / asset.name).write_bytes(asset.read_bytes())

    return {"k": k, "n_layers": n_layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--mask", default=MASK_NPY)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    # GATE-BEFORE-REMOVE: refuse to touch a tensor without a recorded pass.
    rec = assert_gate_passed()
    print(f"[gate-before-remove] pass record OK: s1_rho={rec.get('s1_rho')} "
          f"pass={rec.get('pass')} pass_d2_security={rec.get('pass_d2_security')}", flush=True)

    keep_mask = np.load(args.mask)
    result = apply_physical_v4(args.checkpoint, keep_mask, args.out)
    print(f"k={result['k']} n_layers={result['n_layers']} -> {args.out}", flush=True)
    return 0


def _write_stacked_surgery_fixture(ckpt_dir: Path, n_layers: int, n_experts: int,
                                   dim_a: int, dim_b: int, seed: int = 0) -> None:
    """Synthetic stacked checkpoint: routed experts + router + shared_expert + mtp block."""
    import torch
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    prefix = "model.language_model"
    tensors: dict[str, "torch.Tensor"] = {}
    for layer in range(n_layers):
        tensors[f"{prefix}.layers.{layer}.mlp.experts.gate_up_proj"] = torch.randn(
            n_experts, dim_a, dim_b, generator=gen)
        tensors[f"{prefix}.layers.{layer}.mlp.experts.down_proj"] = torch.randn(
            n_experts, dim_b, dim_a, generator=gen)
        tensors[f"{prefix}.layers.{layer}.mlp.gate.weight"] = torch.randn(
            n_experts, dim_a, generator=gen)
        tensors[f"{prefix}.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = torch.randn(
            dim_a, dim_b, generator=gen)
        tensors[f"{prefix}.layers.{layer}.mlp.shared_expert_gate.weight"] = torch.randn(
            dim_a, generator=gen)
    # MTP block with its own stacked experts — must be copied byte-identical, never sliced.
    tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.randn(
        n_experts, dim_a, dim_b, generator=gen)
    tensors["mtp.layers.0.mlp.gate.weight"] = torch.randn(n_experts, dim_a, generator=gen)
    weight_map = {key: "model.safetensors" for key in tensors}
    save_file(tensors, ckpt_dir / "model.safetensors")
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))
    (ckpt_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {"num_experts": n_experts, "num_hidden_layers": n_layers},
    }))
    (ckpt_dir / "tokenizer_config.json").write_text(json.dumps({"model_type": "qwen"}))


def _self_check() -> None:
    """Assert-based self-check on a synthetic stacked fixture (no GPU, seconds)."""
    import tempfile

    import torch
    from safetensors import safe_open

    from scripts.prune_apply_physical import build_uniform_keep_mask

    n_layers, n_experts, dim_a, dim_b, k = 2, 8, 4, 5, 5

    scores = np.random.RandomState(0).rand(n_layers, n_experts)
    protected = np.zeros((n_layers, n_experts), dtype=bool)
    protected[1, 7] = True  # coldest-scored expert in layer 1, protected -> must survive
    keep_mask = build_uniform_keep_mask(scores, protected, k)
    assert (keep_mask.sum(axis=1) == k).all() and keep_mask[1, 7]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt, out = tmp_path / "ckpt", tmp_path / "out"
        _write_stacked_surgery_fixture(ckpt, n_layers, n_experts, dim_a, dim_b)

        result = apply_physical_v4(ckpt, keep_mask, out)
        assert result["k"] == k

        # config rewrites text_config.num_experts (NOT num_local_experts).
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["text_config"]["num_experts"] == k
        assert "num_local_experts" not in cfg

        prefix = "model.language_model"
        with safe_open(ckpt / "model.safetensors", framework="pt") as fi, \
             safe_open(out / "model.safetensors", framework="pt") as fo:
            for layer in range(n_layers):
                kept = sorted(np.where(keep_mask[layer])[0].tolist())
                for suf in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "mlp.gate.weight"):
                    key = f"{prefix}.layers.{layer}.{suf}"
                    orig = fi.get_tensor(key)
                    new = fo.get_tensor(key)
                    assert new.shape[0] == k, f"{key} kept {new.shape[0]} != {k}"
                    # every kept row is byte-identical to its original, in sorted order.
                    assert torch.equal(new, orig[kept]), f"{key} sliced rows not byte-identical"
                # shared_expert + shared_expert_gate untouched.
                for suf in ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert_gate.weight"):
                    key = f"{prefix}.layers.{layer}.{suf}"
                    assert torch.equal(fi.get_tensor(key), fo.get_tensor(key)), f"{key} touched"
            # mtp block byte-identical.
            for key in ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.gate.weight"):
                assert torch.equal(fi.get_tensor(key), fo.get_tensor(key)), f"{key} touched"

            # protected expert (layer 1, orig idx 7) survives byte-identical at its new position.
            kept1 = sorted(np.where(keep_mask[1])[0].tolist())
            new_idx = kept1.index(7)
            orig7 = fi.get_tensor(f"{prefix}.layers.1.mlp.experts.gate_up_proj")[7]
            surv = fo.get_tensor(f"{prefix}.layers.1.mlp.experts.gate_up_proj")[new_idx]
            assert torch.equal(orig7, surv), "protected expert weight must survive unmodified"

    # Gate-before-remove guard: raises when the pass record is absent or not pass:true.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = tmp_path / "missing.json"
        raised = False
        try:
            assert_gate_passed(missing)
        except RuntimeError:
            raised = True
        assert raised, "guard must raise when the gate result is missing"

        for bad in ({"pass_ship": False, "pass_d2_security": True},
                    {"pass_ship": True, "pass_d2_security": False},
                    {"pass_ship": True},
                    {"pass": True, "pass_d2_security": True}):  # two-sided pass alone is NOT the ship bar
            p = tmp_path / "bad.json"
            p.write_text(json.dumps(bad))
            raised = False
            try:
                assert_gate_passed(p)
            except RuntimeError:
                raised = True
            assert raised, f"guard must raise for {bad}"

        # Ship pass = non-inferiority: pass_ship:true even with the stricter two-sided pass:false.
        good = tmp_path / "good.json"
        good.write_text(json.dumps({"pass": False, "pass_ship": True,
                                    "pass_d2_security": True, "s1_rho": 0.8}))
        assert_gate_passed(good)  # must NOT raise

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
