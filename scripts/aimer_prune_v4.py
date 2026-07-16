"""Stacked-tensor AIMER weight-norm expert importance scorer for the v4 judge (GATE4-04, Plan 26-01).

Re-targets scripts/aimer_prune.py's AIMER formula (UNCHANGED) at the v4 merged
checkpoint's stacked-tensor layout. The score per (layer, expert) is still

    score = P / sqrt(N * Q)

with P = L1 norm, Q = squared L2 norm, N = element count over ALL of that
expert's weights (its slice of experts.gate_up_proj + experts.down_proj). By
Cauchy-Schwarz it is scale-invariant and bounded in [1/sqrt(N), 1]. Only the
tensor-ACCESS pattern changes: v3 iterated per-expert unstacked keys; v4 stores
one STACKED tensor per layer

    <prefix>.layers.{L}.mlp.experts.gate_up_proj  [256, 1024, 2048]
    <prefix>.layers.{L}.mlp.experts.down_proj      [256, 2048, 512]

under a `model.language_model.` prefix (VL composite model). We reduce along
dim 0 (the expert axis) per layer instead of iterating keys.

Load-bearing safety (26-01 threat model):
- The prefix is DERIVED from model.safetensors.index.json's weight_map (never
  hardcoded) and asserted; a missing/renamed expert key RAISES (Pitfall 1 — never
  score a zero array).
- Only the 40 model.language_model.layers.{0..39} MoE blocks are scored. The
  mtp.layers.* Multi-Token-Prediction block ALSO carries stacked experts but is
  NOT part of the 256-expert routed set and is EXCLUDED.
- Never loads the model into a framework (pure safe_open streaming) — GB10-safe
  (T-26-04: an in-process bf16 load OOMs the unified pool).

Main() runs the full Plan 26-01 Task 1 pipeline: score AIMER (mean across the 3
merged judge seeds s0/s1/s2), confirm the merge-of-record (SC1), build the k=224
uniform keep-mask (never dropping a protected expert), and pin the protected
mask sha256 into output/prune-v4/protected_manifest_v4.json.

Usage:
    .venv-tinker/bin/python -m scripts.aimer_prune_v4 --self-check   # CPU, seconds
    .venv-tinker/bin/python -m scripts.aimer_prune_v4                # full pipeline (streams the real checkpoints)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prune_apply_physical import build_uniform_keep_mask  # noqa: E402

# --- v4 judge config (26-01) -------------------------------------------------
N_LAYERS = 40
N_EXPERTS = 256
K_ANCHOR = 224  # the single authorized compression point routed from Gate B

JUDGE_SEEDS = [
    "models/Qwen3.6-35B-A3B-judge-v4-s0-merged",
    "models/Qwen3.6-35B-A3B-judge-v4-s1-merged",
    "models/Qwen3.6-35B-A3B-judge-v4-s2-merged",
]
MERGE_OF_RECORD = "models/Qwen3.6-35B-A3B-judge-v4-s1-merged"  # SC1 checkpoint
PROTECTED_MASK = "output/sieve-v4/protected_expert_mask.npy"    # 25-01, [40,256] bool

SCORES_OUT = "output/prune-v4/aimer_scores_judge_v4.npy"
MASK_OUT = "output/prune-v4/masks/aimer_k224.npy"
MANIFEST_OUT = "output/prune-v4/protected_manifest_v4.json"

EXPERT_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def derive_prefix(weight_map: dict[str, str]) -> str:
    """Derive the routed-MoE key prefix from the weight_map (Pitfall 1: never hardcode).

    The v4 checkpoint carries TWO stacked-expert families: the routed language
    model (`model.language_model.layers.{L}...`) and the MTP block
    (`mtp.layers.{L}...`). The routed set is the one under `.language_model.`;
    the MTP block is excluded. We return the language-model prefix (everything
    before `.layers.`) and assert exactly one such prefix exists.
    """
    prefixes = set()
    for key in weight_map:
        if ".mlp.experts.gate_up_proj" in key and ".layers." in key:
            prefix = key.split(".layers.")[0]
            if not prefix.startswith("mtp"):  # exclude the MTP block
                prefixes.add(prefix)
    if len(prefixes) != 1:
        raise KeyError(
            f"expected exactly one non-mtp routed-expert prefix, found {sorted(prefixes)}"
        )
    return prefixes.pop()


def _expert_keys(prefix: str, layer: int) -> tuple[str, ...]:
    return tuple(f"{prefix}.layers.{layer}.{suf}" for suf in EXPERT_SUFFIXES)


def compute_aimer_scores_v4(
    checkpoint_dir: str | Path, n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS
) -> np.ndarray:
    """Return [n_layers, n_experts] float32 AIMER scores for a stacked-tensor checkpoint.

    Streams each layer's two stacked expert tensors one at a time (never holds
    more than one [n_experts, dim_a, dim_b] tensor + scalar accumulators).
    """
    checkpoint_dir = Path(checkpoint_dir)
    weight_map = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefix = derive_prefix(weight_map)

    # Fail loud on any missing/renamed expert key BEFORE streaming (Pitfall 1).
    wanted = [k for layer in range(n_layers) for k in _expert_keys(prefix, layer)]
    missing = [k for k in wanted if k not in weight_map]
    if missing:
        raise KeyError(
            f"{len(missing)} expert tensor keys missing from weight_map, "
            f"e.g. {sorted(missing)[:3]}"
        )

    P = np.zeros((n_layers, n_experts), dtype=np.float64)
    Q = np.zeros((n_layers, n_experts), dtype=np.float64)
    N = np.zeros((n_layers, n_experts), dtype=np.int64)

    for layer in range(n_layers):
        for key in _expert_keys(prefix, layer):
            with safe_open(checkpoint_dir / weight_map[key], framework="pt") as f:
                w = f.get_tensor(key).float()  # [n_experts, dim_a, dim_b]
            assert w.shape[0] == n_experts, f"{key} axis-0 {w.shape[0]} != {n_experts}"
            P[layer] += w.abs().sum(dim=(1, 2)).numpy()
            Q[layer] += (w**2).sum(dim=(1, 2)).numpy()
            N[layer] += w[0].numel()  # per-expert element count (identical for every expert)

    scores = (P / np.sqrt(N * Q)).astype(np.float32)
    assert np.isfinite(scores).all(), "AIMER scores must be finite for every expert"
    return scores


def verify_merge_of_record(checkpoint_dir: str | Path) -> dict:
    """SC1: confirm the merge-of-record has no adapter files and the right arch/dims."""
    checkpoint_dir = Path(checkpoint_dir)
    adapters = sorted(p.name for p in checkpoint_dir.glob("*adapter*"))
    if adapters:
        raise AssertionError(f"merge-of-record has adapter files (not fully merged): {adapters}")
    config = json.loads((checkpoint_dir / "config.json").read_text())
    arch = config.get("architectures")
    tc = config.get("text_config", {})
    ne, nh = tc.get("num_experts"), tc.get("num_hidden_layers")
    assert arch == ["Qwen3_5MoeForConditionalGeneration"], f"unexpected architectures {arch}"
    assert ne == N_EXPERTS, f"num_experts {ne} != {N_EXPERTS}"
    assert nh == N_LAYERS, f"num_hidden_layers {nh} != {N_LAYERS}"
    return {"adapters": adapters, "architectures": arch, "num_experts": ne, "num_hidden_layers": nh}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=JUDGE_SEEDS,
                    help="1+ merged checkpoints; multiple = elementwise MEAN (shared judge profile, A1)")
    ap.add_argument("--protected-mask", default=PROTECTED_MASK)
    ap.add_argument("--scores-out", default=SCORES_OUT)
    ap.add_argument("--mask-out", default=MASK_OUT)
    ap.add_argument("--manifest-out", default=MANIFEST_OUT)
    ap.add_argument("--k", type=int, default=K_ANCHOR)
    args = ap.parse_args()

    # SC1: confirm the merge-of-record before scoring anything.
    mor = verify_merge_of_record(MERGE_OF_RECORD)
    print(f"[SC1] merge-of-record OK: {mor}", flush=True)

    # Score AIMER, mean across s0/s1/s2 (v3 shared-profile convention, [ASSUMED A1]).
    all_scores = []
    for ckpt in args.checkpoints:
        print(f"[aimer] scoring {ckpt} ...", flush=True)
        all_scores.append(compute_aimer_scores_v4(ckpt))
    scores = (np.mean(all_scores, axis=0).astype(np.float32)
              if len(all_scores) > 1 else all_scores[0])
    assert scores.shape == (N_LAYERS, N_EXPERTS) and np.isfinite(scores).all()

    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.scores_out, scores)
    print(f"[aimer] scores {scores.shape} min={scores.min():.4f} max={scores.max():.4f} "
          f"mean={scores.mean():.4f} -> {args.scores_out}", flush=True)

    # Build the k=224 UNIFORM keep-mask (exactly k/layer; protected always kept;
    # feasible because max_protected_per_layer=98 <= 224). Uniform is required so
    # 26-02's physical surgery can slice every layer to a single num_experts=224.
    protected = np.load(args.protected_mask)
    assert protected.dtype == bool and protected.shape == (N_LAYERS, N_EXPERTS)
    keep = build_uniform_keep_mask(scores, protected, args.k)
    assert (keep.sum(axis=1) == args.k).all(), "keep-mask must be exactly k/layer (uniform)"
    assert bool(np.all((~protected) | keep)), "keep-mask dropped a protected expert"

    Path(args.mask_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.mask_out, keep)
    print(f"[mask] k={args.k} kept exactly {args.k}/layer, protected_retained=True "
          f"-> {args.mask_out}", flush=True)

    # Pin the protected mask sha256 into a NEW v4-specific manifest (T-26-06).
    manifest = {
        "requirement": "GATE4-04",
        "protected_mask_npy": args.protected_mask,
        "mask_npy_sha256": _sha256(args.protected_mask),
        "max_protected_per_layer": int(protected.sum(axis=1).max()),
        "k_anchor": args.k,
        "merge_of_record": MERGE_OF_RECORD,
        "note": "v4-specific — do NOT reuse v3's output/sieve/prune_set_for_phase13.json",
    }
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] pinned protected sha256={manifest['mask_npy_sha256'][:12]}... "
          f"-> {args.manifest_out}", flush=True)
    return 0


def _write_stacked_fixture(ckpt_dir: Path, n_layers: int, n_experts: int,
                           dim_a: int, dim_b: int, scale: float = 1.0, seed: int = 0,
                           include_mtp: bool = True) -> None:
    """Tiny synthetic STACKED-tensor checkpoint under the model.language_model. prefix.

    Mirrors prune_apply_physical._write_fixture_checkpoint but stacks experts along
    dim 0 and prefixes keys like the real v4 checkpoint. Optionally writes an
    mtp.layers.0 stacked-expert block that MUST be excluded from scoring.
    """
    import torch
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    prefix = "model.language_model"
    tensors: dict[str, "torch.Tensor"] = {}
    for layer in range(n_layers):
        tensors[f"{prefix}.layers.{layer}.mlp.experts.gate_up_proj"] = (
            torch.randn(n_experts, dim_a, dim_b, generator=gen) * scale
        )
        tensors[f"{prefix}.layers.{layer}.mlp.experts.down_proj"] = (
            torch.randn(n_experts, dim_b, dim_a, generator=gen) * scale
        )
        tensors[f"{prefix}.layers.{layer}.mlp.gate.weight"] = (
            torch.randn(n_experts, dim_a, generator=gen)
        )
        tensors[f"{prefix}.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = (
            torch.randn(dim_a, dim_b, generator=gen)
        )
    if include_mtp:
        # An MTP block with its OWN stacked experts — must NOT be scored.
        tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = (
            torch.randn(n_experts, dim_a, dim_b, generator=gen)
        )
        tensors["mtp.layers.0.mlp.experts.down_proj"] = (
            torch.randn(n_experts, dim_b, dim_a, generator=gen)
        )
    weight_map = {k: "model.safetensors" for k in tensors}
    save_file(tensors, ckpt_dir / "model.safetensors")
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _self_check() -> None:
    """Assert-based self-check on a synthetic stacked-tensor fixture (no GPU, seconds)."""
    import tempfile

    n_layers, n_experts, dim_a, dim_b = 3, 6, 4, 5
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_a, ckpt_b = tmp_path / "a", tmp_path / "b"
        _write_stacked_fixture(ckpt_a, n_layers, n_experts, dim_a, dim_b, scale=1.0)
        _write_stacked_fixture(ckpt_b, n_layers, n_experts, dim_a, dim_b, scale=5.0)

        s_a = compute_aimer_scores_v4(ckpt_a, n_layers, n_experts)
        s_b = compute_aimer_scores_v4(ckpt_b, n_layers, n_experts)
        s_a2 = compute_aimer_scores_v4(ckpt_a, n_layers, n_experts)

        assert s_a.shape == (n_layers, n_experts)
        assert np.isfinite(s_a).all()
        assert np.allclose(s_a, s_b, rtol=1e-4), "AIMER score must be scale-invariant"
        assert np.array_equal(s_a, s_a2), "AIMER score must be deterministic"
        # bound [1/sqrt(N), 1]; N = per-expert elements over BOTH stacked tensors
        n = dim_a * dim_b * 2
        lower = 1.0 / np.sqrt(n)
        assert (s_a >= lower - 1e-6).all() and (s_a <= 1.0 + 1e-6).all(), \
            "AIMER score must be bounded in [1/sqrt(N), 1]"

        # Prefix derivation excludes the mtp block (only the language_model prefix returned).
        wm = json.loads((ckpt_a / "model.safetensors.index.json").read_text())["weight_map"]
        assert derive_prefix(wm) == "model.language_model"

        # Missing/renamed expert key RAISES (Pitfall 1) — never a zero array.
        bad = tmp_path / "bad"
        _write_stacked_fixture(bad, n_layers, n_experts, dim_a, dim_b, include_mtp=False)
        idx = bad / "model.safetensors.index.json"
        wm_bad = json.loads(idx.read_text())["weight_map"]
        wm_bad.pop("model.language_model.layers.1.mlp.experts.down_proj")
        idx.write_text(json.dumps({"metadata": {}, "weight_map": wm_bad}))
        raised = False
        try:
            compute_aimer_scores_v4(bad, n_layers, n_experts)
        except KeyError:
            raised = True
        assert raised, "a missing expert key must raise (never silently score zero)"

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
