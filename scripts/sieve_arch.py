"""Architecture-awareness helper for the MoE-Sieve profiler + mask + k-sweep stack (GATE4-02).

Every architecture fact the Sieve tooling needs (layer count, expert count, per-layer
DeltaNet-vs-Attention stratum, live module traversal path, task-token IDs) is derived
from the loaded model.config / tokenizer / profiling JSONL here, in one place, so the
consumers never hardcode the v3 numbers (48 layers, 128 experts) in a load-bearing
position. Config-derived on the v4 base (Qwen3.6-35B-A3B judge): 40 layers, 256 experts,
30 "deltanet" + 10 "attention" strata. Config-derived on a v3 Qwen3-30B-A3B config:
48 layers, 128 experts, uniform strata (no layer_types key).

Dependency-light: numpy + stdlib only at module top. torch is imported lazily inside
resolve_moe_layers() (duck-typed getattr walk) so the other four functions stay
importable in a CPU-only, torch-optional test environment.

Usage:
    python -m scripts.sieve_arch --self-check
"""
from __future__ import annotations

from typing import Any

ATTENTION_STRATUM = "attention"
DELTANET_STRATUM = "deltanet"

# Candidate module-tree roots for the live decoder layer list, relative to the
# (PeftModel-unwrapped) model object. Tried in order; the FIRST root that yields
# a non-empty layer list wins. Phase 20-04 found the LIVE in-memory tree is FLAT
# (model.model.layers) even though the on-disk save/load convention nests under
# language_model -- so the flat path is tried first, not the ROADMAP's literal guess.
_MOE_LAYER_ROOT_CANDIDATES = (
    "model.layers",
    "model.language_model.layers",
    "language_model.layers",
)


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """Get `key` from a dict-like OR attribute-style config object.

    Real transformers PretrainedConfig objects do NOT implement .get() -- only
    attribute access -- while test fixtures are typically plain dicts. Support both.
    """
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _text_config(config: Any) -> Any:
    """Resolve the text-config sub-object, or `config` itself if there is none.

    Works on both a composite VL config (text_config sub-object) and a plain
    text-only config (v3 Qwen3-MoE has no text_config key).
    """
    return _cfg(config, "text_config", config)


def arch_dims(config: Any) -> tuple[int, int]:
    """Return (n_layers, n_experts) from a HF config or its text_config.

    v4 base: (40, 256). v3 Qwen3-MoE config: (48, 128).
    """
    tc = _text_config(config)
    n_layers = _cfg(tc, "num_hidden_layers")
    n_experts = _cfg(tc, "num_experts")
    return (int(n_layers), int(n_experts))


def layer_strata(config: Any) -> list[str]:
    """Return a list[str] of length n_layers: ATTENTION_STRATUM or DELTANET_STRATUM.

    Resolution order:
      1. config.layer_types[i] == "full_attention" -> attention, else deltanet.
      2. config.full_attention_interval: attention iff (i % interval) == interval - 1.
      3. Neither key present (v3 uniform base): all DELTANET_STRATUM. Never raises.
    """
    tc = _text_config(config)
    n_layers = int(_cfg(tc, "num_hidden_layers") or 0)
    layer_types = _cfg(tc, "layer_types", None)
    if layer_types is not None:
        return [
            ATTENTION_STRATUM if lt == "full_attention" else DELTANET_STRATUM
            for lt in layer_types
        ]
    interval = _cfg(tc, "full_attention_interval", None)
    if interval:
        return [
            ATTENTION_STRATUM if (i % interval) == interval - 1 else DELTANET_STRATUM
            for i in range(n_layers)
        ]
    return [DELTANET_STRATUM] * n_layers


def resolve_moe_layers(model: Any) -> list[tuple[int, Any]]:
    """Return [(layer_idx, mlp_module), ...] for every decoder layer with an mlp.gate.

    Unwraps PeftModel first (model.get_base_model() when present). Tries the
    candidate roots in _MOE_LAYER_ROOT_CANDIDATES in order, picking the FIRST
    that resolves to a non-empty layer list. Raises RuntimeError naming every
    tried path if none resolve -- a silent zero-hook run (the exact 20-04
    failure mode) is impossible.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    tried = []
    for path in _MOE_LAYER_ROOT_CANDIDATES:
        obj = base
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        tried.append(path)
        if obj is None:
            continue
        layers = list(obj)
        if not layers:
            continue
        result = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and getattr(mlp, "gate", None) is not None:
                result.append((idx, mlp))
        return result

    raise RuntimeError(
        f"resolve_moe_layers: no candidate root resolved to a non-empty layer list "
        f"(tried: {tried})"
    )


def infer_dims_from_records(records: list[dict]) -> tuple[int, int]:
    """Return (n_layers, n_experts) inferred from profiling JSONL records.

    n_layers = max(layer_idx) + 1. n_experts = max(int expert-id across the
    expert_counts_* dicts) + 1. Used where no model object is loaded (mask
    extraction, cross-seed load). Returns (0, 0) for an empty record list.
    """
    n_layers = 0
    n_experts = 0
    for rec in records:
        layer_idx = int(rec.get("layer_idx", 0))
        n_layers = max(n_layers, layer_idx + 1)
        for key in ("expert_counts_total", "expert_counts_wp_gen", "expert_counts_wp_judge"):
            counts = rec.get(key) or {}
            for expert_id in counts:
                n_experts = max(n_experts, int(expert_id) + 1)
    return (n_layers, n_experts)


def resolve_task_token_ids(
    tokenizer: Any, default_gen: int, default_judge: int
) -> tuple[int | None, int | None]:
    """Return (gen_id, judge_id): the defaults iff BOTH task tokens are in the
    tokenizer's vocab, else (None, None).

    On a base without the "<wp_gen>"/"<wp_judge>" tokenizer extension (the v4
    judge, vocab 248320, has none -- see 20-04), callers degrade to total-only
    routing tagging instead of silently matching the wrong numeric IDs.
    """
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if "<wp_gen>" in vocab and "<wp_judge>" in vocab:
        return (default_gen, default_judge)
    return (None, None)


def gb10_load_kwargs() -> dict:
    """Return from_pretrained kwargs that are OOM-safe on GB10 unified memory.

    THE TRAP (root cause of the Phase-25-01 profiler OOM, PID 2474645 killed at
    ~62% of a 67 GiB load): `device_map="auto"` inspects available memory via
    torch.cuda.mem_get_info(), which on a GB10 reports the WHOLE 121 GiB unified
    pool as GPU-free. accelerate then treats "cuda:0 ~118 GiB" and "cpu ~110 GiB"
    as two independent pools (~230 GiB) and balances the model across BOTH --
    reserving GPU headroom means part of the model is placed on "cpu". But CPU
    and GPU are the SAME physical RAM here, so the CPU-resident shards (~54 GiB
    anon at death) plus the unified-GPU shards collide and the kernel OOM-kills
    the load. GPU-mapped pages are pinned/unswappable, so swap cannot rescue it.

    THE FIX: pin every module to a SINGLE device so there is no CPU+GPU split,
    and set low_cpu_mem_usage=True so shards stream to that device via meta-init
    instead of a full CPU materialization. A 67 GiB bf16 model then occupies the
    unified pool once (~67 GiB of 121 GiB), leaving room for bs=1 activations.

    CUDA-absent (CPU-only test/merge host) -> {"": "cpu"}, mirroring the merge
    scripts' proven placement.

    Splat into from_pretrained:  AutoModelForX.from_pretrained(path, dtype=...,
    **sieve_arch.gb10_load_kwargs(), trust_remote_code=True)
    Never combine with a separate device_map=/low_cpu_mem_usage= kwarg (the splat
    owns both) -- a duplicate keyword is a TypeError, caught by the load-safety test.
    """
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    return {"device_map": {"": device}, "low_cpu_mem_usage": True}


# ---------------------------------------------------------------------------
# Self-check (--self-check): assert-based, no GPU, no network
# ---------------------------------------------------------------------------


def demo() -> None:
    # arch_dims: v4 (composite text_config) and v3 (plain) configs
    v4_config = {"text_config": {"num_hidden_layers": 40, "num_experts": 256}}
    assert arch_dims(v4_config) == (40, 256)
    v3_config = {"num_hidden_layers": 48, "num_experts": 128}
    assert arch_dims(v3_config) == (48, 128)

    # layer_strata: real v4 pattern -- 10x(3 linear_attention + 1 full_attention)
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 10
    v4_strata_config = {"text_config": {"num_hidden_layers": 40, "layer_types": layer_types}}
    strata = layer_strata(v4_strata_config)
    assert len(strata) == 40
    assert strata.count(ATTENTION_STRATUM) == 10
    assert strata.count(DELTANET_STRATUM) == 30
    attn_idx = [i for i, s in enumerate(strata) if s == ATTENTION_STRATUM]
    assert attn_idx == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39], attn_idx

    # layer_strata: v3 uniform fallback (neither key present) -> all-deltanet, never raises
    assert layer_strata(v3_config) == [DELTANET_STRATUM] * 48

    # resolve_moe_layers: stub model traversal (flat model.layers root)
    class StubMLP:
        def __init__(self):
            self.gate = object()

    class StubLayer:
        def __init__(self):
            self.mlp = StubMLP()

    class StubInner:
        def __init__(self, n):
            self.layers = [StubLayer() for _ in range(n)]

    class StubModel:
        def __init__(self, n):
            self.model = StubInner(n)

    resolved = resolve_moe_layers(StubModel(40))
    assert len(resolved) == 40
    assert resolved[0][0] == 0 and resolved[39][0] == 39

    class EmptyModel:
        pass

    try:
        resolve_moe_layers(EmptyModel())
        raise AssertionError("expected RuntimeError on an empty/unresolvable tree")
    except RuntimeError:
        pass

    # infer_dims_from_records
    records = [
        {"layer_idx": 0, "expert_counts_total": {"0": 5, "255": 2}},
        {"layer_idx": 39, "expert_counts_total": {"1": 3}},
    ]
    assert infer_dims_from_records(records) == (40, 256)
    assert infer_dims_from_records([]) == (0, 0)

    # resolve_task_token_ids
    class StubTokNoTaskTokens:
        def get_vocab(self):
            return {"a": 0, "b": 1}

    assert resolve_task_token_ids(StubTokNoTaskTokens(), 151669, 151670) == (None, None)

    class StubTokWithTaskTokens:
        def get_vocab(self):
            return {"<wp_gen>": 151669, "<wp_judge>": 151670}

    assert resolve_task_token_ids(StubTokWithTaskTokens(), 151669, 151670) == (151669, 151670)

    # gb10_load_kwargs: single-device placement + streaming, never bare "auto"
    # (torch-optional env: skip if torch is absent, the helper needs torch.cuda)
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        kw = gb10_load_kwargs()
        assert set(kw) == {"device_map", "low_cpu_mem_usage"}, kw
        assert kw["low_cpu_mem_usage"] is True
        assert list(kw["device_map"].keys()) == [""], kw
        assert kw["device_map"][""] in (0, "cpu"), kw
        assert kw["device_map"] != "auto"

    print("OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        demo()
    else:
        print(__doc__)
