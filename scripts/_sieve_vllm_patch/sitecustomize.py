"""Auto-loaded via PYTHONPATH inside the vLLM serving container (SIEVE-04 k-sweep).

Applies the k-sweep expert mask at inference: for every MoE layer, forces the
router (gate) logit of masked-out experts to -inf before vLLM's internal
softmax+top-k expert selection runs. -inf logit -> ~0 softmax weight -> the
remaining (kept) experts' softmax renormalizes automatically over just the
kept set -- this is the exact renormalization semantics of
scripts/sieve_expert_mask_inference.apply_mask, applied as a live forward hook
instead of a standalone tensor op (that module lives on the host and is not
importable inside this container, so the -inf masking math is duplicated here
in ~10 lines rather than shared -- see that module's docstring for the proof).

No weights are modified, no gradients, no training: this is a runtime routing
mask only, uninstalled the moment the container is stopped.

MoE-block class resolution (GATE4-02 T-22-02): the v4 base (Qwen3.6-35B-A3B,
model_type qwen3_5_moe) is served by a DIFFERENT vLLM model-executor class than
v3's Qwen3MoeSparseMoeBlock. _resolve_moe_block_class() walks an ORDERED
candidate list -- qwen3_5_moe/qwen3_next (v4) FIRST, qwen3_moe (v3) fallback --
importing each and picking the first that resolves to a wrappable class. The
exact qwen3_5_moe/qwen3_next module+class names are BEST-KNOWN, not yet
confirmed against the installed vLLM (that confirmation happens inside the
serving container in Plan 22-02 / Phase 25, not on this host -- vLLM is not
importable here). The resolver additionally tolerates an unknown-but-present
class by scanning a resolved module for a single *SparseMoeBlock class, so an
exact class-name miss still resolves as long as the module itself is right.

Enabled only when SIEVE_KEEP_MASK_NPY points to a readable [n_layers,
n_experts] bool .npy keep-mask (bind-mounted read-only into the container by
scripts/serve_30_70_vllm.sh when SIEVE_MASK_NPY is set on the host). Absent ->
this file does nothing and the model serves fully unmasked (the k="full" arm).

Fails LOUD (raises) if the mask env var is set but no candidate MoE-block class
resolves -- serving a "masked" arm that is secretly unmasked would silently
corrupt the k-sweep (T-11-08), so we'd rather the container fail to boot. This
is the SAME discipline as v3, extended to the new class.
"""
import importlib
import os
import sys

# Ordered candidate list: (module_path, class_name). qwen3_5_moe/qwen3_next (v4
# base) tried FIRST, qwen3_moe (v3) as fallback. Exact v4 module/class names are
# best-known pending live confirmation inside the serving container (22-02).
_MOE_BLOCK_CANDIDATES = [
    ("vllm.model_executor.models.qwen3_5_moe", "Qwen3_5MoeSparseMoeBlock"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextSparseMoeBlock"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"),
]


def _resolve_moe_block_class(candidates):
    """Return the first resolvable MoE-block class from an ordered candidate list.

    Each candidate is (module_path, class_name). A candidate resolves if:
      1. module_path imports successfully, AND
      2a. class_name exists on the module, OR
      2b. exactly one *SparseMoeBlock class exists on the module (tolerant
          fallback for an unknown-but-present class name).

    Tolerates ImportError/AttributeError per candidate (tries the next one).
    Raises RuntimeError naming every tried candidate if none resolve.
    """
    tried = []
    for module_path, class_name in candidates:
        tried.append(f"{module_path}.{class_name}")
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is not None:
            return cls
        scanned = [
            v for k, v in vars(mod).items()
            if k.endswith("SparseMoeBlock") and isinstance(v, type)
        ]
        if len(scanned) == 1:
            return scanned[0]
    raise RuntimeError(
        f"[sieve-mask] no MoE-block class resolved (tried: {tried})"
    )


def _install(mask_path: str) -> None:
    import numpy as np
    import torch

    keep = np.load(mask_path)  # [n_layers, n_experts] bool
    neg = -1.0e9

    moe_block_cls = _resolve_moe_block_class(_MOE_BLOCK_CANDIDATES)
    orig_init = moe_block_cls.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        layer_idx = self.experts.layer_id
        if layer_idx is None or layer_idx >= keep.shape[0]:
            print(f"[sieve-mask] WARNING: no mask row for layer_idx={layer_idx}, "
                  "serving unmasked for this layer", file=sys.stderr, flush=True)
            return
        # NOTE 1: vLLM's model __init__ runs under a torch "default device"
        # context (torch.utils._device) that silently pins bare
        # torch.tensor(...) calls to cuda:0 -- mixing that with a genuinely
        # CPU torch.from_numpy(...) in the same op crashes ("two devices").
        # NOTE 2: vLLM captures CUDA graphs for this forward path (warmup +
        # real serving), which forbids any *unpinned* CPU->CUDA copy inside
        # the captured region -- so the additive mask tensor must be built
        # ONCE here (explicit device=self.gate.weight.device, no ambient
        # context, no per-call H2D copy) rather than per-forward-call
        # inside the hook.
        row_np = keep[layer_idx]
        target_device = self.gate.weight.device
        add = torch.zeros(row_np.shape[0], dtype=torch.float32, device=target_device)
        if not row_np.all():
            masked_idx = torch.tensor(np.where(~row_np)[0], dtype=torch.long,
                                       device=target_device)
            add.index_fill_(0, masked_idx, neg)
        n_kept = int(row_np.sum())
        n_total = int(row_np.size)

        def _hook(module, inp, out):
            ref = out[0] if isinstance(out, tuple) else out
            add_ref = add if add.dtype == ref.dtype else add.to(dtype=ref.dtype)
            if isinstance(out, tuple):
                return (ref + add_ref,) + tuple(out[1:])
            return ref + add_ref

        self.gate.register_forward_hook(_hook)
        print(f"[sieve-mask] layer {layer_idx}: kept {n_kept}/{n_total} experts", flush=True)

    moe_block_cls.__init__ = _patched_init
    print(f"[sieve-mask] patched {moe_block_cls.__module__}.{moe_block_cls.__name__} "
          f"-- mask={mask_path}", flush=True)


_MASK_PATH = os.environ.get("SIEVE_KEEP_MASK_NPY")

if _MASK_PATH:
    if not os.path.exists(_MASK_PATH):
        raise RuntimeError(f"[sieve-mask] SIEVE_KEEP_MASK_NPY set but not found: {_MASK_PATH}")

    try:
        _install(_MASK_PATH)
    except Exception as e:  # pragma: no cover - must not silently no-op
        print(f"[sieve-mask] FATAL: failed to install expert-mask patch: {e}",
              file=sys.stderr, flush=True)
        raise
