"""Auto-loaded via PYTHONPATH inside the vLLM serving container (SIEVE-04 k-sweep).

Applies the k-sweep expert mask at inference: for every Qwen3-MoE layer, forces
the router (gate) logit of masked-out experts to -inf before vLLM's internal
softmax+top-k expert selection runs. -inf logit -> ~0 softmax weight -> the
remaining (kept) experts' softmax renormalizes automatically over just the
kept set -- this is the exact renormalization semantics of
scripts/sieve_expert_mask_inference.apply_mask, applied as a live forward hook
instead of a standalone tensor op (that module lives on the host and is not
importable inside this container, so the -inf masking math is duplicated here
in ~10 lines rather than shared -- see that module's docstring for the proof).

No weights are modified, no gradients, no training: this is a runtime routing
mask only, uninstalled the moment the container is stopped.

Enabled only when SIEVE_KEEP_MASK_NPY points to a readable [n_layers,
n_experts] bool .npy keep-mask (bind-mounted read-only into the container by
scripts/serve_30_70_vllm.sh when SIEVE_MASK_NPY is set on the host). Absent ->
this file does nothing and the model serves fully unmasked (the k="full" arm).

Fails LOUD (raises) if the mask env var is set but the patch cannot install --
serving a "masked" arm that is secretly unmasked would silently corrupt the
k-sweep (T-11-08), so we'd rather the container fail to boot.
"""
import os
import sys

_MASK_PATH = os.environ.get("SIEVE_KEEP_MASK_NPY")

if _MASK_PATH:
    if not os.path.exists(_MASK_PATH):
        raise RuntimeError(f"[sieve-mask] SIEVE_KEEP_MASK_NPY set but not found: {_MASK_PATH}")

    import numpy as np

    _keep = np.load(_MASK_PATH)  # [n_layers, n_experts] bool
    _NEG = -1.0e9

    def _install() -> None:
        import torch
        from vllm.model_executor.models import qwen3_moe as _q3

        _orig_init = _q3.Qwen3MoeSparseMoeBlock.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            layer_idx = self.experts.layer_id
            if layer_idx is None or layer_idx >= _keep.shape[0]:
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
            row_np = _keep[layer_idx]
            target_device = self.gate.weight.device
            add = torch.zeros(row_np.shape[0], dtype=torch.float32, device=target_device)
            if not row_np.all():
                masked_idx = torch.tensor(np.where(~row_np)[0], dtype=torch.long,
                                           device=target_device)
                add.index_fill_(0, masked_idx, _NEG)
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

        _q3.Qwen3MoeSparseMoeBlock.__init__ = _patched_init
        print(f"[sieve-mask] patched Qwen3MoeSparseMoeBlock -- mask={_MASK_PATH}", flush=True)

    try:
        _install()
    except Exception as e:  # pragma: no cover - must not silently no-op
        print(f"[sieve-mask] FATAL: failed to install expert-mask patch: {e}",
              file=sys.stderr, flush=True)
        raise
