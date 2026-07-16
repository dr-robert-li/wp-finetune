"""Auto-loaded via PYTHONPATH inside the vLLM serving container (Phase 25 routing
profile). READ-side sibling of scripts/_sieve_vllm_patch/sitecustomize.py.

The mask patch hooks every MoE block's `self.gate` to WRITE (-inf on masked
experts). This patch hooks the SAME gate to READ: it accumulates a per-layer,
per-expert selection count over every forward pass, so the v4 judge's routing
distribution can be profiled by SERVING the model (via vLLM's own memory manager,
which fits the 121 GiB GB10 pool) instead of an in-process from_pretrained load
(which OOMs -- see .planning/debug/v4-judge-load-oom-recurrence.md).

Counting contract is byte-identical to the in-process reference
scripts.profile_base_model.RoutingCollector.make_hook:
  router_logits = gate(...)                 # [n_tokens, n_experts]
  idx = topk(router_logits, k=TOPK).indices # [n_tokens, TOPK]
  counts[layer][e] += 1  for each e in idx  # per-expert top-k selection frequency
The downstream E_eff/Jaccard math (scripts.drive_v4_routing_profile finalize)
reconstructs a RoutingCollector from the dumped [n_layers, n_experts] array and
calls the existing compute_eeff / compute_jaccard_stability / write_profiling_jsonl
UNCHANGED, so the served profile is directly comparable to the merged-model one.

v4 judge has NO <wp_gen>/<wp_judge> task-token extension, so the reference
collector degrades to total-only tagging (gen_id/judge_id=None -> every non-pad
token is "other" -> counted in _counts_total). This patch is therefore total-only
by construction: no per-token-position type alignment is needed inside vLLM's
flattened continuous batch. Serve JUDGE prompts only.

MUST serve with --enforce-eager: the hook runs Python per forward, which vLLM
skips during CUDA-graph REPLAY. Eager mode fires the hook on every real forward.
(The mask patch avoids this by editing the gate output tensor -- a captured op
that replays -- but a counting hook cannot express accumulation as a pure
replayable graph op without fixed token-shape gymnastics, so we disable capture.)

MoE-block class resolution reuses the mask patch's ordered candidate list
(qwen3_5_moe/qwen3_next FIRST for v4, qwen3_moe fallback for v3).

Env:
  SIEVE_PROFILE_OUT   (required) host->container path for the counts .npy dump.
  SIEVE_PROFILE_TOPK  (default 8) experts counted per token; MUST match
                      profile_v4_judge.py's top_k_jaccard (8) for comparability.
  SIEVE_PROFILE_FLUSH_SECS (default 5) background snapshot cadence (safety: the
                      final total is authoritative, periodic flush guards against
                      a missed shutdown signal).

Absent SIEVE_PROFILE_OUT -> this file does nothing (model serves normally).
"""
import atexit
import importlib
import os
import sys
import threading

_MOE_BLOCK_CANDIDATES = [
    ("vllm.model_executor.models.qwen3_5_moe", "Qwen3_5MoeSparseMoeBlock"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextSparseMoeBlock"),
    ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeSparseMoeBlock"),
]


def _resolve_moe_block_class(candidates):
    """First resolvable MoE-block class from an ordered candidate list.

    Mirrors the mask patch resolver: import module, take class_name, else the
    single *SparseMoeBlock class present; raise naming every tried candidate.
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
    raise RuntimeError(f"[sieve-profile] no MoE-block class resolved (tried: {tried})")


class _CountStore:
    """Per-layer expert-selection counters, lazily sized to the gate width.

    Counters live on the gate's device (GPU) as float32 [n_experts] and are
    updated with a pure scatter_add of the top-k indices -- no per-token Python
    loop, no .item()/.tolist() in the hot path. A background thread snapshots a
    CPU copy to disk every FLUSH_SECS; the final atexit flush is authoritative.
    """

    def __init__(self, out_path: str, top_k: int, flush_secs: float):
        self.out_path = out_path
        self.top_k = top_k
        self.flush_secs = flush_secs
        self._counts = {}      # layer_idx -> torch.FloatTensor [n_experts] (device)
        self._n_experts = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._started = False

    def _ensure_thread(self):
        if self._started or self.flush_secs <= 0:
            return
        self._started = True
        t = threading.Thread(target=self._flush_loop, name="sieve-profile-flush", daemon=True)
        t.start()

    def add(self, layer_idx, router_logits):
        import torch
        if router_logits.dim() != 2:
            router_logits = router_logits.reshape(-1, router_logits.shape[-1])
        n_experts = router_logits.shape[-1]
        idx = torch.topk(router_logits, k=min(self.top_k, n_experts), dim=-1).indices.reshape(-1)
        with self._lock:
            if self._n_experts is None:
                self._n_experts = n_experts
            buf = self._counts.get(layer_idx)
            if buf is None:
                buf = torch.zeros(n_experts, dtype=torch.float32, device=router_logits.device)
                self._counts[layer_idx] = buf
            buf.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        self._ensure_thread()

    def _snapshot(self):
        """Return a dense [n_layers, n_experts] numpy array from current counters."""
        import numpy as np
        with self._lock:
            if not self._counts or self._n_experts is None:
                return None
            n_layers = max(self._counts) + 1
            arr = np.zeros((n_layers, self._n_experts), dtype=float)
            for layer_idx, buf in self._counts.items():
                arr[layer_idx] = buf.detach().to("cpu").numpy()
        return arr

    def flush(self):
        import numpy as np
        arr = self._snapshot()
        if arr is None:
            return
        tmp = self.out_path + ".tmp"
        np.save(tmp, arr)
        os.replace(tmp, self.out_path)

    def _flush_loop(self):
        while not self._stopped.wait(self.flush_secs):
            try:
                self.flush()
            except Exception as exc:  # never let a flush error kill serving
                print(f"[sieve-profile] flush error (non-fatal): {exc}", file=sys.stderr, flush=True)

    def close(self):
        self._stopped.set()
        try:
            self.flush()
            print(f"[sieve-profile] final flush -> {self.out_path}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[sieve-profile] final flush error: {exc}", file=sys.stderr, flush=True)


def _install(out_path: str) -> None:
    top_k = int(os.environ.get("SIEVE_PROFILE_TOPK", "8"))
    flush_secs = float(os.environ.get("SIEVE_PROFILE_FLUSH_SECS", "5"))
    store = _CountStore(out_path, top_k=top_k, flush_secs=flush_secs)
    atexit.register(store.close)

    moe_block_cls = _resolve_moe_block_class(_MOE_BLOCK_CANDIDATES)
    orig_init = moe_block_cls.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        layer_idx = getattr(self.experts, "layer_id", None)
        if layer_idx is None:
            print("[sieve-profile] WARNING: block has no experts.layer_id; layer unprofiled",
                  file=sys.stderr, flush=True)
            return

        def _hook(module, inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            store.add(layer_idx, logits)

        self.gate.register_forward_hook(_hook)

    moe_block_cls.__init__ = _patched_init
    print(f"[sieve-profile] patched {moe_block_cls.__module__}.{moe_block_cls.__name__} "
          f"-- out={out_path} top_k={top_k} flush={flush_secs}s (serve with --enforce-eager)",
          flush=True)


_OUT_PATH = os.environ.get("SIEVE_PROFILE_OUT")
if _OUT_PATH:
    _install(_OUT_PATH)
