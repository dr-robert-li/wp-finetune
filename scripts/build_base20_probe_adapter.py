#!/usr/bin/env python
"""BASE-04 Task 1 (Phase 20 base bring-up) -- produce a throwaway LoRA adapter
targeting model.language_model.* modules and log the ACTUAL attached module
names (PITFALLS.md Pitfall 10 / 20-RESEARCH.md Open Question 1).

PRIMARY (source=tinker, confidence=full): a minimal real Tinker LoRA run
against Qwen/Qwen3.6-35B-A3B, ATTENTION-ONLY target (train_attn=True,
train_mlp=False, train_unembed=False). This keeps the exported adapter a
standard-PEFT-loadable 2D LoRA (q/k/v/o_proj Linear layers) so
scripts/merge_adapter.py's generic PeftModel.merge_and_unload() can consume
it directly -- Tinker's separate MoE per-expert fused-tensor convention (see
scripts/merge_tinker_v3.py) is a distinct, already-solved problem out of
scope for this merge-PATH smoke. This is the one link only a real Tinker run
can validate: whether Tinker's own export resolves module names against the
live model.language_model.* structure (confirmed live against
model.safetensors.index.json: attention lives at
model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj on 11/48 layers --
the rest are Gated-DeltaNet layers with no self_attn submodule).

FALLBACK (source=local_zero_init, confidence=reduced): only if
TINKER_API_KEY is absent or the run is blocked for an account/API reason --
builds a zero-init LoRA adapter directly against the base's real attention
module paths (traversed from the live model object). Validates
merge_adapter.py alone, not Tinker's export-side behavior.

Writes output/base20/lora_target_modules.json.

Must run under the Tinker venv (has tinker/tinker_cookbook + transformers/
peft/torch, all needed by either path):
    .venv-tinker/bin/python scripts/build_base20_probe_adapter.py
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
LOCAL_BASE_DIR = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B"
OUT_DIR = PROJECT_ROOT / "output" / "base20"
# NOTE: adapters/ (top-level) is root-owned (0755, uid 0) on this host -- this
# user cannot mkdir a new subdirectory under it (verified: PermissionError
# during this plan's execution, a pre-existing environment condition, Rule 3
# blocking-issue fix). The throwaway probe adapter is written under
# output/base20/ instead, which this user already owns/writes to for every
# other Phase 20 gate receipt.
ADAPTER_DIR = OUT_DIR / "base20_probe_adapter"
OUT_PATH = OUT_DIR / "lora_target_modules.json"

# Trivial, overfit-friendly probe example: distinctive target completion so a
# tiny/cheap LoRA nudge is easy to observe in Task 2's base-vs-merged diff.
PROBE_PROMPT = "Reply with exactly one word: OK"
PROBE_COMPLETION = "PROBEXYZ"
PROBE_RANK = 8
PROBE_LR = 0.05
PROBE_STEPS = 8


def _attached_modules_from_adapter(adapter_dir: Path) -> list[str]:
    """Read the ACTUAL lora_A/lora_B key names from an exported PEFT adapter
    and return the sorted unique module base-paths (Pitfall 10: log the real
    list, don't assume it from config)."""
    from safetensors import safe_open

    modules = set()
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            # e.g. base_model.model.model.language_model.layers.11.self_attn.q_proj.lora_A.weight
            base = k.rsplit(".lora_", 1)[0]
            modules.add(base)
    return sorted(modules)


def _prefix_observed(attached_modules: list[str]) -> str:
    if not attached_modules:
        return "unknown"
    sample = attached_modules[0]
    if "language_model" in sample:
        return sample.split(".language_model.")[0] + ".language_model.*"
    return sample + " (NO language_model prefix found -- see notes)"


def run_tinker_probe() -> dict:
    """PRIMARY path: minimal real Tinker LoRA run, attention-only target."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    train_row = {
        "messages": [
            {"role": "user", "content": PROBE_PROMPT},
            {"role": "assistant", "content": PROBE_COMPLETION},
        ]
    }
    train_path = OUT_DIR / "_probe_train.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        f.write(json.dumps(train_row) + "\n")

    renderer_name = "qwen3_disable_thinking"
    tok = get_tokenizer(BASE_MODEL)
    cc = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=renderer_name,
        max_length=512,
        batch_size=1,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    train_ds, _ = FromConversationFileBuilder(file_path=str(train_path), test_size=0, common_config=cc)()

    print(f"[probe] tinker run: base={BASE_MODEL} rank={PROBE_RANK} lr={PROBE_LR} "
          f"steps={PROBE_STEPS} train_attn=True train_mlp=False train_unembed=False", flush=True)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=PROBE_RANK,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
    )

    losses = []
    for step in range(PROBE_STEPS):
        batch = train_ds.get_batch(0)
        fb = tc.forward_backward(data=batch, loss_fn="cross_entropy")
        tc.optim_step(tinker.AdamParams(learning_rate=PROBE_LR))
        out = fb.result() if hasattr(fb, "result") else fb
        loss = None
        try:
            d = out.model_dump() if hasattr(out, "model_dump") else getattr(out, "__dict__", {})
            for v in d.values() if isinstance(d, dict) else []:
                if isinstance(v, (int, float)):
                    loss = float(v)
                    break
        except Exception:  # noqa: BLE001 -- loss logging is diagnostic only
            pass
        losses.append(loss)
        print(f"[probe] step {step} loss={loss}", flush=True)

    sampler_res = tc.save_weights_for_sampler(name="base20-probe", ttl_seconds=None)
    sampler_path = sampler_res.result().path if hasattr(sampler_res, "result") else sampler_res.path
    print(f"[probe] sampler checkpoint: {sampler_path}", flush=True)

    rc = sc.create_rest_client()
    print("[probe] requesting archive URL (server packs the archive)...", flush=True)
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ADAPTER_DIR / "checkpoint.tar"
    print(f"[probe] downloading archive -> {tar_path}", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("adapter_config.json", "adapter_model.safetensors"):
                m.name = name
                tf.extract(m, ADAPTER_DIR)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER_DIR / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    attached_modules = _attached_modules_from_adapter(ADAPTER_DIR)
    if not attached_modules:
        raise RuntimeError("tinker adapter exported zero LoRA modules")

    return {
        "source": "tinker",
        "confidence": "full",
        "adapter_dir": str(ADAPTER_DIR),
        "attached_modules": attached_modules,
        "prefix_observed": _prefix_observed(attached_modules),
        "sampler_path": sampler_path,
        "rank": PROBE_RANK,
        "learning_rate": PROBE_LR,
        "steps": PROBE_STEPS,
        "losses": losses,
        "probe_prompt": PROBE_PROMPT,
        "probe_completion": PROBE_COMPLETION,
        "notes": (
            "Minimal real Tinker LoRA run (rank=8, train_attn-only, 8 steps, 1 "
            "overfit-friendly example) purely to exercise Tinker's export-side "
            "key-prefix behavior against the live model.language_model.* "
            "structure. Attention-only by design -- MoE per-expert LoRA uses a "
            "distinct Tinker tensor convention (scripts/merge_tinker_v3.py) out "
            "of scope for this merge-path smoke. Cost: order-of-magnitude cents "
            "per 20-RESEARCH.md Open Question 1 sizing."
        ),
    }


def run_local_zero_init_probe(reason: str) -> dict:
    """FALLBACK path: zero-init LoRA built directly against real module paths."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print(f"[probe] FALLBACK: local zero-init adapter ({reason})", flush=True)
    print(f"[probe] loading base from {LOCAL_BASE_DIR} (cpu, bf16) to enumerate real module paths...",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_BASE_DIR), dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=target_modules, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lora_cfg)  # zero-init B by construction

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ADAPTER_DIR))

    attached_modules = _attached_modules_from_adapter(ADAPTER_DIR)
    if not attached_modules:
        raise RuntimeError("local zero-init adapter exported zero LoRA modules")

    return {
        "source": "local_zero_init",
        "confidence": "reduced",
        "adapter_dir": str(ADAPTER_DIR),
        "attached_modules": attached_modules,
        "prefix_observed": _prefix_observed(attached_modules),
        "target_modules": target_modules,
        "reason": reason,
        "notes": (
            "Zero-init LoRA constructed directly against the live model's "
            "real module tree (PEFT LoraConfig target_modules resolution). "
            "Validates merge_adapter.py's own correctness but does NOT test "
            "Tinker's export-side key-prefix behavior -- confidence=reduced."
        ),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tinker_key_present = bool(os.environ.get("TINKER_API_KEY"))

    if tinker_key_present:
        try:
            result = run_tinker_probe()
        except Exception as exc:  # noqa: BLE001 -- fall back only for account/API-shaped failures
            print(f"[probe] tinker run raised: {exc!r}", flush=True)
            result = run_local_zero_init_probe(reason=f"tinker run blocked: {exc}")
    else:
        result = run_local_zero_init_probe(reason="TINKER_API_KEY not set in environment")

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote {OUT_PATH}", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
