"""P0 v3: CPU-only raw-HF + PEFT 0.18.1 merge of v1 30_70 LoRA adapter.

Runs inside unsloth-headless container. Produces:
  models/qwen3-30b-wp-30_70-merged-v2/

v3 vs v2 (post-OOM forensics 2026-05-29):
- v2 (Unsloth FastLanguageModel + max_memory={0:'80GiB','cpu':'30GiB'}) OOMed at
  adapter-load step on GB10 unified memory. Root cause: max_memory is an
  accelerate placement HINT, not a hard cap; Unsloth's bf16 from_pretrained path
  pins ~80+ GiB GPU-mapped pages, PEFT untie of tied embeddings + target_parameters
  scratch pushed total NVRM allocation past the 121 GiB unified ceiling.
- v3 removes GPU entirely: device_map={"":"cpu"}, torch_dtype=bfloat16,
  low_cpu_mem_usage=True. Raw HF + PEFT 0.18.1 ParamWrapper.merge() (verified
  fuses target_parameters per RESEARCH §"PEFT 0.18.1 merge" + memory 1668).
- Peak RAM est.: base 60 GiB + adapter 5 GiB + merge scratch 10 GiB = ~75 GiB.
  Floor in host launcher remains 80 GiB but is now ACCURATE (CPU-only path).

Carried 4.3 mitigations:
- load_in_4bit=False (4-bit destroys Qwen3-MoE routing per 4.3-01 SUMMARY)
- atomic rename, idempotent skip-if-exists
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path

BASE = "models/Qwen3-30B-A3B"
ADAPTER = "adapters/qwen3-30b-wp-30_70"
OUT = "models/qwen3-30b-wp-30_70-merged-v2"


def main() -> int:
    # Bound CPU thread allocator BEFORE torch import to keep MKL/OMP from over-grabbing.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Force CPU path; no CUDA context init.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)

    out_path = Path(OUT)
    if out_path.exists() and (out_path / "config.json").exists():
        print(f"[P0v3] IDEMPOTENT-SKIP: {OUT}/config.json already exists. Delete dir to re-merge.")
        return 0

    print(f"[P0v3] Loading base bf16 from {BASE} (CPU-only, low_cpu_mem_usage=True) ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    print(f"[P0v3] Base loaded. param count: {sum(p.numel() for p in base.parameters()) / 1e9:.2f}B")

    print(f"[P0v3] Loading adapter from {ADAPTER} (CPU) ...")
    model = PeftModel.from_pretrained(
        base,
        ADAPTER,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    print("[P0v3] Adapter attached. Running merge_and_unload() ...")

    # PEFT 0.18.1 ParamWrapper.merge() fuses target_parameters into base weights.
    # merge_and_unload() also handles modules_to_save (embed_tokens, lm_head)
    # by swapping in the trained replacements.
    merged = model.merge_and_unload()
    print("[P0v3] Merge complete. Freeing PEFT wrapper ...")

    # Drop the PeftModel reference; merged holds the underlying base with fused weights.
    del model
    gc.collect()

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        print(f"[P0v3] Cleaning stale tmp dir: {tmp_path}")
        shutil.rmtree(tmp_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[P0v3] Saving merged bf16 to {tmp_path} (safe_serialization=True) ...")
    merged.save_pretrained(
        str(tmp_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print(f"[P0v3] Saving tokenizer from adapter dir (carries added special tokens) ...")
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.save_pretrained(str(tmp_path))

    del merged, base, tok
    gc.collect()

    print(f"[P0v3] Atomic rename: {tmp_path} -> {out_path}")
    tmp_path.rename(out_path)

    cfg = out_path / "config.json"
    if not cfg.exists():
        print(f"[P0v3] ERROR: config.json missing after merge at {cfg}")
        return 1

    size_gb = sum(
        f.stat().st_size for f in out_path.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    print(f"[P0v3] DONE. Output size: {size_gb:.1f} GiB at {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
