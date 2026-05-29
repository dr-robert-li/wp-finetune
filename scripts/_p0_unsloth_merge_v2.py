"""P0: Re-merge v1 30_70 LoRA adapter via Unsloth FastLanguageModel.

Runs inside unsloth-headless container. Produces:
  models/qwen3-30b-wp-30_70-merged-v2/

This corrects the partial-30_70 baseline (raw-PEFT merge silently dropped the
MoE-expert target_parameters per RESEARCH Pitfall 5). PEFT 0.18.1 ParamWrapper.merge()
now correctly fuses target_parameters; Unsloth save_pretrained_merged(save_method=
'merged_16bit') invokes that path.

Carried 4.3 mitigation patterns:
- load_in_4bit=False (4-bit destroys Qwen3-MoE routing per 4.3-01 SUMMARY)
- max_memory cap bounds peak allocation on GB10 unified memory
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

BASE = "models/Qwen3-30B-A3B"
ADAPTER = "adapters/qwen3-30b-wp-30_70"
OUT = "models/qwen3-30b-wp-30_70-merged-v2"


def main() -> int:
    import torch
    from unsloth import FastLanguageModel

    out_path = Path(OUT)
    if out_path.exists() and (out_path / "config.json").exists():
        print(f"[P0] IDEMPOTENT-SKIP: {OUT}/config.json already exists. Delete dir to re-merge.")
        return 0

    max_memory = {0: "80GiB", "cpu": "30GiB"}
    print(
        f"[P0] Loading base bf16 from {BASE} "
        f"(load_in_4bit=False, max_memory={max_memory}) ..."
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        max_memory=max_memory,
    )

    print(f"[P0] Loading adapter from {ADAPTER} ...")
    model.load_adapter(ADAPTER)

    # Stage to a tmp dir, then atomic rename to canonical path.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        print(f"[P0] Cleaning stale tmp dir: {tmp_path}")
        shutil.rmtree(tmp_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[P0] Merging + saving bf16 (save_method='merged_16bit') to {tmp_path} ...")
    model.save_pretrained_merged(str(tmp_path), tokenizer, save_method="merged_16bit")

    print(f"[P0] Atomic rename: {tmp_path} -> {out_path}")
    tmp_path.rename(out_path)

    cfg = out_path / "config.json"
    if not cfg.exists():
        print(f"[P0] ERROR: config.json missing after merge at {cfg}")
        return 1

    size_gb = sum(
        f.stat().st_size for f in out_path.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    print(f"[P0] DONE. Output size: {size_gb:.1f} GiB at {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
