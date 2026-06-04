#!/usr/bin/env python
"""04.3-03 memory de-risk: does a PLAIN transformers BitsAndBytesConfig 4-bit load of the
57GB bf16 merged-v2 base free shards promptly (peak ~25 GB) instead of Unsloth's ~100 GB?

If yes: pre-quantize merged-v2 once on the live desktop (safe), then the Task-2 4-bit arms
load 4-bit directly. Run under scripts/_binding_dryrun_watchdog.sh-style free-RAM watchdog.

  --save DIR   after loading, save_pretrained the 4-bit model to DIR (the pre-quant step).
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/qwen3-30b-wp-30_70-merged-v2")
    ap.add_argument("--save", default=None, help="save the 4-bit model to this dir (pre-quant)")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    print(f"[tf-4bit] loading {args.base} via transformers BitsAndBytesConfig "
          f"(device_map=cuda:0, low_cpu_mem_usage=True) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},          # force all on device 0; no max_memory -> no pre-alloc bloat
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"[tf-4bit] LOADED in {time.time()-t0:.0f}s. resident footprint check:", flush=True)
    try:
        nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"[tf-4bit]   param bytes ~= {nbytes/1e9:.2f} GB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tf-4bit]   (footprint calc failed: {e})", flush=True)

    if args.save:
        print(f"[tf-4bit] saving 4-bit checkpoint to {args.save} ...", flush=True)
        model.save_pretrained(args.save)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.base).save_pretrained(args.save)
        print(f"[tf-4bit] SAVED 4-bit checkpoint to {args.save}", flush=True)

    print("[tf-4bit] DONE", flush=True)


if __name__ == "__main__":
    main()
