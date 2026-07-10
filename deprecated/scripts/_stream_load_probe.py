#!/usr/bin/env python
"""04.3-03 memory de-risk: does a STREAMING bf16 load avoid the ~100 GiB unified-memory
double-hold? Loads a model with transformers AutoModelForCausalLM using device_map={"":0}
+ low_cpu_mem_usage=True (shard-by-shard placement, CPU source buffer freed per shard) —
the standard accelerate streaming path. Prints LOAD COMPLETE + peak if it finishes; the
host-side curve watchdog records the available-RAM trajectory. NO binding probe, NO capture
— this only answers "does streaming placement plateau near the weight size (~57-63 GiB) or
climb toward the OOM floor like the Unsloth/quantizing loaders?".
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM

path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3-30b-wp-30_70-merged-v2"
print(f"[stream-probe] loading {path} "
      f"(AutoModelForCausalLM, dtype=bf16, device_map={{'':0}}, low_cpu_mem_usage=True) ...",
      flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
dt = time.time() - t0
print(f"[stream-probe] LOAD COMPLETE in {dt:.1f}s", flush=True)
try:
    print(f"[stream-probe] cuda max_memory_allocated="
          f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
except Exception as e:
    print(f"[stream-probe] (cuda mem stat unavailable: {e})", flush=True)

import gc
del model
gc.collect()
torch.cuda.empty_cache()
print("[stream-probe] released", flush=True)
