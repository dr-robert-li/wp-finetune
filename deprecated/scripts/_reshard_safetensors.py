#!/usr/bin/env python
"""04.3-03 memory de-risk: LOW-MEMORY on-disk re-shard of a safetensors model into many
small shards, so a streaming load holds only ~one small shard CPU-side at a time (kills the
mega-shard double-hold that peaks ~100 GiB on the GB10 unified pool).

Reads tensors ONE AT A TIME via safe_open (lazy) and writes ~max-gib buckets — peak RAM is
~one bucket (~5 GiB), NEVER a full model load. Writes model.safetensors.index.json + copies
all non-weight files (config, tokenizer, chat_template, ...). Source dir is NEVER modified.

Usage: python scripts/_reshard_safetensors.py <src_dir> <dst_dir> [--max-gib 5]
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-gib", type=float, default=5.0)
    args = ap.parse_args()
    src, dst = args.src, args.dst
    max_bytes = int(args.max_gib * 2**30)
    os.makedirs(dst, exist_ok=True)

    # Resolve the source shard list (prefer the index weight_map; else any *.safetensors).
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        shard_files = sorted(set(wm.values()))
    else:
        shard_files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    print(f"[reshard] src={src} shards={len(shard_files)} max_bucket={args.max_gib}GiB", flush=True)

    # Stream every tensor in a stable order into <= max_bytes buckets.
    weight_map = {}
    total_size = 0
    bucket, bucket_bytes, part = {}, 0, 1
    out_parts = []  # (filename, [keys]) — filenames finalized after we know the count

    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    pending = []  # list of (key, tensor) flushed lazily to bound memory
    # We cannot know the final part count up front for the HF naming convention
    # (model-XXXXX-of-NNNNN), so first pass: write provisional parts, then rename.
    provisional = []

    def flush(bkt):
        nonlocal part
        fname = f"__part_{part:05d}.safetensors"
        save_file(bkt, os.path.join(dst, fname), metadata={"format": "pt"})
        provisional.append((fname, list(bkt.keys())))
        print(f"[reshard] wrote {fname} ({len(bkt)} tensors, "
              f"{sum(tensor_nbytes(v) for v in bkt.values())/2**30:.2f} GiB)", flush=True)
        part += 1

    for sf in shard_files:
        with safe_open(os.path.join(src, sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                nb = tensor_nbytes(t)
                if bucket_bytes + nb > max_bytes and bucket:
                    flush(bucket)
                    bucket, bucket_bytes = {}, 0
                bucket[key] = t
                bucket_bytes += nb
                total_size += nb
    if bucket:
        flush(bucket)

    # Finalize HF shard names model-XXXXX-of-NNNNN.safetensors + index.
    n = len(provisional)
    for i, (prov, keys) in enumerate(provisional, start=1):
        final = f"model-{i:05d}-of-{n:05d}.safetensors"
        os.rename(os.path.join(dst, prov), os.path.join(dst, final))
        for k in keys:
            weight_map[k] = final
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        indent=2,
    )
    print(f"[reshard] index written: {len(weight_map)} tensors, "
          f"{total_size/2**30:.1f} GiB across {n} shards", flush=True)

    # Copy every non-weight file (config, tokenizer, generation_config, chat_template, ...).
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, fn))
            print(f"[reshard] copied {fn}", flush=True)
    print("[reshard] DONE", flush=True)


if __name__ == "__main__":
    main()
