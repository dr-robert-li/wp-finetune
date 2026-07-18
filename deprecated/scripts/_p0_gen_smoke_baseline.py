"""PR2b: generate + commit baseline (merged-v2) outputs for the smoke no-op canary.

Boots vLLM on the v3 baseline (models/qwen3-30b-wp-30_70-merged-v2), generates
the 10 manifest prompts at temperature=0, writes data/phase4_4/smoke_baseline_outputs.json.
Committed artifact: deterministic reference for Stage 2's baseline-similarity check
(merge no-op canary) — no second live vLLM instance needed at smoke time.

Run once (needs GPU + ~15 min vLLM boot). Re-run only if the baseline model changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm  # noqa: E402

BASELINE_MODEL = "models/qwen3-30b-wp-30_70-merged-v2"
MANIFEST = "data/phase4_4/smoke_prompts.json"
OUT = "data/phase4_4/smoke_baseline_outputs.json"
NAME = "wp-smoke-baseline-vllm"
PORT = 8011


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASELINE_MODEL)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    print(f"[baseline-gen] model={args.model} prompts={len(manifest)}")
    if args.dry_run:
        print("[baseline-gen] dry-run: manifest OK, serve helper importable")
        return 0

    try:
        boot_vllm(args.model, NAME, PORT, args.gpu_mem_util)
        served = wait_healthy(PORT, NAME)
        outs = generate(PORT, served, manifest, args.max_tokens)
    finally:
        stop_vllm(NAME)

    records = []
    for p, o in zip(manifest, outs):
        records.append({
            "source_val_idx": p["source_val_idx"],
            "kind": p["kind"],
            "task_token": p["task_token"],
            "baseline_output": o,
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"baseline_model": args.model, "n": len(records), "outputs": records},
              open(args.output, "w"), indent=2)
    empty = sum(1 for r in records if not r["baseline_output"].strip())
    print(f"[baseline-gen] wrote {len(records)} -> {args.output} ({empty} empty)")
    return 1 if empty else 0


if __name__ == "__main__":
    sys.exit(main())
