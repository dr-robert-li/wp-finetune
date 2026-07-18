#!/usr/bin/env python
"""Phase 23-02 extension: merge s0 and s2 promoted judge-v4 adapters onto the
local base, reusing build_judge03_merge_serve.py's proven
_download_promoted_adapter/_run_merge (240/240 guard, fused-expert merge) --
unchanged, just parameterized for seeds 0 and 2 instead of the single
promoted seed (s1, already merged at
models/Qwen3.6-35B-A3B-judge-v4-s1-merged). Sequential, CPU, one at a time.

Usage: .venv-tinker/bin/python scripts/eval4_ext_merge_seeds.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_judge03_merge_serve import _download_promoted_adapter, _run_merge, _merged_dir, _dir_size_gib

OUT = PROJECT_ROOT / "output" / "eval4" / "ext_q8"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in (0, 2):
        t0 = time.time()
        print(f"=== merging seed {seed} ===", flush=True)
        sampler_path = _download_promoted_adapter(seed)
        guard = _run_merge(seed)
        merged_dir = _merged_dir(seed)
        merge_ok = (guard["merged_target_module_count"] == guard["expected_target_module_count"]
                    and guard["merged_target_module_count"] > 0)
        results[f"s{seed}"] = {
            "sampler_path": sampler_path,
            "merged_dir": merged_dir,
            "merge_ok": merge_ok,
            "guard": guard,
            "merged_size_gib": _dir_size_gib(merged_dir),
            "duration_sec": round(time.time() - t0, 1),
        }
        print(f"=== seed {seed} done: merge_ok={merge_ok} size={results[f's{seed}']['merged_size_gib']}GiB "
              f"({results[f's{seed}']['duration_sec']}s) ===", flush=True)
        if not merge_ok:
            raise RuntimeError(f"seed {seed} merge guard failed: {guard}")

    out_path = OUT / "merge_s0_s2_manifest.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
