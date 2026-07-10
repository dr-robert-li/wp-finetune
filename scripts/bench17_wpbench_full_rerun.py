"""BENCH-01 (Phase 17-01): full 344-test wp-bench rerun on the v1.2 gen model,
served via vLLM bf16 on port 8020 — the same shipping stack that produced the
0.4484 Gate-1 figure (traced via output/eval3/eval3_final_comparison.json ->
output/sieve/optimal_k.json full_arm.wp_bench, itself produced by
scripts/run_eval_reasoning.py::_run_wpbench through scripts/sieve_ksweep_run.py).

Reuses that exact function (same request_timeout=1800s, max_tokens=2048,
concurrency=4, enable_thinking=False via scripts/_wpbench_pth, temperature=0.0
default) so the fresh number is comparable. Only the port differs (8020, the
dgx_toolbox.yaml/config/wp-bench.yaml documented default) vs. the sieve driver's
port 8021 — cosmetic, not a stack difference.

Adds the Phase 15 LOCKED lesson: gate capture on a REAL one-token(+) generation
succeeding, not vLLM's /v1/models health response.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout  # noqa: E402
import scripts.run_eval_reasoning as rer  # noqa: E402

MODEL_DIR = "models/qwen3-30b-wp-30_70-reasoning-merged-v4"
CONTAINER_NAME = "wp-bench17-vllm"
PORT = 8020
GPU_MEM_UTIL = 0.55
OUT_DIR = PROJECT_ROOT / "output" / "bench17"
TAG = "full_gate_rerun"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rer.PORT = PORT  # match config/wp-bench.yaml / dgx_toolbox.yaml documented port
    t0 = time.time()
    result: dict = {}
    try:
        boot_vllm(MODEL_DIR, CONTAINER_NAME, PORT, GPU_MEM_UTIL)
        served = wait_healthy(PORT, CONTAINER_NAME)

        # Phase 15 LOCKED lesson: gate on a REAL generation, not /health.
        warm = generate(PORT, served,
                        [{"instruction": "Reply with exactly one word: OK",
                          "source_val_idx": "warmup"}],
                        max_tokens=16)
        if not warm or not warm[0].strip():
            raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
        print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}",
              file=sys.stderr)

        result = rer._run_wpbench(TAG, OUT_DIR)
    finally:
        stop_vllm(CONTAINER_NAME)

    result["served_model_dir"] = MODEL_DIR
    result["port"] = PORT
    result["wall_clock_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "raw_run_meta.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ran") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(3)
