"""wp-bench base-model anchor: full 344-test wp-bench run on the UNTRAINED
foundation model (models/Qwen3-30B-A3B), served via vLLM bf16 on the same
stack/config as the Phase 17 v1.2 gen-model rerun (see
deprecated/scripts/bench17_wpbench_full_rerun.py and
output/bench17/wpbench_full_gate_rerun.json), so the gen model's comparative
story has a base anchor to sit next to.

Reuses scripts/run_eval_reasoning.py::_run_wpbench unmodified (same
request_timeout=1800s, max_tokens=2048, concurrency=4, enable_thinking=False
via scripts/_wpbench_pth, temperature=0.0 default), same seed 1337, same
344-test full suite. The base model was never trained on the <wp_gen> task
tokens, so the harness prompts hit it as plain text — that is the point of
this anchor.

Adds the Phase 15 LOCKED lesson: gate capture on a REAL one-token(+)
generation succeeding, not vLLM's /v1/models health response.
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

MODEL_DIR = "models/Qwen3-30B-A3B"
CONTAINER_NAME = "wp-bench-base-anchor-vllm"
PORT = 8020
GPU_MEM_UTIL = 0.55
OUT_DIR = PROJECT_ROOT / "output" / "bench17"
TAG = "base_anchor"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rer.PORT = PORT
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
    (OUT_DIR / "raw_run_meta_base_anchor.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result.get("ran") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(3)
