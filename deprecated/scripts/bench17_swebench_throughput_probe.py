"""BENCH-02 Wave-0 throughput probe (Phase 17-02, Task 2).

Measures real generation throughput (prefill tok/s, decode tok/s, per-instance
wall-clock) for the v1.2 gen model at SWE-bench-scale prompts (oracle retrieval,
10-20k input tokens), serving via the same vLLM/bf16 shipping stack used for
BENCH-01. Feeds the wall-clock projections that Task 3's decision rule consumes
to pre-register the SWE-bench scope BEFORE any real patch-generation/eval run
happens.

Model max_position_embeddings=40960; SWE-bench oracle prompts run much longer
than wp-bench's short prompts, so this probe sizes vLLM for FEWER slots /
LARGER per-slot context than the Phase 15 judge recipe (--parallel 4,
11264/slot won't hold a 20k-token prompt) -- MAX_MODEL_LEN=24576 here.

Real-generation warm-up gate (Phase 15 lesson): capture is gated on one
non-empty real generation succeeding, not on /v1/models health.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    SERVE_SCRIPT,
    VllmBootTimeout,
    wait_healthy,
    stop_vllm,
    _dump_boot_log,
)

MODEL_DIR = PROJECT_ROOT / "models" / "qwen3-30b-wp-30_70-reasoning-merged-v4"
CONTAINER_NAME = "wp-bench17-swebench-vllm"
PORT = 8020
GPU_MEM_UTIL = 0.55
MAX_MODEL_LEN = 24576
MAX_TOKENS = 2048
CONCURRENCY = 2  # fewer slots, larger per-slot ctx vs Phase 15 judge recipe (--parallel 4)
SEED = 0
OUT_DIR = PROJECT_ROOT / "output" / "bench17"

# 8 real SWE-bench Lite oracle instances, tokenized and picked to spread across
# the 10-20k input-token band (measured this session with this model's own
# tokenizer, not assumed).
INSTANCE_IDS = [
    "psf__requests-863",
    "pytest-dev__pytest-11148",
    "pylint-dev__pylint-7080",
    "sympy__sympy-17022",
    "scikit-learn__scikit-learn-14894",
    "sympy__sympy-22840",
    "sympy__sympy-24102",
    "matplotlib__matplotlib-18869",
]


def boot_vllm_wide_ctx() -> None:
    env = {
        **os.environ,
        "CONTAINER_NAME": CONTAINER_NAME,
        "PORT": str(PORT),
        "MODEL_DIR": str(MODEL_DIR),
        "GPU_MEM_UTIL": str(GPU_MEM_UTIL),
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
    }
    print(f"[vllm] booting {CONTAINER_NAME} on :{PORT} max_model_len={MAX_MODEL_LEN}")
    subprocess.run(
        ["bash", SERVE_SCRIPT], env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def build_prompt(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def generate_one(client, served_model: str, instance_id: str, text: str) -> dict:
    """Stream one completion; capture TTFT (prefill proxy) + decode tok/s."""
    t_start = time.time()
    ttft = None
    n_chunks = 0
    content_parts: list[str] = []
    stream = client.chat.completions.create(
        model=served_model,
        messages=build_prompt(text),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        seed=SEED,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = None
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta and ttft is None:
                ttft = time.time() - t_start
            if delta:
                n_chunks += 1
                content_parts.append(delta)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    t_end = time.time()
    wall_clock = t_end - t_start
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    decode_time = (wall_clock - ttft) if ttft is not None else None
    return {
        "instance_id": instance_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "wall_clock_s": round(wall_clock, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if (prompt_tokens and ttft) else None,
        "decode_tok_s": round((completion_tokens - 1) / decode_time, 1)
        if (completion_tokens and decode_time and completion_tokens > 1)
        else None,
        "patch_nonempty": bool("".join(content_parts).strip()),
    }


def main() -> int:
    import openai

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    print("Loading SWE-bench Lite oracle dataset (princeton-nlp/SWE-bench_Lite_oracle)...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")
    by_id = {row["instance_id"]: row for row in ds}
    missing = [i for i in INSTANCE_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"Instance IDs not found in dataset: {missing}")

    t0 = time.time()
    served = None
    results: list[dict] = []
    try:
        boot_vllm_wide_ctx()
        served = wait_healthy(PORT, CONTAINER_NAME)

        client = openai.OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="none")

        # Phase 15 LOCKED lesson: real-generation warm-up, not /health.
        warm = client.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
            max_tokens=16,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        warm_text = (warm.choices[0].message.content or "").strip()
        if not warm_text:
            raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
        print(f"[warmup] real-generation OK (served_model={served!r}): {warm_text[:80]!r}")

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [
                pool.submit(generate_one, client, served, iid, by_id[iid]["text"])
                for iid in INSTANCE_IDS
            ]
            for f in futures:
                r = f.result()
                print(f"[probe] {r['instance_id']}: prompt={r['prompt_tokens']} "
                      f"completion={r['completion_tokens']} ttft={r['ttft_s']}s "
                      f"wall={r['wall_clock_s']}s prefill={r['prefill_tok_s']}tok/s "
                      f"decode={r['decode_tok_s']}tok/s")
                results.append(r)
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        return 3
    finally:
        stop_vllm(CONTAINER_NAME)

    total_wall = time.time() - t0

    valid = [r for r in results if r["prefill_tok_s"] and r["decode_tok_s"]]
    n = len(valid)
    avg_prefill = round(sum(r["prefill_tok_s"] for r in valid) / n, 1) if n else None
    avg_decode = round(sum(r["decode_tok_s"] for r in valid) / n, 1) if n else None
    avg_wall_per_instance = round(sum(r["wall_clock_s"] for r in valid) / n, 1) if n else None
    avg_prompt_tokens = round(sum(r["prompt_tokens"] for r in valid) / n, 1) if n else None

    # --- Wall-clock projections per candidate scope ---
    # Generation-only: avg_wall_per_instance (measured, concurrency=2) scaled
    # linearly to concurrency=2 (this probe's own concurrency), i.e. instances
    # / CONCURRENCY * avg_wall_per_instance. This is the concurrent-serving
    # assumption for the real run (same CONCURRENCY as this probe).
    #
    # Docker build/test-run overhead per instance: MEASURED this session only
    # for PHP (Task 1: ~35s eval run per instance after a ~60s one-time shared
    # base+env image build across the whole repo -- see
    # output/bench17/arm64_probe/gold.arm64_probe{1,2}.json). Classic Python
    # SWE-bench per-instance overhead was NOT measured this session (Task 1's
    # scope is PHP-only per the plan); using a documented, conservative
    # literature-sourced estimate instead (a few minutes/instance -- repo
    # checkout + pip/conda env install + running the repo's real test suite is
    # heavier than PHP's composer install + phpunit single-file run), flagged
    # explicitly as UNMEASURED-this-session in the pre-registration doc.
    PHP_DOCKER_OVERHEAD_S_PER_INSTANCE = 35.0  # measured, Task 1
    PYTHON_DOCKER_OVERHEAD_S_PER_INSTANCE_ESTIMATE = 180.0  # unmeasured, literature-based estimate

    def project(n_instances: int, overhead_s: float) -> dict:
        if avg_wall_per_instance is None:
            return {"n_instances": n_instances, "error": "no valid measurements"}
        gen_s = (n_instances / CONCURRENCY) * avg_wall_per_instance
        docker_s = n_instances * overhead_s  # docker builds/evals not measured as concurrency-scaled here (conservative, serial estimate)
        total_s = gen_s + docker_s
        return {
            "n_instances": n_instances,
            "generation_wall_clock_s": round(gen_s, 1),
            "docker_overhead_s": round(docker_s, 1),
            "total_projected_s": round(total_s, 1),
            "total_projected_h": round(total_s / 3600, 2),
        }

    projections = {
        "lite_300_python": project(300, PYTHON_DOCKER_OVERHEAD_S_PER_INSTANCE_ESTIMATE),
        "php_multilingual_43": project(43, PHP_DOCKER_OVERHEAD_S_PER_INSTANCE),
        "verified_500_python": project(500, PYTHON_DOCKER_OVERHEAD_S_PER_INSTANCE_ESTIMATE),
        "lite_300_plus_php_43": {
            "total_projected_h": round(
                project(300, PYTHON_DOCKER_OVERHEAD_S_PER_INSTANCE_ESTIMATE)["total_projected_h"]
                + project(43, PHP_DOCKER_OVERHEAD_S_PER_INSTANCE)["total_projected_h"],
                2,
            )
        },
    }

    receipt = {
        "probe": "swebench_throughput_probe",
        "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "served_model_name": served,
        "serving_config": {
            "engine": "vLLM",
            "dtype": "bf16",
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "concurrency_tested": CONCURRENCY,
            "port": PORT,
        },
        "sampling_config": {
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "enable_thinking": False,
            "retrieval_style": "oracle",
            "dataset": "princeton-nlp/SWE-bench_Lite_oracle",
        },
        "warmup": {"gated_on_real_generation": True, "warmup_response": warm_text},
        "per_instance_results": results,
        "summary": {
            "n_instances_probed": len(results),
            "n_valid_measurements": n,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_prefill_tok_s": avg_prefill,
            "avg_decode_tok_s": avg_decode,
            "avg_wall_clock_s_per_instance": avg_wall_per_instance,
            "probe_total_wall_clock_s": round(total_wall, 1),
        },
        "wall_clock_projections": projections,
        "projection_methodology": (
            "generation_wall_clock_s = (n_instances / concurrency_tested) * "
            "avg_wall_clock_s_per_instance (measured this probe, concurrency=2). "
            "docker_overhead_s = n_instances * per-instance overhead: PHP overhead "
            "(35s) is MEASURED this session (Task 1, arm64_probe1/2). Python overhead "
            "(180s) is an UNMEASURED-this-session literature-based estimate -- Task 1's "
            "live build validation is PHP-only per the plan. Both overheads assumed "
            "serial (not concurrency-scaled) -- conservative (upper-bound) choice."
        ),
    }
    out_path = OUT_DIR / "swebench_throughput_probe.json"
    out_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"\nWritten: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
