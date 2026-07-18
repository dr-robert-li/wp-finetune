"""BASE-03 DeltaNet-on-aarch64 vLLM serving smoke (Phase 20 base bring-up).

Proves Qwen3.6-35B-A3B's Gated-DeltaNet layers execute on aarch64/GB10 via
vLLM WITH CUDA-graph capture enabled on the FIRST attempt (Pitfall 2 — an
eager-only smoke gives a false pass for vLLM #35945, which only fires during
graph capture). Falls back to --enforce-eager once if capture crashes, and
records the use_kernels=False decision (declining the non-allowlisted
Atlas-Inference/gdn community kernel, per this plan's threat model T-20-03a).

Writes output/base20/deltanet_smoke.json (BASE-03 gate receipt).

Usage:
    python -m scripts.smoke_deltanet_base20
    python scripts/smoke_deltanet_base20.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same direct-execution sys.path fix as scripts/download_model.py /
# scripts/smoke_load_base20.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    boot_vllm,
    wait_healthy,
    generate,
    stop_vllm,
    VllmBootTimeout,
)

IMAGE = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
MODEL_DIR = "models/Qwen3.6-35B-A3B"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
CONTAINER_NAME = "base20-deltanet-smoke"
PORT = 8020
GPU_MEM_UTIL = 0.80
# Pitfall 3 lesson (900s for the old 57 GiB base) raised: this base is 67 GiB.
BOOT_TIMEOUT_SEC = 1200
MIN_VLLM_VERSION = (0, 19)
OUT_DIR = PROJECT_ROOT / "output" / "base20"
OUTPUT_PATH = OUT_DIR / "deltanet_smoke.json"

USE_KERNELS_RATIONALE = (
    "use_kernels=False (PyTorch fallback) chosen by default per 20-RESEARCH.md "
    "Alternatives Considered: the community Atlas-Inference/gdn Hub kernel "
    "requires trust_remote_code=True against a repo not yet on HF's trusted-"
    "kernels allowlist (SUS verdict, Package Legitimacy Audit) for only a "
    "1.38x prefill speedup (0.73s vs 0.53s at 1024 tokens; decode throughput "
    "flat ~16 tok/s either way, memory-bandwidth bound). `kernels` was not "
    "installed and the Hub kernel was not loaded. Flipping to use_kernels=True "
    "in any later phase requires a checkpoint:human-verify (gate=blocking-"
    "human) per this plan's threat-model entry T-20-03a."
)


def write_receipt(status: str, **fields) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {"status": status, **fields}
    OUTPUT_PATH.write_text(json.dumps(receipt, indent=2))
    return receipt


def check_liveness() -> None:
    """Assumption A1: trivial docker/nvidia-smi liveness check before any real smoke."""
    docker_ok = subprocess.run(["docker", "ps"], capture_output=True, text=True).returncode == 0
    nvidia_ok = subprocess.run(["nvidia-smi"], capture_output=True, text=True).returncode == 0
    if not docker_ok or not nvidia_ok:
        raise RuntimeError(f"A1 liveness check failed: docker_ok={docker_ok} nvidia_ok={nvidia_ok}")
    print(f"[A1] docker + nvidia-smi liveness check OK")


def resolve_vllm_version() -> str:
    """FIRST smoke action (A3): log the container's resolved vLLM version — a
    recorded fact, not an assumption from the :latest tag."""
    r = subprocess.run(
        ["docker", "run", "--rm", IMAGE, "python3", "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"could not resolve vLLM version from {IMAGE}: {r.stderr}")
    version = r.stdout.strip().splitlines()[-1].strip()
    print(f"[A3] container vLLM version: {version}")
    return version


def assert_min_vllm_version(version: str) -> None:
    parts = tuple(int(x) for x in str(version).split(".")[:2])
    if parts < MIN_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM {version} < required {'.'.join(str(x) for x in MIN_VLLM_VERSION)}"
        )


def run_smoke() -> dict:
    check_liveness()
    vllm_version = resolve_vllm_version()
    assert_min_vllm_version(vllm_version)

    fallback_used = False
    cuda_graph_capture = "enabled"
    extra_env = {"LANGUAGE_MODEL_ONLY": "1"}

    print("[boot] attempt 1: CUDA-graph capture ENABLED (no --enforce-eager)")
    boot_vllm(MODEL_DIR, CONTAINER_NAME, PORT, GPU_MEM_UTIL,
              serve_script=SERVE_SCRIPT, extra_env=extra_env)
    try:
        served = wait_healthy(PORT, CONTAINER_NAME, timeout=BOOT_TIMEOUT_SEC)
    except VllmBootTimeout as e:
        print(f"[boot] attempt 1 (CUDA-graph capture) failed: {e}\n"
              f"[boot] retrying once with --enforce-eager (vLLM #35945 documented fallback)")
        stop_vllm(CONTAINER_NAME)
        fallback_used = True
        cuda_graph_capture = "eager_fallback"
        extra_env["ENFORCE_EAGER"] = "1"
        boot_vllm(MODEL_DIR, CONTAINER_NAME, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT, extra_env=extra_env)
        served = wait_healthy(PORT, CONTAINER_NAME, timeout=BOOT_TIMEOUT_SEC)

    # Carry-forward lesson 2: gate on a REAL generation, not /v1/models health.
    warm = generate(PORT, served,
                     [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                     max_tokens=16)
    warm_gen_ok = bool(warm and warm[0].strip())
    if not warm_gen_ok:
        raise RuntimeError(f"real-generation warm-up returned empty output: {warm!r}")
    print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}")

    return write_receipt(
        "pass",
        vllm_version=vllm_version,
        cuda_graph_capture=cuda_graph_capture,
        fallback_used=fallback_used,
        warm_gen_ok=warm_gen_ok,
        warm_gen_sample=warm[0].strip()[:200],
        use_kernels=False,
        use_kernels_rationale=USE_KERNELS_RATIONALE,
        gpu_mem_util=GPU_MEM_UTIL,
        served_model=served,
        model_dir=MODEL_DIR,
    )


def main() -> int:
    try:
        result = run_smoke()
    except Exception as exc:  # noqa: BLE001 — smoke script: any exception is a gate failure
        write_receipt("fail", failing_field="exception", error=str(exc))
        print(f"BASE-03 SMOKE FAILED: {exc}")
        return 1
    finally:
        stop_vllm(CONTAINER_NAME)

    print("BASE-03 SMOKE PASSED")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
