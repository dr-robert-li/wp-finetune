"""_run_parse_check_ckpt72.py — One-shot RTRN-04 parse-check launcher for ckpt-72."""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72"
BASE = "models/qwen3-30b-wp-30_70-merged"
VAL = "data/reasoning_dataset/openai_val.jsonl"

# GB10 unified-memory pre-flight: refuse to launch if free RAM is below this floor.
# Raised from 40 → 60 GiB after 2026-05-28 10:13 NVRM OOM at 16 GiB available:
# 4-bit code path was on disk but its transient quantization peak still allocated
# enough to crash the driver when Chromium ate the margin mid-load.
MIN_FREE_MEM_GIB = 60


def _check_free_memory_gib() -> float:
    """Return available system memory in GiB (host view of GB10 unified pool)."""
    if not shutil.which("free"):
        return float("inf")
    out = subprocess.check_output(["free", "-b"], text=True)
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            available_bytes = int(parts[6]) if len(parts) >= 7 else int(parts[3])
            return available_bytes / (1024 ** 3)
    return float("inf")


def main() -> None:
    from scripts.dgx_toolbox import get_toolbox

    free_gib = _check_free_memory_gib()
    print(f"[pre-flight] Available unified memory: {free_gib:.1f} GiB (floor: {MIN_FREE_MEM_GIB} GiB)")
    if free_gib < MIN_FREE_MEM_GIB:
        print(
            f"[pre-flight] ABORT: {free_gib:.1f} GiB available < {MIN_FREE_MEM_GIB} GiB floor. "
            "Loading Qwen3-30B into the GB10 unified pool below this floor risks an OOM "
            "cascade that kills the desktop and may reboot the host. Free memory "
            "(close Waveterm/VS Code/browsers, or `systemctl isolate multi-user.target`) "
            "and retry."
        )
        sys.exit(2)

    dgx = get_toolbox()
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log = PROJECT_ROOT / "logs" / "phase4.3" / f"parse_check_ckpt72_{ts}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Parse-check ckpt-72 (RTRN-04) — log: {log} ===")

    print("ensure_ready('unsloth_studio')...")
    r = dgx.ensure_ready("unsloth_studio")
    print(r.report() if hasattr(r, "report") else r)

    # Resolve container name from dgx config so we can stream output directly
    # rather than buffer through dgx.execute(capture=True), which loses all
    # output if the subprocess dies mid-call (as happened 2026-05-28 10:13).
    container_name = dgx._containers["unsloth_studio"]["container_name"]
    workdir = dgx._containers["unsloth_studio"].get("workdir")

    cmd = [
        "python", "-u", "-m", "scripts.checkpoint_parse_check",
        "--checkpoint-dir", CHECKPOINT,
        "--base", BASE,
        "--val-jsonl", VAL,
        "--n", "3",
        "--threshold", "0.05",
    ]
    docker_cmd = ["docker", "exec"]
    if workdir:
        docker_cmd += ["-w", workdir]
    docker_cmd += [container_name] + cmd
    print("Executing:", " ".join(docker_cmd))

    # Stream stdout+stderr to log file line-buffered so partial output survives
    # if subprocess crashes (CUDA OOM, NVRM, SIGKILL).
    with log.open("w", buffering=1) as f:
        f.write(f"=== CMD ===\n{' '.join(docker_cmd)}\n=== OUTPUT ===\n")
        f.flush()
        proc = subprocess.Popen(docker_cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -1
        f.write(f"\n=== RETURNCODE ===\n{rc}\n")

    print(f"\n=== returncode: {rc} ===")
    print(f"log: {log}")
    if rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
