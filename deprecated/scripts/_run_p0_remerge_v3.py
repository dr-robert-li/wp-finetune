"""_run_p0_remerge_v3.py — P0 host launcher for CPU-only raw-HF+PEFT re-merge.

Companion to scripts/_p0_unsloth_merge_v3.py. v3 path forensics summary:
- v2 OOMed at 15:28:19 on 2026-05-29 during adapter-load step. GB10 unified
  memory exhausted by Unsloth GPU-pinned allocations + PEFT untie scratch.
- v3 routes the entire merge through CPU (CUDA_VISIBLE_DEVICES=''). Peak
  RAM est. ~75 GiB; no NVRM allocator pressure.

Launcher additions vs v2:
- Drops kernel page cache BEFORE launch (`echo 1 > /proc/sys/vm/drop_caches`,
  best-effort; requires root or skipped) to maximise reclaimable headroom
- Sidecar memory monitor: `free -h` snapshot every 30 s into a forensic log
  for post-mortem if it OOMs again
- Tighter pre-flight floor (90 GiB available; CPU-only path needs ~75 GiB
  working set + 15 GiB host/container baseline)
- Streaming-log Popen pattern preserved (subprocess.Popen with line-buffered
  stdout file handle survives in-container Python crashes)
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_FREE_MEM_GIB = 90  # raised from 80 — v3 needs ~75 GiB working set


def _check_free_memory_gib() -> float:
    if not shutil.which("free"):
        return float("inf")
    out = subprocess.check_output(["free", "-b"], text=True)
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            available_bytes = int(parts[6]) if len(parts) >= 7 else int(parts[3])
            return available_bytes / (1024 ** 3)
    return float("inf")


def _drop_caches() -> None:
    """Best-effort kernel page-cache drop. Skips silently if not permitted."""
    subprocess.run(["sync"], check=False)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")
        print("[pre-flight] Dropped kernel page cache (vm/drop_caches=1)")
    except PermissionError:
        print(
            "[pre-flight] WARN: /proc/sys/vm/drop_caches not writable (non-root). "
            "Skipping cache drop. Consider: sudo sysctl vm.drop_caches=1 before launching."
        )
    except OSError as exc:
        print(f"[pre-flight] WARN: drop_caches failed: {exc}")


def _spawn_memory_monitor(log_path: Path) -> subprocess.Popen[bytes]:
    """Sidecar: append `free -h` snapshot every 30 s to a forensic log."""
    script = (
        'while true; do '
        '  printf "=== %s ===\\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; '
        '  free -h; '
        '  echo; '
        '  sleep 30; '
        'done'
    )
    return subprocess.Popen(
        ["bash", "-c", script],
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group so we can SIGTERM cleanly
    )


def main() -> None:
    from scripts.dgx_toolbox import get_toolbox

    free_gib = _check_free_memory_gib()
    print(f"[pre-flight] Available unified memory: {free_gib:.1f} GiB (floor: {MIN_FREE_MEM_GIB} GiB)")
    if free_gib < MIN_FREE_MEM_GIB:
        print(
            f"[pre-flight] ABORT: {free_gib:.1f} GiB < {MIN_FREE_MEM_GIB} GiB floor. "
            "Close heavy apps (Chromium, VS Code, Waveterm) and retry. "
            "v3 CPU-only merge peak working set ~75 GiB; 15 GiB headroom required."
        )
        sys.exit(2)

    _drop_caches()
    free_after = _check_free_memory_gib()
    print(f"[pre-flight] Post-drop available: {free_after:.1f} GiB")

    dgx = get_toolbox()
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log = PROJECT_ROOT / "logs" / "phase4.4" / f"p0_remerge_v3_{ts}.log"
    memlog = PROJECT_ROOT / "logs" / "phase4.4" / f"p0_remerge_v3_{ts}.memwatch.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== P0 v3 CPU-only re-merge ===")
    print(f"=== merge log:   {log}")
    print(f"=== memwatch:    {memlog}")
    print("ensure_ready('unsloth_studio')...")
    r = dgx.ensure_ready("unsloth_studio")
    print(r.report() if hasattr(r, "report") else r)

    container_name = dgx._containers["unsloth_studio"]["container_name"]
    workdir = dgx._containers["unsloth_studio"].get("workdir")

    # CPU-only: -e CUDA_VISIBLE_DEVICES= forces no-CUDA path inside container.
    cmd = ["python", "-u", "-m", "scripts._p0_unsloth_merge_v3"]
    docker_cmd = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES="]
    if workdir:
        docker_cmd += ["-w", workdir]
    docker_cmd += [container_name] + cmd
    print("Executing:", " ".join(docker_cmd))

    memwatch = _spawn_memory_monitor(memlog)
    try:
        with log.open("w", buffering=1) as f:
            f.write(f"=== CMD ===\n{' '.join(docker_cmd)}\n=== OUTPUT ===\n")
            f.flush()
            proc = subprocess.Popen(docker_cmd, stdout=f, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=7200)  # 2h cap; CPU merge est. 30-45 min
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -1
            f.write(f"\n=== RETURNCODE ===\n{rc}\n")
    finally:
        try:
            os.killpg(os.getpgid(memwatch.pid), signal.SIGTERM)
            memwatch.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    print(f"\n=== returncode: {rc} ===")
    print(f"log:      {log}")
    print(f"memwatch: {memlog}")
    if rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
