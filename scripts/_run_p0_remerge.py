"""_run_p0_remerge.py — P0 host launcher for Unsloth re-merge of v1 30_70.

Council-deliberated pre-Phase-4.4 prerequisite (Q7 option b). Produces a
correctly-merged baseline at models/qwen3-30b-wp-30_70-merged-v2/ so 4.4
HARD-gate comparisons are full-vs-full instead of partial-vs-partial.

Pattern carried from scripts/_run_parse_check_ckpt72.py:
- Pre-flight RAM floor (80 GiB per RESEARCH §"Memory math" Pitfall 1)
- subprocess.Popen streaming-log (survives CUDA OOM / NVRM kills)
- docker exec direct (bypasses dgx.execute capture buffer loss)
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_FREE_MEM_GIB = 80


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


def main() -> None:
    from scripts.dgx_toolbox import get_toolbox

    free_gib = _check_free_memory_gib()
    print(f"[pre-flight] Available unified memory: {free_gib:.1f} GiB (floor: {MIN_FREE_MEM_GIB} GiB)")
    if free_gib < MIN_FREE_MEM_GIB:
        print(
            f"[pre-flight] ABORT: {free_gib:.1f} GiB < {MIN_FREE_MEM_GIB} GiB floor. "
            "Close heavy apps (Chromium, VS Code, Waveterm) and retry. "
            "Merge transient peak per RESEARCH Pitfall 1 needs ~80 GiB headroom."
        )
        sys.exit(2)

    dgx = get_toolbox()
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    log = PROJECT_ROOT / "logs" / "phase4.4" / f"p0_remerge_{ts}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== P0 Unsloth re-merge — log: {log} ===")
    print("ensure_ready('unsloth_studio')...")
    r = dgx.ensure_ready("unsloth_studio")
    print(r.report() if hasattr(r, "report") else r)

    container_name = dgx._containers["unsloth_studio"]["container_name"]
    workdir = dgx._containers["unsloth_studio"].get("workdir")

    cmd = ["python", "-u", "-m", "scripts._p0_unsloth_merge_v2"]
    docker_cmd = ["docker", "exec"]
    if workdir:
        docker_cmd += ["-w", workdir]
    docker_cmd += [container_name] + cmd
    print("Executing:", " ".join(docker_cmd))

    with log.open("w", buffering=1) as f:
        f.write(f"=== CMD ===\n{' '.join(docker_cmd)}\n=== OUTPUT ===\n")
        f.flush()
        proc = subprocess.Popen(docker_cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=7200)  # 2h cap
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
