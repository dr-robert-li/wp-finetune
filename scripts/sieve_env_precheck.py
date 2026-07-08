"""Environment pre-check for the training-free MoE-Sieve phase (Phase 11 Wave 0).

Gates disk (>= 150 GiB free, headroom for two ~57GB s0/s2 merges) and memory
(>= 70 GiB MemAvailable, matches scripts/train_model.py's pre-check) before any
GPU/disk work starts. Also records (does not gate on) statsmodels.stats.weightstats
.ttost_ind availability so plan 11-05 can pick library-vs-hand-rolled TOST
deterministically.

Usage (rtk-prefixed per project CLAUDE.md):
    rtk python3 scripts/sieve_env_precheck.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_DISK_FREE_GIB = 150.0  # headroom for two ~57GB s0/s2 merges + export tars
MIN_MEM_AVAILABLE_GIB = 70.0  # matches train_model.py MIN_FREE_MEMORY_GB
DISK_CHECK_PATH = PROJECT_ROOT / "models" / "_staging"


# ---------------------------------------------------------------------------
# Gate A: disk
# ---------------------------------------------------------------------------


def _check_disk(free_gib: float) -> bool:
    """Gate A: disk free (GiB) must be >= MIN_DISK_FREE_GIB."""
    return free_gib >= MIN_DISK_FREE_GIB


def _disk_free_gib(path: Path) -> float:
    """Read free bytes on the filesystem holding `path` via shutil.disk_usage."""
    check_path = path if path.exists() else path.parent
    _total, _used, free = shutil.disk_usage(check_path)
    return free / (1024 ** 3)


# ---------------------------------------------------------------------------
# Gate B: memory
# ---------------------------------------------------------------------------


def _check_mem(available_gib: float) -> bool:
    """Gate B: MemAvailable (GiB) must be >= MIN_MEM_AVAILABLE_GIB."""
    return available_gib >= MIN_MEM_AVAILABLE_GIB


def _mem_available_gib() -> float:
    """Parse /proc/meminfo MemAvailable (kB) -> GiB."""
    meminfo = Path("/proc/meminfo").read_text()
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split(":")[1].strip().split()[0])
            return kb / (1024 ** 2)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _top_rss_processes(limit: int = 5) -> list[tuple[int, str, int]]:
    """Top-`limit` processes by VmRSS (kB), parsed from /proc/*/status. No psutil dep."""
    procs = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            status = (entry / "status").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        name = ""
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("VmRSS:"):
                parts = line.split(":", 1)[1].strip().split()
                rss_kb = int(parts[0]) if parts else 0
        if rss_kb:
            procs.append((pid, name, rss_kb))
    procs.sort(key=lambda p: p[2], reverse=True)
    return procs[:limit]


# ---------------------------------------------------------------------------
# Gate C: statsmodels TOST availability (recorded, non-blocking)
# ---------------------------------------------------------------------------


def _check_statsmodels() -> bool:
    """Record whether statsmodels.stats.weightstats.ttost_ind is importable.

    Does NOT fail the run on absence -- plan 11-05 hand-rolls TOST if False.
    """
    try:
        import statsmodels.stats.weightstats as w  # noqa: PLC0415

        return hasattr(w, "ttost_ind")
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    disk_free_gib = _disk_free_gib(DISK_CHECK_PATH)
    mem_available_gib = _mem_available_gib()
    statsmodels_ttost_available = _check_statsmodels()

    disk_ok = _check_disk(disk_free_gib)
    mem_ok = _check_mem(mem_available_gib)
    all_hard_gates_pass = disk_ok and mem_ok

    print("=" * 60, file=sys.stderr)
    print("  SIEVE ENV PRE-CHECK", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(
        f"  Disk free ({DISK_CHECK_PATH}): {disk_free_gib:.1f} GiB "
        f"(need >= {MIN_DISK_FREE_GIB:.0f} GiB) -> {'OK' if disk_ok else 'FAIL'}",
        file=sys.stderr,
    )
    print(
        f"  Mem available: {mem_available_gib:.1f} GiB "
        f"(need >= {MIN_MEM_AVAILABLE_GIB:.0f} GiB) -> {'OK' if mem_ok else 'FAIL'}",
        file=sys.stderr,
    )
    print(
        f"  statsmodels ttost_ind available: {statsmodels_ttost_available} (informational only)",
        file=sys.stderr,
    )

    if not disk_ok:
        print(
            f"\n  ACTION REQUIRED: free up disk under {DISK_CHECK_PATH} "
            f"(need {MIN_DISK_FREE_GIB - disk_free_gib:.1f} GiB more) before launching s0/s2 merges.",
            file=sys.stderr,
        )
    if not mem_ok:
        print("\n  Top-5 RSS processes:", file=sys.stderr)
        for pid, name, rss_kb in _top_rss_processes(5):
            print(f"    pid={pid:<8} {name:<20} {rss_kb // 1024} MB", file=sys.stderr)
        print(
            f"\n  ACTION REQUIRED: free {MIN_MEM_AVAILABLE_GIB - mem_available_gib:.1f} GiB "
            f"more RAM before proceeding.",
            file=sys.stderr,
        )

    print("=" * 60, file=sys.stderr)

    print(
        json.dumps(
            {
                "disk_free_gib": round(disk_free_gib, 1),
                "mem_available_gib": round(mem_available_gib, 1),
                "statsmodels_ttost_available": statsmodels_ttost_available,
                "all_hard_gates_pass": all_hard_gates_pass,
            }
        )
    )

    return 0 if all_hard_gates_pass else 1


def _self_check() -> None:
    """Runnable self-check on gate logic with injected fake values (no live hardware)."""
    assert _check_disk(10.0) is False, "10 GiB free should fail the 150 GiB disk gate"
    assert _check_disk(500.0) is True, "500 GiB free should pass the 150 GiB disk gate"
    assert _check_disk(150.0) is True, "exactly 150 GiB should pass (>=)"
    assert _check_mem(10.0) is False, "10 GiB available should fail the 70 GiB mem gate"
    assert _check_mem(500.0) is True, "500 GiB available should pass the 70 GiB mem gate"
    assert _check_mem(70.0) is True, "exactly 70 GiB should pass (>=)"
    print("self-check OK: gate helpers behave correctly on fake values", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
    sys.exit(main())
