"""PR2e: W0-03 dual-stage smoke orchestrator — stage1 (CPU) -> stage2 (vLLM).

Runs the cheap CPU degenerate pre-flight first; only on exit 0 proceeds to the
vLLM-served full smoke. Halts (and surfaces the halt marker) on any stage failure.

Usage:
  python scripts/w0_03_smoke_run.py                  # full dual-stage
  python scripts/w0_03_smoke_run.py --skip-stage1    # vLLM stage only
  python scripts/w0_03_smoke_run.py --stage2-arg --max-baseline-sim --stage2-arg 0.90
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE1 = "scripts.w0_03_smoke_stage1_cpu"
STAGE2 = "scripts.w0_03_smoke_stage2_vllm"


def _run(module: str, extra: list[str]) -> int:
    cmd = [sys.executable, "-u", "-m", module] + extra
    print(f"\n=== {module} {' '.join(extra)} ===")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="W0-03 dual-stage smoke orchestrator")
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage2-arg", action="append", default=[],
                    help="pass-through arg to stage 2 (repeatable)")
    args = ap.parse_args()

    dry = ["--dry-run"] if args.dry_run else []

    if not args.skip_stage1:
        rc1 = _run(STAGE1, dry)
        if rc1 != 0:
            print(f"\nSMOKE HALT at Stage 1 (exit {rc1}). See output/04.4_smoke_halt.md")
            return rc1
        print("[orchestrator] Stage 1 clean -> Stage 2")

    rc2 = _run(STAGE2, dry + args.stage2_arg)
    if rc2 == 0:
        print("\nSMOKE_PASS=true — Stage 2 certifying gate passed.")
    else:
        print(f"\nSMOKE_PASS=false — Stage 2 exit {rc2}. See output/04.4_smoke_halt.md")
    return rc2


if __name__ == "__main__":
    sys.exit(main())
