"""_launch_reasoning_training.py — One-shot launcher for Phase 4.3 reasoning training.

Usage (from project root, outside container):
    python -m scripts._launch_reasoning_training [--dry-run]

This script:
1. Validates the DGX toolbox preconditions.
2. Ensures the unsloth_studio container is ready.
3. Auto-detects any existing checkpoint-* directories and passes --resume
   if a partial run exists (but adapter_config.json does NOT yet exist).
4. Executes training via dgx.execute() with idempotency_check so a completed
   run is a no-op on re-invocation.
5. Does NOT capture stdout — streams directly to terminal so log file (via
   nohup redirect) captures live step output for the checkpoint-abort observer.

Abort: kill this process to stop the host-side executor; also kill the
in-container training process:
    docker exec unsloth-headless pkill -f "scripts.train_model"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is on PYTHONPATH for dgx_toolbox import
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_NAME = "qwen3-30b-wp-30_70-reasoning"
CONFIG_FILE = "config/train_config_reasoning.yaml"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / EXPERIMENT_NAME
IDEMPOTENCY_CHECK = str(ADAPTER_DIR / "adapter_config.json")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    from scripts.dgx_toolbox import get_toolbox  # noqa: PLC0415

    dgx = get_toolbox()

    # Step 1: Pre-flight validation
    print("=== Phase 4.3 Reasoning Training Launcher ===")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Config:     {CONFIG_FILE}")
    print(f"Adapter:    {ADAPTER_DIR}")
    print()

    result = dgx.validate(["toolbox", "config", "memory:70"])
    if not result.ok:
        print("FATAL: Pre-flight validation failed:")
        print(result.report())
        sys.exit(1)
    print("Pre-flight validation: PASSED")

    # Step 2: Ensure container is ready
    print("Ensuring unsloth_studio container is ready ...")
    dgx.ensure_ready("unsloth_studio")
    print("Container: READY")

    # Step 3: Determine if we need --resume (partial run with checkpoint-* but no adapter_config.json)
    train_cmd = [
        "python", "-m", "scripts.train_model",
        "--config", CONFIG_FILE,
    ]

    if ADAPTER_DIR.exists():
        checkpoints = sorted(
            ADAPTER_DIR.glob("checkpoint-*"),
            key=lambda p: int(re.search(r"\d+", p.name).group()),
        )
        adapter_exists = (ADAPTER_DIR / "adapter_config.json").exists()
        if checkpoints and not adapter_exists:
            latest_ckpt = checkpoints[-1]
            print(f"Found {len(checkpoints)} checkpoint(s). Latest: {latest_ckpt.name}")
            print(f"Resuming training from {latest_ckpt}")
            train_cmd.extend(["--resume", str(latest_ckpt)])
        elif adapter_exists:
            print("adapter_config.json already exists — idempotency check will skip training.")

    if dry_run:
        train_cmd.append("--dry-run")
        print(f"DRY-RUN mode: {' '.join(train_cmd)}")
    else:
        print(f"Training command: {' '.join(train_cmd)}")

    print()
    print("Launching training (streaming output — DO NOT capture) ...")
    print("To stop: kill this process, then run:")
    print("  docker exec unsloth-headless pkill -f 'scripts.train_model'")
    print()

    # Step 4: Execute — no capture, no timeout, streams directly to stdout/stderr
    result = dgx.execute(
        "unsloth_studio",
        *train_cmd,
        idempotency_check=IDEMPOTENCY_CHECK if not dry_run else None,
        timeout=None,
    )

    print()
    if result.ok:
        print("=== Training launcher: SUCCESS ===")
        print(result.summary())
    else:
        print("=== Training launcher: FAILED ===")
        print(result.summary())
        sys.exit(1)


if __name__ == "__main__":
    main()
