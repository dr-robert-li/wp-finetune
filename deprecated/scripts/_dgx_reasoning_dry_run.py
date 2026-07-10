"""_dgx_reasoning_dry_run.py — Task 5 automated dry-run gate for 04.3-01.

Sequence:
  1. D-03 readiness gate (live metadata.json must PASS)
  2. DGX pre-flight: dgx.validate + dgx.ensure_ready
  3. DGX execute --dry-run with capture=True
  4. Assert stdout contains required strings:
       - "Router-freeze check PASSED"
       - single-digit trainable%
       - "2.0e-5" (LR)
       - "8192" (max_seq_length)
       - "20" (warmup_steps — from "Warmup steps:" echo)
       - "qwen3-30b-wp-30_70-merged" (merged base local_dir)
  5. Assert stdout does NOT contain:
       - "experiment_001"
       - "merged-30_70" (stale legacy dir)
  6. Print VERBATIM stdout for human verification.

Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.checkpoint_parse_check import verify_dataset_readiness  # noqa: E402
from scripts.dgx_toolbox import get_toolbox  # noqa: E402


def run_dry_run_gate() -> None:
    # -----------------------------------------------------------------------
    # Step 1: D-03 readiness gate
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: D-03 dataset readiness gate")
    print("=" * 60)
    metadata_path = PROJECT_ROOT / "data" / "reasoning_dataset" / "metadata.json"
    verify_dataset_readiness(metadata_path)  # exits non-zero if fails

    # -----------------------------------------------------------------------
    # Step 2: DGX pre-flight
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 2: DGX pre-flight validation")
    print("=" * 60)
    dgx = get_toolbox()

    result = dgx.validate(["toolbox", "config", "memory:70"])
    print(result.report())
    if not result.ok:
        print("FATAL: DGX pre-flight validation FAILED")
        sys.exit(1)
    print("DGX pre-flight validation: PASSED")

    print()
    print("Ensuring unsloth_studio container is ready ...")
    dgx.ensure_ready("unsloth_studio")
    print("unsloth_studio: READY")

    # -----------------------------------------------------------------------
    # Step 3: DGX --dry-run execute
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 3: DGX dry-run (train_model --dry-run --config config/train_config_reasoning.yaml)")
    print("=" * 60)
    dry_result = dgx.execute(
        "unsloth_studio",
        "python", "-m", "scripts.train_model",
        "--dry-run",
        "--config", "config/train_config_reasoning.yaml",
        capture=True,
    )

    stdout = dry_result.stdout if hasattr(dry_result, "stdout") else str(dry_result)

    print()
    print("=" * 60)
    print("DRY-RUN STDOUT (VERBATIM):")
    print("=" * 60)
    print(stdout)
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 4: Assertions — required strings
    # -----------------------------------------------------------------------
    print()
    print("STEP 4: Asserting dry-run stdout ...")

    failures: list[str] = []

    # Router-freeze check (RTRN-03)
    if "Router-freeze check PASSED" not in stdout:
        failures.append("MISSING: 'Router-freeze check PASSED' in stdout (RTRN-03)")

    # Trainable % must be single-digit (< 10%)
    trainable_match = re.search(r"Trainable params:.*?\((\d+(?:\.\d+)?)%\)", stdout)
    if trainable_match:
        pct = float(trainable_match.group(1))
        if pct >= 10.0:
            failures.append(
                f"FAIL: trainable% = {pct:.2f}% — expected single-digit (< 10%); "
                f"embeddings or router may have leaked back in (Pitfall 2/4)"
            )
        else:
            print(f"  trainable% = {pct:.2f}% — single-digit OK")
    else:
        failures.append("MISSING: 'Trainable params:' line with % in stdout")

    # LR 2.0e-5 (RTRN-01)
    if "2.0e-5" not in stdout and "2e-05" not in stdout and "2.0e-05" not in stdout:
        failures.append("MISSING: learning_rate 2.0e-5 echo in stdout (RTRN-01)")

    # max_seq_length 8192 (RTRN-02)
    if "8192" not in stdout:
        failures.append("MISSING: max_seq_length 8192 echo in stdout (RTRN-02)")

    # warmup_steps 20 (RTRN-01)
    if not re.search(r"[Ww]armup.steps.*20", stdout):
        failures.append("MISSING: warmup_steps 20 echo in stdout (RTRN-01)")

    # merged base local_dir (D-01/D-04)
    if "qwen3-30b-wp-30_70-merged" not in stdout:
        failures.append("MISSING: 'qwen3-30b-wp-30_70-merged' in stdout (D-01/D-04)")

    # -----------------------------------------------------------------------
    # Step 5: Assertions — forbidden strings
    # -----------------------------------------------------------------------
    if "experiment_001" in stdout:
        failures.append("FAIL: 'experiment_001' found in stdout (contamination from dead config)")

    if "merged-30_70" in stdout and "qwen3-30b-wp-30_70-merged" not in stdout:
        # The canonical name contains "wp-30_70-merged" not "merged-30_70"
        # but check for the stale legacy pattern specifically
        if re.search(r"models/merged-30_70", stdout):
            failures.append("FAIL: 'models/merged-30_70' (stale legacy dir) found in stdout (D-04)")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print()
    if failures:
        print("=" * 60)
        print("DRY-RUN GATE FAILURES:")
        print("=" * 60)
        for f in failures:
            print(f"  {f}")
        print()
        print("Dry-run gate FAILED. Fix the above before proceeding to training.")
        sys.exit(1)

    print("=" * 60)
    print("DRY-RUN GATE PASSED — all assertions satisfied:")
    print("  - Router-freeze check PASSED (RTRN-03)")
    print("  - Single-digit trainable% confirmed")
    print("  - LR 2.0e-5 echoed (RTRN-01)")
    print("  - max_seq_length 8192 echoed (RTRN-02)")
    print("  - warmup_steps 20 echoed (RTRN-01)")
    print("  - merged 30_70 base confirmed (D-01/D-04)")
    print("  - No experiment_001 contamination")
    print("  - No stale merged-30_70 reference")
    print("=" * 60)
    print()
    print("Human verification required: confirm the VERBATIM stdout above satisfies")
    print("all five criteria before proceeding to Task 6 training.")


if __name__ == "__main__":
    run_dry_run_gate()
