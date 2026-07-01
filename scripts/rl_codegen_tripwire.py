"""Phase 08.2 RVAL-03 — in-run wp-bench codegen TRIP-WIRE.

Provides a DECISION function (check_codegen_tripwire) that is a pure function of
an injectable score, and an EXECUTION function (run_codegen_probe) that calls the
real wp-bench runner (GPU path, exercised only by Plan 05's live smoke).

Design: split decision from execution so the dry-run and offline tests never need
vLLM or a Tinker checkpoint.  The decision feeds into rl_train's EXISTING
check_halt -> return-True -> emergency-checkpoint seam (RVAL-03 / D-08.2).

Constants are sourced from _rlev01_wpbench_ckpt (single source of truth; threat
T-082-05 ensures a silent constant change fails CI).

Sub-floor thresholds (from phase09_rerun RLEV_FINAL_REPORT.md):
  knowledge >= 0.45
  exec      >= 0.375
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Single source of truth — CODEGEN_BAR_V12 derived from the built building block.
# T-082-05: any silent loosening of BASELINE_V12 in _rlev01_wpbench_ckpt breaks
# the test that asserts abs(CODEGEN_BAR_V12 - 0.4616) < 1e-9.
# ---------------------------------------------------------------------------
from scripts._rlev01_wpbench_ckpt import BASELINE_V12 as CODEGEN_BAR_V12  # noqa: E402

# Sub-floor thresholds derived from Phase-10 regression report:
# step-500 showed knowledge=0.41 (< 0.45 floor) and exec=0.34 (< 0.375 floor).
KNOWLEDGE_FLOOR: float = 0.45
EXEC_FLOOR: float = 0.375


def check_codegen_tripwire(
    wpbench_score: Optional[float],
    bar: float = CODEGEN_BAR_V12,
    sub_scores: Optional[dict] = None,
) -> Optional[str]:
    """Pure decision function — injectable score, no GPU, no vLLM.

    Returns a halt-reason string when ANY of the following fire:
      - wpbench_score is not None and below ``bar`` (overall composite gate)
      - sub_scores["knowledge"] is present and below KNOWLEDGE_FLOOR
      - sub_scores["exec"] is present and below EXEC_FLOOR

    Returns None when:
      - wpbench_score is None (probe not run / score unavailable — skip silently)
      - wpbench_score >= bar AND no sub-floor is breached

    The caller (run_training_step) folds the returned reason into halt_reason
    via ``halt_reason = halt_reason or check_codegen_tripwire(...)``, so the
    SAME emergency-checkpoint + return-True path fires (no second mechanism).

    Args:
        wpbench_score: composite wp-bench score or None (probe skipped / offline).
        bar: halt threshold (defaults to CODEGEN_BAR_V12 = 0.4616).
        sub_scores: optional dict with keys "knowledge" and/or "exec"; used to
            enforce per-dimension floors even when the composite is at bar.

    Returns:
        Non-None halt reason string if the trip-wire fires, else None.
    """
    if wpbench_score is None:
        return None

    # Overall composite gate
    if wpbench_score < bar:
        return (
            f"CODEGEN TRIP-WIRE HALT: wp-bench={wpbench_score:.4f} < bar={bar:.4f} "
            f"(v1.2 SFT bar CODEGEN_BAR_V12)"
        )

    # Sub-floor checks (even if overall passes the composite bar)
    if sub_scores:
        knowledge = sub_scores.get("knowledge")
        if knowledge is not None and knowledge < KNOWLEDGE_FLOOR:
            return (
                f"CODEGEN TRIP-WIRE HALT: knowledge sub-floor breach "
                f"knowledge={knowledge:.4f} < {KNOWLEDGE_FLOOR}"
            )
        exec_score = sub_scores.get("exec")
        if exec_score is not None and exec_score < EXEC_FLOOR:
            return (
                f"CODEGEN TRIP-WIRE HALT: exec sub-floor breach "
                f"exec={exec_score:.4f} < {EXEC_FLOOR}"
            )

    return None


def run_codegen_probe(model_dir: str, tag: str, step: int) -> dict:
    """EXECUTION path — calls the real wp-bench runner (GPU / live path only).

    Merges/serves the checkpoint via the reused _wpbench_with_boot building block
    from scripts.run_eval_reasoning.  This function is NEVER called in offline
    tests or dry-runs — it requires a running vLLM instance and a merged checkpoint.

    The test/offline path uses ``args.codegen_score_override`` (injected directly
    into check_codegen_tripwire) and never calls this function.

    Args:
        model_dir: path to the merged checkpoint directory.
        tag: label for this probe (used in output paths and vLLM served name).
        step: training step number (for output organisation).

    Returns:
        dict with at minimum:
          - "wpbench_score": float | None
          - "knowledge": float | None
          - "exec": float | None
          - "baseline_v12": CODEGEN_BAR_V12
          - "passes_hard_gate": bool
    """
    # Lazy import — GPU deps must NOT be loaded at module import time.
    # This ensures the module is safe to import in offline/test environments.
    from pathlib import Path  # noqa: PLC0415

    # Import the real wp-bench runner (requires vLLM, CUDA, etc.)
    from scripts.run_eval_reasoning import _wpbench_with_boot  # noqa: PLC0415

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "output" / "rl_eval" / f"wpbench_probe_step{step}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    served_name = f"wp-probe-step{step}-{tag}-vllm"
    res = _wpbench_with_boot(model_dir, served_name, tag, 0.55, out_dir)

    score = res.get("wpbench_score")
    res["baseline_v12"] = CODEGEN_BAR_V12
    res["passes_hard_gate"] = score is not None and score >= CODEGEN_BAR_V12
    return res
