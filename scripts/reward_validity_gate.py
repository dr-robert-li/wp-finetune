"""Standing reward-validity gate — Phase 08.2 (SC1 / RVAL-01).

Reusable entrypoint: given a candidate reward form (by name from FORMS, or a
custom callable), runs the offline oracle trajectory correlation and returns a
GateResult indicating valid/invalid.

STANDING RULE (D-08.2):
    No reward goes to GPU until its oracle correlation CI-lower > 0.
    pairwise_rank_agreement PASSES (CI-lower +0.15>0).
    fix_correctness_BASELINE FAILS (CI-lower -0.87, includes 0).
    See .planning/phases/08.2-reward-validity/08.2-GATE-RULE.md for full rule.

Usage (Plans 03, 04 and future candidates):
    from scripts.reward_validity_gate import run_validity_gate, GateResult

    result = run_validity_gate("pairwise_rank_agreement")
    assert result.valid, f"Gate FAILED for {result.form}: ci_lo={result.ci_lo:.3f}"

    # Register and test a new candidate form:
    def my_reward(model, gt): ...
    result = run_validity_gate("my_reward", form_fn=my_reward)

Imports: reuses scripts._reward_validity_oracle (single source of truth for FORMS,
bootstrap_corr_lo, and the trajectory pipeline). Does NOT copy any logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

# Single source of truth: FORMS, bootstrap_corr_lo, and the pipeline builder.
from scripts._reward_validity_oracle import (
    FORMS,
    FIX_CORR,
    CKPTS,
    bootstrap_corr_lo,
    build_oracle_pipeline,
    model_overall_map,
    _load_teacher_from_val,
)


@dataclass
class GateResult:
    """Result of a reward-validity gate run.

    Attributes:
        form              : Name of the candidate reward form evaluated.
        spearman_vs_target: Point Spearman correlation of the form's checkpoint
                            trajectory vs the teacher-Spearman target trajectory.
        ci_lo             : 95% bootstrap lower bound of the Spearman correlation.
        ci_hi             : 95% bootstrap upper bound of the Spearman correlation.
        valid             : True iff ci_lo > 0 — the standing gate rule (D-08.2).
        n_ckpts           : Number of checkpoints used in the trajectory correlation.
    """
    form: str
    spearman_vs_target: float
    ci_lo: float
    ci_hi: float
    valid: bool
    n_ckpts: int

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"GateResult(form={self.form!r}, spearman={self.spearman_vs_target:.3f}, "
            f"ci=[{self.ci_lo:.3f},{self.ci_hi:.3f}], valid={self.valid}, "
            f"n_ckpts={self.n_ckpts})  [{status}]"
        )


def _build_custom_form_series(
    form_fn: Callable[[List[float], List[float]], float],
) -> tuple[dict, list, dict]:
    """Compute per-checkpoint reward series for a custom form function.

    Rebuilds the full checkpoint pipeline (loads model_overall_map + teacher GT)
    and applies form_fn to each checkpoint's (model_vec, gt_vec). Used when the
    caller passes form_fn for a new candidate form not in FORMS.

    Returns:
        (form_series, present, target) — same shape as build_oracle_pipeline() output
        but with the custom form's series instead of the FORMS dict.
    """
    from scipy.stats import spearmanr

    teacher_map = _load_teacher_from_val()
    mom = {ck: model_overall_map(ck) for ck in CKPTS}
    present = [ck for ck in CKPTS if mom[ck]]

    common = set(teacher_map)
    for ck in present:
        common &= set(mom[ck])
    common_sorted = sorted(common)

    target: dict[str, float] = {}
    form_series: dict[str, float] = {}
    for ck in present:
        m_vec = [mom[ck][i] for i in common_sorted]
        g_vec = [teacher_map[i] for i in common_sorted]
        target[ck] = float(spearmanr(m_vec, g_vec).statistic)
        form_series[ck] = form_fn(m_vec, g_vec)

    return form_series, present, target


def run_validity_gate(
    form_name: str,
    form_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    n_boot: int = 2000,
) -> GateResult:
    """Score a candidate reward form against the oracle trajectory.

    Args:
        form_name : Key for the reward form. Must be one of the keys in FORMS
                    (from scripts._reward_validity_oracle) OR a new name when
                    form_fn is provided. Use "fix_correctness_BASELINE" for the
                    known-invalid baseline (uses the external FIX_CORR series).
        form_fn   : Optional custom callable (model: list[float], gt: list[float])
                    -> float. Required if form_name is not in FORMS.
        n_boot    : Bootstrap resampling count (default 2000, matching oracle).

    Returns:
        GateResult with valid=True iff ci_lo>0 (the standing gate rule, D-08.2).

    Raises:
        ValueError: if form_name is not in FORMS and form_fn is not provided.

    Examples:
        # Test the known-valid replacement reward:
        r = run_validity_gate("pairwise_rank_agreement")
        # r.valid is True, r.ci_lo > 0

        # Test the known-invalid proxy (Goodhart baseline):
        f = run_validity_gate("fix_correctness_BASELINE")
        # f.valid is False

        # Register a new candidate form:
        def my_form(model, gt): return sum(m > g for m, g in zip(model, gt)) / len(model)
        r = run_validity_gate("my_form", form_fn=my_form)
    """
    # Resolve whether this is a built-in FORMS entry or a custom callable
    is_builtin = form_name in FORMS

    if not is_builtin and form_fn is None:
        raise ValueError(
            f"Unknown form {form_name!r}. Available: {list(FORMS.keys())}. "
            f"Pass form_fn=<callable> to register a new candidate."
        )

    if is_builtin and form_fn is None:
        # --- Built-in path: reuse build_oracle_pipeline() (avoids double data load) ---
        target, reward_series, present, _n_common = build_oracle_pipeline()

        if form_name == "fix_correctness_BASELINE":
            # External FIX_CORR series (not computed from (model,gt) vectors)
            form_series = {n: v for n, v in FIX_CORR.items() if n in present}
        else:
            form_series = reward_series[form_name]

    else:
        # --- Custom callable path: rebuild pipeline applying form_fn per checkpoint ---
        form_series, present, target = _build_custom_form_series(form_fn)

    # Align checkpoint keys between form_series and target
    ck_aligned = [n for n in present if n in form_series and n in target]
    xs = [form_series[n] for n in ck_aligned]
    ys = [target[n] for n in ck_aligned]

    point, lo, hi = bootstrap_corr_lo(xs, ys, n_boot=n_boot)
    valid = bool(lo == lo and lo > 0)  # NaN-safe: NaN==NaN is False -> valid=False

    return GateResult(
        form=form_name,
        spearman_vs_target=point,
        ci_lo=lo,
        ci_hi=hi,
        valid=valid,
        n_ckpts=len(ck_aligned),
    )
