"""
scripts/tost_gate.py

SIEVE-05: TOST (Two One-Sided Tests) equivalence gate for the training-free
MoE-Sieve k-sweep. Declares the smallest expert-count budget k that is
statistically equivalent to the full (unmasked) model on wp-bench at
epsilon=2pp, AND retains all 1,480 protected experts, AND clears the
judge-ensemble-rho regression bar.

statsmodels.stats.weightstats.ttost_ind is NOT installed in this environment
(confirmed by scripts/sieve_env_precheck.py: statsmodels_ttost_available=false,
recorded in 11-01-SUMMARY.md). Per 11-RESEARCH.md's Don't-Hand-Roll guidance,
this falls back to a hand-rolled two-one-sided-t-test using scipy.stats.t
(same review bar as scripts/bootstrap_gate.py -- Welch/unequal-variance,
no new dependency).

Functions
---------
tost_equivalence(a, b, epsilon, alpha=0.05) -> bool
    Plain-bool TOST verdict (tests/test_tost_gate.py contract).

tost_equivalent(k_scores, full_scores, epsilon, alpha=0.05) -> dict
    Full TOST record: {equivalent, p_lower, p_upper, mean_diff, ci}.

run_gate(...) -> dict
    Orchestrates the k-sweep: loads per-item wp-bench arrays for k in
    {13, 32, 64} (candidate budgets) against the full arm, evaluates all
    three SIEVE-05 sub-gates per k, and declares optimal_k (smallest k
    passing all three) or no_equivalent_k=true.

CLI
---
python3 scripts/tost_gate.py
    Reads output/sieve/k_sweep_results.json + output/relabel/gate_noise_floors.json,
    writes output/sieve/optimal_k.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as _t_dist

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_KS = [13, 32, 64]  # ascending -- "smallest k" declaration order
EPSILON_DEFAULT = 0.02  # 2pp equivalence margin, 11-CONTEXT.md / SIEVE-05

# Canonical (Tinker-native) ship-artifact bars, per 11-CONTEXT.md "Ship artifact"
# table. Recorded here for traceability ONLY -- per established fact (serving
# gap ~3pp, sanity_gate_recalibration.json), the ACTUAL judge-rho regression
# reference used to gate k is the vLLM-measured full arm from k_sweep_results.json
# (like-for-like, same stack both sides), not these canonical numbers.
CANONICAL_ENSEMBLE_RHO = 0.842
CANONICAL_S1_RHO = 0.827

_KSWEEP_DIR = _PROJECT_ROOT / "output" / "sieve" / "ksweep"


# ---------------------------------------------------------------------------
# Core TOST (hand-rolled two one-sided t-tests, Welch unequal-variance)
# ---------------------------------------------------------------------------


def _tost_core(a: np.ndarray, b: np.ndarray, epsilon: float, alpha: float = 0.05) -> dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    mean_diff = float(a.mean() - b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = float(np.sqrt(v1 / n1 + v2 / n2))

    if se == 0.0:
        # Degenerate (zero pooled variance, e.g. identical constant arrays):
        # equivalence collapses to a direct point comparison against epsilon.
        equivalent = bool(abs(mean_diff) < epsilon)
        return {
            "equivalent": equivalent,
            "p_lower": 0.0 if equivalent else 1.0,
            "p_upper": 0.0 if equivalent else 1.0,
            "mean_diff": mean_diff,
            "ci": [mean_diff, mean_diff],
        }

    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )

    # Lower one-sided test: H0: mean_diff <= -epsilon vs H1: mean_diff > -epsilon
    t_lower = (mean_diff - (-epsilon)) / se
    p_lower = float(1.0 - _t_dist.cdf(t_lower, df))
    # Upper one-sided test: H0: mean_diff >= epsilon vs H1: mean_diff < epsilon
    t_upper = (mean_diff - epsilon) / se
    p_upper = float(_t_dist.cdf(t_upper, df))

    equivalent = bool(p_lower < alpha and p_upper < alpha)

    ci_lo = float(mean_diff - _t_dist.ppf(1 - alpha, df) * se)
    ci_hi = float(mean_diff + _t_dist.ppf(1 - alpha, df) * se)

    return {
        "equivalent": equivalent,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "mean_diff": mean_diff,
        "ci": [ci_lo, ci_hi],
    }


def tost_equivalence(a: np.ndarray, b: np.ndarray, epsilon: float, alpha: float = 0.05) -> bool:
    """Plain-bool TOST verdict. tests/test_tost_gate.py contract."""
    return bool(_tost_core(a, b, epsilon, alpha)["equivalent"])


def tost_equivalent(
    k_scores: np.ndarray, full_scores: np.ndarray, epsilon: float, alpha: float = 0.05
) -> dict[str, Any]:
    """Full TOST record: {equivalent, p_lower, p_upper, mean_diff, ci}."""
    return _tost_core(k_scores, full_scores, epsilon, alpha)


# ---------------------------------------------------------------------------
# Per-item wp-bench loading
# ---------------------------------------------------------------------------


def _load_per_item_wpbench(jsonl_path: Path) -> list[float]:
    """Per-item scalar score from a wp-bench jsonl: `score` for knowledge-type
    records, `correctness` for execution-type records. Not reweighted to match
    the official aggregate `overall` formula -- immaterial here since the
    measured k-sweep gaps (full 0.4484 -> k64 0.2275 -> k32 0.0546) are an
    order of magnitude past epsilon=0.02 either way.
    """
    items: list[float] = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            val = rec.get("score", rec.get("correctness"))
            if val is not None:
                items.append(float(val))
    return items


def _find_ksweep_jsonl(k_label: str) -> Path | None:
    matches = sorted(_KSWEEP_DIR.glob(f"gen_k{k_label}/wp_bench_results_*.jsonl"))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gate(
    k_sweep_path: Path,
    noise_floors_path: Path,
    epsilon: float = EPSILON_DEFAULT,
    alpha: float = 0.05,
) -> dict[str, Any]:
    sweep = json.loads(k_sweep_path.read_text())["sweep"]
    by_k = {arm["k"]: arm for arm in sweep}
    full_arm = by_k.get("full")
    if full_arm is None:
        raise ValueError(f"{k_sweep_path} has no k='full' arm -- cannot TOST against it")

    full_jsonl = _find_ksweep_jsonl("full")
    if full_jsonl is None:
        raise FileNotFoundError("no per-item wp-bench jsonl found for the full arm")
    full_items = _load_per_item_wpbench(full_jsonl)

    noise_floors = json.loads(noise_floors_path.read_text())
    # s1-s2 cross-seed sd/2SE is the seed-noise-floor convention this project
    # uses for judge-rho regression bars (see gate_noise_floors.json G1_rule).
    seed_noise_floor = float(noise_floors["s1-s2"]["two_se"])

    ref_ensemble_rho = float(full_arm["judge_ensemble_rho"])
    ref_s1_rho = float(full_arm["judge_single_s1_rho"])

    bar_ensemble = ref_ensemble_rho - seed_noise_floor
    bar_s1 = ref_s1_rho - seed_noise_floor
    canonical_bar_ensemble = CANONICAL_ENSEMBLE_RHO - seed_noise_floor
    canonical_bar_s1 = CANONICAL_S1_RHO - seed_noise_floor

    per_k: dict[str, Any] = {}
    optimal_k: int | None = None

    for k in CANDIDATE_KS:
        k_str = str(k)
        arm = by_k.get(k_str)
        k_jsonl = _find_ksweep_jsonl(k_str)

        if arm is None or k_jsonl is None:
            per_k[k_str] = {
                "measured": False,
                "tost": None,
                "protected_retained": None,
                "judge_rho_bar_passed": None,
                "equivalent_k": False,
                "note": (
                    "Never executed. Deliberately not run per 11-04-SUMMARY FINAL "
                    "STATE ADDENDUM: the measured collapse (full 0.4484 -> k64 0.2275 "
                    "-> k32 0.0546) is monotone and catastrophic, so smaller/harder-"
                    "masked k is bounded worse by monotonicity -- cannot pass TOST "
                    "regardless of measurement."
                ),
            }
            continue

        k_items = _load_per_item_wpbench(k_jsonl)
        tost = tost_equivalent(np.array(k_items), np.array(full_items), epsilon, alpha)
        protected_retained = bool(arm.get("protected_retained", False))

        judge_rho = arm.get("judge_ensemble_rho")
        if judge_rho is None:
            judge_rho_bar_passed = False
            judge_rho_note = "judge_ensemble_rho not measured at this k (sweep session ended before judge capture)."
        else:
            judge_rho_bar_passed = bool(judge_rho >= bar_ensemble)
            judge_rho_note = None

        equivalent_k = bool(tost["equivalent"] and protected_retained and judge_rho_bar_passed)

        per_k[k_str] = {
            "measured": True,
            "tost": tost,
            "wp_bench": arm.get("wp_bench"),
            "protected_retained": protected_retained,
            "judge_ensemble_rho": judge_rho,
            "judge_rho_bar_passed": judge_rho_bar_passed,
            "judge_rho_note": judge_rho_note,
            "equivalent_k": equivalent_k,
        }

        if equivalent_k and optimal_k is None:
            optimal_k = k

    no_equivalent_k = optimal_k is None

    return {
        "epsilon": epsilon,
        "tost_reference": "vLLM-measured full arm in k_sweep_results.json (same stack both sides, per sanity_gate_recalibration.json)",
        "full_arm": {
            "wp_bench": full_arm.get("wp_bench"),
            "judge_ensemble_rho": ref_ensemble_rho,
            "judge_single_s1_rho": ref_s1_rho,
        },
        "seed_noise_floor": seed_noise_floor,
        "judge_rho_bar_ensemble": bar_ensemble,
        "judge_rho_bar_s1_fallback": bar_s1,
        "canonical_bars_for_traceability": {
            "note": "Tinker-native 0.842/0.827 minus seed_noise_floor -- NOT the gating reference (serving gap, see sanity_gate_recalibration.json). Recorded for traceability only.",
            "ensemble": canonical_bar_ensemble,
            "s1_fallback": canonical_bar_s1,
        },
        "per_k": per_k,
        "optimal_k": optimal_k if optimal_k is not None else "full",
        "no_equivalent_k": no_equivalent_k,
    }


def main() -> int:
    result = run_gate(
        k_sweep_path=_PROJECT_ROOT / "output" / "sieve" / "k_sweep_results.json",
        noise_floors_path=_PROJECT_ROOT / "output" / "relabel" / "gate_noise_floors.json",
    )
    out_path = _PROJECT_ROOT / "output" / "sieve" / "optimal_k.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"optimal_k={result['optimal_k']} no_equivalent_k={result['no_equivalent_k']}", file=sys.stderr)
    print(json.dumps({"optimal_k": result["optimal_k"], "no_equivalent_k": result["no_equivalent_k"]}))
    return 0


def _self_check() -> None:
    """Runnable self-check: synthetic equivalent + non-equivalent arrays."""
    rng = np.random.default_rng(42)
    equiv_a = rng.normal(0.85, 0.01, 100)
    equiv_b = rng.normal(0.855, 0.01, 100)
    assert tost_equivalence(equiv_a, equiv_b, epsilon=0.02) is True, "tight arrays should be equivalent"

    noneq_a = rng.normal(0.85, 0.01, 100)
    noneq_b = rng.normal(0.60, 0.01, 100)
    assert tost_equivalence(noneq_a, noneq_b, epsilon=0.02) is False, "far-apart arrays should NOT be equivalent"

    full = tost_equivalent(equiv_a, equiv_b, epsilon=0.02)
    assert set(full.keys()) == {"equivalent", "p_lower", "p_upper", "mean_diff", "ci"}
    assert isinstance(full["equivalent"], bool)

    print("self-check OK: tost_equivalence/tost_equivalent behave correctly on synthetic arrays", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
    sys.exit(main())
