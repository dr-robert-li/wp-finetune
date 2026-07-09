"""Winner-selection rule for Phase 13 pruning (PRUNE-05).

Consumes the per-variant gate results that 13-04/13-05 produce (gen wp-bench,
judge 3-seed ensemble rho, judge parse-rate, per-dimension D2_security
retention, protected_retained) and applies the eligibility gate pinned in
13-CONTEXT / output/sieve/prune_set_for_phase13.json:

    eligible iff ALL of:
        gen_wp_bench        >= 0.4284  (vLLM-measured gen bar)
        judge_ensemble_rho  >= 0.7555  (vLLM-measured judge bar)
        judge_parse_rate    >= 0.95
        d2_security_retention within 2pp of d2_security_baseline (never MORE
            than 2pp below baseline -- being at or above baseline is fine)
        protected_retained is True
        physically feasible: k >= max over layers of protected_count[layer]
            (PRUNE-06 requires a UNIFORM per-layer keep-count K; a layer
            whose protected-expert count exceeds K cannot be realized without
            dropping a protected expert -- derived this session, not
            advisory. Real protected mask: layer 1 alone carries 40 protected
            experts, so ratio=75/K=32 is NEVER a shippable winner.)

Among eligible variants: prefer the SMALLER k (higher compression); ties
broken by higher d2_security_retention. If the eligible set is empty, return
an explicit no_winner verdict with the per-variant failure reasons (a
legitimate outcome per 13-CONTEXT -- the phase ships unpruned).

Input contract (one dict per (method, ratio) candidate -- 13-04/13-05 merge
their gen-axis + judge-axis + per-dimension gated-eval records into this
shape before calling select_winner):
    {
        "method": "aimer" | "reap",
        "ratio": 25 | 50 | 75,
        "gen_wp_bench": float,
        "judge_ensemble_rho": float,
        "judge_parse_rate": float,
        "d2_security_retention": float,   # measured, pruned variant
        "d2_security_baseline": float,    # unpruned-model reference
        "protected_retained": bool,
    }

Usage (CLI, real gate results once 13-04/13-05 populate output/prune/gated/):
    python -m scripts.prune_selection \
        --records-dir output/prune/gated \
        --protected-mask output/profiling/reasoning-merged-v4/protected_expert_mask.npy \
        --out output/prune/selection.json

    python -m scripts.prune_selection --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prune_gated_eval import RATIO_TO_K  # noqa: E402  (reuse, do not redefine)

# vLLM-measured regression bars (output/sieve/prune_set_for_phase13.json
# regression_bars block) -- same constants prune_gated_eval.py gates on.
GEN_WPBENCH_FLOOR = 0.4284
JUDGE_ENS_RHO_FLOOR = 0.7555
JUDGE_PARSE_FLOOR = 0.95
D2_SECURITY_TOLERANCE_PP = 0.02

# Required per-variant record fields (missing/None fields fail closed --
# never treated as a silent pass; T-13-02 mitigation).
REQUIRED_FIELDS = (
    "method", "ratio", "gen_wp_bench", "judge_ensemble_rho", "judge_parse_rate",
    "d2_security_retention", "d2_security_baseline", "protected_retained",
)


def max_protected_per_layer(protected: np.ndarray) -> int:
    """Max count of protected experts in any single layer -- the K floor below
    which NO ratio can physically ship (PRUNE-06's uniform-per-layer constraint)."""
    return int(protected.sum(axis=1).max())


def evaluate_variant(record: dict, max_protected: int) -> tuple[bool, list[str]]:
    """Return (eligible, reasons). reasons lists every failed check (empty if eligible)."""
    reasons: list[str] = []

    for field in REQUIRED_FIELDS:
        if record.get(field) is None:
            reasons.append(f"missing_field:{field}")
    if reasons:
        return False, reasons  # can't evaluate further checks without the data

    if record["ratio"] not in RATIO_TO_K:
        reasons.append(f"unknown_ratio:{record['ratio']}")
        return False, reasons
    k = RATIO_TO_K[record["ratio"]]

    if record["gen_wp_bench"] < GEN_WPBENCH_FLOOR:
        reasons.append(f"gen_wp_bench {record['gen_wp_bench']:.4f} < floor {GEN_WPBENCH_FLOOR}")
    if record["judge_ensemble_rho"] < JUDGE_ENS_RHO_FLOOR:
        reasons.append(
            f"judge_ensemble_rho {record['judge_ensemble_rho']:.4f} < floor {JUDGE_ENS_RHO_FLOOR}"
        )
    if record["judge_parse_rate"] < JUDGE_PARSE_FLOOR:
        reasons.append(
            f"judge_parse_rate {record['judge_parse_rate']:.4f} < floor {JUDGE_PARSE_FLOOR}"
        )
    d2_delta = record["d2_security_baseline"] - record["d2_security_retention"]
    if d2_delta > D2_SECURITY_TOLERANCE_PP:
        reasons.append(
            f"d2_security regressed {d2_delta:.4f} > tolerance {D2_SECURITY_TOLERANCE_PP}"
        )
    if not record["protected_retained"]:
        reasons.append("protected_retained is False")
    if k < max_protected:
        reasons.append(
            f"physically infeasible: k={k} < max_protected_per_layer={max_protected}"
        )

    return (len(reasons) == 0), reasons


def select_winner(results: list[dict], protected: np.ndarray) -> dict:
    """Apply the full eligibility gate to every variant and pick the winner.

    Returns:
        {
            "verdict": "winner" | "no_winner",
            "winner": {"method", "ratio", "k"} | None,
            "max_protected_per_layer": int,
            "per_variant": [{"method", "ratio", "k", "eligible", "reasons"}, ...],
        }
    """
    max_protected = max_protected_per_layer(protected)

    per_variant = []
    eligible_variants = []
    for record in results:
        eligible, reasons = evaluate_variant(record, max_protected)
        k = RATIO_TO_K.get(record.get("ratio"))
        entry = {
            "method": record.get("method"),
            "ratio": record.get("ratio"),
            "k": k,
            "eligible": eligible,
            "reasons": reasons,
        }
        per_variant.append(entry)
        if eligible:
            eligible_variants.append((record, k))

    if not eligible_variants:
        return {
            "verdict": "no_winner",
            "winner": None,
            "max_protected_per_layer": max_protected,
            "per_variant": per_variant,
        }

    # Prefer smaller k (higher compression); ties -> higher d2_security_retention.
    winner_record, winner_k = min(
        eligible_variants, key=lambda rk: (rk[1], -rk[0]["d2_security_retention"])
    )
    return {
        "verdict": "winner",
        "winner": {
            "method": winner_record["method"],
            "ratio": winner_record["ratio"],
            "k": winner_k,
        },
        "max_protected_per_layer": max_protected,
        "per_variant": per_variant,
    }


def load_variant_records(records_dir: str | Path) -> list[dict]:
    """Merge output/prune/gated/{method}_{ratio}_{gen,judge,d2}.json triples into
    one per-variant record each. A missing d2 file leaves d2 fields as None
    (fails eligibility closed, per T-13-02 -- never silently skipped)."""
    records_dir = Path(records_dir)
    variants: dict[tuple[str, int], dict] = {}

    for path in sorted(records_dir.glob("*.json")):
        stem = path.stem  # e.g. "aimer_25_gen"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix, axis = parts
        method_ratio = prefix.rsplit("_", 1)
        if len(method_ratio) != 2:
            continue
        method, ratio_str = method_ratio
        try:
            ratio = int(ratio_str)
        except ValueError:
            continue

        key = (method, ratio)
        variant = variants.setdefault(
            key,
            {
                "method": method, "ratio": ratio,
                "gen_wp_bench": None, "judge_ensemble_rho": None, "judge_parse_rate": None,
                "d2_security_retention": None, "d2_security_baseline": None,
                "protected_retained": None,
            },
        )
        data = json.loads(path.read_text())

        if axis == "gen":
            variant["gen_wp_bench"] = data.get("wp_bench")
            if data.get("protected_retained") is not None:
                variant["protected_retained"] = data["protected_retained"]
            if "d2_security_retention" in data:
                variant["d2_security_retention"] = data["d2_security_retention"]
            if "d2_security_baseline" in data:
                variant["d2_security_baseline"] = data["d2_security_baseline"]
        elif axis == "judge":
            variant["judge_ensemble_rho"] = data.get("judge_ensemble_rho")
            variant["judge_parse_rate"] = data.get("parse_rate")
            if data.get("protected_retained") is not None:
                variant["protected_retained"] = data["protected_retained"]
        elif axis == "d2":
            variant["d2_security_retention"] = data.get("d2_security_retention")
            variant["d2_security_baseline"] = data.get("d2_security_baseline")

    return list(variants.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records-dir", default="output/prune/gated")
    ap.add_argument(
        "--protected-mask",
        default="output/profiling/reasoning-merged-v4/protected_expert_mask.npy",
    )
    ap.add_argument("--out", default="output/prune/selection.json")
    args = ap.parse_args()

    protected = np.load(args.protected_mask)
    results = load_variant_records(args.records_dir)
    verdict = select_winner(results, protected)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))

    if verdict["verdict"] == "winner":
        print(f"WINNER: {verdict['winner']}")
    else:
        print("NO WINNER -- phase ships unpruned")
        for v in verdict["per_variant"]:
            if not v["eligible"]:
                print(f"  {v['method']} ratio={v['ratio']}: {v['reasons']}")
    print(f"Wrote {out_path}")
    return 0


def _self_check() -> None:
    """Assert-based self-check on a tiny synthetic gate-result table."""
    n_layers, n_experts = 4, 128
    protected = np.zeros((n_layers, n_experts), dtype=bool)
    protected[1, :40] = True  # layer 1 carries 40 protected experts (mirrors real data)
    assert max_protected_per_layer(protected) == 40

    clean_25 = {
        "method": "aimer", "ratio": 25, "gen_wp_bench": 0.45, "judge_ensemble_rho": 0.78,
        "judge_parse_rate": 0.97, "d2_security_retention": 0.90, "d2_security_baseline": 0.90,
        "protected_retained": True,
    }
    infeasible_75 = {
        "method": "aimer", "ratio": 75, "gen_wp_bench": 0.46, "judge_ensemble_rho": 0.80,
        "judge_parse_rate": 0.98, "d2_security_retention": 0.91, "d2_security_baseline": 0.90,
        "protected_retained": True,
    }  # k=32 < max_protected(40) -> physically infeasible even though bars pass
    d2_regression_50 = {
        "method": "aimer", "ratio": 50, "gen_wp_bench": 0.44, "judge_ensemble_rho": 0.77,
        "judge_parse_rate": 0.96, "d2_security_retention": 0.60, "d2_security_baseline": 0.90,
        "protected_retained": True,
    }  # d2 regressed 30pp > 2pp tolerance -> disqualified

    verdict = select_winner([clean_25, infeasible_75, d2_regression_50], protected)
    assert verdict["verdict"] == "winner"
    assert verdict["winner"] == {"method": "aimer", "ratio": 25, "k": 96}
    by_ratio = {v["ratio"]: v for v in verdict["per_variant"]}
    assert by_ratio[25]["eligible"] is True
    assert by_ratio[75]["eligible"] is False
    assert any("infeasible" in r for r in by_ratio[75]["reasons"])
    assert by_ratio[50]["eligible"] is False
    assert any("d2_security regressed" in r for r in by_ratio[50]["reasons"])

    no_winner_verdict = select_winner([infeasible_75, d2_regression_50], protected)
    assert no_winner_verdict["verdict"] == "no_winner"
    assert no_winner_verdict["winner"] is None

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
