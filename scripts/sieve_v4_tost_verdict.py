#!/usr/bin/env python
"""CI-aware TOST verdict scorer for the v4 judge MoE-Sieve k-sweep (Plan 25-02, GATE4-03).

Reads output/sieve-v4/k_sweep_results_v4.json (produced by sieve_ksweep_v4_run.py),
re-scores each arm's judge capture with the EXACT s1 rho pipeline eval_relabel.py uses
(parse_judge_scores + _derive_prose_overall, val:wj_rows index join, spearmanr vs
val_labels_v1), and for every masked k computes the paired-bootstrap CI of
(masked_s1_rho - full_s1_rho) over the common items and applies TOST at epsilon=0.02.

CI-aware (pre-registration carry-forward #4): the disposition uses the bootstrap
lower/upper BOUND, not the point estimate. equivalent iff [ci_lo, ci_hi] ⊂ (-eps, +eps).

Reference = the same-stack vLLM `full` arm measured in THIS sweep (all-keep mask) — NOT
the llama.cpp Q8 0.8067 nor the Tinker 0.8358.

Verdict: optimal_k = smallest k that (a) passes CI-aware TOST vs the full arm AND
(b) retains every protected expert; else optimal_k = "full" (no_winner). Both valid,
both route Phase 26.

Usage:
    .venv-tinker/bin/python scripts/sieve_v4_tost_verdict.py output/sieve-v4/k_sweep_results_v4.json
    .venv-tinker/bin/python scripts/sieve_v4_tost_verdict.py --self-check   # no GPU, synthetic
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

EPSILON = 0.02
N_BOOT = 2000
BOOT_SEED = 7
VAL_DATASET = PROJECT_ROOT / "data/reasoning_dataset/openai_val.jsonl"
VAL_LABELS = PROJECT_ROOT / "output/relabel/val_labels_v1.json"
RESULTS_PATH = PROJECT_ROOT / "output/sieve-v4/k_sweep_results_v4.json"
VERDICT_PATH = PROJECT_ROOT / "output/sieve-v4/optimal_k_v4.json"


# ---- scoring (byte-identical to eval_relabel.py's join) ---------------------

def _wj_rows() -> list[int]:
    rows = [json.loads(l) for l in open(VAL_DATASET) if l.strip()]
    return [i for i, r in enumerate(rows) if next(
        (m["content"] for m in r["messages"] if m["role"] == "user"), ""
    ).startswith("<wp_judge>")]


def load_labels() -> dict[str, float]:
    return {k: v for k, v in json.load(open(VAL_LABELS)).items() if k.startswith("val:")}


def score_capture(capture_path: str | Path) -> tuple[dict[str, float], int]:
    """Return ({val:key -> judge overall}, parse_fail_count) for one capture jsonl."""
    from eval.eval_judge import _derive_prose_overall
    from eval.output_parsers import load_dim_map, parse_judge_scores

    dm = load_dim_map()
    dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
    wj = _wj_rows()

    scores: dict[str, float] = {}
    parse_fail = 0
    for line in open(capture_path):
        r = json.loads(line)
        if "index" not in r:
            continue
        parsed = parse_judge_scores(r["response"], "auto")
        if not parsed or not parsed.get("dimension_scores"):
            parse_fail += 1
            continue
        o = (float(parsed["overall"]) if "overall" in parsed
             else _derive_prose_overall(parsed["dimension_scores"], dw))
        scores[f"val:{wj[r['index']]}"] = o
    return scores, parse_fail


# ---- TOST (CI-aware, paired bootstrap of rho difference) --------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def paired_bootstrap_delta(masked_vals: np.ndarray, full_vals: np.ndarray,
                           label_vals: np.ndarray, n_boot: int = N_BOOT,
                           seed: int = BOOT_SEED) -> dict:
    """Paired-bootstrap CI of (spearman(masked,labels) - spearman(full,labels)).

    All three arrays are aligned over the SAME common items (index i is the same
    val item in each). Returns point mean_diff + 95% percentile CI [lo, hi].
    """
    m, f, l = np.asarray(masked_vals, float), np.asarray(full_vals, float), np.asarray(label_vals, float)
    n = len(l)
    point = _spearman(m, l) - _spearman(f, l)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = _spearman(m[idx], l[idx]) - _spearman(f[idx], l[idx])
        if not np.isnan(d):
            deltas.append(d)
    deltas = np.sort(deltas)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return {"mean_diff": point, "ci": [lo, hi], "n": n}


def tost_equivalent(ci: list[float], eps: float = EPSILON) -> bool:
    """CI-aware TOST: equivalent iff the whole CI lies strictly within (-eps, +eps)."""
    lo, hi = ci
    return (lo > -eps) and (hi < eps)


def tost_from_scores(masked: dict[str, float], full: dict[str, float],
                     labels: dict[str, float], eps: float = EPSILON) -> dict:
    common = sorted(set(masked) & set(full) & set(labels))
    if len(common) < 3:
        return {"equivalent": False, "mean_diff": None, "ci": [None, None],
                "n": len(common), "note": "insufficient common items"}
    res = paired_bootstrap_delta(
        np.array([masked[k] for k in common]),
        np.array([full[k] for k in common]),
        np.array([labels[k] for k in common]), seed=BOOT_SEED,
    )
    res["equivalent"] = tost_equivalent(res["ci"], eps)
    return res


# ---- verdict over the whole sweep -------------------------------------------

def _k_int(k) -> int | None:
    try:
        return int(k)
    except (ValueError, TypeError):
        return None


def compute_verdict(results_path: Path = RESULTS_PATH, eps: float = EPSILON) -> dict:
    data = json.loads(Path(results_path).read_text())
    sweep = data.get("sweep", data if isinstance(data, list) else [])
    by_k = {r["k"]: r for r in sweep}
    if "full" not in by_k:
        raise SystemExit("no 'full' arm in k_sweep_results_v4.json — cannot set TOST reference")

    full = by_k["full"]
    full_scores, _ = score_capture(PROJECT_ROOT / full["judge_capture"])
    labels = load_labels()

    per_k = {}
    for k, arm in by_k.items():
        if k == "full":
            continue
        masked_scores, parse_fail = score_capture(PROJECT_ROOT / arm["judge_capture"])
        tost = tost_from_scores(masked_scores, full_scores, labels, eps)
        per_k[k] = {
            "measured": True,
            "judge_single_s1_rho": arm.get("judge_single_s1_rho"),
            "parse_fail": parse_fail,
            "protected_retained": arm.get("protected_retained"),
            "kept_experts_per_layer_min": (min(arm["kept_experts_per_layer"])
                                           if arm.get("kept_experts_per_layer") else None),
            "tost": tost,
        }

    # smallest k (max compression) that passes CI-aware TOST AND retains protected experts
    candidates = sorted((k for k in per_k
                         if per_k[k]["tost"]["equivalent"] and per_k[k]["protected_retained"]),
                        key=lambda k: _k_int(k) if _k_int(k) is not None else 1 << 30)
    if candidates:
        optimal_k = candidates[0]
        no_winner = False
    else:
        optimal_k = "full"
        no_winner = True

    if no_winner:
        phase26 = ("no_winner (optimal_k=full): no sub-full k is CI-aware-equivalent to the "
                   "same-stack full arm. Phase 26 prune may still be attempted per ROADMAP, or the "
                   "reopened v4 compression question closes with v3 (30.2 GiB Q8) staying canonical.")
    else:
        phase26 = (f"optimal_k={optimal_k}: the v4 judge compresses at this expert budget. Phase 26 "
                   f"merges + prunes at k={optimal_k} and re-checks vs v3's 30.2 GiB Q8.")

    return {
        "requirement": "GATE4-03",
        "epsilon": eps,
        "ci_aware": True,
        "tost_reference": {
            "arm": "full",
            "judge_single_s1_rho": full.get("judge_single_s1_rho"),
            "source": "same-stack vLLM full arm (all-keep mask) measured in this sweep",
            "note": "NOT the llama.cpp Q8 0.8067 nor the Tinker 0.8358 — both TOST sides same stack",
        },
        "full_s1_rho": full.get("judge_single_s1_rho"),
        "per_k": per_k,
        "optimal_k": optimal_k,
        "no_equivalent_k": no_winner,
        "no_winner": no_winner,
        "disposition": "no_winner" if no_winner else "optimal_k",
        "phase26_routing": phase26,
    }


def _self_check() -> None:
    """No-GPU synthetic check: one clearly-equivalent arm, one clearly-not."""
    rng = np.random.default_rng(0)
    n = 121
    labels = {f"val:{i}": float(v) for i, v in enumerate(rng.uniform(0, 1, n))}
    lab = np.array([labels[f"val:{i}"] for i in range(n)])

    # full arm: correlated with labels (rho ~ 0.8)
    full_vals = 0.8 * lab + 0.2 * rng.uniform(0, 1, n)
    full = {f"val:{i}": float(full_vals[i]) for i in range(n)}

    # equivalent arm: full + tiny noise -> rho difference ~ 0, CI inside +-0.02
    eq_vals = full_vals + rng.normal(0, 0.002, n)
    equivalent = {f"val:{i}": float(eq_vals[i]) for i in range(n)}

    # non-equivalent arm: near-random -> rho collapses, CI well below -0.02
    ne_vals = rng.uniform(0, 1, n)
    nonequiv = {f"val:{i}": float(ne_vals[i]) for i in range(n)}

    r_eq = tost_from_scores(equivalent, full, labels)
    r_ne = tost_from_scores(nonequiv, full, labels)

    print(f"[self-check] equivalent arm: mean_diff={r_eq['mean_diff']:+.4f} "
          f"ci={[round(c, 4) for c in r_eq['ci']]} equivalent={r_eq['equivalent']}")
    print(f"[self-check] non-equiv  arm: mean_diff={r_ne['mean_diff']:+.4f} "
          f"ci={[round(c, 4) for c in r_ne['ci']]} equivalent={r_ne['equivalent']}")

    assert r_eq["equivalent"] is True, "clearly-equivalent arm must pass TOST"
    assert r_ne["equivalent"] is False, "clearly-non-equivalent arm must fail TOST"

    # CI-aware guard: a point diff inside 2pp but a CI spilling past -2pp is NOT equivalent
    assert tost_equivalent([-0.01, 0.01]) is True
    assert tost_equivalent([-0.03, 0.01]) is False, "CI spilling past -eps must fail (point-inside is not enough)"
    assert tost_equivalent([-0.005, 0.05]) is False, "CI spilling past +eps must fail"
    print("[self-check] TOST CI-aware dispositions OK")


def main() -> int:
    if "--self-check" in sys.argv:
        _self_check()
        return 0
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_PATH
    verdict = compute_verdict(results)
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2))
    print(f"verdict: optimal_k={verdict['optimal_k']} disposition={verdict['disposition']}")
    print(f"full_s1_rho={verdict['full_s1_rho']}")
    for k, e in verdict["per_k"].items():
        t = e["tost"]
        print(f"  k={k}: s1_rho={e['judge_single_s1_rho']} parse_fail={e['parse_fail']} "
              f"protected_retained={e['protected_retained']} "
              f"tost.equivalent={t['equivalent']} mean_diff={t['mean_diff']} ci={t['ci']}")
    print(f"wrote {VERDICT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
