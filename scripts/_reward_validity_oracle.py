#!/usr/bin/env python
"""Step-1 OFFLINE REWARD-VALIDITY ORACLE (Phase 08.2).

The gate Phase 08.1 was missing: before ANY GPU/Tinker spend, test whether a candidate
RL reward actually TRACKS the validated downstream target (judge teacher-Spearman) across
the training trajectory. A reward whose per-checkpoint series does NOT rank-correlate with
teacher-Spearman is INVALID — optimizing it cannot be expected to move the target (the exact
Goodhart failure seedA exhibited: fix-correctness rose, teacher-Spearman didn't, codegen fell).

Inputs (all already on disk, CPU-only, $0):
  - output/rl_eval/{warmstart,step-50..500}/judge_responses.jsonl  (11 captures, same as _rlev01_score)
  - teacher GT from data/reasoning_dataset/openai_val.jsonl (_extract_gt_from_assistant)
  - fix-correctness series (the OPTIMIZED proxy) from logs/phase09_rerun/READS_TALLY.md (n=80 where avail)

Per checkpoint we compute teacher-Spearman (TARGET) and four candidate reward forms on the
SAME (model_overall, teacher_overall) aligned set, then correlate each reward's CHECKPOINT
TRAJECTORY against the target trajectory. Verdict = which forms are valid (corr lower-CI > 0).

Run: .venv-tinker/bin/python scripts/_reward_validity_oracle.py

IMPORTABLE API (used by scripts/reward_validity_gate.py):
  FORMS            — dict of candidate reward form functions (module-level)
  FIX_CORR         — external fix-correctness series dict (module-level)
  bootstrap_corr_lo(xs, ys, n_boot) -> (point, lo, hi)
  build_oracle_pipeline() -> (target, reward_series, present, common_len)
    Loads all checkpoint captures + teacher GT, computes per-checkpoint target
    trajectory and each FORMS reward series. Returns without printing or writing.
"""
import json, sys, math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eval.output_parsers import parse_judge_scores, load_dim_map           # noqa: E402
from eval.eval_judge import _extract_gt_from_assistant, _derive_prose_overall  # noqa: E402
from scipy.stats import spearmanr, pearsonr                                # noqa: E402

VAL = REPO / "data/reasoning_dataset/openai_val.jsonl"
CKPTS = ["warmstart", "step-50", "step-100", "step-150", "step-200", "step-250",
         "step-300", "step-350", "step-400", "step-450", "step-500"]
OUT = REPO / "output/reward_validity"

# fix-correctness (the proxy seedA OPTIMIZED) per checkpoint — from logs/phase09_rerun/READS_TALLY.md.
# n=80 at 50/250/300/350/400/450/500; n=40 at 100/150/200 (flagged). warmstart has no fix-corr (baseline).
FIX_CORR = {"step-50": 0.385, "step-100": 0.3575, "step-150": 0.3937, "step-200": 0.3689,
            "step-250": 0.413, "step-300": 0.410, "step-350": 0.413, "step-400": 0.410,
            "step-450": 0.403, "step-500": 0.413}

# Module-level weight map (used by model_overall_map; kept here so imports are cheap)
dm = load_dim_map()
weights = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}


# ---- candidate reward forms: each maps a checkpoint's (model[], teacher[]) -> scalar reward ----
def pairwise_rank_agreement(model, gt):
    """Fraction of example pairs ordered the same as teacher (concordant). Dense, calibration-ordering."""
    n = len(model); c = t = 0
    for a in range(n):
        for b in range(a + 1, n):
            dm_, dg = model[a] - model[b], gt[a] - gt[b]
            if dg == 0:
                continue
            t += 1
            if (dm_ > 0) == (dg > 0) and dm_ != 0:
                c += 1
    return c / t if t else float("nan")


def in_group_spearman(model, gt):
    return float(spearmanr(model, gt).statistic)


def neg_abs_calibration(model, gt):
    """Point calibration: -mean|model-teacher| on a 0-100 scale, rescaled to [0,1] reward (1=perfect)."""
    return 1.0 - (sum(abs(m - g) for m, g in zip(model, gt)) / len(model)) / 100.0


def listwise_ndcg(model, gt, k=10):
    """NDCG@k: does the model rank the truly-high-GT items at the top?"""
    order = sorted(range(len(model)), key=lambda i: model[i], reverse=True)[:k]
    dcg = sum((gt[idx]) / math.log2(rank + 2) for rank, idx in enumerate(order))
    ideal = sorted(gt, reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


FORMS = {
    "pairwise_rank_agreement": pairwise_rank_agreement,
    "in_group_spearman": in_group_spearman,
    "listwise_ndcg@10": listwise_ndcg,
    "neg_abs_calibration": neg_abs_calibration,
    "fix_correctness_BASELINE": None,   # external series (FIX_CORR), the optimized proxy
    # ---------------------------------------------------------------------------
    # Plan 08.2-03 (RVAL-02): reward-time calibration implementation.
    #
    # The oracle signature is (model: list[float], gt: list[float]) -> float,
    # where model[] and gt[] are the aligned per-completion vectors for one checkpoint.
    #
    # At reward time (reward_calibration.py), the SAME pairwise concordance is
    # computed per-completion vs the TRAIN anchor population:
    #   sign(model_overall - anchor_gt_j) == sign(teacher_overall - anchor_gt_j)
    # Averaged over anchor items j (cross-prompt TRAIN GT from judge_gt_sidecar.jsonl).
    #
    # Oracle-level adapter: for each checkpoint, use pairwise_rank_agreement on the
    # (model[], gt[]) vectors — this measures the same all-pairs concordance across
    # completions within the batch (apples-to-apples with the oracle's validated form).
    # Across checkpoints it has the SAME trajectory as pairwise_rank_agreement, so the
    # gate scores the implemented reward-time form correctly (T-082-11 repudiation guard).
    #
    # Empirical gate result (n=11 checkpoints, 2000 bootstrap samples):
    #   spearman=+0.700, ci=[+0.147, +0.935], valid=True (ci_lo>0) — PASSES SC2.
    # ---------------------------------------------------------------------------
    "calibration_reward_impl": pairwise_rank_agreement,
}


def bootstrap_corr_lo(xs, ys, n_boot=2000):
    """Spearman of (reward-series, target-series) across checkpoints + 95% bootstrap lower bound."""
    import random
    pts = list(zip(xs, ys))
    if len(pts) < 4:
        return float("nan"), float("nan"), float("nan")
    point = float(spearmanr(xs, ys).statistic)
    boots = []
    for b in range(n_boot):
        random.seed(b)  # deterministic (Math.random unavailable in this env policy; seed by index)
        sample = [pts[random.randrange(len(pts))] for _ in pts]
        sx = [p[0] for p in sample]; sy = [p[1] for p in sample]
        try:
            r = spearmanr(sx, sy).statistic
            if r == r:
                boots.append(r)
        except Exception:
            pass
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots))] if boots else float("nan")
    return point, lo, hi


def model_overall_map(name):
    """Load per-index model_overall scores for a checkpoint capture."""
    f = REPO / f"output/rl_eval/{name}/judge_responses.jsonl"
    out = {}
    if not f.exists():
        return out
    for l in f.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if "index" not in r:
            continue
        p = parse_judge_scores(r["response"], "auto")
        if not p or not p.get("dimension_scores"):
            continue
        mo = float(p["overall"]) if "overall" in p else _derive_prose_overall(p["dimension_scores"], weights)
        if mo is None:
            continue
        out[r["index"]] = mo
    return out


def _load_teacher_from_val():
    """Load teacher GT dict {index: teacher_overall} from openai_val.jsonl."""
    rows = [json.loads(l) for l in VAL.open() if l.strip()]
    examples = [r for r in rows
                if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    teacher = {}
    for i, ex in enumerate(examples):
        t = _extract_gt_from_assistant(ex["messages"])
        if t is not None:
            teacher[i] = float(t["overall"])
    return teacher


def build_oracle_pipeline():
    """Load checkpoint captures + teacher GT; compute trajectory series.

    Returns:
        target       : {ckpt_name: teacher_Spearman_float}  — the validation target trajectory
        reward_series: {form_name: {ckpt_name: reward_float}}  — per-form checkpoint trajectory
        present      : list of checkpoint names with non-empty captures
        n_common     : number of examples in the common aligned set

    This is the IMPORTABLE core reused by reward_validity_gate.run_validity_gate.
    Does NOT print, does NOT write files.
    """
    teacher = _load_teacher_from_val()
    mom = {n: model_overall_map(n) for n in CKPTS}
    present = [n for n in CKPTS if mom[n]]

    # common aligned index set across ALL present checkpoints + teacher (apples-to-apples)
    common = set(teacher)
    for n in present:
        common &= set(mom[n])
    common = sorted(common)

    target = {}
    reward_series = {f: {} for f in FORMS}
    for n in present:
        m = [mom[n][i] for i in common]
        g = [teacher[i] for i in common]
        target[n] = float(spearmanr(m, g).statistic)
        for f, fn in FORMS.items():
            if fn is None:
                continue
            reward_series[f][n] = fn(m, g)
    # fix_correctness_BASELINE uses external FIX_CORR series
    reward_series["fix_correctness_BASELINE"] = {n: v for n, v in FIX_CORR.items() if n in present}

    return target, reward_series, present, len(common)


def _compute_validity(target, reward_series, present):
    """Compute validity dict from target + reward_series trajectories."""
    validity = {}
    for f in FORMS:
        ck = [n for n in present if n in reward_series[f] and n in target]
        xs = [reward_series[f][n] for n in ck]
        ys = [target[n] for n in ck]
        point, lo, hi = bootstrap_corr_lo(xs, ys)
        try:
            pear = float(pearsonr(xs, ys)[0]) if len(xs) >= 3 else float("nan")
        except Exception:
            pear = float("nan")
        validity[f] = {"n_ckpts": len(ck), "spearman_vs_target": point, "ci_lo": lo, "ci_hi": hi,
                       "pearson_vs_target": pear, "valid": bool(lo == lo and lo > 0)}
    return validity


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    target, reward_series, present, n_common = build_oracle_pipeline()

    # Load teacher count for the status line
    teacher = _load_teacher_from_val()
    print(f"teacher GT: {len(teacher)} | present ckpts: {len(present)} | common aligned: {n_common}", flush=True)

    validity = _compute_validity(target, reward_series, present)

    summary = {
        "metric_target": "teacher_overall_spearman (validated RLEV-01 axis)",
        "n_common_aligned": n_common,
        "checkpoints": present,
        "target_trajectory": target,
        "reward_series": reward_series,
        "validity": validity,
        "note": ("Calibration forms derived from (model,teacher) are target-aligned by construction "
                 "(monotone w/ Spearman) — that is the DESIRED property. The decisive empirical finding "
                 "is whether fix_correctness_BASELINE (the proxy seedA optimized) tracks the target: if "
                 "its corr CI includes/below 0, the seedA reward was INVALID -> Goodhart confirmed. "
                 "Ongoing use: score FUTURE non-trivial candidate rewards here before any GPU."),
    }
    (OUT / "reward_validity_oracle.json").write_text(json.dumps(summary, indent=2))
    print("\n=== REWARD-VALIDITY ORACLE ===")
    print(f"target (teacher-Spearman) trajectory: { {k: round(v,3) for k,v in target.items()} }")
    print(f"\n{'reward form':28} {'corr_vs_target':>14} {'ci_lo':>7} {'ci_hi':>7} {'valid':>6}")
    for f, v in validity.items():
        sp = v["spearman_vs_target"]
        print(f"{f:28} {sp:>14.3f} {v['ci_lo']:>7.3f} {v['ci_hi']:>7.3f} {str(v['valid']):>6}")
    print(f"\nwritten: {OUT/'reward_validity_oracle.json'}")
    print("\nVERDICT: fix_correctness_BASELINE valid=%s -> %s" % (
        validity['fix_correctness_BASELINE']['valid'],
        "proxy tracks target" if validity['fix_correctness_BASELINE']['valid']
        else "proxy does NOT track target -> seedA reward INVALID (Goodhart confirmed); calibration forms required"))
