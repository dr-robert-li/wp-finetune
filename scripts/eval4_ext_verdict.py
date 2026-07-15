#!/usr/bin/env python3
"""Phase 23-02 extension: apply the pre-registered UNEQUIVOCAL WIN rule
(output/eval4/ext_q8_preregistration.md) to v4's Q8 GGUF 3-seed ensemble vs
v3's shipped Q8 ensemble (0.8056, output/packaging/pkg03_ens8192_results.json).

Mirrors eval_relabel_ensemble.py's parse_capture()/index-join logic verbatim
(inlined -- that script executes its whole body at import time, so it cannot
be imported) so v3 and v4 per-item scores are derived identically. Adds the
paired item-resample bootstrap of the rho delta.

Usage:
  python3 scripts/eval4_ext_verdict.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")
import numpy as np
from scipy.stats import spearmanr

from eval.eval_judge import _derive_prose_overall
from eval.output_parsers import load_dim_map, parse_judge_scores

_DM = load_dim_map()
_DW = {k: v for k, v in _DM["dimension_weights"].items() if not k.startswith("_")}
_ROWS = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
_WJ_ROWS = [i for i, r in enumerate(_ROWS)
            if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]


def parse_capture(path):
    """Verbatim mirror of scripts/relabel/eval_relabel_ensemble.py:parse_capture."""
    d = {}
    pf = 0
    for line in open(path):
        r = json.loads(line)
        if "index" not in r:
            continue
        parsed = parse_judge_scores(r["response"], "auto")
        if not parsed or not parsed.get("dimension_scores"):
            pf += 1
            continue
        o = float(parsed["overall"]) if "overall" in parsed else _derive_prose_overall(parsed["dimension_scores"], _DW)
        d[f"val:{_WJ_ROWS[r['index']]}"] = o
    return d, pf

V3_CAPS = [f"output/packaging/ens8192/q8_s{i}/judge_responses.jsonl" for i in range(3)]
V4_CAPS = [f"output/eval4/ext_q8/q8_s{i}/judge_responses.jsonl" for i in range(3)]
LABELS_PATH = "output/relabel/val_labels_v1.json"
V3_POINT = 0.8056  # output/packaging/pkg03_ens8192_results.json ensemble.q8.rho
V3_RECEIPT = "output/packaging/pkg03_ens8192_results.json"
OUT_PATH = "output/eval4/ext_q8_results.json"
N_RESAMPLES = 10_000
SEED = 1337


def median_ensemble(caps, min_seeds=2):
    per_seed, pf = [], []
    for c in caps:
        d, f = parse_capture(c)
        per_seed.append(d)
        pf.append(f)
    labels = {k: v for k, v in json.load(open(LABELS_PATH)).items() if k.startswith("val:")}
    ens = {}
    for k in labels:
        vals = [d[k] for d in per_seed if k in d]
        if len(vals) >= min_seeds:
            ens[k] = float(np.median(vals))
    return ens, pf, labels


def llama_cpp_version():
    r = subprocess.run(
        [os.path.expanduser("~/llama.cpp/build/bin/llama-cli"), "--version"],
        capture_output=True, text=True,
    )
    out = (r.stdout + r.stderr).strip().splitlines()
    return out[0] if out else "unknown"


def paired_bootstrap(v3_ens, v4_ens, labels, n=N_RESAMPLES, seed=SEED):
    common = sorted(set(v3_ens) & set(v4_ens) & set(labels))
    rng = np.random.default_rng(seed)
    n_items = len(common)
    v3_arr = np.array([v3_ens[k] for k in common])
    v4_arr = np.array([v4_ens[k] for k in common])
    lab_arr = np.array([labels[k] for k in common])
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, n_items)
        r3 = spearmanr(v3_arr[idx], lab_arr[idx]).statistic
        r4 = spearmanr(v4_arr[idx], lab_arr[idx]).statistic
        deltas[i] = r4 - r3
    deltas_sorted = np.sort(deltas)
    ci_lower = float(deltas_sorted[int(0.025 * n)])
    ci_upper = float(deltas_sorted[int(0.975 * n)])
    return {
        "n_common_items": n_items,
        "n_resamples": n,
        "seed": seed,
        "mean_delta": float(deltas.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_gt_0": ci_lower > 0,
    }


def main():
    v4_single = {}
    for i, cap in enumerate(V4_CAPS):
        d, pf = parse_capture(cap)
        labels_full = {k: v for k, v in json.load(open(LABELS_PATH)).items() if k.startswith("val:")}
        common = sorted(set(d) & set(labels_full))
        rho = spearmanr([d[k] for k in common], [labels_full[k] for k in common]).statistic
        v4_single[f"s{i}"] = {"rho": float(rho), "n": len(common), "parse_fail": pf}

    v3_ens, v3_pf, labels = median_ensemble(V3_CAPS)
    v4_ens, v4_pf, _ = median_ensemble(V4_CAPS)

    common_v3 = sorted(set(v3_ens) & set(labels))
    v3_point = float(spearmanr([v3_ens[k] for k in common_v3], [labels[k] for k in common_v3]).statistic)

    common_v4 = sorted(set(v4_ens) & set(labels))
    v4_point = float(spearmanr([v4_ens[k] for k in common_v4], [labels[k] for k in common_v4]).statistic)

    # v4 ensemble's own unpaired bootstrap CI (fallback rule ingredient), same convention as eval_relabel_ensemble.py
    rng = np.random.default_rng(7)
    v4_arr = np.array([v4_ens[k] for k in common_v4])
    lab_arr = np.array([labels[k] for k in common_v4])
    n_v4 = len(common_v4)
    boots = sorted(spearmanr(v4_arr[idx], lab_arr[idx]).statistic
                   for idx in (rng.integers(0, n_v4, n_v4) for _ in range(2000)))
    v4_ci = [float(boots[50]), float(boots[1949])]

    item_mismatch = set(v3_ens) != set(v4_ens)
    rule_fired = "fallback_ci_lower_vs_v3_point" if item_mismatch else "paired_bootstrap"

    pb = None
    win_a = v4_point > V3_POINT
    if rule_fired == "paired_bootstrap":
        pb = paired_bootstrap(v3_ens, v4_ens, labels)
        win_b = pb["ci_lower_gt_0"]
        unequivocal_win = bool(win_a and win_b)
    else:
        win_b = v4_ci[0] > V3_POINT
        unequivocal_win = bool(win_b)

    result = {
        "requirement": "Phase-23-02-EXTENSION",
        "title": "v4 judge on shipped Q8 GGUF llama.cpp stack vs v3's shipped 0.8056",
        "preregistration": "output/eval4/ext_q8_preregistration.md",
        "llama_cpp_version": llama_cpp_version(),
        "config": {
            "max_tokens": 8192,
            "engine": "llama.cpp llama-server, CUDA (GB10), -ngl 999 --jinja, --parallel 4",
            "val_set": "data/reasoning_dataset/openai_val.jsonl (121 wp_judge items)",
            "labels": LABELS_PATH,
            "ensemble_rule": "per-item median of the 3 seed overalls (>=2 seeds required)",
        },
        "v4_single_seed": v4_single,
        "v4_ensemble": {
            "rho": v4_point, "ci": v4_ci, "n": n_v4,
            "parse_fail_per_seed": v4_pf,
            "source": V4_CAPS,
        },
        "v3_ensemble_recomputed": {
            "rho": v3_point, "n": len(common_v3),
            "parse_fail_per_seed": v3_pf,
            "source": V3_CAPS,
            "note": "recomputed here from the same v3 raw captures for the paired join; "
                    f"original shipped receipt point estimate = {V3_POINT} ({V3_RECEIPT})",
        },
        "v3_shipped_point": V3_POINT,
        "v3_shipped_receipt": V3_RECEIPT,
        "item_id_mismatch": item_mismatch,
        "rule_fired": rule_fired,
        "paired_bootstrap": pb,
        "criteria": {
            "a_point_gt_v3": win_a,
            "b_ci_lower_gt_0_or_fallback": win_b,
        },
        "unequivocal_win": unequivocal_win,
        "secondary_reads": {
            "v4_bf16_vllm_served_s1": 0.7872,
            "v4_bf16_vllm_capture_s1": 0.8358,
            "v4_bf16_vllm_capture_ensemble": 0.8160,
            "v3_bf16_llama_cpp_ensemble": 0.8100,
            "note": "does llama.cpp lift v4's numerics ceiling the way it matched vLLM for v3 (0.8100 vs 0.8075)?",
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(result, open(OUT_PATH, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH}")
    print(f"\nUNEQUIVOCAL_WIN = {unequivocal_win}  (rule_fired={rule_fired})")


if __name__ == "__main__":
    main()
