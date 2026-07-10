#!/usr/bin/env python
"""RLEV-01 teacher-Spearman scorer (offline, from captured responses).

The calibrated_canonical path excludes every row when rubric calibrated_overall is
unavailable (this env), which collaterally zeroes the teacher pairing too. This scorer
computes the TEACHER-GT Spearman directly — the baseline-comparable SOFT metric (the v4
baseline's 0.1534) — reusing the SAME parsers as eval_judge's online path, without the
canonical gate. Fixed teacher GT across checkpoints → clean cross-checkpoint comparison.

Run (any venv with scipy): .venv-tinker/bin/python scripts/_rlev01_score.py
"""
import json, sys, importlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eval.output_parsers import parse_judge_scores, load_dim_map
from eval.eval_judge import _extract_gt_from_assistant, _derive_prose_overall
from scipy.stats import spearmanr

VAL = REPO / "data/reasoning_dataset/openai_val.jsonl"
CKPTS = ["warmstart", "step-50", "step-100", "step-150", "step-200", "step-250",
         "step-300", "step-350", "step-400", "step-450", "step-500"]

dm = load_dim_map()
weights = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}

rows = [json.loads(l) for l in VAL.open() if l.strip()]
examples = [r for r in rows
            if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]

# Teacher GT per filtered-index (fixed across all checkpoints)
teacher = {}
for i, ex in enumerate(examples):
    t = _extract_gt_from_assistant(ex["messages"])
    if t is not None:
        teacher[i] = float(t["overall"])
print(f"teacher GT available: {len(teacher)}/{len(examples)} examples", flush=True)


def model_overall_map(name):
    f = REPO / f"output/rl_eval/{name}/judge_responses.jsonl"
    out = {}
    if not f.exists():
        return out
    for l in f.open():
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if "index" not in r:   # skip provenance header line
            continue
        p = parse_judge_scores(r["response"], "auto")
        if not p or not p.get("dimension_scores"):
            continue
        if "overall" in p:
            mo = float(p["overall"])
        else:
            mo = _derive_prose_overall(p["dimension_scores"], weights)
            if mo is None:
                continue
        out[r["index"]] = mo
    return out


mom = {n: model_overall_map(n) for n in CKPTS}
present = [n for n in CKPTS if mom[n]]

per_ckpt = {}
for n in CKPTS:
    idx = [i for i in mom[n] if i in teacher]
    if len(idx) >= 2:
        sp = spearmanr([mom[n][i] for i in idx], [teacher[i] for i in idx]).statistic
        per_ckpt[n] = {"spearman": (None if sp != sp else float(sp)), "n": len(idx)}
    else:
        per_ckpt[n] = {"spearman": None, "n": len(idx)}
    print(f"  {n}: spearman={per_ckpt[n]['spearman']} n={per_ckpt[n]['n']}", flush=True)

# Common aligned indices across all captured checkpoints + teacher
common = set(teacher)
for n in present:
    common &= set(mom[n])
common = sorted(common)
print(f"common aligned indices across {present}: n={len(common)}", flush=True)

aligned = {}
if common:
    gt = [teacher[i] for i in common]
    for n in present:
        aligned[n] = float(spearmanr([mom[n][i] for i in common], gt).statistic)

boot = {}
if "warmstart" in present and len(common) >= 5:
    bg = importlib.import_module("scripts.bootstrap_gate")
    gt = [teacher[i] for i in common]
    base = [mom["warmstart"][i] for i in common]
    for n in present:
        if n == "warmstart":
            continue
        cand = [mom[n][i] for i in common]
        try:
            boot[n] = bg.bootstrap_spearman_improvement(cand, gt, base, n_boot=2000)
        except Exception as e:  # noqa: BLE001
            boot[n] = {"error": repr(e)}

summary = {
    "metric": "teacher_overall_spearman (SOFT, baseline-comparable)",
    "per_ckpt": per_ckpt,
    "n_common_aligned": len(common),
    "aligned_spearman_common_set": aligned,
    "bootstrap_vs_warmstart": boot,
    "captured": present,
}
(REPO / "output/rl_eval/rlev01_teacher_summary.json").write_text(json.dumps(summary, indent=2))
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
