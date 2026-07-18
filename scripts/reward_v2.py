#!/usr/bin/env python3
"""Reward-v2 — grounded, stream-separated judge reward (Phase B1 of the RL redo).

Design (07-02 analysis + 08.1 lessons):
  1. GROUNDED DEFECT TERM: per-dimension agreement with the relabel-v1 GT
     (median dims from the 2026-07-03 campaign), not just overall proximity.
     A policy can Goodhart a scalar overall; matching 8 independent dims that
     were each anchored to a frozen rubric is much harder to fake.
  2. STREAM SEPARATION (MO-GRPO): components are returned SEPARATELY —
     calib / defect / format. At train time each stream is z-normalized within
     the GRPO group independently and then combined, so no stream's variance
     can drown the others (the 70/30-composite failure mode).
  3. ANTI-HACK is a train-time trip-wire (perturbation margin), not a reward
     term here: offline replay cannot generate new policy outputs.

Offline-replayable by construction: score(response_text, gt) uses only the
capture text + GT files, so B2's oracle gate can run over historical captures.

GT files (built by build_gt() below from output/relabel/results/*):
  data/relabel_v1/gt_dims_train.json  {"train:RIDX": {dims..., overall, verdict}}
  data/relabel_v1/gt_dims_val.json    {"val:RIDX":   {dims..., overall, verdict}}
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rubric dim name (relabel GT) <-> parser D-code (eval/dim_map.json)
RUBRIC2D = {
    "wpcs_compliance": "D1_wpcs",
    "security": "D2_security",
    "sql_safety": "D3_sql",
    "performance": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "i18n": "D6_i18n",
    "accessibility": "D7_a11y",
    "error_handling": "D8_errors",
    "code_quality": "D9_structure",
}
DIM_KEYS = set(RUBRIC2D)

GT_TRAIN = "data/relabel_v1/gt_dims_train.json"
GT_VAL = "data/relabel_v1/gt_dims_val.json"


def build_gt():
    """Aggregate per-dim medians + majority verdict from all relabel passes."""
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob("output/relabel/results/*.json")):
        try:
            rows = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for e in rows:
            iid = e.get("id", "")
            if iid.startswith("SENTINEL::") or ":" not in iid:
                continue
            for k, v in (e.get("judge") or {}).items():
                if k in DIM_KEYS and isinstance(v, (int, float)):
                    acc[iid][k].append(float(v))
                elif k == "overall_score" and isinstance(v, (int, float)):
                    acc[iid]["_overall"].append(float(v))
                elif k == "verdict" and v in ("PASS", "FAIL"):
                    acc[iid]["_verdict"].append(v)
    out = {"train": {}, "val": {}}
    for iid, d in acc.items():
        side = "train" if iid.startswith("train:") else "val"
        entry = {k: median(vs) for k, vs in d.items() if k in DIM_KEYS}
        if d.get("_overall"):
            entry["overall"] = median(d["_overall"])
        if d.get("_verdict"):
            entry["verdict"] = Counter(d["_verdict"]).most_common(1)[0][0]
        out[side][iid] = entry
    os.makedirs("data/relabel_v1", exist_ok=True)
    json.dump(out["train"], open(GT_TRAIN, "w"), indent=1)
    json.dump(out["val"], open(GT_VAL, "w"), indent=1)
    print(f"gt_dims: train={len(out['train'])} val={len(out['val'])}")
    return out


def load_gt(side="val"):
    return json.load(open(GT_VAL if side == "val" else GT_TRAIN))


def score(response_text: str, gt: dict) -> dict:
    """Reward-v2 streams for one response vs one GT entry. Each in [0,1].

    calib  : 1 - |pred_overall - gt_overall| / 100
    defect : mean over dims PRESENT IN BOTH of (1 - |pred - gt|/10),
             plus verdict agreement folded in as one extra pseudo-dim.
             Grounded: dims the GT omits (N/A) are never scored.
    format : 1 if parseable with dims + [/REASONING] present, else partial/0.
    """
    from eval.output_parsers import parse_judge_scores  # noqa: PLC0415

    parsed = parse_judge_scores(response_text, "auto")
    has_close = "[/REASONING]" in response_text
    if not parsed or not parsed.get("dimension_scores"):
        return {"calib": 0.0, "defect": 0.0,
                "format": 0.3 if has_close else 0.0, "parseable": False}

    fmt = 1.0 if has_close else 0.6

    # calib stream
    calib = 0.0
    pred_overall = parsed.get("overall")
    if pred_overall is not None and "overall" in gt:
        calib = max(0.0, 1.0 - abs(float(pred_overall) - gt["overall"]) / 100.0)

    # defect stream: per-dim agreement on the D-code intersection
    pred_dims = parsed["dimension_scores"]  # D-coded, 0-10
    agrees = []
    for rname, dcode in RUBRIC2D.items():
        if rname in gt and dcode in pred_dims:
            agrees.append(max(0.0, 1.0 - abs(float(pred_dims[dcode]) - gt[rname]) / 10.0))
    if "verdict" in gt and pred_overall is not None:
        pred_verdict = "PASS" if float(pred_overall) >= 65 else "FAIL"
        agrees.append(1.0 if pred_verdict == gt["verdict"] else 0.0)
    defect = sum(agrees) / len(agrees) if agrees else 0.0

    return {"calib": calib, "defect": defect, "format": fmt, "parseable": True,
            "n_dims_scored": len(agrees)}


def combine(streams: dict, w_calib=0.4, w_defect=0.5, w_format=0.1) -> float:
    """Blended scalar for OFFLINE analysis only. At GRPO time do NOT use this:
    z-normalize each stream within the group first (MO-GRPO), then sum —
    see rl_train integration notes in 08.1."""
    return (w_calib * streams["calib"] + w_defect * streams["defect"]
            + w_format * streams["format"])


if __name__ == "__main__":
    build_gt()
    # smoke: score one v1.3 response against val GT
    gt = load_gt("val")
    rows = [json.loads(l) for l in open("output/relabel/eval_s1_ep3/judge_responses.jsonl")
            if '"index"' in l]
    val_rows = [json.loads(l) for l in open("data/reasoning_dataset/openai_val.jsonl") if l.strip()]
    wj = [i for i, r in enumerate(val_rows)
          if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    r0 = rows[0]
    s = score(r0["response"], gt[f"val:{wj[r0['index']]}"])
    print("smoke stream scores:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    print("blended (offline):", round(combine(s), 4))
