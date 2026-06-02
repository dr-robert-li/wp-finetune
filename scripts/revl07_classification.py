"""REVL-07 — classification confusion matrix (SOFT/informational).

Treats the judge task as binary PASS/FAIL at swept score thresholds: a prediction is
PASS iff model_overall >= threshold; ground truth is PASS iff gt_canonical >= threshold
(gt_canonical = rubric_calibrated_overall, the REVL-01A canonical GT — see dim_map.json).
Both are on the 0-100 scale. For each threshold we compute TP/TN/FP/FN + precision/
recall/F1/accuracy, and report the F1-optimal threshold (consumed by REVL-05 stratified
sampling). SOFT gate: always records, never blocks the merge.

Reads output/eval_reasoning/reasoning_merged/eval_judge_results.pairs.jsonl (per-example
model_overall + gt_canonical). Rows missing either value are excluded + counted.

Usage:
  python -m scripts.revl07_classification \
      --pairs output/eval_reasoning/reasoning_merged/eval_judge_results.pairs.jsonl \
      --out output/04.4_classification_matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = "output/eval_reasoning/reasoning_merged/eval_judge_results.pairs.jsonl"
DEFAULT_OUT = "output/04.4_classification_matrix.json"
# Standard WP-judge PASS line is 7.0/10 == 70/100; sweep around it.
DEFAULT_THRESHOLDS = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]


def _num(x):
    return x if isinstance(x, (int, float)) else None


def confusion_at(pairs, threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for m, g in pairs:
        pred = m >= threshold
        truth = g >= threshold
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {"threshold": threshold, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def revl07(pairs_jsonl: str, out_path: str, thresholds=None) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    pp = PROJECT_ROOT / pairs_jsonl if not os.path.isabs(pairs_jsonl) else Path(pairs_jsonl)
    rows = [json.loads(l) for l in open(pp) if l.strip()]
    usable, excluded = [], 0
    for r in rows:
        m, g = _num(r.get("model_overall")), _num(r.get("gt_canonical"))
        if m is None or g is None:
            excluded += 1
            continue
        usable.append((m, g))
    matrices = [confusion_at(usable, t) for t in thresholds]
    best = max(matrices, key=lambda d: d["f1"]) if matrices else None
    result = {
        "gate": "REVL-07",
        "gate_class": "soft",
        "n_total": len(rows),
        "n_usable": len(usable),
        "n_excluded": excluded,
        "gt_source": "gt_canonical (rubric_calibrated_overall)",
        "thresholds": matrices,
        "f1_optimal_threshold": best["threshold"] if best else None,
        "f1_optimal": best["f1"] if best else None,
    }
    op = PROJECT_ROOT / out_path if not os.path.isabs(out_path) else Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-07 classification confusion matrix (SOFT)")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    res = revl07(args.pairs, args.out)
    print(f"[revl07] n_usable={res['n_usable']} excluded={res['n_excluded']} "
          f"F1-opt threshold={res['f1_optimal_threshold']} F1={res['f1_optimal']:.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
