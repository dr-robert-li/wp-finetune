"""Diff v1 vs v2 calibration models.

Compares the pre-pivot v1 model (saved as `*.json.v1.bak`) against the
post-pivot v2 model (current `*.json`) trained on the schema-normalized
calibration dataset (FAIL N grew from 47 to 100 train, 19 to 45 holdout).

Outputs:
  data/calibration/diff_v1_v2.md   markdown report
  data/calibration/diff_v1_v2.json structured diff

Compares:
  1. Gate metrics on the v2 holdout (v1 model scored on extended holdout vs v2 model)
  2. Top-N feature importance shifts (which features moved up/down/in/out)
  3. Per-row verdict + score prediction agreement v1 vs v2 on the v2 holdout
  4. Sign-flip / large-shift flags

Run after `scripts/calibrate_rubric.py` finishes the v2 fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_rubric import build_feature_matrix, load_jsonl, DIMS  # noqa: E402

V1_CLF = ROOT / "models/calibration/verdict_classifier.json.v1.bak"
V1_REG = ROOT / "models/calibration/overall_regressor.json.v1.bak"
V2_CLF = ROOT / "models/calibration/verdict_classifier.json"
V2_REG = ROOT / "models/calibration/overall_regressor.json"
HOLDOUT = ROOT / "data/calibration/holdout.jsonl"

OUT_MD = ROOT / "data/calibration/diff_v1_v2.md"
OUT_JSON = ROOT / "data/calibration/diff_v1_v2.json"
TOP_N = 20


def load_models() -> tuple[xgb.XGBClassifier, xgb.XGBClassifier, xgb.XGBRegressor, xgb.XGBRegressor]:
    v1c = xgb.XGBClassifier(); v1c.load_model(str(V1_CLF))
    v2c = xgb.XGBClassifier(); v2c.load_model(str(V2_CLF))
    v1r = xgb.XGBRegressor(); v1r.load_model(str(V1_REG))
    v2r = xgb.XGBRegressor(); v2r.load_model(str(V2_REG))
    return v1c, v2c, v1r, v2r


def metrics_clf(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else None,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }
    return out


def metrics_reg(model, X, y) -> dict:
    pred = model.predict(X)
    sp = spearmanr(y, pred)
    pr = pearsonr(y, pred)
    return {
        "spearman": float(sp.statistic if hasattr(sp, "statistic") else sp[0]),
        "pearson": float(pr.statistic if hasattr(pr, "statistic") else pr[0]),
        "mae": float(np.mean(np.abs(y - pred))),
        "n": int(len(y)),
    }


def importance_dict(model, feature_names: list[str]) -> dict[str, float]:
    """Map feature_name -> importance (gain). Zeros omitted."""
    booster = model.get_booster()
    booster.feature_names = feature_names
    raw = booster.get_score(importance_type="gain")
    return raw  # only includes nonzero


def diff_importances(v1: dict[str, float], v2: dict[str, float]) -> dict:
    keys = sorted(set(v1) | set(v2))
    rows = []
    for k in keys:
        a = v1.get(k, 0.0)
        b = v2.get(k, 0.0)
        rows.append({"feature": k, "v1_importance": a, "v2_importance": b,
                     "delta": b - a, "abs_delta": abs(b - a),
                     "appeared_in_v2": a == 0 and b > 0,
                     "vanished_in_v2": a > 0 and b == 0})
    rows.sort(key=lambda r: r["abs_delta"], reverse=True)
    top_v1 = sorted(v1.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]
    top_v2 = sorted(v2.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]
    return {"top_v1": top_v1, "top_v2": top_v2, "biggest_shifts": rows[:TOP_N],
            "n_appeared": sum(1 for r in rows if r["appeared_in_v2"]),
            "n_vanished": sum(1 for r in rows if r["vanished_in_v2"])}


def per_row_agreement(v1c, v2c, v1r, v2r, X, rows) -> dict:
    p1c = (v1c.predict_proba(X)[:, 1] >= 0.5).astype(int)
    p2c = (v2c.predict_proba(X)[:, 1] >= 0.5).astype(int)
    p1r = v1r.predict(X)
    p2r = v2r.predict(X)
    agree_verdict = int((p1c == p2c).sum())
    disagreements = []
    for i, r in enumerate(rows):
        if p1c[i] != p2c[i] or abs(p1r[i] - p2r[i]) > 10.0:
            disagreements.append({
                "row_id": r.get("row_id"), "source": r.get("source"),
                "subtlety": r.get("subtlety"), "gt_verdict": r.get("gt_verdict"),
                "gt_overall": r.get("gt_overall"),
                "v1_verdict_pred": "PASS" if p1c[i] == 0 else "FAIL",
                "v2_verdict_pred": "PASS" if p2c[i] == 0 else "FAIL",
                "v1_overall_pred": float(p1r[i]),
                "v2_overall_pred": float(p2r[i]),
            })
    return {
        "n_rows": int(len(X)),
        "verdict_agreement": agree_verdict / len(X),
        "verdict_agree_count": agree_verdict,
        "score_mean_abs_diff": float(np.mean(np.abs(p1r - p2r))),
        "score_max_abs_diff": float(np.max(np.abs(p1r - p2r))),
        "disagreement_rows": disagreements[:30],
        "n_disagreements_total": len(disagreements),
    }


def main() -> None:
    holdout = load_jsonl(HOLDOUT)
    X, feat_names = build_feature_matrix(holdout)
    y_verdict = np.array([1 if r["gt_verdict"] == "FAIL" else 0 for r in holdout], dtype=int)
    y_overall = np.array([float(r["gt_overall"]) for r in holdout], dtype=float)

    v1c, v2c, v1r, v2r = load_models()

    v1_clf_m = metrics_clf(v1c, X, y_verdict)
    v2_clf_m = metrics_clf(v2c, X, y_verdict)
    v1_reg_m = metrics_reg(v1r, X, y_overall)
    v2_reg_m = metrics_reg(v2r, X, y_overall)

    imp_clf = diff_importances(importance_dict(v1c, feat_names),
                                importance_dict(v2c, feat_names))
    imp_reg = diff_importances(importance_dict(v1r, feat_names),
                                importance_dict(v2r, feat_names))

    agree = per_row_agreement(v1c, v2c, v1r, v2r, X, holdout)

    diff = {
        "holdout_path": str(HOLDOUT.relative_to(ROOT)),
        "holdout_n": len(holdout),
        "verdict_classifier": {
            "v1_on_v2_holdout": v1_clf_m,
            "v2_on_v2_holdout": v2_clf_m,
            "delta_accuracy": v2_clf_m["accuracy"] - v1_clf_m["accuracy"],
            "importance_diff": imp_clf,
        },
        "overall_regressor": {
            "v1_on_v2_holdout": v1_reg_m,
            "v2_on_v2_holdout": v2_reg_m,
            "delta_spearman": v2_reg_m["spearman"] - v1_reg_m["spearman"],
            "delta_mae": v2_reg_m["mae"] - v1_reg_m["mae"],
            "importance_diff": imp_reg,
        },
        "per_row_agreement": agree,
    }

    OUT_JSON.write_text(json.dumps(diff, indent=2))

    # Markdown
    md = []
    md.append("# v1 vs v2 calibration model diff")
    md.append("")
    md.append("v1 = pre-pivot model (trained on 527-row dataset, FAIL N=47).  ")
    md.append("v2 = post-schema-normalization model (trained on 580-row dataset, FAIL N=100).  ")
    md.append(f"Both scored against the v2 holdout ({len(holdout)} rows; v1's original holdout was 39 rows).")
    md.append("")
    md.append("## 1. Holdout gate metrics")
    md.append("")
    md.append("| Head | Metric | v1 | v2 | Δ | v2 gate |")
    md.append("|---|---|---:|---:|---:|:---:|")
    da = diff["verdict_classifier"]["delta_accuracy"]
    da_arrow = "↑" if da > 0 else "↓" if da < 0 else "·"
    md.append(f"| Verdict | accuracy | {v1_clf_m['accuracy']:.4f} | {v2_clf_m['accuracy']:.4f} | {da_arrow} {da:+.4f} | ≥ 0.85 → {'PASS' if v2_clf_m['accuracy']>=0.85 else 'FAIL'} |")
    if v1_clf_m["auc"] is not None:
        md.append(f"| Verdict | AUC | {v1_clf_m['auc']:.4f} | {v2_clf_m['auc']:.4f} | {v2_clf_m['auc']-v1_clf_m['auc']:+.4f} | — |")
    ds = diff["overall_regressor"]["delta_spearman"]
    ds_arrow = "↑" if ds > 0 else "↓" if ds < 0 else "·"
    dp = v2_reg_m["pearson"] - v1_reg_m["pearson"]
    dp_arrow = "↑" if dp > 0 else "↓" if dp < 0 else "·"
    md.append(f"| Overall | pearson | {v1_reg_m['pearson']:.4f} | {v2_reg_m['pearson']:.4f} | {dp_arrow} {dp:+.4f} | ≥ 0.75 → {'PASS' if v2_reg_m['pearson']>=0.75 else 'FAIL'} |")
    md.append(f"| Overall | spearman | {v1_reg_m['spearman']:.4f} | {v2_reg_m['spearman']:.4f} | {ds_arrow} {ds:+.4f} | (informational) |")
    md.append(f"| Overall | MAE | {v1_reg_m['mae']:.4f} | {v2_reg_m['mae']:.4f} | {v2_reg_m['mae']-v1_reg_m['mae']:+.4f} | — |")
    md.append("")

    md.append("## 2. Per-row prediction agreement (v1 vs v2 on v2 holdout)")
    md.append("")
    md.append(f"- Verdict agreement: {agree['verdict_agreement']*100:.1f}% ({agree['verdict_agree_count']}/{agree['n_rows']})")
    md.append(f"- Overall score mean |Δ|: {agree['score_mean_abs_diff']:.2f}")
    md.append(f"- Overall score max |Δ|: {agree['score_max_abs_diff']:.2f}")
    md.append(f"- Total disagreement rows (verdict mismatch OR |score Δ| > 10): {agree['n_disagreements_total']}")
    md.append("")
    if agree["disagreement_rows"]:
        md.append("### Disagreement sample (up to 30)")
        md.append("")
        md.append("| row_id | source | subtlety | gt_verdict | gt_overall | v1 verdict | v2 verdict | v1 score | v2 score |")
        md.append("|---|---|---|---|---:|---|---|---:|---:|")
        for r in agree["disagreement_rows"]:
            md.append(f"| {r['row_id']} | {r['source']} | {r['subtlety']} | {r['gt_verdict']} | {r['gt_overall']:.1f} | "
                     f"{r['v1_verdict_pred']} | {r['v2_verdict_pred']} | {r['v1_overall_pred']:.1f} | {r['v2_overall_pred']:.1f} |")
        md.append("")

    for label, idiff in [("Verdict classifier", imp_clf), ("Overall regressor", imp_reg)]:
        md.append(f"## 3. {label} feature importance")
        md.append("")
        md.append(f"- Features that appeared in v2 (zero in v1, nonzero in v2): {idiff['n_appeared']}")
        md.append(f"- Features that vanished in v2 (nonzero in v1, zero in v2): {idiff['n_vanished']}")
        md.append("")
        md.append(f"### Top {TOP_N} features by absolute shift")
        md.append("")
        md.append("| feature | v1 imp | v2 imp | Δ | flag |")
        md.append("|---|---:|---:|---:|---|")
        for r in idiff["biggest_shifts"]:
            flag = "appeared" if r["appeared_in_v2"] else ("vanished" if r["vanished_in_v2"] else "")
            md.append(f"| {r['feature']} | {r['v1_importance']:.4f} | {r['v2_importance']:.4f} | {r['delta']:+.4f} | {flag} |")
        md.append("")
        md.append(f"### Top {TOP_N} v1")
        md.append("")
        for f, imp in idiff["top_v1"]:
            md.append(f"- `{f}`  {imp:.4f}")
        md.append("")
        md.append(f"### Top {TOP_N} v2")
        md.append("")
        for f, imp in idiff["top_v2"]:
            md.append(f"- `{f}`  {imp:.4f}")
        md.append("")

    md.append("## 4. Verdict")
    md.append("")
    v2_clf_pass = v2_clf_m["accuracy"] >= 0.85
    v2_reg_pass = v2_reg_m["pearson"] >= 0.75
    if v2_clf_pass and v2_reg_pass:
        md.append("**v2 gates PASS.** Safe to ship as the calibration model for Phase 1c.")
    else:
        md.append("**v2 gates FAIL.** Investigate before replacing v1.")
    md.append("")
    md.append(f"v1→v2 agreement on extended holdout: {agree['verdict_agreement']*100:.1f}% verdict, mean |Δ| score {agree['score_mean_abs_diff']:.2f}.  ")
    md.append("- ≥ 95% verdict agreement: v1 stands; v2 just tightens. Phase 1b pilot results valid under either.")
    md.append("- 85–95%: flag disagreement set above for human review before treating Phase 1b as final.")
    md.append("- < 85%: v1 results retroactively suspect; consider re-running Phase 1b under v2.")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"\nSummary:")
    print(f"  v1 holdout acc: {v1_clf_m['accuracy']:.4f}  → v2: {v2_clf_m['accuracy']:.4f}  (Δ {da:+.4f})")
    print(f"  v1 holdout spearman: {v1_reg_m['spearman']:.4f}  → v2: {v2_reg_m['spearman']:.4f}  (Δ {ds:+.4f})")
    print(f"  v1↔v2 verdict agreement: {agree['verdict_agreement']*100:.1f}%  ({agree['n_disagreements_total']} disagreement rows)")


if __name__ == "__main__":
    main()
