"""Phase 1a step 6-8: XGBoost dual-head calibration of rubric scorer.

Reads:
  data/calibration/train.jsonl
  data/calibration/holdout.jsonl

Fits two heads with 5-fold inner CV grid search:
  - Verdict classifier (XGBClassifier): target gt_verdict (PASS|FAIL)
  - Overall regressor   (XGBRegressor):  target gt_overall (0-100)

Features:
  - 241 binary indicators (triggered_check_id ∈ CHECK_REGISTRY)
  -   9 per-dim raw scores (None -> NaN; XGBoost handles missing natively)

Gates on holdout (post-pivot: 65 rows = 20 PASS anchors + 45 boundary FAIL):
  - verdict accuracy >= 0.85
  - overall  Pearson  >= 0.75   (was Spearman >= 0.70 pre-pivot; rationale in
                                  data/calibration/audit/council_transcripts.md)

Persists:
  models/calibration/verdict_classifier.json
  models/calibration/overall_regressor.json
  config/rubric_calibration.yaml
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xgboost as xgb

from eval.rubric_definitions import CHECK_REGISTRY

DIMS = [
    "D1_wpcs", "D2_security", "D3_sql", "D4_perf", "D5_wp_api",
    "D6_i18n", "D7_a11y", "D8_errors", "D9_structure",
]
TRAIN_PATH = ROOT / "data/calibration/train.jsonl"
HOLDOUT_PATH = ROOT / "data/calibration/holdout.jsonl"
MODELS_DIR = ROOT / "models/calibration"
CONFIG_PATH = ROOT / "config/rubric_calibration.yaml"

CLASSIFIER_GATE = 0.85
REGRESSOR_GATE_PEARSON = 0.75  # Pearson on holdout; rationale in data/calibration/audit/council_transcripts.md

HP_GRID_CLF = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_child_weight": mcw}
    for d in (3, 4, 5)
    for n in (100, 200, 400)
    for lr in (0.05, 0.1)
    for mcw in (5, 10)
]
HP_GRID_REG = list(HP_GRID_CLF)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Return (X[n_rows, n_features], feature_names)."""
    check_ids = sorted(CHECK_REGISTRY.keys())
    feature_names = check_ids + [f"score::{d}" for d in DIMS]
    n_feat = len(feature_names)
    X = np.full((len(rows), n_feat), np.nan, dtype=np.float64)
    cid_idx = {cid: i for i, cid in enumerate(check_ids)}
    dim_off = len(check_ids)
    for row_i, r in enumerate(rows):
        # Binary check indicators: default 0, set 1 if triggered
        for i in range(len(check_ids)):
            X[row_i, i] = 0.0
        for cid in r.get("triggered_checks_flat", []):
            j = cid_idx.get(cid)
            if j is not None:
                X[row_i, j] = 1.0
        # Per-dim scores: None -> NaN, else float
        dim_scores = r.get("dim_scores") or {}
        for di, d in enumerate(DIMS):
            v = dim_scores.get(d)
            if isinstance(v, (int, float)):
                X[row_i, dim_off + di] = float(v)
            # else: leave NaN
    return X, feature_names


def cv_score_classifier(X: np.ndarray, y: np.ndarray, hp: dict, scale_pos_weight: float, n_splits: int = 5) -> float:
    """Return mean ROC-AUC over stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        m = xgb.XGBClassifier(
            **hp,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        try:
            aucs.append(roc_auc_score(y[va], p))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))


def cv_score_regressor(X: np.ndarray, y: np.ndarray, hp: dict, n_splits: int = 5) -> float:
    """Return mean Spearman over k-fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    sps = []
    for tr, va in kf.split(X):
        m = xgb.XGBRegressor(
            **hp,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        m.fit(X[tr], y[tr])
        p = m.predict(X[va])
        if len(set(y[va].tolist())) < 2 or len(set(p.tolist())) < 2:
            sps.append(0.0)
            continue
        sp = spearmanr(y[va], p)
        sps.append(float(sp.statistic) if hasattr(sp, "statistic") else float(sp[0]))
    return float(np.mean(sps))


def grid_search(score_fn, grid: list[dict], **kwargs) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Linear scan; returns (best_hp, best_score, all_scores)."""
    results = []
    best_hp, best_score = None, -1e9
    for i, hp in enumerate(grid):
        s = score_fn(hp=hp, **kwargs)
        results.append((hp, s))
        if s > best_score:
            best_score, best_hp = s, hp
        print(f"  [{i+1}/{len(grid)}] {hp}  score={s:+.4f}  (best={best_score:+.4f})")
    return best_hp, best_score, results


def hash_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:16]


def main():
    print(f"Loading train: {TRAIN_PATH.relative_to(ROOT)}")
    train_rows = load_jsonl(TRAIN_PATH)
    print(f"Loading holdout: {HOLDOUT_PATH.relative_to(ROOT)}")
    holdout_rows = load_jsonl(HOLDOUT_PATH)
    print(f"  train={len(train_rows)}  holdout={len(holdout_rows)}")

    X_train, feat_names = build_feature_matrix(train_rows)
    X_hold, _ = build_feature_matrix(holdout_rows)
    print(f"  feature dim: {X_train.shape[1]} ({len(CHECK_REGISTRY)} checks + {len(DIMS)} per-dim scores)")

    # Targets
    y_verdict_train = np.array([1 if r["gt_verdict"] == "FAIL" else 0 for r in train_rows], dtype=np.int32)
    y_verdict_hold = np.array([1 if r["gt_verdict"] == "FAIL" else 0 for r in holdout_rows], dtype=np.int32)
    y_overall_train = np.array([float(r["gt_overall"]) for r in train_rows], dtype=np.float64)
    y_overall_hold = np.array([float(r["gt_overall"]) for r in holdout_rows], dtype=np.float64)

    n_pos = int(y_verdict_train.sum())
    n_neg = int(len(y_verdict_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0
    print(f"  classifier: train PASS={n_neg} FAIL={n_pos}  scale_pos_weight={scale_pos_weight:.3f}")

    # ---- Verdict classifier grid search ----
    print("\n[1/2] Verdict classifier 5-fold CV grid search ...")
    t0 = time.time()
    best_clf_hp, best_clf_auc, _ = grid_search(
        cv_score_classifier, HP_GRID_CLF, X=X_train, y=y_verdict_train,
        scale_pos_weight=scale_pos_weight,
    )
    print(f"  Best CV AUC: {best_clf_auc:.4f}  HPs: {best_clf_hp}  ({time.time()-t0:.1f}s)")

    clf = xgb.XGBClassifier(
        **best_clf_hp,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    clf.fit(X_train, y_verdict_train)

    # ---- Overall regressor grid search ----
    print("\n[2/2] Overall regressor 5-fold CV grid search ...")
    t0 = time.time()
    best_reg_hp, best_reg_sp, _ = grid_search(
        cv_score_regressor, HP_GRID_REG, X=X_train, y=y_overall_train,
    )
    print(f"  Best CV Spearman: {best_reg_sp:.4f}  HPs: {best_reg_hp}  ({time.time()-t0:.1f}s)")

    reg = xgb.XGBRegressor(
        **best_reg_hp,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    reg.fit(X_train, y_overall_train)

    # ---- Holdout evaluation ----
    print("\nHoldout evaluation (45-row boundary):")
    p_verdict = clf.predict(X_hold)
    p_verdict_proba = clf.predict_proba(X_hold)[:, 1]
    clf_acc = float(accuracy_score(y_verdict_hold, p_verdict))
    try:
        clf_auc = float(roc_auc_score(y_verdict_hold, p_verdict_proba))
    except ValueError:
        clf_auc = float("nan")
    cm = confusion_matrix(y_verdict_hold, p_verdict).tolist()
    print(f"  Verdict acc={clf_acc:.4f}  AUC={clf_auc:.4f}  confusion={cm}")

    p_overall = reg.predict(X_hold)
    if len(set(y_overall_hold.tolist())) < 2:
        reg_sp = 0.0
        reg_pe = 0.0
    else:
        sp = spearmanr(y_overall_hold, p_overall)
        reg_sp = float(sp.statistic) if hasattr(sp, "statistic") else float(sp[0])
        pe = pearsonr(y_overall_hold, p_overall)
        reg_pe = float(pe.statistic) if hasattr(pe, "statistic") else float(pe[0])
    reg_mae = float(np.mean(np.abs(p_overall - y_overall_hold)))
    print(f"  Overall Spearman={reg_sp:+.4f}  Pearson={reg_pe:+.4f}  MAE={reg_mae:.2f}")

    # ---- Feature importances ----
    def top_k(model, k=20) -> list[dict]:
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1][:k]
        return [{"feature": feat_names[i], "importance": float(imps[i])} for i in order if imps[i] > 0]

    clf_imp = top_k(clf, 20)
    reg_imp = top_k(reg, 20)
    print("\nTop-20 classifier features:")
    for x in clf_imp:
        print(f"  {x['importance']:.4f}  {x['feature']}")
    print("\nTop-20 regressor features:")
    for x in reg_imp:
        print(f"  {x['importance']:.4f}  {x['feature']}")

    # ---- Gates ----
    clf_pass = clf_acc >= CLASSIFIER_GATE
    reg_pass = reg_pe >= REGRESSOR_GATE_PEARSON
    print("\nGates:")
    print(f"  classifier accuracy >= {CLASSIFIER_GATE}: {'PASS' if clf_pass else 'FAIL'} ({clf_acc:.4f})")
    print(f"  regressor Pearson   >= {REGRESSOR_GATE_PEARSON}: {'PASS' if reg_pass else 'FAIL'} ({reg_pe:+.4f})")

    # ---- Persist ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf_path = MODELS_DIR / "verdict_classifier.json"
    reg_path = MODELS_DIR / "overall_regressor.json"
    clf.save_model(str(clf_path))
    reg.save_model(str(reg_path))
    print(f"\nSaved: {clf_path.relative_to(ROOT)}")
    print(f"Saved: {reg_path.relative_to(ROOT)}")

    config = {
        "schema_version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "holdout_path": str(HOLDOUT_PATH.relative_to(ROOT)),
            "train_sha16": hash_path(TRAIN_PATH),
            "holdout_sha16": hash_path(HOLDOUT_PATH),
            "n_train": len(train_rows),
            "n_holdout": len(holdout_rows),
            "feature_dim": int(X_train.shape[1]),
            "check_registry_size": len(CHECK_REGISTRY),
        },
        "feature_schema": {
            "check_ids_sorted": sorted(CHECK_REGISTRY.keys()),
            "dim_order": DIMS,
        },
        "verdict_classifier": {
            "model_path": str(clf_path.relative_to(ROOT)),
            "hp": best_clf_hp,
            "scale_pos_weight": scale_pos_weight,
            "cv_auc_mean": best_clf_auc,
            "holdout_accuracy": clf_acc,
            "holdout_auc": clf_auc,
            "holdout_confusion": cm,
            "top_features": clf_imp,
            "gate": {"metric": "holdout_accuracy", "threshold": CLASSIFIER_GATE, "pass": clf_pass},
        },
        "overall_regressor": {
            "model_path": str(reg_path.relative_to(ROOT)),
            "hp": best_reg_hp,
            "cv_spearman_mean": best_reg_sp,
            "holdout_spearman": reg_sp,
            "holdout_pearson": reg_pe,
            "holdout_mae": reg_mae,
            "top_features": reg_imp,
            "gate": {"metric": "holdout_pearson", "threshold": REGRESSOR_GATE_PEARSON, "pass": reg_pass},
        },
        "gates_overall_pass": clf_pass and reg_pass,
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Saved: {CONFIG_PATH.relative_to(ROOT)}")

    if not (clf_pass and reg_pass):
        print("\n!! GATES FAILED — see config/rubric_calibration.yaml for top features. "
              "Return to Phase 0.10 to expand LLM-check coverage on dominant wrong-direction features.")
        sys.exit(2)


if __name__ == "__main__":
    main()
