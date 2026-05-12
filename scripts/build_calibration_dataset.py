"""Phase 1a step 3-5: Assemble calibration train/holdout split for XGBoost dual-head fit.

Joins:
  - output/diagnostic/pass_anchors_features.jsonl   (500 PASS anchors)
  - output/diagnostic/seed_scorer_features.jsonl    (145 FAIL seeds)

Derives gt_overall + gt_verdict per row (see SOURCES table below).

File-based holdout (45 rows):
  - 25 from ugc_boundary_seeds.json (all boundary subtlety)
  - 20 from human_annotated_seeds.json where defect_subtlety == "boundary"

Train (~600 rows):
  - 500 PASS anchors (after wp-bench leakage drop)
  -   7 clear-cut human FAIL
  -  60 clear-cut UGC FAIL (from ugc_seeds.json)
  -  33 boundary-subtlety rows from ugc_seeds.json (not the holdout file)

Outputs:
  data/calibration/train.jsonl
  data/calibration/holdout.jsonl
  data/calibration/leakage_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANCHORS_FEATURES = ROOT / "output" / "diagnostic" / "pass_anchors_features.jsonl"
SEED_FEATURES = ROOT / "output" / "diagnostic" / "seed_scorer_features.jsonl"
SEED_SOURCES = {
    "human": ROOT / "deps/wp-finetune-data/human_seeds/human_annotated_seeds.json",
    "ugc": ROOT / "deps/wp-finetune-data/ugc_seeds.json",
    "ugc_boundary": ROOT / "deps/wp-finetune-data/ugc_boundary_seeds.json",
}
WPBENCH_KNOWLEDGE_DIR = ROOT / "wp-bench/datasets/suites/wp-core-v1/knowledge"
OUT_DIR = ROOT / "data" / "calibration"

ANCHOR_CLAMP = 95.0
PASS_THRESHOLD = 70.0  # used only as derivation fallback (not gating)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_seed_lookup() -> dict[str, dict]:
    """Index every seed by seed_id with its source + raw record."""
    out: dict[str, dict] = {}
    for source, path in SEED_SOURCES.items():
        seeds = json.loads(path.read_text())
        for s in seeds:
            sid = s["seed_id"]
            out[sid] = {"source": source, "raw": s}
    return out


def derive_human_overall(seed: dict) -> Optional[float]:
    """Human seeds: gt_overall = mean(non-null per-dim 0-10 scores) * 10."""
    dims = (seed.get("human_critique") or {}).get("dimensions", {})
    vals = []
    for payload in dims.values():
        if isinstance(payload, dict):
            s = payload.get("score")
            if isinstance(s, (int, float)):
                vals.append(float(s))
    if not vals:
        return None
    return (sum(vals) / len(vals)) * 10.0


def derive_gt(seed_id: str, seed_lookup: dict[str, dict]) -> tuple[Optional[float], Optional[str], str]:
    """Return (gt_overall, gt_verdict, subtlety). None if cannot derive."""
    rec = seed_lookup.get(seed_id)
    if rec is None:
        return None, None, "unknown"
    source = rec["source"]
    raw = rec["raw"]
    subtlety = raw.get("defect_subtlety", "unknown")

    if source == "human":
        overall = derive_human_overall(raw)
        return overall, "FAIL", subtlety  # all human seeds are defective annotations

    # UGC + UGC-boundary share schema
    hr = raw.get("human_reasoning") or {}
    overall = hr.get("overall_score")
    verdict = hr.get("verdict")
    if overall is None or verdict is None:
        return None, None, subtlety
    return float(overall), str(verdict).upper(), subtlety


def extract_wpbench_tokens() -> set[str]:
    """Collect identifier-like tokens (>= 4 chars) referenced anywhere in wp-bench knowledge JSONs."""
    if not WPBENCH_KNOWLEDGE_DIR.exists():
        print(f"  WARN: {WPBENCH_KNOWLEDGE_DIR} not found; skipping leakage audit")
        return set()
    token_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
    tokens: set[str] = set()
    for jf in sorted(WPBENCH_KNOWLEDGE_DIR.glob("*.json")):
        text = jf.read_text()
        for m in token_re.finditer(text):
            tokens.add(m.group(1))
    return tokens


def anchor_row(anchor: dict, leaked: bool) -> dict:
    return {
        "row_id": f"anchor::{anchor.get('source_repo')}::{anchor.get('function_name')}",
        "source": "pass_anchor",
        "split": "train",
        "subtlety": "clear-cut",
        "triggered_checks_flat": anchor.get("triggered_checks_flat", []),
        "dim_scores": anchor.get("rubric_dim_scores", {}),
        "dim_na": anchor.get("dimension_na", []),
        "rubric_overall": float(anchor.get("rubric_overall", 0.0)),
        "gt_overall": min(float(anchor.get("rubric_overall", 95.0)), ANCHOR_CLAMP),
        "gt_verdict": "PASS",
        "_meta": {
            "function_name": anchor.get("function_name"),
            "source_repo": anchor.get("source_repo"),
            "leaked": leaked,
        },
    }


def seed_row(feat: dict, gt_overall: float, gt_verdict: str, subtlety: str, split: str) -> dict:
    return {
        "row_id": f"seed::{feat['source']}::{feat['seed_id']}",
        "source": feat["source"],
        "split": split,
        "subtlety": subtlety,
        "triggered_checks_flat": feat.get("triggered_checks_flat", []),
        "dim_scores": feat.get("rubric_dim_scores_full", feat.get("rubric_dim_scores_0_10", {})),
        "dim_na": feat.get("rubric_dim_na", []),
        "rubric_overall": float(feat.get("rubric_overall_0_100", 0.0)),
        "gt_overall": float(gt_overall),
        "gt_verdict": gt_verdict,
        "_meta": {"seed_id": feat["seed_id"], "seed_type": feat.get("seed_type")},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    print(f"Loading anchor features: {ANCHORS_FEATURES.relative_to(ROOT)}")
    anchors = load_jsonl(ANCHORS_FEATURES)
    print(f"  {len(anchors)} anchors")

    print(f"Loading seed features: {SEED_FEATURES.relative_to(ROOT)}")
    seed_feats = load_jsonl(SEED_FEATURES)
    print(f"  {len(seed_feats)} seed feature rows")

    print("Loading raw seed records for GT derivation")
    seed_lookup = load_seed_lookup()
    print(f"  {len(seed_lookup)} seed records across {len(SEED_SOURCES)} sources")

    print("Computing wp-bench leakage tokens")
    wpbench_tokens = extract_wpbench_tokens()
    print(f"  {len(wpbench_tokens)} candidate identifier tokens in wp-bench knowledge")

    # ---- Anchor rows + leakage audit ----
    anchor_rows: list[dict] = []
    leaked_anchors: list[dict] = []
    for a in anchors:
        fn = a.get("function_name") or ""
        # Strip leading "ClassName::" if present
        bare = fn.split("::")[-1] if "::" in fn else fn
        leaked = bool(bare) and bare in wpbench_tokens
        row = anchor_row(a, leaked=leaked)
        if leaked:
            leaked_anchors.append({"function_name": fn, "source_repo": a.get("source_repo")})
        else:
            anchor_rows.append(row)
    audit["wpbench_tokens"] = len(wpbench_tokens)
    audit["anchors_total"] = len(anchors)
    audit["anchors_dropped_leakage"] = len(leaked_anchors)
    audit["anchors_kept"] = len(anchor_rows)
    audit["leaked_sample"] = leaked_anchors[:20]
    print(f"  Leakage: dropped {len(leaked_anchors)}/{len(anchors)} anchors")

    # ---- Seed rows ----
    holdout_rows: list[dict] = []
    train_seed_rows: list[dict] = []
    dropped_seeds: list[dict] = []

    for feat in seed_feats:
        sid = feat["seed_id"]
        source = feat["source"]
        gt_overall, gt_verdict, subtlety = derive_gt(sid, seed_lookup)
        if gt_overall is None or gt_verdict is None:
            dropped_seeds.append({"seed_id": sid, "source": source, "reason": "no_gt"})
            continue
        # Holdout rule: ugc_boundary file OR human-boundary
        if source == "ugc_boundary":
            split = "holdout"
        elif source == "human" and subtlety == "boundary":
            split = "holdout"
        else:
            split = "train"
        row = seed_row(feat, gt_overall, gt_verdict, subtlety, split)
        if split == "holdout":
            holdout_rows.append(row)
        else:
            train_seed_rows.append(row)

    audit["seeds_total"] = len(seed_feats)
    audit["seeds_dropped_no_gt"] = len(dropped_seeds)
    audit["seeds_to_holdout"] = len(holdout_rows)
    audit["seeds_to_train"] = len(train_seed_rows)
    audit["dropped_seeds_sample"] = dropped_seeds[:20]

    # ---- Write splits ----
    train_path = OUT_DIR / "train.jsonl"
    holdout_path = OUT_DIR / "holdout.jsonl"

    with train_path.open("w") as f:
        for r in anchor_rows + train_seed_rows:
            f.write(json.dumps(r) + "\n")
    with holdout_path.open("w") as f:
        for r in holdout_rows:
            f.write(json.dumps(r) + "\n")

    audit["train_rows"] = len(anchor_rows) + len(train_seed_rows)
    audit["holdout_rows"] = len(holdout_rows)
    audit["train_label_dist"] = {
        "PASS": sum(1 for r in anchor_rows + train_seed_rows if r["gt_verdict"] == "PASS"),
        "FAIL": sum(1 for r in anchor_rows + train_seed_rows if r["gt_verdict"] == "FAIL"),
    }
    audit["holdout_label_dist"] = {
        "PASS": sum(1 for r in holdout_rows if r["gt_verdict"] == "PASS"),
        "FAIL": sum(1 for r in holdout_rows if r["gt_verdict"] == "FAIL"),
    }
    audit["train_subtlety_dist"] = {}
    for r in anchor_rows + train_seed_rows:
        k = r["subtlety"]
        audit["train_subtlety_dist"][k] = audit["train_subtlety_dist"].get(k, 0) + 1
    audit["holdout_subtlety_dist"] = {}
    for r in holdout_rows:
        k = r["subtlety"]
        audit["holdout_subtlety_dist"][k] = audit["holdout_subtlety_dist"].get(k, 0) + 1

    audit_path = OUT_DIR / "leakage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    print()
    print(f"Wrote {train_path.relative_to(ROOT)}: {audit['train_rows']} rows "
          f"(PASS={audit['train_label_dist']['PASS']}, FAIL={audit['train_label_dist']['FAIL']})")
    print(f"Wrote {holdout_path.relative_to(ROOT)}: {audit['holdout_rows']} rows "
          f"(PASS={audit['holdout_label_dist']['PASS']}, FAIL={audit['holdout_label_dist']['FAIL']})")
    print(f"Wrote {audit_path.relative_to(ROOT)}")
    print(f"  Train subtlety: {audit['train_subtlety_dist']}")
    print(f"  Holdout subtlety: {audit['holdout_subtlety_dist']}")


if __name__ == "__main__":
    main()
