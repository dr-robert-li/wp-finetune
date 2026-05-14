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
import random
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

# Reserve a deterministic slice of PASS anchors into holdout so the verdict
# classifier gate is discriminating (FAIL seeds alone would let "predict FAIL
# always" trivially hit ≥0.85 accuracy).
HOLDOUT_PASS_ANCHORS = 20
HOLDOUT_PASS_ANCHOR_SEED = 1729  # different from training seed to avoid coupling

# wp-bench leakage scope: only flag anchors from the WordPress core repo (the only
# repo wp-bench-core knowledge bank corresponds to). Plugin anchors share no
# provenance with the bench and shouldn't be policed by it.
LEAKAGE_AT_RISK_REPOS = {"wordpress-develop"}


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
    """Return (gt_overall, gt_verdict, subtlety). None if cannot derive.

    Schema-tolerant: convention is schema-follows-`seed_type`, not -file.
    `critique_then_fix` seeds carry `human_critique` (per-dim 0-10 scores);
    `deep_judge_cot` seeds carry `human_reasoning` (overall_score + verdict).
    Both can appear in any source file. Earlier code branched on `source` and
    silently dropped cross-schema seeds — see audit at
    `data/calibration/audit/AUDIT-2026-05-14.md`.
    """
    rec = seed_lookup.get(seed_id)
    if rec is None:
        return None, None, "unknown"
    raw = rec["raw"]
    subtlety = raw.get("defect_subtlety", "unknown")

    # Prefer explicit overall_score + verdict (deep_judge_cot schema)
    hr = raw.get("human_reasoning") or {}
    overall = hr.get("overall_score")
    verdict = hr.get("verdict")
    if overall is not None and verdict is not None:
        return float(overall), str(verdict).upper(), subtlety

    # Fall back to per-dim scores (critique_then_fix schema). All FAIL pool
    # seeds, so verdict is implicit FAIL when only the critique block exists.
    overall = derive_human_overall(raw)
    if overall is not None:
        return overall, "FAIL", subtlety

    return None, None, subtlety


def extract_wpbench_call_sites() -> set[str]:
    """Collect bare function names referenced as CALL SITES (name followed by '(').

    Tightened from a permissive identifier-token scan to avoid over-flagging:
    bare 4+-char tokens caught JSON keys + prose + every common WP word.
    Call-site form (`name(`) limits matches to actual function references.
    """
    if not WPBENCH_KNOWLEDGE_DIR.exists():
        print(f"  WARN: {WPBENCH_KNOWLEDGE_DIR} not found; skipping leakage audit")
        return set()
    call_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
    names: set[str] = set()
    for jf in sorted(WPBENCH_KNOWLEDGE_DIR.glob("*.json")):
        text = jf.read_text()
        for m in call_re.finditer(text):
            names.add(m.group(1))
    return names


def anchor_row(anchor: dict, leaked: bool, split: str = "train") -> dict:
    return {
        "row_id": f"anchor::{anchor.get('source_repo')}::{anchor.get('function_name')}",
        "source": "pass_anchor",
        "split": split,
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

    print("Computing wp-bench leakage call-sites (tightened: name followed by '(' "
          f"AND source_repo in {sorted(LEAKAGE_AT_RISK_REPOS)})")
    wpbench_calls = extract_wpbench_call_sites()
    print(f"  {len(wpbench_calls)} call-site names in wp-bench knowledge")

    # ---- Anchor rows + leakage audit ----
    kept_anchors: list[dict] = []
    leaked_anchors: list[dict] = []
    for a in anchors:
        fn = a.get("function_name") or ""
        repo = a.get("source_repo") or ""
        # Strip leading "ClassName::" if present
        bare = fn.split("::")[-1] if "::" in fn else fn
        # Only police WP-core anchors (only repo wp-bench-core knowledge corresponds to).
        leaked = repo in LEAKAGE_AT_RISK_REPOS and bool(bare) and bare in wpbench_calls
        if leaked:
            leaked_anchors.append({"function_name": fn, "source_repo": repo})
        else:
            kept_anchors.append(a)
    audit["wpbench_call_sites"] = len(wpbench_calls)
    audit["leakage_at_risk_repos"] = sorted(LEAKAGE_AT_RISK_REPOS)
    audit["anchors_total"] = len(anchors)
    audit["anchors_dropped_leakage"] = len(leaked_anchors)
    audit["anchors_kept"] = len(kept_anchors)
    audit["leaked_sample"] = leaked_anchors[:20]
    print(f"  Leakage: dropped {len(leaked_anchors)}/{len(anchors)} anchors")

    # Reserve a deterministic random slice of PASS anchors into holdout so the
    # verdict-classifier gate is discriminating (otherwise holdout = all-FAIL
    # boundary seeds and "predict FAIL always" trivially exceeds 0.85 acc).
    rng = random.Random(HOLDOUT_PASS_ANCHOR_SEED)
    indices = list(range(len(kept_anchors)))
    rng.shuffle(indices)
    n_holdout_anchors = min(HOLDOUT_PASS_ANCHORS, len(kept_anchors))
    holdout_anchor_ix = set(indices[:n_holdout_anchors])
    anchor_rows: list[dict] = []
    holdout_anchor_rows: list[dict] = []
    for i, a in enumerate(kept_anchors):
        if i in holdout_anchor_ix:
            holdout_anchor_rows.append(anchor_row(a, leaked=False, split="holdout"))
        else:
            anchor_rows.append(anchor_row(a, leaked=False, split="train"))
    audit["holdout_pass_anchors"] = n_holdout_anchors
    audit["holdout_pass_anchor_seed"] = HOLDOUT_PASS_ANCHOR_SEED

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

    all_train = anchor_rows + train_seed_rows
    all_holdout = holdout_anchor_rows + holdout_rows
    with train_path.open("w") as f:
        for r in all_train:
            f.write(json.dumps(r) + "\n")
    with holdout_path.open("w") as f:
        for r in all_holdout:
            f.write(json.dumps(r) + "\n")

    audit["train_rows"] = len(all_train)
    audit["holdout_rows"] = len(all_holdout)
    audit["train_label_dist"] = {
        "PASS": sum(1 for r in all_train if r["gt_verdict"] == "PASS"),
        "FAIL": sum(1 for r in all_train if r["gt_verdict"] == "FAIL"),
    }
    audit["holdout_label_dist"] = {
        "PASS": sum(1 for r in all_holdout if r["gt_verdict"] == "PASS"),
        "FAIL": sum(1 for r in all_holdout if r["gt_verdict"] == "FAIL"),
    }
    audit["train_subtlety_dist"] = {}
    for r in all_train:
        k = r["subtlety"]
        audit["train_subtlety_dist"][k] = audit["train_subtlety_dist"].get(k, 0) + 1
    audit["holdout_subtlety_dist"] = {}
    for r in all_holdout:
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
