"""Audit the 79 FAIL seeds dropped during Phase 1a calibration GT-derivation.

Read-only against existing artifacts. Reproduces the drop logic from
scripts/build_calibration_dataset.py (derive_gt, derive_human_overall) but
classifies each dropped seed by *specific* missing field instead of
aggregating to "no_gt", then KS-tests rubric feature distributions of kept
vs dropped seeds (stratified by source) to detect non-random drop bias that
would propagate into the Phase 1b calibrated rubric.

Outputs (under data/calibration/audit/):
  - dropped_seeds_classified.jsonl   one row per seed (all 145), granular reason
  - ks_results.json                  per-(source, feature) KS test results
  - AUDIT-<date>.md                  decision report w/ Green/Yellow/Red verdict
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

SEED_FEATURES = ROOT / "output/diagnostic/seed_scorer_features.jsonl"
SEED_SOURCES = {
    "human": ROOT / "deps/wp-finetune-data/human_seeds/human_annotated_seeds.json",
    "ugc": ROOT / "deps/wp-finetune-data/ugc_seeds.json",
    "ugc_boundary": ROOT / "deps/wp-finetune-data/ugc_boundary_seeds.json",
}
TRAIN_SPLIT = ROOT / "data/calibration/train.jsonl"
HOLDOUT_SPLIT = ROOT / "data/calibration/holdout.jsonl"

OUT_DIR = ROOT / "data/calibration/audit"
CLASSIFIED_PATH = OUT_DIR / "dropped_seeds_classified.jsonl"
KS_PATH = OUT_DIR / "ks_results.json"
REPORT_PATH = OUT_DIR / f"AUDIT-{date.today().isoformat()}.md"

KS_FEATURES = ["rubric_overall_0_100", "rubric_triggered_check_count"] + [
    f"D{i}_{name}" for i, name in enumerate(
        ["wpcs", "security", "sql", "perf", "wp_api", "i18n", "a11y", "errors", "structure"], start=1)
]
MIN_N_FOR_KS = 5
ALPHA = 0.05


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def classify_seed(source: str, raw: dict) -> tuple[Optional[float], Optional[str], str]:
    """Return (gt_overall, gt_verdict, classification).

    Mirrors the schema-tolerant `derive_gt()` in
    `scripts/build_calibration_dataset.py:96-128` (post-pivot patch). Convention
    is schema-follows-`seed_type` (not -file): `deep_judge_cot` seeds carry
    `human_reasoning`; `critique_then_fix` seeds carry `human_critique`. Both
    can live in any source file.

    Classification labels (no actual drops expected post-patch):
      `kept_via_human_reasoning`   - resolved through explicit overall_score + verdict
      `kept_via_human_critique`    - resolved through per-dim mean × 10 fallback
      `legacy_drop_no_reasoning`   - lacks human_reasoning AND human_critique (true gap)
      `legacy_drop_critique_empty` - both blocks present but neither parseable
    """
    hr = raw.get("human_reasoning") or {}
    overall = hr.get("overall_score")
    verdict = hr.get("verdict")
    if overall is not None and verdict is not None:
        return float(overall), str(verdict).upper(), "kept_via_human_reasoning"

    hc = raw.get("human_critique") or {}
    dims = hc.get("dimensions") or {}
    vals = []
    for payload in dims.values():
        if isinstance(payload, dict):
            s = payload.get("score")
            if isinstance(s, (int, float)):
                vals.append(float(s))
    if vals:
        return (sum(vals) / len(vals)) * 10.0, "FAIL", "kept_via_human_critique"

    if not hr and not hc:
        return None, None, "legacy_drop_no_reasoning"
    return None, None, "legacy_drop_critique_empty"


def load_seed_lookup() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for source, path in SEED_SOURCES.items():
        for s in json.loads(path.read_text()):
            out[s["seed_id"]] = {"source": source, "raw": s}
    return out


def stratified_ks(rows: list[dict]) -> dict:
    """KS test per (source × feature). Skips cells with n < MIN_N_FOR_KS."""
    by_source: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in rows:
        src = r["source"]
        bucket = "kept" if r["kept"] else "dropped"
        feats = r["_features"]
        if not feats:  # seed had no entry in seed_scorer_features.jsonl
            continue
        s = by_source.setdefault(src, {})
        for fname in KS_FEATURES:
            v = feats.get(fname)
            if v is None:
                continue
            s.setdefault(fname, {"kept": [], "dropped": []})[bucket].append(float(v))

    results = []
    n_tests = 0
    for src, fmap in by_source.items():
        for fname, samples in fmap.items():
            kept = samples["kept"]
            dropped = samples["dropped"]
            n_kept = len(kept)
            n_dropped = len(dropped)
            entry: dict = {
                "source": src,
                "feature": fname,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
            }
            if n_kept < MIN_N_FOR_KS or n_dropped < MIN_N_FOR_KS:
                entry["status"] = "insufficient_n"
            else:
                ks = stats.ks_2samp(kept, dropped)
                entry["statistic"] = float(ks.statistic)
                entry["p_value"] = float(ks.pvalue)
                entry["status"] = "tested"
                n_tests += 1
            results.append(entry)

    bonferroni = ALPHA / n_tests if n_tests else None
    for e in results:
        if e["status"] == "tested":
            e["significant_raw"] = e["p_value"] < ALPHA
            e["significant_bonferroni"] = bonferroni is not None and e["p_value"] < bonferroni
    return {
        "alpha": ALPHA,
        "n_tests": n_tests,
        "bonferroni_alpha": bonferroni,
        "min_n_for_ks": MIN_N_FOR_KS,
        "results": results,
    }


def feature_dict(feat_row: Optional[dict]) -> dict:
    """Flatten the rubric features we KS-test on into a single name→value dict.

    `rubric_dim_scores_full` carries D1..D9 with explicit nulls for NA dims;
    those nulls naturally drop out of the KS sample (we skip None below).
    """
    if feat_row is None:
        return {}
    out = {
        "rubric_overall_0_100": feat_row.get("rubric_overall_0_100"),
        "rubric_triggered_check_count": feat_row.get("rubric_triggered_check_count"),
    }
    full = feat_row.get("rubric_dim_scores_full") or feat_row.get("rubric_dim_scores_0_10") or {}
    for k, v in full.items():
        out[k] = v
    return {k: v for k, v in out.items() if v is not None}


def boundary_skew(rows: list[dict]) -> dict:
    by_source: dict[str, dict[str, dict[str, int]]] = {}
    for r in rows:
        src = r["source"]
        bucket = "kept" if r["kept"] else "dropped"
        is_boundary = r["defect_subtlety"] == "boundary"
        s = by_source.setdefault(src, {"kept": {"boundary": 0, "other": 0},
                                        "dropped": {"boundary": 0, "other": 0}})
        s[bucket]["boundary" if is_boundary else "other"] += 1
    out = {}
    for src, b in by_source.items():
        kn = b["kept"]["boundary"] + b["kept"]["other"]
        dn = b["dropped"]["boundary"] + b["dropped"]["other"]
        kf = (b["kept"]["boundary"] / kn) if kn else None
        df = (b["dropped"]["boundary"] / dn) if dn else None
        ratio = (df / kf) if (kf and df is not None) else None
        out[src] = {
            "kept_boundary_frac": kf, "dropped_boundary_frac": df,
            "ratio_dropped_over_kept": ratio,
            "counts": b,
        }
    return out


def render_report(rows: list[dict], ks: dict, skew: dict, split_check: dict) -> str:
    # Drop-reason table: source × reason counts
    reason_table: dict[tuple[str, str], int] = {}
    for r in rows:
        if not r["kept"]:
            reason_table[(r["source"], r["drop_reason"])] = reason_table.get((r["source"], r["drop_reason"]), 0) + 1

    lines = [f"# Dropped-seed audit — {date.today().isoformat()}", "",
             "Read-only audit of the 79 FAIL seeds dropped during Phase 1a calibration GT derivation.",
             "Generated by `scripts/audit_dropped_seeds.py`. Reproduces drop logic from",
             "`scripts/build_calibration_dataset.py:82-115, 239-257`.", "",
             "## 1. Drop-reason split", "",
             "| Source | Reason | Count |", "|---|---|---:|"]
    for (src, reason), count in sorted(reason_table.items()):
        lines.append(f"| {src} | `{reason}` | {count} |")
    lines.append(f"| **TOTAL** | dropped | {sum(reason_table.values())} |")
    lines.append(f"| **TOTAL** | kept | {sum(1 for r in rows if r['kept'])} |")
    lines.append("")

    lines += ["## 2. KS distribution test (kept vs dropped, stratified by source)", "",
              f"alpha={ks['alpha']}, n_tests={ks['n_tests']}, "
              f"bonferroni_alpha={ks['bonferroni_alpha']:.4g}" if ks["bonferroni_alpha"] else
              f"alpha={ks['alpha']}, n_tests=0 (no testable cells)", "",
              "| Source | Feature | n_kept | n_dropped | KS stat | p_raw | sig α | sig Bonferroni |",
              "|---|---|---:|---:|---:|---:|:---:|:---:|"]
    for e in ks["results"]:
        if e["status"] == "insufficient_n":
            lines.append(f"| {e['source']} | {e['feature']} | {e['n_kept']} | {e['n_dropped']} | — | — | n<5 | n<5 |")
        else:
            sig_a = "✓" if e["significant_raw"] else "·"
            sig_b = "✓" if e["significant_bonferroni"] else "·"
            lines.append(f"| {e['source']} | {e['feature']} | {e['n_kept']} | {e['n_dropped']} | "
                         f"{e['statistic']:.3f} | {e['p_value']:.3g} | {sig_a} | {sig_b} |")
    lines.append("")

    lines += ["## 3. Boundary-subtlety check", "",
              "Drops over-representing `boundary` cases → calibration boundary biased toward easy FAILs.", "",
              "| Source | kept boundary frac | dropped boundary frac | ratio (dropped/kept) |",
              "|---|---:|---:|---:|"]
    for src, s in skew.items():
        kf = "—" if s["kept_boundary_frac"] is None else f"{s['kept_boundary_frac']:.2f}"
        df_ = "—" if s["dropped_boundary_frac"] is None else f"{s['dropped_boundary_frac']:.2f}"
        rr = "—" if s["ratio_dropped_over_kept"] is None else f"{s['ratio_dropped_over_kept']:.2f}×"
        lines.append(f"| {src} | {kf} | {df_} | {rr} |")
    lines.append("")

    lines += ["## 4. Sanity checks", "",
              f"- seed feature rows: {split_check['n_feature_rows']} (expect 145)",
              f"- kept (gt derivable): {split_check['n_kept']} (expect 66)",
              f"- dropped (no gt): {split_check['n_dropped']} (expect 79)",
              f"- dropped seeds absent from train+holdout: "
              f"{split_check['dropped_absent_from_splits']}/{split_check['n_dropped']}",
              ""]

    # Decision
    sig_dims = sum(1 for e in ks["results"] if e.get("significant_bonferroni"))
    max_skew = max((s["ratio_dropped_over_kept"] or 0) for s in skew.values()) if skew else 0
    if sig_dims == 0 and max_skew < 1.5:
        verdict = "**GREEN** — no significant drift, drops scatter across reasons. Ship Phase 1b verdicts as-is."
    elif sig_dims <= 2 and max_skew < 1.5:
        verdict = (f"**YELLOW** — {sig_dims} dim(s) drift post-Bonferroni; max boundary skew {max_skew:.2f}×. "
                   "Ship pilot, treat results as advisory pending backfill of worst dropped boundary seeds.")
    else:
        verdict = (f"**RED** — {sig_dims} dim(s) drift post-Bonferroni; max boundary skew {max_skew:.2f}×. "
                   "Hold Phase 1b results, backfill, re-fit, diff coefficients before consuming.")
    lines += ["## 5. Decision", "", verdict, ""]

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feats = load_jsonl(SEED_FEATURES)
    feats_by_id = {f["seed_id"]: f for f in feats}
    seed_lookup = load_seed_lookup()

    rows: list[dict] = []
    for sid, rec in seed_lookup.items():
        gt_overall, gt_verdict, reason = classify_seed(rec["source"], rec["raw"])
        kept = reason.startswith("kept")
        feat_row = feats_by_id.get(sid)
        rows.append({
            "seed_id": sid,
            "source": rec["source"],
            "seed_type": rec["raw"].get("seed_type"),
            "defect_subtlety": rec["raw"].get("defect_subtlety", "unknown"),
            "kept": kept,
            "drop_reason": None if kept else reason,
            "gt_overall": gt_overall,
            "gt_verdict": gt_verdict,
            "_features": feature_dict(feat_row),
        })

    # Persist classified jsonl (without _features to keep small)
    with CLASSIFIED_PATH.open("w") as f:
        for r in rows:
            r_out = {k: v for k, v in r.items() if k != "_features"}
            f.write(json.dumps(r_out) + "\n")

    ks = stratified_ks(rows)
    KS_PATH.write_text(json.dumps(ks, indent=2))

    skew = boundary_skew(rows)

    # Sanity: dropped seed_ids must NOT appear in train/holdout
    in_splits = set()
    for sp in [TRAIN_SPLIT, HOLDOUT_SPLIT]:
        if sp.exists():
            for line in sp.open():
                d = json.loads(line)
                meta = d.get("_meta", {})
                if "seed_id" in meta:
                    in_splits.add(meta["seed_id"])
    dropped_ids = {r["seed_id"] for r in rows if not r["kept"]}
    absent = sum(1 for sid in dropped_ids if sid not in in_splits)
    split_check = {
        "n_feature_rows": len(feats),
        "n_kept": sum(1 for r in rows if r["kept"]),
        "n_dropped": sum(1 for r in rows if not r["kept"]),
        "dropped_absent_from_splits": absent,
    }

    REPORT_PATH.write_text(render_report(rows, ks, skew, split_check))

    print(f"Wrote {CLASSIFIED_PATH.relative_to(ROOT)}: {len(rows)} rows "
          f"(kept={split_check['n_kept']}, dropped={split_check['n_dropped']})")
    print(f"Wrote {KS_PATH.relative_to(ROOT)}: {ks['n_tests']} KS tests "
          f"(bonferroni alpha {ks['bonferroni_alpha']})")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Sanity: {absent}/{len(dropped_ids)} dropped seeds absent from train+holdout splits")


if __name__ == "__main__":
    main()
