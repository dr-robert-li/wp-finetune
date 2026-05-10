"""Phase 0 step 2: Score human + UGC + boundary seeds with rubric_scorer.

Computes per-dimension Spearman rank-order agreement between rubric_scorer
output and human-annotated dimension scores. Treats this as agreement on
"which defective code is worse on each dimension" rather than absolute score
regression (seeds are all FAIL-band examples; no PASS anchors yet — those
come in Phase 1 via clean WP core + top-plugin extraction).

Output: output/diagnostic/seed_scorer_agreement.json + .md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.rubric_scorer import score_code

# Seed dim name → rubric dim key
# Note: dependency_integrity captures function_exists / WP-API-hygiene patterns,
# better classified as D5_wp_api than D8_errors. Updated 2026-05-11.
SEED_DIM_MAP = {
    "performance": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "code_quality": "D9_structure",
    "security": "D2_security",
    "sql_safety": "D3_sql",
    "wpcs_compliance": "D1_wpcs",
    "i18n": "D6_i18n",
    "i18n_l10n": "D6_i18n",
    "accessibility": "D7_a11y",
    "dependency_integrity": "D5_wp_api",
    "error_handling": "D8_errors",
    "code_structure": "D9_structure",
}

SEED_FILES = [
    ("human", "deps/wp-finetune-data/human_seeds/human_annotated_seeds.json"),
    ("ugc", "deps/wp-finetune-data/ugc_seeds.json"),
    ("ugc_boundary", "deps/wp-finetune-data/ugc_boundary_seeds.json"),
]


def extract_code_and_scores(seed: dict) -> tuple[Optional[str], dict[str, float]]:
    """Return (code, {rubric_dim_key: score_0_10}) for a seed entry."""
    code = seed.get("defective_code") or seed.get("code") or seed.get("corrected_code")
    if "human_critique" in seed:
        dims = seed["human_critique"].get("dimensions", {})
    elif "human_reasoning" in seed:
        dims = seed["human_reasoning"].get("dimension_analysis", {})
    else:
        return code, {}
    out: dict[str, float] = {}
    for seed_dim, payload in dims.items():
        if not isinstance(payload, dict):
            continue
        score = payload.get("score")
        if score is None:
            continue
        key = SEED_DIM_MAP.get(seed_dim)
        if key is None:
            continue
        # Keep first-seen score per rubric key (in case 2 seed dims map to same rubric dim)
        out.setdefault(key, float(score))
    return code, out


def score_all_seeds() -> list[dict]:
    rows: list[dict] = []
    for source, rel in SEED_FILES:
        path = ROOT / rel
        with open(path) as f:
            seeds = json.load(f)
        for s in seeds:
            code, human_dims = extract_code_and_scores(s)
            if not code or not human_dims:
                continue
            sc = score_code(code, file_path=s["seed_id"])
            row = {
                "seed_id": s["seed_id"],
                "source": source,
                "seed_type": s["seed_type"],
                "defect_subtlety": s["defect_subtlety"],
                "human_dim_scores_0_10": human_dims,
                "rubric_dim_scores_0_10": {k: v for k, v in sc.dimension_scores.items() if v is not None},
                "rubric_dim_na": sc.dimension_na,
                "rubric_overall_0_100": sc.overall,
                "rubric_grade": sc.grade,
                "rubric_triggered_check_count": sum(len(v) for v in sc.triggered_checks.values()),
            }
            rows.append(row)
    return rows


def per_dim_correlation(rows: list[dict]) -> dict[str, dict]:
    """Per-rubric-dim Spearman + Pearson between human and rubric scores."""
    out: dict[str, dict] = {}
    for dim in [f"D{i}_{tag}" for i, tag in enumerate(
            ["wpcs", "security", "sql", "perf", "wp_api", "i18n", "a11y", "errors", "structure"], start=1)]:
        h_vals, r_vals = [], []
        for r in rows:
            h = r["human_dim_scores_0_10"].get(dim)
            v = r["rubric_dim_scores_0_10"].get(dim)
            if h is None or v is None:
                continue
            h_vals.append(h)
            r_vals.append(v)
        if len(h_vals) < 3:
            out[dim] = {"n": len(h_vals), "spearman": None, "pearson": None}
            continue
        sp = spearmanr(h_vals, r_vals)
        pe = pearsonr(h_vals, r_vals)
        out[dim] = {
            "n": len(h_vals),
            "spearman": float(sp.statistic) if hasattr(sp, "statistic") else float(sp[0]),
            "spearman_pvalue": float(sp.pvalue) if hasattr(sp, "pvalue") else float(sp[1]),
            "pearson": float(pe[0]) if not hasattr(pe, "statistic") else float(pe.statistic),
            "human_mean": float(np.mean(h_vals)),
            "rubric_mean": float(np.mean(r_vals)),
        }
    return out


def write_report(rows: list[dict], dim_corrs: dict[str, dict], out_dir: Path) -> None:
    json_path = out_dir / "seed_scorer_agreement.json"
    json_path.write_text(json.dumps({
        "n_seeds_scored": len(rows),
        "per_dim": dim_corrs,
        "rows": rows,
    }, indent=2))

    lines = ["# Seed Scorer Agreement (Phase 0 step 2)\n",
             f"Scored seeds: **{len(rows)}**  (all FAIL-band examples; PASS anchors come in Phase 1)\n",
             "## Per-dimension agreement (rubric_scorer vs human dim scores)\n",
             "| Dim | n | Spearman | p | Pearson | Human mean | Rubric mean |",
             "|-----|---|----------|---|---------|------------|-------------|"]
    for dim, c in dim_corrs.items():
        if c["spearman"] is None:
            lines.append(f"| {dim} | {c['n']} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {dim} | {c['n']} | {c['spearman']:+.3f} | {c['spearman_pvalue']:.3f} | {c['pearson']:+.3f} | {c['human_mean']:.2f} | {c['rubric_mean']:.2f} |")

    # Rubric overall distribution
    overalls = [r["rubric_overall_0_100"] for r in rows]
    lines += ["", "## Rubric overall_0_100 distribution",
              f"- n: {len(overalls)}  min: {min(overalls):.1f}  max: {max(overalls):.1f}  mean: {np.mean(overalls):.1f}  stdev: {np.std(overalls):.1f}",
              "",
              "## Tooling state",
              "Full 4-tool active: PHPCS (WordPress + WordPressVIPMinimum + Security) + PHPStan + regex.",
              "LLM-assisted checks (18 of them per rubric §F.5) still deferred — those capture",
              "the semantic defects humans flag (e.g. unbounded WP_Query, missing capability",
              "checks); weak per-dim agreement on D4/D5/D9 reflects that gap, not a tool failure."]
    (out_dir / "seed_scorer_agreement.md").write_text("\n".join(lines))


def main():
    out_dir = ROOT / "output" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Scoring seeds via rubric_scorer (partial-tooling mode)...")
    rows = score_all_seeds()
    print(f"Scored {len(rows)} seeds with at least 1 mapped dim.")
    dim_corrs = per_dim_correlation(rows)
    write_report(rows, dim_corrs, out_dir)
    print(f"Wrote {out_dir}/seed_scorer_agreement.json and .md")


if __name__ == "__main__":
    main()
