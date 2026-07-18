#!/usr/bin/env python3
"""Normalize the v5-labelling source assets into two unified tables.

Two entity types, two tables (they are different things and are NOT forced into one):

1. repos_catalog.csv  — one row per unique REPO (2,226 after merging 60 duplicate source rows). Union of the four catalog CSVs'
   columns (18 common + 5 variant, nulled where absent) plus two derived columns:
   `repo_type` (plugin|theme) and `catalog_tier` (top|poor). This is the corpus
   the extraction stage (LABELING_PLAN.md §3) clones.

2. seed_units.jsonl   — one row per ANNOTATED CODE UNIT (118 = 93 general + 25
   boundary). The two UGC seed files share one schema with a seed_type-dependent
   field split; this unifies them:
       code        := code            (deep_judge_cot)  | defective_code (critique_then_fix)
       annotation  := human_reasoning (deep_judge_cot)  | human_critique (critique_then_fix)
       corrected_code stays, null for deep_judge_cot rows.
   Output fields align with the labeling schema in LABELING_PLAN.md §6 where they
   overlap (dimensions_addressed, subtlety) so seeds can be loaded as pre-filled
   partial label rows. seed_units.csv is a flat export (list fields JSON-encoded);
   the JSONL is canonical.

The 27 client-engagement human seeds (human_annotated_seeds.json) are deliberately
NOT ingested — calibration-only pending clearance (LABELING_PLAN.md §13).

Usage:
    python3 normalize_assets.py              # writes repos_catalog.csv, seed_units.jsonl, seed_units.csv
    python3 normalize_assets.py --self-check # no writes; validates invariants on the source files
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

CATALOGS = {
    "wp_top1000_plugins_final.csv": ("plugin", "top"),
    "wp_poor_plugins_final.csv": ("plugin", "poor"),
    "wp_top100_themes_final.csv": ("theme", "top"),
    "wp_poor_themes_final.csv": ("theme", "poor"),
}
SEED_FILES = ["ugc_seeds.json", "ugc_boundary_seeds.json"]

# ponytail: fixed column order beats set-union nondeterminism — derived first,
# then the 18 common columns in top-plugins order, then the 5 variants.
REPO_COLUMNS = [
    "repo_type", "catalog_tier",
    "rank", "name", "slug", "active_installs", "rating_pct", "rating_5star",
    "editor_support", "last_updated", "homepage", "github_url", "wp_url",
    "total_known_vulns", "unpatched_vulns", "max_cvss", "max_cvss_vuln",
    "max_cvss_severity", "vuln_types", "top_cwes", "latest_vuln_date",
    "sample_vuln_titles", "tags", "download_url",
    "requires_wp", "tested_up_to", "github_repo_type", "num_ratings", "parent_theme",
]

SEED_COLUMNS = [
    "seed_id", "seed_source", "seed_type", "annotation_type", "defect_subtlety",
    "dimensions_addressed", "code", "annotation", "corrected_code",
    "source_platform", "source_url", "source_file",
]


def load_catalog_rows():
    """One row per unique (slug, repo_type). Two real dup classes in the sources:
    48 repos appear in BOTH tiers (high-install AND poorly-rated/vulnerable — e.g.
    hostinger, facebook-for-woocommerce), and 12 rows are exact same-file repeats.
    Cross-tier membership is DATA (catalog_tier becomes "poor,top"); same-file
    repeats keep the first row. Field merge prefers non-empty values."""
    merged = {}
    for fname, (repo_type, tier) in CATALOGS.items():
        with open(HERE / fname, newline="") as fh:
            for r in csv.DictReader(fh):
                k = (r["slug"], repo_type)
                if k not in merged:
                    r["repo_type"] = repo_type
                    r["_tiers"] = {tier}
                    merged[k] = r
                else:
                    m = merged[k]
                    m["_tiers"].add(tier)
                    for c, v in r.items():
                        if v and not m.get(c):
                            m[c] = v
    rows = []
    for m in merged.values():
        m["catalog_tier"] = ",".join(sorted(m.pop("_tiers")))
        rows.append(m)
    return rows


def normalize_seed(row, source_file):
    st = row["seed_type"]
    if st == "deep_judge_cot":
        code, annotation = row.get("code"), row.get("human_reasoning")
    else:  # critique_then_fix
        code, annotation = row.get("defective_code"), row.get("human_critique")
    return {
        "seed_id": row["seed_id"],
        "seed_source": source_file,
        "seed_type": st,
        "annotation_type": row.get("annotation_type"),
        "defect_subtlety": row.get("defect_subtlety"),
        "dimensions_addressed": row.get("dimensions_addressed"),
        "code": code,
        "annotation": annotation,
        "corrected_code": row.get("corrected_code"),
        "source_platform": row.get("source_platform"),
        "source_url": row.get("source_url"),
        "source_file": row.get("source_file"),
    }


def load_seed_rows():
    out = []
    for fname in SEED_FILES:
        data = json.loads((HERE / fname).read_text())
        rows = data if isinstance(data, list) else data.get("seeds", [])
        out.extend(normalize_seed(r, fname) for r in rows)
    return out


def check(rows_r, rows_s):
    assert len(rows_r) == 2226, f"expected 2226 unique repos (2286 source rows - 60 dups), got {len(rows_r)}"
    assert len({(r["slug"], r["repo_type"]) for r in rows_r}) == len(rows_r), "dup slug+type after merge"
    dual = sum(1 for r in rows_r if r["catalog_tier"] == "poor,top")
    assert dual == 48, f"expected 48 dual-tier repos, got {dual}"
    for r in rows_r:
        assert r["repo_type"] in ("plugin", "theme") and r["catalog_tier"] in ("top", "poor", "poor,top")
        assert not (set(r) - set(REPO_COLUMNS)), f"unmapped catalog column: {set(r) - set(REPO_COLUMNS)}"
    assert len(rows_s) == 118, f"expected 118 seeds, got {len(rows_s)}"
    assert len({s["seed_id"] for s in rows_s}) == 118, "dup seed_id"
    for s in rows_s:
        # every seed must carry code + its annotation, whichever variant produced it
        assert s["code"] and s["annotation"], f"seed {s['seed_id']} missing code/annotation"
        assert s["seed_type"] in ("deep_judge_cot", "critique_then_fix"), s["seed_type"]
        if s["seed_type"] == "critique_then_fix":
            assert s["corrected_code"], f"critique_then_fix seed {s['seed_id']} lacks corrected_code"


def main():
    rows_r, rows_s = load_catalog_rows(), load_seed_rows()
    check(rows_r, rows_s)
    if "--self-check" in sys.argv:
        print("self-check OK (2226 unique repos, 118 seeds, invariants hold; no files written)")
        return 0
    with open(HERE / "repos_catalog.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPO_COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows_r)
    with open(HERE / "seed_units.jsonl", "w") as fh:
        for s in rows_s:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(HERE / "seed_units.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SEED_COLUMNS)
        w.writeheader()
        for s in rows_s:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in s.items()})
    print(f"wrote repos_catalog.csv ({len(rows_r)} rows), seed_units.jsonl/.csv ({len(rows_s)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
