"""Phase 0 step 2: Score human + UGC + boundary seeds with rubric_scorer.

Computes per-dimension Spearman rank-order agreement between rubric_scorer
output and human-annotated dimension scores. Treats this as agreement on
"which defective code is worse on each dimension" rather than absolute score
regression (seeds are all FAIL-band examples; no PASS anchors yet — those
come in Phase 1 via clean WP core + top-plugin extraction).

Set RUBRIC_USE_LLM_CHECKS=1 + --workers N to enable LLM-check pass with
parallel agent invocation. Default deterministic-only (PHPCS + PHPStan + regex).

Output: output/diagnostic/seed_scorer_agreement.json + .md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _score_one(seed: dict, source: str) -> Optional[dict]:
    code, human_dims = extract_code_and_scores(seed)
    if not code or not human_dims:
        return None
    sc = score_code(code, file_path=seed["seed_id"])
    triggered_flat = sorted({cid for ids in sc.triggered_checks.values() for cid in ids})
    return {
        "seed_id": seed["seed_id"],
        "source": source,
        "seed_type": seed["seed_type"],
        "defect_subtlety": seed["defect_subtlety"],
        "human_dim_scores_0_10": human_dims,
        "rubric_dim_scores_0_10": {k: v for k, v in sc.dimension_scores.items() if v is not None},
        "rubric_dim_na": sc.dimension_na,
        "rubric_overall_0_100": sc.overall,
        "rubric_grade": sc.grade,
        "rubric_triggered_check_count": len(triggered_flat),
        "llm_checks_skipped": sc.llm_checks_skipped,
        # Calibration feature payload (Phase 1a). Additive; existing consumers ignore.
        "triggered_checks_flat": triggered_flat,
        "floor_rules_applied": list(sc.floor_rules_applied),
        "rubric_dim_scores_full": dict(sc.dimension_scores),
    }


_write_lock = threading.Lock()


def _load_partial(partial_path: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    done_ids: set[str] = set()
    if not partial_path.exists():
        return rows, done_ids
    with partial_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            done_ids.add(r["seed_id"])
    return rows, done_ids


def score_all_seeds(workers: int = 1, partial_path: Optional[Path] = None,
                    time_budget_sec: Optional[float] = None) -> list[dict]:
    tasks: list[tuple[dict, str]] = []
    for source, rel in SEED_FILES:
        with open(ROOT / rel) as f:
            seeds = json.load(f)
        for s in seeds:
            tasks.append((s, source))

    rows: list[dict] = []
    done_ids: set[str] = set()
    if partial_path is not None:
        rows, done_ids = _load_partial(partial_path)
        if rows:
            print(f"  Resuming: {len(rows)} seeds already scored in {partial_path.name}")
    todo = [(s, src) for s, src in tasks if s["seed_id"] not in done_ids]
    if not todo:
        print("  All seeds already scored.")
        return rows

    partial_fp = partial_path.open("a") if partial_path else None
    start = time.time()
    budget_exhausted = False

    def _persist(row: dict) -> None:
        if partial_fp is None:
            return
        with _write_lock:
            partial_fp.write(json.dumps(row) + "\n")
            partial_fp.flush()

    try:
        if workers <= 1:
            for i, (s, source) in enumerate(todo):
                if time_budget_sec is not None and (time.time() - start) > time_budget_sec:
                    budget_exhausted = True
                    print(f"  Time budget {time_budget_sec}s exhausted at seed {i}/{len(todo)}")
                    break
                row = _score_one(s, source)
                if row is not None:
                    rows.append(row)
                    _persist(row)
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(todo)}] (+{len(rows)-len(done_ids)} new) elapsed {time.time()-start:.0f}s")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_score_one, s, src): (s, src) for s, src in todo}
                completed = 0
                for fut in as_completed(futs):
                    completed += 1
                    try:
                        row = fut.result()
                    except Exception as e:
                        print(f"  WARN: scoring failed: {e}")
                        continue
                    if row is not None:
                        rows.append(row)
                        _persist(row)
                    if completed % 5 == 0:
                        print(f"  [{completed}/{len(todo)}] elapsed {time.time()-start:.0f}s rows={len(rows)}")
                    if time_budget_sec is not None and (time.time() - start) > time_budget_sec:
                        budget_exhausted = True
                        print(f"  Time budget {time_budget_sec}s exhausted at {completed}/{len(todo)}")
                        for f in futs:
                            f.cancel()
                        break
    finally:
        if partial_fp is not None:
            partial_fp.close()

    if budget_exhausted:
        print(f"  PARTIAL: {len(rows)}/{len(tasks)} seeds scored. Re-run to continue.")
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


def write_report(rows: list[dict], dim_corrs: dict[str, dict], out_dir: Path, suffix: str = "") -> None:
    json_path = out_dir / f"seed_scorer_agreement{suffix}.json"
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
              "## Tooling state"]
    llm_skipped_vals = [r.get("llm_checks_skipped") for r in rows if "llm_checks_skipped" in r]
    if llm_skipped_vals and all(v == 0 for v in llm_skipped_vals):
        lines.append("Full 5-tool active: PHPCS (WordPress + WordPressVIPMinimum + Security) + PHPStan + regex + LLM-assisted (41 binary YES/NO checks per rubric §F.5).")
    elif llm_skipped_vals and all(v > 0 for v in llm_skipped_vals):
        lines.append("Deterministic 4-tool only: PHPCS (WordPress + WordPressVIPMinimum + Security) + PHPStan + regex. LLM-assisted checks (rubric §F.5) deferred — set RUBRIC_USE_LLM_CHECKS=1 to enable.")
    else:
        lines.append("Mixed run: some seeds scored with LLM checks, others without (see per-row llm_checks_skipped).")
    (out_dir / f"seed_scorer_agreement{suffix}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (LLM-call latency is dominant)")
    parser.add_argument("--output-suffix", default="", help="Append to output filename (e.g. '_llm')")
    parser.add_argument("--time-budget-sec", type=float, default=None,
                        help="Exit cleanly after this many seconds; resume on next run via partial.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from .partial.jsonl, skipping already-scored seeds")
    parser.add_argument("--emit-features", action="store_true",
                        help="Also write output/diagnostic/seed_scorer_features.jsonl "
                             "(one row per seed) for Phase 1a calibration.")
    parser.add_argument("--features-output",
                        default="output/diagnostic/seed_scorer_features.jsonl",
                        help="Path for --emit-features JSONL output.")
    args = parser.parse_args()

    out_dir = ROOT / "output" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "LLM ON" if os.environ.get("RUBRIC_USE_LLM_CHECKS") == "1" else "deterministic-only"
    print(f"Scoring seeds via rubric_scorer ({mode}, workers={args.workers}) ...")
    partial_path = out_dir / f"seed_scorer_agreement{args.output_suffix}.partial.jsonl" if args.resume or args.time_budget_sec else None
    rows = score_all_seeds(
        workers=args.workers,
        partial_path=partial_path,
        time_budget_sec=args.time_budget_sec,
    )
    print(f"Scored {len(rows)} seeds with at least 1 mapped dim.")
    dim_corrs = per_dim_correlation(rows)
    suffix = args.output_suffix
    write_report(rows, dim_corrs, out_dir, suffix=suffix)
    print(f"Wrote {out_dir}/seed_scorer_agreement{suffix}.json and .md")

    if args.emit_features:
        features_path = ROOT / args.features_output
        features_path.parent.mkdir(parents=True, exist_ok=True)
        with features_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {features_path} ({len(rows)} feature rows)")


if __name__ == "__main__":
    main()
