"""Phase 0.11: Extract deterministic PASS-anchor pool for Phase 1 calibration.

Subsamples the phase1_extraction/output/passed/ pool (already Claude-judged at
7-10 per dim) and re-scores each via the 4-tool rubric_scorer (PHPCS x3 +
PHPStan + regex). Keeps only functions where:
  - rubric_scorer.overall >= 90
  - PHPCS WordPress + WordPressVIPMinimum + Security all clean (0 errors)
  - PHPStan clean (no errors)
  - No negative-regex hits

Output: output/diagnostic/pass_anchors.jsonl  — fills the 90-100 calibration
range that human + UGC + boundary seeds (all FAIL-band) cannot cover.

Usage:
    python -m scripts.extract_pass_anchors --target-anchors 500 --sample-pool 1500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.rubric_scorer import score_code, run_phpcs, run_phpstan

PASSED_DIR = ROOT / "data" / "phase1_extraction" / "output" / "passed"


def iter_passed_functions() -> Iterator[dict]:
    for fn in sorted(os.listdir(PASSED_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(PASSED_DIR / fn) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue
        if isinstance(d, dict):
            d = [d]
        for item in d:
            yield item


def stratified_sample(items: list[dict], n: int) -> list[dict]:
    """Stratify by Claude overall (8/9/10 buckets) then random-sample within."""
    buckets: dict[int, list[dict]] = {8: [], 9: [], 10: []}
    for it in items:
        scores = (it.get("assessment", {}) or {}).get("scores", {})
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        if not vals:
            continue
        bucket = round(sum(vals) / len(vals))
        if bucket in buckets:
            buckets[bucket].append(it)
    per_bucket = max(1, n // 3)
    out: list[dict] = []
    for b, pool in buckets.items():
        random.shuffle(pool)
        out.extend(pool[:per_bucket])
    random.shuffle(out)
    return out[:n]


def is_deterministic_anchor(code: str, min_overall: float = 90.0) -> tuple[bool, dict]:
    """Score one snippet and decide whether it qualifies as a 4-tool PASS anchor."""
    diagnostics: dict = {}
    # Cheap pre-filter — full score_code already runs all 4 tools
    sc = score_code(code)
    diagnostics["overall"] = sc.overall
    diagnostics["grade"] = sc.grade
    diagnostics["triggered_check_count"] = sum(len(v) for v in sc.triggered_checks.values())
    diagnostics["triggered_checks"] = sc.triggered_checks
    diagnostics["dimension_scores"] = sc.dimension_scores
    diagnostics["dimension_na"] = sc.dimension_na
    diagnostics["floor_rules_applied"] = sc.floor_rules_applied
    diagnostics["llm_checks_skipped"] = sc.llm_checks_skipped
    if sc.overall < min_overall:
        diagnostics["reject_reason"] = f"overall {sc.overall:.1f} < {min_overall}"
        return False, diagnostics
    # Re-confirm each tool independently clean (paranoia — score_code aggregates)
    for standard in ("WordPress", "WordPressVIPMinimum", "Security"):
        r = run_phpcs(code, standard=standard)
        if r.get("_unavailable"):
            diagnostics["reject_reason"] = f"phpcs {standard} unavailable"
            return False, diagnostics
        errs = r.get("totals", {}).get("errors", 0)
        if errs:
            diagnostics["reject_reason"] = f"phpcs {standard} {errs} errors"
            return False, diagnostics
    ps = run_phpstan(code)
    if ps.get("_unavailable"):
        diagnostics["reject_reason"] = "phpstan unavailable"
        return False, diagnostics
    n_errs = (ps.get("totals", {}) or {}).get("file_errors", 0)
    if n_errs:
        diagnostics["reject_reason"] = f"phpstan {n_errs} errors"
        return False, diagnostics
    diagnostics["reject_reason"] = None
    return True, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-anchors", type=int, default=500,
                        help="Stop after this many anchors qualify")
    parser.add_argument("--sample-pool", type=int, default=1500,
                        help="Stratified sample size from passed pool")
    parser.add_argument("--min-overall", type=float, default=90.0,
                        help="Minimum rubric overall to qualify as anchor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output",
                        default="output/diagnostic/pass_anchors.jsonl")
    parser.add_argument("--emit-features", action="store_true",
                        help="Persist triggered_checks_flat + dimension_na + floor_rules_applied "
                             "+ rubric_triggered_check_count + llm_checks_skipped for calibration.")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Loading passed pool from {PASSED_DIR} ...")
    all_items = list(iter_passed_functions())
    print(f"  Loaded {len(all_items)} functions")

    sampled = stratified_sample(all_items, n=args.sample_pool)
    print(f"  Stratified-sampled {len(sampled)} candidates")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_anchors = 0
    n_rejected = 0
    reject_reasons: dict[str, int] = {}
    with out_path.open("w") as f:
        for i, item in enumerate(sampled):
            if n_anchors >= args.target_anchors:
                break
            code = item.get("body") or item.get("code")
            if not code:
                continue
            ok, diag = is_deterministic_anchor(code, min_overall=args.min_overall)
            if ok:
                anchor = {
                    "function_name": item.get("function_name"),
                    "source_repo": item.get("source_repo"),
                    "source_file": item.get("source_file"),
                    "training_tags": item.get("training_tags", []),
                    "code": code,
                    "rubric_overall": diag["overall"],
                    "rubric_dim_scores": diag["dimension_scores"],
                    "claude_assessment": item.get("assessment", {}),
                }
                if args.emit_features:
                    triggered_flat = sorted({
                        cid for ids in diag["triggered_checks"].values() for cid in ids
                    })
                    anchor["triggered_checks_flat"] = triggered_flat
                    anchor["dimension_na"] = list(diag["dimension_na"])
                    anchor["floor_rules_applied"] = list(diag["floor_rules_applied"])
                    anchor["rubric_triggered_check_count"] = diag["triggered_check_count"]
                    anchor["llm_checks_skipped"] = diag["llm_checks_skipped"]
                f.write(json.dumps(anchor) + "\n")
                f.flush()
                n_anchors += 1
                if n_anchors % 25 == 0:
                    print(f"  [{i+1}/{len(sampled)}] {n_anchors} anchors qualified")
            else:
                n_rejected += 1
                reason = diag.get("reject_reason", "unknown")
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    print(f"\nDone. {n_anchors} anchors written to {out_path}")
    print(f"Rejected {n_rejected} candidates.")
    print("Top reject reasons:")
    for r, n in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:5d}  {r}")


if __name__ == "__main__":
    main()
