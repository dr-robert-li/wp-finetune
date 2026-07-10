"""REVL-01 GT-readiness preflight — HARD gate before the W1-W6 cascade.

Council-mandated (2026-05-30): before paying vLLM boots, verify the CANONICAL GT
(rubric calibrated_overall) is available + non-degenerate on the eval set, and
report teacher-GT coverage. Model-side parse coverage is measured during the
cascade (needs the served model); this preflight covers the GT side, which is
deterministic + CPU-only.

Exit 0 = GT ready (calibrated coverage OK + variance preflight passes).
Exit 1 = GT degenerate/missing -> do NOT launch cascade (REVL-01A would be invalid).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_judge import _extract_code_from_judge_prompt, _extract_gt_from_assistant, _gt_variance_ok  # noqa: E402
from eval.output_parsers import load_dim_map  # noqa: E402
from eval.rubric_scorer import score_code  # noqa: E402

DEFAULT_DATASET = "data/reasoning_dataset/openai_val.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-01 GT-readiness preflight (HARD)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="min fraction of judge rows with calibrated_overall GT")
    args = ap.parse_args()

    dm = load_dim_map()
    pf = dm["gt_variance_preflight"]

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    judge = [r for r in rows
             if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    if args.limit:
        judge = judge[: args.limit]

    calibrated_gt = []
    n_calibrated = 0
    n_teacher = 0
    n_raw_only = 0
    for r in judge:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        code = _extract_code_from_judge_prompt(user)
        rub = score_code(code)
        if rub.calibrated_overall is not None:
            n_calibrated += 1
            calibrated_gt.append(float(rub.calibrated_overall))
        else:
            n_raw_only += 1
        if _extract_gt_from_assistant(r["messages"]) is not None:
            n_teacher += 1

    n = len(judge)
    coverage = n_calibrated / n if n else 0.0
    var_ok, var_detail = _gt_variance_ok(calibrated_gt, pf["min_stdev"], pf["min_unique_ranks"])

    print(f"[preflight] dataset={args.dataset} judge_rows={n}")
    print(f"[preflight] calibrated_overall coverage: {n_calibrated}/{n} = {coverage:.2%} "
          f"(min {args.min_coverage:.0%})")
    print(f"[preflight] rows missing calibrated (would be EXCLUDED): {n_raw_only}")
    print(f"[preflight] teacher-GT (dataset target) coverage: {n_teacher}/{n} (REVL-01B SOFT)")
    print(f"[preflight] canonical GT variance: {var_detail}")

    coverage_ok = coverage >= args.min_coverage
    ok = coverage_ok and var_ok
    print(f"[preflight] coverage_ok={coverage_ok} variance_ok={var_ok} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[preflight] DO NOT LAUNCH CASCADE — REVL-01A canonical GT not ready.")
        if not coverage_ok:
            print(f"  calibrated coverage {coverage:.2%} < {args.min_coverage:.0%}")
        if not var_ok:
            print(f"  GT variance degenerate (rank-collapse risk): {var_detail}")
        return 1
    print("[preflight] REVL-01A canonical GT ready. Cascade may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
