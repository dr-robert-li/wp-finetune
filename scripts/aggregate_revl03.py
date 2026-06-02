"""REVL-03 aggregator — reads the dispatched agents' results, computes the gate.

Reads output/eval_reasoning/revl03_claude_eval.jsonl (one JSON object per sample,
written by the orchestrating session's agents) and writes revl03_aggregate.json:
  - dimension_coverage_rate = mean over samples of
        count(true in dimension_coverage) / N_DIMS
    where N_DIMS is the per-sample dimension_coverage key count (the model's real
    rubric is 8 dims — see scripts.revl03_evaluator_agent.REVL03_DIMENSIONS — NOT a
    naive 9; i18n + error_handling are structurally absent from the prose output and
    are NOT in the denominator, consistent with the REVL-01 dim_map.json reconciliation).
  - score_reasoning_consistency_rate = mean over samples of
        count(true in consistency restricted to CLAIMED dims) / count(claimed dims)
    where claimed dims = dims marked true in that sample's dimension_coverage.
    A sample claiming ZERO dims is scored 0.0 (NOT count/0 — that would crash the gate).
  - mean_coherence = mean coherence (1-5)
  - pass = dimension_coverage_rate >= 0.80

No LLM API imported anywhere.

Usage:
  python -m scripts.aggregate_revl03 \
      --eval-jsonl output/eval_reasoning/revl03_claude_eval.jsonl \
      --aggregate-out output/eval_reasoning/revl03_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = "output/eval_reasoning/revl03_claude_eval.jsonl"
DEFAULT_AGG = "output/eval_reasoning/revl03_aggregate.json"
PASS_THRESHOLD = 0.80


def _truthy_count(d) -> int:
    return sum(1 for v in (d or {}).values() if v is True)


def aggregate(eval_jsonl: str, aggregate_out: str, threshold: float = PASS_THRESHOLD) -> dict:
    ep = PROJECT_ROOT / eval_jsonl if not os.path.isabs(eval_jsonl) else Path(eval_jsonl)
    samples = [json.loads(l) for l in open(ep) if l.strip()]
    n = len(samples)

    coverage_scores, consistency_scores, coherences = [], [], []
    for s in samples:
        cov = s.get("dimension_coverage", {}) or {}
        n_dims = len(cov)  # the model's real rubric is 8 dims; denominator = keys present
        coverage_scores.append((_truthy_count(cov) / n_dims) if n_dims else 0.0)

        claimed = [k for k, v in cov.items() if v is True]
        cons = s.get("score_reasoning_consistency", {}) or {}
        if claimed:  # div-by-zero guard: zero-claim sample -> 0.0, never count/0
            consistent = sum(1 for k in claimed if cons.get(k) is True)
            consistency_scores.append(consistent / len(claimed))
        else:
            consistency_scores.append(0.0)

        c = s.get("coherence")
        if isinstance(c, (int, float)):
            coherences.append(float(c))

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    dimension_coverage_rate = _mean(coverage_scores)
    result = {
        "n_samples": n,
        "dimension_coverage_rate": dimension_coverage_rate,
        "score_reasoning_consistency_rate": _mean(consistency_scores),
        "mean_coherence": _mean(coherences),
        "threshold": threshold,
        "pass": bool(n > 0 and dimension_coverage_rate >= threshold),
    }
    out_path = PROJECT_ROOT / aggregate_out if not os.path.isabs(aggregate_out) else Path(aggregate_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-03 aggregator")
    ap.add_argument("--eval-jsonl", default=DEFAULT_EVAL)
    ap.add_argument("--aggregate-out", default=DEFAULT_AGG)
    ap.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    args = ap.parse_args()
    res = aggregate(args.eval_jsonl, args.aggregate_out, args.threshold)
    print(f"[revl03] {json.dumps(res)}", file=sys.stderr)
    print(f"[revl03] REVL-03 {'PASS' if res['pass'] else 'FAIL'} "
          f"(dimension_coverage_rate {res['dimension_coverage_rate']:.3f} "
          f"vs >= {args.threshold})", file=sys.stderr)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
