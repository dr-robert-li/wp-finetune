#!/usr/bin/env python
"""P4 REVL-05 invalid-PHP sentinel CHECKER (project venv).

Pairs the sentinel dataset (all should_fail:true) with Tinker-captured judge responses
(by filtered-list index) and counts FALSE-PASSES. Gate: ZERO false-passes.

False-pass definition (the model approved clearly-broken code):
  - judge_output verdict == "PASS"  (the REVL-05 critical signal), OR
  - verdict absent AND overall_score is high (>= --pass-score, default 60).

Usage:
  python scripts/check_invalid_php_sentinel.py \
      --dataset data/reasoning_dataset/invalid_php_sentinel.jsonl \
      --responses-jsonl output/eval_reasoning/reasoning_v2_tinker/sentinel_responses.jsonl \
      --output output/format_stability/invalid_php_sentinel_summary.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.output_parsers import parse_judge_scores  # noqa: E402

_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"?(PASS|FAIL|NEEDS_REVISION|FAILED|PASSED)"?', re.IGNORECASE)


def _verdict(text: str):
    m = _VERDICT_RE.search(text or "")
    if not m:
        return None
    v = m.group(1).upper()
    if v.startswith("PASS"):
        return "PASS"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/reasoning_dataset/invalid_php_sentinel.jsonl")
    ap.add_argument("--responses-jsonl", required=True)
    ap.add_argument("--output", default="output/format_stability/invalid_php_sentinel_summary.json")
    ap.add_argument("--pass-score", type=float, default=60.0,
                    help="overall_score >= this (with no explicit verdict) counts as a pass")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    responses = {}
    for line in open(args.responses_jsonl):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "__provenance__" in rec:
            continue
        responses[int(rec["index"])] = rec.get("response", "")

    results = []
    false_pass = 0
    unparseable = 0
    correct_fail = 0
    for i, row in enumerate(rows):
        text = responses.get(i, "")
        verdict = _verdict(text)
        parsed = parse_judge_scores(text, "auto") if text else None
        overall = parsed.get("overall") if parsed else None
        if verdict == "PASS":
            is_false_pass = True
        elif verdict == "FAIL":
            is_false_pass = False
        elif overall is not None and overall >= args.pass_score:
            is_false_pass = True
        else:
            is_false_pass = False
        if not text or (verdict is None and overall is None):
            unparseable += 1
        if is_false_pass:
            false_pass += 1
        elif verdict == "FAIL":
            correct_fail += 1
        results.append({
            "index": i,
            "defect_category": row.get("metadata", {}).get("defect_category"),
            "verdict": verdict,
            "overall_score": overall,
            "false_pass": is_false_pass,
            "response_head": (text or "")[:160],
        })

    n = len(rows)
    summary = {
        "n": n, "false_pass": false_pass, "correct_fail": correct_fail,
        "unparseable": unparseable, "pass_score_threshold": args.pass_score,
        "gate_pass": false_pass == 0,
        "rows": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[sentinel] n={n} false_pass={false_pass} correct_fail={correct_fail} "
          f"unparseable={unparseable} -> GATE {'PASS' if false_pass == 0 else 'FAIL'}",
          file=sys.stderr)
    for r in results:
        if r["false_pass"]:
            print(f"  FALSE-PASS [{r['defect_category']}] verdict={r['verdict']} "
                  f"overall={r['overall_score']}", file=sys.stderr)
    print(f"[sentinel] summary -> {out}", file=sys.stderr)
    return 0 if false_pass == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
