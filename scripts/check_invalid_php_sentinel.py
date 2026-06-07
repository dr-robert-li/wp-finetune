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
                    help="overall_score >= this (with no explicit verdict) counts as a RAW pass")
    ap.add_argument("--threshold", type=float, default=70.0,
                    help="v1.2 verdict policy: overall < threshold -> effective FAIL")
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
    raw_false_pass = 0       # model APPROVED (verdict PASS or no-verdict+high score)
    policy_false_pass = 0    # effective verdict PASS under the v1.2 policy
    unparseable = 0
    correct_fail = 0
    for i, row in enumerate(rows):
        text = responses.get(i, "")
        verdict = _verdict(text)
        parsed = parse_judge_scores(text, "auto") if text else None
        overall = parsed.get("overall") if parsed else None
        # RAW: did the model's own output approve the broken code?
        if verdict == "PASS":
            raw_fp = True
        elif verdict == "FAIL":
            raw_fp = False
        elif overall is not None and overall >= args.pass_score:
            raw_fp = True
        else:
            raw_fp = False
        # POLICY: effective verdict = FAIL if model FAIL OR overall < threshold.
        if verdict == "FAIL":
            policy_fp = False
        elif overall is not None:
            policy_fp = overall >= args.threshold
        else:
            policy_fp = verdict == "PASS"  # no score, model said PASS -> approved
        if not text or (verdict is None and overall is None):
            unparseable += 1
        raw_false_pass += raw_fp
        policy_false_pass += policy_fp
        if verdict == "FAIL" or (overall is not None and overall < args.threshold):
            correct_fail += 1
        results.append({
            "index": i,
            "defect_category": row.get("metadata", {}).get("defect_category"),
            "verdict": verdict,
            "overall_score": overall,
            "raw_false_pass": raw_fp,
            "policy_false_pass": policy_fp,
            "response_head": (text or "")[:160],
        })

    n = len(rows)
    summary = {
        "n": n, "raw_false_pass": raw_false_pass, "policy_false_pass": policy_false_pass,
        "correct_fail": correct_fail, "unparseable": unparseable,
        "pass_score_threshold": args.pass_score, "policy_threshold": args.threshold,
        "gate_pass": policy_false_pass == 0,
        "rows": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[sentinel] n={n} RAW_false_pass={raw_false_pass} "
          f"POLICY_false_pass={policy_false_pass} (thr={args.threshold}) "
          f"correct_fail={correct_fail} unparseable={unparseable} -> "
          f"GATE {'PASS' if policy_false_pass == 0 else 'FAIL'}", file=sys.stderr)
    for r in results:
        if r["raw_false_pass"] or r["policy_false_pass"]:
            tag = "POLICY+RAW" if r["policy_false_pass"] else "raw-only(boundary)"
            print(f"  [{tag}] {r['defect_category']} verdict={r['verdict']} "
                  f"overall={r['overall_score']}", file=sys.stderr)
    print(f"[sentinel] summary -> {out}", file=sys.stderr)
    return 0 if policy_false_pass == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
