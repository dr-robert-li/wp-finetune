#!/usr/bin/env python
"""Two-sided judge-verdict gate: confusion matrix on the val wp_judge rows.

The invalid-PHP sentinel is one-sided (all should_fail) — a FAIL-everything model passes
it trivially while destroying the judge. This guards the OTHER direction: on the 121 val
wp_judge rows (which carry the teacher's PASS/FAIL verdict), does the corrective model
catch teacher-FAILs WITHOUT over-failing teacher-PASSes?

Reports BOTH:
  - raw  : the model's emitted `verdict` token.
  - policy: effective verdict under the v1.2 policy (FAIL if overall_score < --threshold
            OR model verdict==FAIL; PASS only if overall>=threshold and not FAIL).

Key metric for promotion (don't regress): false-FAIL rate on teacher-PASS rows
(over-strictness) must stay low; recall on teacher-FAIL should rise vs the P4 model.

Usage:
  python scripts/check_verdict_confusion.py \
      --dataset data/reasoning_dataset/openai_val.jsonl \
      --responses-jsonl output/eval_reasoning/reasoning_v3_tinker/judge_responses.jsonl \
      --output output/format_stability/verdict_confusion/wp-reasoning-v3.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.output_parsers import parse_judge_scores  # noqa: E402

_V = re.compile(r'"verdict"\s*:\s*"?(PASS|FAIL|PASSED|FAILED)"?', re.IGNORECASE)


def _verdict(text):
    m = _V.search(text or "")
    if not m:
        return None
    return "PASS" if m.group(1).upper().startswith("PASS") else "FAIL"


def _overall(text):
    p = parse_judge_scores(text, "auto") if text else None
    return p.get("overall") if p else None


def _effective(v, o, thr):
    """Policy: FAIL if model says FAIL or overall<thr; PASS if overall>=thr and not FAIL."""
    if v == "FAIL":
        return "FAIL"
    if o is not None:
        return "PASS" if o >= thr else "FAIL"
    if v == "PASS":
        return "PASS"
    return "UNK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--responses-jsonl", required=True)
    ap.add_argument("--threshold", type=float, default=70.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    examples = [r for r in rows
                if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    responses = {}
    for line in open(args.responses_jsonl):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "__provenance__" in rec:
            continue
        responses[int(rec["index"])] = rec.get("response", "")

    def teacher_verdict(r):
        a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        return _verdict(a)

    cm_raw = {}  # (teacher, model_raw) -> count
    cm_pol = {}
    rows_out = []
    for i, r in enumerate(examples):
        tv = teacher_verdict(r)
        text = responses.get(i, "")
        mv = _verdict(text)
        ov = _overall(text)
        ev = _effective(mv, ov, args.threshold)
        cm_raw[(tv, mv)] = cm_raw.get((tv, mv), 0) + 1
        cm_pol[(tv, ev)] = cm_pol.get((tv, ev), 0) + 1
        rows_out.append({"index": i, "teacher": tv, "model_raw": mv,
                         "overall": ov, "model_policy": ev})

    def rate(cm, teacher, model):
        tot = sum(v for (t, _), v in cm.items() if t == teacher)
        hit = cm.get((teacher, model), 0)
        return (hit, tot, (hit / tot if tot else float("nan")))

    summary = {
        "threshold": args.threshold,
        "n": len(examples),
        "teacher_dist": {v: sum(c for (t, _), c in cm_raw.items() if t == v)
                         for v in ("PASS", "FAIL", None)},
        "raw": {
            "false_FAIL_on_teacherPASS": rate(cm_raw, "PASS", "FAIL"),
            "recall_on_teacherFAIL": rate(cm_raw, "FAIL", "FAIL"),
            "confusion": {f"{t}->{m}": c for (t, m), c in sorted(cm_raw.items(), key=lambda x: str(x[0]))},
        },
        "policy": {
            "false_FAIL_on_teacherPASS": rate(cm_pol, "PASS", "FAIL"),
            "recall_on_teacherFAIL": rate(cm_pol, "FAIL", "FAIL"),
            "confusion": {f"{t}->{m}": c for (t, m), c in sorted(cm_pol.items(), key=lambda x: str(x[0]))},
        },
        "rows": rows_out,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    rfp = summary["policy"]["false_FAIL_on_teacherPASS"]
    rec = summary["policy"]["recall_on_teacherFAIL"]
    print(f"[confusion] teacher_dist={summary['teacher_dist']}", file=sys.stderr)
    print(f"[confusion] POLICY false-FAIL on teacher-PASS = {rfp[0]}/{rfp[1]} = {rfp[2]:.3f} "
          f"(over-strictness guard)", file=sys.stderr)
    print(f"[confusion] POLICY recall on teacher-FAIL      = {rec[0]}/{rec[1]} = {rec[2]:.3f}", file=sys.stderr)
    print(f"[confusion] -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
