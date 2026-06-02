"""REVL-05 — build the human-review pack (CoT-only) + sign-off sentinel mechanism.

Generates output/v1.2_human_review.md: a stratified sample of CoT judge responses from
the reasoning-merged model, side-by-side with the code under review, the model's scores,
the REVL-03 opaque-evaluator verdict, and the calibrated GT. The human reads each, then
appends `HUMAN_APPROVED: <ISO timestamp>` (or `HUMAN_REJECTED: <reason>`) to the file.
The merge-promotion step refuses to run without the sentinel (check_sentinel).

CoT-only (10 samples, not the original 20 mixed): REVL-06 is N/A — the model is judge-only
and emits no CtF `<corrected_code>`, so there is no CtF stream to review (see GATE-LEDGER).

Stratification (dual purpose — general quality sign-off AND adjudicating the REVL-03
MARGINAL result): samples are drawn across the REVL-03 dimension-coverage spectrum so the
reviewer sees both the thorough-prose majority and the terse-JSON failure mode that drags
the coverage rate toward its 0.80 floor:
  - 4 full-coverage (1.0) prose responses  — the 65/85 majority
  - 3 terse-JSON low-coverage (<=0.25)     — the mode driving REVL-03 marginal
  - 3 mid-coverage (0.25 < f < 1.0)        — borderline judgment calls

Usage:
  python -m scripts.build_human_review            # generate the pack
  python -m scripts.build_human_review --check    # verify sentinel present (exit 0/1)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.capture_reasoning_responses import extract_reasoning  # noqa: E402

CAPTURED = "output/eval_reasoning/reasoning_merged/captured_responses.jsonl"
REVL03_EVAL = "output/eval_reasoning/revl03_claude_eval.jsonl"
PAIRS = "output/eval_reasoning/reasoning_merged/eval_judge_results.pairs.jsonl"
OUT_MD = "output/v1.2_human_review.md"
SENTINEL_RE = re.compile(r"^HUMAN_(APPROVED|REJECTED):\s*\S", re.MULTILINE)


def _coverage_fraction(eval_row) -> float:
    d = eval_row.get("dimension_coverage", {}) or {}
    return (sum(1 for v in d.values() if v is True) / len(d)) if d else 0.0


def _load(path):
    p = PROJECT_ROOT / path if not os.path.isabs(path) else Path(path)
    return [json.loads(l) for l in open(p) if l.strip()]


def select_samples(captured, revl03, pairs):
    """Stratified CoT-only selection by REVL-03 coverage band (deterministic)."""
    cap = {c["example_idx"]: c for c in captured}
    rev = {r["sample_id"]: r for r in revl03}
    pid = {p["index"]: p for p in pairs}
    cots = sorted(i for i, c in cap.items() if c["task_type"] == "cot" and i in rev)

    full, low, mid = [], [], []
    for i in cots:
        f = _coverage_fraction(rev[i])
        if f >= 1.0:
            full.append(i)
        elif f <= 0.25:
            low.append(i)
        else:
            mid.append(i)
    # deterministic: take from the front of each sorted band
    picks = full[:4] + low[:3] + mid[:3]
    return picks, cap, rev, pid


def _num(x):
    return x if isinstance(x, (int, float)) else None


def build(out_md: str = OUT_MD) -> str:
    captured = _load(CAPTURED)
    revl03 = _load(REVL03_EVAL)
    pairs = _load(PAIRS)
    picks, cap, rev, pid = select_samples(captured, revl03, pairs)

    lines = []
    lines.append("# REVL-05 — Human Review Pack (v1.2 reasoning-merged)")
    lines.append("")
    lines.append("**Model:** `models/qwen3-30b-wp-30_70-reasoning-merged` "
                 "(merge-certified). **Generated:** by `scripts/build_human_review.py`.")
    lines.append("")
    lines.append("## What you are signing off")
    lines.append("")
    lines.append("1. **REVL-05 (primary):** is the reasoning quality of this judge model "
                 "good enough to declare v1.2 complete and promote the merge?")
    lines.append("2. **REVL-03 marginal (secondary):** the automated opaque-evaluator gate "
                 "scored dimension-coverage **0.814** with a 95% CI **[0.751, 0.871]** that "
                 "straddles the 0.80 floor — a statistical tie. Your read on these samples "
                 "is the tiebreaker. The samples below are stratified to show BOTH the "
                 "thorough-prose majority AND the terse-JSON mode that drags the rate down.")
    lines.append("")
    lines.append(f"**Sample:** {len(picks)} CoT judge responses (CtF excluded — REVL-06 N/A, "
                 "model is judge-only). Stratified by REVL-03 coverage band.")
    lines.append("")
    lines.append("## How to sign off")
    lines.append("")
    lines.append("After reading all samples, append ONE line to the END of this file and save:")
    lines.append("")
    lines.append("- Approve: `HUMAN_APPROVED: 2026-06-02T14:00:00Z` (use a real ISO timestamp)")
    lines.append("- Reject:  `HUMAN_REJECTED: <one-line reason>`")
    lines.append("")
    lines.append("Then run `python -m scripts.build_human_review --check` (exit 0 = sentinel "
                 "present). The merge-promotion step refuses to run without it.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for n, idx in enumerate(picks, 1):
        c = cap[idx]
        r = rev[idx]
        p = pid.get(idx, {})
        f = _coverage_fraction(r)
        band = "FULL" if f >= 1.0 else ("TERSE/LOW" if f <= 0.25 else "MID")
        reasoning = extract_reasoning(c.get("response", "") or "")
        prompt = c.get("prompt", "") or ""
        mo = _num(p.get("model_overall"))
        gc = _num(p.get("gt_canonical"))
        gt = _num(p.get("gt_teacher"))

        lines.append(f"## Sample {n}/{len(picks)} — example_idx {idx} "
                     f"[{band} coverage {f:.3f}]")
        lines.append("")
        cons = r.get("score_reasoning_consistency", {}) or {}
        claimed = [k for k, v in (r.get("dimension_coverage") or {}).items() if v is True]
        cons_rate = (sum(1 for k in claimed if cons.get(k) is True) / len(claimed)) if claimed else 0.0
        lines.append(f"- REVL-03 evaluator: coherence {r.get('coherence')}/5, "
                     f"dimension_coverage {f:.3f}, "
                     f"score-reasoning consistency {cons_rate:.2f} (on claimed dims)")
        lines.append(f"- Scores: model_overall {mo}, gt_calibrated {round(gc,1) if gc is not None else None}, "
                     f"gt_teacher {gt}")
        lines.append("")
        lines.append("### Code under review")
        lines.append("")
        lines.append("```")
        lines.append(prompt.strip()[:2000])
        lines.append("```")
        lines.append("")
        lines.append("### Model reasoning + judgment")
        lines.append("")
        lines.append("```")
        lines.append(reasoning.strip()[:3000])
        lines.append("```")
        lines.append("")
        lines.append("### REVL-03 opaque-evaluator dimension_coverage")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.get("dimension_coverage"), indent=2))
        lines.append("```")
        lines.append("")
        lines.append("**Your note (optional):** ____________________")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Sign-off")
    lines.append("")
    lines.append("<!-- Append HUMAN_APPROVED: <ISO ts>  OR  HUMAN_REJECTED: <reason> below -->")
    lines.append("")

    op = PROJECT_ROOT / out_md if not os.path.isabs(out_md) else Path(out_md)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text("\n".join(lines))
    return str(op)


def check_sentinel(out_md: str = OUT_MD) -> bool:
    op = PROJECT_ROOT / out_md if not os.path.isabs(out_md) else Path(out_md)
    if not op.exists():
        return False
    return SENTINEL_RE.search(op.read_text()) is not None


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-05 human review pack")
    ap.add_argument("--out", default=OUT_MD)
    ap.add_argument("--check", action="store_true",
                    help="Check the sign-off sentinel is present; exit 0 if so, 1 if not.")
    args = ap.parse_args()
    if args.check:
        ok = check_sentinel(args.out)
        print("HUMAN_APPROVED/REJECTED sentinel: "
              + ("PRESENT" if ok else "ABSENT"), file=sys.stderr)
        return 0 if ok else 1
    path = build(args.out)
    print(f"[revl05] review pack written -> {path}", file=sys.stderr)
    print("[revl05] human must append HUMAN_APPROVED: <ts> (or HUMAN_REJECTED: <reason>) "
          "then run --check.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
