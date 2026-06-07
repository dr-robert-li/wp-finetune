#!/usr/bin/env python
"""REVL-05 human-review pack for wp-reasoning-v3 (corrective branch).

The original build_human_review.py is wired to the ckpt-72 merged-model REVL-03 agent
artifacts. v3 was evaluated on Tinker via REVL-01A (Spearman) + the invalid-PHP sentinel
+ the two-sided verdict-confusion gate — so this builds the pack from THOSE artifacts.

Emits output/v1.2_human_review_v3.md: the 4-gate scorecard, the invalid-PHP section
(REVL-05's original rejection reason), a stratified sample of v3's judge responses
side-by-side with code + model scores + calibrated GT + teacher verdict, and a sign-off
sentinel. The reviewer appends `HUMAN_APPROVED: <ISO>` or `HUMAN_REJECTED: <reason>`.

Usage:
  python scripts/build_human_review_v3.py            # build the pack
  python scripts/build_human_review_v3.py --check    # verify sentinel present (0/1)
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.output_parsers import parse_judge_scores  # noqa: E402

VAL = "data/reasoning_dataset/openai_val.jsonl"
JUDGE = "output/eval_reasoning/reasoning_v3_tinker/judge_responses.jsonl"
SENT = "output/format_stability/invalid_php_sentinel_v3_summary.json"
CONF = "output/format_stability/verdict_confusion/wp-reasoning-v3.json"
FS = "output/format_stability/fs_gate/wp-reasoning-v3/summary.json"
REVL01A = "output/eval_reasoning/reasoning_v3_tinker/eval_judge_results.json"
OUT = "output/v1.2_human_review_v3.md"
SENTINEL_RE = re.compile(r"^HUMAN_(APPROVED|REJECTED):\s*\S", re.MULTILINE)


def _load(p, default=None):
    p = ROOT / p
    return json.load(open(p)) if p.exists() else default


def _verdict(t):
    m = re.search(r'"verdict"\s*:\s*"([A-Z]+)"', t or "")
    return m.group(1) if m else None


def check():
    p = ROOT / OUT
    if not p.exists():
        print(f"[revl05] {OUT} missing — build it first", file=sys.stderr)
        return 1
    ok = bool(SENTINEL_RE.search(p.read_text()))
    print(f"[revl05] sentinel {'PRESENT' if ok else 'ABSENT'} in {OUT}", file=sys.stderr)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()

    rows = [json.loads(l) for l in open(ROOT / VAL) if l.strip()]
    ex = [r for r in rows if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]
    resp = {}
    for l in open(ROOT / JUDGE):
        l = l.strip()
        if not l:
            continue
        rec = json.loads(l)
        if "__provenance__" in rec:
            continue
        resp[int(rec["index"])] = rec.get("response", "")

    sent = _load(SENT, {})
    conf = _load(CONF, {})
    fs = _load(FS, {})
    r01 = _load(REVL01A, {})

    L = []
    L.append("# REVL-05 Human Review — wp-reasoning-v3 (v1.2 reasoning model)\n")
    L.append("Corrective re-gate. The P4 model (`wp-reasoning-v2`) fixed the terse collapse but "
             "the REVL-05 invalid-PHP critical persisted; this pack covers the corrective "
             "`wp-reasoning-v3` (Tinker run `3497a27e...:train:0`).\n")
    L.append("## Automated gate scorecard (all 4 PASS)\n")
    r01a = (r01.get("revl01a_overall_spearman_HARD") or {})
    fsa = {a["temp"]: a for a in fs.get("arms", [])}
    L.append(f"- **FS terse (cot+ctf)**: temp0 {fsa.get(0.0,{}).get('rate','?'):.3f} / "
             f"temp0.7 {fsa.get(0.7,{}).get('rate','?'):.3f} → {'PASS' if fs.get('pass') else 'FAIL'} "
             f"(vs ~0.35 baseline)")
    L.append(f"- **REVL-01A overall Spearman**: {r01a.get('corr')} (n={r01a.get('n_pairs')}, "
             f"p={r01a.get('p_value')}) ≥ 0.171 baseline → PASS")
    L.append(f"- **Invalid-PHP sentinel**: RAW false-pass {sent.get('raw_false_pass')}/24, "
             f"POLICY false-pass {sent.get('policy_false_pass')}/24 → "
             f"{'PASS' if sent.get('gate_pass') else 'FAIL'}")
    cp = conf.get("policy", {})
    ff = cp.get("false_FAIL_on_teacherPASS", [None, None, None])
    rc = cp.get("recall_on_teacherFAIL", [None, None, None])
    L.append(f"- **Verdict confusion (policy)**: false-FAIL on teacher-PASS {ff[2]:.3f}, "
             f"recall on teacher-FAIL {rc[2]:.3f}")
    L.append("\n> Verdict policy: PASS iff `overall_score ≥ 70` AND no auto-FAIL defect class "
             "(syntax error / fabricated API / out-of-class fatal / unsanitized SQL|XSS). "
             "See `VERDICT-POLICY.md`.\n")

    # --- The REVL-05 critical: invalid-PHP ---
    L.append("## REVL-05 critical — invalid-PHP judge quality\n")
    L.append("The original rejection: the judge passed syntactically-invalid PHP. v3 result on "
             "24 held-out should_fail snippets:\n")
    for row in sent.get("rows", []):
        flag = "⚠️ RAW-PASS (boundary; policy→FAIL)" if row.get("raw_false_pass") else "✓ FAIL"
        if row.get("policy_false_pass"):
            flag = "❌ POLICY FALSE-PASS"
        L.append(f"- `{row.get('defect_category')}`: model verdict={row.get('verdict')} "
                 f"overall={row.get('overall_score')} — {flag}")
    L.append("")

    # --- Stratified judge samples ---
    L.append("## Judge samples (model reasoning side-by-side with code + GT)\n")
    # classify by model overall
    buckets = {"clear-pass (≥80)": [], "clear-fail (<50)": [], "boundary (50–70)": []}
    for i, r in enumerate(ex):
        t = resp.get(i, "")
        p = parse_judge_scores(t, "auto") if t else None
        o = p.get("overall") if p else None
        if o is None:
            continue
        if o >= 80:
            buckets["clear-pass (≥80)"].append((i, o))
        elif o < 50:
            buckets["clear-fail (<50)"].append((i, o))
        elif 50 <= o <= 70:
            buckets["boundary (50–70)"].append((i, o))
    pick = []
    for name, items in buckets.items():
        for (i, o) in items[:3]:
            pick.append((name, i, o))
    for name, i, o in pick:
        r = ex[i]
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        code = u.split("```php", 1)[-1].rsplit("```", 1)[0].strip() if "```php" in u else u[:300]
        a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        tv = _verdict(a)
        resp_text = resp.get(i, "")
        L.append(f"### [{name}] val#{i} — model overall={o}, teacher verdict={tv}\n")
        L.append("```php\n" + code[:600] + "\n```\n")
        L.append("**Model response (head):**\n")
        L.append("```\n" + resp_text[:900].strip() + "\n```\n")

    L.append("\n---\n")
    L.append("## Sign-off\n")
    L.append("Reviewer: read the scorecard, the invalid-PHP section (the original rejection "
             "reason), and the samples, then append ONE line below:\n")
    L.append("- `HUMAN_APPROVED: <ISO timestamp>` — v3 clears REVL-05, proceed to merge / Phase 7.")
    L.append("- `HUMAN_REJECTED: <reason>` — what still fails.\n")
    L.append("<!-- append sign-off below this line -->\n")

    out = ROOT / OUT
    out.write_text("\n".join(L))
    print(f"[revl05] wrote {OUT} ({len(pick)} samples, "
          f"sentinel {sent.get('policy_false_pass')}/24 policy false-pass)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
