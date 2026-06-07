# v1.2 reasoning-judge verdict policy (corrective branch)

**Context:** P4 (`wp-reasoning-v2`) fixed the terse collapse and passed REVL-01A but
**failed the invalid-PHP sentinel 4/24** — it approved a real `fn()->` parse error (@51),
two clearly-broken snippets (@57), and a fabricated `wp_*` call (@100). The two failures are
distinct and need distinct fixes.

## The two bugs

1. **Boundary leniency (@51–57 → PASS).** Teacher PASS targets run down to `overall_score`
   49, so the model faithfully learned a lenient PASS boundary. This is a *threshold* problem.
2. **Scoring blind spot (@100).** The model gave fabricated/non-existent-API code 10/10 — no
   threshold catches a 100. Only *training* on invalid-PHP/fabricated negatives fixes this.

## Policy

**Effective verdict = PASS iff `overall_score >= 70` AND no auto-FAIL defect class is
present; otherwise FAIL.**

Auto-FAIL defect classes (disqualifying regardless of other quality):
- syntax / parse error (won't compile)
- fabricated / non-existent WordPress API (fatal at runtime)
- out-of-context fatal `$this` / `self` / `parent` (no class scope)
- unsanitized SQL injection or unescaped output (XSS) on request data
- missing nonce/capability on a state-changing handler

## How the policy is enforced (two layers, defense in depth)

- **Training (model-internal):** 30 `should_fail` negatives
  (`scripts/build_reasoning_negatives.py`) teach the auto-FAIL classes to score the relevant
  dimensions 1–4 → `overall_score < 40` → `verdict: FAIL`. This fixes the scoring blind spot
  and pulls the @51–57 cases down into clear-FAIL territory.
- **Post-hoc (deterministic):** at evaluation/consumption, `overall_score < 70` ⇒ effective
  FAIL. This guarantees boundary consistency without flipping teacher verdicts (which would
  contradict their PASS-leaning prose). Downstream consumes `overall_score`, not the raw
  `verdict` token, so this is a policy layer, not a model hack.

## Why not relabel teacher rows?

The overlap is small (20 PASS rows < 70, 5 FAIL ≥ 70). Flipping a teacher PASS@65 to FAIL
while leaving its "acceptable, minor issues" prose intact injects a prose↔verdict
contradiction — worse training signal than the leniency it fixes. The post-hoc threshold
achieves the same boundary consistency cleanly and verifiably.

## Gates (two-sided — a FAIL-everything model must NOT pass)

1. **FS terse** (cot+ctf): rate ≤10%, Wilson-upper ≤15%.
2. **REVL-01A** overall Spearman ≥ merged-v2 baseline (0.171).
3. **Invalid-PHP sentinel** (`scripts/check_invalid_php_sentinel.py`, 24 held-out): zero
   effective-false-pass.
4. **Verdict confusion** (`scripts/check_verdict_confusion.py`, 121 val rows): recall on
   teacher-FAIL rises vs P4 **without** materially worsening false-FAIL on teacher-PASS
   (over-strictness guard). P4 baseline: false-FAIL 40.3%, recall 63.8% (policy).

Promotion requires all four.
