---
status: awaiting_human_verify
trigger: "eval-na-inflation-and-triage-formula"
created: 2026-04-06T00:00:00Z
updated: 2026-04-06T00:10:00Z
---

## Current Focus

hypothesis: CONFIRMED - Three separate issues in eval pipeline produce misleading metrics
test: Applying targeted fixes per specification
expecting: After fixes, metrics are transparent about N/A dimensions and triage uses two-axis ranking
next_action: Apply fixes to eval_gen.py, triage_ratios.py, update tests, update CHANGELOG

## Symptoms

expected: Eval metrics should be transparent about N/A dimensions and not inflate scores. Triage should not collapse orthogonal metrics into one number.
actual: Three issues inflate or obscure eval results:
  Issue 2 (N/A inflation in eval_gen.py): Dimensions marked N/A are excluded from pass_rate_8 and overall mean. Code testable on only 2/9 dimensions gets 100% pass rate. security_pass_rate defaults to 1.0 when no security code detected.
  Issue 3 (Overall score weight redistribution in rubric_scorer.py): N/A dimensions have their weights redistributed to applicable ones. Narrow-scope code (e.g., only D1_wpcs applicable) gets overall=100.
  Issue 4 (Triage formula mixes scales in triage_ratios.py): `0.6 * pass_rate + 0.4 * spearman` conflates pass rates (proportion) with Spearman correlation (agreement magnitude) as if they're the same scale.
errors: No crashes — metrics are computed but misleading.
reproduction: Run eval on any ratio. Inspect per-dimension N/A counts (>90% N/A on most dims). Check overall scores (all near 100).
started: Since pipeline was written.

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-04-06T00:00:00Z
  checked: symptoms provided by user
  found: Three distinct issues in eval pipeline
  implication: Need to read all four files before making any changes

- timestamp: 2026-04-06T00:01:00Z
  checked: eval_gen.py _compute_summary()
  found: per_dimension dict only has {mean, pass_rate_8, na_count} — no na_rate or pass_rate_8_inclusive. security_pass_rate defaults to 1.0 when no security vals. No n_applicable_dims in summary.
  implication: Issue 2 confirmed. N/A examples excluded from pass_rate_8 denominator, making narrow-scope code look perfect.

- timestamp: 2026-04-06T00:01:00Z
  checked: rubric_scorer.py compute_overall()
  found: N/A weight redistribution is in compute_overall(); RubricScore dataclass has no n_applicable_dims field. The spec says transparency fix is in eval_gen.py summary (n_applicable_dims), not in the scorer.
  implication: Issue 3 fix = add n_applicable_dims to summary in eval_gen.py (per-example via RubricScore.dimension_na) and expose it in per_dimension dict.

- timestamp: 2026-04-06T00:01:00Z
  checked: triage_ratios.py compute_overall_score() and triage_ratios()
  found: Single blended score 0.6*((phpcs+security)/2)+0.4*spearman. triage_table shows "Overall Score Ranking (formula)" with blended score. TriageResult namedtuple has no separate gen_quality_score/judge_calibration fields.
  implication: Issue 4 confirmed. Need to split into two axes; 5pp elimination rule must use gen_quality_score only.

- timestamp: 2026-04-06T00:01:00Z
  checked: tests/test_triage.py TestComputeOverallScore
  found: Tests call compute_overall_score() directly and verify the blended formula. These tests must be updated after the refactor.
  implication: Test updates needed for triage score splitting.

- timestamp: 2026-04-06T00:01:00Z
  checked: eval_gate.py
  found: eval_gate reads security_pass_rate from gen results. It uses results.get("security_pass_rate", 0.0) — so if security_pass_rate becomes None, the gate will treat it as 0.0, which will FAIL the security gate. This is actually correct behavior (unknown = fail-safe). No cascading breakage.
  implication: eval_gate.py needs no changes. The None handling is safe for gate purposes.

## Resolution

root_cause: Three separate issues: (1) eval_gen.py excluded N/A examples from pass_rate_8 denominator and defaulted security_pass_rate=1.0 when no security code; (2) rubric_scorer.py weight redistribution (intentional per-example behavior) was insufficiently transparent — no n_applicable_dims exposed; (3) triage_ratios.py blended proportions (phpcs/security rates) with Spearman correlation in a single weighted sum despite them being different scales.
fix: (1) Added pass_rate_8_inclusive, na_rate, n_applicable_dims_mean to eval_gen.py summary; security_pass_rate now returns None when no applicable examples. (2) No change to rubric_scorer.py — transparency fix in eval_gen.py summary. (3) Replaced blended score with two independent axes: gen_quality_score=(phpcs+security)/2 for 5pp elimination, judge_calibration=spearman as separate reporting axis; TriageResult gains gen_quality_scores and judge_calibrations fields; triage table shows both axes.
verification: 174/174 tests pass. Execution-validated: pass_rate_8_inclusive=0.25 vs pass_rate_8=1.0 for 75% N/A scenario; security_pass_rate=None when all N/A; gen_quality_scores and judge_calibrations fields populated correctly; triage table contains both Gen Quality and Spearman sections.
files_changed: [eval/eval_gen.py, scripts/triage_ratios.py, tests/test_eval_gen.py, tests/test_triage.py, CHANGELOG.md]
