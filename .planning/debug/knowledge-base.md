# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## eval-judge-gt-source — eval_judge.py uses rubric_scorer as GT instead of test dataset assistant response
- **Date:** 2026-04-06
- **Error patterns:** Spearman, near-zero, correlation, rubric_scorer, ground truth, GT, score_code, eval_judge, variance, stdev
- **Root cause:** eval_judge.py called score_code(code) unconditionally for GT. rubric_scorer produces near-zero variance (stdev≈0.4) on high-quality code. The test set's assistant response contains scored judge output with real variance (min=10, max=100, stdev=14.2).
- **Fix:** Added _extract_gt_from_assistant() helper that parses GT scores from the test example's assistant response JSON. Added _GT_FIELD_TO_DIM mapping for 5 fields (wpcs_compliance, security_score, performance_score, i18n_score, accessibility_score). Dimensions not in GT fall back lazily to rubric_scorer.
- **Files changed:** eval/eval_judge.py, tests/test_eval_judge.py, CHANGELOG.md
---

