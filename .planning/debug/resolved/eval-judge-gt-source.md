---
status: resolved
trigger: "eval_judge.py uses rubric_scorer.score_code() for ground truth instead of the test set's own GT scores from the assistant response"
created: 2026-04-06T00:00:00Z
updated: 2026-04-06T00:00:00Z
symptoms_prefilled: true
goal: find_and_fix
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: CONFIRMED. eval_judge.py line 261 calls score_code(code) for GT. Fix: extract GT from test example's assistant response JSON. The GT response uses field names {overall_score, wpcs_compliance, security_score, performance_score, i18n_score, accessibility_score, documentation_score} which differ from DIM_NAME_MAP field names. Need custom mapping. Keep rubric_scorer as fallback.
test: implement fix in eval_judge.py, add helper _extract_gt_from_assistant, update tests
expecting: fix removes rubric_scorer GT path, test dataset GT variance is real (min=10, max=100)
next_action: implement fix

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: eval_judge.py should compare model's judge scores against GT scores already present in the test dataset's assistant response (real variance: min=10, max=100, mean=77.1, stdev=14.2, with 451 examples below 70)
actual: eval_judge.py ignores the test set's GT scores and re-scores code using rubric_scorer.score_code(), which gives 95-100 to all examples (stdev=0.4). Spearman correlation is -0.01.
errors: No crash — the correlation is computed but meaningless due to near-zero GT variance.
reproduction: Run eval_judge on any ratio. All produce near-zero Spearman because GT has no variance.
started: Since eval_judge.py was written. The rubric_scorer was always the GT source.

## Eliminated
<!-- APPEND only - prevents re-investigating -->

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-04-06T00:05:00Z
  checked: eval/eval_judge.py line 261
  found: gt = score_code(code) — unconditionally calls rubric_scorer on extracted code
  implication: GT scores have near-zero variance; Spearman is meaningless

- timestamp: 2026-04-06T00:05:00Z
  checked: data/final_dataset/openai_test.jsonl — wp_judge examples
  found: 1855 examples; assistant response is JSON with overall_score, wpcs_compliance, security_score, performance_score, i18n_score, accessibility_score, documentation_score; real variance min=10, max=100, mean=77.1, stdev=14.2
  implication: GT scores exist in the test set with real variance; just need to extract them

- timestamp: 2026-04-06T00:05:00Z
  checked: eval/rubric_definitions.py DIM_NAME_MAP
  found: DIM_NAME_MAP maps wpcs_compliance->D1_wpcs, security_score->D2_security, but test set uses performance_score (not "performance"), i18n_score (not "i18n_l10n"), accessibility_score (not "accessibility"), documentation_score (no matching dimension key at all). No sql_safety, wp_api_usage, error_handling, code_structure fields in GT.
  implication: Need custom GT field -> dim_key mapping; test set only covers 6 dimensions (no D3_sql, D5_wp_api, D8_errors, D9_structure). documentation_score has no dim_key equivalent.

- timestamp: 2026-04-06T00:05:00Z
  checked: eval/eval_gen.py
  found: Uses score_code() to score MODEL-GENERATED code — this is correct (no GT in gen examples). Not the same problem.
  implication: eval_gen.py does not need to be changed.

- timestamp: 2026-04-06T00:05:00Z
  checked: eval/eval_gate.py
  found: Reads pre-computed eval_judge_results.json; no dependency on GT source. No change needed.
  implication: eval_gate.py does not need to be changed.

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: eval_judge.py line 261 called score_code(code) unconditionally for GT. rubric_scorer produces near-zero variance (stdev≈0.4) on high-quality code because it starts at 10 and only deducts for detected issues. The test set's assistant response contains scored judge output with real variance (min=10, max=100, stdev=14.2).

fix: Added _extract_gt_from_assistant() helper that parses GT scores from the test example's assistant response JSON. Added _GT_FIELD_TO_DIM mapping (5 fields: wpcs_compliance, security_score, performance_score, i18n_score, accessibility_score). Dimensions not in GT fall back lazily to rubric_scorer. Full rubric_scorer fallback for unparseable responses (1/3164 examples). Removes score_code() as primary GT source.

verification: 164 unit tests pass. Execution test: 3164 wp_judge examples parsed, 1 failure, overall GT variance: min=9, max=100, stdev=12.6. New test_extract_gt_from_assistant() covers 6 scenarios.

files_changed: [eval/eval_judge.py, tests/test_eval_judge.py, CHANGELOG.md]
