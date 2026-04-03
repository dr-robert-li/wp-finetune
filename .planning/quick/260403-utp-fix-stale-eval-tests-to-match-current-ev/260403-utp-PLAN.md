---
phase: quick
plan: 260403-utp
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/test_eval_gen.py
  - tests/test_eval_judge.py
  - tests/test_eval_gate.py
autonomous: true
requirements: [EVAL-TESTS]

must_haves:
  truths:
    - "pytest tests/test_eval_gen.py collects and passes all tests"
    - "pytest tests/test_eval_judge.py collects and passes all tests"
    - "pytest tests/test_eval_gate.py collects and passes all tests"
  artifacts:
    - path: "tests/test_eval_gen.py"
      provides: "Tests for eval_gen public API surface"
    - path: "tests/test_eval_judge.py"
      provides: "Tests for eval_judge public API surface"
    - path: "tests/test_eval_gate.py"
      provides: "Tests for eval_gate public API surface"
  key_links:
    - from: "tests/test_eval_gen.py"
      to: "eval/eval_gen.py"
      via: "imports _compute_summary, _extract_php_code"
      pattern: "from eval\\.eval_gen import"
    - from: "tests/test_eval_judge.py"
      to: "eval/eval_judge.py"
      via: "imports parse_judge_response, _safe_spearman"
      pattern: "from eval\\.eval_judge import"
    - from: "tests/test_eval_gate.py"
      to: "eval/eval_gate.py"
      via: "imports check_gates, load_thresholds"
      pattern: "from eval\\.eval_gate import"
---

<objective>
Fix 3 stale Wave-0 eval test files so they match the current source API surfaces.

Purpose: These tests were written spec-first before implementation. The source modules were refactored (phpcs-based scoring replaced by rubric_scorer, eval_gate return type changed, threshold dict expanded) but tests were never updated. All 3 test files fail at import time or with wrong assertions.

Output: 3 passing test files that verify current eval module behavior without GPU/model/external deps.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@eval/eval_gen.py
@eval/eval_judge.py
@eval/eval_gate.py
@eval/rubric_scorer.py
@eval/rubric_definitions.py

<interfaces>
<!-- Current source API that tests must align to -->

From eval/eval_gen.py:
- `_extract_php_code(text: str) -> str` — extracts PHP from fenced blocks
- `_compute_summary(rubric_scores: list[RubricScore]) -> dict` — aggregates scores into summary with keys: total, overall_mean, overall_median, grade_distribution, per_dimension, floor_rules, phpcs_pass_rate, security_pass_rate
- `run_eval(dataset_path, limit, output_path, model) -> dict` — main runner (needs vLLM, not unit-testable)
- NO `run_phpcs`, `compute_pass_rate`, `classify_security` — these were removed during rubric refactor

From eval/eval_judge.py:
- `parse_judge_response(response: str) -> Optional[dict]` — parses JSON from model response (exists, tests are correct)
- `_safe_spearman(xs, ys) -> dict` — returns {"corr", "p_value", "n_pairs"}
- NO `invert_phpcs_errors` — removed during rubric refactor

From eval/eval_gate.py:
- `load_thresholds(config_path) -> dict` — returns dict with keys: overall_mean_target, overall_spearman_target, gen_dimension_targets, judge_dimension_targets, phpcs_pass_target, spearman_target, security_pass_target
- `check_gates(results: dict, thresholds: dict) -> tuple[bool, list[dict]]` — returns (all_passed, gate_rows) where each gate_row is {"gate", "target", "actual", "passed"}
- `_FALLBACK_THRESHOLDS` — module-level dict with all threshold defaults

From eval/rubric_scorer.py:
- `run_phpcs(code: str, standard: str = "WordPress") -> dict` — phpcs runner lives HERE now
- `score_code(code: str, file_path: str = "<generated>") -> RubricScore`
- `class RubricScore` — has .overall, .grade, .dimension_scores, .dimension_na, .floor_rules_applied, .triggered_checks
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Rewrite test_eval_gen.py to test current API</name>
  <files>tests/test_eval_gen.py</files>
  <action>
Rewrite tests/test_eval_gen.py to test the actual eval_gen.py API surface. The old tests imported `run_phpcs`, `compute_pass_rate`, `classify_security` which no longer exist in eval_gen. The module now delegates to `rubric_scorer.score_code()` and aggregates via `_compute_summary()`.

New test structure:
1. `test_extract_php_code()` — test `_extract_php_code` with: fenced ```php block, generic ``` block, raw PHP text (no fences). Assert correct extraction for each.
2. `test_compute_summary_basic()` — create 3-5 mock `RubricScore` objects (import from `eval.rubric_scorer`), pass to `_compute_summary()`, assert: total count correct, overall_mean/median computed correctly, phpcs_pass_rate and security_pass_rate present and reasonable.
3. `test_compute_summary_empty()` — pass empty list to `_compute_summary()`, assert returns `{"total": 0}`.

Import from eval.eval_gen: `_extract_php_code`, `_compute_summary`
Import from eval.rubric_scorer: `RubricScore`

For mock RubricScore objects, instantiate directly using the dataclass fields (check rubric_scorer.py for the exact field names: overall, grade, dimension_scores, dimension_na, floor_rules_applied, triggered_checks). Use realistic dimension keys from DIMENSION_WEIGHTS (D1_correctness, D2_security, etc.).

Do NOT test `run_eval()` — it requires vLLM. Keep all tests pure unit tests with no external deps.
  </action>
  <verify>
    <automated>cd /home/robert_li/Desktop/projects/wp-finetune && python -m pytest tests/test_eval_gen.py -x -v 2>&1 | tail -20</automated>
  </verify>
  <done>All tests in test_eval_gen.py pass. No import errors. Tests exercise _extract_php_code and _compute_summary against current API.</done>
</task>

<task type="auto">
  <name>Task 2: Rewrite test_eval_judge.py to test current API</name>
  <files>tests/test_eval_judge.py</files>
  <action>
Rewrite tests/test_eval_judge.py to test the actual eval_judge.py API surface. The old test imported `invert_phpcs_errors` which no longer exists. `parse_judge_response` still exists and those tests are mostly correct.

New test structure:
1. `test_judge_output_parsing()` — KEEP the existing test logic (it is correct). Tests: valid JSON, fenced JSON, malformed response, missing overall_score. Import `parse_judge_response` from `eval.eval_judge`.
2. `test_spearman_computation()` — KEEP the existing test (uses scipy directly, still valid).
3. `test_safe_spearman_edge_cases()` — NEW test for `_safe_spearman` from eval_judge: test with <2 items returns corr=0.0, test with all-identical values returns corr=0.0, test with valid pairs returns dict with keys "corr", "p_value", "n_pairs".

REMOVE: `test_score_inversion` entirely (tests nonexistent `invert_phpcs_errors`).

Import from eval.eval_judge: `parse_judge_response`, `_safe_spearman`
Do NOT import `invert_phpcs_errors`.
  </action>
  <verify>
    <automated>cd /home/robert_li/Desktop/projects/wp-finetune && python -m pytest tests/test_eval_judge.py -x -v 2>&1 | tail -20</automated>
  </verify>
  <done>All tests in test_eval_judge.py pass. No import errors. Tests exercise parse_judge_response and _safe_spearman against current API.</done>
</task>

<task type="auto">
  <name>Task 3: Rewrite test_eval_gate.py to test current API</name>
  <files>tests/test_eval_gate.py</files>
  <action>
Rewrite tests/test_eval_gate.py to match the current check_gates signature and return type. Two breaking changes:
1. `check_gates()` now returns `(bool, list[dict])` not `(bool, list[str])`. Each dict has keys: gate, target, actual, passed.
2. Thresholds dict must include ALL keys from `_FALLBACK_THRESHOLDS`: overall_mean_target, overall_spearman_target, gen_dimension_targets, judge_dimension_targets, phpcs_pass_target, spearman_target, security_pass_target.
3. Results dict must include keys that check_gates reads: overall_mean, gen_dimension_pass_rates, overall_spearman, judge_dimension_correlations, phpcs_pass_rate, spearman_corr, security_pass_rate.

New test structure using FULL thresholds/results dicts:

1. `test_gate_pass()` — All metrics above all thresholds. Assert `passed is True` and `all(row["passed"] for row in gate_rows)`.
2. `test_gate_fail_phpcs()` — Set phpcs_pass_rate below phpcs_pass_target. Assert `passed is False`. Assert at least one gate_row has gate containing "phpcs" and passed=False.
3. `test_gate_fail_spearman()` — Set spearman_corr below spearman_target. Assert `passed is False`. Assert at least one gate_row has gate containing "spearman" and passed=False.
4. `test_gate_fail_security()` — Set security_pass_rate below security_pass_target. Assert `passed is False`. Assert at least one gate_row has gate containing "security" and passed=False.
5. `test_gate_reads_thresholds_from_config()` — KEEP logic but update: config YAML must include ALL threshold keys (overall_mean_target, overall_spearman_target, gen_dimension_targets: {}, judge_dimension_targets: {}). Results dict must include all required keys. Update assertions to check gate_rows (list of dicts) not failures (list of strings).

Use a helper function to build a full passing results dict and full thresholds dict, then override specific values per test. This avoids KeyError from missing keys.

Import from eval.eval_gate: `check_gates`, `load_thresholds`, `_FALLBACK_THRESHOLDS`
  </action>
  <verify>
    <automated>cd /home/robert_li/Desktop/projects/wp-finetune && python -m pytest tests/test_eval_gate.py -x -v 2>&1 | tail -20</automated>
  </verify>
  <done>All tests in test_eval_gate.py pass. No import errors. Tests use correct check_gates return type (bool, list[dict]) and full threshold/results dicts.</done>
</task>

</tasks>

<verification>
Run all 3 test files together:
```bash
cd /home/robert_li/Desktop/projects/wp-finetune && python -m pytest tests/test_eval_gen.py tests/test_eval_judge.py tests/test_eval_gate.py -v
```
All tests pass with 0 failures, 0 errors, 0 import failures.
</verification>

<success_criteria>
- All 3 test files import without error
- All tests pass (pytest exit code 0)
- Tests exercise current API surfaces: _extract_php_code, _compute_summary, parse_judge_response, _safe_spearman, check_gates, load_thresholds
- No tests reference removed functions: run_phpcs (from eval_gen), compute_pass_rate, classify_security, invert_phpcs_errors
- All tests remain pure unit tests (no GPU, no vLLM, no external services)
</success_criteria>

<output>
After completion, create `.planning/quick/260403-utp-fix-stale-eval-tests-to-match-current-ev/260403-utp-SUMMARY.md`
</output>
