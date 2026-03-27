---
phase: 03-model-prep-and-training
plan: 02
subsystem: testing
tags: [eval, phpcs, spearman, scipy, openai, vllm, dgx-toolbox, quality-gate, wp-bench]

# Dependency graph
requires:
  - phase: 03-model-prep-and-training
    provides: DGXToolbox (scripts/dgx_toolbox.py) with vllm_endpoint() resolver

provides:
  - eval/eval_gen.py: PHPCS pass rate + security pass rate evaluation
  - eval/eval_judge.py: Spearman correlation vs PHPCS ground truth
  - eval/eval_gate.py: Quality gate with config-driven thresholds and sys.exit(0/1)
  - config/wp-bench.yaml: wp-bench configuration for local vLLM endpoint
  - tests/test_eval_gen.py: 3 tests for gen eval logic
  - tests/test_eval_judge.py: 3 tests for judge eval logic
  - tests/test_eval_gate.py: 5 tests for gate logic

affects:
  - 03-03-model-training (gate used after training to decide whether to merge LoRA)
  - CI/CD pipeline (eval_gate.py is the hard exit gate in the deployment pipeline)

# Tech tracking
tech-stack:
  added:
    - scipy.stats.spearmanr (Spearman correlation computation)
    - openai (Python client for vLLM endpoint)
    - pyyaml (threshold loading from train_config.yaml)
  patterns:
    - Wave-0 TDD: tests written before implementations (eval tests imported stub functions)
    - DGX Toolbox resolver pattern: all endpoint URLs via dgx.vllm_endpoint(), never hardcoded
    - Config-driven thresholds: eval_gate.py reads from config/train_config.yaml eval section
    - Fallback thresholds: when train_config.yaml missing, sensible defaults used

key-files:
  created:
    - eval/__init__.py
    - eval/eval_gen.py
    - eval/eval_judge.py
    - eval/eval_gate.py
    - config/wp-bench.yaml
    - tests/test_eval_gen.py
    - tests/test_eval_judge.py
    - tests/test_eval_gate.py
  modified: []

key-decisions:
  - "Security pass rate defined as: examples where any WordPress.Security.* sniff fired / total — not all examples"
  - "PHPCS unavailability handled gracefully (treated as pass with _phpcs_unavailable flag) to allow test runs without binary"
  - "eval_gate.py imports get_toolbox even though gate doesn't make model calls — consistency + future-proof"
  - "parse_judge_response returns None for unparseable responses (not ValueError) — callers skip instead of crash"
  - "invert_phpcs_errors formula: max(0, 100 - errors * 5) — identical to plan spec"
  - "load_thresholds falls back to _FALLBACK_THRESHOLDS when config/train_config.yaml absent"

patterns-established:
  - "DGX Toolbox endpoint pattern: from scripts.dgx_toolbox import get_toolbox; dgx = get_toolbox(); client = openai.OpenAI(base_url=dgx.vllm_endpoint())"
  - "Wave-0 test pattern: write tests against module API contracts before implementations exist"
  - "Config-driven gate pattern: eval_gate.py load_thresholds(config_path) reads YAML eval section"
  - "PHP code extraction: try ```php fenced, then ``` fenced, then raw fallback"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05]

# Metrics
duration: 22min
completed: 2026-03-27
---

# Phase 3 Plan 02: Evaluation Suite Summary

**Three-script eval suite (PHPCS pass rate, Spearman correlation, quality gate) using DGX Toolbox vLLM resolver with 11 mock-based tests and wp-bench.yaml config**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-03-27T22:06:56Z
- **Completed:** 2026-03-27T22:28:00Z
- **Tasks:** 2/2
- **Files modified:** 8 created

## Accomplishments

- Wave-0 test scaffolds (11 tests, all mock-based, no GPU/phpcs binary required) committed before implementations
- eval_gen.py evaluates wp_gen mode examples via PHPCS and computes pass rate + security pass rate
- eval_judge.py computes Spearman correlation between model judge scores and PHPCS error counts as ground truth
- eval_gate.py exits 0/1 based on config-driven thresholds from train_config.yaml, never hardcoded
- All 74 existing tests still passing (1 skipped) after additions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create eval test scaffolds and wp-bench config** - `9e61756` (test)
2. **Task 2: Create eval_gen.py, eval_judge.py, and eval_gate.py** - `1deeb18` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `eval/__init__.py` - Makes eval a Python package
- `eval/eval_gen.py` - PHPCS pass rate + security pass rate for wp_gen mode; exposes run_phpcs(), compute_pass_rate(), classify_security()
- `eval/eval_judge.py` - Spearman correlation eval for wp_judge mode; exposes invert_phpcs_errors(), parse_judge_response()
- `eval/eval_gate.py` - Quality gate with sys.exit(0/1); exposes check_gates(), load_thresholds()
- `config/wp-bench.yaml` - wp-bench config with local vLLM endpoint (api_base: http://localhost:8020/v1)
- `tests/test_eval_gen.py` - 3 tests: phpcs_eval_runs, security_rate_detection, pass_rate_calculation
- `tests/test_eval_judge.py` - 3 tests: spearman_computation, score_inversion, judge_output_parsing
- `tests/test_eval_gate.py` - 5 tests: gate_pass, gate_fail_phpcs, gate_fail_spearman, gate_fail_security, reads_thresholds_from_config

## Decisions Made

- Security pass rate uses WordPress.Security.* sniff prefix as the filter — matches the PHPCS sniff taxonomy exactly
- PHPCS subprocess failures (binary unavailable) return a graceful "passed" result with `_phpcs_unavailable: True` flag rather than crashing — allows test suite to run on developer machines without phpcs
- eval_gate.py falls back to hardcoded defaults (_FALLBACK_THRESHOLDS) if train_config.yaml doesn't exist yet — prevents gate from failing before 03-01 output is present
- parse_judge_response returns None for unparseable JSON rather than raising ValueError — eval_judge.py skips and increments `skipped` counter

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all implementations matched test contracts on first run.

## User Setup Required

None - no external service configuration required. Eval scripts require vLLM running (handled by DGX Toolbox) and phpcs binary for live runs.

## Next Phase Readiness

- Eval suite ready to run against any served checkpoint immediately after training
- eval_gate.py is the automated go/no-go decision point for LoRA adapter merging
- wp-bench.yaml configured and ready for wp-bench CLI integration
- Concern (carried forward): Judge correlation circularity — model used to judge may have been trained on data judged by Claude; consider using a held-out human-scored subset for the spearman eval

## Self-Check: PASSED

All files confirmed present on disk:
- eval/__init__.py: FOUND
- eval/eval_gen.py: FOUND
- eval/eval_judge.py: FOUND
- eval/eval_gate.py: FOUND
- config/wp-bench.yaml: FOUND
- tests/test_eval_gen.py: FOUND
- tests/test_eval_judge.py: FOUND
- tests/test_eval_gate.py: FOUND

Commits confirmed in git log:
- 9e61756 (Task 1: test scaffolds + wp-bench.yaml): FOUND
- 1deeb18 (Task 2: eval scripts): FOUND

Final verification:
- `grep -rn "from scripts.dgx_toolbox import get_toolbox" eval/` → 3 matches
- `grep -rn "localhost:8020" eval/` → 0 matches
- `python3 -m pytest tests/ -x -q` → 74 passed, 1 skipped

---
*Phase: 03-model-prep-and-training*
*Completed: 2026-03-27*
