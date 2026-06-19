---
phase: 08-reward-infrastructure
plan: "04"
subsystem: antihack-eval-set
tags: [antihack, ci-aware-gate, perturbation, grpo-01, d-11, wave-4]
dependency_graph:
  requires: [08-03]
  provides:
    - scripts.build_antihack_set (3-axis perturbation + fixture CI gate + acceptance report)
    - output/antihack_validation/acceptance_report.json (4 CI bounds per axis, gate_pass)
    - tests/test_antihack.py (20 tests, CI-gate + schema + structural + coverage)
  affects: [09-gspo-trainer, 10-rl-eval]
tech_stack:
  added: []
  patterns:
    - Pure-Python perturbation (no model calls in perturbation functions)
    - CI-aware gate (hi_perturbed < lo_clean via bootstrap_ci) — D-09
    - Fixture-backed acceptance report (no live vLLM required for gate logic proof)
    - AST-based grep gate in tests (anthropic.Anthropic() instantiation count == 0)
    - Lazy import of reward_pipeline inside scoring functions (keeps --help + tests fast)
key_files:
  created:
    - scripts/build_antihack_set.py
    - output/antihack_validation/acceptance_report.json
    - output/antihack_validation/antihack_verbose_padding.jsonl
    - output/antihack_validation/antihack_template_critique_collapse.jsonl
    - output/antihack_validation/antihack_self_preference_swap.jsonl
  modified:
    - tests/test_antihack.py (un-stubbed + extended)
decisions:
  - "_load_source_records uses 'overall' key (eval_gen_results.jsonl schema); 'rubric_overall' accepted as fallback for forward compat — PATTERNS used 'rubric_overall' but real file has 'overall'"
  - "Fixture-backed acceptance report: live 45-case scoring deferred (no vLLM available in exec env); report clearly labelled report_type=fixture_backed; gate logic proof is complete"
  - "Agent(run_in_background=True) is a Claude Code orchestration primitive, not callable Python; script prepares batches, dispatch pattern documented in docstring; lazy-import of compute_reward inside score_and_gate keeps tests fast"
  - "test_ci_aware_not_bare_point: max-variance alternating [0, 0.8] / [0.2, 1.0] arrays with n_boot=2000; asserts gate_pass=False to prove overlapping CIs fail even when perturbed mean < clean mean"
  - "TestAntihackBuildScript.test_no_anthropic_api_in_script: uses ast.walk to check for actual Call nodes (not grep on raw text); excludes comment/docstring hits — one docstring line mentions the prohibition but has 0 AST-level instantiations"
metrics:
  duration_seconds: 357
  completed_date: "2026-06-20"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
requirements_satisfied: [GRPO-01]
---

# Phase 8 Plan 04: Anti-Hack Eval Set Summary

**One-liner:** 3-axis pure-Python perturbation (verbose padding, template-critique collapse, self-preference swap) on real >=65-score PHP outputs, with CI-aware gate (hi_perturbed < lo_clean via bootstrap_ci) and fixture-backed acceptance report publishing 4 CI bounds per axis.

## Tasks Completed

| # | Task | Commit | Key Output |
|---|------|--------|------------|
| 1 | 3-axis perturbation script + source filtering | `32aa0fa` | scripts/build_antihack_set.py |
| 2 | CI-aware gate + acceptance report + test suite | `a7f0c33` | tests/test_antihack.py (20 tests), output/antihack_validation/acceptance_report.json |

## What Was Built

### `scripts/build_antihack_set.py`

**Source filtering:**
- `_load_source_records(path, min_score=65.0)` — filters real gen+judge JSONL to records with `overall >= 65.0` (Pitfall 7: perturb MEDIUM-HIGH quality originals only). Handles both `overall` (eval_gen_results.jsonl schema) and `rubric_overall` fallback key.

**Three perturbation functions (pure Python, no model calls):**
1. `perturb_verbose_padding(php_code)` — injects 3-6 inert PHP comment blocks/docblocks + closing docblock at random line positions. Simulates score inflation via documentation bloat.
2. `perturb_template_critique_collapse(php_code)` — appends a boilerplate critique PHP docblock with 3-5 stock phrases. Simulates judge reasoning collapse to generic templates.
3. `perturb_self_preference_swap(php_code)` — prepends a self-preference header (`// [REFERENCE IMPLEMENTATION]` / `// [JUDGE TARGET]`) and removes duplicate `<?php`. Simulates bias toward the judge's own training-target output.

All three functions are in `PERTURBATION_AXES` dict for uniform dispatch.

**Batch construction:**
- `build_axis_batches(records, cases_per_axis=15)` — samples records, applies each perturbation, emits `{case_id, axis, php_perturbed, php_clean, source_overall}` dicts.
- `write_axis_batches(axis_batches, output_dir)` — EVAL-06 JSONL (one object per line): `antihack_verbose_padding.jsonl`, `antihack_template_critique_collapse.jsonl`, `antihack_self_preference_swap.jsonl`.

**CI-aware gate:**
- `compute_axis_gate(perturbed_rewards, clean_rewards, axis_name)` — calls `bootstrap_ci` from `scripts.compute_concentration`; gate passes iff `hi_perturbed < lo_clean`. Reports all 4 CI bounds: `lo_perturbed`, `hi_perturbed`, `lo_clean`, `hi_clean` + `gate_pass`, `perturbed_ci`, `clean_ci` lists. Lazy import of `bootstrap_ci` and `numpy` inside function.

**Acceptance report:**
- `build_fixture_acceptance_report(axis_batches, output_dir)` — fixture-backed: uses `np.random.seed(2024)` + `np.random.uniform` to produce realistic synthetic reward arrays (perturbed 0.28-0.44, clean 0.60-0.80) and runs the actual `compute_axis_gate` function. All 3 axes pass (hi_p < lo_c).
- `score_and_gate(axis_batches, output_dir, ...)` — live-scoring path: calls `compute_group_rewards` per axis; requires `EVAL_JUDGE_BASE_URL`. Lazy-imports `reward_pipeline`. Documented as live follow-up.

**CLI:**
```
python -m scripts.build_antihack_set --help
python -m scripts.build_antihack_set --fixture-gate   # proves gate logic, no vLLM
python -m scripts.build_antihack_set --score-and-gate # requires vLLM judge endpoint
```

### `output/antihack_validation/acceptance_report.json`

```json
{
  "report_type": "fixture_backed",
  "all_axes_pass": true,
  "gate_criterion": "hi_perturbed < lo_clean (D-09 CI-aware)",
  "axes": {
    "verbose_padding":           {"gate_pass": true, "hi_perturbed": 0.378, "lo_clean": 0.666, ...},
    "template_critique_collapse":{"gate_pass": true, "hi_perturbed": 0.326, "lo_clean": 0.685, ...},
    "self_preference_swap":      {"gate_pass": true, "hi_perturbed": 0.363, "lo_clean": 0.646, ...}
  }
}
```
All 4 CI bounds (`lo_perturbed`, `hi_perturbed`, `lo_clean`, `hi_clean`) + `gate_pass` present per axis.

### `tests/test_antihack.py` (20 tests, all pass)

| Class | Tests | What it checks |
|-------|-------|---------------|
| `TestAntihackCIGate` | 3 | CI gate math: pass when separated, fail when overlapping, 4-bound report structure |
| `TestAntihackCaseCoverage` | 3 | All 3 axis batches built from real source JSONL |
| `TestAntihackAxisGate` | 3 | Per-axis `compute_axis_gate` returns gate_pass + 4 CI bounds |
| `TestAntihackAcceptanceReport` | 5 | Report exists, has all axes, 4-bound schema, CI-aware criterion string, [lo,hi] list shape |
| `TestAntihackBuildScript` | 5 | 0 `anthropic.Anthropic()` AST nodes, `bootstrap_ci` in gate source, `hi_p < lo_c` formula, `_load_source_records` filter boundary, 3-axis PERTURBATION_AXES dict |
| `test_bootstrap_ci_importable` | 1 | Wave 0 import gate |

**Key test — `test_ci_aware_not_bare_point` (D-09 proof):**
Uses max-variance alternating arrays `[0.0, 0.8, 0.1, 0.9, ...]` / `[0.2, 1.0, 0.3, 1.0, ...]` where `perturbed.mean()=0.45 < clean.mean()=0.65` but CIs fully overlap. Asserts `gate_pass=False` — proving the gate uses CI bounds, not bare point comparison.

## Deviations from Plan

### Auto-fixed / structural adjustments

**1. [Rule 1 - Schema Bug] Source JSONL field name is 'overall', not 'rubric_overall'**
- **Found during:** Orientation (read actual JSONL schema)
- **Issue:** PATTERNS.md `_load_source_records` code block uses `rec.get("rubric_overall", 0.0)`. The actual file (`eval_gen_results.jsonl`) uses `"overall"` as the score key. Using the PATTERNS key would silently default everything to 0.0, filtering all records out.
- **Fix:** `_load_source_records` checks `score_key` (`"overall"`) first, then `fallback_key` (`"rubric_overall"`). Both field names documented in docstring.
- **Files:** scripts/build_antihack_set.py

**2. [Rule 3 - Blocking Gap] Agent(run_in_background=True) is not callable Python**
- **Found during:** Design (advisor review before implementation)
- **Issue:** `Agent(...)` is a Claude Code orchestration construct, not a Python class. Writing `Agent(run_in_background=True)` in a `.py` script would raise `NameError`. The plan's agent-spawn pattern is the dispatch *model*, not literal Python code.
- **Fix:** `score_and_gate()` calls `compute_group_rewards` directly for in-process convenience (requires live vLLM). The SKILL.md agent dispatch pattern is documented in the module docstring for human orchestrators. Lazy import of `reward_pipeline` inside `score_and_gate()` keeps tests + `--help` fast.
- **Files:** scripts/build_antihack_set.py

**3. [Plan Note] Fixture-backed acceptance report instead of live 45-case run**
- **Per-plan note:** "If generating the full 45-case live set requires the live vLLM judge endpoint (which may be unavailable in this execution environment), build the construction script + a deterministic/fixture-backed gate test that proves the CI-aware disposition logic, and document any live-run step as a follow-up in SUMMARY.md — do NOT block the plan on live infrastructure."
- **Resolution:** `build_fixture_acceptance_report()` uses `np.random.seed(2024)` + synthetic reward arrays to prove gate logic. Report is labelled `report_type=fixture_backed`. The live follow-up is documented below.

**4. [Rule 2 - Missing Guard] AST-based anthropic check in tests**
- **Found during:** Test design (advisor review)
- **Issue:** Plain `grep -c 'anthropic.Anthropic('` would match docstring comments (e.g., line 44 is "NO direct anthropic.Anthropic( calls in the reward compute path"). The acceptance criterion needs to check for actual Python instantiation, not comment mentions.
- **Fix:** `TestAntihackBuildScript.test_no_anthropic_api_in_script` uses `ast.walk` to check for `ast.Call` nodes where `.func.attr == "Anthropic"`. Result: 0 actual instantiations. The one comment-mention is correctly excluded.
- **Files:** tests/test_antihack.py

## Known Follow-Ups (Live 45-Case Construction)

The live scoring run requires the local vLLM judge endpoint (EVAL_JUDGE_BASE_URL) serving the frozen wp_judge canonical checkpoint. Steps when vLLM is available:

```bash
# 1. Start vLLM with frozen wp_judge checkpoint
export EVAL_JUDGE_BASE_URL="http://localhost:8000/v1"

# 2. Run full pipeline (perturbation + live scoring + CI gate)
python -m scripts.build_antihack_set \
    --source-jsonl output/eval_reasoning_v4_winner/eval_gen_results.jsonl \
    --output-dir output/antihack_validation/ \
    --cases-per-axis 15 \
    --score-and-gate

# 3. Check gate results
cat output/antihack_validation/acceptance_report.json
```

For agent-dispatch scoring (D-08-03 SKILL.md pattern), spawn background agents per batch file:
```
Agent(
  model="sonnet",
  description="Score antihack batch: axis=verbose_padding",
  prompt="Score each PHP case in output/antihack_validation/antihack_verbose_padding.jsonl
    using reward_pipeline.compute_reward(). Write results to
    output/antihack_validation/scored_verbose_padding.jsonl as JSONL:
    {case_id, scalar, breakdown_dict}. Use EVAL_JUDGE_BASE_URL.",
  run_in_background=True
)
```

## Threat Surface Scan

No new network endpoints or auth paths introduced.

T-08-06 mitigations:
- `scripts/build_antihack_set.py`: 0 `anthropic.Anthropic()` AST instantiations (verified by TestAntihackBuildScript)
- `score_and_gate()` dispatches via `reward_pipeline.compute_group_rewards` (local vLLM only)
- AST-based test gate in CI suite ensures this property is maintained

T-08-07 mitigations:
- `acceptance_report.json` publishes all 4 CI bounds per axis (not just pass/fail)
- Gate criterion string `"hi_perturbed < lo_clean (D-09 CI-aware)"` written to report
- Test `test_gate_criterion_is_ci_aware` verifies criterion string presence

T-08-08 mitigations:
- `_load_source_records` filters to `overall >= 65.0` (Pitfall 7)
- 17 qualifying records found in eval_gen_results.jsonl (all with rubric overall 65-100)

## Self-Check: PASSED

Files created/exist:
- [x] scripts/build_antihack_set.py (created)
- [x] tests/test_antihack.py (modified — un-stubbed + extended)
- [x] output/antihack_validation/acceptance_report.json (created by fixture run)
- [x] output/antihack_validation/antihack_verbose_padding.jsonl
- [x] output/antihack_validation/antihack_template_critique_collapse.jsonl
- [x] output/antihack_validation/antihack_self_preference_swap.jsonl

Commits verified:
- [x] 32aa0fa feat(08-04): build_antihack_set.py — 3-axis perturbation + source filtering
- [x] a7f0c33 feat(08-04): test_antihack.py — CI-aware gate unit tests + acceptance report schema

Acceptance criteria:
- [x] `--help` runs; argparse exposes --source-jsonl, --output-dir, --cases-per-axis
- [x] `_load_source_records` filters to overall >= 65.0 (Pitfall 7)
- [x] Three distinct perturbation functions (one per D-11 axis), pure-Python (no Anthropic API)
- [x] perturbed + matched-clean batches written as JSONL (3 axis JSONL files)
- [x] gate computed as `hi_perturbed < lo_clean` via bootstrap_ci (not bare point)
- [x] acceptance report publishes lo_perturbed, hi_perturbed, lo_clean, hi_clean per axis + gate_pass
- [x] `anthropic.Anthropic(` AST instantiation count in build_antihack_set.py == 0
- [x] `test_ci_aware_not_bare_point` proves overlapping CIs fail the gate (D-09)
- [x] `pytest tests/test_antihack.py -q`: 20 passed
- [x] `pytest tests/ -q`: 421 passed
