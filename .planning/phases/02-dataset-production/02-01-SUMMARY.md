---
phase: 02-dataset-production
plan: 01
subsystem: pipeline
tags: [judge, config, checkpoint, backoff, batch-api, anthropic, yaml, pytest]

# Dependency graph
requires:
  - phase: 01-pipeline-ready
    provides: scripts/utils.py with extract_json, call_with_backoff, load/save_checkpoint, batch_or_direct, Batch API helpers

provides:
  - Judge config with raised >= 8 threshold and security auto-FAIL rule
  - Deflated N/A scoring (7 not 10) for i18n and accessibility dimensions
  - Rejection templates for proactive security training examples
  - phase1_clone.py with checkpoint resume support
  - phase1_extract.py with checkpoint resume support
  - phase1_judge.py fully hardened with utils.py (extract_json, call_with_backoff, checkpoints, batch routing, security auto-FAIL enforcement)
  - Wave 0 test scaffolds: test_config.py (4 tests) and test_pipeline_integration.py (2 tests)

affects: [phase2-synthetic-gen, phase3-cot, judge-pipeline-execution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Checkpoint-resume pattern: load_checkpoint at start of main(), skip if in completed set, save_checkpoint after each repo"
    - "Batch routing: batch_or_direct(len(functions)) before judging loop, >= 50 uses Batch API"
    - "Security auto-FAIL: post-parse score check, security < 5 overrides verdict regardless of other scores"
    - "TDD RED/GREEN: tests written against config files and mocked pipeline imports before config changes applied"

key-files:
  created:
    - tests/test_config.py
    - tests/test_pipeline_integration.py
  modified:
    - config/judge_system.md
    - config/synthetic_prompts.yaml
    - scripts/phase1_clone.py
    - scripts/phase1_extract.py
    - scripts/phase1_judge.py

key-decisions:
  - "Judge PASS threshold raised from >= 7 to >= 8 — stricter quality bar required before any pipeline execution (Pitfall 1 from research)"
  - "Security auto-FAIL implemented in judge.py (not just config) — enforcement must be code-level not doc-level"
  - "N/A scoring deflated to 7 (not 10) — prevents dimension inflation on functions with no i18n/accessibility"
  - "Rejection templates added at config level with 3 sub-keys (proactive_nonce, proactive_capability, proactive_escaping)"
  - "Batch API path uses judge_functions_batch() helper to keep main() clean; batch_id not yet persisted per-repo in checkpoint (deferred)"

patterns-established:
  - "Pattern: All Phase 1 scripts use load_checkpoint at main() entry, save_checkpoint after each repo"
  - "Pattern: judge_function() uses call_with_backoff() + extract_json() — no bare client.messages.create or json.loads in pipeline scripts"
  - "Pattern: Security auto-FAIL is enforced via _apply_security_auto_fail() called on every judged result"

requirements-completed: [DATA-01, DATA-02, DATA-03]

# Metrics
duration: 25min
completed: 2026-03-26
---

# Phase 2 Plan 01: Config Updates and Phase 1 Script Hardening Summary

**Judge threshold raised to >= 8 with security auto-FAIL rule; Phase 1 clone/extract/judge scripts hardened with checkpoint resume, call_with_backoff, extract_json, and Batch API routing**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-26T05:30:00Z
- **Completed:** 2026-03-26T05:55:11Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Updated `config/judge_system.md` with >= 8 threshold, SECURITY AUTO-FAIL rule (< 5), and N/A scoring deflated to 7
- Added `rejection_templates` section to `config/synthetic_prompts.yaml` with 3 sub-keys (proactive_nonce, proactive_capability, proactive_escaping), 2 templates each
- Hardened all three Phase 1 scripts with utils.py: checkpoint resume, exponential backoff, robust JSON parsing, batch API routing, and security auto-FAIL enforcement
- Created Wave 0 test scaffolds with 6 tests total (4 config + 2 pipeline checkpoint integration); all 32 tests across test suite pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Update config files and create Wave 0 test scaffolds** - `9a2f2ce` (feat)
2. **Task 2: Harden Phase 1 scripts with utils.py integration** - `9fb8a1b` (feat)

**Plan metadata:** [pending final commit] (docs: complete plan)

## Files Created/Modified

- `config/judge_system.md` - Raised threshold to >= 8, added SECURITY AUTO-FAIL, deflated N/A from 10 to 7
- `config/synthetic_prompts.yaml` - Added rejection_templates section with 6 prompt templates across 3 keys
- `tests/test_config.py` - 4 tests: threshold, security auto-FAIL, N/A scoring, rejection templates
- `tests/test_pipeline_integration.py` - 2 tests: checkpoint skip behavior in clone and extract
- `scripts/phase1_clone.py` - Added from scripts.utils import, checkpoint resume in main()
- `scripts/phase1_extract.py` - Added from scripts.utils import, checkpoint resume in main()
- `scripts/phase1_judge.py` - Full rewrite with utils.py integration: removed REQUEST_INTERVAL/time.sleep, replaced json parsing with extract_json(), client.messages.create with call_with_backoff(), added checkpoint per repo, batch_or_direct routing, _apply_security_auto_fail()

## Decisions Made

- Raised PASS threshold from >= 7 to >= 8 as required (Pitfall 1 from research: config changes before pipeline execution)
- Enforced security auto-FAIL in code (phase1_judge.py) not just documentation — config is authoritative but code is the gate
- N/A scoring deflated to 7 to prevent inflation when most functions lack i18n/accessibility output
- Batch API path structured as helper function `judge_functions_batch()` — keeps main() readable
- Batch ID not persisted per-repo in checkpoint state (out of scope for this plan; tracked as deferred)

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Config files are ready for pipeline execution with raised threshold, security auto-FAIL, and rejection templates
- Phase 1 scripts are hardened and resume-capable — safe to run against real repos
- Phase 2 (02-02) can now execute: gap analysis and synthetic generation scripts
- Blocker to watch: batch_id per-repo not persisted to checkpoint (if a batch job is in-flight and process crashes, the batch_id is lost); this is a known limitation for Phase 1 execution

---
*Phase: 02-dataset-production*
*Completed: 2026-03-26*

## Self-Check: PASSED

All created files exist. Both task commits verified in git log.
