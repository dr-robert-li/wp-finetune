---
phase: 01-pipeline-ready
plan: 01
subsystem: testing
tags: [python, pytest, anthropic, checkpoint, backoff, batch-api, preflight, phpcs]

# Dependency graph
requires: []
provides:
  - "scripts/utils.py with 9 exported functions: extract_json, call_with_backoff, load_checkpoint, save_checkpoint, batch_or_direct, make_batch_request, submit_batch, poll_batch, parse_batch_results"
  - "scripts/preflight.py with run_preflight() validating ANTHROPIC_API_KEY, php, phpcs, WordPress-Extra"
  - "tests/test_utils.py with 11 unit tests covering all utils functions"
  - "tests/test_preflight.py with 4 unit tests covering all preflight checks"
  - "checkpoints/ directory with .gitkeep and gitignore rules"
affects:
  - "scripts/phase1_judge.py (should import call_with_backoff, extract_json from utils)"
  - "All phase2, phase3 scripts (should import from scripts.utils)"
  - "02-repo-list plan (tests infrastructure shared with this plan)"

# Tech tracking
tech-stack:
  added: [pytest, anthropic]
  patterns:
    - "4-strategy JSON extraction: raw parse -> ```json fence -> plain fence -> outermost braces"
    - "Exponential backoff with retry-after header reading and 10% jitter"
    - "Atomic checkpoint writes via tmp+rename"
    - "Batch API routing at threshold=50 items"
    - "TDD: RED (failing imports) -> GREEN (all pass)"

key-files:
  created:
    - scripts/__init__.py
    - scripts/utils.py
    - scripts/preflight.py
    - checkpoints/.gitkeep
    - .gitignore
    - tests/__init__.py
    - tests/test_utils.py
    - tests/test_preflight.py
  modified: []

key-decisions:
  - "Batch threshold hardcoded at 50 (BATCH_THRESHOLD constant) matching PIPE-04 spec"
  - "Checkpoint uses phase name as key so multiple pipeline stages can coexist without collision"
  - "preflight.py catches FileNotFoundError from subprocess.run so tests pass on machines without php/phpcs installed"
  - "datetime.now(timezone.utc) used instead of deprecated utcnow()"

patterns-established:
  - "Test isolation: all subprocess calls in preflight are mocked so tests pass without real tools installed"
  - "Checkpoint directory defaults to PROJECT_ROOT/checkpoints/, overridable via checkpoint_dir param for testing"
  - "call_with_backoff takes **kwargs and passes to client.messages.create for forward compatibility"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05]

# Metrics
duration: 25min
completed: 2026-03-26
---

# Phase 1 Plan 01: Shared Utilities and Pre-flight Summary

**extract_json with 4-strategy fallback chain, call_with_backoff with retry-after jitter, atomic checkpointing via tmp+rename, Batch API routing at threshold=50, and pre-flight validation for ANTHROPIC_API_KEY/php/phpcs/WordPress-Extra**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-26T03:30:00Z
- **Completed:** 2026-03-26T03:55:00Z
- **Tasks:** 2 (TDD: RED then GREEN)
- **Files modified:** 8

## Accomplishments

- 9-function utils.py covering JSON extraction, exponential backoff, atomic checkpoints, Batch API helpers, and routing decisions
- preflight.py validating all 4 environment requirements (API key, php, phpcs, WordPress-Extra standard)
- 15 passing unit tests with complete isolation via mocking (no real API or PHPCS needed to run tests)
- checkpoints/ directory wired into .gitignore so runtime state doesn't pollute git history

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test infrastructure and test stubs (RED phase)** - `c2936eb` (test)
2. **Task 2: Implement scripts/utils.py and scripts/preflight.py (GREEN phase)** - `e5e82a4` (feat)

**Plan metadata:** pending (this commit)

## Files Created/Modified

- `scripts/__init__.py` - Empty package init, makes `from scripts.utils import` work
- `scripts/utils.py` - 9 utility functions for all pipeline scripts to import
- `scripts/preflight.py` - Pre-flight validation with sys.exit(1) on any failure
- `checkpoints/.gitkeep` - Sentinel file so checkpoints/ dir exists in git
- `.gitignore` - Ignores checkpoints/*.json, checkpoints/*.tmp, Python artifacts
- `tests/__init__.py` - Empty package init for test discovery
- `tests/test_utils.py` - 11 unit tests: extract_json (6), checkpoint (2), backoff (2), routing (1)
- `tests/test_preflight.py` - 4 unit tests: missing phpcs, missing API key, missing WP standard, all pass

## Decisions Made

- Batch threshold is 50 (not 49, not 51) — exact spec from PIPE-04 "items >= 50 use Batch API"
- Atomic checkpoint write uses Path.rename() (not shutil.move) — same filesystem, guaranteed atomic on Linux
- preflight.py checks API key FIRST before subprocess calls — fail fast on most common missing config
- FileNotFoundError handling in subprocess calls ensures tests pass on CI machines without php/phpcs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] preflight.py subprocess calls crash when php/phpcs not installed**
- **Found during:** Task 2 (GREEN phase, test run)
- **Issue:** `subprocess.run(["php", "--version"])` raises `FileNotFoundError` when php binary is absent; test `test_missing_api_key` (which patches env but not subprocess) crashed
- **Fix:** Wrapped all subprocess.run calls in try/except FileNotFoundError, treating missing binary as returncode != 0
- **Files modified:** scripts/preflight.py, tests/test_preflight.py (added subprocess mock to test_missing_api_key)
- **Verification:** All 15 tests pass after fix
- **Committed in:** e5e82a4 (Task 2 commit)

**2. [Rule 1 - Bug] datetime.utcnow() deprecation in save_checkpoint**
- **Found during:** Task 2 (GREEN phase, test warnings)
- **Issue:** Python 3.13 emits DeprecationWarning for datetime.utcnow()
- **Fix:** Changed to datetime.now(timezone.utc) with proper timezone import
- **Files modified:** scripts/utils.py
- **Verification:** 15 passed with no warnings
- **Committed in:** e5e82a4 (Task 2 commit)

**3. [Rule 3 - Blocking] anthropic package not installed**
- **Found during:** Task 2 (first test run)
- **Issue:** `import anthropic` failed — package absent from system Python
- **Fix:** Ran `pip install anthropic --system`
- **Files modified:** none (system package install)
- **Verification:** import succeeds, all tests pass
- **Committed in:** n/a (system package)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 3 blocking)
**Impact on plan:** All fixes necessary for correctness and test isolation. No scope creep.

## Issues Encountered

- `rtk git add` reported compact success output but files were committed via a different session's commit (e5e82a4). All task files verified committed via `git ls-files --cached`.

## User Setup Required

None - no external service configuration required for the utility layer. Tests run entirely with mocks.

## Next Phase Readiness

- All 9 utils functions available for import: `from scripts.utils import extract_json, call_with_backoff, ...`
- `scripts/preflight.py` is the standard entry gate before any pipeline run
- Plan 01-02 (repo list management) can proceed — shared test infrastructure is in place
- Plan 01-03+ scripts should replace inline JSON parsing and rate limiting with utils imports

---
*Phase: 01-pipeline-ready*
*Completed: 2026-03-26*

## Self-Check: PASSED

- scripts/utils.py: FOUND
- scripts/preflight.py: FOUND
- tests/test_utils.py: FOUND
- tests/test_preflight.py: FOUND
- checkpoints/.gitkeep: FOUND
- .gitignore: FOUND
- commit c2936eb (RED phase): FOUND
- commit e5e82a4 (GREEN phase): FOUND
- pytest tests/test_utils.py tests/test_preflight.py: 15 passed
