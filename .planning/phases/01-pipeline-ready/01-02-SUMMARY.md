---
phase: 01-pipeline-ready
plan: 02
subsystem: data-pipeline
tags: [python, csv, yaml, pytest, wordpress, quality-filtering]

requires: []

provides:
  - scripts/csv_to_repos.py — CSV-to-repos.yaml conversion script with filter, tier assignment, and path inference
  - config/repos.yaml — Fully populated repository config with 1 core + 49 plugins + 6 themes
  - tests/test_csv_to_repos.py — 11-test suite covering all filter/tier/schema behaviors
  - tests/fixtures/sample_plugins.csv — 5-row test fixture (2 pass, 3 excluded)
  - tests/fixtures/sample_themes.csv — 5-row test fixture (2 pass, 3 excluded)

affects:
  - phase1_clone.py consumes repos.yaml (name, url, quality_tier, paths, skip_paths keys)
  - Future phases that depend on repository diversity for training data

tech-stack:
  added: [pyyaml, csv (stdlib), pytest]
  patterns:
    - TDD red-green cycle with fixture-driven tests
    - Quality-tier assignment from multi-signal CSV data (vulns + rating)
    - Tag-based path filter inference via keyword matching dict
    - Filter guard with explicit "+" suffix stripping for numeric fields

key-files:
  created:
    - scripts/csv_to_repos.py
    - scripts/__init__.py
    - tests/test_csv_to_repos.py
    - tests/__init__.py
    - tests/fixtures/sample_plugins.csv
    - tests/fixtures/sample_themes.csv
    - config/repos.yaml
  modified: []

key-decisions:
  - "quality_tier=trusted requires zero total_known_vulns AND zero unpatched AND rating >= 90; otherwise assessed"
  - "active_installs '+' suffix stripped before int() conversion (e.g. '10000000+' -> 10000000)"
  - "github_url must start with 'https://github.com/' — git clone access required"
  - "If >100 repos pass filter, top 100 by active_installs retained"
  - "STANDARD_SKIP list = vendor, node_modules, tests, test, assets, css, js"
  - "WordPress Core hardcoded as first entry with quality_tier=core — not sourced from CSV"

patterns-established:
  - "filter_row: single function with explicit int/float parsing guards for missing/empty fields"
  - "infer_path_filters: keyword-in-tag matching, first match wins, _default fallback"
  - "make_entry: slug as name key, ensure .git suffix on github_url"

requirements-completed: [REPO-01, REPO-02, REPO-03, REPO-04]

duration: 2min
completed: 2026-03-26
---

# Phase 1 Plan 02: CSV-to-Repos Converter Summary

**Automated WordPress repo selection from ranked CSVs: 49 plugins + 6 themes with quality_tier (trusted/assessed/core) and tag-based path filters, replacing manual curation**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-26T03:38:33Z
- **Completed:** 2026-03-26T03:40:52Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 7

## Accomplishments

- Created `scripts/csv_to_repos.py` with four exported functions: `filter_row`, `assign_quality_tier`, `infer_path_filters`, `convert_csvs_to_repos`
- Generated `config/repos.yaml` with 1 core + 49 plugins + 6 themes from real CSV data, all meeting REPO-01 through REPO-04 requirements
- Built 11-test pytest suite with fixtures covering pass/fail filter cases, both quality tiers, schema validation, '+' suffix parsing, and tag-based path inference

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test fixtures and test_csv_to_repos.py (RED phase)** - `4136589` (test)
2. **Task 2: Implement csv_to_repos.py and generate repos.yaml (GREEN phase)** - `e5e82a4` (feat)

_TDD tasks: RED commit (failing tests) → GREEN commit (implementation + generated repos.yaml)_

## Files Created/Modified

- `scripts/csv_to_repos.py` — Filter/tier/path inference functions + CLI entry point that writes repos.yaml
- `scripts/__init__.py` — Package init enabling `from scripts.csv_to_repos import ...` in tests
- `config/repos.yaml` — 1 core + 49 plugins + 6 themes with quality_tier, paths, skip_paths, description
- `tests/test_csv_to_repos.py` — 11 pytest tests covering all behaviors from VALIDATION.md map
- `tests/__init__.py` — Package init for test discovery
- `tests/fixtures/sample_plugins.csv` — 5-row fixture: rows 1-2 pass, rows 3-5 excluded (no github/low installs/unpatched)
- `tests/fixtures/sample_themes.csv` — 5-row fixture: rows 1-2 pass, rows 3-5 excluded (no github/low rating/unpatched)

## Decisions Made

- Used slug field as the `name` key in repos.yaml (matches downstream phase1_clone.py pattern)
- `infer_path_filters` uses first-match-wins on TAG_TO_PATH_FILTERS dict ordering — "page builder" checked before "seo" etc.
- Real data produced 49 plugins (below 100 cap) so sort+truncation path was not needed in this run; code handles it when >100 pass
- `scripts/__init__.py` added as deviation (Rule 3: blocking import error) — pytest could not import from `scripts` package without it

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added scripts/__init__.py to enable package import**
- **Found during:** Task 2 (GREEN phase, running pytest)
- **Issue:** `ModuleNotFoundError: No module named 'scripts.csv_to_repos'` — Python treats `scripts/` as a namespace package but pytest's import mode requires an `__init__.py` for consistent import resolution
- **Fix:** Created empty `scripts/__init__.py`
- **Files modified:** scripts/__init__.py
- **Verification:** All 11 tests pass after addition
- **Committed in:** e5e82a4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking import)
**Impact on plan:** Necessary for test discovery. No scope creep.

## Issues Encountered

- RTK filter cached an older pytest log in tee output, making it appear tests still failed after fix. Confirmed 11 tests passing via RTK summary line ("Pytest: 11 passed") on subsequent runs.

## User Setup Required

None - no external service configuration required. CSV data files are already present at `/home/robert_li/Desktop/data/wp-finetune-data/`.

## Next Phase Readiness

- `config/repos.yaml` ready for `scripts/phase1_clone.py` consumption (name, url, quality_tier, paths, skip_paths keys present)
- 49 plugins + 6 themes provides sufficient diversity for training data generation
- Plan 01-01 (pipeline hardening utilities) can proceed independently

---
*Phase: 01-pipeline-ready*
*Completed: 2026-03-26*
