---
phase: 1
slug: pipeline-ready
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (Wave 0 installs) |
| **Config file** | none — Wave 0 creates |
| **Quick run command** | `pytest tests/test_utils.py -x -q` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_utils.py -x -q`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | PIPE-01 | unit | `pytest tests/test_preflight.py::test_missing_phpcs -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | PIPE-01 | unit | `pytest tests/test_preflight.py::test_missing_api_key -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | PIPE-02 | unit | `pytest tests/test_utils.py::test_checkpoint_roundtrip -x` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | PIPE-02 | unit | `pytest tests/test_utils.py::test_checkpoint_atomic -x` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | PIPE-03 | unit (mock) | `pytest tests/test_utils.py::test_backoff_retries -x` | ❌ W0 | ⬜ pending |
| 01-01-06 | 01 | 1 | PIPE-03 | unit (mock) | `pytest tests/test_utils.py::test_backoff_retry_after -x` | ❌ W0 | ⬜ pending |
| 01-01-07 | 01 | 1 | PIPE-04 | unit | `pytest tests/test_utils.py::test_routing_threshold -x` | ❌ W0 | ⬜ pending |
| 01-01-08 | 01 | 1 | PIPE-05 | unit | `pytest tests/test_utils.py::test_extract_json -x` | ❌ W0 | ⬜ pending |
| 01-01-09 | 01 | 1 | PIPE-05 | unit | `pytest tests/test_utils.py::test_extract_json_failure -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | REPO-01 | unit | `pytest tests/test_csv_to_repos.py::test_core_preserved -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | REPO-02 | unit | `pytest tests/test_csv_to_repos.py::test_min_plugins -x` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | REPO-03 | unit | `pytest tests/test_csv_to_repos.py::test_min_themes -x` | ❌ W0 | ⬜ pending |
| 01-02-04 | 02 | 1 | REPO-04 | unit | `pytest tests/test_csv_to_repos.py::test_entry_schema -x` | ❌ W0 | ⬜ pending |
| 01-02-05 | 02 | 1 | REPO-04 | unit | `pytest tests/test_csv_to_repos.py::test_quality_tier_logic -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — package init
- [ ] `tests/test_utils.py` — stubs for PIPE-02, PIPE-03, PIPE-04, PIPE-05
- [ ] `tests/test_preflight.py` — stubs for PIPE-01
- [ ] `tests/test_csv_to_repos.py` — stubs for REPO-01 through REPO-04
- [ ] `tests/fixtures/sample_plugins.csv` — 5-row sample for CSV tests
- [ ] `tests/fixtures/sample_themes.csv` — 5-row sample for CSV tests
- [ ] `pip install pytest` — framework installation

*Existing infrastructure covers no phase requirements — all tests are new.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Batch API actually submits to Anthropic | PIPE-04 | Requires live API key and real batch | Submit 2-item batch, verify completion status |
| Killed script resumes from checkpoint | PIPE-02 | Requires process kill mid-run | Start phase1_judge on 10 items, kill at item 5, restart, verify items 1-5 skipped |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
