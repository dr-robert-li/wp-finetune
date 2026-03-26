---
phase: 2
slug: dataset-production
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (confirmed in Phase 1) |
| **Config file** | none — invoked via `pytest tests/` from project root |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DATA-01 | unit | `pytest tests/test_pipeline_integration.py::test_clone_checkpoint_skip -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DATA-02 | unit | `pytest tests/test_pipeline_integration.py::test_extract_checkpoint_skip -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | DATA-03 | unit | `pytest tests/test_utils.py -x` | ✅ | ⬜ pending |
| 02-01-04 | 01 | 1 | DATA-03 | unit | `pytest tests/test_config.py::test_judge_threshold_v2 -x` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 1 | DATA-05 | unit | `pytest tests/test_phase2_mutate.py::test_phpcs_required -x` | ❌ W0 | ⬜ pending |
| 02-01-06 | 01 | 1 | DATA-06 | unit | `pytest tests/test_config.py::test_rejection_templates_exist -x` | ❌ W0 | ⬜ pending |
| 02-01-07 | 01 | 1 | DATA-08 | unit | `pytest tests/test_phase2_judge_dataset.py::test_rate_limiting -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | DATA-10 | unit | `pytest tests/test_export.py::test_gen_judge_ratio -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | DATA-10 | unit | `pytest tests/test_export.py::test_php_lint_validation -x` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 2 | DATA-11 | unit | `pytest tests/test_export.py::test_metadata_fields -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline_integration.py` — covers DATA-01, DATA-02 (checkpoint skip behavior)
- [ ] `tests/test_config.py` — covers DATA-03 threshold, DATA-06 rejection templates
- [ ] `tests/test_phase2_mutate.py` — covers DATA-05 PHPCS hard-fail guard
- [ ] `tests/test_phase2_judge_dataset.py` — covers DATA-08 rate limiting fix
- [ ] `tests/test_export.py` — covers DATA-10 ratio enforcement, PHP lint, DATA-11 metadata

*Existing `tests/test_utils.py` (15 passing tests) covers extract_json, call_with_backoff, checkpoints, batch routing.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full pipeline produces ≥10K examples | DATA-11 | Requires real API calls + repo cloning | Run full pipeline, check final_dataset/metadata.json |
| Batch API completes within 24h window | DATA-03 | Requires live Anthropic API | Submit batch, monitor completion |
| Spot-check 20 random examples for quality | DATA-11 | Requires human/Claude Code review | Sample from final_dataset/, review for correctness + teaching quality |
| Taxonomy coverage across all 12 categories | DATA-04 | Requires full pipeline run | Check gap_report.json after Phase 2 generate |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
