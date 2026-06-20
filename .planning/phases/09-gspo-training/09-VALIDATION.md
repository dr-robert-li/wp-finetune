---
phase: 9
slug: gspo-training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-20
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | {path or "none — Wave 0 installs"} |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/` |
| **Estimated runtime** | ~{N} seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** {N} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 1 | REQ-{XX} | T-{N}-01 / — | {expected secure behavior or "N/A"} | unit | `{command}` | ✅ / ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> Planner fills this map per task (GRPO-05/06/07/08). RL-loop math (GSPO sequence-IS ratio,
> RSPO stop-gradient floor, router-shift / KL-sample-train computation, protected-expert
> retention check, judge-reward noise/CV measurement) is unit-testable on synthetic tensors
> WITHOUT a live Tinker run — these are the Nyquist-sampled invariants. Full Tinker training
> runs are manual (see below).

---

## Wave 0 Requirements

- [ ] `{tests/test_file.py}` — stubs for REQ-{XX}
- [ ] `{tests/conftest.py}` — shared fixtures
- [ ] `{framework install}` — if no framework detected

*Planner completes during Wave 0.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end RL training run on Tinker | GRPO-05/06 | Cloud GPU cost + wall-clock; cannot run in CI | Dry-run flag first, then real run; inspect per-step metrics JSONL |
| Auto-halt on router-shift/KL breach | GRPO-07/08 | Requires live rollout/train divergence | Verify halt fires from per-step monitor on threshold breach |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
