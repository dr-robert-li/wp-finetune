---
phase: quick
plan: 260620-bwf
subsystem: documentation
tags: [docs, changelog, journal, readme, project-status, phase-7, phase-8]
dependency_graph:
  requires: [08-04-SUMMARY, 08-REVIEW, STATE.md, ROADMAP.md]
  provides: [CHANGELOG.md, JOURNAL.md, README.md, PROJECT.md]
  affects: []
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified:
    - CHANGELOG.md
    - JOURNAL.md
    - README.md
    - PROJECT.md
decisions: []
metrics:
  duration: ~15min
  completed: 2026-06-20
---

# Quick Task 260620-bwf: Update All Docs — Changelog, Journal, README, PROJECT.md

**One-liner:** Synced CHANGELOG [Unreleased], JOURNAL, README Project Status table, and PROJECT.md Current Status checklist with Phase 7 closure (1,480 experts, CI-aware Jaccard gate, council-approved 2026-06-19) and Phase 8 reward infrastructure (composite 70/30 pipeline, security terminal gate, MO-GRPO, VeRPO, 424 tests, anti-hack eval set).

## Tasks Completed

| Task | Name | Status | Files Modified |
|------|------|--------|----------------|
| 1 | CHANGELOG + JOURNAL | Done | CHANGELOG.md, JOURNAL.md |
| 2 | README + PROJECT.md | Done | README.md, PROJECT.md |
| 3 | Stage + commit + push | Done | (all 4 docs + plan dir) |

## Changes Made

### CHANGELOG.md
Added new Added/Fixed subsections at the TOP of the existing `[Unreleased]` section (before pre-existing D-03 entries). Covers:
- Phase 7: CI-aware Jaccard gate (PROF-03, `jaccard_ci_lower=0.9426≥0.94`), 1,480 protected experts, E_eff concentration (PROF-04), L45-47 broadening accepted, mask immutable + shippable
- Phase 8 Added: composite 70/30 pipeline, security TERMINAL fail-CLOSED gate (SC2), MO-GRPO normalization, VeRPO difficulty-weighted partial credit (D1_wpcs+D5_wp_api, 59 IDs), `judge_score_single()` RC-A wrapper, injectable recalibration-offset loader (+3.58, D-V4-09), anti-hack eval set (3-axis, CI-aware CR-03 fix, fixture-backed report), 424-test TDD suite
- Phase 8 Fixed: CR-01 (phpcs startup assertion), CR-03 (combined-group normalization), CR-02 (module-relative path anchor)

### JOURNAL.md
Prepended `## 2026-06-19` entry before existing `## 2026-06-15` entry (after `---` separator, line 5). Topics: Phase 7 council sign-off + mask immutability; Phase 8 fail-CLOSED security gate design lesson (CR-01 catch); judge recalibration inheritance delivered (+3.58, D-V4-09, promised in 06-14 entry); MO-GRPO normalization as reward-hacking guardrail (design intent); live anti-hack UAT deferred = deferred, not incomplete.

### README.md
Project Status table:
- v1.2 row: `**Next**` → `**Complete**`
- v2.0 row: rebuilt from old "MoE-Sieve Phases 7-9" → v2.0 RL Alignment with per-phase rows: Phase 7 Complete, Phase 8 Complete, Phase 9 Next, Phase 10 Planned
- v3.0 row: rebuilt from old "GRPO & Deploy Phases 10-14" → v3.0 MoE-Sieve, Pruning & Packaging Phases 11-15 all Planned

Current line: replaced stale "re-execution in progress / Phase 1 re-judging" with true current position (v1.2 promoted 2026-06-14, Phase 7 closed 2026-06-19, Phase 8 complete, Phase 9 GSPO next).

### PROJECT.md
- L96 "Not yet started. Next milestone step." → replaced with Phase C complete note + promoted model path + waiver
- Current Status checklist expanded: marked Phase C complete, added Phase 7 + Phase 8 complete rows, Phase 9 as next step, Phase 10 planned, Phase D/E groupings corrected to v3.0 Phases 11-15

## Deviations from Plan

None. Plan executed exactly as written. One design clarification: v2.0/v3.0 labels in README/PROJECT required content swap (not just renumbering) since old docs had v2.0=MoE-Sieve / v3.0=GRPO; ROADMAP canonical order is v2.0=RL Alignment (7-10) / v3.0=MoE-Sieve+Pruning+Packaging (11-15). Applied full content+number rebuild per advisor guidance.

## Known Stubs

None.

## Threat Flags

None — documentation-only changes, no source code modified.

## Self-Check: PASSED

- CHANGELOG.md: Phase 7 + Phase 8 Added/Fixed sections present at top of [Unreleased]
- JOURNAL.md: `## 2026-06-19` entry exists before `## 2026-06-15` entry
- README.md: v1.2 shows Complete; v2.0 shows RL Alignment Phases 7-10; v3.0 shows MoE-Sieve Phases 11-15
- PROJECT.md: Phase C note updated; Current Status checklist has Phase 7, Phase 8 complete rows and Phase 9 as next
