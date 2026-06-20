---
phase: 10
slug: rl-comparative-evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `10-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing — 471 tests passing as of Phase 9) |
| **Config file** | `pyproject.toml` / `pytest.ini` (existing) |
| **Quick run command** | `pytest tests/test_bootstrap_gate.py tests/test_rlev02_report.py -x -q --tb=short` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~60s quick / full suite minutes |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_bootstrap_gate.py tests/test_rlev02_report.py -x -q`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds (quick), full suite at wave merges

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-W0a | 01 | 0 | RLEV-01 | — | N/A | unit | `pytest tests/test_bootstrap_gate.py -x` | ❌ W0 | ⬜ pending |
| 10-01-W0b | 01 | 0 | RLEV-01 | — | N/A | unit | `pytest tests/test_bootstrap_gate.py::test_spearman_improvement -x` | ❌ W0 | ⬜ pending |
| 10-01-W0c | 01 | 0 | RLEV-01 | — | N/A | unit | `pytest tests/test_bootstrap_gate.py::test_wpbench_gate -x` | ❌ W0 | ⬜ pending |
| 10-01-W0d | 01 | 0 | RLEV-01 | — | N/A | unit | `pytest tests/test_bootstrap_gate.py::test_pertask_floor -x` | ❌ W0 | ⬜ pending |
| 10-01-W0e | 01 | 0 | RLEV-02 | — | N/A | unit | `pytest tests/test_rlev02_report.py -x` | ❌ W0 | ⬜ pending |
| 10-01-W0f | 01 | 0 | RLEV-02 | — | N/A | unit | `pytest tests/test_rlev02_report.py::test_conjunctive_gate -x` | ❌ W0 | ⬜ pending |
| 10-01-W0g | 01 | 0 | RLEV-02 | T-10-01 | anti-hack hi_perturbed_rl < lo_clean_v12 (no reward-hack) | unit | `pytest tests/test_rlev02_report.py::test_antihack_gate -x` | ❌ W0 | ⬜ pending |
| 10-01-W0h | 01 | 0 | RLEV-02 | — | N/A | unit | `pytest tests/test_rlev02_report.py::test_jaccard_retention -x` | ❌ W0 | ⬜ pending |
| 10-01-Wre | 01 | 1 | RLEV-01 | — | N/A | integration | `pytest tests/test_eval_integration.py -x` | ✅ (Phase 4.4) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/bootstrap_gate.py` — CI-aware bootstrap-lower-bound gate (wraps `bootstrap_ci` from `scripts/compute_concentration.py`; adds pair-level Spearman-improvement bootstrap for the primary judge gate). Wave 0 deliverable.
- [ ] `scripts/rlev02_report.py` — RLEV-02 report generator + five-part conjunctive gate aggregator. Wave 0 deliverable.
- [ ] `tests/test_bootstrap_gate.py` — RLEV-01 dim CI gate, Spearman improvement, wp-bench aggregate gate, per-task floor (against dry-run fixtures).
- [ ] `tests/test_rlev02_report.py` — RLEV-02 report parsing (`reward_breakdown`), conjunctive gate, anti-hack CI comparison, Jaccard retention.

*Existing test infrastructure covers `eval_gen`, `eval_judge`, `rubric_scorer` — those are NOT Wave 0 gaps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live Tinker RL checkpoint export + real comparison numbers | RLEV-01, RLEV-02 | Credential-gated Phase 9 live run; not reproducible in CI | After Phase 9 live run lands: merge RL checkpoint via `merge_tinker_v3.py`, serve via vLLM, run full eval, run `bootstrap_gate.py` + `rlev02_report.py` |
| Human review of full v1.2-SFT-vs-RL comparison table (5 sub-gates + 9 dims + wp-bench) | RLEV-02 (D-10-04) | Conjunctive gate + human sign-off before v3.0 declared | Present comparison table; reviewer confirms all 5 sub-gates pass or surfaces regression + suggested fix |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (`bootstrap_gate.py`, `rlev02_report.py` + their tests)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
