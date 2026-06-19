---
phase: 8
slug: reward-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-19
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `08-RESEARCH.md` § "Validation Architecture". All acceptance gates are **CI-aware** (report bootstrap CIs; lower bound must clear the bar) per D-09 / D-V4-10, reusing `scripts.compute_concentration.bootstrap_ci`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (repo-existing) |
| **Config file** | repo pytest convention (no new config — match existing tests/ layout) |
| **Quick run command** | `rtk cargo test`-equivalent → `python -m pytest tests/test_reward_pipeline.py -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~30–90 s (deterministic signals; judge component mocked in unit tests, live only in the integration test) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_reward_pipeline.py -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 08-01-* | 01 | 1 | GRPO-01 | — | judge offset read from artifact, not hardcoded | unit | `pytest tests/test_reward_pipeline.py -k offset_loader -q` | ❌ W0 | ⬜ pending |
| 08-01-* | 01 | 1 | GRPO-01 | — | `judge_score_single` reuses `_judge_create` thinking-off guard | unit | `pytest tests/test_reward_pipeline.py -k judge_single -q` | ❌ W0 | ⬜ pending |
| 08-02-* | 02 | 2 | GRPO-03 | — | within-group var-norm `(x-μ)/(σ+ε)`, ε floor on zero-variance | unit | `pytest tests/test_reward_pipeline.py -k mogrpo -q` | ❌ W0 | ⬜ pending |
| 08-02-* | 02 | 2 | GRPO-04 | — | VeRPO difficulty `1-pass_rate` on WP-standards subset only (D-08-06) | unit | `pytest tests/test_reward_pipeline.py -k verpo -q` | ❌ W0 | ⬜ pending |
| 08-03-* | 03 | 3 | GRPO-02 | T-08-SEC | CRITICAL_FLOOR_RULE(D2_security) → reward=0 as terminal override AFTER normalize+combine, fail-CLOSED (D-08-05) | unit | `pytest tests/test_reward_pipeline.py -k security_gate -q` | ❌ W0 | ⬜ pending |
| 08-03-* | 03 | 3 | GRPO-01 | — | composite = 70% verifiable (35 PHPCS / 35 VeRPO) + 30% judge; no single-signal dominance | unit | `pytest tests/test_reward_pipeline.py -k composite -q` | ❌ W0 | ⬜ pending |
| 08-03-* | 03 | 3 | GRPO-01..04 | T-08-SEC | 50-case integration incl. SC2 secure-fail-but-high-quality → reward exactly 0 | integration | `pytest tests/test_reward_pipeline_integration.py -q` | ❌ W0 | ⬜ pending |
| 08-04-* | 04 | 4 | GRPO-01 (D-11) | — | 45 perturbed adversarial cases (15/axis) score CI-aware below clean baseline: `hi_perturbed < lo_clean` | integration | `pytest tests/test_antihack.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_reward_pipeline.py` — unit stubs for GRPO-01..04 (offset loader, judge_single, MO-GRPO, VeRPO, security gate, composite)
- [ ] `tests/test_reward_pipeline_integration.py` — 50-case integration harness (known-good / known-bad / SC2 secure-fail)
- [ ] `tests/test_antihack.py` — anti-hack CI-aware gate (`hi_perturbed < lo_clean`)
- [ ] `tests/conftest.py` — fixtures: mock judge client, sample rollout group, sample gen+judge JSONL records, recalibration-artifact stub
- [ ] Confirm `scripts.compute_concentration.bootstrap_ci` importable for the anti-hack gate

*Existing pytest infrastructure covers the runner; new test files above are the Wave 0 deliverables.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live vLLM judge endpoint reachability | GRPO-01 | Requires the local judge server running; not available in CI | Integration test runs against the live endpoint when present; unit tests mock it. Verify endpoint fail-fast error path manually before Phase 9. |
| Anti-hack adversarial-case quality (semantic plausibility) | D-11 | Claude Code agents score candidates during set construction; the *judgment* of "is this a realistic hack" is human-auditable | Spot-check a sample of the 45 generated cases per axis. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
