---
phase: 11
slug: compression-packaging
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-08
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: 11-RESEARCH.md § Validation Architecture. Scope: TRAINING-FREE Sieve (locked 2026-07-08).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (defaults; `tests/conftest.py` exists, no root config file) |
| **Config file** | none — Wave 0 keeps defaults |
| **Quick run command** | `pytest tests/test_concentration.py tests/test_bootstrap_ci.py tests/test_jaccard_stability.py -q` |
| **Full suite command** | `pytest tests/ -q` (362 passing at Phase 7 close; add alongside, never replace) |
| **Estimated runtime** | quick ~10s · full ~2-3 min |

---

## Sampling Rate

- **After every task commit:** `pytest tests/test_<new-file>.py -x` (fast, mocked)
- **After every plan wave:** `pytest tests/ -q` (full suite — catches regressions in reused Phase 7 code)
- **Before `/gsd-verify-work`:** full suite green PLUS at least one real (non-mocked) DGX profiling run and one real wp-bench k-sweep pass — mocked tests cannot validate routing/quality on real hardware
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | SIEVE-01 | — | Cross-seed routing profile produced; protected experts subset-verified in retained set | unit + integration | `pytest tests/test_sieve_protected_retention.py -x` | ❌ Wave 0 | pending |
| TBD | TBD | 0 | SIEVE-02 | — | Training-free path: N/A rationale documented (data-routing spec applies only to superseded training path) | doc check | — | n/a | pending |
| TBD | TBD | — | SIEVE-03 | — | Ratio traceability (30/70) to Phase 4/7 artifacts — no new decision | manual-only | — (traceability check against closed decisions, not executable behavior) | n/a | pending |
| TBD | TBD | 1+ | SIEVE-04 | — | K-sweep at 13/32/64 executes; per-k wp-bench + judge-rho records emitted | integration | `pytest tests/test_sieve_ksweep_mask.py -x` (mocked) + real DGX run | ❌ Wave 0 | pending |
| TBD | TBD | 1+ | SIEVE-05 | — | TOST gate (epsilon=2pp) declares optimal k; protected experts retained at optimal k | unit | `pytest tests/test_tost_gate.py -x` | ❌ Wave 0 | pending |

---

## Wave 0 Gaps

- [ ] `tests/test_sieve_protected_retention.py` — SIEVE-01 (protected-expert subset check on new hot/cold classification)
- [ ] `tests/test_tost_gate.py` — SIEVE-05 (TOST logic; model on `test_bootstrap_ci.py` conventions)
- [ ] `tests/test_sieve_ksweep_mask.py` — SIEVE-04 (expert-masking-at-inference logic, mockable without GPU)
- [ ] Cross-seed overlap script + test — resolves Open Question 2 (one Sieve profile vs union-of-3)
- [ ] Environment pre-check script (disk/memory, pattern per `wp-moe.md` `/proc/meminfo` pre-check) — Pitfall 4/OQ3 (disk verified 1.6T free 2026-07-08; script still guards reruns)
