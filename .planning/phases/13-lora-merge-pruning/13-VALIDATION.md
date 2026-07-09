---
phase: 13
slug: lora-merge-pruning
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-10
---

# Phase 13 — Validation Strategy

> Source: 13-RESEARCH.md § Validation Architecture. Scope: weight-level expert pruning
> (AIMER primary, REAP conditional), gate-before-remove, no training.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 via `.venv-tinker/bin/python -m pytest` (system pytest broken — NEVER use it) |
| **Config file** | none — plain `tests/` auto-discovery |
| **Quick run command** | `.venv-tinker/bin/python -m pytest tests/test_aimer_prune.py tests/test_reap_prune.py tests/test_prune_overlap.py -x -q` |
| **Full suite command** | `.venv-tinker/bin/python -m pytest tests/ -q` |

---

## Sampling Rate

- **Per task commit:** quick unit run (synthetic fixtures, no GPU/model load; module-level `pytest.importorskip` Wave-0 pattern)
- **Per wave merge:** full suite + ≥1 real-hardware gated-eval smoke (Phase 11 precedent — masking/scoring bugs don't surface in synthetic tests)
- **Phase gate:** full suite green + AIMER@25% gated eval (both models) with a real comparison table before `/gsd-verify-work`

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists | Status |
|---|---|---|---|---|---|
| MERGE-01 | Traceability record; merged checkpoints exist + load | manual/doc | assert paths exist | n/a | pending |
| PRUNE-01 | `compute_aimer_scores()` → [48,128] float, scale-invariant, deterministic | unit | `pytest tests/test_aimer_prune.py -x -q` | ❌ Wave 0 | pending |
| PRUNE-01 | AIMER keep-masks never drop protected experts (reuse masking contract) | unit | `pytest tests/test_sieve_ksweep_mask.py -x -q` | ✅ existing | pending |
| PRUNE-02 | `compute_reap_scores()` → [48,128] from synthetic forward-hook fixture (no GPU) | unit | `pytest tests/test_reap_prune.py -x -q` | ❌ Wave 0 | pending |
| PRUNE-03 | Gated eval bars: wp-bench ≥ 0.4484−2pp; judge rho ≥ 0.8075−0.052; parse-rate ≥ 95% | integration (real HW) | sieve_ksweep_run-style driver on AIMER/REAP masks | ❌ Wave 0 | pending |
| PRUNE-04 | Per-layer Jaccard between AIMER and REAP keep-masks at matched ratios | unit | `pytest tests/test_prune_overlap.py -x -q` | ❌ Wave 0 | pending |
| PRUNE-05 | Selection rule (bars + D2_security floor) correct on synthetic gate table | unit | `pytest tests/test_prune_selection.py -x -q` | ❌ Wave 0 | pending |
| PRUNE-06 | Physical surgery: uniform per-layer expert count, router renorm, model loads + coherent output | unit + manual smoke | `pytest tests/test_prune_physical.py -x -q` + generate() smoke | ❌ Wave 0 | pending |

---

## Wave 0 Gaps

- [ ] `tests/test_aimer_prune.py` (PRUNE-01, synthetic weights)
- [ ] `tests/test_reap_prune.py` (PRUNE-02, synthetic hook fixture)
- [ ] `tests/test_prune_overlap.py` (PRUNE-04, synthetic masks)
- [ ] `tests/test_prune_selection.py` (PRUNE-05, synthetic gate table)
- [ ] `tests/test_prune_physical.py` (PRUNE-06, synthetic small tensors)
- Framework install: none needed
