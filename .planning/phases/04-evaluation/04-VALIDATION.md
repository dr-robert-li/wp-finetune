---
phase: 04
slug: evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-03
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (Python scripts; no existing test config) |
| **Config file** | none — Wave 0 creates test files |
| **Quick run command** | `python -m pytest tests/test_eeff.py tests/test_triage.py -x -q` |
| **Full suite command** | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` |
| **Estimated runtime** | ~5 seconds (unit tests), ~30 min (full eval per adapter) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` (on first available results)
- **Before `/gsd:verify-work`:** Full suite must be green + triage_decision.md written
- **Max feedback latency:** 5 seconds (unit), 30 min (integration)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | EVAL-05, GATE-02 | unit | `pytest tests/test_eeff.py -x -q` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | GATE-02 | unit | `pytest tests/test_triage.py -x -q` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | EVAL-01, EVAL-02, EVAL-03, EVAL-04 | integration | `python -m eval.eval_gate --results-dir output/eval_triage/ratio_{r}` | ❌ W0: output dir not yet created | ⬜ pending |
| 04-02-02 | 02 | 2 | EVAL-04 | smoke | `python -c "from eval.eval_gate import run_gate; print('OK')"` | ✅ eval/ exists | ⬜ pending |
| 04-03-01 | 03 | 3 | EVAL-05, GATE-02 | human | Human reviews triage_decision.md | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eeff.py` — unit tests for E_eff computation (entropy, effective expert count, JSONL output)
- [ ] `tests/test_triage.py` — unit tests for GATE-02 elimination logic (hard gates, 5pp rule)
- [ ] `output/eval_triage/` — create directory structure before eval loop
- [ ] `wp-bench/` — clone `github.com/WordPress/wp-bench` and run setup
- [ ] Verify vLLM LoRA loading works with adapters that have `modules_to_save` tensors

---

## Requirement → Test Coverage

| Req ID | Description | Test | Verified By |
|--------|-------------|------|-------------|
| EVAL-01 | PHPCS pass rate >95% | Integration: eval_gen.py output | eval_gate exit code |
| EVAL-02 | Judge Spearman >0.85 | Integration: eval_judge.py output | eval_gate exit code |
| EVAL-03 | Security pass rate >98% | Integration: eval_gen.py security check | eval_gate exit code |
| EVAL-04 | Run via DGX Toolbox containers | Smoke: eval scripts importable in container | dgx.execute smoke test |
| EVAL-05 | All 3 gates pass for ≥1 ratio | Integration: eval_gate.run_gate() exits 0 | triage_decision.md |
| GATE-02 | High bar elimination, low bar continuation | Unit: test_triage.py edge cases | pytest exit code |
