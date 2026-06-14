---
phase: 07
slug: router-profiling-protected-expert-set
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-14
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing repo infra; tests/ + tests/phase4_4/ already present) |
| **Config file** | none — repo-root pytest discovery (existing) |
| **Quick run command** | `pytest tests/ -q` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~45 seconds (synthetic/mock-data unit tests, no GPU, no model load) |

---

## Sampling Rate

- **After every task commit:** Run the task's scoped `pytest tests/test_<file>.py -x -q`
- **After every plan wave:** Run `pytest tests/ -q` (full suite)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | PROF-01, PROF-02, PROF-03 | T-07-01 / T-07-03 | Profiler writes only to reasoning-merged-v4 dir (path-collision guard); malformed JSONL skip-and-log | unit | `pytest tests/test_routing_collector.py tests/test_jaccard_stability.py tests/test_eeff.py -x -q` | ✅ | ⬜ pending |
| 07-01-02 | 01 | 1 | PROF-04, D-09 | — | CI-aware ci_lower disposition (not point-bar); non-empty D-08 join assertion | unit | `pytest tests/test_concentration.py tests/test_bootstrap_ci.py -x -q` | ✅ | ⬜ pending |
| 07-01-03 | 01 | 1 | PROF-04 (D-03/D-04 mask) | T-07-02 | Mask/report contain counts + bool only — no secrets/PII | unit | `pytest tests/test_protected_mask.py -x -q` | ✅ | ⬜ pending |
| 07-02-01 | 02 | 2 | PROF-03, GATE-01 (output) | T-07-04 / T-07-06 | Baseline JSONL untouched (`git diff --stat`); D-08 join non-empty (no silent zero-match) | output-schema | `python3 -c "import numpy as np,json,pathlib; d=pathlib.Path('output/profiling/reasoning-merged-v4'); m=np.load(d/'protected_expert_mask.npy'); assert m.shape==(48,128) and m.dtype==bool; rr=[json.loads(l) for l in (d/'routing_report.jsonl').read_text().splitlines() if l.strip()]; assert len(rr)==48 and all(r.get('ratio')=='30_70' for r in rr); print('OK')"` | ❌ (run-produced) | ⬜ pending |
| 07-02-02 | 02 | 2 | PROF-05, GATE-01 | — | N/A rationale prose, no fabricated multi-ratio matrix | doc-content | `python3 -c "t=open('output/profiling/reasoning-merged-v4/routing_report_rationale.md').read(); assert 'PROF-05' in t and 'GATE-01' in t and '30/70' in t and ('NO_SURVIVORS' in t.upper() or 'no_survivors' in t.lower()); assert 'final_dataset/ratio_30_70' in t; print('rationale OK')"` | ❌ (run-produced) | ⬜ pending |
| 07-02-03 | 02 | 2 | GATE-01 (sign-off) | — | Human review of E_eff delta + protected set | checkpoint:human-verify | — (manual, see Manual-Only) | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*PROF-01/02 → test_routing_collector.py; PROF-03 (subsample-vs-FULL Jaccard) → test_jaccard_stability.py + 07-02-01 output gate; PROF-04 → test_concentration.py; D-09 → test_bootstrap_ci.py; D-03/D-04 → test_protected_mask.py; PROF-05/GATE-01 → 07-02-02 rationale + 07-02-03 sign-off.*

---

## Wave 0 Requirements

*Existing pytest infrastructure (tests/, tests/phase4_4/) covers all phase requirements. No separate Wave 0 plan — Plan 07-01 uses in-task TDD (`tdd="true"` + `<behavior>` blocks): each task writes its synthetic-data test stubs alongside the implementation, so all five new test files (test_routing_collector.py, test_jaccard_stability.py, test_concentration.py, test_bootstrap_ci.py, test_protected_mask.py) are created within the wave-1 tasks. pytest is already installed; no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Merged-model GPU forward-pass profiling run (full-set + 10% subsample) producing routing_report.jsonl / mask / sensitivity artifacts | PROF-01..05 (runtime) | Requires the DGX ngc-pytorch container + ~60 GB VRAM to load the 30B merged checkpoint; cannot run in CI or on CPU (CUDA guard exits 2). The 07-02-01 one-liner *validates* the artifacts but the run that *produces* them is GPU-bound. | Invoke wp-finetune:run-profiling skill on DGX; confirm "Registered 48 hooks" startup log; then run the 07-02-01 output-schema one-liner against the produced artifacts. |
| Human sign-off on E_eff comparison + protected expert set | GATE-01 | Visual/judgment review of routing-shift plausibility (modest deltas, conservative mask, sensitivity spread) before closing the phase for Phases 11/13 | Open concentration_report.json (D-08 delta, 48 rows), protected_expert_mask.json, sensitivity_table.json, routing_report_rationale.md; confirm per task 07-02-03 how-to-verify steps; type "approved". |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (07-02-03 is the only manual task; it follows two automated-gated tasks)
- [x] Wave 0 covers all MISSING references (in-task TDD stubs cover all five new test files; no MISSING automated commands)
- [x] No watch-mode flags (all commands use `-q`/`-x`; no `--watch`)
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-14
