---
phase: 11-compression-packaging
plan: 01
subsystem: testing
tags: [pytest, moe-sieve, tost, disk-precheck, wave-0-stub]

# Dependency graph
requires: []
provides:
  - "scripts/sieve_env_precheck.py — disk/mem/statsmodels gates before s0/s2 merges"
  - "4 Wave-0 pytest scaffolds gating SIEVE-01/04/05 + cross-seed overlap implementation"
affects: [11-02-merge-checkpoints, 11-03-routing-profiles, 11-04-ksweep, 11-05-tost-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0 stub convention: pytest.importorskip('scripts.<future-module>') so new
       test files SKIP cleanly (never ERROR) while their target module is absent,
       flipping SKIP->RED->GREEN once a later wave lands the module"
    - "Per-test (not module-level) importorskip when a single test file mixes an
       already-existing-artifact assertion with future-module contract tests"

key-files:
  created:
    - scripts/sieve_env_precheck.py
    - tests/test_sieve_protected_retention.py
    - tests/test_tost_gate.py
    - tests/test_sieve_ksweep_mask.py
    - tests/test_sieve_cross_seed_overlap.py
  modified: []

key-decisions:
  - "Env pre-check reads MemAvailable and lists top-5 RSS via /proc/*/status parsing only (no psutil, no ps subprocess), matching the project's zero-extra-dependency convention"
  - "statsmodels.stats.weightstats.ttost_ind absence (confirmed: not installed) is recorded, non-blocking — plan 11-05 will hand-roll TOST"
  - "test_sieve_protected_retention.py mixes a today-runnable mask-artifact assertion with future-module retention-check tests using per-test importorskip, so the file doesn't fully SKIP despite depending partly on a module that doesn't exist yet"

patterns-established:
  - "Future sieve modules named scripts.sieve_protected_retention, scripts.tost_gate, scripts.sieve_expert_mask_inference, scripts.sieve_cross_seed_overlap — later waves must land these exact module/function names for the Wave-0 tests to flip green (retention_check, tost_equivalence, build_ksweep_mask, jaccard/pairwise_layer_jaccard)"

requirements-completed: [SIEVE-01, SIEVE-04, SIEVE-05]

coverage:
  - id: D1
    description: "sieve_env_precheck.py gates disk>=150GiB and mem>=70GiB, records statsmodels availability, exits 0 on current hardware"
    requirement: "SIEVE-01"
    verification:
      - kind: other
        ref: "python3 scripts/sieve_env_precheck.py (exit=0, JSON: disk_free_gib=1557.5, mem_available_gib=111.2, statsmodels_ttost_available=false, all_hard_gates_pass=true)"
        status: pass
      - kind: unit
        ref: "scripts/sieve_env_precheck.py __main__ self-check (_self_check asserts gate helpers on fake 10/150/500 GiB and 10/70/500 GiB values)"
        status: pass
    human_judgment: false
  - id: D2
    description: "4 Wave-0 pytest scaffolds collect cleanly; protected-retention mask-count assertion passes today; other tests SKIP pending waves 2-4; no regressions in existing suite"
    requirement: "SIEVE-04, SIEVE-05"
    verification:
      - kind: unit
        ref: ".venv-tinker/bin/python -m pytest tests/test_sieve_protected_retention.py tests/test_tost_gate.py tests/test_sieve_ksweep_mask.py tests/test_sieve_cross_seed_overlap.py -v (1 passed, 5 skipped, 0 errors)"
        status: pass
      - kind: integration
        ref: ".venv-tinker/bin/python -m pytest tests/ -q --ignore=tests/test_preflight.py (666 passed, 8 skipped, 7 pre-existing tinker-auth failures unrelated to this plan, no new collection errors)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-08
status: complete
---

# Phase 11 Plan 01: Wave-0 Sieve Scaffolding Summary

**Env pre-check script (disk/mem/statsmodels gates) plus 4 pytest Wave-0 scaffolds (importorskip stubs) gating SIEVE-01/04/05 before any MoE-Sieve GPU/disk work starts**

## Performance

- **Duration:** 25 min
- **Started:** 2026-07-08T07:12:00Z (approx, session start)
- **Completed:** 2026-07-08T07:37:10Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- `scripts/sieve_env_precheck.py`: gates disk free (>=150 GiB, under `models/_staging`) and `/proc/meminfo` MemAvailable (>=70 GiB), zero hard dependency on psutil/subprocess (pure `/proc/*/status` VmRSS parsing for the top-5-consumer report). Records `statsmodels.stats.weightstats.ttost_ind` availability (confirmed absent on this machine) without failing the run. Exits 0 on current hardware (1557.5 GiB free, 111.2 GiB available).
- 4 Wave-0 test scaffolds create concrete red-to-green targets for waves 2-4: `test_sieve_protected_retention.py` (SIEVE-01), `test_tost_gate.py` (SIEVE-05), `test_sieve_ksweep_mask.py` (SIEVE-04), `test_sieve_cross_seed_overlap.py` (Open Question 2). Each SKIPs cleanly via `pytest.importorskip` while its target module (`scripts.sieve_protected_retention`, `scripts.tost_gate`, `scripts.sieve_expert_mask_inference`, `scripts.sieve_cross_seed_overlap`) is absent.
- The protected-retention mask-count assertion (`protected_expert_mask.npy` sum == 1480, shape [48,128] bool) runs and passes TODAY since the Phase-7 artifact and `extract_protected_mask.py` already exist — confirmed via per-test (not module-level) importorskip so only the two future retention-check tests skip, not the whole file.

## Task Commits

Each task was committed atomically:

1. **Task 1: Environment pre-check script (disk, memory, statsmodels)** - `02bd0de` (feat)
2. **Task 2: Wave-0 test scaffolds for SIEVE-01/04/05 + cross-seed overlap** - `4f13598` (test)

**Plan metadata:** (this commit, see below)

## Files Created/Modified
- `scripts/sieve_env_precheck.py` - Gate A (disk >=150GiB), Gate B (mem >=70GiB, top-5 RSS on failure), Gate C (statsmodels ttost_ind, informational), JSON summary + assert-based self-check
- `tests/test_sieve_protected_retention.py` - SIEVE-01: mask-artifact assertion (runs today) + retention_check contract tests (per-test importorskip on scripts.sieve_protected_retention)
- `tests/test_tost_gate.py` - SIEVE-05: TOST equivalence gate contract (importorskip scripts.tost_gate)
- `tests/test_sieve_ksweep_mask.py` - SIEVE-04: k=13 union-with-protected mask contract (importorskip scripts.sieve_expert_mask_inference)
- `tests/test_sieve_cross_seed_overlap.py` - Open Question 2: per-layer Jaccard across judge seeds (importorskip scripts.sieve_cross_seed_overlap)

## Decisions Made
- statsmodels is confirmed NOT installed in this environment (`ModuleNotFoundError`); Gate C records `statsmodels_ttost_available: false` and does not block — plan 11-05 will implement hand-rolled TOST per the 11-CONTEXT.md/11-RESEARCH Don't-Hand-Roll guidance for this case.
- Future module/function names locked in by these test scaffolds: `scripts.sieve_protected_retention.retention_check(retained: dict[int, set[int]], mask: np.ndarray) -> bool`, `scripts.tost_gate.tost_equivalence(a, b, epsilon) -> bool`, `scripts.sieve_expert_mask_inference.build_ksweep_mask(routing_counts, protected_mask, k) -> np.ndarray[bool]`, `scripts.sieve_cross_seed_overlap.jaccard(a, b) -> float` and `pairwise_layer_jaccard(seed_topk: dict[str, list[set]]) -> dict[tuple[str,str], list[float]]`. Later-wave implementers must match these signatures for the tests to flip green without rewrites.

## Deviations from Plan

### Auto-fixed Issues

None required beyond the below infra workaround — no bugs, missing functionality, or blocking issues surfaced in the plan's own file scope.

---

**Total deviations:** 0 auto-fixed
**Impact on plan:** Plan executed as written. One pre-existing environment issue was worked around for verification (documented below), not fixed, since it is out of this plan's scope.

## Issues Encountered

**Pre-existing, out-of-scope environment issue (not caused by this plan):** the system `python3`'s pytest install is a broken editable dev checkout (`pytest 6.0.0rc2.dev33+g7f7a36478.d20260625`, installed via a `.pth` file pointing at `~/.claude/jobs/fa2ded79/tmp/.../src`) that is incompatible with Python 3.13's `ast` module (`TypeError: required field "lineno" missing from alias` inside `_pytest/assertion/rewrite.py`). This breaks `python3 -m pytest` for ANY test file, confirmed by reproducing the identical failure on a pre-existing, untouched test (`tests/test_bootstrap_ci.py`) before running any of this plan's new files. This is global conda-environment contamination unrelated to the sieve scaffolding and out of this task's scope (fixing it would require a `pip install`/reinstall of the global pytest package, which Rule 3's package-manager-install exclusion blocks from auto-fix).

Verification was instead performed with `.venv-tinker/bin/python -m pytest` (an isolated venv with a clean pytest 9.1.1), which:
- Confirms all 4 new files: collect correctly, `TestProtectedMaskArtifact::test_mask_shape_dtype_and_count` PASSES today, the other 5 tests SKIP (2 per-test skips in `test_sieve_protected_retention.py`, 3 whole-module skips for the other 3 files) — 0 errors.
- Full suite (`tests/ -q --ignore=tests/test_preflight.py`; `test_preflight.py` excluded because this venv lacks `python-dotenv`, an unrelated pre-existing gap in the tinker venv, not the system env): 666 passed, 8 skipped, 7 pre-existing failures — all 7 are `tinker.TinkerError` / judge-dispatch timeout tests unrelated to this plan's files (require `TINKER_API_KEY` / live judge endpoints not available in Wave-0 CPU/local-only scope). No new collection errors were introduced by the 4 sieve test files or the env-precheck script.

**Recommendation for the human/orchestrator:** the system `python3` pytest environment should be repaired (reinstall pytest into the conda env) before Wave-0 GPU work in later plans, since it currently blocks running the FULL project test suite (`tests/ -q`, 362+ tests) via the canonical `python3 -m pytest` path described in 11-VALIDATION.md. This is flagged here for visibility, not fixed, per scope boundaries.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Env pre-check is ready to gate the 2x57GB s0/s2 merges in plan 11-02 (disk/mem confirmed sufficient: 1557.5 GiB free, 111.2 GiB available).
- 4 concrete Wave-0 test contracts are in place for waves 2-4 implementers (SIEVE-01 retention check, SIEVE-04 k-sweep mask, SIEVE-05 TOST gate, cross-seed overlap for Open Question 2).
- Blocker to flag upstream: system `python3`'s pytest install is broken (see Issues Encountered) — recommend repairing before relying on `pytest tests/ -q` for later-wave verification gates.

---
*Phase: 11-compression-packaging*
*Completed: 2026-07-08*

## Self-Check: PASSED
All 5 created files found on disk; both task commits (02bd0de, 4f13598) found in git log.
