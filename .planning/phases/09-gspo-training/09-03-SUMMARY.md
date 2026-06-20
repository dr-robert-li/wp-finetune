---
phase: 09-gspo-training
plan: 03
subsystem: rl-reward
tags: [asyncio, claude-agent, subprocess, content-hash-cache, imputation, tdd]

# Dependency graph
requires:
  - phase: 09-02
    provides: "Wave-0 test scaffolding (test_rl_train.py stubs, conftest.py with mock_tinker_client)"
  - phase: 08-reward-infrastructure
    provides: "reward_pipeline.py (consumed unmodified); judge_imputed_from_group pattern"
  - phase: scripts/claude_agent.py
    provides: "generate_json subprocess path (claude --print --no-session-persistence --tools '')"
provides:
  - "scripts/rl_judge_dispatch.py: async Claude-consistency scorer (score_judge_consistency, score_with_cache, score_judge_consistency_batch)"
  - "tests/test_rl_judge_dispatch.py: unit tests (14 tests, all green)"
  - "JUDGE_SYSTEM rubric prompt for hackability reduction (D-09-05 T-09-RWD-HACK)"
  - "Content-hash cache (SHA-256 over php_code[:512]+critique_text[:512])"
  - "120s per-sample async timeout with group-mean imputation"
affects: [09-04, 09-05, 09-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Async-over-blocking: asyncio.wait_for + loop.run_in_executor wraps blocking subprocess calls"
    - "Content-hash cache: module-level dict keyed by SHA-256 of truncated inputs"
    - "Group-mean imputation: failed batch slots receive mean of valid scores (mirrors reward_pipeline)"
    - "N-vote median: call scorer N times, take median (default 1; callers increase for noise reduction)"
    - "Clamp to [0,1] before returning score (prevents out-of-range values in advantage computation)"

key-files:
  created:
    - scripts/rl_judge_dispatch.py
    - tests/test_rl_judge_dispatch.py
  modified: []

key-decisions:
  - "Async-over-blocking pattern: score_judge_consistency is kept sync (subprocess); asyncio.wait_for + run_in_executor lifts it to async — avoids rewriting claude_agent.py"
  - "None not cached: parse failures do not populate _score_cache so subsequent calls retry (avoids silent persistent failures)"
  - "Score clamped to [0,1] before return (Rule 2 guard): prevents out-of-range LLM values from propagating to advantage computation downstream"
  - "Test concurrency safety: failure simulation keyed on php_code content not call-count to avoid asyncio scheduling non-determinism"
  - "autouse clear_dispatch_cache fixture in test file (not conftest.py — conftest owned by 09-02)"

patterns-established:
  - "Test: use importorskip for module-level guard, autouse fixture to clear module-level cache"
  - "Test: async batch tests call asyncio.run() from sync test (no pytest-asyncio dependency)"
  - "Test: key failure simulation on input content not call order when testing concurrent async code"

requirements-completed: [GRPO-05]

# Metrics
duration: 18min
completed: 2026-06-20
---

# Phase 9 Plan 03: RL Judge Dispatch Summary

**Async Claude-consistency scorer with SHA-256 content-hash cache, 120s asyncio.wait_for timeout, and group-mean imputation — all routing via scripts.claude_agent subprocess (never the Anthropic API)**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-20T08:00:00Z
- **Completed:** 2026-06-20T08:07:36Z
- **Tasks:** 2 (both TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- Implemented `scripts/rl_judge_dispatch.py` with three exported functions: `score_judge_consistency`, `score_with_cache`, `score_judge_consistency_batch`
- Single-sample scorer calls `generate_json` via `scripts.claude_agent` subprocess with a structured JUDGE_SYSTEM rubric prompt (reduces hackability per D-09-05 T-09-RWD-HACK mitigation)
- Content-hash cache (SHA-256 over first 512 chars of each input) prevents redundant subprocess calls for repeated (code, critique) pairs; None results not cached so failures retry
- Async batch dispatcher uses `asyncio.gather` + `asyncio.wait_for(timeout=120s)` with `loop.run_in_executor` to lift the blocking subprocess into async context
- Group-mean imputation: failed/timed-out slots receive mean of valid batch scores (mirrors `reward_pipeline.judge_imputed_from_group`); all-failed fallback to neutral 0.5
- Score clamped to `[0,1]` before return (Rule 2 auto-fix: prevents out-of-range LLM output from corrupting advantage computation)
- 14 unit tests all green: cache-hit-skips-subprocess, timeout-imputes-from-group-mean, all-cached-zero-scorer-calls, order-preserving, all-failed-neutral-fallback, acceptance criteria guards

## Task Commits

Each task was committed atomically:

1. **RED tests** - `aa0d564` (test: add failing tests for rl_judge_dispatch)
2. **Task 1+2 GREEN implementation + test fixes** - `b6ec1d9` (feat: implement rl_judge_dispatch)

**Plan metadata:** (final commit — see below)

_TDD: RED commit then GREEN commit per task_

## Files Created/Modified

- `scripts/rl_judge_dispatch.py` — Async cached Claude-consistency scorer (173 lines)
- `tests/test_rl_judge_dispatch.py` — Unit tests (14 tests, all green)

## Decisions Made

- **Async-over-blocking pattern:** `score_judge_consistency` kept synchronous (subprocess); `asyncio.wait_for` + `run_in_executor` lifts it to async. This avoids modifying `scripts/claude_agent.py` and keeps the sync path clean for direct calls.
- **None not cached:** When `generate_json` returns None (parse failure), `_score_cache` is NOT populated — next call retries the subprocess. This avoids persistent silent failures.
- **Score clamped to [0,1]:** Even though the rubric asks for 0.0–1.0, LLM output is not trusted. Clamp applied before return so downstream advantage computation never receives out-of-range values (Rule 2 auto-fix).
- **N-vote median exposed:** `n_votes` param defaults to 1; 09-04/05 callers will raise this for noise suppression (D-09-05 guard 2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Score clamped to [0,1] before returning**
- **Found during:** Task 1 (single-sample scorer)
- **Issue:** Plan behavior spec says "float in [0,1]" but the PATTERNS example does not clamp; an out-of-range LLM value would flow directly into advantage computation
- **Fix:** Added `score = max(0.0, min(1.0, score))` after float cast, before appending to votes list
- **Files modified:** `scripts/rl_judge_dispatch.py`
- **Verification:** Unit test `test_score_judge_consistency_returns_float_in_range` confirms [0,1]
- **Committed in:** `b6ec1d9`

**2. [Rule 1 - Bug] Fixed AST traversal bug in acceptance criteria test**
- **Found during:** Task 1 GREEN verification
- **Issue:** `test_no_anthropic_api_import` used `n.name` on `ast.Import` nodes but `ast.Import` has `.names` (list of aliases), not `.name`; this caused `AttributeError` in the subprocess check
- **Fix:** Changed `[n.name for n in ast.walk(tree) if isinstance(n, ast.Import)]` to `[a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]`
- **Files modified:** `tests/test_rl_judge_dispatch.py`
- **Verification:** Test passes after fix
- **Committed in:** `b6ec1d9`

**3. [Rule 1 - Bug] Fixed asyncio scheduling race in batch imputation tests**
- **Found during:** Task 2 GREEN verification (`test_timeout_imputes_from_group_mean`)
- **Issue:** Tests used call-count to determine which sample fails (3rd call = timeout), but asyncio.gather runs tasks concurrently so call order is non-deterministic — the 3rd call could be any sample, making the "second score must be 0.6" assertion flaky
- **Fix:** Changed failure logic to key on `php_code` content (e.g., `score_map["<?php echo 3;"] = "TIMEOUT"`) instead of incrementing call count
- **Files modified:** `tests/test_rl_judge_dispatch.py`
- **Verification:** `test_timeout_imputes_from_group_mean` passes consistently
- **Committed in:** `b6ec1d9`

---

**Total deviations:** 3 auto-fixed (1 Rule 2 correctness guard, 2 Rule 1 test bugs)
**Impact on plan:** All auto-fixes necessary for correctness and test reliability. No scope creep.

## Issues Encountered

- `n.name` vs `n.names` on `ast.Import` nodes: caught during GREEN verification, fixed inline
- asyncio concurrency ordering: first batch imputation test failed flakily due to call-count keying; fixed by content-keying

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. All external surface is via existing `scripts/claude_agent.py` subprocess path. Threat mitigations as specified:
- T-09-INJECT: php_code and critique inserted in fenced blocks inside `_build_consistency_prompt`; `--tools ""` prevents tool access; output parsed as single numeric field
- T-09-RWD-HACK: JUDGE_SYSTEM rubric (not free-form) reduces hackability; score is the consistency signal only (not the reward)
- T-09-SELFPREF: N-vote parameter exposed for 09-04/05 to raise; temp-0 is subprocess default

## Next Phase Readiness

- `score_judge_consistency_batch(samples)` is ready for 09-04 (reward combination step) to call
- Cache is session-scoped (module-level dict); 09-04/05 can clear or pre-warm it
- `n_votes` defaults to 1; raise it in 09-04/05 for production noise suppression
- No blockers for 09-04

---
*Phase: 09-gspo-training*
*Completed: 2026-06-20*

## Self-Check: PASSED

- `scripts/rl_judge_dispatch.py` EXISTS: confirmed
- `tests/test_rl_judge_dispatch.py` EXISTS: confirmed
- RED commit `aa0d564` EXISTS: confirmed
- GREEN commit `b6ec1d9` EXISTS: confirmed
- All 14 tests PASS: confirmed (`pytest tests/test_rl_judge_dispatch.py -q` → 14 passed)
- Acceptance criteria: 0 anthropic imports, 0 run_in_background, 3 claude_agent refs, 2 asyncio.gather, 2 wait_for
