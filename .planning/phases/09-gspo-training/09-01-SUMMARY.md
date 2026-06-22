---
phase: 09-gspo-training
plan: 01
subsystem: rl-data
tags: [rl, prompts, data-assembly, phase9, grpo, tinker]
dependency_graph:
  requires:
    - data/reasoning_dataset/openai_train.jsonl   # Phase 4.2 audited corpus
    - data/reasoning_dataset/openai_val.jsonl      # held-out (leakage guard only)
  provides:
    - data/rl_prompts/wp_gen_train.jsonl           # 68 wp_gen rollout prompts
    - data/rl_prompts/wp_judge_train.jsonl         # 482 wp_judge rollout prompts
    - data/rl_prompts/PROVENANCE.md                # source lineage + leakage assertion
    - scripts/build_rl_prompts.py                  # deterministic, re-runnable assembler
    - scripts/tinker_rl_data.py                    # Tinker prompt-only data adapter
  affects:
    - scripts/rl_train.py (09-04)                  # rollout sampler unblocked
tech_stack:
  added: []
  patterns:
    - sha256 dedup + val-leakage guard (content-hash cross-check)
    - prompt-only OpenAI chat schema (empty assistant turn)
    - lazy tinker import for test-environment compatibility
key_files:
  created:
    - scripts/build_rl_prompts.py
    - scripts/tinker_rl_data.py
    - data/rl_prompts/wp_gen_train.jsonl
    - data/rl_prompts/wp_judge_train.jsonl
    - data/rl_prompts/PROVENANCE.md
  modified: []
decisions:
  - "Gen pool source: openai_train.jsonl contains 73 raw <wp_gen> user turns (68 after dedup); no seed augmentation needed — pool is non-empty and auditable"
  - "Manual JSONL load is primary path in tinker_rl_data.py (not cookbook) — allows import without tinker and matches test lazy-import convention"
  - "Global dedup: sha256 seen-set shared across both gen and judge pools to prevent cross-pool duplicates"
  - "Neither-tagged rows (8 replay rows): excluded by tag-split rule, counted in PROVENANCE"
metrics:
  duration: 2
  completed_date: "2026-06-20"
  tasks_completed: 2
  files_created: 5
  files_modified: 0
---

# Phase 09 Plan 01: RL Prompt Pool Assembly Summary

## One-Liner

Deterministic assembly of 68 wp_gen + 482 wp_judge RL rollout prompt pools from the Phase-4.2 audited corpus, with sha256 dedup, val-leakage guard, PROVENANCE.md lineage record, and a tinker-free Tinker data adapter.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Audit prompt sources and build assembly script | 223aa1b | build_rl_prompts.py, wp_gen_train.jsonl, wp_judge_train.jsonl, PROVENANCE.md |
| 2 | Tinker prompt-only data adapter | 63d893f | tinker_rl_data.py |

## What Was Built

### Task 1: build_rl_prompts.py + data/rl_prompts/

Reads `data/reasoning_dataset/openai_train.jsonl` (Phase-4.2 audited corpus, 563 rows), splits by leading tag:

- `<wp_gen>` user turns → `wp_gen_train.jsonl` (68 prompts after 5 dedup drops)
- `<wp_judge>` user turns → `wp_judge_train.jsonl` (482 prompts, no dedup drops)
- 8 untagged replay rows → excluded

Output schema: OpenAI chat format with empty assistant turn (`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": ""}]}`). Completions generated at RL sampling time.

PROVENANCE.md records: source file, row counts, dedup count, split rule, val-leakage count (0), output file sha256s. Satisfies T-09-POISON (audited-only source) and T-09-LEAK (cross-check against openai_val.jsonl) threat mitigations.

Script is fully idempotent — re-running produces byte-identical output (no timestamps, deterministic iteration order).

### Task 2: scripts/tinker_rl_data.py

Mirrors `scripts/tinker_reasoning_data.py` with:
- `BASE_MODEL = "Qwen/Qwen3-30B-A3B"`, `RENDERER_NAME = "qwen3_disable_thinking"`, `MAX_LENGTH = 8192`
- `load_rl_prompts(pool)` returns user-turn-only prompt dicts; strips the empty assistant turn so the model generates completions at sampling time
- Primary path: manual JSONL load (no tinker dependency)
- Secondary path (commented): `FromConversationFileBuilder` with `TrainOnWhat.NONE` if cookbook exposes prompt-only mode
- Tinker import is lazy (inside `_load_via_cookbook`) — module imports without tinker in CI/test environments

## Deviations from Plan

None — plan executed exactly as written.

Note: The plan's `read_first` listed `ugc_seeds.json` as "candidate gen-task seeds". These were examined but not needed: `openai_train.jsonl` already contained 73 `<wp_gen>` user turns (68 after dedup), satisfying the non-empty pool requirement from audited sources alone. No seed augmentation was performed; using seeds would have introduced a second source requiring separate provenance tracking.

## Verification Results

All acceptance criteria met:

- Both JSONL files non-empty; every line parses as JSON with `messages[0].role=="user"` and `messages[1].content==""`.
- All gen prompts start with `<wp_gen>`, all judge prompts start with `<wp_judge>` (plan check commands exit 0).
- Val-set leakage check: 0 prompts leaked (sha256 cross-check against 141 val rows).
- Idempotency: byte-identical output on re-run (sha256 of all 3 output files unchanged).
- `python -c "import scripts.tinker_rl_data"` succeeds without tinker installed.
- `load_rl_prompts("gen")` returns 68 dicts, `load_rl_prompts("judge")` returns 482 dicts, no assistant target in any returned prompt.

## Known Stubs

None.

## Threat Flags

None — all new files consume existing audited data; no new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check: PASSED

- `data/rl_prompts/wp_gen_train.jsonl`: FOUND
- `data/rl_prompts/wp_judge_train.jsonl`: FOUND
- `data/rl_prompts/PROVENANCE.md`: FOUND
- `scripts/build_rl_prompts.py`: FOUND
- `scripts/tinker_rl_data.py`: FOUND
- Task 1 commit 223aa1b: FOUND
- Task 2 commit 63d893f: FOUND
