---
phase: 21-sft-training-generation-judge-models
plan: 04
subsystem: evaluation
tags: [judge-format, vllm, gb10, qwen3.6, smoke, baseline, wave-2]

# Dependency graph
requires:
  - phase: 21-sft-training-generation-judge-models
    provides: "21-01 Wave-0 gate (renderer/format decisions); Phase 20 v4 serving harness (serve_base20_vllm.sh + boot_vllm serve_script/extra_env)"
provides:
  - "scripts/smoke_judge_format_base21.py -- raw-base judge-format-compliance smoke (serve + parse-fail-rate)"
  - "output/base21/judge01_format_smoke.json -- JUDGE-01 baseline receipt: raw Qwen3.6-35B-A3B parse_fail_rate 1.0 (30/30) vs 0.18 community anchor, recorded BEFORE any judge-training result is read"
affects: [21-06-eval-gate, JUDGE-03-interpretation]

tech-stack:
  added: []
  patterns:
    - "Raw-base diagnostic baselines are recorded before dependent training results are read -- a 100% parse-fail rate on the untrained base is a valid, expected anchor, not a failure"

key-files:
  created:
    - scripts/smoke_judge_format_base21.py
    - output/base21/judge01_format_smoke.json
  modified: []

key-decisions:
  - "parse_fail_rate 1.0 recorded as the honest untrained-base baseline -- NOT a gate on judge SFT (per plan: a high raw-base rate is EXPECTED and is exactly what the judge SFT trains away)"
  - "Root cause of 30/30 parse-fail: the raw base's always-on thinking mode emits 'Here's a thinking process:' prose and exhausts max_tokens=2048 before any rubric JSON is produced -- format noncompliance, not truncation artifact (max_tokens>=2048 recorded per carry-forward lesson 1; the model never reaches JSON at any length in this mode)"
  - "JUDGE-03 interpretation guidance: trained seeds must beat this baseline decisively; the v1.2-era old base showed the same class of raw failure (3/4 ratios unparseable)"

metrics:
  duration: ~55min
  completed: 2026-07-14

requirements-completed: [JUDGE-01]

status: complete
---

# Phase 21 Plan 04: Raw-Base Judge-Format-Compliance Smoke (JUDGE-01) Summary

**Raw (no-adapter) Qwen3.6-35B-A3B parse-fails 30/30 real wp_judge prompts (parse_fail_rate 1.0, above the 0.18 community anchor) — root cause is the base's always-on thinking mode emitting reasoning prose that exhausts 2048 tokens before any rubric JSON; recorded as the honest untrained baseline BEFORE any judge SFT result is read, exactly the diagnostic anchor JUDGE-01 exists to establish.**

## Performance

- **Duration:** ~55 min (single vLLM serve cycle: boot + 30 generations at max_tokens=2048)
- **Completed:** 2026-07-14 (2026-07-13T20:33Z)
- **Tasks:** 1
- **Files modified:** 2 (both created)

## Accomplishments

- `scripts/smoke_judge_format_base21.py`: mirrors `bench_wpbench_base_anchor.py` / `smoke_deltanet_base20.py` structurally — `boot_vllm` (Phase 20 `serve_base20_vllm.sh`, `LANGUAGE_MODEL_ONLY=1`, gpu_mem_util 0.80, 1200s boot timeout for the 67 GiB base) → `wait_healthy` → real-generation warm-up gate (Phase 15 LOCKED lesson) → 30 seed-1337-sampled `<wp_judge>` prompts from `openai_val.jsonl` with the `config/judge_system.md` rubric as system message → `parse_judge_scores(text, "auto")` per completion → receipt → `stop_vllm` in `finally`.
- Parse-fail definition matches `scripts/relabel/eval_relabel.py` exactly: fail = `not parsed or not parsed.get("dimension_scores")`.
- Truncation-safety honored (carry-forward lesson 1): `max_tokens=2048` recorded in the receipt; inspection of `sample_failures` confirms the failures are format noncompliance (thinking-mode prose from token 0), not rubric JSON cut mid-emission.
- Receipt `output/base21/judge01_format_smoke.json`: `n_prompts=30`, `n_parse_ok=0`, `n_parse_fail=30`, `parse_fail_rate=1.0`, `community_anchor_rate=0.18`, `vs_anchor="above"`, `max_tokens=2048`, `temperature=0.0`, `served_model_dir`, 5 `sample_failures` for inspection.
- Plan verify assertion passes: `n_prompts>=20`, counts sum, rate in [0,1], `vs_anchor` present. Script `ast.parse`s clean and greps for `parse_judge_scores`.
- Clean teardown verified: no vLLM containers, port 8020 free, GPU idle (5W).

## Baseline Interpretation (for JUDGE-03 / 21-06)

- The raw base is at **100% judge-format noncompliance** — far above the 18% community anchor. This is an UNTRAINED-base measurement and is expected: the base was never trained on `<wp_judge>` task tokens or the rubric output contract.
- **Root cause:** Qwen3.6's always-on thinking mode. Every completion opens with "Here's a thinking process:" analysis prose and consumes the full 2048-token budget before emitting any `<judge_output>`/JSON block. `strip_think()` cannot help — the prose is not wrapped in `<think>` tags at serve time.
- **Guidance:** trained judge seeds (21-03's 3-seed relabel-SFT) must beat this baseline decisively — post-SFT parse-fail near 0 is the signal the format training landed. The v1.2-era old base exhibited the same raw-failure class (3/4 ratios produced unparseable judge output pre-training). A high raw rate does NOT invalidate the judge training; this receipt is a diagnostic anchor, not a go/no-go.

## Task Commits

1. **Task 1: Raw-base judge-format-compliance smoke** — `4d48720` (feat) — `feat(21-04): JUDGE-01 raw-base judge-format smoke -- parse_fail_rate 1.0 vs 0.18 anchor`

## Files Created/Modified

- `scripts/smoke_judge_format_base21.py` — raw-base judge-format smoke (serve + warm-up gate + parse-fail measurement + finally teardown)
- `output/base21/judge01_format_smoke.json` — JUDGE-01 baseline receipt (force-added per the base20/base21 gate-receipt precedent)

## Decisions Made

See `key-decisions` in frontmatter.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None — real serve, real generations, real parser; no mocked calls or fabricated receipts.

## Issues Encountered

None. Single serve cycle, clean boot, clean teardown (no orphan containers, port 8020 free, GPU back to idle).

## User Setup Required

None.

## Next Phase Readiness

- **JUDGE-01: SATISFIED.** The raw-base parse-fail baseline (1.0 vs 0.18 anchor) is measured on 30 real generations, truncation-safe (max_tokens=2048 recorded), and committed BEFORE any judge-training result is read or promoted (Wave 4, 21-06).
- 21-06's eval gate now has the anchor to contextualize JUDGE-03's post-SFT parse rate and rho against.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-14*

## Self-Check: PASSED

Both created files verified present on disk; task commit hash (4d48720) verified present in git log; plan verify assertion exits 0.
