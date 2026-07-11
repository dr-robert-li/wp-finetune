---
phase: 19-next-base-rerun-roadmap
plan: 01
subsystem: planning-roadmap
tags: [next-base, qwen3.6, moe, roadmap, v4.0, pipeline-rerun]
dependency-graph:
  requires: [PIPELINE.md, output/relabel/gap_closure_summary.json, output/packaging/MODEL_CARD.md]
  provides: [.planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md, .planning/V4-RERUN-ROADMAP.md]
  affects: [.planning/ROADMAP.md, .planning/REQUIREMENTS.md, .planning/STATE.md, CHANGELOG.md, JOURNAL.md]
tech-stack:
  added: []
  patterns: [live-verification-before-lock, cost-anchored-to-measured-actuals, pre-registered-success-criteria]
key-files:
  created:
    - .planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md
    - .planning/V4-RERUN-ROADMAP.md
    - .planning/phases/19-next-base-rerun-roadmap/19-01-SUMMARY.md
  modified:
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
    - .planning/STATE.md
    - CHANGELOG.md
    - JOURNAL.md
decisions:
  - "Base LOCKED: Qwen/Qwen3.6-35B-A3B (35B/3B, 256 experts 8-routed+1-shared, hybrid Gated-DeltaNet/Gated-Attention, Apache-2.0) — all five rationale axes live-verified via direct HF/Tinker fetches, not inherited from prior research"
  - "Fallback pre-authorized: Qwen/Qwen3.5-35B-A3B (same family, live-verified); dense Qwen3.6-27B documented as a methodology-change alternative, not equated"
  - "New finding: gen+judge pair no longer fits GB10 concurrently at bf16 (65.2 GiB/checkpoint vs old base's 56.8 GiB; 130.4 GiB pair exceeds 121 GB host) — Stage 5 quantization becomes a memory-driven hard prerequisite, not an optional size lever"
  - "Relabel-campaign reuse recommended (v1.3's 603-item human-relabeled set is base-agnostic) with an explicit re-open condition, not silently deferred"
  - "Execution is a FUTURE v4.0 milestone gated on explicit human sign-off; this phase produced the plan only"
metrics:
  duration: "~55 min"
  completed: "2026-07-11"
status: complete
---

# Phase 19 Plan 01: Next-Base Rerun Roadmap Summary

Live-verified and LOCKED `Qwen/Qwen3.6-35B-A3B` as the v4.0 rerun base, then wrote a costed, evidence-linked
roadmap (`.planning/V4-RERUN-ROADMAP.md`) mapping every PIPELINE.md stage and conditional gate to it.

## What was built

**Task 1 — `19-NEXT-BASE-SELECTION.md`.** Re-verified the front-runner live (not from the prior-session
research doc alone) via direct `curl` fetches: HF model card raw READMEs for Qwen3.6-35B-A3B, Qwen3.5-35B-A3B
(fallback), and Qwen3-30B-A3B (current base, for comparison); the HF models-search API for ecosystem
coverage (Unsloth, bartowski llama.cpp GGUF, NVIDIA NVFP4, QuantTrio AWQ); and Tinker's live models table
for support/pricing/context caps. All five LOCKED rationale axes covered with source-cited evidence:
architecture match (256 experts, 8-routed+1-shared, hybrid DeltaNet/Attention, confirmed verbatim against
the research doc's claim), GB10 121 GB memory budget, Tinker/Unsloth/vLLM (>=0.19.0)/llama.cpp support,
Apache-2.0 license, and coding-benchmark deltas (SWE-bench Verified 73.4, LiveCodeBench v6 80.4,
Terminal-Bench 2.0 51.5 — all matched the planner's pre-verification numbers exactly). Base LOCKED.

**Task 2 — `.planning/V4-RERUN-ROADMAP.md`.** Maps all 8 PIPELINE.md stages/gates (data pipeline, gen SFT,
judge SFT/relabel, final eval, Gate A/RL, Gate B/MoE-Sieve, Gate C/prune, packaging) each with expected
delta vs Qwen3-30B-A3B, the carried-forward known result, the re-test gate, and a cost estimate anchored to
named v3.0/v3.1 actuals (Tinker ~$1.83/run, GB10 profiling 6h30m, wp-bench ~19min, ens8192 ~2h/6 arms, GGUF
~1h/model). The three no-winner gates (RL rejected, Sieve optimal_k=full, prune no_winner) are carried
forward as conditional re-test stages per PIPELINE.md's own guidance. Both architecture-delta work items
(Sieve/protected-mask tooling adaptation for mixed DeltaNet/Attention layers + shared expert; eos/pad
token-ID alignment, QwenLM/Qwen3.6 discussion #96) are scheduled before their dependent stages. All six
carry-forward lessons documented. Pre-registered success criteria: judge rho >0.85 single-seed or >0.87
ensemble, framed against the 0.8075 shipping figure and ~0.157 ceiling gap. Both Claude's-Discretion items
resolved: a proposed 10-phase v4.0 structure (Phases 20-29, mirroring PIPELINE order with dependencies) and
a relabel-reuse recommendation with an explicit re-open condition.

**Task 3 — Closeout.** ROADMAP.md gets the V4-RERUN-ROADMAP pointer and Phase 19 marked Complete (progress
table + detail block). REQUIREMENTS.md NEXT-01/NEXT-02 flipped to complete in both the v3.1 checklist and
the traceability table. STATE.md position advanced to Phase 19 complete (frontmatter + Current Position),
explicitly noting Phase 18 status was left untouched. CHANGELOG.md `[Unreleased]` gained a summary entry.
JOURNAL.md gained a first-person, semi-formal entry (no em dashes) describing the base lock, the memory
finding, and the roadmap. Committed as Dr. Robert Li (no AI co-author trailer), pushed to
`phase10-execution`.

## Key finding beyond the plan's literal ask

Live verification surfaced a real, previously-undocumented constraint: at bf16, the Qwen3.6-35B-A3B
gen+judge pair is 130.4 GiB, exceeding the GB10 host's 121 GB — unlike the current base's pair (113.6 GiB,
fits with headroom). This converts Stage 5 quantization from an optional size lever into a hard
concurrent-serving prerequisite for v4.0. Captured in both the selection doc (Axis 2) and the roadmap
(Stage 5), not left for a future session to discover mid-run.

## Deviations from Plan

None — plan executed as written. The bf16 memory-budget finding above was produced by the plan's own
required arithmetic (Axis 2 verification), not an out-of-scope addition.

## Verification

- `SELECTION_OK`, `ROADMAP_OK`, `CLOSEOUT_OK` — all three per-task automated grep gates passed.
- No weights downloaded, no training run, no code added — diff is 9 files, all `.md`.
- Commit `abc8ea7` authored `Dr. Robert Li <dr.robert.li.au@gmail.com>`, no AI trailer, pushed
  (`5decf41..abc8ea7 phase10-execution -> phase10-execution`).
- 18-CONTEXT.md and `deps/dgx-toolbox` (unrelated in-progress Phase 18 artifacts) were left untouched,
  per the execution constraint not to disturb Phase 18's paused work.

## Self-Check: PASSED

- FOUND: `.planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md`
- FOUND: `.planning/V4-RERUN-ROADMAP.md`
- FOUND commit `abc8ea7` in `git log --oneline --all`
