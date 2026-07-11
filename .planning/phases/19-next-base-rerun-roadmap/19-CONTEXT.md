# Phase 19: Next-Base Rerun Roadmap - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning
**Source:** User goal directive (session-scoped /goal, 2026-07-11) — express path, no discuss-phase

<domain>
## Phase Boundary

A planning-only phase. Deliverable: a costed, evidence-linked roadmap document for rerunning the full
locked pipeline (PIPELINE.md) on the latest capable Qwen-family base, carrying every v3.0/v3.1 lesson
forward. No downloads, no training, no code — the roadmap gates a future milestone (v4.0) that starts
only on human approval.

</domain>

<decisions>
## Implementation Decisions

### Base selection (NEXT-01)
- Candidates already researched this session (see 19-RESEARCH-BASESCAN.md in this directory): Qwen3.6-35B-A3B (front-runner), Qwen3.5-35B-A3B (safer twin), Qwen3.6-27B dense (wildcard, methodology change) (INPUT — verify key claims before locking)
- Selection rationale must cover: architecture match to the pipeline (MoE routing, tokenizer/task-token extension), GB10 memory budget (121 GB), Tinker/Unsloth/vLLM/llama.cpp support, license, coding-benchmark deltas (LOCKED — NEXT-01)
- Known architecture deltas MUST be addressed as roadmap work items: hybrid Gated-DeltaNet layers + 256 experts + shared expert break the current Sieve profiler/protected-mask tooling assumptions; eos/pad token-ID mismatch needs an alignment step before SFT (LOCKED — carry into NEXT-02)

### Roadmap doc (NEXT-02)
- Maps EVERY PIPELINE.md stage to the new base with expected deltas; conditional no-winner gates (RL, Sieve, prune) carried forward as re-test stages, not dropped (LOCKED)
- Rough compute/cost estimates per stage (GB10 wall-clock from v3.0 actuals as the baseline; Tinker spend from v1.3 actuals ~$2/run-class) (LOCKED)
- Carry-forward lessons: truncation-aware evals (8192-token caps), real-generation warm-up gating, --parallel context splitting, CI-aware gates, pre-registration discipline for benchmarks, double-grep archive rule (LOCKED)
- Roadmap lands as a doc in .planning/ (e.g. .planning/V4-RERUN-ROADMAP.md) + ROADMAP.md gets a pointer; execution is a FUTURE milestone requiring human sign-off (LOCKED)
- JOURNAL entry, STATE/CHANGELOG updates, commit+push as dr-robert-li (LOCKED)

### Claude's Discretion
- v4.0 milestone phase structure (how many phases, what order) — informed by PIPELINE.md stage list
- Whether judge-ceiling expectations justify re-running the relabel campaign vs reusing v1.3 labels

</decisions>

<canonical_refs>
## Canonical References

- `PIPELINE.md` — the locked pipeline being re-targeted
- `.planning/phases/19-next-base-rerun-roadmap/19-RESEARCH-BASESCAN.md` — Qwen family scan (2026-07-11)
- `output/relabel/gap_closure_summary.json` — judge-ceiling evidence (stronger base = the lever)
- `output/packaging/MODEL_CARD.md` — v3.0/v3.1 final numbers (comparison baseline)
- `.planning/ROADMAP.md` — milestone history + v3.1 section

</canonical_refs>

<specifics>
## Specific Ideas

- v3.0 actuals for estimates: router profiling 6h30m GB10; SFT grid ~Tinker $2/run; wp-bench full run ~19 min; ens8192 judge eval ~2h/6 arms; GGUF convert ~1h/model
- The 0.157 judge-rho gap to teacher ceiling was the declared motivation for a stronger base — the roadmap should pre-register what "success" looks like (e.g. judge rho > 0.85 single-seed, or ensemble > 0.87)

</specifics>

<deferred>
## Deferred Ideas

- Actually downloading/running the new base — future v4.0 milestone
- Non-Qwen bases — out of scope per user directive

</deferred>

---

*Phase: 19-next-base-rerun-roadmap*
*Context gathered: 2026-07-11 via user goal directive express path*
