# Phase 17: Benchmark Expansion — wp-bench + SWE-bench Generation Eval - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning
**Source:** User goal directive (session-scoped /goal, 2026-07-11) — express path, no discuss-phase

<domain>
## Phase Boundary

Give the shipped two-model pair current, honest benchmark numbers. Two evals: (1) a full wp-bench run
on the v1.2 generation model via the shipping stack, (2) a SWE-bench generation-mode eval positioning
the model against a public coding benchmark. Results go into MODEL_CARD.md with out-of-domain caveats.
No training, no model changes, no packaging (Phase 18 owns publication).

</domain>

<decisions>
## Implementation Decisions

### Benchmarks
- Full wp-bench run (unlimited, not WPBENCH_LIMIT subset) on v1.2 gen model, shipping stack, compared against the 0.4484 Gate-1 receipt (LOCKED — user goal)
- SWE-bench generation eval as full as feasible: "full SWE-bench gen eval" per user; scope constrained only by what the DGX Spark (GB10, aarch64, 121 GB unified) toolchain can honestly EVALUATE, and the chosen scope + constraints must be pre-registered before results are read (LOCKED — BENCH-02)
- Generation-mode (non-agentic patch generation) is the SWE-bench protocol — this is a completion model with task tokens, not an agent scaffold (LOCKED)

### Execution
- All heavy inference runs on local serving (vLLM or llama.cpp per PIPELINE.md patterns); no paid API for generation (LOCKED — project rule)
- Serving lessons from Phase 15 apply: real-generation warm-up gate before capture (not /health), context window sized as parallel × per-slot need (LOCKED)
- Results recorded as JSON artifacts under output/ with config + seed, same convention as prior eval receipts (LOCKED)

### Documentation
- MODEL_CARD.md gains a Benchmarks section with both results + explicit out-of-domain caveat for SWE-bench (LOCKED — BENCH-03)
- JOURNAL.md entry (semi-formal voice), STATE/CHANGELOG updates, commit+push as dr-robert-li, no AI co-author trailer (LOCKED — user goal)

### Claude's Discretion
- SWE-bench variant choice (full test split vs Verified vs Lite) — decide from measured feasibility: aarch64 Docker evaluation support, disk, and wall-clock; document the choice and why. If local aarch64 evaluation of generated patches is not feasible, an alternative honest path (e.g. sb-cli cloud evaluation, or patch-generation + apply/lint-only validation with the limitation stated) may be selected, provided the limitation is pre-registered.
- Retrieval context style for generation-mode SWE-bench (oracle vs BM25) — pick the standard that keeps results comparable to published numbers; document.
- Prompt/template adaptation for the fine-tuned task-token model vs base Qwen template.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline + serving
- `PIPELINE.md` — locked v3.0 pipeline stages, serving entrypoints, gate conventions
- `scripts/run_packaging_recipe.md` — llama.cpp/GGUF serving recipe from Phase 15
- `output/packaging/gate1_bf16_baseline.json` — wp-bench 0.4484 Gate-1 receipt (comparison target)
- `output/packaging/MODEL_CARD.md` — card to extend with Benchmarks section

### Eval harness
- `wp-bench/` — wp-bench runner (see wp-bench/README.md, AGENTS.md)
- `eval/eval_gen.py`, `scripts/run_eval_reasoning.py` — OpenAI-compatible eval transport + orchestration patterns
- `.planning/phases/15-packaging/` — Phase 15 summaries (serving lessons)

</canonical_refs>

<specifics>
## Specific Ideas

- Model under test: `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (v1.2 gen model, canonical)
- wp-bench prior receipts: 0.4484 (Gate 1, bf16, shipping stack); 0.4616 codegen bar from v1.2 promotion
- DGX Spark: GB10, aarch64, 121 GB unified memory — SWE-bench Docker eval images are predominantly x86_64; feasibility must be measured, not assumed
- SWE-bench is Python-repo patch generation; this model is WordPress/PHP-specialized — the number will be low and that is fine; honesty over vanity

</specifics>

<deferred>
## Deferred Ideas

- Agentic SWE-bench (SWE-agent / mini-SWE-agent scaffold) — out of scope; generation-mode only this phase
- Publishing benchmark numbers to HF — Phase 18
- Any re-training or prompt-tuning to chase SWE-bench score — never in scope for this milestone

</deferred>

---

*Phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval*
*Context gathered: 2026-07-11 via user goal directive express path*
