# Phase 18: Production Sweep & HuggingFace Publication - Context

**Gathered:** 2026-07-11
**Status:** Ready for planning
**Source:** User goal directive (session-scoped /goal, 2026-07-11) — express path, no discuss-phase

<domain>
## Phase Boundary

Two halves. (1) Production sweep: repo docs current and mutually consistent, stale artifacts to
deprecated/, streamlined layout an outside user can follow. (2) Publication: package the two-model pair
(v1.2 gen + v1.3 judge, Q8 GGUF ship tier) and publish to HuggingFace with the full-lineage model card,
then validate the published artifacts round-trip. No new training, no new evals (Phase 17 numbers are
final inputs).

</domain>

<decisions>
## Implementation Decisions

### Repo sweep (PUB-01)
- README/PROJECT.md/PIPELINE.md/STATE agree with each other and with shipped artifacts; benchmarks section of README reflects Phase 17 numbers (LOCKED)
- Stale/one-off artifacts move to deprecated/ with README notes — but LESSON FROM PHASE 17: the Phase 16 sweep mis-archived string-path-referenced active deps (scripts/_wpbench_pth, _wpbench_shim). Any move must grep BOTH import statements AND string literals referencing the path before archiving (LOCKED)
- Root stays clean: no stray logs/artifacts at repo root (LOCKED)

### Packaging (PUB-02)
- Ship set: v1.2 gen model bf16 safetensors (models/qwen3-30b-wp-30_70-reasoning-merged-v4, 57 GB, 13 shards) + v1.3 judge Q8_0 GGUF all 3 ensemble seeds (models/_gguf/wp-v1.3-judge-s{0,1,2}.Q8_0.gguf, 32.5 GB each) with single-seed s1 documented as the leaner fallback (LOCKED — matches ship decision + PKG-03)
- MODEL_CARD.md (full lineage + quantization ladder + Phase 17 Benchmarks) is the source for the HF card(s); tokenizer with <wp_gen>/<wp_judge> task tokens ships with the gen model (LOCKED)
- bf16 judge GGUFs and base-model GGUF do NOT ship (size, no deployment need) (LOCKED)

### HF publication (PUB-03)
- Account: the authenticated `iamchum` HF account (verified working this session) (LOCKED)
- Visibility: PUBLIC — the goal says "publishing"; the card is written for outside users (LOCKED)
- Upload via huggingface_hub (hf upload / upload_large_folder for the 57 GB safetensors dir); single files are all under HF's 50 GB LFS cap (LOCKED)
- Post-upload validation: files listed via API, GGUF header readable/loadable, and a smoke gen + judge prompt round-trip from the DOWNLOADED artifact (not the local copy) (LOCKED — PUB-03)
- Commit+push repo docs as dr-robert-li, no AI co-author trailer (LOCKED)

### Claude's Discretion
- Repo layout: two model repos (one per model, HF-loader-friendly) with cross-linked cards vs one combined repo — pick what serves HF tooling best and document; suggested names wp-qwen3-30b-a3b-wp-gen-v1.2 / wp-qwen3-30b-a3b-wp-judge-v1.3-gguf or similar
- Whether to also publish the judge s1 bf16 safetensors export for vLLM users (models/tinker_export/v1.3*) — only if it exists complete and validates; otherwise GGUF-only judge is fine
- Card frontmatter (license inherits Apache-2.0 from Qwen3 base, tags, pipeline_tag)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `output/packaging/MODEL_CARD.md` — full-lineage card incl. Phase 17 Benchmarks section
- `output/packaging/gate2_quantization_decision.md`, `output/packaging/pkg03_ens8192_results.json` — quantization evidence cited by the card
- `PIPELINE.md` — pipeline doc that README links; must stay consistent
- `.planning/phases/16-pipeline-lockdown/` — prior sweep conventions + deprecated/README pattern
- `.planning/phases/17-benchmark-expansion-wp-bench-swe-bench-generation-eval/17-03-SUMMARY.md` — final benchmark numbers

</canonical_refs>

<specifics>
## Specific Ideas

- Upload volume ~155 GB total (57 + 3×32.5) — check disk/network headroom, use resumable upload paths, run as background process with progress logging
- Serving examples in cards: vLLM for gen (task-token prompt examples), llama.cpp/Ollama for judge Q8 GGUF (3-seed median ensemble recipe + s1 single-seed fallback)
- Phase 15 E2E validation receipts (pkg05_e2e_validation.json) can seed the smoke-prompt set

</specifics>

<deferred>
## Deferred Ideas

- Ollama registry publication, Open-WebUI demo — out of scope
- AWQ/further quant tiers — ladder closed at Q8
- Next-base work — Phase 19

</deferred>

---

*Phase: 18-production-sweep-huggingface-publication*
*Context gathered: 2026-07-11 via user goal directive express path*
