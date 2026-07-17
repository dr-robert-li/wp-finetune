# Phase 27 — Packaging & Publication Refresh (CONTEXT seed)

Seeded ahead of planning to capture two locked user directives (2026-07-17). `/gsd-plan-phase 27` should expand this into full plans; the decisions below are LOCKED inputs, not open questions.

## LOCKED DECISION 1 — canonical model flips v3 → v4

The canonical deliverable is now the **v4 judge** (Qwen3.6-35B-A3B base), not v3. Phase 27 publishes the v4 artifact chosen by Gate C (Phase 26):
- Gate C prune PASSES → publish the **pruned v4** (~33.6 GiB Q8).
- Gate C prune FAILS → publish the merged-**unpruned v4** (37.8 GiB Q8).
Either way the shipped model is v4. PROJECT.md / MODEL_CARD / README / HF card lineage all update v3→v4 here. Accepted tradeoff: larger artifact than v3's 30.2 GiB at statistically-tied quality (0.8067 vs 0.8056), in exchange for the newer base.

## LOCKED DECISION 2 — HF model card is OPERATOR-specific

The Hugging Face model card is for the **operator who pulls and runs the model**, NOT a history of the project. It must NOT recount the pipeline, training runs, phase history, or compression/methodology narrative. Those live in the **GitHub repo** (PIPELINE.md, JOURNAL.md, CHANGELOG.md, phase docs) — the card **links out** to the repo for anyone wanting the deep methodology.

**Card scope (what it DOES contain):**
1. **What it is / what it's for** — a WordPress-code review judge: hand it a PHP snippet, it returns a structured rubric verdict. State the intended use + the boundary (it judges, it doesn't generate; generation is out of scope — point at a current base model for that).
2. **Acquisition** — which HF repo + which GGUF file/quant to download; minimal "get it" steps.
3. **Use** — the operator quickstart: serve with llama.cpp, feed a snippet, read the rubric back. Request/response shape, the 3-seed ensemble note if relevant, any serving flags that matter.
4. **Performance / evals** — the judge Spearman-rho headline + the benchmark table (per-dimension retention, serving-config comparison, the base anchor row). The numbers an operator needs to trust it.
5. **Links out** — GitHub repo for training / compression / packaging methodology; license; provenance/lineage one-liner.

**NOT in the card** (→ redirect to GitHub): the v3/v4 study narrative, the MoE-Sieve/k-sweep/AIMER prune methodology, the OOM/served-profile engineering, Tinker/RL history, the pre-registration discipline. A one-line "how this was built → see repo" pointer is enough.

**Style reference:** the existing operator-first README rewrite (quickstart-led, ~150 lines) is the tone to match — the card is the even-tighter operator surface of that.
