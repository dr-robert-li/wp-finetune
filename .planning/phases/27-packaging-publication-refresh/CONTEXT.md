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

Do NOT adapt the v3 card (`output/packaging/hf_cards/judge_gguf_README.md`) — it is a *negative example*, written in exactly the pipeline-narrative style this decision forbids. Write the v4 card fresh.

## LOCKED DECISION 3 — publish v4 to a NEW HF repo (2026-07-17)

Target: **`iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`** (new repo).

v4 is a different base model (Qwen3.6-35B-A3B, 224/256 experts after prune) than v3 (Qwen3-30B-A3B), so it gets its own lineage, license block, and eval table. The v3 repo `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` **stays up untouched** as the older artifact — this phase must not clobber or rewrite it. Rejected: versioning v4 inside the v3 repo (would file a Qwen3.6 model under a repo name that says `qwen3-30b`).

## LOCKED DECISION 4 — card names the generation base explicitly (2026-07-17)

The card states the boundary (it judges, it does not generate) and points operators at **`Qwen/Qwen3.6-35B-A3B`** by name as the base to use for generation. Accepted tradeoff: this goes stale when Qwen ships a newer base; concrete day-one usefulness wins over v3's vaguer "use a current base model" phrasing.

## SCOPE CORRECTION — judge-only, not a pair (from 27-RESEARCH.md, 2026-07-17)

The ROADMAP Phase 27 section says "Q8 GGUF **pair** conversion" and cites a "134 GiB bf16 pair". **That wording is stale**, inherited from the v3.0 template. The gen role was retired as a deliverable on 2026-07-15 (`PROJECT.md:14`, `README.md`, `JOURNAL.md` — every fine-tuned gen candidate regressed below the raw Qwen3.6 base). Phase 27 packages and publishes **one model**: the pruned v4 judge at `models/Qwen3.6-35B-A3B-judge-v4-pruned-k224` (60 GB bf16 on disk, 224/256 experts). Phase 27 should correct the stale ROADMAP/REQUIREMENTS wording rather than silently plan around it.

**~33.6 GiB Q8 is a PROJECTION, not a measurement** — `output/prune-v4/selection_v4.json` says so explicitly. Measuring it is this phase's job; no plan may treat it as a known value.

## LOCKED DECISION 5 — ship tier is Q6_K, human-confirmed (2026-07-17)

Ship: **Q6_K, 23.47 GiB** (`output/pkg-v4/wp-judge-v4-pruned-k224.Q6_K.gguf`). Human-confirmed after the executor flagged the call `human_judgment: true` in 27-03. This resolves that flag — Plans 27-04/27-05 consume `pkg4_quantization_ladder.json`'s `ship_tier`/`ship_gguf` as the single source.

Measured ladder (all four rungs, single-seed s1, n=121 wp_judge subset, max_tokens 2048):

| tier | rho | Δ vs f16 floor | n | parse_fail | size |
|---|---|---|---|---|---|
| f16 | 0.8002 | — (the floor) | 121 | 0 | 57.10 GiB |
| Q8_0 | 0.7851 | −1.51pp | 121 | 0 | 30.37 GiB |
| **Q6_K** | **0.8063** | **+0.61pp** | 121 | **0** | **23.47 GiB** ← ships |
| Q5_K_M | 0.8060 | +0.58pp | 120 | 1 | 20.36 GiB |

**Rationale — read this before citing any rho above.** All four rungs are **statistically indistinguishable**. Q6 scored *above* its own f16 source, which is impossible as a true effect (a lossy compression cannot exceed its source), proving the ~1–2pp spread is single-seed sampling noise, not signal. Selecting on "highest rho" would be selecting noise. The only discriminator that separates tiers is **parse_fail**: Q5_K_M alone produced a failure (1/121 — a degenerate repetition loop that never emitted a closing structured verdict). For a judge whose contract is a parseable rubric, that is a functional defect, not a quality wobble — though 1/121 is weak evidence in isolation. Q6_K is the smallest zero-parse-failure tier.

One-line rationale: **smallest artifact at statistically-equal measured quality with zero parse failures** — explicitly NOT "highest rho won".

This **overrides** 27-03-PLAN's literal mechanical stop rule (lowest rho-passing tier), which would have picked Q5_K_M.

**Consequence for LOCKED DECISION 1's tradeoff:** the "accepted tradeoff: larger artifact than v3's 30.2 GiB" is **void**. Q6_K at 23.47 GiB is ~22% SMALLER than v3's shipped artifact, at tied quality, on a newer base. The card must not repeat the "larger artifact" framing.

**Superseded finding:** 27-02-SUMMARY.md's headline ("Q8 is NOT lossless / the prune increased quantization sensitivity") was **falsified** by the Q6 rung — see `ladder_q8.json`'s `revised_interpretation` and `pkg4_quantization_ladder.json`'s `noise_floor_finding`. Raw measurements are immutable; only the interpretation was corrected. Do not repeat the falsified claim anywhere, including the card.
