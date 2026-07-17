---
phase: 27-packaging-publication-refresh
plan: 04
subsystem: packaging
tags: [huggingface, model-card, gguf, docs, publication]

requires:
  - phase: 27-packaging-publication-refresh
    provides: "Plan 27-03's frozen ship_tier=Q6_K / ship_gguf (pkg4_quantization_ladder.json), human-confirmed via CONTEXT.md LOCKED DECISION 5"
provides:
  - "Fresh, operator-only v4 HF model card (five LOCKED-DECISION-2 sections, no methodology narrative, ship-tier/size/rho read from the ladder JSON, every rho labelled with its stack + seed config)"
  - "Canonical model flipped v3 -> v4 across README.md, PROJECT.md, .planning/PROJECT.md, and output/packaging/MODEL_CARD.md, with the v3 repo kept live/untouched and framed only as the superseded prior artifact"
  - "output/pkg-v4/pub4_upload_manifest.json + scripts/_pub4_upload.sh prepared (not run) for Plan 27-05's gated publish"
affects: [27-05-publish]

tech-stack:
  added: []
  patterns:
    - "Operator-surface card discipline: five fixed sections (what/for, acquisition, use, performance, links out), fresh-written, no phase/gate/methodology narrative -- methodology lives in the repo's MODEL_CARD.md/PIPELINE.md/JOURNAL.md, the HF card only links to it"
    - "Every published rho is stamped with its stack AND seed configuration; a 3-seed ensemble number and a single-seed number are never juxtaposed without the label on the same line"
    - "Historical footer preservation: when a stale dated snapshot in .planning/PROJECT.md is superseded, add a new dated block above it rather than editing the old one out of existence"

key-files:
  created:
    - output/pkg-v4/hf_cards/judge_v4_README.md
    - output/pkg-v4/pub4_upload_manifest.json
    - scripts/_pub4_upload.sh
  modified:
    - README.md
    - PROJECT.md
    - .planning/PROJECT.md
    - output/packaging/MODEL_CARD.md

key-decisions:
  - "Card frontmatter reuses the v3 card's YAML field SET (structure only) with v4 values; card body is a fresh write, not an adaptation of judge_gguf_README.md's body (that file is a documented negative example)"
  - "README.md's stale 'v1.3 stays canonical' / 'v4 judge stays on the bench' narrative (written before Phase 26's prune resolved) was corrected in place rather than left to contradict the now-canonical v4 flip -- Rule 1 (stale/incorrect content), not scope creep: the acceptance criteria required every line naming the v1.3 repo to carry 'superseded'/'prior' framing, which the property table and Quickstart section could not satisfy without being rewritten to v4"
  - "HF Hub's per-file size limit was fetched live (huggingface.co/docs/hub/storage-limits, 2026-07-17) rather than assumed: <200GB recommended split threshold, 500GB hard cap. The 23.47 GiB ship GGUF needs no llama-gguf-split; recorded in the manifest's per_file_limit_checked block with its source URL and the exact quoted text"
  - "output/packaging/MODEL_CARD.md's v3 section (title, compression lineage, quant table, all v3 numbers) was left completely untouched -- only a superseded-pointer banner was added at the top and a new v4 lineage section appended, per the task's 'extend, do not gut' instruction (git diff --numstat: +57/-17)"

requirements-completed: [PUB4-01]

coverage:
  - id: D1
    description: "Fresh operator-only v4 HF model card exists with the five LOCKED-DECISION-2 sections, v4 frontmatter naming Qwen/Qwen3.6-35B-A3B, the ladder-chosen Q6_K ship file + real measured size, a llama.cpp quickstart with real serving flags and request/response shape, an eval table where every row names its stack/seed config, and zero excluded methodology acronyms or other-stack numbers"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "27-04-PLAN.md Task 1 <verify> python assertion block (frontmatter fields, ship-tier filename present, --jinja present, methodology-acronym negative grep, 0.8134/0.8533/33.6 negative grep, unlabelled-ensemble check, card shorter than README) -- all pass"
        status: pass
    human_judgment: false
  - id: D2
    description: "PROJECT.md, README.md and output/packaging/MODEL_CARD.md name the v4 judge and iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf as canonical with measured numbers and stack/seed provenance; the v3 repo appears only as the superseded prior artifact; the dangling #the-v40-finding-qwen36 anchor into the v3 card body is repointed; 'retired as a deliverable' and '0 parseable' load-bearing records survived; MODEL_CARD.md was extended not gutted"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "27-04-PLAN.md Task 2 <verify> python assertion block (v4 repo canonical in both docs, 0.8134/0.8533 absent, Qwen3.6-35B-A3B present, dangling anchor absent, load-bearing sentences intact, v3-repo-line superseded/prior framing, ship_tier named, MODEL_CARD.md numstat additions>deletions) -- all pass"
        status: pass
    human_judgment: false
  - id: D3
    description: "output/pkg-v4/pub4_upload_manifest.json targets exactly the new v4 repo with disk-verified sizes and the ladder's ship_gguf (single file, HF Hub per-file limit checked live and recorded), plus the card as README.md; scripts/_pub4_upload.sh preserves _pub03_upload.sh's stall-watchdog + sequential-retry logic byte-for-byte (diff confirmed) with only manifest path/log path/repo target changed, and does not execute upload-large-folder. Nothing pushed."
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "27-04-PLAN.md Task 3 <verify> python assertion block (single repo, correct repo_id, disk-size match, README.md entry present, v3 repo absent, watchdog markers present, manifest path correct, upload-large-folder not executable) -- all pass; bash -n syntax check pass; watchdog-diff check pass; token-leak grep pass"
        status: pass
    human_judgment: false

duration: ~30min
completed: 2026-07-17
status: complete
---

# Phase 27 Plan 04: Model Card + Canonical Flip + Upload Manifest Summary

**Wrote a fresh operator-only v4 HF model card (Q6_K/23.47 GiB, rho 0.8063 single-seed, no MTP head disclosed), flipped the canonical model v3->v4 across README.md/PROJECT.md/MODEL_CARD.md with the v3 repo kept live as the superseded prior artifact, and prepared (not run) the v4 upload manifest + `_pub4_upload.sh` for Plan 27-05.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-07-17T10:30:00Z (approx, immediately after human confirmation of LOCKED DECISION 5)
- **Completed:** 2026-07-17T10:59:55Z
- **Tasks:** 3
- **Files modified:** 7 (3 created, 4 modified)

## Accomplishments

- **Fresh v4 HF model card** (`output/pkg-v4/hf_cards/judge_v4_README.md`, 96 lines, tighter than README.md's 157). Five sections only: what it is/for (names `Qwen/Qwen3.6-35B-A3B` explicitly as the generation base per LOCKED DECISION 4), acquisition (ship-tier filename + size read from `pkg4_quantization_ladder.json`, not hardcoded), use (real llama.cpp serving flags, real request/response shape pulled from `data/reasoning_dataset/openai_val.jsonl`), performance (every rho stamped with its stack + seed config; the v3 3-seed ensemble number is present but explicitly labelled and not treated as a like-for-like delta), links out. Zero methodology acronyms (AIMER/REAP/MoE-Sieve/GSPO/Tinker/k-sweep), zero bf16-vLLM numbers, zero `33.6` projection.
- **No-MTP disclosure.** The card states plainly, in operator terms, that the shipped GGUF has no MTP/speculative-decoding head and why (the prune left the MTP layer at 256 experts, GGUF's `expert_count` is a global field, `--no-mtp` conversion was required).
- **Canonical model flipped v3 -> v4** across `README.md` (badges, "the model is" paragraph, `## The model` table, Quickstart, Benchmarks table, the `## The v4.0 finding` section's stale "v1.3 stays canonical" conclusion), `PROJECT.md` (canonical-deliverable paragraph, keeping "retired as a deliverable" and "0 parseable" verbatim), `.planning/PROJECT.md` (new Key Decisions row + a fresh dated footer block, with the prior 2026-07-15 snapshot preserved below it rather than deleted), and `output/packaging/MODEL_CARD.md` (extended with a full v4 prune-to-ship lineage section: routing profile, AIMER gate-before-remove result, the GGUF ladder table, ship rationale — v3's section is marked superseded-but-published at the top and otherwise untouched, +57/-17 lines).
- **LOCKED DECISION 5's correction propagated everywhere it needed to.** The "larger artifact than v3" framing from LOCKED DECISION 1 is gone from every doc touched; all now correctly state Q6_K (23.47 GiB) is ~22% *smaller* than v3's 30.2 GiB.
- **Upload manifest + script prepared, not run.** `output/pkg-v4/pub4_upload_manifest.json` targets exactly one repo (the new v4 one) with disk-verified sizes for the ship GGUF and the card (as `README.md`). Fetched HuggingFace Hub's live storage-limits doc to confirm the per-file threshold (<200GB recommended, 500GB hard cap) rather than assuming a number — the 23.47 GiB file needs no `llama-gguf-split`. `scripts/_pub4_upload.sh` is a byte-for-byte copy of `_pub03_upload.sh`'s stall-watchdog + sequential-retry logic (confirmed via `diff`), with only the manifest path, log path, and repo/log strings changed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Write the v4 operator-only HF model card from scratch** - `9076b4d` (feat)
2. **Task 2: Flip the canonical model v3 -> v4 across PROJECT.md, README.md and MODEL_CARD.md** - `8c2fd1e` (docs)
3. **Task 3: Upload manifest + `_pub4_upload.sh` (prepared, not run)** - `00b2b8c` (feat)

_No TDD split — this plan's tasks are documentation authoring and a data/script artifact, not application logic under test-first development._

## Files Created/Modified

- `output/pkg-v4/hf_cards/judge_v4_README.md` - fresh v4 operator-only HF model card
- `README.md` - badges, model paragraph, property table, Quickstart, Benchmarks, `## The v4.0 finding` conclusion, License, all flipped to v4-canonical
- `PROJECT.md` - canonical-deliverable paragraph flipped to v4, load-bearing sentences preserved
- `.planning/PROJECT.md` - Key Decisions row for the v3->v4 flip; footer flipped to v4 with the 2026-07-15 snapshot preserved as history below it
- `output/packaging/MODEL_CARD.md` - superseded-pointer banner added at top; new v4 prune-to-ship lineage section appended; v3's own lineage section left untouched
- `output/pkg-v4/pub4_upload_manifest.json` - single-repo upload manifest (v4 GGUF + card), HF per-file-limit check recorded
- `scripts/_pub4_upload.sh` - copy of `_pub03_upload.sh` adapted for the v4 manifest/repo/log paths only

## Decisions Made

- Card frontmatter YAML field set copied structurally from the v3 card (license/base_model/pipeline_tag/library_name/language/tags); body is a fresh write per CONTEXT.md LOCKED DECISION 2 — `judge_gguf_README.md`'s body was never opened past its frontmatter.
- README.md's `## The model` property table and Quickstart section were rewritten to v4 (not just the top-of-file badges/paragraph the plan's `<action>` called out explicitly) because the acceptance criteria requires every line naming the v1.3 repo to carry "superseded"/"prior" framing — leaving those sections pointing at v1.3 as the primary download target would both fail that check and mislead an operator into serving the wrong model.
- README.md's `## The v4.0 finding` section conclusion ("v1.3 stays canonical... v4 stays on the bench... in progress") was corrected in place to reflect that the prune resolved and canonical flipped, rather than left stale next to the now-contradictory badges above it.
- HF Hub's per-file size limit was fetched live from `huggingface.co/docs/hub/storage-limits` (2026-07-17) instead of assumed, per the plan's explicit "confirm at run time" instruction; recorded with its source URL and quoted text in the manifest's `per_file_limit_checked` block.
- `output/packaging/MODEL_CARD.md`'s v3 lineage content was left byte-identical except for a new banner and heading-title change on the "why v4 now ships" section — the file was extended, not rewritten, per the plan's "this is where the compression narrative belongs" instruction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug/stale content] README.md's `## The model` table, Quickstart, and `## The v4.0 finding` conclusion still pointed at v1.3 as the primary/canonical artifact**
- **Found during:** Task 2, while verifying the acceptance criterion "every line naming the v1.3 repo must contain 'superseded' or 'prior'"
- **Issue:** The plan's `<action>` for Task 2 explicitly called out the badges and the "model is" paragraph + callout, but the property table (`## The model`) and the `## Quickstart` code blocks still named `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` as the repo to download and serve, with no superseded/prior framing on those lines — this would fail the acceptance grep and, more importantly, would tell an operator reading the README to download the wrong (superseded) model.
- **Fix:** Rewrote the property table to v4 values (repo, base, ship tier, rho, serving) with a "Prior release ... superseded" row for v1.3; rewrote the Quickstart's `hf download`/`llama-server` commands to the v4 repo and filename; rewrote the Benchmarks table to the v4 ladder numbers with stack/seed labels; corrected the `## The v4.0 finding` section's stale "v1.3 stays canonical, v4 stays on the bench, in progress" conclusion to state the prune resolved and canonical flipped, with the historical mid-flight numbers (0.8067 vs 0.8056 at that point) kept but clearly marked as "at that point."
- **Files modified:** `README.md`
- **Verification:** Full Task 2 `<verify>` python assertion block passes; every line matching `wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` in README.md now contains "superseded" or "prior".
- **Committed in:** `8c2fd1e` (Task 2 commit)

**2. [Rule 1 - Bug/stale content] `.planning/PROJECT.md`'s dated footer still declared v1.3 canonical and Phases 24-27 "not run"**
- **Found during:** Task 2, reviewing `.planning/PROJECT.md` for canonical-model references per the task's read-list
- **Issue:** The file's closing `*Last updated: 2026-07-15 ... Canonical deliverable = the v1.3 WP Judge ... v3 stays canonical, v4.0 recorded as a diagnostic milestone. Phases 24-27 not run*` line directly contradicts LOCKED DECISION 1/5 and the actual state of the project (Phases 22/25/26/27 have since run and flipped canonical to v4).
- **Fix:** Added a new dated block above it (2026-07-17) stating the v4 flip with measured numbers and a pointer to the receipts, and preserved the old 2026-07-15 snapshot below it verbatim as a labelled historical record rather than deleting it — consistent with the project's established pattern of not erasing prior interpretations (see `pkg4_quantization_ladder.json`'s `noise_floor_finding` treatment in Plan 27-03).
- **Files modified:** `.planning/PROJECT.md`
- **Verification:** Manual read-through; the new block and the preserved old block are both present and internally consistent with their respective dates.
- **Committed in:** `8c2fd1e` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1, stale/incorrect content discovered while satisfying the plan's own acceptance criteria)
**Impact on plan:** Both fixes were necessary for the acceptance criteria to pass and for the docs to be internally consistent and honest post-flip; neither introduces new scope beyond "make the canonical-flip claim true everywhere it's made." No architectural changes, no new artifacts beyond what the plan specified.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. The HF Hub storage-limits check used a plain `curl` to a public docs page; no credentials involved.

## Next Phase Readiness

- **Plan 27-05 (publish, human-gated) can now run `scripts/_pub4_upload.sh` against `output/pkg-v4/pub4_upload_manifest.json`** — both are prepared and verified but nothing has been pushed to HuggingFace.
- The v4 card (`output/pkg-v4/hf_cards/judge_v4_README.md`) is ready to ship as the new repo's `README.md` exactly as manifested.
- `27-VALIDATION.md`'s operator-only tone bar is explicitly manual verification, reserved for Plan 27-05's blocking human gate — Task 1's automated checks cover structure/content exclusions but not tone.
- No blockers.

---
*Phase: 27-packaging-publication-refresh*
*Completed: 2026-07-17*

## Self-Check: PASSED

- FOUND: `output/pkg-v4/hf_cards/judge_v4_README.md`
- FOUND: `output/pkg-v4/pub4_upload_manifest.json`
- FOUND: `scripts/_pub4_upload.sh`
- FOUND: `README.md`
- FOUND: `PROJECT.md`
- FOUND: `.planning/PROJECT.md`
- FOUND: `output/packaging/MODEL_CARD.md`
- FOUND commit `9076b4d` (Task 1)
- FOUND commit `8c2fd1e` (Task 2)
- FOUND commit `00b2b8c` (Task 3)
