---
phase: 19-next-base-rerun-roadmap
verified: 2026-07-11T01:10:00Z
status: gaps_found
score: 5/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "The next base is selected and LOCKED with documented rationale, and its load-bearing claims (existence, license, architecture, benchmarks) are verified against live sources — NEXT-01"
    status: partial
    reason: >
      Independent live re-fetch of the exact source cited by the selection doc
      (huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md) confirms all quoted
      figures are accurate (35B/3B, 256 experts 8-routed+1-shared, hybrid DeltaNet/Attention,
      Apache-2.0, SWE-bench Verified 73.4, LiveCodeBench v6 80.4, Terminal-Bench 2.0 51.5 —
      all exact matches). BUT the same README states, two lines above the quoted param count,
      `pipeline_tag: image-text-to-text` and `Type: Causal Language Model with Vision Encoder`,
      and carries a full "Vision Language" benchmark section (MMMU, MMMU-Pro, Mathvista, etc.).
      The current base (`Qwen/Qwen3-30B-A3B`, independently re-checked) is `pipeline_tag:
      text-generation` / `Type: Causal Language Models` — text-only. Axis 1 (architecture
      match) in 19-NEXT-BASE-SELECTION.md never mentions the vision encoder or the modality
      change, despite quoting figures from the same paragraph. Axis 2's bf16 memory arithmetic
      (65.2 GiB/checkpoint = 35B params x 2 bytes) uses the "Language Model" param count only;
      the README's own structure implies the vision tower is a separate, additional component,
      so the true per-checkpoint bf16 footprint (and the 130.4 GiB pair figure used to justify
      Stage 5 quantization as a "hard prerequisite") is likely understated, not fabricated.
      This is exactly the class of blind spot the phase's own threat model (T-19-01) says Task 1
      must close before the roadmap authorizes a costly future milestone.
    artifacts:
      - path: ".planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md"
        issue: "Axis 1 omits the vision encoder / VL modality change vs the current text-only base; Axis 2 memory arithmetic does not account for the vision tower's parameter contribution to bf16 size"
    missing:
      - "Add an explicit sub-finding to Axis 1: Qwen3.6-35B-A3B is a vision-language checkpoint (image-text-to-text, Causal LM + Vision Encoder), unlike the current text-only Qwen3-30B-A3B base. State whether the pipeline's task-token SFT (text-only data) is expected to leave the vision tower untouched/frozen, and confirm Tinker/Unsloth/vLLM LoRA fine-tuning and merge paths handle a VL checkpoint the same way as a text-only one."
      - "Re-derive or bound the vision-tower parameter count and re-state the bf16 per-checkpoint / pair-size figures in Axis 2 (and V4-RERUN-ROADMAP.md Stage 5) either confirming 65.2/130.4 GiB already includes it or revising upward with the delta noted."
deferred: []
---

# Phase 19: Next-Base Rerun Roadmap Verification Report

**Phase Goal:** Costed, evidence-linked roadmap for rerunning locked PIPELINE.md on the latest Qwen base, all v3.0 lessons carried forward.
**Verified:** 2026-07-11
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Next base selected and LOCKED, load-bearing claims verified live (NEXT-01) | ⚠️ PARTIAL | All quoted figures independently re-verified accurate via direct `curl` of the same HF README (200 OK; 35B/3B, 256 experts 8-routed+1-shared, hybrid DeltaNet/Attention, Apache-2.0, SWE-bench 73.4, LiveCodeBench v6 80.4, Terminal-Bench 51.5 — exact matches). Gap: the model is confirmed a vision-language checkpoint (`pipeline_tag: image-text-to-text`, "Vision Encoder"), unlike the current text-only base; this is not addressed in Axis 1 or Axis 2's memory arithmetic. See gaps. |
| 2 | Roadmap maps every PIPELINE.md stage with deltas, re-test gates, cost estimates (NEXT-02) | ✓ VERIFIED | PIPELINE.md's 5 stages + 3 conditional gates (8 total) all present in `V4-RERUN-ROADMAP.md` with (a)/(b)/(c)/(d) structure; confirmed via `grep -nE "^#+ (Stage|Conditional Gate)" PIPELINE.md` (5 stage headers + 3 gate headers) cross-referenced against roadmap section headers 1:1. |
| 3 | Three no-winner gates (RL, Sieve, prune) carried forward as conditional re-test stages, not dropped | ✓ VERIFIED | All three gates present with explicit "carried-forward known result" (REJECTED / no headroom / no winner) and a distinct re-test gate condition each. |
| 4 | Two architecture-delta work items (Sieve/DeltaNet tooling, eos/pad alignment) captured as explicit roadmap work items | ✓ VERIFIED | Both items have a named section, scheduled before their dependent stage (Gate B / Stage 2-3 resp.), and appear in the proposed v4.0 phase table (Phase 25, Phase 20). |
| 5 | Carry-forward lessons + pre-registered success criteria documented | ✓ VERIFIED | All six lessons present with source citations (Jaccard CI, truncation, warm-up, `--parallel`, pre-registration, double-grep). Success criteria: rho >0.85 single-seed / >0.87 ensemble, independently confirmed against `output/relabel/gap_closure_summary.json` (ceiling 0.9844, gap 0.157) and `output/packaging/MODEL_CARD.md` (0.8075/0.8017) — figures match exactly, not fabricated. |
| 6 | ROADMAP.md points to the roadmap doc; execution flagged FUTURE v4.0 milestone requiring human sign-off | ✓ VERIFIED | ROADMAP.md line 833 pointer to V4-RERUN-ROADMAP.md; both docs and Closing gate state explicit human-sign-off requirement; commit diff (`abc8ea7`) is 9 files, all `.md`, no code/downloads. |

**Score:** 5/6 truths verified (1 partial — see gap)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `19-NEXT-BASE-SELECTION.md` | Locked base, 5-axis rationale, live-verified | ⚠️ PRESENT, ONE AXIS INCOMPLETE | Exists, substantive, all 5 axes present; Axis 1/2 miss the VL-modality fact (see gap) |
| `.planning/V4-RERUN-ROADMAP.md` | Full PIPELINE stage map | ✓ VERIFIED | Exists, 278 lines, all 8 stages/gates + work items + lessons + criteria + phase proposal |
| `.planning/ROADMAP.md` (Phase 19 pointer + completion) | Pointer + Complete | ✓ VERIFIED | Line 826/832-834/872 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| Selection doc base name | V4-RERUN-ROADMAP.md base references | Same locked model throughout | ✓ WIRED | `Qwen/Qwen3.6-35B-A3B` consistent across both docs, no drift |
| Roadmap cost figures | v3.0/v3.1 actuals (STATE/MODEL_CARD/output/*) | Citation-traceable | ✓ WIRED | Spot-checked 11 cited artifact paths — all exist; spot-checked rho figures (0.8075/0.8017/0.7554/0.157/0.9844) against source JSON/MD — all exact matches, no fabrication |
| V4-RERUN-ROADMAP.md stages | PIPELINE.md stages/gates | 1:1 coverage | ✓ WIRED | `grep -nE "^#+ (Stage|Conditional Gate)"` on PIPELINE.md yields exactly the 8 sections mapped in the roadmap |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|--------------|--------|----------|
| NEXT-01 | 19-01-PLAN.md | Base researched + selected, rationale documented | ⚠️ PARTIAL | REQUIREMENTS.md flipped to `[x]` Complete; codebase evidence shows the rationale doc is strong but incomplete on architecture modality (see gap) |
| NEXT-02 | 19-01-PLAN.md | Roadmap maps every stage, gates carried forward, cost estimates | ✓ SATISFIED | REQUIREMENTS.md `[x]` Complete; independently confirmed |

### Anti-Patterns Found

None (TBD/FIXME/XXX/TODO/placeholder scan on both new docs: clean). No debt markers.

### Closeout Consistency

- REQUIREMENTS.md NEXT-01/NEXT-02: flipped `[x]` in both v3.1 checklist and Traceability table.
- ROADMAP.md: Phase 19 marked Complete in progress table and detail block, pointer present.
- STATE.md: `stopped_at` updated, explicitly notes Phase 19 ran ahead of Phase 18 (express path, disclosed, not silent).
- CHANGELOG.md: `[Unreleased]` entry added, matches doc content.
- JOURNAL.md: first-person entry present, no em dashes in body prose (only the established title-separator convention, consistent with every other entry in the file).
- Commit `abc8ea7` authored "Dr. Robert Li <dr.robert.li.au@gmail.com>", no AI co-author trailer, 9 files all `.md`, pushed to `phase10-execution` (confirmed in `git log`).

### Human Verification Required

None triggered independently of the gap above — the gap is a documentable, fixable content gap (add a sub-finding + re-derive one arithmetic line), not a behavior that needs human/runtime observation.

### Gaps Summary

The phase substantially achieves its goal: NEXT-02 (the roadmap doc) is thorough, accurate, and its cost
figures independently trace to real artifacts with no fabrication found. NEXT-01 (base selection) is
mostly rigorous — genuinely live-verified against primary sources rather than trusting the prior research
doc, which is exactly what the plan asked for. However, an independent re-fetch of the exact HF README the
selection doc cites reveals a real, material fact the doc's Axis 1 (architecture match) and Axis 2 (memory
budget) both miss: `Qwen/Qwen3.6-35B-A3B` is a vision-language model (`image-text-to-text`, "Causal
Language Model with Vision Encoder"), while the current production base (`Qwen3-30B-A3B`, also re-checked)
is text-only. This is not a fabricated-number problem — every number quoted is accurate — it is an
incomplete-analysis problem on the exact axis ("architecture match") this phase's own threat model
(T-19-01) flagged as the highest-severity risk to mitigate before a document that gates real GPU/Tinker
spend gets human sign-off. Recommend: amend `19-NEXT-BASE-SELECTION.md` Axis 1 with the VL-modality finding
and its tooling implications (does text-only LoRA SFT via Tinker/Unsloth touch/require the vision tower?),
and re-derive Axis 2's bf16 figures to confirm whether 65.2/130.4 GiB already includes vision-tower weights
or needs revising upward. This is a small, scoped fix, not a redo of either document.

---

_Verified: 2026-07-11_
_Verifier: Claude (gsd-verifier)_

---

## Gap Fix Applied (2026-07-11, executor)

Both `missing` items addressed; ready for re-verification:

- **Investigation:** fetched `config.json` + `model.safetensors.index.json` for Qwen3.6-35B-A3B and the
  Qwen-org model list (`?author=Qwen&search=Qwen3.6`). Confirmed VL
  (`Qwen3_5MoeForConditionalGeneration`, 333 `model.visual.*` tensors, SigLIP-class tower depth 27 /
  hidden 1152). **No text-only sibling exists** — all four Qwen-org 3.6 repos are `image-text-to-text`,
  and the fallback Qwen3.5-35B-A3B is also "Causal Language Model with Vision Encoder". Selection NOT
  changed; ranking cannot flip on modality since every candidate shares it.
- **Axis 1 amended** with a modality sub-finding: VL confirmed with primary-source evidence; text-only
  serving is vendor-documented (vLLM `--language-model-only`); text-only MoE-LoRA SFT leaves the tower
  untouched; a VL merge-path bring-up check (`model.language_model.*` key prefix) added to
  V4-RERUN-ROADMAP Phase 20.
- **Axis 2 re-derived from measurement:** safetensors-index `total_size` = 71,903,645,408 B = **67.0
  GiB/checkpoint** (includes vision tower + MTP; the prior 65.2 GiB was the LM-only lower bound). Pair =
  134.0 GiB full / ~130.4 GiB LM-only — both exceed 121 GB, so the "quantization is a hard prerequisite"
  conclusion is unchanged (strengthened). Q8 projection revised to ~35.6 GiB/checkpoint, ~71.3 GiB pair.
  V4-RERUN-ROADMAP Stage 5 updated with the same figures.
