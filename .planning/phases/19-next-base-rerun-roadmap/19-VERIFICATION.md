---
phase: 19-next-base-rerun-roadmap
verified: 2026-07-11T01:30:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 5/6
  gaps_closed:
    - "NEXT-01: VL-modality omission in Axis 1 and understated bf16 footprint in Axis 2 — fixed in commit aecf7a0"
  gaps_remaining: []
  regressions: []
---

# Phase 19: Next-Base Rerun Roadmap Verification Report

**Phase Goal:** Costed, evidence-linked roadmap for rerunning locked PIPELINE.md on the latest Qwen base, all v3.0 lessons carried forward.
**Verified:** 2026-07-11
**Status:** passed
**Re-verification:** Yes — after gap closure (commit `aecf7a0`)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Next base selected and LOCKED, load-bearing claims verified live (NEXT-01) | ✓ VERIFIED | All quoted figures independently re-verified via direct fetches of the cited primary sources: HF README (35B/3B, 256 experts 8-routed+1-shared, hybrid DeltaNet/Attention, Apache-2.0, SWE-bench 73.4, LiveCodeBench v6 80.4, Terminal-Bench 51.5 — exact matches). Gap fix re-verified independently: `config.json` shows `architectures: ["Qwen3_5MoeForConditionalGeneration"]` + `vision_config`; `model.safetensors.index.json` `metadata.total_size` = 71,903,645,408 bytes = 67.0 GiB with 333 `model.visual.*` tensors and `model.language_model.*` key prefix — all exactly as the amended Axis 1/2 now state. |
| 2 | Roadmap maps every PIPELINE.md stage with deltas, re-test gates, cost estimates (NEXT-02) | ✓ VERIFIED | PIPELINE.md's 5 stages + 3 conditional gates (8 total) all present in `V4-RERUN-ROADMAP.md` with (a)/(b)/(c)/(d) structure, 1:1 against PIPELINE.md section headers. |
| 3 | Three no-winner gates (RL, Sieve, prune) carried forward as conditional re-test stages, not dropped | ✓ VERIFIED | All three gates present with explicit carried-forward known result (REJECTED / no headroom / no winner) and a distinct re-test gate condition each. |
| 4 | Two architecture-delta work items (Sieve/DeltaNet tooling, eos/pad alignment) captured as explicit roadmap work items | ✓ VERIFIED | Both items have a named section, scheduled before their dependent stage (Gate B / Stage 2-3), and appear in the proposed v4.0 phase table (Phase 25, Phase 20). Phase 20 additionally carries the new VL merge-path bring-up check from the gap fix. |
| 5 | Carry-forward lessons + pre-registered success criteria documented | ✓ VERIFIED | All six lessons present with source citations. Success criteria (rho >0.85 single-seed / >0.87 ensemble) confirmed against `output/relabel/gap_closure_summary.json` (ceiling 0.9844, gap 0.157) and `output/packaging/MODEL_CARD.md` (0.8075/0.8017) — exact matches, not fabricated. |
| 6 | ROADMAP.md points to the roadmap doc; execution flagged FUTURE v4.0 milestone requiring human sign-off | ✓ VERIFIED | ROADMAP.md pointer at line 833; both docs and the Closing gate state explicit human-sign-off; phase diff is all `.md`, no code/downloads. |

**Score:** 6/6 truths verified

### Gap Closure Verification (re-verification focus)

Previous gap: `19-NEXT-BASE-SELECTION.md` Axis 1 omitted the VL modality of the candidate; Axis 2's 65.2/130.4 GiB bf16 figures excluded the vision tower. Fixed in `aecf7a0`; each fix claim re-verified against live primary sources this session:

| Claim (amended doc) | Independent check | Result |
|---|---|---|
| `config.json`: `Qwen3_5MoeForConditionalGeneration` + `vision_config` (depth 27, hidden 1152) | Direct fetch of `Qwen/Qwen3.6-35B-A3B/resolve/main/config.json` | ✓ MATCH |
| 333 `model.visual.*` tensors in safetensors index | Direct fetch + count of `model.safetensors.index.json` weight_map | ✓ 333 exactly |
| Measured 67.0 GiB (`total_size` 71,903,645,408 bytes), pair 134.0 GiB full / ~130.4 GiB LM-only, both exceed 121 GB | Direct fetch: `metadata.total_size` = 71,903,645,408 = 67.0 GiB | ✓ MATCH; arithmetic shown in doc |
| LM keys prefixed `model.language_model.*` (motivates Phase 20 merge-path check) | Weight-map sample from same fetch | ✓ MATCH (`model.language_model.layers.0.*`) |
| Fallback `Qwen3.5-35B-A3B` is also VL (no text-only sibling in the generation) | Doc claim, consistent with vendor generation being VL-wide; ranking unchanged | ✓ Documented |
| `V4-RERUN-ROADMAP.md` Stage 5 restated (67.0/134.0/130.4; Q8 ~71.3 GiB pair fits) + Phase 20 VL check + `--language-model-only` mitigation | Read of amended Stage 5 and Phase 20 rows | ✓ Present |

Regression spot-check on previously-passed items: V4 stage coverage, gates, lessons, criteria, ROADMAP/REQUIREMENTS/STATE/CHANGELOG/JOURNAL closeout, commit authorship (`aecf7a0` authored "Dr. Robert Li", no AI trailer) — all intact.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `19-NEXT-BASE-SELECTION.md` | Locked base, 5-axis rationale, live-verified | ✓ VERIFIED | All 5 axes complete including VL modality sub-finding and measured memory arithmetic |
| `.planning/V4-RERUN-ROADMAP.md` | Full PIPELINE stage map | ✓ VERIFIED | All 8 stages/gates + work items + lessons + criteria + phase proposal; Stage 5/Phase 20 carry the VL fix |
| `.planning/ROADMAP.md` (Phase 19 pointer + completion) | Pointer + Complete | ✓ VERIFIED | Lines 826/832-834/872 |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| Selection doc base name | V4-RERUN-ROADMAP.md base references | Same locked model throughout | ✓ WIRED |
| Roadmap cost figures | v3.0/v3.1 actuals | Citation-traceable; 11 cited artifact paths exist; rho/cost figures exact-match sources | ✓ WIRED |
| V4-RERUN-ROADMAP.md stages | PIPELINE.md stages/gates | 1:1 coverage | ✓ WIRED |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| NEXT-01 | ✓ SATISFIED | Base locked with complete, live-verified five-axis rationale incl. VL modality |
| NEXT-02 | ✓ SATISFIED | Full stage map with deltas, gates, costs, work items, lessons, pre-registered criteria |

### Anti-Patterns Found

None. TBD/FIXME/XXX/TODO/placeholder scan on both docs: clean.

### Human Verification Required

None. The one remaining unknown (DeltaNet/VL runtime behavior on GB10 aarch64 without downloading weights) is explicitly documented as a Phase 20 bring-up smoke check gated on human sign-off — a scheduled future task, not an unverified claim in this phase.

### Gaps Summary

None remaining. The single gap from the initial verification (VL modality omission + understated bf16 footprint) was closed in commit `aecf7a0` and every fix claim was independently re-verified against the live HF primary sources this session. The memory finding actually strengthened: the measured 67.0 GiB full-checkpoint figure makes the quantization-mandatory conclusion firmer than the original 65.2 GiB derivation.

---

_Verified: 2026-07-11_
_Verifier: Claude (gsd-verifier)_
