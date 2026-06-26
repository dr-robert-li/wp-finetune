---
phase: 02-dataset-production
verified: 2026-06-26T00:00:00Z
status: passed
score: 5/5 (4 verified + C5 accepted via override 2026-06-26)
override_accepted: true
overrides:
  - must_have: "The wp_gen and wp_judge example counts follow approximately 40/60 gen/judge split"
    reason: >-
      Static 40/60 export target (a Phase-2 user decision) superseded by the
      ratio_30_70..ratio_70_30 export sweep; the 30/70 ratio adapter was selected
      as the Phase 4 triage winner (ROADMAP line 41) and is the basis for all
      downstream work (Phases 4.3, 7-10). Downstream training consumes the
      per-ratio split dirs (data/final_dataset/ratio_30_70/openai_train.jsonl),
      NOT the top-level blend whose ratio is therefore moot. This also resolves
      the REQUIREMENTS DATA-11 (~50/50) vs ROADMAP C5 (40/60) target conflict —
      both static targets are superseded by the empirical 30/70 selection.
    accepted_by: "Dr. Robert Li"
    accepted_at: "2026-06-26T12:31:27Z"
re_verification: true
re_verification_note: >-
  The prior 02-VERIFICATION.md (dated 2026-03-26, status gaps_found, score 0/5)
  PREDATED pipeline execution. The pipeline actually ran 2026-03-29 via the
  /run-data-pipeline skill (ROADMAP line 36) and outputs were produced/refreshed
  through 2026-04-23. The prior "nothing ran / dirs empty" finding is obsolete.
  Outputs also moved from repo-root (phase1_extraction/, final_dataset/) to
  under data/. This report verifies against current disk state.
re_verification_data:
  previous_status: gaps_found
  previous_score: 0/5
  gaps_closed:
    - "C1: repos cloned + PHP functions extracted with metadata (was 'dirs empty')"
    - "C2: PHPCS pre-filter + Claude judging with passed/failed separated (82165 passed / 12349 failed)"
    - "C3: gap analysis + synthetic generation executed"
    - "C4: data/final_dataset/ produced with OpenAI/Alpaca/raw formats, 80/10/10, task tokens"
  gaps_remaining: []
  gaps_accepted_via_override:
    - "C5: gen/judge split ~92/8 vs 40/60 target — ACCEPTED 2026-06-26 (Dr. Robert Li). Static target superseded by the ratio_30_70..70_30 sweep + 30/70 Phase-4 winner; see overrides block."
  regressions: []
gaps:
  - truth: "The wp_gen and wp_judge example counts follow approximately 40/60 gen/judge split"
    status: accepted_override
    reason: >-
      Canonical dataset (data/final_dataset/wordpress_finetune.jsonl) contains
      72033 <wp_gen> vs 5996 <wp_judge> tokens = 92.3% gen / 7.7% judge —
      strongly gen-skewed, missing the 40/60 target (and also missing the
      ~50/50 target that REQUIREMENTS.md DATA-11 states — the two source docs
      disagree on the target). Separately, metadata.json's reported ratio is a
      REPORTING BUG: gen_ratio_actual=1.0 / judge_ratio_actual=0.0 is impossible
      given gen_count=72033 / judge_count=11219 (which would compute to ~86/14).
      Neither the buggy field NOR the recomputed count matches target. The
      ratio_30_70 .. ratio_70_30/ subdirs (Mar 29) show the 40/60 single-target
      was superseded by a downstream ratio SWEEP; ROADMAP line 41 records the
      30/70 ratio adapter as the eventual Phase 4 winner. This may be an
      intentional, accepted deviation — needs human confirmation (override path
      below).
    artifacts:
      - path: "data/final_dataset/wordpress_finetune.jsonl"
        issue: "92.3/7.7 gen/judge token split vs ~40/60 target"
      - path: "data/final_dataset/metadata.json"
        issue: "gen_ratio_actual=1.0 / judge_ratio_actual=0.0 is a contradictory reporting bug; gen_count/judge_count compute to ~86/14, still off-target"
    missing:
      - "Either rebalance gen/judge toward 40/60 (or the 50/50 DATA-11 target), OR record an override confirming the single 40/60 target was intentionally replaced by the ratio_* sweep + 30/70 winner"
      - "Fix the *_ratio_actual computation in export_dataset.generate_metadata() so the field is internally consistent with the counts"
      - "Reconcile the REQUIREMENTS.md DATA-11 (~50/50) vs ROADMAP C5 (40/60) target disagreement"
human_verification:
  # [RESOLVED 2026-06-26 — Dr. Robert Li] "Was the static 40/60 gen/judge target formally
  # abandoned in favor of the ratio sweep + 30/70 Phase-4 winner?" YES — confirmed via the C5
  # override (see overrides block). No longer a pending item.
  - test: "Resolve the cross-format vintage patchwork in data/final_dataset/"
    expected: >-
      openai_*/wordpress_finetune (86542, Apr 23) vs alpaca_*/raw_* (19198, Apr 11)
      are NOT the same dataset snapshot. Confirm which is canonical for training
      and whether alpaca/raw should be regenerated from the 86542-example set.
    why_human: "Which snapshot is authoritative is a project decision, not derivable from disk"
  - test: "Spot-check 20 random examples from data/final_dataset/openai_train.jsonl"
    expected: "Realistic WordPress PHP, task tokens present, rejection/security examples proactively add nonce/capability/escaping"
    why_human: "Content quality requires human judgment"
---

# Phase 2: Dataset Production Verification Report

**Phase Goal:** The full three-phase data pipeline executes against real repositories and produces a clean, split, multi-format training dataset
**Verified:** 2026-06-26
**Status:** passed (C5 accepted via override — see frontmatter `overrides:`; accepted by Dr. Robert Li 2026-06-26)
**Re-verification:** Yes — replaces stale 2026-03-26 report that PREDATED execution

## Re-Verification Context

The prior `02-VERIFICATION.md` was dated **2026-03-26** and concluded `gaps_found` / 0-of-5 / "pipeline never ran, output dirs empty." That report **predates the actual pipeline execution on 2026-03-29** (`/run-data-pipeline` skill, ROADMAP line 36), with outputs refreshed through 2026-04-23. Outputs also relocated from repo root to under `data/`. Its script-readiness assessment (17/17 plan must_haves) remains valid and is not re-litigated here. This report verifies the **execution outputs** that the old report could not see.

## Goal Achievement — Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Repos shallow-cloned + PHP functions extracted with metadata | ✓ VERIFIED | `data/phase1_extraction/output/extracted/` has 204 per-repo files; passed records carry full metadata (`function_name, class_context, docblock, body, start_line, dependencies, sql_patterns, hooks_used, line_count, source_repo, source_file, quality_tier, assessment, training_tags`) |
| 2 | PHPCS pre-filter before Claude judging; passed/failed in separate files | ✓ VERIFIED | `passed/` = 82165 functions across 204 files; `failed/` = 12349 functions across 204 files; separate directories; records carry `quality_tier` + `assessment` (judge verdict). `php_lint_failures: 0` in metadata |
| 3 | Gap analysis identifies under-represented categories + synthetic generation fills gaps | ✓ VERIFIED (residual noted) | `data/phase2_synthetic/gap_report.json` (15517 passed functions analyzed, per-tag deficits w/ have/need/deficit/fill_pct); synthetic output present (`generated/` 18, `judged/` 6, `mutated/` 1, `judge_training/` 23). Major gaps filled in final coverage (e.g. prepared-statements 494). **Residual:** `taxonomy_gaps_remaining` lists ~40 long-tail categories (walkers, save_post, list-table = 1–6 each). Passes "identify + fill"; would fail a strict "every category ≥20" bar |
| 4 | `data/final_dataset/` ≥10,000 in OpenAI JSONL + Alpaca JSON + raw JSONL, 80/10/10 split, task tokens | ✓ VERIFIED (cross-format patchwork — WARNING) | OpenAI: train 69233 / val 8654 / test 8655 = **86542**, exact **80.00/10.00/10.00** split, `<wp_gen>`/`<wp_judge>` tokens present. All three formats exist, each >10k. **WARNING:** vintage mismatch — openai_*/wordpress_finetune are Apr-23 (86542); alpaca_*/raw_* are Apr-11 (**19198** = 15358+1919+1921). The three formats are NOT the same snapshot. Literal criterion met (3 formats, each ≥10k, each 80/10/10, tokens present); cross-format incoherence is a non-blocking flag |
| 5 | wp_gen/wp_judge counts follow ~40/60 gen/judge split | ✗ FAILED | Canonical `wordpress_finetune.jsonl`: **72033 `<wp_gen>` / 5996 `<wp_judge>` = 92.3% / 7.7%** — strongly gen-skewed, misses 40/60. metadata `*_ratio_actual` (1.0/0.0) is a contradictory reporting bug; even its raw counts recompute to ~86/14. Off-target by every measure. See dedicated analysis below |

**Score:** 4/5 success criteria verified; C5 accepted via override (30/70 sweep superseded the static 40/60 target) → phase closed as **passed**

## Criterion 5 Deep Analysis (the requested honest verdict)

Three candidate explanations were posed; the evidence supports **all three simultaneously**:

- **(c) Reporting bug — CONFIRMED.** `metadata.json` reports `gen_ratio_actual: 1.0` and `judge_ratio_actual: 0.0`. This is impossible: `judge_count: 11219` is non-zero, so judge ratio cannot be 0.0. The `*_ratio_actual` fields are computed incorrectly in `export_dataset.generate_metadata()`.
- **(a) Real unmet gap — CONFIRMED.** Independent of the buggy field, the actual data is gen-skewed. Authoritative token count on the canonical file: **gen 92.3% / judge 7.7%** (72033 / 5996; the remaining ~8513 of 86542 records carry neither token — rejection/other). Even the metadata `gen_count`/`judge_count` recompute to ~86.5/13.5. No reading lands near 40/60 (or the ~50/50 DATA-11 target).
- **(b) Superseded target — LIKELY, needs human confirmation.** `data/final_dataset/ratio_30_70/ … ratio_70_30/` subdirs (Mar 29) contain full re-exported datasets at five different gen/judge ratios — i.e. the single 40/60 export target was replaced by a downstream ratio SWEEP. ROADMAP line 41 records the **30/70 ratio adapter** as the Phase 4 triage winner. This strongly suggests intentional supersession, but intent cannot be confirmed from disk — routed to human verification + override path.

**Plain verdict:** Criterion 5 is genuinely unmet by the canonical dataset (≈92/8 gen-skewed). It is *not* merely a reporting bug — the bug is real but secondary. The likely intended resolution is an accepted target change (sweep + 30/70 winner), which a human should confirm via override.

## Requirements Coverage (DATA-01 … DATA-11)

| Req | Description | Status | Evidence |
|-----|-------------|--------|----------|
| DATA-01 | Repos shallow-cloned | ✓ SATISFIED | 204 repos represented across extracted/passed/failed |
| DATA-02 | PHP functions extracted with metadata | ✓ SATISFIED | extracted/ 204 files; 14-field metadata records |
| DATA-03 | Judge: PHPCS pre-filter + Claude, passed/failed separated | ✓ SATISFIED | passed 82165 / failed 12349, separate dirs, `assessment`+`quality_tier` |
| DATA-04 | Gap analysis vs taxonomy | ✓ SATISFIED | gap_report.json with per-tag deficits |
| DATA-05 | Mutation: contrastive bad→good pairs | ✓ SATISFIED | phase2 `mutated/` output present |
| DATA-06 | Generate: synthetic fills gaps | ✓ SATISFIED | phase2 `generated/` (18); coverage shows filled categories |
| DATA-07 | Judge synthetics, failed revised | ✓ SATISFIED | phase2 `judged/` output present |
| DATA-08 | Rubric-scored judge training data | ✓ SATISFIED | phase2 `judge_training/` (23) |
| DATA-09 | Phase 3 CoT reasoning chains | ✓ SATISFIED | data/phase3_cot/output/ 7 CoT artifacts (gen/judge/security/contrastive/rubric) |
| DATA-10 | Export OpenAI+Alpaca+Raw, tokens, 80/10/10 | ✓ SATISFIED (w/ C4 cross-format WARNING) | OpenAI 86542 @ 80/10/10 w/ tokens; alpaca/raw exist but are a 19198 vintage snapshot |
| DATA-11 | ≥10,000 examples with ~50/50 wp_gen/wp_judge | ⚠️ PARTIAL — split NOT satisfied | ≥10,000 ✓ (86542). Split ✗: actual ≈92/8. **Note:** REQUIREMENTS.md says ~50/50 here while ROADMAP C5 says 40/60 — the two contracts disagree; actual misses BOTH. Marked [x] Complete in REQUIREMENTS.md but the split sub-clause is not met |

## Anti-Patterns / Notes

- `// TODO:` markers appear inside generated PHP **training content** (e.g. the Elementor `process_element_export_import_content` example) — these are properties of the source/synthetic code being learned, NOT phase debt markers in pipeline code. Not a blocker.
- `*_backup_20260411` and `{output`/`{output}` sibling directories exist under data/phase1/2/3 — stale backup/scratch artifacts, not part of the canonical pipeline output. Cosmetic.

## Gaps Summary

The pipeline executed end-to-end and the goal is substantially achieved: real repos extracted, PHPCS+Claude judged with clean passed/failed separation, gaps analyzed and filled, and a clean 86542-example dataset split exactly 80/10/10 with task tokens. **One criterion (C5) genuinely fails:** the gen/judge balance is ~92/8, far from the 40/60 target, compounded by a contradictory `*_ratio_actual` reporting bug. Evidence (the `ratio_*` sweep + the 30/70 Phase-4 winner) suggests the static 40/60 target was intentionally superseded — but that is a human decision, not a disk fact, so C5 is reported as a gap with an override path rather than rubber-stamped. A secondary non-blocking warning: the three export formats are not the same vintage (openai/wordpress = 86542 Apr-23; alpaca/raw = 19198 Apr-11).

**Resolution — OVERRIDE ACCEPTED (2026-06-26, Dr. Robert Li).** The C5 deviation is confirmed intentional: the static 40/60 export target (a Phase-2 user decision) was superseded by the `ratio_30_70..ratio_70_30` sweep, and the **30/70** ratio adapter was selected as the Phase 4 triage winner (ROADMAP line 41) — the basis for all downstream work (Phases 4.3, 7–10). Downstream training consumes `data/final_dataset/ratio_30_70/`, not the top-level blend, so the blend's ≈92/8 ratio is moot. This also resolves the REQUIREMENTS DATA-11 (~50/50) vs ROADMAP C5 (40/60) conflict — both static targets are superseded by the empirical 30/70 selection. See the `overrides:` block in this file's frontmatter. **Phase 2 status → passed.**

Two non-blocking items remain documented (intentionally NOT fixed during closeout — track separately if desired):
1. The `*_ratio_actual` reporting bug in `export_dataset.generate_metadata()` (`gen_ratio_actual: 1.0 / judge_ratio_actual: 0.0` is impossible given non-zero `judge_count`).
2. The alpaca/raw vs openai cross-format vintage mismatch (alpaca/raw are a 19,198-example Apr-11 snapshot vs the 86,542 Apr-23 openai set). No consumers of alpaca/raw were found in `scripts/`/`config/`, so it is cosmetic for current training.

---

_Verified: 2026-06-26_
_Verifier: Claude (gsd-verifier) — re-verification against current disk state_
