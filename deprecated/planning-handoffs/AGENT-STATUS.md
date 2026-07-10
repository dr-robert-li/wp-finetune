# AGENT-STATUS — wp-qwen3-moe live dashboard

**Last updated:** 2026-05-14T12:21:30Z (local 2026-05-14 22:21 GMT+10)
**Run cycle:** 2 (hourly refresh; no new commits since cycle 1)
**Agent role:** read-only planning observer; writes only this file

---

## 1. Currently engaged phase

**Phase 1b — Stratified Re-Judge (v2 calibration corpus rebuild)** — IN PROGRESS

- Active work: Stratified 20K re-judge of model output via local vLLM (Qwen3.6-35B-A3B-FP8, port 30000), feeding through the newly-calibrated XGBoost dual-head (`calibrated_overall`, `calibrated_verdict`) emitted by `score_code`.
- Last live signal: pilot PID 136878 noted in journal at 1h57m / 448 of 1000 rows when last checked (entry timestamp 2026-05-14 late-evening). No new commit in last ~3h.
- Most recent commit on this phase: `01bca50 feat(phase-1b): stratified re-judge pilot + calibrated_overall clip`.
- Untracked artifact: `data/phase1b/rejudge_full_20k.jsonl` (still 0 B — bulk run not yet kicked off; pilot output `data/phase1b/rejudge_pilot_1k.jsonl` ~725.8 KB, unchanged since cycle 1).

Note: The formal ROADMAP.md numbering (Phase 4, 4.1-4.4, 5-15) is **stale** — Phase 0 diagnostic and the seed-anchored rebuild (Phase 1a/1b/...) were inserted on 2026-05-12 ("Data-Quality-First Rebuild Over Elastic Distillation" plan approved). STATE.md still points to Phase 4.1/4.2; the actual recent commit stream is entirely Phase 0.x → 1a → 1b. Treat Phase 0/1a as the source of truth for in-progress work; Phase 4.1/4.2 are paused but unblocked-on-paper.

## 2. Next work

1. **Finish Phase 1b pilot review** — verify pilot output (`rejudge_pilot_1k.jsonl`) under v2 calibrated rubric, confirm bucket distributions match expectation.
2. **Kick the full 20K stratified re-judge** — populate `data/phase1b/rejudge_full_20k.jsonl` against the same vLLM endpoint.
3. **Phase 1c follow-up (carried from journal):** address regressor compression on hard-FAILs (within-FAIL ordinal signal loss flagged by Grok-4 council dissent); add features that discriminate hard-FAIL boundaries or expand GT range from 11 discrete FAIL scores.
4. **Phase 1d gate (from phase-0.3 commit):** add explicit gate on full 9-dim schema + 95% parse rate before any new train; current adapter only emits 6/9 dims (`sql_safety`, `wp_api_usage`, `error_handling`, `code_structure` missing).
5. **Stale Phase 4.2 work (paused, not blocking 1b):** if/when the rebuild path is revisited, `.planning/phases/04.2-reasoning-dataset-assembly/.continue-here.md` carries the resume context — validation script needs the 10-example batches + haiku + ThreadPoolExecutor pattern; `assemble_reasoning_dataset.py` already written.

## 3. Recently completed (last ~10)

| When | Item |
|------|------|
| 2026-05-14 | `01bca50` Phase 1b stratified re-judge pilot + calibrated_overall clip |
| 2026-05-14 | `ef429d0` DGX_TOOLBOX_ISSUES.md relocated to deps/ |
| 2026-05-14 | `8b1b2eb` Phase 1a schema-tolerant `derive_gt` + Pearson gate (v2 calibration) — drops 79 → 0, train 527 → 580, FAIL 47 → 100 |
| 2026-05-14 | `4930beb` Journal: phase 1a complete, both calibration gates pass (verdict acc 0.9231, regressor Spearman +0.8382 v1 / Pearson +0.7659 v2) |
| 2026-05-14 | `8363772` Phase 1a Steps 3-8 — calibration split + XGBoost dual-head |
| 2026-05-14 | `ba2c3bd` Journal: Phase 1a step 1 done, step 2 running |
| 2026-05-14 | `3de049d` Phase 1a Step 1 — 500 PASS-anchor pool via hybrid backend |
| 2026-05-14 | `2293775` Phase 1a Claude CLI OAuth into container + bind-mount |
| 2026-05-13 | dgx-toolbox issues #10 (torchcodec aarch64) + #11 (venv dispatch) filed and resolved (`45cc8a5`) |
| 2026-05-12 | `7ed07f3` Phase 0.3 final — Spearman -0.046 vs human GT → full Phase 1 rebuild justified, schema collapse 6/9 dims documented |

## 4. Full project progress table

Source of truth for this table: ROADMAP.md + JOURNAL.md cross-reference. Phases marked with ‡ are the (newer) seed-anchored rebuild path inserted on 2026-05-12 that supersedes the original v1.0 MVP path mid-stream — ROADMAP.md has not yet been updated to reflect them as first-class phases.

| Phase | Title | Status | Summary |
|-------|-------|--------|---------|
| 1 (orig) | Pipeline Ready | ✓ | All pipeline scripts hardened; repos.yaml populated. v1.0 work. |
| 2 (orig) | Dataset Production | ✓ | 86,542 examples, train/val/test split, multi-format export. Phase 2 gap closure complete. |
| 3 (orig) | Model Prep and Training | ✓ | Qwen3-30B-A3B fine-tuned (experiment_001 complete; 30/70 adapter; loss 0.298). |
| 4 (orig) | Evaluation (Triage) | ◐ | 30/70 declared winner via human override 2026-04-06; full eval Spearman 0.096 / parse rate 13.2% — verdict invalidated by Phase 0.3 (vs human GT: Spearman -0.046). Triage is dead. |
| 4.1 | Reasoning Data Gen (v1.2 inserted) | ✓ | 196 CoT + 179 CtF bulk examples accepted; human audit approved 20/20 each. Paused mid-flight by rebuild pivot. |
| 4.2 | Reasoning Dataset Assembly | ◐ | Validation script + assemble script written; consistency validation blocked on claude --print batch-size bottleneck. Paused 2026-04-23. |
| 4.3 | Reasoning Fine-Tune | ☐ | Not started. Unsloth PEFT stacking question (A vs B) unresolved. |
| 4.4 | Reasoning Eval & Merge | ☐ | Not started. wp-bench gate deferred here from triage. |
| 5 | Packaging | ☐ | Deferred to v3.0 Phase 15. |
| 6 | Adaptive Training Planner | ✓ | Complete 2026-04-01. v1.1 milestone. |
| 7 | Router Profiling & Protected Experts | ☐ | Blocked on 4.4. |
| 8 | Reward Infrastructure | ☐ | Blocked on 7. |
| 9 | GSPO Training | ☐ | Blocked on 8. |
| 10 | RL Comparative Eval | ☐ | Blocked on 9; gates v3.0. |
| 11 | Post-RL MoE-Sieve | ☐ | Blocked on 10. |
| 12 | MoE-Sieve Comparative Eval | ☐ | Blocked on 11. |
| 13 | LoRA Merge & Pruning (AIMER) | ☐ | Blocked on 12. |
| 14 | Final Comparative Eval | ☐ | Blocked on 13. |
| 15 | Packaging | ☐ | Final compression + HF Hub. |
| **0** ‡ | Diagnostic baseline (rebuild path) | ✓ | 0.1–0.3 + 0.4–0.13 follow-ups all done. Finding: base model E_eff `<wp_judge>` 69.96 vs `<wp_gen>` 59.48 (+10.48 delta) vindicates routing thesis; Spearman vs human GT -0.046 → rebuild justified. |
| **1a** ‡ | Seed-anchored calibration (XGBoost dual-head) | ✓ | Steps 1-8 done. v2 model PASS: verdict acc 0.8769, Pearson 0.7659. Schema-tolerant `derive_gt`; sklearn<1.7 pinned. |
| **1b** ‡ | Stratified 20K re-judge | ◐ | Pilot running (448/1000 at last check); full 20K bulk pending. Pilot output 725.8 KB; full output still 0 B as of cycle 2. |
| **1c** ‡ | Hard-FAIL regressor improvement | ☐ | Carried forward from 1a council dissent; not started. |
| **1d** ‡ | 9-dim schema + 95% parse-rate gate | ☐ | Required before any retrain; not started. |
| **2** ‡ | Diagnostic pilot (500-step LoRA on 5K slice) | ☐ | Per updated plan. |
| **3** ‡ | Full SFT on v2 dataset | ☐ | Per updated plan. |
| **4** ‡ | Independent eval (no Claude in loop) | ☐ | Per updated plan. |
| **5** ‡ | RL alignment via GSPO + verifiable rewards | ☐ | Per updated plan. |
| **6** ‡ | Compression — MoE-Sieve → LoRA merge → AIMER/REAP | ☐ | Per updated plan. |
| **7** ‡ | Packaging — BF16 + AWQ-4 + GGUF, HF + Ollama | ☐ | Per updated plan. |
| **8** ‡ | Parallel Qwen3.6-base variant | ☐ | Same dataset / task tokens, newer base. |

Legend: ✓ done · ◐ in progress / paused · ☐ pending

## 5. Open questions / blockers

- **ROADMAP vs reality drift:** ROADMAP.md still describes the v1.0/v1.2/v2.0/v3.0 path (Phase 4.x → 7-15). The actual work has forked to the seed-anchored rebuild path (Phase 0 → 1a → 1b → ...). STATE.md "current focus" line still says Phase 4.1. Recommend the rebuild path either replace or be slotted into the formal ROADMAP next time the user updates it. Flagged but not for this agent to fix.
- **Phase 4.2 limbo:** Phase 4.1 outputs (196 CoT + 179 CtF) are still on disk and uncommitted (`data/phase4_reasoning/...`). If rebuild path subsumes them, they may either be discarded or repurposed as additional FAIL-seed annotations once `derive_gt` schema-tolerance permits.
- **Unsloth PEFT stacking (Option A nested LoRA vs Option B LoRA on merged):** still unresolved — was a blocker for old Phase 4.3, will resurface for the rebuild Phase 3 (Full SFT).
- **Phase 1c (hard-FAIL regressor compression):** Grok council dissent flagged real loss of within-FAIL ordinal signal; documented as next regressor iteration but not yet ticketed.
- **Phase 1d (9-dim schema + 95% parse gate):** required before retrain; the 30/70 adapter currently emits only 6 of 9 dimensions.
- **Working tree state:** 16 modified files + 36 untracked (slight growth from cycle 1 — `.claude/scheduled_tasks.lock` + `.planning/AGENT-STATUS.md` now in the untracked set). Heavy uncommitted weight remains in `data/phase4_reasoning/`, `data/phase1_extraction/`, `data/phase1b/`, plus the `scripts/judge_*` / `scripts/_judge_*` family. STATE.md and ROADMAP.md both still have local edits.
- **Container ecosystem abrasiveness:** journal entry 2026-05-14 morning flags this explicitly — torchcodec aarch64 gaps, unsloth-studio bootstrap, PEFT pinning, sparkrun var-expansion bugs. Affects throughput, not a specific phase blocker.
- **Cycle-2 delta:** zero new commits, zero new artifacts since cycle 1 (~11 minutes prior). Pilot likely still mid-run or stalled; recommend a focused check next refresh if `rejudge_pilot_1k.jsonl` byte count has not advanced.

## 6. Source-of-truth notes (this cycle)

Files consulted:

- `.planning/AGENT-STATUS.md` (prior cycle-1 dashboard — read for cycle counter)
- `.planning/ROADMAP.md` (full file >25K tokens, deferred to outline + prior-cycle extraction; no new entries in git diff)
- `.planning/STATE.md` (full read; no change vs cycle 1)
- `.planning/phases/` directory listing — all 7 phase folders enumerated (`01-pipeline-ready`, `02-dataset-production`, `03-model-prep-and-training`, `04-evaluation`, `04.1-reasoning-data-generation-inserted`, `04.2-reasoning-dataset-assembly`, `06-adaptive-training-planner`); each contains the PLAN/SUMMARY/CONTEXT/RESEARCH/VERIFICATION set as expected — no new files added since cycle 1
- `data/phase1b/` (pilot 725.8 KB unchanged; full 20K still 0 B)
- `rtk git log --oneline -30` (HEAD still `01bca50`)
- `rtk git log --since="1 hour ago"` (only `01bca50` returned)
- `rtk git log --since="3 hours ago"` (top three commits returned — no progress since `01bca50`)
- `rtk git status` (16 modified, 36 untracked)

No errors reading any file. No git operation errors. Cycle 2 is a no-change-since-cycle-1 confirmation pass.

Next refresh: ~hourly cadence. Will drop to ~10 min if commits-in-last-5-min detected.
