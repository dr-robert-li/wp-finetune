---
phase: quick-260620-bwf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - CHANGELOG.md
  - JOURNAL.md
  - README.md
  - PROJECT.md
autonomous: true
requirements: [DOC-SYNC]
must_haves:
  truths:
    - "CHANGELOG [Unreleased] documents Phase 7 closure and Phase 8 reward infrastructure"
    - "JOURNAL has a new top entry covering Phase 7 closure (2026-06-19) and Phase 8 reward work"
    - "README Project Status table reflects v1.2 complete, Phase 7 closed, Phase 8 complete, correct v2.0/v3.0 milestone numbering"
    - "PROJECT.md Current Status reflects the true current position (next = Phase 9 GSPO)"
    - "Doc changes committed and pushed to origin/main with no unrelated files swept in"
  artifacts:
    - path: "CHANGELOG.md"
      provides: "Updated [Unreleased] section"
      contains: "reward"
    - path: "JOURNAL.md"
      provides: "New 2026-06 dated entry"
      contains: "## 2026-06"
    - path: "README.md"
      provides: "Corrected Project Status table + Current line"
    - path: "PROJECT.md"
      provides: "Corrected Current Status checklist"
  key_links:
    - from: "doc content"
      to: ".planning/STATE.md + .planning/ROADMAP.md + 08-*-SUMMARY.md"
      via: "executor reads source-of-truth artifacts, derives facts (does not invent)"
      pattern: "STATE.md|ROADMAP.md|SUMMARY"
---

<objective>
Synchronize project documentation with the current state of the project, then commit and push.

The latest dated JOURNAL entry (2026-06-15) is stale: it left Phase 7 sign-off OPEN and predates both the Phase 7 closure (council-reviewed APPROVED, Dr. Robert Li, 2026-06-19) and all Phase 8 "Reward Infrastructure" work (2026-06-19/20). CHANGELOG [Unreleased] has no Phase 7/8 entries. README's Project Status table and PROJECT.md's Current Status are structurally stale — they predate the v2.0/v3.0 roadmap reorder and still describe v1.2 as "Next" and Phase 1 re-judging as "Current".

Purpose: Make the human-facing docs (CHANGELOG, JOURNAL, README, PROJECT) factually match the project's true position so future readers and planning sessions start from accurate context.
Output: Updated CHANGELOG.md, JOURNAL.md, README.md, PROJECT.md — committed and pushed to origin/main.

SCOPE: Documentation-only. NO source code changes. Touch ONLY the four markdown files above. Do NOT hand-edit STATE.md or ROADMAP.md — they are tool-managed and already current; they are READ as the source of truth, not modified. docs/ skill files are generic skill descriptions with no status drift — leave them alone.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@CHANGELOG.md
@JOURNAL.md
@README.md
@PROJECT.md

# Source-of-truth artifacts for Phase 8 factual content (read, derive — do not invent):
@.planning/phases/08-reward-infrastructure/08-01-SUMMARY.md
@.planning/phases/08-reward-infrastructure/08-02-SUMMARY.md
@.planning/phases/08-reward-infrastructure/08-03-SUMMARY.md
@.planning/phases/08-reward-infrastructure/08-04-SUMMARY.md
@.planning/phases/08-reward-infrastructure/08-REVIEW.md
@.planning/phases/08-reward-infrastructure/08-HUMAN-UAT.md

<facts>
<!-- Ground truth already established. Executor must read the source artifacts above to flesh out detail, but these are the load-bearing facts the docs must assert. -->

TRUE CURRENT POSITION (per STATE.md frontmatter `status: milestone_complete`, ROADMAP checkboxes, git log):
- v1.0 MVP: complete (Phase 5 packaging deferred to v3.0, intentional)
- v1.1 Adaptive Training (Phase 6): complete
- v1.2 Judge Reasoning Fine-Tune (Phases 4.1–4.4): COMPLETE and PROMOTED — v4-winner canonical at models/qwen3-30b-wp-30_70-reasoning-merged-v4 (D-V4-10 waiver + REVL-05 sign-off, 2026-06-14)
- Phase 7 Router Profiling & Protected Expert Set: CLOSED 2026-06-19 — council-reviewed APPROVED by Dr. Robert Li
- Phase 8 Reward Infrastructure: COMPLETE — all 4 plans [x], 08-REVIEW status=clean. One human UAT test is PENDING (live 45-case anti-hack scoring) because it needs a live vLLM judge endpoint not available until Phase 9 infra; describe Phase 8 as "complete — in-scope deliverables built/tested; one live-endpoint UAT deferred to Phase 9".
- NEXT: Phase 9 GSPO Training.

PHASE 7 CLOSURE (source: STATE.md "Current Position" block, lines ~31-43):
- Profiling of canonical v1.2 model on matched 30/70 stimulus: 34,855 examples, 785.8M tokens, GB10 6h30m, rc=0
- All automated gates green: PROF-03 jaccard_ci_lower=0.9426≥0.94 (D-09 CI-aware gate cleared the bar where a point estimate would have failed — 6/48 layers below 0.94, L35 as low as 0.60); PROF-04 concentration E_eff gen 60.7 < judge 72.7
- 1,480 experts protected (mean-threshold, conservative); late-layer L45–47 E_eff +7 ACCEPTED as lawful routing shift; two judgment items accepted unanimously by SOTA council (GPT-5.5 / Opus 4.8 / Gemini 3.1 Pro)
- protected_expert_mask.json/.npy shippable to Phases 11/13

PHASE 8 REWARD INFRASTRUCTURE (source: 08-*-SUMMARY.md + ROADMAP Phase 8 + git log b202d38..5e15c76):
- Composite reward pipeline scripts/reward_pipeline.py: 70% verifiable (PHPCS + security + WP standards) / 30% frozen wp_judge score (GRPO-01, GRPO-02)
- Security TERMINAL hard gate: secure-failing generation → total reward 0 regardless of other signals (fail-closed; RuntimeError on empty trigger set per SC2)
- MO-GRPO within-group normalization: each signal normalized by within-group variance before combination; no single dominant signal can inflate total (GRPO-03)
- VeRPO partial credit on WP-standards subset: each check weighted by difficulty estimated from pass-rate across group samples; rare-pass checks contribute more (GRPO-04)
- judge_score_single() RC-A wrapper (enable_thinking=False guard fixing the unclosed-<think> parse failure) + injectable judge recalibration-offset loader inheriting the +3.58 calibration constant (D-V4-09) so reward-time judge matches gate-time judge
- Anti-hack eval set (D-11): build_antihack_set.py 3-axis perturb-real (verbose padding / template-critique collapse / self-preference swap) + background-agent scoring + CI-aware gate (hi_perturbed < lo_clean); CR-03 fix scores perturbed+clean in ONE combined compute_group_rewards call so normalization spans both groups
- Composite weights 35/35/30 (phpcs/verpo/judge) locked per D-08
- 08-REVIEW resolved status=clean; one live-endpoint UAT (45-case anti-hack scoring) deferred to Phase 9 vLLM infra

DRIFT LOCATIONS (executor must fix EXACTLY these — do not re-discover):
- README.md Project Status table (lines ~43-53): v1.2 shows "Next" (is COMPLETE/promoted); Phase 7 absent/under "Planned" (is CLOSED); milestone numbering is STRUCTURALLY WRONG — table predates the v2.0/v3.0 reorder. Table currently shows v2.0=MoE-Sieve(7-9) / v3.0=GRPO(10-14). ACTUAL per ROADMAP.md: v2.0 = Phases 7-10 (Router Profiling, Reward Infra, GSPO Training, RL Eval); v3.0 = Phases 11-15 (Post-RL MoE-Sieve, Sieve Eval, Merge+Pruning, Final Eval, Packaging). Fix the table to match ROADMAP's canonical structure.
- README.md "Current:" line (~L55): still narrates "Full pipeline re-execution in progress / Phase 1 re-judging underway" — replace with the true current position (v1.2 promoted, Phase 7 closed, Phase 8 complete, next = Phase 9 GSPO).
- PROJECT.md "Current Status" checklist (~L245-252): Phase C marked "next step" and milestone groupings (Phase D = v2.0 Phases 7-9, Phase E = v3.0 Phases 10-14) are stale vs ROADMAP reorder.
- PROJECT.md (~L96): "*Not yet started. Next milestone step.*" under Phase C — stale.
- ROADMAP.md is the CANONICAL milestone structure — align README/PROJECT numbering to it, do not invent a new scheme.
</facts>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update CHANGELOG [Unreleased] and prepend new JOURNAL entry</name>
  <files>CHANGELOG.md, JOURNAL.md</files>
  <action>
    Read STATE.md, ROADMAP.md (Phase 7/8 sections), and the 08-*-SUMMARY.md / 08-REVIEW.md / 08-HUMAN-UAT.md artifacts already referenced in context to derive accurate facts — do NOT invent numbers, commit hashes, or outcomes.

    CHANGELOG.md — update the existing [Unreleased] section (keep-a-changelog Added/Changed/Fixed grouping; do not create a new version heading). Add entries covering: (a) Phase 7 router-profiling closure — protected expert mask (1,480 experts, mean-threshold conservative), CI-aware Jaccard gate (jaccard_ci_lower=0.9426≥0.94 per D-09), E_eff concentration (gen 60.7 < judge 72.7), council-reviewed sign-off; (b) Phase 8 reward infrastructure — composite 70/30 reward pipeline (scripts/reward_pipeline.py), security terminal fail-closed gate (SC2), MO-GRPO within-group normalization, VeRPO difficulty-weighted partial credit, judge_score_single() RC-A wrapper + judge recalibration-offset loader (+3.58 / D-V4-09 inheritance), anti-hack eval set (build_antihack_set.py, 3-axis perturb-real, CI-aware gate, CR-03 combined-group fix). Group these under Added/Changed/Fixed as appropriate; cite the requirement IDs (PROF-*, GRPO-01..04, D-08, D-11) where they sharpen the entry.

    JOURNAL.md — prepend a NEW dated entry at the TOP, immediately after the header block (after the `---` separator on line 5, before the 2026-06-15 entry). Newest-first ordering. Match the existing voice: first-person, reflective, technical, format `## 2026-06-19 — <Title>` (use 2026-06-19/20 dates appropriate to the work; a single entry is fine, two if Phase 7 closure and Phase 8 read as distinct days). Cover: closing Phase 7 (the 06-15 entry left sign-off open — now signed, council-reviewed, mask immutable); and building the Phase 8 reward infrastructure (the first RL-facing artifact — composite reward, the security terminal gate, MO-GRPO normalization as the reward-hacking guardrail, VeRPO, the judge recalibration inheritance promised in the 06-14 entry now delivered, the anti-hack set and its CI-aware gate). Connect to the prior journal's recurring theme (be as rigorous about the measurement/reward as about the model — reward-hacking is a model exploiting a gate that doesn't know its own floor). Do NOT touch any existing entry.
  </action>
  <verify>
    <automated>grep -q "## 2026-06-19" JOURNAL.md &amp;&amp; grep -iq "reward" CHANGELOG.md &amp;&amp; grep -iq "anti-hack\|MO-GRPO\|reward pipeline" CHANGELOG.md &amp;&amp; grep -iq "protected\|Phase 7\|profiling" CHANGELOG.md &amp;&amp; echo PASS</automated>
  </verify>
  <done>CHANGELOG [Unreleased] documents Phase 7 closure + Phase 8 reward infra under proper Added/Changed/Fixed grouping; JOURNAL has a new top-of-file 2026-06-19/20 entry in the existing voice; no existing JOURNAL entry modified.</done>
</task>

<task type="auto">
  <name>Task 2: Correct README and PROJECT status to true current position</name>
  <files>README.md, PROJECT.md</files>
  <action>
    Use ROADMAP.md as the canonical milestone structure (do not invent numbering).

    README.md Project Status table (~L43-53): (1) change v1.2 Judge Reasoning row Status from "Next" to "Complete" (v4-winner promoted, D-V4-10 waiver + REVL-05 sign-off); (2) rebuild the v2.0 and v3.0 rows to match the ROADMAP reorder — v2.0 = Phases 7-10 (7 Router Profiling [Complete/closed 2026-06-19], 8 Reward Infrastructure [Complete], 9 GSPO Training [Next], 10 RL Comparative Eval [Planned]); v3.0 = Phases 11-15 (11 Post-RL MoE-Sieve, 12 Sieve Eval, 13 Merge+Pruning, 14 Final Eval, 15 Packaging — all Planned). Keep the v1.0/v1.1 rows accurate (already correct). (3) Replace the "Current:" line (~L55) — remove the stale "pipeline re-execution / Phase 1 re-judging" narrative; write the true position: v1.2 reasoning model complete + promoted (canonical merged-v4), Phase 7 router profiling closed (protected expert mask shipped, council-approved 2026-06-19), Phase 8 reward infrastructure complete (composite 70/30 reward pipeline built + tested; one live-endpoint anti-hack UAT deferred to Phase 9 infra), next = Phase 9 GSPO dual-mode RL training.

    PROJECT.md Current Status checklist (~L245-252): mark Phase C (Phase 4 eval/triage) complete, add v1.2 (4.1-4.4) complete + promoted, Phase 7 complete (closed 2026-06-19), Phase 8 complete; update the v2.0/v3.0 milestone groupings to match the ROADMAP reorder (v2.0 = Phases 7-10, v3.0 = Phases 11-15); mark next = Phase 9 GSPO. Also fix the stale "*Not yet started. Next milestone step.*" note under Phase C (~L96) to reflect that Phase 4 triage is complete. Keep edits surgical — only correct stale status/numbering, do not rewrite prose sections that are still accurate.

    Confirm Phase 8 closure nuance from 08-REVIEW.md (status: clean) and 08-HUMAN-UAT.md (one pending live-endpoint test) before writing "complete" — phrase it as complete with the live UAT deferred to Phase 9, not as "in progress".
  </action>
  <verify>
    <automated>! grep -q "v1.2 Judge Reasoning.*Next" README.md &amp;&amp; ! grep -iq "re-judging underway\|re-execution in progress" README.md &amp;&amp; ! grep -q "Phase C:.*next step" PROJECT.md &amp;&amp; grep -iq "Phase 9\|GSPO" README.md &amp;&amp; echo PASS</automated>
  </verify>
  <done>README status table shows v1.2 complete, Phase 7 closed, Phase 8 complete, and v2.0/v3.0 numbering matches ROADMAP (7-10 / 11-15); README Current line states the true position with next=Phase 9; PROJECT Current Status checklist and the Phase C note are corrected; no accurate prose was rewritten.</done>
</task>

<task type="auto">
  <name>Task 3: Commit scoped doc changes and push to origin/main</name>
  <files>CHANGELOG.md, JOURNAL.md, README.md, PROJECT.md</files>
  <action>
    The working tree is dirty with many UNRELATED untracked/modified files (data/, scripts/judge_*.py, logs/, *.augmented.jsonl, modified data JSONs, etc.). Do NOT use `git add -A` or `git add .` — that would sweep unrelated work into a docs commit.

    Stage ONLY the four doc files plus this quick-task plan directory, using explicit paths (use rtk per CLAUDE.md):
      rtk git add CHANGELOG.md JOURNAL.md README.md PROJECT.md .planning/quick/260620-bwf-update-all-docs-changelog-and-journal-md/

    Verify the staged set BEFORE committing: `rtk git status` must show only those paths staged and nothing else added. If anything unrelated is staged, unstage it (`git restore --staged <path>`) and re-stage correctly.

    Commit with the project's docs(...) convention. The git author is already correctly configured as Dr. Robert Li / dr.robert.li.au@gmail.com — do NOT change author or create a branch; commit directly on main. Suggested message:
      docs: sync CHANGELOG, JOURNAL, README, PROJECT with Phase 7 closure + Phase 8 reward infrastructure

    Then push: `rtk git push` to origin/main.

  </action>
  <verify>
    <automated>rtk git status --porcelain | grep -E "^[AM] +(CHANGELOG|JOURNAL|README|PROJECT)\.md" ; rtk git log --oneline -1 | grep -iq "docs" &amp;&amp; git status -sb | grep -q "## main...origin/main$" &amp;&amp; echo PUSHED</automated>
  </verify>
  <done>Only the four doc files (+ this plan dir) are committed in a single docs(...) commit; no unrelated files swept in; commit is pushed to origin/main; `git status` shows the branch in sync with origin/main with no remaining staged doc changes.</done>
</task>

</tasks>

<verification>
- JOURNAL.md: new top-of-file 2026-06-19/20 entry covering Phase 7 closure + Phase 8 reward infra; existing entries untouched.
- CHANGELOG.md: [Unreleased] documents Phase 7 closure and Phase 8 reward infrastructure (reward pipeline, MO-GRPO, VeRPO, security gate, anti-hack set).
- README.md: status table accurate (v1.2 complete, Phase 7 closed, Phase 8 complete, v2.0=7-10 / v3.0=11-15), Current line states next=Phase 9 GSPO, no Phase-1-re-judging narrative.
- PROJECT.md: Current Status checklist + Phase C note corrected; milestone numbering matches ROADMAP.
- Git: single scoped docs commit on main, pushed to origin/main; no unrelated working-tree files staged.
</verification>

<success_criteria>
- All four docs factually match the project's true position (v1.2 promoted, Phase 7 closed, Phase 8 complete, next = Phase 9 GSPO).
- Facts are derived from STATE.md / ROADMAP.md / 08-*-SUMMARY.md — not invented.
- No source code changed; only the four markdown files touched.
- Scoped commit pushed to origin/main; working tree's unrelated files left unstaged.
</success_criteria>

<output>
Create `.planning/quick/260620-bwf-update-all-docs-changelog-and-journal-md/260620-bwf-SUMMARY.md` when done.
</output>
