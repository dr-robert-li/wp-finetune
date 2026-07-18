# deprecated/planning-handoffs/ and ../planning-reviews/

Dangling session-scoped documents swept out of the repo root and `.planning/` during the Phase 16
cleanup. None of them is referenced by the active pipeline (`PIPELINE.md`, `scripts/`, `eval/`, the
`wp-finetune:*` skills). They are kept for provenance, not deleted.

## planning-handoffs/ — session handoffs and status

Ephemeral "where I left off" notes and the live status dashboard. Each was a resume pointer for a specific
work session; the durable record of every decision lives in `JOURNAL.md`, `.planning/STATE.md`, and the
per-phase `SUMMARY.md`/`VERIFICATION.md` files (which stayed in `.planning/`).

- `root.continue-here.md` — the last root-level resume note (superseded by STATE.md).
- `phase04.1.continue-here.md`, `phase08.2.continue-here.md`, `phase10.continue-here.md` — per-phase resume notes.
- `04.3-continue-here-PREPIVOT-discriminator.md` — pre-Tinker-pivot handoff (was already under an `_archive` dir).
- `09-HANDOFF.md`, `09-DUAL-DRY-RUN-HANDOFF.md`, `09-LOCAL-RL-HANDOFF.md` — RL (GSPO) run handoffs. The RL
  track was rejected; see PIPELINE.md conditional gate A.
- `AGENT-STATUS.md` — the live agent dashboard (a running status mirror of STATE.md).

## ../planning-reviews/ — phase review docs

Standalone cross-AI / human review write-ups. Their outcomes were folded into the phase closures; the docs
themselves are not part of the streamlined pipeline.

- `07-HUMAN-REVIEW.md` — Phase 7 human/council review of the protected-expert set.
- `08-REVIEW.md` — Phase 8 reward-infrastructure review.
- `09-REVIEW.md` — Phase 9 GSPO training review.

## References

Active-pipeline breadcrumbs that pointed at these files were updated to the deprecated paths (comments in
`scripts/rl_judge_dispatch.py`, `scripts/serve_consistency_vllm.sh`) or genericized
(`scripts/validate_reasoning_consistency.py`). Historical mentions inside `.planning/STATE.md`,
`ROADMAP.md`, and `CHANGELOG.md` are dated log entries and were left as written.
