---
phase: 09-gspo-training
plan: "06"
subsystem: skill-orchestration
tags: [skill, tinker, gspo, rl-training, orchestration]
dependency_graph:
  requires: [09-01, 09-03, 09-04, 09-05]
  provides: [run-rl-training-skill]
  affects: [operator-runbooks, claude-skills]
tech_stack:
  added: []
  patterns: [tinker-serviceClient, gspo-forward-backward-custom, rspo-floor, mo-grpo-normalization]
key_files:
  created:
    - .claude/skills/wp-finetune:run-rl-training/SKILL.md
  modified: []
decisions:
  - "GSPO (forward_backward_custom + RSPO stop-gradient floor) is primary/default loss; no flag required — locked D-09-03"
  - "GRPO importance_sampling is instability fallback only via --grpo-fallback / --no-gspo"
  - "Router gates frozen on Tinker — protected-expert Jaccard is monitor-only, no enforcement — locked D-09-02"
  - "Dispatch boundary: Agent(run_in_background=true) = telemetry monitor only; judge scoring = claude_agent subprocess inside rl_train.py"
  - "Dry-run (--dry-run) is mandatory preflight step before any real training run"
metrics:
  duration: "~15 min (context continuation from prior session)"
  completed: "2026-06-20"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 09 Plan 06: Tinker-Native run-rl-training Skill Summary

Authored `.claude/skills/wp-finetune:run-rl-training/SKILL.md`: 9-step Tinker-native RL training orchestrator with GSPO as primary/default loss, GRPO as documented fallback, mandatory dry-run preflight, background telemetry monitor, KL/MoE auto-halt handling, anti-hack regression gate, and Dispatch boundary rule.

## Task Results

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create run-rl-training skill | 427254a | .claude/skills/wp-finetune:run-rl-training/SKILL.md |

## Skill Structure

The skill mirrors the run-training structural pattern but fully reframes to Tinker primitives:

- **Step 0a–0d**: Confirm base model, dataset, experiment name, configuration
- **Step 1**: Configure base command (GSPO default, GRPO fallback optional)
- **Step 2**: Validate Tinker credentials and protected-expert mask
- **Step 3**: Spawn background telemetry monitor via `Agent(run_in_background=true)`
- **Step 4**: Mandatory dry run (`--dry-run`)
- **Step 5**: Run RL training — `rl_train.py` with full flag reference
- **Step 6**: Handle auto-halt outcomes (KL hard halt / e_frac hard halt)
- **Step 7**: Anti-hack regression gate (CI-aware, bootstrap lower bound)
- **Step 8**: Stop telemetry monitor (`touch output/rl_checkpoints/_stop`)
- **Step 9**: Verify checkpoint manifest + write run summary

## Acceptance Gates Verified

```
Gate 1 (dgx/unsloth/docker exec — non-quoted lines): 0 refs — PASS
Gate 2 (GRPO.*primary|200-step|gated upgrade): 0 refs — PASS
Plan verify command: PASS
Must-contain: run-rl-training, save_weights_for_sampler, use_gspo, Deviations,
              --dry-run, Dispatch boundary, frozen — all PASS
rtk prefix: 9 shell examples prefixed — PASS
GPU/thermal text: 0 refs — PASS
```

## Deviations from Plan

None — plan executed exactly as written. The skill correctly:
- Documents GSPO as PRIMARY (use_gspo defaults True, no flag needed)
- Documents GRPO as FALLBACK only (`--grpo-fallback` / `--no-gspo`)
- Contains zero DGX/unsloth/docker references
- Omits any "GRPO primary", "200-step gate", or "gated upgrade" framing

## Known Stubs

None. Skill is a runbook/orchestration document — no data-wiring stubs applicable.

## Threat Flags

None. Skill is a runbook document — no new network endpoints or auth paths introduced. Credential hygiene rule (Key Rule 6) explicitly directs operators to use `~/.tinker` or env, never hardcode tokens.

## Self-Check: PASSED

- `.claude/skills/wp-finetune:run-rl-training/SKILL.md` exists: CONFIRMED
- Commit 427254a exists: CONFIRMED (`git log` verified)
- All gates: PASS
