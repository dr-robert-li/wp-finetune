---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03-02-PLAN.md
last_updated: "2026-03-27T22:28:00.000Z"
last_activity: 2026-03-27 — Completed 03-02 eval suite (eval_gen, eval_judge, eval_gate + 11 tests, 74 total passing)
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 9
  completed_plans: 7
  percent: 55
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** A single self-hostable model that generates WPCS-compliant WordPress code and catches critical defects via structured 9-dimension rubric scoring
**Current focus:** Phase 3 - Model Prep and Training

## Current Position

Phase: 3 of 4 (Model Prep and Training)
Plan: 2 of 3 in current phase (03-01, 03-02 complete)
Status: In progress
Last activity: 2026-03-27 — Completed 03-02 eval suite (eval_gen, eval_judge, eval_gate + 11 tests, 74 total passing)

Progress: [██████░░░░] 55%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 9 min
- Total execution time: 0.62 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-pipeline-ready | 2 | 27 min | 13 min |
| 02-dataset-production | 2 | ~10 min | ~5 min |
| 03-model-prep-and-training | 2 | 34 min | 17 min |

**Recent Trend:**
- Last 5 plans: 03-02 (22 min), 03-01 (12 min), 01-02 (2 min), 01-01 (25 min)
- Trend: steady

*Updated after each plan completion*
| Phase 02-dataset-production P01 | 25 | 2 tasks | 7 files |
| Phase 02-dataset-production P03 | 4 | 2 tasks | 3 files |
| Phase 03-model-prep-and-training P01 | 12 | 2 tasks | 5 files |
| Phase 03-model-prep-and-training P02 | 22 | 2 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Download Qwen3-30B-A3B (native MoE) BEFORE tokenizer extension and fine-tuning
- [Init]: Use Batch API for bulk judging (50% cost savings, required by PIPE-04)
- [Init]: Eval scripts must be written before training completes (gate cannot function otherwise)
- [Init]: Keep LoRA adapter separate from base model until all three eval thresholds pass
- [01-02]: quality_tier=trusted requires zero total_known_vulns AND zero unpatched AND rating >= 90; otherwise assessed
- [01-02]: active_installs '+' suffix stripped before int() conversion (e.g. '10000000+' -> 10000000)
- [01-02]: github_url must start with 'https://github.com/' — git clone access required
- [01-02]: If >100 repos pass filter, top 100 by active_installs retained
- [01-02]: WordPress Core hardcoded as first entry with quality_tier=core — not sourced from CSV
- [01-01]: Batch threshold hardcoded at 50 (BATCH_THRESHOLD constant) per PIPE-04 spec
- [01-01]: Checkpoint uses phase name as key so multiple pipeline stages coexist without collision
- [01-01]: preflight.py catches FileNotFoundError so tests pass on machines without php/phpcs
- [Phase 02-01]: Judge PASS threshold raised from >= 7 to >= 8 — stricter quality bar required before any pipeline execution (Pitfall 1 from research)
- [Phase 02-01]: Security auto-FAIL enforced in judge.py code (_apply_security_auto_fail function) — code-level gate not just config documentation
- [Phase 02-01]: N/A scoring deflated to 7 (from 10) for i18n and accessibility dims — prevents inflation on functions with no relevant output
- [Phase 02-01]: Rejection templates use 3 sub-keys (proactive_nonce, proactive_capability, proactive_escaping) in synthetic_prompts.yaml — aligned with security training taxonomy
- [Phase 02-02]: PHPCS hard-fail guard added at module level in phase2_mutate.py — no silent fallback on FileNotFoundError from verify_mutation_detectable
- [Phase 02-02]: batch results saved to disk immediately in phase2_judge_dataset after parse_batch_results (24h expiry protection, Pitfall 3)
- [Phase 02-02]: security auto-FAIL enforced in _apply_security_auto_fail() in phase2_judge.py — score < 5 forces FAIL verdict
- [Phase 02-03]: round() used instead of int() for gen/judge ratio calculation to avoid float precision truncation (0.60/0.40=1.4999... causes int to give 29 not 30)
- [Phase 02-03]: utils.py checkpoints save every 100 examples in phase3_cot.py (authoritative resume); per-500 progress JSONL files kept for additional recovery
- [Phase 02-03]: deduplicate() uses SHA-256 of assistant message content as reliable duplicate signal
- [Phase 03-01]: load_in_4bit=False LOCKED in prepare_tokenizer.py — no QLoRA for MoE (Qwen3-30B-A3B is MoE, QLoRA destabilizes routing)
- [Phase 03-01]: Mean embedding init: new token rows set to mean of existing embed_tokens rows (not random) for stable early training
- [Phase 03-01]: Model saved back to local_dir after embedding resize — ensures model and tokenizer vocab sizes are consistent
- [Phase 03-01]: All Phase 3 hyperparameters externalized in config/train_config.yaml (no hardcoded values in scripts)

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: RESOLVED — switched to Qwen3-30B-A3B native MoE (CMoE/ToMoE had no serving stack support)
- [Phase 4]: AWQ quantization for Qwen3-30B-A3B — verify vLLM support (likely native since it's an official Qwen model)
- [Phase 3]: Judge correlation circularity — decide whether to use a different Claude model or human-scored subset for eval before training starts

## Session Continuity

Last session: 2026-03-28T00:12:00.000Z
Stopped at: Completed 03-01-PLAN.md
Resume file: .planning/phases/03-model-prep-and-training/03-01-SUMMARY.md
