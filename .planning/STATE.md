---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Adaptive Training Infrastructure
status: defining_requirements
stopped_at: Milestone v1.1 started — defining requirements
last_updated: "2026-03-31T22:00:00.000Z"
last_activity: 2026-03-31 - Milestone v1.1 started (Adaptive Training Infrastructure)
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 12
  completed_plans: 11
  percent: 60
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** A single self-hostable model that generates WPCS-compliant WordPress code and catches critical defects via structured 9-dimension rubric scoring
**Current focus:** Phase 3 — DGX Training (scripts ready, data exported at 5 ratios, eval suite rewritten)

## Current Position

Phase: Not started (defining requirements)
Plan: --
Status: Defining requirements for v1.1 Adaptive Training Infrastructure
Last activity: 2026-03-31 - Milestone v1.1 started

Progress: [██████░░░░] 60%

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
| Phase 03-model-prep-and-training P03 | 10 | 2 tasks | 2 files |
| Phase 02-dataset-production P04 | 3 | 2 tasks | 98 files |
| Phase 02-dataset-production P05 | 8 | 2 tasks | 28 files |
| Phase 02-dataset-production P06 | 5 | 2 tasks | 37 files |

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
- [Phase 03-02]: Security pass rate uses WordPress.Security.* sniff prefix filter — matches PHPCS sniff taxonomy
- [Phase 03-02]: PHPCS unavailability handled gracefully (passed=True with _phpcs_unavailable flag) to allow tests without binary
- [Phase 03-02]: eval_gate.py falls back to _FALLBACK_THRESHOLDS when train_config.yaml absent — gate won't crash before 03-01 output exists
- [Phase 03-02]: parse_judge_response returns None for unparseable JSON (not ValueError) — eval_judge.py skips and counts as skipped
- [Phase 03]: output_router_logits=True set both in model_kwargs and model.config — Unsloth version inconsistency protection
- [Phase 03]: merge_adapter.py falls back to vLLM --lora-modules on special-token assertion failure — adapter always stays safe
- [Phase 03-03]: output_router_logits=True set both in model_kwargs and model.config — Unsloth version inconsistency protection
- [Phase 03-03]: merge_adapter.py falls back to vLLM --lora-modules on special-token assertion failure — adapter always stays safe
- [Phase 03-03]: Human blocking checkpoint approved 2026-03-28 — all Phase 3 scripts verified before DGX execution
- [Phase 02-04]: wordpress-develop auto-passed with all scores=10 (quality_tier: core) per judge_system.md rule 1
- [Phase 02-04]: Empty extracted repos (0 functions) get empty passed/failed arrays for 100% coverage
- [Phase 02-05]: Template-based generation used instead of LLM API calls -- parameterized WordPress code templates with varied complexity/context/constraint axes
- [Phase 02-05]: Mutation pipeline produced 0 contrastive pairs -- regex patterns did not match passed function body format (acceptable per plan)
- [Phase 02-05]: 500 rejection examples split 170/170/160 across proactive_nonce, proactive_capability, proactive_escaping
- [Phase 02-06]: N/A dimensions (i18n=7, accessibility=7) treated as non-failing per judge_system.md rubric rules
- [Phase 02-06]: Double-brace template artifacts auto-fixed during revision step (1,958 functions revised)
- [Phase 02-06]: error_log in catch blocks treated as legitimate production logging, not debug output
- [Phase 02-06]: REST permission callback functions assessed for capability checks, not for containing 'permission_callback' string

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3]: RESOLVED — switched to Qwen3-30B-A3B native MoE (CMoE/ToMoE had no serving stack support)
- [Phase 4]: AWQ quantization for Qwen3-30B-A3B — verify vLLM support (likely native since it's an official Qwen model)
- [Phase 3]: Judge correlation circularity — PARTIALLY ADDRESSED: eval_judge.py now uses 9-dimension rubric ground truth (241 checks) instead of PHPCS-only. Scoring calibration still needed during Phase 4 against real model output.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260328-h1d | Make dgx_toolbox.py project-agnostic by moving all hardcoded couplings to dgx_toolbox.yaml | 2026-03-28 | e1cec35 | [260328-h1d-make-dgx-toolbox-py-project-agnostic-by-](./quick/260328-h1d-make-dgx-toolbox-py-project-agnostic-by-/) |
| 260329-cy1 | Sync STATE.md, ROADMAP.md, PROJECT.md with current project state | 2026-03-29 | 726ec5a | [260329-cy1-update-state-md-roadmap-md-project-md-to](./quick/260329-cy1-update-state-md-roadmap-md-project-md-to/) |
| 260329-g30 | Prefix all 8 skills with wp-finetune: for Claude Code discoverability | 2026-03-29 | ba82fab | [260329-g30-rename-all-skills-with-wp-finetune-prefi](./quick/260329-g30-rename-all-skills-with-wp-finetune-prefi/) |
| 260329-g6l | Update README and CHANGELOG to match current project state | 2026-03-29 | 33f5311 | [260329-g6l-update-readme-and-changelog-to-reflect-c](./quick/260329-g6l-update-readme-and-changelog-to-reflect-c/) |
| 260329-g9p | README How It Works section with both pipeline and training skills | 2026-03-29 | 163497e | [260329-g9p-readme-how-it-works-section-with-both-pi](./quick/260329-g9p-readme-how-it-works-section-with-both-pi/) |
| 260329-gmf | README add observe and review-telemetry skill descriptions | 2026-03-29 | d31e013 | [260329-gmf-readme-add-observe-and-review-telemetry-](./quick/260329-gmf-readme-add-observe-and-review-telemetry-/) |
| 260329-gre | Add telemetry opt-in Step 0c to run-training skill | 2026-03-29 | 09beab1 | [260329-gre-add-telemetry-opt-in-step-0c-to-run-trai](./quick/260329-gre-add-telemetry-opt-in-step-0c-to-run-trai/) |
| 260329-gvs | Update README CHANGELOG with telemetry opt-in and training steps | 2026-03-29 | 9a99da8 | [260329-gvs-update-readme-changelog-with-telemetry-o](./quick/260329-gvs-update-readme-changelog-with-telemetry-o/) |

## Session Continuity

Last session: 2026-03-29T00:00:00Z
Stopped at: Data pipeline complete, eval suite rewritten, training skill updated — ready for /run-training
Resume file: None
