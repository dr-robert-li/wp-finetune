---
phase: "09"
name: "GSPO Training"
audit_date: "2026-06-20"
asvs_level: 1
auditor: "claude-sonnet-4-6 / secure-phase"
threats_total: 15
threats_verified: 15
threats_open: 0
status: SECURED
---

# Phase 9 Security Audit — GSPO Training

## Summary

All 15 unique threat IDs declared across the six Phase 9 PLAN.md threat registers have
verified code-level mitigations or accepted dispositions. No HIGH-severity threats are
open. The phase is cleared to ship under the ASVS Level 1 / block-on-HIGH rule.

---

## Accepted-Risks Log

Threats with disposition `accept` must appear here per audit protocol.

| Threat ID | Plans | Rationale |
|-----------|-------|-----------|
| T-09-SC | 01, 02, 03, 04, 05, 06 | Supply-chain risk accepted. No new packages introduced across all six plans. Dependency surface unchanged from Phase 8 baseline. |

---

## Per-Threat Verification Table

Threat IDs are unique; IDs appearing in multiple plans are de-duped to a single row
with the originating plan noted.

| Threat ID | Severity | Disposition | Plans | Status | Evidence |
|-----------|----------|-------------|-------|--------|----------|
| T-09-POISON | MEDIUM | mitigate | 01 | CLOSED | `scripts/build_rl_prompts.py` docstring lines 1-30 excludes val/pre_vendorfilter/synthetic; `data/rl_prompts/PROVENANCE.md` lines 63-69 records lineage to Phase-4.2 train corpus only |
| T-09-LEAK | MEDIUM | mitigate | 01 | CLOSED | `scripts/build_rl_prompts.py` lines 95-108 loads val sha256 set; lines 157-160 rejects any row matching val hash; `data/rl_prompts/PROVENANCE.md` line 45: "Assertion: NO val-set user-content sha256 appears in either RL prompt pool." Val-leakage dropped: 0 |
| T-09-SC | LOW | accept | 01-06 | CLOSED | See accepted-risks log above |
| T-09-STALE | LOW | mitigate | 02, 06 | CLOSED | `.claude/skills/wp-finetune:run-rl-training/SKILL.md` Deviations section: GSPO primary, GRPO fallback; 0 DGX references in Phase 9 block; ROADMAP.md Phase 9 corrected (commit 7ba1783) |
| T-09-RWD-HACK | HIGH | mitigate | 03 | CLOSED | `scripts/rl_judge_dispatch.py` line 32: `from scripts.claude_agent import generate_json`; `scripts/claude_agent.py` lines 130-132: `--print`, `--no-session-persistence`, `--tools ""` (subprocess, no Anthropic API, tools disabled); `reward_pipeline.py` unmodified (Phase 8 commit b202d38 confirmed in 09-04-SUMMARY.md line 130: `git diff --stat scripts/reward_pipeline.py` → empty) |
| T-09-INJECT | HIGH | mitigate | 03 | CLOSED (residual) | `scripts/rl_judge_dispatch.py` JUDGE_SYSTEM rubric lines 40-68 (structured 0.0-1.0 anchored rubric); PHP code in fenced ```php``` blocks lines 98-101; prompt requests single numeric `consistency_score` field; score clamped `max(0.0, min(1.0, score))` line 149; N-vote median `statistics.median(scores)` line 158. **Residual:** clamp+median bound damage but cannot prevent within-[0,1] manipulation; monitored via N-vote variance logging |
| T-09-SELFPREF | MEDIUM | mitigate | 03 | CLOSED (research-gated) | `scripts/rl_train.py` `_panickssery_spot_check` lines 494-527; called at line 581 every `step % 50 == 0`; WR-04 fix confirmed: lines 508-510 use `getattr(bd, ...)` not `isinstance(bd, dict)`; logs alert if `|fix_correctness - judge_consistency| > 0.3`. **Research-gated:** quantification of self-preference bias remains RESEARCH-GATED per 09-CONTEXT.md D-09-05 R1; spot-check is the declared mitigation, not elimination |
| T-09-NOISE | LOW | mitigate | 03 | CLOSED | N-vote median: `statistics.median(scores)` `scripts/rl_judge_dispatch.py` line 158; collision-safe cache key: `json.dumps([php_code[:512], critique_text[:512]])` line 80 (WR-02 fix); group-mean imputation for timeouts lines 291-307; `_SUBPROCESS_TIMEOUT_S = 110` < asyncio 120s (WR-03 partial mitigation) |
| T-09-RWD-CAP | HIGH | mitigate | 04 | CLOSED | `scripts/rl_rollouts.py` line 54: `assert judge_consistency_weight <= 0.5` (import-time); line 196: `if weight > 0.5: raise ValueError` (runtime); default `judge_consistency_weight = 0.3` line 51; CR-05 fix confirmed: `collect_rollouts` lines 566-573 pass `fix_correctness_scores[i]` raw (not z-scored) to `combine_judge_reward`; line 571 comment: "Combine on the RAW [0, 1] scale (D-09-05 guard 1)... CR-05" |
| T-09-SECDROP | HIGH | mitigate | 04 | CLOSED | `scripts/rl_rollouts.py` `build_trajectory_groups` lines 255-263: `getattr(breakdown, "security_fail", False)` drops member when True; `scripts/reward_pipeline.py` line 590: `RewardBreakdown` instantiated with `security_fail=sec_fail` where `sec_fail = _security_fail(rubric)` line 579; gen path uses real `RewardResult` from `reward_pipeline.compute_group_rewards` which sets `security_fail` on the breakdown object — attribute reliably present. CR-06 fix confirmed: `_inline_remove_constant_reward_groups` uses `_group_by_id` per-prompt grouping |
| T-09-CRED | HIGH | mitigate | 05, 06 | CLOSED | `scripts/rl_train.py` docstring line 14: "no hardcoded credentials. ServiceClient() reads ~/.tinker or env."; grep of all Phase 9 scripts: 0 matches for `sk-`, `api_key=`, `TINKER_TOKEN` literal; `scripts/claude_agent.py` credentials sourced from environment only |
| T-09-CKPT | MEDIUM | mitigate | 05 | CLOSED | `scripts/rl_train.py` `_save_checkpoint` line 429: `tc.save_weights_for_sampler(name=name, ttl_seconds=None)`; emergency checkpoint before halt at line 624 also uses `ttl_seconds=None`; no ephemeral (non-None ttl) checkpoint calls found |
| T-09-ROUTE | MEDIUM | mitigate | 05 | CLOSED | `scripts/rl_train.py` `check_halt` lines 321-368: per-step `e_frac_with_tokens:mean` monitoring with soft 0.7 / HARD 0.5 thresholds; `protected_mask_jaccard` lines 376-419 monitors expert overlap against `output/profiling/reasoning-merged-v4/protected_expert_mask.npy`; grep: 0 matches for `train_router` or `router=` in Phase 9 scripts (router frozen per D-09-02) |
| T-09-DIVERGE | HIGH | mitigate | 05 | CLOSED | CR-04 fix in `scripts/rl_train.py` `run_training_step` lines 535-641: `forward_backward` → `_compute_kl_metrics` → `check_halt` → emergency checkpoint + return on halt; `tc.optim_step()` at line 628 only reached on safe path; `_KL_COMPUTE_FAILED_SENTINEL = 1e9` line 273 prevents silent 0.0 swallow; RSPO floor: `rspo_floored_ratio` `ratio.clamp(min=1.0)` lines 141-165 |
| T-09-RWD-REG | HIGH | mitigate | 06 | CLOSED | `.claude/skills/wp-finetune:run-rl-training/SKILL.md` Step 7: anti-hack regression gate `scripts/eval_judge.py --mode regression`; Key Rule 5: "do not auto-promote if regression gate fails"; gate wired as mandatory pre-promotion step in the skill runbook |

---

## Threat Flags from SUMMARY.md Files

All six SUMMARY.md files were inspected for `## Threat Flags` or `## Threat Surface Scan` sections.

| Plan | Section Title | Declared Flags |
|------|---------------|----------------|
| 09-01 | `## Threat Flags` | None |
| 09-02 | (no threat flags section; test scaffolding plan — no new surfaces) | None |
| 09-03 | `## Threat Surface Scan` | None — "All external surface is via existing `scripts/claude_agent.py` subprocess path" |
| 09-04 | `## Threat Surface Scan` | None — "module imports `scripts.reward_pipeline` and `scripts.rl_judge_dispatch` (both existing, no new surfaces)" |
| 09-05 | `## Threat Flags` | None — "no new network endpoints, auth paths, or schema changes beyond what is in the plan's threat model" |
| 09-06 | `## Threat Flags` | None — "Skill is a runbook document — no new network endpoints or auth paths introduced" |

No unregistered threat flags. No new attack surface appeared during implementation that
lacks a threat mapping.

---

## Controls Verifiable Only at Live Run

The following mitigations were verified at the code level; their effectiveness at runtime
depends on environment conditions that cannot be confirmed from static analysis:

1. **T-09-CRED (credential hygiene):** `ServiceClient()` reads `~/.tinker` or env. The
   audit confirms no hardcoded credentials in code. Whether `~/.tinker` is correctly
   provisioned on the Tinker cloud host is a deployment-time control.

2. **T-09-RWD-REG (regression gate):** `eval_judge.py --mode regression` is declared in
   the SKILL runbook (Step 7) with a hard gate on promotion. Confirmed present in
   SKILL.md. Whether the operator executes Step 7 before promoting is an operational
   control.

3. **T-09-INJECT (prompt injection residual):** Score clamping and N-vote median bound
   manipulation to [0,1]. Actual prompt injection attempts and their effect on model
   behavior can only be observed during live inference.

4. **T-09-SELFPREF (self-preference):** Panickssery spot-check is wired and fires every
   50 steps. Whether the logged divergence alerts prompt operator intervention is a
   runtime/operational control.

---

## Review Bugs Verified Fixed

The following critical bugs from 09-REVIEW.md were cross-verified during this audit:

| Bug ID | Description | Fix Location | Verified |
|--------|-------------|--------------|---------|
| CR-04 | `optim_step` called before halt check | `rl_train.py` `run_training_step` seam | Yes — halt check at line ~582, `optim_step` at line 628 |
| CR-05 | Z-scored values fed to `combine_judge_reward` instead of raw [0,1] | `rl_rollouts.py` `collect_rollouts` lines 566-573 | Yes — no pre-normalization; comment explicitly states "RAW [0, 1] scale" |
| CR-06 | Per-batch filter instead of per-prompt in `remove_constant_reward_groups` | `rl_rollouts.py` `_inline_remove_constant_reward_groups` | Yes — `_group_by_id` used for per-prompt grouping |
| WR-04 | `_panickssery_spot_check` inert due to `isinstance(bd, dict)` | `rl_train.py` lines 508-510 | Yes — `getattr` used |
| WR-02 | Cache key collision via string concatenation | `rl_judge_dispatch.py` line 80 | Yes — `json.dumps([...])` separator |
| WR-03 | Thread leak on subprocess timeout | `rl_judge_dispatch.py` lines 200-202 | Partial — `_SUBPROCESS_TIMEOUT_S=110` < asyncio 120s; thread exits before asyncio cancel |
