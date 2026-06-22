---
phase: 09-gspo-training
verified: 2026-06-20T20:08:00+10:00
status: human_needed
score: 4/4 must-haves verified (loop wiring level)
overrides_applied: 1
overrides:
  - must_have: "Protected experts monitored via routing regularizer / KL penalty enforcing router shift"
    reason: >
      Router gates are FROZEN on Tinker (LoraConfig has no train_router arg — D-09-02).
      Routing regularizer is implemented as monitor-only: Jaccard similarity against
      protected_expert_mask.npy logged every N steps; e_frac halt thresholds (soft 0.7,
      hard 0.5) act as KL-budget enforcement. The goal says "monitored via routing
      regularizer / KL penalty" — monitor-only Jaccard + global KL halt satisfies the
      monitoring intent under Tinker constraints. Deviation surfaced in 09-02-PLAN.md
      notes, 09-05-PLAN.md notes, SKILL.md Deviations section (D-09-02), and
      09-VALIDATION.md.
    accepted_by: phase-instructions
    accepted_at: 2026-06-20T00:00:00+10:00
human_verification:
  - test: "Live training run executes end-to-end with cloud credentials"
    expected: >
      `python scripts/rl_train.py --total-steps 10` reaches collect_rollouts, dispatches
      gen and judge rollouts, computes advantages, runs build_loss_step (GSPO path),
      logs RLEV fields (kl_sample_train_v1, e_frac_with_tokens_mean, reward_breakdown),
      writes checkpoint at step 10, exits cleanly.
    why_human: >
      Requires live Tinker cloud credentials. Also blocked by the judge-client gap below:
      main() passes `args` to collect_rollouts, which reads args.judge_client /
      args.judge_model / args.n_votes (rl_rollouts.py lines 525-526, 562), but
      _parse_args() never registers those three CLI arguments (verified: no
      add_argument for --judge-client, --judge-model, --n-votes in rl_train.py lines
      649-719). Live run AttributeErrors before step 0. This is the documented
      out-of-scope follow-up noted in phase instructions; it does not block the loop
      wiring goal but must be fixed before a real training run.
  - test: "Reward breakdown in RLEV JSONL contains non-zero judge signals"
    expected: >
      reward_breakdown field in metrics/rl_metrics_*.jsonl shows non-zero
      judge_consistency and fix_correctness values after a real run, confirming
      dual-mode reward flow reaches the log sink.
    why_human: >
      Requires a live Tinker run with judge dispatch active. Dry-run uses a mock
      client and does not invoke rl_judge_dispatch.py.
---

# Phase 9: GSPO Training Verification Report

**Phase Goal:** Dual-mode RL refines BOTH generation quality (`<wp_gen>`) and judge-reasoning quality (`<wp_judge>`) on the full Qwen3-30B-A3B MoE, with router-shift stabilization. GSPO (sequence-level) is the primary RL objective; judge gets judge >= gen RL budget. Gen rewards = PHPCS+security+VeRPO; judge rewards = score-reasoning consistency (Claude evaluator agent) + fix correctness. Protected experts from Phase 7 monitored via routing regularizer.

**Verified:** 2026-06-20T20:08:00+10:00
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GSPO sequence-level loss (forward_backward_custom + RSPO floor) is the DEFAULT primary objective | VERIFIED | `build_loss_step(use_gspo=True)` default; calls `tc.forward_backward_custom(data, loss_fn)` with `rspo_floored_ratio` closure (`seq_ratio.clamp(min=1.0)`). `_parse_args` sets `use_gspo=True` via `set_defaults`. GRPO fallback requires explicit `--grpo-fallback`. `_dry_run` exercises GSPO path. |
| 2 | Dual-mode rollouts: judge budget >= gen budget (n_judge = round(batch_size * 0.6)) | VERIFIED | `JUDGE_RATIO = 0.6` at rl_rollouts.py module level; `sample_interleaved_prompts` stamps `_origin` ("gen"/"judge"); `collect_rollouts` routes by `_origin`. Integration test `test_both_gen_and_judge_rollouts_reach_advantages` asserts both gen-* and judge-* group_ids survive to advantages. |
| 3 | KL halt check runs BEFORE optim_step (CR-04 ordering) | VERIFIED | `run_training_step` ordering confirmed: `build_loss_step` -> `_compute_kl_metrics` -> `check_halt` -> if halt_reason: `_save_checkpoint` + `return True` (no optim_step); else `tc.optim_step()`. `_KL_COMPUTE_FAILED_SENTINEL = 1e9` ensures failure trips hard halt. Integration test `test_hard_kl_halts_before_optim_step` (kl_v1=0.9) asserts halted=True AND optim_step NOT called. |
| 4 | Protected expert mask loaded and monitored via Jaccard; D-09-02 deviation surfaced | VERIFIED (override) | `protected_mask_jaccard` loads `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` (shape [48,128]). Dry-run output shows `jaccard=0.0020` (mask loaded successfully). Frozen router D-09-02 deviation surfaced in 09-02-PLAN notes, 09-05-PLAN notes, SKILL.md Deviations section, and 09-VALIDATION.md. |

**Score:** 4/4 truths verified (1 via documented override for D-09-02 deviation)

---

### Seven Critical Bug Fixes (09-REVIEW.md)

All 7 BLOCKER-class bugs found in code review are FIXED. Verification evidence:

| Bug | Description | Fix Verification |
|-----|-------------|-----------------|
| CR-01 | Wrong sampling client (save_weights_for_sampler returns checkpoint ref, not client with .sample()) | `main()` line 858 calls `tc.save_weights_and_get_sampling_client()`. Integration test `test_main_wires_sampling_client_and_loads_pools` asserts this method called. `_FakeSamplingClient.sample_calls` non-empty. |
| CR-02 | Empty pools (gen_pool/judge_pool hardcoded []) | `main()` lines 842-843: `gen_pool = load_rl_prompts("gen")`, `judge_pool = load_rl_prompts("judge")`. Pools confirmed non-empty: gen=68, judge=482. Integration test asserts `load_rl_prompts` called for both. |
| CR-03 | Tag filtering drops all rollouts (routing by non-existent "tag" field) | `sample_interleaved_prompts` stamps `_origin` key via `_stamp_origin()`. `collect_rollouts` routes by `item.get("_origin") == "gen"/"judge"`. No "tag" field routing. |
| CR-04 | optim_step called before halt check (commits divergent update) | Ordering fixed: build_loss_step -> _compute_kl_metrics -> check_halt -> (if halt) emergency checkpoint + return True -> (safe path only) tc.optim_step(). Integration test confirms. |
| CR-05 | Z-scores fed into combine_judge_reward (double-normalization) | `collect_rollouts` passes raw fix_correctness values to `combine_judge_reward()`. MO-GRPO normalization happens downstream in advantage centering only. |
| CR-06 | Constant filter on entire batch (not per-prompt) | `_inline_remove_constant_reward_groups` groups by `group_id` via `_group_by_id()`, checks std within each prompt group separately. |
| CR-07 | Unconditional ImportError raises (bypasses cookbook, errors at import) | Unconditional raise removed. `compute_rollout_advantages` uses inline implementations as explicit primary path with clear docstring. No try/except with broken fallback. |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/rl_train.py` | Main RL training loop (880 lines) | VERIFIED | Exists, 880 lines, substantive. All CR fixes present. GSPO primary, KL halt ordering correct, sampling client correct. |
| `scripts/rl_rollouts.py` | Rollout + reward collection | VERIFIED | Exists, 781 lines, substantive. _origin routing (CR-03), per-group constant filter (CR-06), raw reward to combine_judge_reward (CR-05). |
| `scripts/rl_judge_dispatch.py` | Claude consistency scorer | VERIFIED | Exists. No Anthropic API import; subprocess path via claude_agent.py. `judge_consistency_weight = 0.3` asserted <= 0.5 at import. |
| `scripts/tinker_rl_data.py` | Prompt pool loader | VERIFIED | Exists. `load_rl_prompts("gen")` and `load_rl_prompts("judge")` both return non-empty lists (68, 482 items). |
| `scripts/build_rl_prompts.py` | Prompt corpus assembler | VERIFIED | Exists. Builds wp_gen_train.jsonl and wp_judge_train.jsonl from reasoning_dataset. |
| `data/rl_prompts/wp_gen_train.jsonl` | Gen prompt pool (OpenAI chat schema) | VERIFIED | Exists, 11.0K, 68 prompts. |
| `data/rl_prompts/wp_judge_train.jsonl` | Judge prompt pool (OpenAI chat schema) | VERIFIED | Exists, 518.9K, 482 prompts. |
| `data/rl_prompts/PROVENANCE.md` | Data provenance documentation | VERIFIED | Exists, 2.7K. |
| `tests/test_rl_train.py` | Unit contract tests (8 named stubs) | VERIFIED | 234 lines. TestRLTrainUnit class with 8 named tests using pytest.importorskip. |
| `tests/test_rl_train_integration.py` | Integration tests (6 tests) | VERIFIED | 475 lines, 6 tests, all 6 PASSED in isolated run. |
| `.claude/skills/wp-finetune:run-rl-training/SKILL.md` | Tinker-native RL skill file | VERIFIED | 319 lines. Deviations section documents D-09-02 and D-09-03. Zero DGX/unsloth/docker references. WR-05 fix: uses `m.get('checkpoints', [])`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `main()` | `load_rl_prompts` | import + call | WIRED | Lines 840-843: gen_pool and judge_pool populated from tinker_rl_data |
| `main()` | `tc.save_weights_and_get_sampling_client()` | Tinker SDK | WIRED | Line 858: correct API for sampling client (CR-01 fix) |
| `main()` | `collect_rollouts()` | call | WIRED | Passes gen_pool, judge_pool, sampling_client, args |
| `collect_rollouts()` | `_origin` routing | `_stamp_origin()` | WIRED | Routes gen vs judge by _origin key, not non-existent "tag" |
| `collect_rollouts()` | `combine_judge_reward()` | raw values | WIRED | Passes fix_correctness and consistency on [0,1] scale (CR-05 fix) |
| `run_training_step()` | `tc.optim_step()` | conditional | WIRED | Only reached after check_halt returns None (CR-04 fix) |
| `build_loss_step()` | `tc.forward_backward_custom()` | GSPO default | WIRED | Called when use_gspo=True (default); rspo_floored_ratio closure |
| `_compute_kl_metrics()` | `_KL_COMPUTE_FAILED_SENTINEL` | return on failure | WIRED | Returns 1e9 on exception, ensuring hard KL halt triggers |
| `protected_mask_jaccard()` | `output/profiling/.../protected_expert_mask.npy` | np.load | WIRED | Dry-run shows jaccard=0.0020, confirming mask loaded |
| CLI args | `judge_client/judge_model/n_votes` in rollouts | _parse_args | NOT WIRED | `_parse_args` has no --judge-client/--judge-model/--n-votes. rl_rollouts.py lines 525-526, 562 reference args.judge_client/args.judge_model/args.n_votes. Live run AttributeErrors. Documented out-of-scope follow-up. |

---

### Data-Flow Trace (Level 4)

| Component | Data Variable | Source | Produces Real Data | Status |
|-----------|--------------|--------|-------------------|--------|
| gen prompt pool | gen_pool | data/rl_prompts/wp_gen_train.jsonl via load_rl_prompts("gen") | Yes — 68 prompts from reasoning_dataset | FLOWING |
| judge prompt pool | judge_pool | data/rl_prompts/wp_judge_train.jsonl via load_rl_prompts("judge") | Yes — 482 prompts | FLOWING |
| GSPO loss | fb_out | tc.forward_backward_custom(data, loss_fn) | Yes — Tinker SDK forward pass (mock in dry-run) | FLOWING (dry-run confirmed) |
| KL metrics | kl_v1, e_frac | _compute_kl_metrics(fb_out, data) | Yes — parses fb_out or returns sentinel | FLOWING |
| RLEV metrics | jsonl row | _log_step() | Yes — writes kl_sample_train_v1, e_frac_with_tokens_mean, reward_breakdown | FLOWING (integration test confirms) |
| judge rewards | consistency_scores | rl_judge_dispatch.py score_judge_consistency_batch | Real on live run; dry-run bypasses | DISCONNECTED (dry-run path only) |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Dry-run exits 0 | `python scripts/rl_train.py --dry-run --total-steps 1` | exit 0 | PASS |
| Full test suite | `python -m pytest tests/ -q --tb=no` | 478 passed | PASS |
| Integration tests | `python -m pytest tests/test_rl_train_integration.py -v` | 6 passed | PASS |
| Prompt pools loadable | `load_rl_prompts("gen"), load_rl_prompts("judge")` | 68, 482 items | PASS |
| No train_router in rl_train.py | `grep -c 'train_router'` | 0 matches | PASS |
| No phase7_profiling ref in rl_train.py | `grep -c 'data/phase7_profiling'` | 0 matches | PASS |
| No hardcoded credentials | `grep -c 'sk-\|api_key='` | 0 matches | PASS |
| GSPO sampling client API used | `grep -c 'save_weights_and_get_sampling_client'` | 2 matches | PASS |
| KL halt gating present | `grep -c 'check_halt'` | 7 matches | PASS |

---

### Probe Execution

No probe scripts found at conventional path `scripts/*/tests/probe-*.sh`. Phase declared dry-run as the verification gate. Dry-run exit 0 confirmed above.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GRPO-05 | 09-01, 09-03, 09-04, 09-06 | Dual-mode prompt corpus assembled with proper origin tagging; judge consistency scorer integrated | SATISFIED | wp_gen_train.jsonl (68), wp_judge_train.jsonl (482), PROVENANCE.md exist. rl_judge_dispatch.py exports score_judge_consistency_batch. _origin routing via _stamp_origin(). |
| GRPO-06 | 09-02, 09-05 | Router-shift stabilization with protected expert monitoring | SATISFIED (D-09-02 override) | Router frozen on Tinker (D-09-02). protected_mask_jaccard() monitors Jaccard vs Phase 7 mask. e_frac halt thresholds (soft 0.7, hard 0.5). KL halt (soft 0.1, hard 0.3). Deviation explicitly documented. |
| GRPO-07 | 09-02, 09-05 | GSPO primary loss (forward_backward_custom + RSPO floor) with GRPO fallback | SATISFIED | build_loss_step defaults use_gspo=True, calls tc.forward_backward_custom with rspo_floored_ratio closure. --grpo-fallback flag exists. D-09-03 locked and documented. |
| GRPO-08 | 09-02, 09-05 | KL/MoE halt before optim_step; RLEV metrics logged; checkpoints persisted | SATISFIED | CR-04 ordering verified. _KL_COMPUTE_FAILED_SENTINEL=1e9. check_halt() covers both KL and e_frac thresholds. _log_step writes RLEV fields. _save_checkpoint writes output/checkpoints/. Integration test confirms. |

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| rl_rollouts.py docstring | `compute_rollout_advantages` docstring says "delegating to cookbook" but code is 100% inline with no try/import of tinker_cookbook | WARNING | Documentation misleads; implementation is correct and tested. Docstring should be updated to say "inline mirroring cookbook semantics." Not a stub — math is real. |
| rl_rollouts.py (WR-02) | Cache key collision in score_with_cache (rl_judge_dispatch.py): uses raw completion text as key, collides if two different prompts have identical completions | WARNING | Low probability in production; could produce incorrect judge scores on cache hits from different contexts. Not fixed in this phase. |
| rl_rollouts.py (WR-03) | Thread leak on timeout in rl_judge_dispatch.py: Future thread not cancelled when timeout fires | WARNING | Background threads accumulate over long runs. Not fixed in this phase. |
| rl_train.py (IN-01) | Vacuous security test: `test_security_reward_nonzero` checks a file the test creates, not a real WordPress file | INFO | Test asserts reward pipeline wired but doesn't validate real security detection. Non-blocking. |
| cli gap | `_parse_args` has no --judge-client, --judge-model, --n-votes; rl_rollouts.py lines 525-526, 562 reference args.judge_client/args.judge_model/args.n_votes | BLOCKER for live run | Live main() AttributeErrors before step 0. Documented out-of-scope follow-up; does not block loop wiring goal. Requires fix before any real training run. |

**Debt marker gate:** No TBD/FIXME/XXX markers found in phase-modified files without tracking references.

---

### Human Verification Required

#### 1. Live training run end-to-end execution

**Test:** Fix judge-client CLI args (add --judge-client, --judge-model, --n-votes to _parse_args), then run `python scripts/rl_train.py --total-steps 10` with Tinker cloud credentials.

**Expected:** Script reaches collect_rollouts without AttributeError, dispatches gen and judge rollouts, computes advantages for both modes, runs GSPO build_loss_step, checks KL/e_frac halt, calls optim_step, logs RLEV fields to JSONL, writes checkpoint at step 10, exits cleanly.

**Why human:** Requires Tinker cloud credentials. Also requires fixing the judge-client CLI gap (three missing --add_argument calls in _parse_args) before the real entrypoint works. Gated by cloud cost; intentionally manual per phase design.

#### 2. Dual-mode RLEV reward_breakdown confirms both gen and judge signals

**Test:** After a real run, inspect metrics/rl_metrics_*.jsonl for reward_breakdown contents.

**Expected:** reward_breakdown contains non-zero values for phpcs_score, security_score, verpo_score (gen signals) AND fix_correctness, judge_consistency (judge signals) across multiple steps.

**Why human:** Requires live run with judge dispatch active. Dry-run mock client never invokes rl_judge_dispatch.py.

---

### Gaps Summary

No gaps block the phase goal at the loop wiring level. All 4 success criteria are VERIFIED in the implementation:

- GSPO sequence-level loss with RSPO floor is correctly implemented and is the default.
- Dual-mode rollout interleaving (judge >= gen budget) is correctly wired via _origin routing.
- KL/MoE halt-before-optim_step ordering is correct and tested with hard KL values.
- Protected expert monitoring is implemented (D-09-02 override: monitor-only due to frozen Tinker router, explicitly surfaced in plans and SKILL.md).

One pre-live-run fix required (not a goal-level gap):
- **Judge-client CLI gap**: Add `--judge-client`, `--judge-model`, `--n-votes` to `_parse_args()` in `scripts/rl_train.py`. Three `parser.add_argument` calls needed. This is the documented out-of-scope follow-up from the phase instructions.

Three warning-level items carried forward (not fixed in this phase):
- WR-02: Cache key collision in rl_judge_dispatch.py
- WR-03: Thread leak on timeout in rl_judge_dispatch.py  
- Cookbook delegation docstring misleads (implementation is inline, correctly so)

---

_Verified: 2026-06-20T20:08:00+10:00_
_Verifier: Claude (gsd-verifier)_
