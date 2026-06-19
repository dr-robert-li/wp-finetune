# Phase 8: Reward Infrastructure - Context

**Gathered:** 2026-06-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Build and independently validate a **composite reward pipeline** (`scripts/reward_pipeline.py` + pytest) *before* any RL training. The pipeline maps a single generation (and its rollout group) to a scalar reward: **70% verifiable signals** (PHPCS pass rate, security scan, WordPress-standards checks) + **30% frozen `wp_judge` score**, with a **security hard gate** (fail → reward 0), **MO-GRPO within-group variance normalization**, and **VeRPO difficulty-weighted partial credit**. Also construct + validate the **anti-hack eval set** (D-11) used as an RL regression check.

**Out of scope (own phases):** running GSPO/GRPO RL (Phase 9), router-shift stabilization / protected-expert routing regularizer (Phase 9, consumes Phase 7 mask), RL-vs-SFT evaluation (Phase 10).

</domain>

<decisions>
## Implementation Decisions

### Signal sourcing (D-08-01)
- **D-08-01:** Reuse the existing `eval/` harness as the single source of truth for deterministic signals — import `eval/rubric_scorer.py` + `eval/llm_checks.py` (PHPCS / security / WP-standards) and `eval/eval_judge.py` for the frozen `wp_judge` invocation. Reward signals MUST match eval signals (no drift).
- Inherit the PHPCS standard and the local vLLM judge endpoint already configured in `eval/` — do not re-pick a standard or stand up a new endpoint.

### Judge recalibration application (D-08-02 / inherits D-V4-09)
- **D-08-02:** Apply the **+3.58** `score_offset` to the **raw** `wp_judge` overall score, **clip to the valid score range**, **then** MO-GRPO normalize. Order is offset → clip → normalize.
- Treat the offset as a **bare point correction**. The artifact CI [1.24, 6.09] / SE 1.25 is documented context only — it does NOT become a weight discount in v1. (Offset is rank-invariant per artifact; this is the simplest correct application.)
- Source of truth for the constant: `output/eval_reasoning_v4_winner/judge_recalibration.json` — read at pipeline load, do not hardcode the number in two places.

### Anti-hack eval set construction (D-08-03 / D-11)
- **D-08-03:** Build adversarial cases by **perturbing real gen+judge outputs** along the three D-11 hack axes — verbose padding, template-critique collapse, self-preference swap. Claude Code agents (`Agent(run_in_background=true)`, per `wp-finetune:run-data-pipeline`) score candidate cases during construction. No external Anthropic API.
- **Pass criterion is CI-aware (D-09):** an adversarial case "passes the anti-hack check" when its reward is **CI-aware below the clean-baseline reward** (bootstrap lower bound clears the bar), not a bare absolute cap.

### Reward output contract (D-08-04)
- **D-08-04:** `reward_pipeline` exposes a **per-sample** entry point that returns **`(scalar, breakdown_dict)`** — breakdown carries each signal (PHPCS, security, VeRPO, judge) at both pre- and post-normalization, for RLEV-02 logging (reward convergence curves, anti-hack results).
- The call **accepts the rollout group** so MO-GRPO within-group variance and VeRPO per-check difficulty (pass rate across group samples) can be computed. Apply an **epsilon floor on zero-variance groups** (avoid divide-by-zero / single-signal blow-up).

### Validation gate hygiene (inherits D-09 / D-V4-10)
- All Phase-8 validation/acceptance gates (anti-hack "all below threshold", integration test on the 50 held-out gen+judge cases) use **CI-aware noise-band dispositions** — report bootstrap CIs, require the lower bound to clear the bar, measured identically on baseline + candidate.

### Resolved from research open questions (2026-06-19)
- **D-08-05 (security-failure definition, GRPO-02):** "Security failure" = a **CRITICAL_FLOOR_RULE for `D2_security` fires** (research Option C). Reward=0 triggers on the rubric's own critical-security classification, NOT on any minor SEC-N* sniff and NOT on a bare `D2_security < 8.0` score cut.
- **D-08-06 (VeRPO scope, GRPO-04):** VeRPO difficulty-weighting applies to the **WP-standards subset only** (`D1_wpcs` + WP-specific sniffs), per the literal GRPO-04 / SC4 text. The other dimensions are covered by the 30% judge component — VeRPO is NOT applied across all 9 dimensions.
- **D-08-07 (judge parse-failure fallback):** When `judge_score_single` returns `None` (parse failure), impute the judge component from the **rollout-group mean**; record the failure in `breakdown_dict`; flag/raise if the per-batch parse-failure rate exceeds **10%**.
- **Accepted research defaults (planner may proceed without re-asking):** 70% verifiable block split **35% PHPCS-overall / 35% VeRPO**, each independently MO-GRPO normalized; anti-hack set size **15 cases per axis = 45 total** (+ the SC2 secure-fail case → fits the 50-case integration set).

### Folded Todos
- **`phase8-inherit-judge-recalibration.md`** (pending) — folded as **D-08-02**: Phase 8 reward MUST consume `judge_recalibration.json` (+3.58, D-V4-09) as a hard input to the 30% `wp_judge` component.
- **`phase7-8-ci-aware-noiseband-gates.md`** (pending) — folded as the **Validation gate hygiene** decision above (D-09 / D-V4-10): CI-aware noise-band gates for all Phase-8 acceptance checks.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` § "Phase 8: Reward Infrastructure" — goal, 5 success criteria, skill/execution constraints
- `.planning/REQUIREMENTS.md` — GRPO-01 (70/30 composite), GRPO-02 (security hard gate), GRPO-03 (MO-GRPO norm), GRPO-04 (VeRPO partial credit); RLEV-02 (RL report fields the breakdown must feed)

### Judge recalibration (D-V4-09 — MUST consume)
- `output/eval_reasoning_v4_winner/judge_recalibration.json` — `score_offset=3.58`, `ci_95=[1.24,6.09]`, `rank_invariant=true`; explicitly tags the Phase-8 reward 30% `wp_judge` component
- `.planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-RECALIBRATION-SUMMARY.md` — derivation (paired mean, n=118, bf16-merge artifact)
- `.planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-D-V4-JUDGE-MECHANISM-DIAGNOSIS.md` — why the offset is significant-but-rank-invariant
- `.planning/todos/pending/phase8-inherit-judge-recalibration.md` — the inheritance obligation

### CI-aware gate disposition (D-09 / D-V4-10)
- `.planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-D-V4-10-WAIVER.md` — origin of CI-aware noise-band gates
- `.planning/phases/07-router-profiling-protected-expert-set/07-CONTEXT.md` §D-09 — codified CI-aware disposition (bootstrap lower bound clears the bar)
- `.planning/todos/pending/phase7-8-ci-aware-noiseband-gates.md` — the gate-hygiene obligation

### Signal sources to reuse
- `eval/rubric_scorer.py`, `eval/llm_checks.py` — deterministic PHPCS / security / WP-standards scoring
- `eval/eval_judge.py`, `eval/eval_gate.py` — frozen `wp_judge` invocation + gating patterns
- `eval/dim_map.json`, `eval/rubric_definitions.py` — 9-dimension rubric schema the judge emits

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/rubric_scorer.py` + `eval/llm_checks.py`: deterministic PHPCS/security/WP-standards scoring — the 70% verifiable signals wrap these directly.
- `eval/eval_judge.py`: existing frozen `wp_judge` invocation against the local vLLM endpoint — the 30% judge component reuses it (then applies the +3.58 offset).
- `output/eval_reasoning_v4_winner/judge_recalibration.json`: the recalibration constant, loaded at runtime.

### Established Patterns
- Per-example JSONL logging (EVAL-06) already exists in `eval_gen.py`/`eval_judge.py` — the reward breakdown_dict mirrors this shape for RLEV-02.
- `Agent(run_in_background=true)` data-pipeline pattern (`wp-finetune:run-data-pipeline`) is the sanctioned way to use Claude Code agents for anti-hack set construction — NO direct Anthropic API.

### Integration Points
- Phase 9 GSPO trainer consumes `reward_pipeline`'s per-sample `(scalar, breakdown)` + group API.
- The frozen `wp_judge` model = the promoted v1.2 canonical checkpoint (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`).

</code_context>

<specifics>
## Specific Ideas

- Reward = 0 floor on security failure must be tested with an explicit "secure-failing but otherwise high-quality" case (SC2).
- Anti-hack three axes are fixed by D-11: verbose padding, template-critique collapse, self-preference bias.
- Recalibration constant must be read from the JSON artifact, never duplicated as a literal across modules.

</specifics>

<deferred>
## Deferred Ideas

- **CI-as-weight-discount for the judge component** — considered (Recalib option B); deferred. If RL shows the judge signal is noisy/over-trusted, revisit shrinking the 30% weight by SE. Not in v1.
- **Synthesize-fresh / hand-curated anti-hack sets** — considered; deferred in favor of perturb-real. Could augment coverage in a later hardening pass.
- **Router-shift stabilization, protected-expert routing regularizer, dual-mode RL rewards (judge score-reasoning consistency, fix correctness)** — Phase 9 (GRPO-05/06/07). Out of Phase 8 scope.

</deferred>

---

*Phase: 8-reward-infrastructure*
*Context gathered: 2026-06-19*
