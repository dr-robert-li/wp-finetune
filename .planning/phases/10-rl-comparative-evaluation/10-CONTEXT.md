# Phase 10: RL Comparative Evaluation - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Compare the **Phase 9 RL model** (GSPO-trained, Tinker-exported) against the **v1.2 SFT baseline** (`output/merge_v4_winner`, wp-bench results in `output/04.4_wp_bench_results.json`) on **wp-bench + all 9 eval dimensions**, and produce the RLEV-02 value-add report. Confirm RL improved judge-reasoning quality (the primary RL target — judge Spearman) **without** regressing generation quality. **wp-bench is a HARD GATE** for v3.0 (MoE-Sieve). Reuses the existing `wp-finetune:run-evaluation` skill + `eval/` infra, extended with RL-specific metrics (reward convergence, router-shift stability, protected-expert retention, anti-hack pass rate).

**Out of scope (own phases):** fresh routing re-profiling on RL-policy logs for sieve selection (Phase 11 — the Phase 7 SFT-era profiles are used here only for the protected-expert retention baseline); MoE-Sieve (Phase 11); LoRA merge + pruning (Phase 13). Phase 10 EVALUATES and gates — it does not retrain. If a dimension regresses, surface to the user with suggested fixes (re-train with adjusted regularizer / reward weights); do not auto-retrain in this phase.

**Hard dependency:** blocked on Phase 9's **live Tinker RL run** completing (credential-gated, still partial in `09-HUMAN-UAT.md`). That run exports the RL checkpoint(s) (`tinker://` → HF archive) and emits the per-step RLEV-01/02 metrics this phase consumes. Phase 10 can be PLANNED now but cannot produce real comparison numbers until that run lands.

</domain>

<decisions>
## Implementation Decisions

### Regression-gate policy (D-10-01) — CI-aware per-dimension
- **D-10-01:** "No dimension regression permitted" is operationalized as **no statistically real regression** under the project's CI-aware noise-band disposition (D-09, inherited). A dimension counts as regressed only if its **bootstrap CI shows a real drop below the v1.2 baseline**, measured identically on baseline + RL candidate; within-noise dips PASS. **Judge Spearman must improve beyond noise** (the primary RL target) — a flat/within-noise judge is a soft fail to surface, not silent pass. Per-dimension (not aggregate-only) so a real gen-dim drop cannot hide behind a judge gain.
- Rationale: strict point-wise zero-regression would fail the gate on pure eval noise even when RL genuinely improved; aggregate-only would mask a real gen regression. CI-aware per-dimension is the established project gate shape.

### Checkpoint selection (D-10-02) — best-by-reward + final, head-to-head
- **D-10-02:** Export and evaluate **two** RL checkpoints — the **best-by-reward-convergence** checkpoint AND the **final-step** checkpoint — run both through the full eval, and pick the winner against the baseline. Robust to late-training divergence / reward overfit at modest extra eval cost. The selected winner is the canonical RL model handed to Phase 11.

### wp-bench hard-gate definition (D-10-03) — aggregate CI-aware + per-task floor
- **D-10-03:** The wp-bench HARD GATE passes when the RL model's **aggregate wp-bench bootstrap lower bound ≥ the v1.2 baseline aggregate point estimate** (`output/04.4_wp_bench_results.json`), **AND no individual wp-bench task catastrophically regresses** (a hard per-task floor — a real, large single-task drop fails the gate even if the aggregate clears). Balances an overall meet-or-exceed bar with catastrophe protection.

### RLEV-02 value-add bar (D-10-04) — five-part conjunctive + human sign-off
- **D-10-04:** Sign-off to v3.0 (gating MoE-Sieve) requires **ALL** of:
  1. Judge Spearman improvement beyond noise (primary RL target).
  2. wp-bench HARD GATE pass (D-10-03).
  3. Anti-hack pass rate **≥ the Phase 8 anti-hack baseline** — confirms gains are real, not reward-hacking.
  4. Protected-expert retention **≥ the Phase 7 baseline** (CI-aware vs `protected_expert_mask`).
  5. Router-shift / KL stability log shows **no routing collapse** over the run.
- A **human review checkpoint** presents the full v1.2-SFT-vs-RL comparison table (all 9 dims + wp-bench + the 5 sub-gates) before the v3.0 gate is declared pass/fail. This is a conjunctive gate, not holistic judgment — reproducible.

### Claude's Discretion
- Exact bootstrap method, N resamples, and CI level (reuse the project's existing CI-aware gate implementation in `eval/eval_gate.py` — do not invent a new one).
- The concrete numeric value of the per-task "catastrophic regression" floor (D-10-03) — derived at planning from baseline task-score spread.
- Eval-harness wiring, serving venue plumbing, report layout/format, telemetry embedding — planner/researcher territory.
- Whether the judge-recalibration +3.58 offset (D-V4-09) applies identically to the RL judge component — confirm at planning (inherited from the frozen judge).

### Folded Todos
- None folded. The two matched todos (`phase7-8-ci-aware-noiseband-gates.md`, `phase8-inherit-judge-recalibration.md`) were already folded into Phase 8 and satisfied by shipped Phase 8 work — they are not Phase-10 scope (their dispositions are *inherited* here as D-10-01's CI-aware basis, not re-implemented). See Reviewed Todos.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` § "Phase 10: RL Comparative Evaluation" — goal, 2 success criteria, skill spec. ⚠ STALENESS FLAG (mirrors Phase 9 D-09-01): the skill text says `dgx.execute("eval_toolbox", ...)` and LLM-eval via `Agent(run_in_background=true)`. The RL model is **Tinker-exported** (`tinker://` → HF archive), not DGX-trained; serving for eval is vLLM-on-DGX (the Phase 4.4 re-bench pattern). And Python-side LLM-judged dims dispatch via the `scripts/claude_agent.py` subprocess path (NOT the Anthropic API; `Agent(run_in_background=true)` is skill-orchestrator-only — the Phase 9 code review confirmed this). Planner should reconcile and note the correction.
- `.planning/REQUIREMENTS.md` — RLEV-01 (RL vs v1.2 SFT on wp-bench + 9 dims, no regression, judge Spearman primary), RLEV-02 (report: reward convergence, router-shift log, protected-expert retention, gen/judge delta, anti-hack results)

### Eval infrastructure (REUSE — extend, do not rebuild)
- `.claude/skills/wp-finetune:run-evaluation/SKILL.md` — the eval skill to extend with RL-specific metrics
- `eval/eval_gate.py` — the existing CI-aware gate (use for D-10-01/03 bootstrap bounds; do not invent a new gate)
- `eval/eval_judge.py`, `eval/eval_gen.py`, `eval/rubric_scorer.py`, `eval/rubric_definitions.py`, `eval/dim_map.json`, `eval/llm_checks.py`, `eval/output_parsers.py` — 9-dim scoring + frozen judge
- `config/wp-bench.yaml`, `wp-bench/` — the wp-bench harness

### Baseline artifacts (v1.2 SFT — the comparison target)
- `output/merge_v4_winner/` — the v1.2 SFT canonical merged model (D-10's baseline model)
- `output/04.4_wp_bench_results.json` — the v1.2 SFT baseline wp-bench scores (D-10-03 reference)
- `output/eval_reasoning_v4_winner/` — v1.2 eval results + `judge_recalibration.json` (+3.58 offset, D-V4-09)

### RL model + metrics (Phase 9 — the candidate)
- `.planning/phases/09-gspo-training/09-CONTEXT.md` — Phase 9 decisions (GSPO primary, frozen router, RLEV fields the run logs)
- `.planning/phases/09-gspo-training/09-HUMAN-UAT.md` — the pending live-run items that produce the checkpoints + metrics this phase consumes
- `scripts/rl_train.py` — emits `output/rl_checkpoints/metrics/rl_metrics.jsonl` (reward convergence, kl_sample_train, e_frac, Jaccard) + the checkpoint manifest (D-10-02 candidate selection)

### Protected experts + anti-hack baselines (Phase 7 / Phase 8)
- `.planning/phases/07-router-profiling-protected-expert-set/07-CONTEXT.md` §D-09 + `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` — protected-expert retention baseline (D-10-04 #4)
- `.planning/phases/08-reward-infrastructure/08-CONTEXT.md` + the Phase 8 anti-hack set builder/scoring — anti-hack pass-rate baseline (D-10-04 #3)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`eval/` suite + `wp-finetune:run-evaluation` skill**: the full 9-dim + wp-bench eval path already exists (used for v1.2). Phase 10 RE-RUNS it on the RL model and EXTENDS it with 4 RL-specific report sections — does not rebuild scoring.
- **`output/04.4_wp_bench_results.json`**: the baseline numbers are already on disk — no need to re-run v1.2; just re-bench the RL model and diff.
- **`eval/eval_gate.py` CI-aware gate**: the bootstrap-lower-bound-vs-bar machinery for D-10-01/03 already exists from prior phases.
- **Phase 8 anti-hack set + Phase 7 protected_expert_mask**: the two baselines D-10-04 compares against.

### Established Patterns
- CI-aware noise-band gate disposition (D-09) — all Phase 10 gates use bootstrap lower bound vs bar, measured identically baseline + candidate.
- Claude Code agents for LLM-judged dims (NOT Anthropic API); Python-side via `scripts/claude_agent.py` subprocess.
- Tinker `tinker://` → HF archive export → vLLM serve for benching (Phase 4.4 re-bench pattern), NOT DGX-local training execution.

### Integration Points
- RL checkpoint(s) export (Phase 9 manifest) → HF archive → vLLM serve → `eval/` 9-dim + wp-bench → `eval_gate.py` CI-aware diff vs v1.2 baseline → RLEV-02 report → human sign-off → v3.0 gate.

</code_context>

<specifics>
## Specific Ideas

- All four gate decisions deliberately reuse the project's existing CI-aware machinery rather than inventing new thresholds — the user consistently favors the bootstrap-lower-bound-vs-bar disposition across phases.
- The judge Spearman improvement is the PRIMARY success signal (RL's whole point was to fix the 0.57 judge bottleneck); gen quality is an anti-regression anchor, not the target.

</specifics>

<deferred>
## Deferred Ideas

- **Fresh RL-policy routing re-profiling** for sieve selection — Phase 11 (Phase 7 profiles are SFT-era; used here only for the retention baseline).
- **MoE-Sieve / merge / pruning** — Phases 11/13.
- **Auto-retrain on regression** — out of scope; Phase 10 surfaces a regression + suggested fix to the user, it does not loop back into training.

### Reviewed Todos (not folded)
- `phase7-8-ci-aware-noiseband-gates.md` — already folded into Phase 8 and satisfied; its CI-aware disposition is *inherited* as D-10-01's basis, not re-implemented. Recommend clearing from `pending/`.
- `phase8-inherit-judge-recalibration.md` — already folded into Phase 8; the +3.58 judge-recalibration offset is a Claude's-Discretion confirm item here, not new scope. Recommend clearing from `pending/`.

</deferred>

---

*Phase: 10-rl-comparative-evaluation*
*Context gathered: 2026-06-20*
