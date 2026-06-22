# Phase 9: GSPO Training - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Dual-mode RL that refines **both** generation quality (`<wp_gen>`) and judge-reasoning quality (`<wp_judge>`) on the full Qwen3-30B-A3B MoE — consuming Phase 8's composite reward pipeline (`scripts/reward_pipeline.py`) and Phase 7's `protected_expert_mask`. GSPO (sequence-level) is the primary RL objective per D-08; judge is the primary bottleneck (Spearman 0.57 vs gen 0.99+ at SFT) and receives equal-or-greater RL budget. Router-shift stabilization (GRPO-07/08) and a protected-expert routing regularizer (GRPO-06) guard MoE routing stability.

**Out of scope (own phases):** RL-vs-SFT comparative evaluation on wp-bench + 9 dims (Phase 10, RLEV-01/02); post-RL MoE-Sieve (Phase 11); merge + pruning (Phase 13). Building/validating the reward pipeline itself was Phase 8 — Phase 9 consumes it, does not rebuild it.

</domain>

<decisions>
## Implementation Decisions

### Execution platform (D-09-01) — LOCKED
- **D-09-01:** RL runs on **Tinker (cloud)**, the same venue as the Phase 4.3/4.4 SFT. The GB10 (124.6 GiB unified) provably cannot host Qwen3-30B-A3B bf16 (`output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`); RL is heavier than SFT (rollouts + reward scoring + gradient), so local execution is off the table.
- The RL loop is built on Tinker primitives: `forward_backward` / `optim_step` / `save_state` + `sampling_client.sample(...)` for rollouts. It is a **custom training loop**, not a turnkey RL recipe.
- ⚠ **ROADMAP staleness flag:** the ROADMAP "Phase 9" skill text still describes DGX-local `dgx.execute("unsloth_studio", ...)` per-epoch execution. That text predates the Tinker pivot and is WRONG for the training venue. The new `wp-finetune:run-rl-training` skill must be Tinker-native; only auxiliary scoring (deterministic PHPCS/security/VeRPO) and Claude-agent dispatch run off-Tinker. Planner should reconcile this and note the ROADMAP correction.

### Router-training scope (D-09-02) — RESEARCH-GATED
- **D-09-02:** Whether RL trains the **router gates** is deferred to research. In SFT the router was FROZEN (LoRA on experts/attention/shared only). GRPO-06 wants full-MoE RL with "gradients to router gates" + a protected-expert regularizer.
- Researcher MUST confirm, on Tinker: (a) can LoRA adapters target the router gates? (b) does Tinker expose per-step routing distributions / per-expert activation frequencies (telemetry needed for router-shift + protected-expert checks)?
- **If frozen** (likely, Tinker-native): router-shift is ~0 by construction, the `protected_expert_mask` becomes a **monitor-only** invariant, and the regularizer is a rarely-firing safety net. This **relaxes GRPO-06's router-gate clause** → must be documented as an explicit, justified deviation and router-RL deferred (the router is reshaped in Phase 13 merge/pruning regardless).
- **If trained:** router-shift stabilization (GRPO-07/08) + KL regularizer (GRPO-06) become load-bearing — the real payoff of Phases 7→9.

### RL algorithm (D-09-03) — GSPO primary, RESEARCH-GATED on feasibility
- **D-09-03:** **GSPO (sequence-level)** is primary per D-08. Researcher confirms it is expressible on Tinker primitives (sequence-level importance ratio from sampling logprobs; RSPO stop-gradient floor for router-shift). GSPO needs sequence logprobs, which Tinker sampling provides.
- **GRPO + Pro-GRPO (expand-then-prune)** is the **fallback only** if GSPO proves infeasible/unstable on Tinker — NOT a planned side-by-side comparison (Phase 10 is the comparative-eval phase; don't spend RL budget comparing here). Note Pro-GRPO's expand-then-prune touches expert structure and interacts with D-09-02.

### Gen vs judge budget (D-09-04)
- **D-09-04:** **Interleaved, judge-weighted.** A single RL run where each batch mixes `<wp_gen>` and `<wp_judge>` samples with judge ≥ gen (start ~60/40 judge/gen; honors GRPO-05 "judge ≥ gen budget").
- Both pathways stay live every batch: gen's verifiable reward (PHPCS+security+VeRPO) acts as an **anti-regression anchor** while judge improves. This deliberately avoids the catastrophic-forgetting risk of two sequential stages — the gen-regression failure already seen once (`.planning/debug/reasoning-merge-gen-regression.md`).

### Judge reward definition (D-09-05) — Claude-agent rubric + MANDATORY anti-regression guards
- **D-09-05:** Judge reward = **score-reasoning consistency** (a separately spawned **Claude Code evaluator agent**, NO Anthropic API — same dispatch as Phase 8 anti-hack scoring; rates 0–1 whether the judge's written critique actually justifies its numeric score: contradictions, unsupported claims, missed real issues) **+ fix correctness** (deterministic PHPCS/security on the critique-then-fix corrected code).
- **This is the highest-risk reward component in Phase 9.** Using an LLM/Claude agent as an RL reward has *documented* training-regression failure modes (see canonical refs — reward hacking, self-preference bias, reward-noise → GRPO advantage collapse). It is a KNOWN hazard, not a hypothetical. Therefore the following guards are **mandatory**, not optional:
  1. **Fix-correctness is the anchor.** The deterministic half carries primary weight; the Claude-consistency reward is **capped** (cannot be the sole/dominant judge-side signal).
  2. **Determinism:** Claude scorer runs at **temperature 0 with N-vote** (median/majority) to suppress reward noise that would weaken GRPO group-relative advantages.
  3. **Rubric, not free-form:** detailed scoring rubric (reduces hackability per the literature).
  4. **Regression gate:** Phase 9 is gated on the **existing Phase 8 anti-hack regression set** + CI-aware disposition (D-09) — judge-scores-up-while-quality-down is caught, not trusted.
- **RESEARCH-GATED open items the researcher MUST resolve before this is trusted** (these are the genuinely project-unknown parts, vs. the known phenomenon):
  - **(R1) Self-preference quantification:** the Qwen judge-reasoning was SFT'd partly on **Claude-distilled CoT** (`data/phase3_cot/`, deep_judge_cot). Run a Panickssery-style check — does the Claude scorer reward Claude-stylistic reasoning over *correct* reasoning? Quantify before trusting.
  - **(R2) Reward-noise budget:** measure Claude-scorer variance on repeated identical inputs; confirm SNR keeps GRPO advantages above the collapse regime (consider Noise-corrected/Dr.GRPO if not).
  - **(R3) Latency / non-stationarity:** Claude agents between rollout and gradient step add wall-clock + an external dependency that can drift run-to-run (model updates mid-training). Engineering + reproducibility risk.

### Router-shift stabilization policy (D-09-06)
- **D-09-06:** **Two-tier automated response** (active only if D-09-02 resolves to a trained router; monitor-only otherwise):
  - Protected-expert deactivation below the Phase 7 baseline frequency → **auto-inject the routing KL regularizer + re-run the epoch** (automated recovery, training keeps moving).
  - Hard router-shift-ratio breach (routing-collapse early warning) → **halt + present metrics to a human**.
- Thresholds inherit the **CI-aware disposition (D-09)**: bootstrap lower bound clears the bar, measured identically on baseline + candidate — absorbs transient noise instead of crying wolf. Per-step shift logging required (GRPO-08).

### Claude's Discretion
- Exact judge/gen interleave ratio (60/40 start) — planner/research may tune; constraint is judge ≥ gen.
- Concrete numeric thresholds (router-shift, protected-expert frequency, consistency-reward cap, N for N-vote) — derived at planning/research time; the *dispositions* above are locked.

### Folded Todos
- None folded into Phase 9. The two pending todos (`phase8-inherit-judge-recalibration.md`, `phase7-8-ci-aware-noiseband-gates.md`) were already folded into **Phase 8** CONTEXT (D-08-02 and the gate-hygiene decision) and are satisfied by shipped Phase 8 work — they are Phase-8 obligations, not Phase-9 scope. Flag for cleanup.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` § "Phase 9: GSPO Training" — goal, 4 success criteria, skill spec (⚠ DGX-execution text is stale per D-09-01)
- `.planning/REQUIREMENTS.md` — GRPO-05 (dual-mode RL, judge ≥ gen budget), GRPO-06 (full-MoE RL + protected-expert regularizer, GSPO primary / GRPO fallback), GRPO-07 (router-shift stabilization / RSPO), GRPO-08 (per-step shift monitoring + auto-halt); RLEV-01/02 (Phase 10 eval fields the RL run must log)

### Execution platform (D-09-01 — Tinker)
- `.planning/TINKER-PIVOT-RESEARCH.md` — why Tinker, primitives (`forward_backward`/`optim_step`/`sampling_client`), `Qwen3-30B-A3B` supported, weight export for downstream phases
- `output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md` — proof GB10 cannot host 30B bf16 (why not DGX-local)

### Reward pipeline + signals (Phase 8 — consume, do not rebuild)
- `.planning/phases/08-reward-infrastructure/08-CONTEXT.md` — full Phase 8 decisions (70/30 split, security terminal gate, MO-GRPO norm, VeRPO, +3.58 recal application order, anti-hack set, D-08-04 per-sample `(scalar, breakdown)` + group entry point)
- `scripts/reward_pipeline.py` — the composite reward entry point Phase 9 calls
- `eval/rubric_scorer.py`, `eval/llm_checks.py`, `eval/eval_judge.py`, `eval/eval_gate.py`, `eval/dim_map.json`, `eval/rubric_definitions.py` — deterministic signal sources + frozen judge
- `output/eval_reasoning_v4_winner/judge_recalibration.json` — +3.58 offset (D-V4-09) the judge component inherits

### Protected experts (Phase 7 — consume)
- `.planning/phases/07-router-profiling-protected-expert-set/07-CONTEXT.md` §D-09 — CI-aware gate disposition
- Phase 7 `protected_expert_mask.json` / `.npy` — the mask monitored/regularized in D-09-02/D-09-06 (07-02-SUMMARY for exact path)

### LLM-as-reward training-regression literature (informs D-09-05 guards + R1–R3)
- [LLM Evaluators Recognize and Favor Their Own Generations — Panickssery et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7f1f0218e45f5414c79c0679633e47bc-Paper-Conference.pdf) — self-preference bias; basis for R1
- [Quantifying and Mitigating Self-Preference Bias of LLM Judges (arXiv 2604.22891)](https://arxiv.org/html/2604.22891v2)
- [Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients (arXiv 2510.18924)](https://arxiv.org/abs/2510.18924) — noisy reward → advantage collapse; basis for R2
- [Gradient Regularization Prevents Reward Hacking in RLHF/RLVR (arXiv 2602.18037)](https://arxiv.org/pdf/2602.18037)
- [Ask a Strong LLM Judge when Your Reward Model is Uncertain (arXiv 2510.20369)](https://arxiv.org/abs/2510.20369)

### Skill patterns to extend (NEW skill `wp-finetune:run-rl-training` created at planning)
- `docs/wp-finetune:run-training.md` — per-epoch loop pattern (⚠ DGX/Unsloth; the RL loop is Tinker-native — reuse structure, not the venue)
- `wp-finetune:adaptive-planner` (between-epoch thermal/power config adjust), `wp-finetune:observe-training` (telemetry team), `wp-finetune:review-telemetry`
- `wp-finetune:run-data-pipeline` SKILL.md — the `Agent(run_in_background=true)` Claude Code agent dispatch pattern reused for D-09-05 consistency scoring (no Anthropic API)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`scripts/reward_pipeline.py`** (Phase 8): per-sample `(scalar, breakdown_dict)` + accepts the rollout group (MO-GRPO within-group norm, VeRPO per-check difficulty). Phase 9 calls this for gen rewards directly; the breakdown feeds RLEV-02 logging.
- **Phase 8 anti-hack set + Claude-agent scoring dispatch** (`build_antihack_set.py`, background-agent scoring): reuse the agent-dispatch pattern for D-09-05 consistency scoring AND as the Phase 9 regression gate.
- **Phase 7 `protected_expert_mask`**: the monitored/regularized set for D-09-02/06.
- **Tinker SDK** (`tinker`, `tinker-cookbook`): training-loop primitives + sampling client.

### Established Patterns
- CI-aware noise-band gate disposition (D-09) — all Phase 9 stability/regression gates use bootstrap lower bound vs bar, measured identically baseline+candidate.
- Claude Code agents for LLM work (NOT Anthropic API) — Phase-wide constraint; applies to the consistency evaluator.
- `dgx_toolbox` per-epoch loop + adaptive-planner + observe-training telemetry — structural template, but venue is Tinker not DGX.

### Integration Points
- Reward: Phase 9 rollouts → `reward_pipeline.py` (gen) + Claude-agent consistency + deterministic fix-correctness (judge) → composite scalar → GSPO/GRPO advantage.
- Routing: per-step routing telemetry (if available, D-09-02) → router-shift ratio (RSPO floor) + protected-expert retention check → D-09-06 two-tier response.
- Output: RL checkpoint exported from Tinker (`tinker://` → HF archive) for Phase 10 eval.

</code_context>

<specifics>
## Specific Ideas

- The user explicitly flagged the Claude-agent reward-regression risk and asked it be researched before locking — hence D-09-05's mandatory guards + R1–R3. Do NOT treat the Claude consistency reward as a free, trustworthy signal; it is the component most likely to silently regress training.
- ROADMAP Phase 9 skill text assumes DGX-local execution — known stale; RL loop is Tinker-native (D-09-01).

</specifics>

<deferred>
## Deferred Ideas

- **Router-RL** — if D-09-02 resolves to a frozen router, training the router gates is deferred (the router is reshaped in Phase 13 merge/pruning regardless).
- **GSPO vs GRPO empirical comparison** — out of scope here; Phase 10 (RLEV-01/02) is the comparative-eval phase.
- **Noise-corrected / Dr.GRPO adoption** — only if R2 shows the Claude-scorer reward noise pushes GRPO advantages into the collapse regime.

### Reviewed Todos (not folded)
- `phase8-inherit-judge-recalibration.md`, `phase7-8-ci-aware-noiseband-gates.md` — already folded into Phase 8 and satisfied by shipped Phase 8 work; not Phase-9 scope. Recommend clearing from `pending/`.

</deferred>

---

*Phase: 9-gspo-training*
*Context gathered: 2026-06-20*
