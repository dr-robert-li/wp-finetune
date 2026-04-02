---
phase: 04
reviewers: [gemini, codex]
reviewed_at: 2026-04-03
plans_reviewed: [04-01-PLAN.md, 04-02-PLAN.md, 04-03-PLAN.md]
---

# Cross-AI Plan Review — Phase 4

## Gemini Review

The plan for Phase 4 is technically sound, surgically targeted, and rigorously aligned with the MoE-Sieve research methodology and hardware constraints of the DGX Spark. It correctly prioritizes a fast, gradient-free profiling pass on the base model to gate further training (D-05), while implementing a sequential evaluation pipeline for existing adapters to manage the 128GB VRAM limit (D-06). The separation of the E_eff profiling logic, triage automation, and human-in-the-loop verification provides a robust safety net before moving into the selective training of Phase 7.

### Strengths
- Correctly identifies the `Qwen3MoeTopKRouter` hook target (`outputs[2]`) as the source for routing indices
- Adheres strictly to sequential adapter evaluation via vLLM `--lora-modules` (only viable path for 30B on 128GB)
- Includes fallback path for vLLM LoRA loading (merging adapters first) to mitigate `modules_to_save` risk
- Plan 01 mandates GPU-free unit tests for E_eff and triage logic
- Explicitly addresses GATE-02 elimination logic preventing "winner's curse"

### Concerns
- **MEDIUM**: Token-type attribution may be diluted if 10% subsample includes truncated sequences — "other" category could inflate
- **LOW**: wp-bench Docker build latency (15-30 min) could block evaluation loop if it fails
- **LOW**: E_eff trend detection via simple linear comparison may be noise-sensitive at 10% subsample

### Suggestions
- Stratify or verify balanced `<wp_gen>`/`<wp_judge>` token counts in subsample
- Log standard deviation of E_eff across layers (bottleneck detection for Phase 8)
- Explicitly log vLLM `/v1/models` check result for debugging

### Risk Assessment: LOW
**Verdict: APPROVED**

---

## Codex Review

### Plan 01 — MEDIUM risk
Well-structured and testable, but threshold semantics are ambiguous (`>` vs `>=`), token-type tagging underspecified for edge cases (padding, truncation, packed samples), and zero-count E_eff handling may distort trend detection. Dual scoring model (eval_gen overall vs triage overall) is a confusion source.

### Plan 02 — HIGH risk
Operationally fragile. Integration mismatch risk with eval_gate.py per-ratio interface. No idempotency strategy for re-runs. LoRA fallback doesn't specify checkpoint location/cleanup/disk impact. GPU memory reclamation between profiling and vLLM not defined. No timeout/retry for eval scripts. wp-bench setup failure has no graceful degraded path beyond --skip-wpbench.

### Plan 03 — MEDIUM risk
Appropriate human gate, but underspecified for edge outcomes: no "all ratios fail" contingency, human override doesn't preserve both automated and final verdicts, assumes wp-bench results exist when Plan 02 allows skipping.

### Cross-Plan Concerns
- **HIGH**: Threshold semantics not normalized across plans
- **HIGH**: Operational reliability for long-running DGX/container execution underdesigned
- **MEDIUM**: Upstream/downstream interfaces implied more than specified
- **MEDIUM**: Partial-run and rerun behavior not clearly defined

### Cross-Plan Suggestions
- Freeze metric semantics and scoring formulas in one shared constants module
- Add artifact validation and resumability before running expensive operations
- Make each major step independently rerunnable
- Document "no survivors" and "wp-bench skipped" outcomes explicitly

### Overall Risk: MEDIUM-HIGH

---

## Consensus Summary

### Agreed Strengths
- **Research grounding**: Both reviewers praised the precise use of MoE-Sieve research (router hook target, E_eff formula, Jaccard stability)
- **Hardware awareness**: Sequential serving correctly respects 128GB VRAM constraint
- **Triage-not-winner design**: Both affirm the Phase 4 → Phase 7 handoff architecture
- **GPU-free unit tests**: Both highlight Plan 01's testability without DGX access
- **LoRA fallback**: Both approve the merge-adapter contingency for modules_to_save

### Agreed Concerns
1. **Token-type attribution edge cases** (Gemini: MEDIUM, Codex: HIGH) — truncated sequences, padding, missing task tokens could dilute profiling signal
2. **Operational reliability** (Gemini: LOW, Codex: HIGH) — wp-bench setup failure, long-running job fragility, no idempotency
3. **Threshold ambiguity** (Codex: HIGH) — `>` vs `>=` for hard gates, "5pp behind" semantics need normalization

### Divergent Views
- **Overall risk**: Gemini rates LOW, Codex rates MEDIUM-HIGH. Gemini focuses on conceptual soundness; Codex focuses on operational execution fragility. Both perspectives are valid — the plans are well-designed but need more defensive execution engineering.
- **E_eff zero-count handling**: Only Codex flagged this. Gemini accepted the approach. Worth a targeted fix.
