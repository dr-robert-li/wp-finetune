---
scope: Training data quality assessment
reviewers: [gemini, claude]
reviewed_at: 2026-04-11T10:15:00Z
dataset: 12,160 exported (93,974 merged)
---

# Cross-AI Review: Training Data Quality

## Consensus Verdict: WEAK — Fix before training

Both reviewers independently concluded the dataset has a strong foundation (real code, good Phase 4 reasoning) undermined by structural problems that will dominate the training signal.

## Agreed Critical Issues

### 1. Judge Training Duplication (75% near-duplicate) — BLOCKER
**Both reviewers rate this as the #1 structural problem.**

- 8,525 of 11,400 judge responses are near-duplicates (only 2,875 unique)
- The top template `{wpcs: 90, security: 100, ...}` appears 764 times
- **Impact:** Model memorizes modal score vectors instead of learning to evaluate code
- **Root cause:** Agents generated scores via heuristic templates, not per-function analysis
- **Fix:** Regenerate judge training with real per-function Claude Code agent evaluation

### 2. Weak Gen Instructions — HIGH
**Both reviewers flag `<wp_gen> Write a WordPress function: {name}` as insufficient.**

- Trains name→body memorization, not instruction-following
- At inference time, intent-based prompts ("Write a function that securely updates user meta...") will fail because the model never saw that form
- **Fix:** Enrich instructions from docstrings, function context, and WordPress API patterns

### 3. Score Dead Zone (65-79) — HIGH
**Both reviewers identify the bimodal score distribution as a calibration failure.**

- Passed clusters at 90-95 (stdev 3.9), failed clusters at 60-65
- The 14-point gap (65-79) is where the most important discrimination happens
- Model can't learn where "okay" ends and "good" begins
- **Fix:** Generate 500-1000 intermediate-quality examples filling the gap

### 4. Hardcoded N/A Dimensions — MEDIUM
- `i18n=7` and `accessibility=7` for ~90% of examples
- Reduces effective dimensionality from 6 to 4
- Model learns these as constants, not assessable qualities

## Agreed Strengths

- **Phase 1 real code corpus is excellent** — 82K functions from 204 repos, diverse, properly judged
- **Phase 4 reasoning data is the best in the pipeline** — 180 CtF + 380 deep CoT with 99.2% critique-fix alignment, 1.6% hallucination rate, real per-function analysis
- **Security patterns well-represented** — nonce verification, capability checks, output escaping, prepared statements all present in training data

## Divergent Views

- **Gemini rates ADEQUATE/WEAK**, Claude rates **WEAK**
- Gemini focuses on intent-based instruction enrichment as the #1 fix
- Claude focuses on excluding templated examples as the #1 fix
- Both are valid — they attack different failure modes (gen vs judge)

## Recommended Fix Priority (Consensus)

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | **Regenerate judge training** with real per-function agent analysis (not templates) | Fixes duplication + hardcoded N/A + thin explanations | Medium |
| 2 | **Enrich gen instructions** from docstrings/function context | Fixes memorization, enables instruction-following | Medium |
| 3 | **Fill score dead zone** (65-79) with intermediate examples | Fixes calibration boundary | Medium |
| 4 | **Scale Phase 4 CoT** from 380 to 2,000+ | Highest-quality signal in the pipeline | Medium |
| 5 | Fix sample weighting field matching | Restores contrastive upweighting | Low |

## Bottom Line

> "A 3,000-example dataset of real agent-generated assessments will produce a better model than 12,160 examples where 70% are templates." — Claude review

> "You are fine-tuning a very expensive, very smart MoE model to be a simple autocomplete engine." — Gemini review

**DO NOT PROCEED to training until fixes 1-3 are applied.**
