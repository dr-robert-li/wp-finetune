# Eval-Harness Prose Compatibility — Findings + Open Design (Phase 4.4 W1-W6)

**Date:** 2026-05-30
**Status:** BLOCKER for REVL-01/02 cascade launch. Council acked Option B + two-GT;
NEW dimension-taxonomy mismatch discovered — modifies parser spec, needs resolution.

## Blocker 1 — JSON-only parser vs prose model output (council ACKED)

`eval/eval_judge.py:133-210` `parse_judge_response` is JSON-only (4 strategies, all
`json.loads`). v1.2 reasoning fine-tune emits dimensional PROSE
(`WPCS Compliance: score 9/10 — explanation`). Both sides of REVL-01 break:
model output → None; GT (assistant target) prose → `_extract_gt_from_assistant` None
→ rubric_scorer fallback. `eval_gen` `_extract_php_code` at risk from `<think></think>`
prefix + reasoning prose (same class as smoke php_lint contamination).

### Council binding direction
- **Option B**: explicit `output_format: json|prose|auto` flag; `auto` default
  (JSON-first, prose fallback) for backward compat. Shared module `eval/output_parsers.py`.
- **Two-GT design**:
  - Canonical GT for selection/reward (Phase-7 MoE-Sieve, Phase-10/11 GRPO) =
    **rubric_scorer** (authoritative, independent oracle). → REVL-01A (HARD).
  - Teacher-target GT (parsed assistant target JSON/prose) = diagnostic only.
    → REVL-01B (SOFT).
- **No silent fallback**: record per-row provenance —
  `model_parse_format: json|prose|fail`, `teacher_parse_format: json|prose|missing|fail`,
  `gt_source_quality: rubric_scorer[_calibrated]`,
  `gt_source_teacher: assistant_target_judge_output|assistant_target_prose|missing`.
- **Parser-coverage preflight** (HARD gate) on reasoning val before REVL-01 runs.
- `eval_gen`: add `strip_think_blocks()` + harden `_extract_php_code()`.

## Blocker 2 — dimension taxonomy mismatch (NEW, needs resolution)

Reasoning prose emits a DIFFERENT 9-dim rubric than eval_judge expects.

| Reasoning prose dim (40/40 in val) | eval_judge internal key | Maps? |
|------------------------------------|-------------------------|-------|
| WPCS Compliance       | D1_wpcs       | ✓ |
| Security              | D2_security   | ✓ |
| SQL Safety            | D3_sql        | ✓ |
| Performance           | D4_perf       | ✓ |
| WP API Usage          | D5_wp_api     | ✓ |
| I18n                  | D6_i18n       | ✓ |
| Accessibility         | D7_a11y       | ✓ |
| Code Quality          | D9_structure (code_structure) | ≈ loose |
| Dependency Integrity  | — (no eval equivalent) | ✗ |
| — (no reasoning equiv)| D8_errors (error_handling) | ✗ |

7/9 clean. 2 diverge: reasoning {Code Quality, Dependency Integrity} vs
eval {error_handling D8, code_structure D9}.

### Open design question (for council)
REVL-01 per-dim Spearman options:
1. **7-dim clean**: map the 7 unambiguous dims; drop Code Quality + Dependency
   Integrity (reasoning) and D8 + D9 (eval) from Spearman. Cleanest, loses 2 dims.
2. **9-dim with mapping**: Code Quality→D9_structure; Dependency Integrity→D8_errors
   (semantic stretch). Full coverage, questionable validity on the 2 mapped dims.
3. **Overall-only HARD + 7-dim per-dim SOFT**: REVL-01A overall Spearman (model overall
   vs rubric overall) is the HARD gate; per-dim reported on the 7 clean dims as SOFT.

Recommendation pending council: likely (3) — overall Spearman is the robust HARD
signal; per-dim on 7 clean dims is diagnostic. Avoids forcing invalid 2-dim mappings.

## Sequencing (regardless of resolution)
1. Resolve Blocker-2 dim-map decision.
2. Build `eval/output_parsers.py` (strip_think, parse_judge_scores(format), prose
   dim extractor with agreed map, extract_php_code) + unit tests.
3. Wire into eval_judge (two-GT + provenance) + eval_gen (think-strip).
4. Parser-coverage preflight on reasoning val (HARD) — Spearman computes on real
   scores, PHPCS runs on think-stripped code.
5. Build W2-02 orchestrator (D-03 baseline re-eval → REVL-02 → REVL-01A/B → REVL-04).
6. Launch W1-W6 cascade.

## Blocker 3 — prose has no `overall_score` (HARD-gate derivation, NEEDS council)

REVL-01A HARD = Spearman(model_overall vs rubric_scorer_overall). But reasoning
PROSE output emits 8 per-dimension `score N/10` lines and NO overall score
(CtF-JSON output DOES carry overall_score). So for prose rows, `model_overall`
must be DERIVED from the per-dim scores.

Options for prose model_overall:
1. **Weighted mean** of model's parsed dim scores via `DIMENSION_WEIGHTS`
   (symmetric with how rubric_scorer computes its overall) — RECOMMENDED.
2. Simple mean of the 6 clean dims.
3. Skip prose rows from overall Spearman (only CtF-json rows contribute) — loses
   most of the signal (85 prose vs 36 json).

Recommendation: (1) weighted mean — matches rubric aggregation, uses all emitted
dims, keeps full sample. Provenance: tag `model_overall_source: prose_weighted_mean
| json_overall`. CtF-json rows use their emitted overall_score directly.

### Validation already done (parsers, Blocker 1+2)
- eval/output_parsers.py: model output 5/5 parse (4 prose + 1 json); teacher GT
  111/121 (108 json trailing-block + 3 prose); gen code extracts clean. 12 tests.
- Committed 0b14735. dim_map.json + parsers are the validated core.

## Downstream impact (why surfacing, not patching)
eval_judge/eval_gen feed Phase-4 triage, Phase-7 MoE-Sieve eval, Phase-10/11 GRPO
reward. If GRPO reward grounds in parse_judge_response, the format-detection +
dimension-map decisions here propagate to the reward signal. Council confirmed:
GRPO reward MUST ground in rubric_scorer (independent oracle), NOT assistant targets.
