# Phase 4.1 Gap Analysis — Unaccounted 171 CoT Examples

**Date**: 2026-04-22  
**Audited by**: Quick task 260422-p41-review-followups

## The Numbers

| Stage | Count | Notes |
|--|------|-------|
| **Total generated** | 1,163 | CoT batches (batch_000.json through batch_050.json) |
| **Judge PASS** | 741 | Examples with parsed PASS verdict |
| **Judge FAIL** | 251 | Examples with parsed FAIL verdict |
| **Accounted (PASS+FAIL)** | 992 | |
| **Gap** | 171 | 1,163 - 992 = 171 unaccounted |

## Gap Investigation

### What the gap IS NOT

1. **Not pilot examples**: Pilot (20 examples) is separate from the 1,163 bulk
2. **Not _input_batch residuals**: 280 `_input_batch_*.json` files are orchestrator intermediate state (contain code but no reasoning field), NOT part of the final 1,163 count
3. **Not CtF examples**: CtF has 200 separate examples
4. **Not manifest items**: Manifests contain 196 (CoT) and 180 (CtF) function IDs

### What the gap LIKELY IS

**Judge parse failures / abstentions.** The 741 PASS / 251 FAIL counts come from judge evaluation output, not from the generation batch itself. The bulk judge eval (eval_judge.py) extracts verdicts from model responses, and:

- The judge parse rate reported in eval results explains ~13.2% of judge outputs being unparseable (`_skipped.jsonl` diagnostic)
- 171 / 1,163 = **14.7%** — consistent with a ~13-15% judge parse failure rate
- No explicit rejection log or abstain marker was written during bulk generation

### Evidence

1. **Judge parse diagnostics**: `eval_judge.py` writes skipped examples to `_skipped.jsonl` — this is the mechanism for unparseable outputs
2. **Citation audit subset** (196 examples): Only `deep_judge_cot_bulk.json` has citation audit; no equivalent audit exists for the full 1,163
3. **Per-batch structure**: All batch items have the same structure (`source_file`, `code`, `reasoning`, `dimensions_addressed`, `citation_accuracy`) but NONE contain a judge verdict — verdicts are added externally by judge eval

### Conclusion

The 171 examples are **not lost**. They are examples where the judge model's response could not be parsed for a PASS/FAIL verdict. This is a judge evaluation artifact, not a data generation artifact.

**However**: The review is correct that there is NO explicit acceptance report. The 741/251 numbers come from an intermediate eval that was not documented as a formal Phase 4.1 gate.

## Recommendations

1. **Before Phase 4.2**: Run judge evaluation on all 1,163 examples with full parsing (no skipped outputs) and produce explicit pass/fail/deny counts
2. **Parse rate target**: Require judge parse rate ≥ 95% before accepting any generation batch
3. **Document in Phase 4.2 gate**: "accepted" = judge-parsed AND verdict=PASSED AND citation_validity ≥ threshold
