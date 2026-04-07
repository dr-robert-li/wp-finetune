STATUS: HUMAN_OVERRIDE — 30_70 ACCEPTED

## Triage Decision

### Hard Gate Results

| Ratio | PHPCS Rate | Spearman | Security Rate | PHPCS Gate | Spearman Gate | Security Gate |
|-------|------------|----------|---------------|------------|---------------|---------------|
| 30_70 | 1.0000 | 0.5698 | 1.0000 | PASS | FAIL | PASS |
| 40_60 | 0.9860 | 0.0000 | 1.0000 | PASS | FAIL | PASS |
| 50_50 | 0.9720 | 0.0000 | 1.0000 | PASS | FAIL | PASS |
| 60_40 | 0.9880 | 0.0000 | 1.0000 | PASS | FAIL | PASS |

### Survivors

Ratios proceeding to Phase 7: NONE

### Eliminated Ratios

- **30_70**: Spearman gate failed: 0.5698 not strictly > 0.85 (strict > 0.85 required)
- **40_60**: Spearman gate failed: 0.0000 not strictly > 0.85 (strict > 0.85 required)
- **50_50**: Spearman gate failed: 0.0000 not strictly > 0.85 (strict > 0.85 required)
- **60_40**: Spearman gate failed: 0.0000 not strictly > 0.85 (strict > 0.85 required)

### wp-bench Scores

wp-bench was skipped (--skip-wpbench). Triage based on static eval gates only. wp-bench differentiation deferred.

### E_eff Summary (Informational)

- **30_70**: {'mean_eeff_total': 69.83565524495702, 'max_eeff_total': 98.25061035593937}
- **40_60**: {'mean_eeff_total': 69.54687653035816, 'max_eeff_total': 97.50789011501928}
- **50_50**: {'mean_eeff_total': 68.98821013160263, 'max_eeff_total': 96.91285692503224}
- **60_40**: {'mean_eeff_total': 68.32307569541484, 'max_eeff_total': 96.29058871036455}
- **70_30**: {'mean_eeff_total': 67.54526181655979, 'max_eeff_total': 95.02210313488297}

### NO_SURVIVORS: Recommendation

All ratios failed hard gates. Consider:
1. Re-examine training data quality
2. Investigate specific failure dimensions
3. Lower gate thresholds if domain warrants

---

## Human Override — 2026-04-06

**Decision**: Accept 30_70 as the triage winner despite Spearman gate failure.

**Rationale**:
- 30_70 is the only ratio with a non-zero Spearman correlation (0.5698). The remaining three ratios scored exactly 0.0000, which signals a systematic evaluation issue (likely insufficient test diversity) rather than a genuine model quality problem.
- 30_70 passes both the PHPCS gate (1.0000 — perfect) and security gate (1.0000 — perfect). The sole failing gate is Spearman < 0.85.
- 30_70 has the highest E_eff mean (69.84) of all ratios evaluated, confirming it produces the best overall output quality.
- The strict Spearman > 0.85 threshold was set for a balanced eval set; the current eval set appears too narrow to produce meaningful Spearman variance. The gate will not be removed — it remains for v1.2 Phase 4.4 (which uses a richer, human-annotated test set).
- wp-bench was skipped in this triage run (--skip-wpbench). wp-bench differentiation is deferred to Phase 4.4 where it becomes a hard gate.

**Winner**: 30_70

**Winning Adapter**: `adapters/qwen3-30b-wp-30_70`

**Override Authority**: Human operator (Robert Li), 2026-04-06

**Downstream Impact**:
- Phase 4.1 (Reasoning Data Generation) may now begin using the 30_70 adapter.
- Phase 4.4 must re-validate 30_70 against wp-bench as a hard gate before adapter merge.
- The Spearman gate threshold will be revisited in Phase 4.4 with the human-annotated seed test set.
