# Seed Scorer Agreement (Phase 0 step 2)

Scored seeds: **145**  (all FAIL-band examples; PASS anchors come in Phase 1)

## Per-dimension agreement (rubric_scorer vs human dim scores)

| Dim | n | Spearman | p | Pearson | Human mean | Rubric mean |
|-----|---|----------|---|---------|------------|-------------|
| D1_wpcs | 22 | +0.087 | 0.702 | +0.085 | 2.36 | 9.99 |
| D2_security | 15 | +0.175 | 0.532 | +0.158 | 2.27 | 7.58 |
| D3_sql | 12 | +0.996 | 0.000 | +1.000 | 3.00 | 3.32 |
| D4_perf | 8 | +0.286 | 0.493 | +0.037 | 1.88 | 9.88 |
| D5_wp_api | 31 | -0.345 | 0.057 | -0.303 | 2.65 | 9.96 |
| D6_i18n | 16 | +nan | nan | +nan | 2.00 | 9.22 |
| D7_a11y | 12 | +nan | nan | +nan | 2.00 | 9.96 |
| D8_errors | 0 | n/a | n/a | n/a | n/a | n/a |
| D9_structure | 15 | -0.071 | 0.800 | -0.071 | 2.93 | 9.99 |

## Rubric overall_0_100 distribution
- n: 145  min: 64.6  max: 100.0  mean: 95.8  stdev: 9.7

## Tooling state
Full 4-tool active: PHPCS (WordPress + WordPressVIPMinimum + Security) + PHPStan + regex.
LLM-assisted checks (18 of them per rubric §F.5) still deferred — those capture
the semantic defects humans flag (e.g. unbounded WP_Query, missing capability
checks); weak per-dim agreement on D4/D5/D9 reflects that gap, not a tool failure.