# Seed Scorer Agreement (Phase 0 step 2)

Scored seeds: **145**  (all FAIL-band examples; PASS anchors come in Phase 1)

## Per-dimension agreement (rubric_scorer vs human dim scores)

| Dim | n | Spearman | p | Pearson | Human mean | Rubric mean |
|-----|---|----------|---|---------|------------|-------------|
| D1_wpcs | 22 | +0.087 | 0.702 | +0.085 | 2.36 | 9.99 |
| D2_security | 15 | +0.000 | 1.000 | -0.081 | 2.27 | 3.92 |
| D3_sql | 12 | +0.996 | 0.000 | +1.000 | 3.00 | 3.32 |
| D4_perf | 8 | +0.286 | 0.493 | +0.488 | 1.88 | 9.87 |
| D5_wp_api | 31 | -0.144 | 0.439 | -0.121 | 2.65 | 9.88 |
| D6_i18n | 16 | +nan | nan | +nan | 2.00 | 9.35 |
| D7_a11y | 12 | +nan | nan | +nan | 2.00 | 9.82 |
| D8_errors | 0 | n/a | n/a | n/a | n/a | n/a |
| D9_structure | 15 | -0.071 | 0.800 | -0.071 | 2.93 | 9.99 |

## Rubric overall_0_100 distribution
- n: 145  min: 53.3  max: 100.0  mean: 93.7  stdev: 11.9

## Tooling state
Full 5-tool active: PHPCS (WordPress + WordPressVIPMinimum + Security) + PHPStan + regex + LLM-assisted (41 binary YES/NO checks per rubric §F.5).