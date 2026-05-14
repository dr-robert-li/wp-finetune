# v1 vs v2 calibration model diff

v1 = pre-pivot model (trained on 527-row dataset, FAIL N=47).  
v2 = post-schema-normalization model (trained on 580-row dataset, FAIL N=100).  
Both scored against the v2 holdout (65 rows; v1's original holdout was 39 rows).

## 1. Holdout gate metrics

| Head | Metric | v1 | v2 | Δ | v2 gate |
|---|---|---:|---:|---:|:---:|
| Verdict | accuracy | 0.8615 | 0.8769 | ↑ +0.0154 | ≥ 0.85 → PASS |
| Verdict | AUC | 0.9811 | 0.9800 | -0.0011 | — |
| Overall | pearson | 0.7388 | 0.7659 | ↑ +0.0271 | ≥ 0.75 → PASS |
| Overall | spearman | 0.6718 | 0.6466 | ↓ -0.0252 | (informational) |
| Overall | MAE | 19.4331 | 16.5637 | -2.8694 | — |

## 2. Per-row prediction agreement (v1 vs v2 on v2 holdout)

- Verdict agreement: 98.5% (64/65)
- Overall score mean |Δ|: 8.02
- Overall score max |Δ|: 36.10
- Total disagreement rows (verdict mismatch OR |score Δ| > 10): 25

### Disagreement sample (up to 30)

| row_id | source | subtlety | gt_verdict | gt_overall | v1 verdict | v2 verdict | v1 score | v2 score |
|---|---|---|---|---:|---|---|---:|---:|
| anchor::sierotki::iWorks_Orphans_Integration_Advanced_Custom_Fields::__construct | pass_anchor | clear-cut | PASS | 95.0 | PASS | PASS | 93.7 | 67.7 |
| seed::human::human_062241e0 | human | boundary | FAIL | 35.0 | PASS | FAIL | 58.9 | 43.3 |
| seed::human::human_62803ec5 | human | boundary | FAIL | 25.0 | FAIL | FAIL | 3.5 | 18.8 |
| seed::human::human_d6d4b402 | human | boundary | FAIL | 26.7 | FAIL | FAIL | 3.5 | 18.8 |
| seed::human::human_c6d3a022 | human | boundary | FAIL | 40.0 | FAIL | FAIL | -6.0 | 8.0 |
| seed::human::human_ae3dbe2a | human | boundary | FAIL | 10.0 | FAIL | FAIL | 0.8 | 12.4 |
| seed::human::human_b4c54cab | human | boundary | FAIL | 15.0 | FAIL | FAIL | 3.5 | 18.9 |
| seed::human::human_53f910a2 | human | boundary | FAIL | 15.0 | FAIL | FAIL | 3.5 | 18.9 |
| seed::human::human_dfbddb50 | human | boundary | FAIL | 25.0 | FAIL | FAIL | 3.5 | 18.9 |
| seed::ugc_boundary::ugc_boundary_001 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -1.8 | 13.8 |
| seed::ugc_boundary::ugc_boundary_003 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | 0.4 | 14.0 |
| seed::ugc_boundary::ugc_boundary_004 | ugc_boundary | boundary | FAIL | 20.0 | FAIL | FAIL | -2.3 | 13.8 |
| seed::ugc_boundary::ugc_boundary_005 | ugc_boundary | boundary | FAIL | 0.0 | PASS | PASS | 91.3 | 81.0 |
| seed::ugc_boundary::ugc_boundary_006 | ugc_boundary | boundary | FAIL | 20.0 | FAIL | FAIL | -2.2 | 10.8 |
| seed::ugc_boundary::ugc_boundary_007 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -3.5 | 14.3 |
| seed::ugc_boundary::ugc_boundary_008 | ugc_boundary | boundary | FAIL | 20.0 | FAIL | FAIL | -6.1 | 8.0 |
| seed::ugc_boundary::ugc_boundary_009 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -2.2 | 10.8 |
| seed::ugc_boundary::ugc_boundary_013 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | 49.5 | 36.7 |
| seed::ugc_boundary::ugc_boundary_014 | ugc_boundary | boundary | FAIL | 20.0 | FAIL | FAIL | -2.4 | 12.7 |
| seed::ugc_boundary::ugc_boundary_016 | ugc_boundary | boundary | FAIL | 20.0 | PASS | PASS | 78.6 | 63.2 |
| seed::ugc_boundary::ugc_boundary_018 | ugc_boundary | boundary | FAIL | 20.0 | PASS | PASS | 74.3 | 59.0 |
| seed::ugc_boundary::ugc_boundary_017 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -1.8 | 34.3 |
| seed::ugc_boundary::ugc_boundary_019 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -7.5 | 10.7 |
| seed::ugc_boundary::ugc_boundary_022 | ugc_boundary | boundary | FAIL | 20.0 | FAIL | FAIL | -3.2 | 7.6 |
| seed::ugc_boundary::ugc_boundary_023 | ugc_boundary | boundary | FAIL | 0.0 | FAIL | FAIL | -1.8 | 11.5 |

## 3. Verdict classifier feature importance

- Features that appeared in v2 (zero in v1, nonzero in v2): 1
- Features that vanished in v2 (nonzero in v1, zero in v2): 2

### Top 20 features by absolute shift

| feature | v1 imp | v2 imp | Δ | flag |
|---|---:|---:|---:|---|
| SEC-N02 | 24.3500 | 0.0000 | -24.3500 | vanished |
| WAPI-P08 | 15.6707 | 29.2019 | +13.5312 |  |
| WPCS-P06 | 28.9349 | 20.8262 | -8.1087 |  |
| score::D8_errors | 1.5209 | 9.4007 | +7.8798 |  |
| WPCS-P01 | 23.2390 | 30.4186 | +7.1796 |  |
| WAPI-N12 | 25.6780 | 32.1130 | +6.4350 |  |
| STR-P02 | 3.8068 | 9.5857 | +5.7788 |  |
| score::D1_wpcs | 7.4346 | 2.1681 | -5.2665 |  |
| score::D5_wp_api | 2.7274 | 7.9482 | +5.2208 |  |
| ERR-P09 | 4.5114 | 9.1045 | +4.5931 |  |
| score::D4_perf | 15.2242 | 19.5198 | +4.2955 |  |
| score::D2_security | 8.2420 | 12.0284 | +3.7864 |  |
| WPCS-N02 | 9.7991 | 6.0962 | -3.7029 |  |
| score::D7_a11y | 0.0000 | 2.4946 | +2.4946 | appeared |
| PERF-P09 | 1.6179 | 0.0000 | -1.6179 | vanished |
| score::D9_structure | 70.5445 | 71.3044 | +0.7598 |  |
| SEC-N04 | 6.9952 | 7.5169 | +0.5217 |  |
| I18N-P01 | 5.9502 | 5.4440 | -0.5062 |  |
| WPCS-P09 | 2.8457 | 3.3320 | +0.4863 |  |

### Top 20 v1

- `score::D9_structure`  70.5445
- `WPCS-P06`  28.9349
- `WAPI-N12`  25.6780
- `SEC-N02`  24.3500
- `WPCS-P01`  23.2390
- `WAPI-P08`  15.6707
- `score::D4_perf`  15.2242
- `WPCS-N02`  9.7991
- `score::D2_security`  8.2420
- `score::D1_wpcs`  7.4346
- `SEC-N04`  6.9952
- `I18N-P01`  5.9502
- `ERR-P09`  4.5114
- `STR-P02`  3.8068
- `WPCS-P09`  2.8457
- `score::D5_wp_api`  2.7274
- `PERF-P09`  1.6179
- `score::D8_errors`  1.5209

### Top 20 v2

- `score::D9_structure`  71.3044
- `WAPI-N12`  32.1130
- `WPCS-P01`  30.4186
- `WAPI-P08`  29.2019
- `WPCS-P06`  20.8262
- `score::D4_perf`  19.5198
- `score::D2_security`  12.0284
- `STR-P02`  9.5857
- `score::D8_errors`  9.4007
- `ERR-P09`  9.1045
- `score::D5_wp_api`  7.9482
- `SEC-N04`  7.5169
- `WPCS-N02`  6.0962
- `I18N-P01`  5.4440
- `WPCS-P09`  3.3320
- `score::D7_a11y`  2.4946
- `score::D1_wpcs`  2.1681

## 3. Overall regressor feature importance

- Features that appeared in v2 (zero in v1, nonzero in v2): 10
- Features that vanished in v2 (nonzero in v1, zero in v2): 8

### Top 20 features by absolute shift

| feature | v1 imp | v2 imp | Δ | flag |
|---|---:|---:|---:|---|
| WAPI-N12 | 2920.4922 | 14677.2119 | +11756.7197 |  |
| STR-P02 | 5515.0884 | 16400.5000 | +10885.4116 |  |
| STR-N03 | 0.0000 | 10327.7559 | +10327.7559 | appeared |
| WAPI-P08 | 806.4765 | 10639.4922 | +9833.0157 |  |
| score::D9_structure | 43681.3320 | 50158.4219 | +6477.0898 |  |
| PERF-P09 | 0.0000 | 5896.5664 | +5896.5664 | appeared |
| score::D2_security | 735.9820 | 3824.3994 | +3088.4174 |  |
| score::D4_perf | 995.5705 | 3582.0679 | +2586.4974 |  |
| WPCS-P06 | 525.2622 | 2938.4299 | +2413.1677 |  |
| WPCS-P01 | 2205.9353 | 4415.0068 | +2209.0715 |  |
| score::D8_errors | 1010.0096 | 2702.5513 | +1692.5416 |  |
| SQL-N16 | 0.0000 | 1097.0546 | +1097.0546 | appeared |
| STR-N12 | 0.0000 | 970.0911 | +970.0911 | appeared |
| SEC-N04 | 638.9397 | 1385.0762 | +746.1365 |  |
| ERR-N07 | 0.0000 | 703.1039 | +703.1039 | appeared |
| score::D5_wp_api | 639.2468 | 1302.3318 | +663.0850 |  |
| ERR-P09 | 341.6246 | 961.3008 | +619.6761 |  |
| A11Y-P05 | 0.0000 | 606.5471 | +606.5471 | appeared |
| ERR-N11 | 0.0000 | 523.5713 | +523.5713 | appeared |
| WPCS-N07 | 0.0000 | 515.9758 | +515.9758 | appeared |

### Top 20 v1

- `score::D9_structure`  43681.3320
- `STR-P02`  5515.0884
- `WAPI-N12`  2920.4922
- `WPCS-P01`  2205.9353
- `SEC-N02`  1334.6898
- `score::D8_errors`  1010.0096
- `score::D4_perf`  995.5705
- `WAPI-P08`  806.4765
- `score::D2_security`  735.9820
- `score::D5_wp_api`  639.2468
- `SEC-N04`  638.9397
- `WPCS-P06`  525.2622
- `score::D1_wpcs`  350.0787
- `ERR-P09`  341.6246
- `WPCS-P09`  338.1639
- `SEC-N03`  249.0915
- `WPCS-N02`  227.5516
- `WPCS-P11`  209.9825
- `I18N-N13`  209.2963
- `WPCS-P07`  176.2292

### Top 20 v2

- `score::D9_structure`  50158.4219
- `STR-P02`  16400.5000
- `WAPI-N12`  14677.2119
- `WAPI-P08`  10639.4922
- `STR-N03`  10327.7559
- `PERF-P09`  5896.5664
- `WPCS-P01`  4415.0068
- `score::D2_security`  3824.3994
- `score::D4_perf`  3582.0679
- `WPCS-P06`  2938.4299
- `score::D8_errors`  2702.5513
- `SEC-N04`  1385.0762
- `score::D5_wp_api`  1302.3318
- `SQL-N16`  1097.0546
- `SEC-N02`  976.1895
- `STR-N12`  970.0911
- `ERR-P09`  961.3008
- `ERR-N07`  703.1039
- `WPCS-N02`  684.5187
- `score::D1_wpcs`  671.2886

## 4. Verdict

**v2 gates PASS.** Safe to ship as the calibration model for Phase 1c.

v1→v2 agreement on extended holdout: 98.5% verdict, mean |Δ| score 8.02.  
- ≥ 95% verdict agreement: v1 stands; v2 just tightens. Phase 1b pilot results valid under either.
- 85–95%: flag disagreement set above for human review before treating Phase 1b as final.
- < 85%: v1 results retroactively suspect; consider re-running Phase 1b under v2.