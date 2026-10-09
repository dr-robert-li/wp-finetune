# Shrink study: stability review (independent)

**Method.** I re-scored all 8 captures against baseline_q6k using `shrink_score.load_capture`/`score`. Every agree_rho, verdict_agree, mean_abs_diff, parse_fail and not_json value matches its score.json exactly. The rho CIs are bootstrap 95% (2000 resamples over paired items, seed 0). Unreadable means parse_fail plus not_json.

**Finding 1: `-fa on` is not a perturbation.** Apart from the provenance line, s_eiq3xxs_fa is byte-identical to mix_eiq3xxs_f16 (0 of 121 responses differ). s_iq4xs_fa is likewise identical to iq4xs_im_f16. Flash attention is already on by default (`-fa auto`) on this GPU. Both `_fa` runs are therefore repeats of the default run and do not count toward the two-perturbed-runs bar.

| variant | rho [95% CI] | verdict | unreadable (pf+not_json) |
|---|---|---|---|
| mix_eiq3xxs_f16 (default) | 0.921 [0.878,0.951] | 96.7% | 0 |
| s_eiq3xxs_fa (= default) | 0.921 [0.878,0.951] | 96.7% | 0 |
| s_eiq3xxs_ub128 | 0.930 [0.889,0.957] | 96.7% | 1 (nj #25, loop to cap) |
| c_eiq3xxs_kvq8 | 0.858 [0.797,0.903] | 95.0% | 1 (pf #115, format break) |
| iq4xs_im_f16 (default) | 0.935 [0.897,0.960] | 99.2% | 1 (pf #89, loop) |
| s_iq4xs_fa (= default) | 0.935 [0.897,0.960] | 99.2% | 1 (pf #89) |
| s_iq4xs_ub128 | 0.926 [0.891,0.950] | 97.5% | 0 |
| c_iq4xs_kvq8 | 0.939 [0.908,0.960] | 98.3% | 0 |

**Paired bootstrap rho differences:**

| comparison | 95% CI | reading |
|---|---|---|
| eiq3xxs ub128 vs kvq8 | [+0.029, +0.124] | kvq8 is a real degradation for the 3-bit file |
| iq4xs ub128 vs eiq3xxs ub128 | [-0.038, +0.038] | no difference |
| iq4xs default vs eiq3xxs default | [-0.029, +0.058] | no difference |

**Bar check:** CI lower bound >= 0.85 on two perturbed runs, and at most 1 of 121 unreadable.
- iq4xs_im_f16 passes on every real perturbation: ub128 (0.891, 0 unreadable) and kvq8 (0.908, 0 unreadable).
- eiq3xxs passes on ub128 (0.889, 1 unreadable, which is at the limit) but fails on kvq8 (0.797). Only one genuine passing perturbation exists, because the -fa run was a no-op. Unreadable outputs (a cap loop on ub128, a format break on kvq8) appear in 2 of its 3 distinct runs.

**Verdict.**
- **iq4xs_im_f16: SHIP as the default (confirmed).** It is stable across ub128 and kvq8 and is the only file that clears the bar.
- **mix_eiq3xxs_f16 (12 GB): SHIP-as-experimental.** Conditions:
  - Use the default KV cache only. Never use kvq8.
  - Document it as more fragile.
  - Before promoting it to a full SHIP: run one more genuine perturbation (e.g. `-ub 256` or a different `-b`/thread count, not `-fa`), and add a repetition guard so cap loops fail fast.
