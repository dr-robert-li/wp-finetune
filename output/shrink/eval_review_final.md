# Shrink study: final independent review

**1. Verification.** I checked every cell of all 19 rows in RESULTS.md against score/resources/memory.json and found 0 mismatches. I re-scored mix_eiq3xxs_f16 and c_eiq3xxs_kvq8 from their judge_responses.jsonl into /tmp. Both reproduce exactly: 0.921 / 96.7% and 0.858 / 95.0% (114/120).

**2. Noise.** Running the baseline weights through a different numeric path already costs agreement: rt_kvq8 scores 0.942 and rt_cpumoe 0.957 against baseline, and only 0.916 against each other. In every same-file pair, 0 of 121 texts are identical. So the practical ceiling is about 0.95, not 1.0. The two eiq3xxs runs differ by 0.063 (paired bootstrap 95% CI [0.020, 0.115]). The 3-bit file looks more fragile to perturbation, and a single run is one draw. Bootstrap CIs (2000 resamples) are 0.07–0.09 wide.

**3. Unreadable answers.**
- Every candidate failure is a repetition loop that hits the 2048-token cap: iq4xs#89, q4km_f16#109, q5km_f16#44, and the not_json items mix_eiq4xs#47 and c_eiq4xs#43. The baseline's longest answer is 1512 tokens.
- rt_kvq8 (baseline weights) also loops once (#65), so one loop per 121 is within noise.
- Two failures are not cap-related. c_eiq3xxs_kvq8#115 stopped early at 1005 tokens without any scores (a format break). iq3xxs#44 stopped at 840 tokens.
- prune_k160 collapses into word loops: 7 unreadable and 19 with no verdict.

**4. The proposed bar.** A fixed 0.90 sits inside every candidate's CI. Each verdict flip moves the rate by 0.8%, and the exact-weights runs already show 1 flip and 1 loop. Better: require the rho CI lower bound ≥ 0.85, and require it on two perturbed runs (default plus kvq8). Count not_json items as unreadable. Add a repetition guard (repeat-penalty or loop stop) so a loop fails fast instead of running to the cap.

| variant | GiB | GPU MiB | agree_rho [95% CI] | verdict | unread | verdict |
|---|---|---|---|---|---|---|
| iq4xs_im_f16 | 15.44 | 15992 | 0.935 [0.897,0.960] | 99.2% | 1 (loop) | **SHIP (default)** |
| c_iq4xs_kvq8 | 15.44 | 15946 | 0.939 [0.909,0.961] | 98.3% | 0 | SHIP (runtime opt) |
| mix_eiq4xs_f16 | 15.88 | 16304 | 0.926 [0.886,0.957] | 98.3% | 0 (+1 loop) | DROP (redundant) |
| mix_eiq3xxs_f16 | 11.99 | 12314 | 0.921 [0.880,0.950] | 96.7% | 0 | MAYBE (small) |
| c_eiq3xxs_kvq8 | 11.99 | 12267 | 0.858 [0.796,0.904] | 95.0% | 1 | DROP |
| q4km_f16 | 17.43 | 18017 | 0.916 [0.869,0.949] | 97.5% | 1 (loop) | DROP |
| q5km_f16 | 20.36 | 20954 | 0.913 [0.865,0.947] | 97.5% | 1 (loop) | DROP |
| rt_cpumoe | 23.47 | 2342 (+23 GB RAM) | 0.957 [0.928,0.976] | 100% | 0 | MAYBE (low-VRAM, 9.9 tok/s) |
| prune_k192 / k160 | 20.38 / 17.3 | – | 0.879 / 0.880 | 98.3 / 99.0% | 0 / 7 | DROP |

**5. Recommendation.**
- **Ship as default:** iq4xs_im_f16. It saves 34% (about 8 GB of VRAM), is the fastest at 69 tok/s, and sits at the noise ceiling.
- **Small:** mix_eiq3xxs_f16 only as "small (experimental)", and only after 2 more perturbed runs clear the bar. Do not pair it with kvq8.
- **Drop:**
  - mix_eiq4xs: larger and slower than iq4xs with no gain.
  - q4km/q5km (both versions), q3km, iq3xxs_im, mix_e4km: larger and/or lower agreement than iq4xs.
  - Both prune variants.
  - rt_kvq8 as a size option: it saves only 47 MiB on this one-prompt memory probe.
