# Shrink study: ub256 review (independent)

**Run reviewed.** `s_eiq3xxs_ub256` serves the same file as `mix_eiq3xxs_f16` (`models/_gguf/shrink-mix-eiq3xxs_f16.gguf`, per serve.log) with `-ub 256` (shrink_run_all.sh:64). It used greedy decoding on the same 121 wp_judge prompts and was scored against baseline_q6k.

## 1. Is this a genuine perturbation? Yes.
| compared with | responses that differ |
|---|---|
| mix_eiq3xxs_f16 (default ub) | 52 / 121 |
| s_eiq3xxs_ub128 | 90 / 121 |

The outputs are not byte-identical to either earlier run, so this does not repeat the `-fa` no-op.

## 2. score.json check
I re-scored the run with `shrink_score.load_capture`/`score`. Every field matches score.json: n=121, parse_fail 0, not_json 0, agree_rho 0.9075, mean_abs_diff 2.397, verdict_agree 0.9587, ref_rho 0.570.

The bootstrap 95% CI on agree_rho (2000 resamples, seed 0) is **[0.863, 0.938]**. Its lower bound clears the 0.85 bar. Unreadable outputs: **0 / 121**.

## 3. Paired bootstrap rho difference (2000 resamples, seed 0, items parsed in all runs)
| comparison | n | point | 95% CI | reading |
|---|---|---|---|---|
| iq4xs_im_f16 minus eiq3xxs_ub256 | 120 | +0.028 | [-0.017, +0.072] | no significant difference |
| mix_eiq3xxs_f16 minus eiq3xxs_ub256 | 121 | +0.013 | [-0.008, +0.037] | ub256 matches the 12GB default |

## 4. Bar check for the 12GB file (genuine perturbations only)
| run | rho CI lower bound | unreadable | pass |
|---|---|---|---|
| ub128 | 0.889 | 1 | yes (at the limit) |
| ub256 | 0.863 | 0 | yes |
| kvq8 | 0.797 | 1 | no; KV quantization is excluded by config |

Two genuine perturbed runs now pass, so the bar is met.

**Caveats**
- The ub256 lower bound (0.863) is the lowest of the passing runs and sits close to 0.85.
- verdict_agree is **95.9%** (about 5 PASS/FAIL flips in 121), compared with 96.7% on the other 12GB runs and 99.2% for iq4xs. A user of this tier should expect roughly 4x the verdict flips of iq4xs.
- I found no repetition guard in the run scripts. The cap loop seen on ub128 is therefore still unmitigated.

## Verdict: SHIP as the 'small' tier (conditional)
- iq4xs_im_f16 remains the default.
- Ship mix_eiq3xxs_f16 only with the default f16 KV cache. Never use kvq8.
- Document the higher verdict-flip rate (95.9 to 96.7% agreement against 99.2%).
- Add a repetition or loop guard, or keep the max_tokens cap with failure detection, before calling the tier GA. If this condition is not acceptable, it stays SHIP-as-experimental.
