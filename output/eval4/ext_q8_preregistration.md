# Phase 23-02 Extension: v4 Judge on the Shipped Serving Stack — Pre-Registration

**Written before any measurement.** Locks the primary metric, the win rule, and the fallback
before s0/s2 merges, GGUF conversion, or serving begin.

## Why this run exists

23-01's judge verdict (`output/eval4/VERDICT-EVAL4.md`) measured v4's judge on **bf16 vLLM**
(served s1 CI-lower 0.7125, capture ensemble CI-lower 0.7563 — both miss the 0.85/0.87
pre-registered targets). `output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md` found the served
figure is bounded by a **Tinker-capture-vs-vLLM-serve engine-numerics ceiling**, not a training
or label defect (capture 0.8358 > vLLM-served 0.7872; a from-scratch fp32-accumulation merge
reproduced the same ~0.78, ruling out the merge as the cause). It explicitly recommends
re-evaluating "against the CAPTURE path or a fixed serving stack."

v3's shipped judge (v1.3) was never served on bf16-vLLM for its final number — it shipped on
**llama.cpp Q8_0 GGUF**, where the 3-seed ensemble scored 0.8056 (`output/packaging/pkg03_ens8192_results.json`),
matching the independent vLLM-bf16-ensemble reference (0.8075) end to end. This extension asks:
does v4's judge, measured on that *same shipped stack* (not vLLM bf16), clear the v3 bar? This is
the deciding measurement for judge-only shipping.

## Primary metric

**v4 Q8_0 GGUF, 3-seed (s0/s1/s2) median ensemble, Spearman rho vs `output/relabel/val_labels_v1.json`,
on the same 121-item `data/reasoning_dataset/openai_val.jsonl` val set used by v3.** Served via
llama.cpp `llama-server` (CUDA, GB10), `--jinja`, temp 0, 8192-token completion cap — identical
harness/config to v3's `scripts/_pkg_gguf_eval_run.sh` / `scripts/_pkg_ens8192_run.sh`. Scored with
the unmodified `scripts/relabel/eval_relabel.py` (single seed) and `scripts/relabel/eval_relabel_ensemble.py`
(3-seed median), the same functions that produced both v3's 0.8056 and v4's capture-path 0.8358/0.8160.

## UNEQUIVOCAL WIN rule (mechanical, applied in this order)

**UNEQUIVOCAL WIN :=**

**(a)** v4 Q8 3-seed ensemble point rho **> 0.8056** (v3's shipped ensemble point estimate)

**AND**

**(b)** Paired bootstrap of the per-item delta, resampling the 121 val items with replacement
(10,000 resamples, `numpy.random.default_rng(1337)`), computing `rho(v4_ens, resample) -
rho(v3_ens, resample)` in each resample using the SAME resampled item indices for both arms (this
is what makes it "paired" — it isolates the item-level rho delta from independent sampling
noise), has **CI-lower (2.5th percentile of the 10,000-delta distribution) > 0**.

Both v3 and v4 per-item ensemble scores are joined on the shared `val:{idx}` key produced by
`eval_relabel_ensemble.py`'s index mapping (v3 source: `output/packaging/ens8192/q8_s{0,1,2}/judge_responses.jsonl`;
v4 source: this run's `output/eval4/ext_q8/q8_s{0,1,2}/judge_responses.jsonl`) — `data/reasoning_dataset/openai_val.jsonl`
is unchanged since 04.1/04.2 (verified: last touching commits predate v3, confirmed same 121-row
`<wp_judge>` subset), so item identity and NEW-label alignment hold across milestones.

**Fallback rule (fires only if (a)+(b) is infeasible due to item-ID mismatch discovered during
scoring):** UNEQUIVOCAL WIN := v4 ensemble CI-lower (from `eval_relabel_ensemble.py`'s own 2000-resample
bootstrap, unpaired) **> 0.8056** (point). The receipt records which rule fired
(`rule_fired: "paired_bootstrap"` or `"fallback_ci_lower_vs_v3_point"`).

## Secondary reads (context only, non-gating)

- v4 single-seed s1 Q8 rho (already-merged checkpoint, first seed converted+served).
- v4 Q8 ensemble vs v4 bf16-vLLM ensemble (0.7872 served-s1 / capture 0.8358 / capture-ensemble
  0.8160) — does llama.cpp lift the numerics ceiling the same way it did for v3 (bf16-llama.cpp
  0.8100 matched vLLM-bf16-ensemble 0.8075)?

## Failure disposition

If v4 does not beat v3 on this stack (rule (a) fails, or (a) passes but (b) fails, or the fallback
misses), that is a **valid recorded outcome**: v3's judge (v1.3, Q8 ensemble 0.8056) stays the
canonical shipping artifact, and this extension's judge-only-ship recommendation is **no** on the
v4 base. No re-running with different seeds/params to chase a pass — the rule is applied
mechanically, once, against whatever the harness produces.

## Scope note

This run measures the judge role only. It does not reopen the gen-role verdict (raw base wins,
per 23-01) or the relabel-campaign re-open condition (still unmet per DIAGNOSTIC_SYNTHESIS.md).

---
*Pre-registered 2026-07-15, before s0/s2 merge, before any GGUF conversion or serving for this extension.*
