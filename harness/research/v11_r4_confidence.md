# R4: confidence and FP/FN risk (T4)

Research only. No harness files were edited. Scripts live in the session scratchpad
(`/tmp/claude-1000/.../scratchpad/r4/`: `units.py`, `run.py`, `ana.py`, `full.py`, `eval_flags.py`, `eval2.py`), with raw runs in `runs/*.jsonl`.
The scratchpad is temporary. The method below is enough to redo the runs.

## Setup

- Server: host `~/work/llama.cpp/build/bin/llama-server` (build 9954), `models/_gguf/shrink-iq4xs-im_f16.gguf`, the flags from `e2e.sh host_up`, `--parallel 4 -c 32768`, port 8099. It ran under `flock /tmp/wpjudge-gpu.lock`, and I checked nvidia-smi before and after. The server was killed after the runs.
- Requests: the harness `PROMPT` and `enable_thinking: false`, parsed with `wpjudge.judge.to_judgment`. Scripts called the server directly with `logprobs=True, top_logprobs=5`.
- **Unit set (69).** Built by `units.py` with `wpjudge.extract`:
  - **fixture (10).** 5 planted units: ground truth (GT) FAIL and capped. 4 clean units plus the template: GT PASS, not capped. Source: `vuln-plugin.expected.json`.
  - **classic-editor (39).** GT assumed to be PASS and not capped (a mature wp.org plugin). `add_redirect_helper` is the known FP.
  - **val (20).** From `data/reasoning_dataset/openai_val.jsonl`. GT is the teacher `<judge_output>`. 8 are FAIL+cap, 8 PASS+no cap and 4 FAIL+no cap (the first ones of each kind in file order).
- **Runs (8 judgments per unit).**

| Run | Settings | Wall |
|---|---|---|
| `g_w1` | temp 0, workers 1, logprobs | 839.5 s, 12.2 s/unit |
| `g_w4a` | temp 0, workers 4, logprobs | 425.2 s, 6.2 s/unit |
| `g_w4b` | temp 0, workers 4, no logprobs | 386.9 s, 5.6 s/unit |
| `t07` | temp 0.7, workers 4, 5 samples per unit (345 requests), logprobs | 2128.6 s, 30.8 s/unit for 5 samples |

- **Instability ground truth.** A unit is "unstable" if its cap state or verdict differs across its 8 judgments.
  - 6 of 69 are unstable: classic `init_actions`, `register_settings`, `settings_2`, `add_edit_php_inline_style`, and val `LP_Admin_Ajax::duplicator`, `LP_Admin_Ajax::init`.
  - Caveat: this definition uses the t07 samples, so the recall of option 3 is partly circular.

## Baseline findings (they shape everything below)

1. **Cap and verdict flips are rare.**
   - At temperature 0.7 the pairwise flip rate is 2.6% for the cap and 3.2% for the verdict.
   - Greedy w1 vs w4a, and w4a vs w4b, each flip on 1 of 69 units (duplicator, sec 6 vs 4).
   - Overall-score noise is large: spread ≥ 10 on 34 of 69 units across all 8 judgments, and on 3 of 69 across the 3 greedy runs only.
2. **The planted fixture bugs are never missed.** All 5 were FAIL and capped in every judgment: 8/8 each, 39 of 39 judged.
3. **`ser_render_search_box` (14 → 42 in v1).**
   - Greedy overall was 14 / 46 / 28 (sec 1 / 4 / 2). Across all judgments the spread is 32.
   - It was FAIL and capped in 7/7 judged runs. The instability is in the overall score only; the verdict and cap are stable.
   - At temp 0.7 it also produced 1 degenerate output: the code echoed back in 68 tokens, which counts as unjudged. The harness retry absorbs that.
4. **The classic-editor CSRF FP lives in the prose only.**
   - `add_redirect_helper` JSON security was 6, 6, 7, None, 7, 6, 6, 6. It was never capped in 8 judgments; overall was 64 to 72.
   - The prose said "Security 4/10" in both greedy runs (v1 had prose 4 and JSON 5).
   - The FP therefore reaches the report only through the critique text, not through the cap.
5. **The dominant error is systematic FN, not noise.**
   - On val, the model said PASS on all 12 GT-FAIL units (overall 58 to 80; the teacher gave 15 to 55). It missed the cap on 7 of 8 GT-capped units (w1) and 6 of 8 (w4a).
   - No re-judge strategy below fixes any of these (option 6).
   - Caveat: the teacher labels are noisy. For example, `wp_cache_delete` is labelled FAIL with security 2.
6. **The prose is harsher than the JSON on security (input for R2).**
   - On 9 of 39 classic units, the prose says security ≤ 4 and the JSON says 6. That is the same pattern as the FP.
   - On 3 of the 7 val FNs (duplicator, `apply_hooks_on_draft_transition`, YITH `form`), the prose said 4. That matches the teacher, while the JSON said 6.
   - So switching the cap to prose scores would fix some FNs but cap classic-editor. Neither source is reliably right near the threshold.
7. **Logprobs work on llama-server build 9954.** `/v1/chat/completions` with `logprobs=true` returns per-token `logprob` and `top_logprobs` for the whole output, including the `<judge_output>` digits and the verdict token.

## Options, evidence

Metrics are per base run (g_w1 / g_w4a).
- "unstable caught" uses the 6-unit set.
- "cap err" = FP+FN against GT. At greedy there are 0 FP, and 7 (w1) / 6 (w4a) FN.
- "verdict err" is 12/69 in both runs.
- "sb" = flags `ser_render_search_box`; "helper" = flags `add_redirect_helper`.

| # | Option | Flagged | Unstable caught | Cap err caught | Verdict err caught | sb | helper | Extra cost | Simplicity |
|---|---|---|---|---|---|---|---|---|---|
| 1a | near_cap: a cap dim in [th−1, th+1] | 5 / 7 (7% / 10%) | 1 / 2 | 0 / 0 | 1 / 2 | no / yes | no | 0 | trivial |
| 1b | near_verdict: overall 50 to 62 (model PASS min = FAIL max = 54) | 2 / 1 | 1 / 1 | 0 | 2 / 1 | no | no | 0 | trivial |
| 2 | prose_cross: a prose cap-dim score on the other side of `cap_threshold` from the JSON | 13 / 13 (19%) | 3 / 1 | 3 / 2 | 3 / 3 | no | **yes / yes** | 0 | small; reuses the `DIM` regex |
| 3 | n samples at temp 0.7, flag if not unanimous | n=3: 3 (4%); n=5: 6 (9%) | 3; 6 (circular) | — | — | no | no | +3× / +5× calls; t07 took 30.8 s/unit for 5 vs 6.2 s/unit greedy | moderate |
| 3m | majority vote of n samples as the answer | — | — | n=3: FN 6; n=5: FN 7 | n=3: 10; n=5: 11 (vs 12) | — | — | as above | moderate |
| 4a | logprob: P(cap dim ≤ th) at the emitted JSON digit, in (0.05, 0.95) | 0 / 0 | 0 | 0 | 0 | no | no | logprobs on | small |
| 4b | logprob: verdict-token p < 0.9 | 7 / 7 (10%) | 3 / 3 | 3 / 2 | 4 / 4 | no | no | logprobs on (see below) | small |
| 5 | re-judge w1 vs w4, flag on cap or verdict difference | 1 (1%) | 1 | 0 | 0 | yes, as \|Δ\| ≥ 10 | no | +1× calls; w1 is 2× slower | small, but doubles time |
| 6 | second pass on flagged units: +2 temp-0.7 samples, majority of 3 | 4b: 7, 2: 13, 1a: 5, 4b∪2: 16, all three: 18 to 20 | — | **0 errors changed, for every flag set** | **0 changed** | — | — | +0.14 to 0.58 calls/unit | moderate |

Notes on each option:

- **1a.** Most hits are the planted fixture 4s, which are stably capped (8/8). Unstable units sit at security 6 to 7, outside the window. `cap_reasons` already shows the score.
- **1b.** It is weak because the model rarely emits overall scores in 50 to 62.
- **2.** It is free, it is the only signal that catches the known FP, and it catches real FNs (3 of 7 in w1).
  - Precision is low: 9 of the 13 are classic-editor units where the prose says 4 and the JSON says 6.
  - It depends on R2. If T2b makes the prose and the JSON agree, this flag goes quiet or changes meaning.
- **3.** It catches everything that flips, by construction, but at 3 to 5× the cost. The majority vote does not reduce cap errors.
- **4a.** It is useless. The digit distributions are spread across neighbours but rarely across the threshold. Examples:
  - planted 4s: {4: .64, 3: .28, 2: .06}, so P(≤ 4) ≈ 1;
  - `add_redirect_helper` security 6: {6: .58, 5: .31, 7: .09, 4: .01}.
  - Cap flips come from a different reasoning path, not from local digit uncertainty.
  - A minimum cap-digit p < 0.6 flags 12 units but only 1 unstable one.
- **4b.** It is the best cheap signal.
  - Precision: 5 of the 7 flagged are errors or unstable. In w1 those are `settings_2` (p .55), duplicator (.51), `LP_Admin_Ajax::init` (.82), `apply_hooks…` (.78) and `form` (.88). The other 2 are fixture planted units at p ≈ .89, which are correct FAILs.
  - Base rate: 16 of 69 units are errors or unstable.
  - Threshold sweep (w1): 0.8 gives 3 flagged; 0.9 gives 7; 0.95 gives 13; 0.99 gives 17 (6 of 12 verdict errors). The knee is at 0.9. On w4a the same thresholds give 2 / 7 / 10 / 19.
- **5.** It mostly re-measures slot nondeterminism. It flags `ser_render_search_box` only as a score difference.
- **6.** The errors are systematic, so re-sampling repeats them. Greedy re-judging at w1 is deterministic and adds nothing.

**Cost of logprobs.**
- At w4: 425.2 vs 386.9 s wall (+10%), and 32.4 vs 28.9 ms per completion token (+12%), one run each.
- Those runs used `top_logprobs=5`. Requesting `logprobs=true` without `top_logprobs` should cost less, but I did not measure it.
- Logprobs did not change the outputs beyond the normal w4 noise: the g_w4a vs g_w4b differences match the slot nondeterminism.

**Coverage of the union 4b ∪ 2** (16 / 17 units, 23 to 25%):
- unstable caught 4/6 (w1) and 3/6 (w4a);
- verdict errors 4/12 and 5/12; cap errors 3/7 and 2/6;
- catches the helper FP; misses `ser_render_search_box`.
- Its verdict and cap are stable, and its score noise moved the report overall by only 0.3 in v1. No cheap signal predicts that unit's score swing.

## Recommendation (the smallest design that helps)

Two free per-unit signals and one summary count. No re-judge knob.

1. **`judge.py`**
   - Add `logprobs=True` (no `top_logprobs`) to both `_judge_create` calls.
   - In `judge_one`, take the logprob of the first non-blank, non-quote token after `"verdict":` inside `<judge_output>`. Store `verdict_p = round(exp(lp), 3)`, or `None` when the server returns no logprobs or there is no verdict.
   - Pass it through `to_judgment`, keeping `to_judgment` pure: it gains a `verdict_p=None` argument.
   - About 10 lines.
2. **`aggregate.py`**, per judged row:
   - `confidence_reasons`: a list containing
     - `"verdict_p"` when `verdict_p is not None and verdict_p < cfg.min_verdict_p`;
     - `"prose_disagrees"` when a cap dimension's prose score, parsed with the existing `DIM` regex, is on the other side of `cfg.cap_threshold` from the JSON score. `None`, N/A or absent counts as "above". Use the same threshold the cap uses, which lines up with T2a.
   - `low_confidence = bool(confidence_reasons)`.
   - Summary: `"low_confidence": <count of judged rows with low_confidence>`.
   - Skipped and unjudged rows: `low_confidence: false, confidence_reasons: [], verdict_p: null`. (Unjudged rows are already counted in coverage.)
   - About 12 lines.
3. **`config.py`**: `min_verdict_p: float = 0.9` (`WPJUDGE_MIN_VERDICT_P`). This is the calibration knob for build or quantisation drift. Set it to 0 to disable.
4. **README**:
   - Add a "low confidence" row to the report fields.
   - Under the caveats, say the flag marks review leads. It does **not** fix the systematic leniency: on val, 7 of 8 GT-capped units were missed and no flag or re-judge recovers most of them.
   - Mention that non-llama.cpp servers may return no logprobs, so `verdict_p` is null and only `prose_disagrees` applies.
5. **Not built:** near-threshold flag, n-sample mode, workers=1 re-judge, second-pass re-judge.
   - Measured benefit is 0 changed errors, at +14 to 400% calls.
   - Add n-sample only if a future model shows noise-driven (not systematic) errors.

Sequencing: build after R2. If T2b changes where scores come from, re-check `prose_disagrees`. If the prose and the JSON are made consistent, drop that reason and keep `verdict_p`.

## Files to change

- `harness/wpjudge/judge.py`: logprobs request, `verdict_p` extraction, `to_judgment(..., verdict_p=None)`.
- `harness/wpjudge/aggregate.py`: `confidence_reasons`, `low_confidence`, summary count. Reuse `DIM`.
- `harness/wpjudge/config.py`: `min_verdict_p`.
- `harness/SPEC.md` / `README.md`: schema fields, knob, caveat.
- `harness/tests/test_judge.py`, `tests/test_aggregate.py`, `tests/e2e.sh` (chk.py).

## Tests

**Unit**
- `test_judge.py`:
  - A fake response whose `logprobs.content` tokens spell `<judge_output>\n{\n "verdict": "FAIL", …` with the FAIL token at logprob ln(0.5) gives `verdict_p == 0.5`.
  - With `logprobs=None`, `verdict_p` is None and the unit is still judged.
  - A verdict token split as `"`,`FA`,`IL` takes the first non-quote piece.
  - The second `_judge_create` path (template rejected) also sends logprobs.
- `test_aggregate.py`:
  - prose "Security: score 4/10" with JSON 6 and `cap_threshold` 4 gives `prose_disagrees`;
  - prose 4 with JSON 4 gives no reason;
  - prose "Security: severity high (4/10)" with JSON missing gives `prose_disagrees`;
  - with `cap_threshold=5`, prose 5 and JSON 6 gives the reason, and prose 4 and JSON 5 does not;
  - `verdict_p` 0.85 gives the reason at the default; `min_verdict_p=0` disables it; `verdict_p` None gives no reason;
  - the summary count equals the number of rows; skipped and unjudged rows are false.

**E2E** (live server, real CLI / API / docker)
- `cli` fixture:
  - every judged row has `0 ≤ verdict_p ≤ 1`, a bool `low_confidence` and a list `confidence_reasons`;
  - `summary.low_confidence` equals the row count;
  - with `WPJUDGE_MIN_VERDICT_P=0`, no row has `"verdict_p"` in its reasons.
- `real` classic-editor:
  - `add_redirect_helper` has `"prose_disagrees"`. It did in 3 of 3 observed runs (v1 report, g_w1, g_w4a). Make this a soft check that warns if it fails, since it depends on the model.
  - Wall time stays within +20% of v1 (221 s).
- `api` / `allinone` (container build b11515): rows have a non-null `verdict_p`. This proves the shipped llama-server build returns logprobs; I verified that on 9954 only.
- Golden unchanged: the planted < clean ordering and the cap assertions still pass with logprobs on.

## Risks

- **Small evidence base.** 69 units, one run per greedy configuration, 5 samples at temp 0.7.
  - The thresholds (0.9) are tuned on that set.
  - The "unstable" set is 6 units and is partly defined by the t07 samples.
- **Weak GT.** The classic-editor GT is assumed clean, and the val GT is teacher labels.
- **Flags do not reduce errors.** The FN problem is model leniency (val), not noise. Users may read `low_confidence: false` as "verified". The README must say otherwise.
- **`prose_disagrees` precision is low.** On classic-editor it fires on about 25% of units, because the prose says security 4 where the JSON says 6. It may become noise, or vanish, depending on the T2b fix.
- **Logprob availability and calibration** vary by server and build (cross-build drift is already documented). If logprobs are absent, `verdict_p` is null and the flag degrades to the prose check only.
- **Logprob overhead is up to about +10% wall.** It was measured with `top_logprobs=5`; without it the overhead should be lower. Re-measure in E2E.
