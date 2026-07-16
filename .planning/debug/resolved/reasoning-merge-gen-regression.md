---
status: resolved
slug: reasoning-merge-gen-regression
trigger: "04.4 — reasoning LoRA merge degrades base WP generation/judge ability (D-IT-02 diagnosis)"
created: "2026-06-10"
updated: "2026-06-10"
phase: 04.4-reasoning-eval-adapter-merge-inserted
goal: find_root_cause
---

# Debug: reasoning-merge-gen-regression

## Symptoms

DATA_START

**Expected behavior:** Merging the wp-reasoning-v3 Tinker LoRA adapter onto the stock
Qwen3-30B-A3B base should ADD judge-reasoning ability while PRESERVING the base model's
WP code-generation ability (wp-bench score >= baseline merged-v2 ~0.4537) and judge-output
fidelity (parse rate <= 5%, Spearman >= ~0.268 baseline).

**Actual behavior:** Every merged-served reasoning candidate DEGRADES generation:
- **v3 (target_modules=all-linear merge):** REVL-04 wp-bench 0.3716 < baseline 0.4537 FAIL
  (execution correlation 0.292 vs baseline 0.417). Judge path DID transfer (plan-02 fidelity
  L3 Spearman >= 0.95, human-approved) — so: good judge, degraded generator. Parse-failure
  rate on the 121-row val judge census = 0.190 (23/121).
- **v4 attempt-1 (lm_head LoRA stage EXCLUDED, attention q_proj KEPT):** parse-failure rate
  0.248 (30/121) — WORSE than v3 0.190 — and judge Spearman REGRESSED to 0.153 (< baseline
  0.268). wp-bench never ran (fail-fast precondition early-exit on the failed parse gate).
  REVL-02 generation PHPCS stayed 1.0 (raw PHP syntax is fine; the failure is in the model's
  ability to emit parseable structured JUDGE output and to score correctly post-merge).

**Key discriminating evidence:** Excluding the manual lm_head LoRA stage did NOT improve the
parse rate — it made it slightly WORSE (0.190 -> 0.248). This FALSIFIES the lm_head hypothesis
(D-IT-04). The generation/judge regression is NOT driven by the lm_head delta. The remaining
merged components are the suspects: (a) the attention q/k/v/o PEFT delta merge, (b) the MoE
per-expert down_proj/gate_up_proj deltas (Unsloth contiguous-block convention), (c) an
interaction / magnitude-scaling issue across components, or (d) a merge-math/convention error
that the 3 anchor gates (which only check MoE-expert weights + expert-block forward, NOT the
generation/judge decode path) do not catch.

**Error messages / failure mode:** No crash. Failure manifests as malformed judge output —
the merged model emits `<think>` reasoning that is never closed / unparseable structured
verdicts (parse_failure_rate), plus degraded code-execution correlation on wp-bench. Both
merged candidates pass the merge ANCHOR gates (tensor byte-exact, fp32-control, forward
router-invariant) yet fail the BEHAVIORAL eval — the anchors certify weight-merge correctness
but not generation/judge-decode preservation.

**Timeline / when it started:** Began as soon as any reasoning adapter was merged onto the
base. The base merged-v2 (no reasoning adapter) scores 0.4537 wp-bench cleanly. The unmerged
Tinker-sampled adapter behaved correctly in Phase 4.3 (REVL-05 human-APPROVED wp-reasoning-v3,
parse/judge quality fixed). So the regression appears at MERGE time, not in the adapter itself —
the served-merged weights diverge behaviorally from the Tinker-sampled adapter that was approved.

**Reproduction:** Serve a merged candidate dir under vLLM (thinking-OFF via
chat_template_kwargs enable_thinking=false, stock 151,936-routing tokenizer, served-name
wp-30_70, port 8021), then run the REVL-01A 121-row parse census
(scripts/_04.4_revl01a_v4.py-style harness) over data/reasoning_dataset/openai_val.jsonl.
Measure parse_failure_rate + judge Spearman. ~1h per variant on GB10 (decode-bound).

DATA_END

## Investigation Constraints / Cost Notes

- GPU serve + 121-row census ~1h/variant; full wp-bench ~2.7h. PREFER CHEAP STATIC TESTS FIRST.
- **Cheapest first (no GPU serve):** static weight-space diagnosis — load base vs merged
  candidate shards, compute per-component relative delta magnitude (attention q/k/v/o vs MoE
  down_proj/gate_up vs embed/lm_head), check for anomalous scaling (the 2026-05-29 probe found a
  council "broadcast" math was 12x over-magnitude before it was falsified — re-audit per-component
  magnitudes the same way). A component whose merged delta is orders-of-magnitude off vs the
  Tinker adapter's intended per-expert/per-proj delta is the prime suspect WITHOUT any serve.
- Then a SMALL targeted census (e.g. 10-20 prompts, not 121) to confirm a hypothesis before a
  full 121-row run.
- Component-ablation merges (attention-only, MoE-only) are the EXPENSIVE confirmation — only
  build/serve them after the static audit narrows the suspect.

## Available Evidence Artifacts (read these, do not re-measure)

- output/eval_reasoning_v4_nolmhead/revl01a_v4.json — v4 parse 0.248, Spearman 0.153
- output/eval_reasoning_v3/04.4_wp_bench_results.json — v3 wp-bench 0.3716 FAIL
- output/merge_v4_nolmhead/merge_report.json — v4 merge components, anchors, per-expert differ
- output/merge_v3/merge_report.json (and models/_staging/...-v3) — v3 merge report
- scripts/merge_tinker_v3.py — the merge implementation (--exclude-lm-head flag, MoE convention)
- .planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-GATE-LEDGER-V4.md
- .planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-0{1,2,3,6,7,8}-SUMMARY.md
- STATE.md "Session 2026-05-29 reasoning MERGE COMPLETE" + "P0 forensics" — Unsloth fused-MoE
  shared-rank convention, PEFT vs Unsloth strided-vs-contiguous B-indexing, gate/up chunk split.

## Candidate Hypotheses (seed — debugger to test/eliminate via scientific method)

- H1: Attention q/k/v/o PEFT delta merge corrupts the decode path (parse/Spearman) —
  test by static magnitude audit of attention deltas, then attention-only ablation census.
- H2: MoE per-expert delta convention error (contiguous-block vs strided B-indexing, or
  gate/up chunk-split order) introduces subtle expert corruption that anchors miss because
  anchors check the same convention they were merged with (circular) — test by an INDEPENDENT
  re-derivation of a few expert deltas vs the served weights.
- H3: A scaling/alpha mismatch (lora_alpha/r scale applied once too many or omitted on one
  component) inflates/deflates one component's delta — static per-component magnitude audit.
- H4: thinking-token / chat-template interaction post-merge: the merged model's `<think>`
  emission diverges from the Tinker-sampled adapter (unterminated think) — compare served
  template behavior vs the adapter's Phase-4.3 sampling config.
- H5: The anchor gates are insufficient (they certify weight-merge but not generation-decode);
  the merge is "correct" yet the adapter itself does not survive bf16 weight-space fusion the
  way it survived Tinker LoRA-runtime application — i.e. LoRA-runtime != weight-merge for this
  adapter. Test: compare Tinker-runtime adapter output vs merged-weight output on identical
  prompts (the definitive "is it the merge or the adapter" discriminator).

## Eliminated

- lm_head LoRA stage (D-IT-04) as the CAUSE: EXCLUDED in v4 attempt-1, parse rate got WORSE
  (0.190 -> 0.248). lm_head is NOT the cause; if anything the lm_head delta was mildly HELPING
  parse fidelity (or the 0.190->0.248 gap is run-to-run noise — see E1). ELIMINATED 2026-06-10.

- MoE/attention merge as a v3-vs-v4 VARIABLE: the v3 and v4 staging models have BYTE-IDENTICAL
  MoE+attention merged weights (E1). The regression is present with the MoE+attention merge ALONE
  (v3 parse 0.190 and wp-bench 0.3716 both already FAIL). So the suspect set is firmly {MoE
  per-expert deltas, attention q/k/v/o PEFT, or weight-merge-vs-LoRA-runtime divergence (H5)} —
  NOT lm_head, and NOT any v3/v4 weight difference (there is none besides lm_head).

- H5 (weight-merge vs LoRA-runtime as ROOT CAUSE of JUDGE regression) ELIMINATED 2026-06-10:
  The Tinker-runtime offline eval of wp-reasoning-v3
  (output/eval_reasoning/reasoning_v3_tinker/) shows Spearman HARD = 0.2626 (121 pairs, 0
  parse failures) — essentially identical to the merged-v2 baseline of 0.2678. So the ADAPTER
  IS CORRECT and survives weight-space fusion in terms of judge content. The adapter does NOT
  diverge from LoRA-runtime in the weight domain. The JUDGE parse regression is entirely in the
  EVALUATION HARNESS, not in the merge math.

- H1/H2/H3 (weight-merge convention errors): ELIMINATED 2026-06-10 for the judge regression.
  The Tinker-runtime offline eval with 0 parse failures and Spearman 0.2626 proves the merged
  model IS producing correct judge output — the parse failures and Spearman drop are harness
  artifacts, not model quality degradation. The weight merge is correct for the judge path.

- H5 / harness confound as explanation for WP-BENCH regression: ELIMINATED 2026-06-10 (E8).
  wp-bench output/eval_reasoning_v3/reasoning_merged/wp_bench_results_20260608_184438.jsonl
  has 0/344 records containing `<think>` tags — the usercustomize.py shim WAS active during
  wp-bench generation. Therefore the generation regression (0.4537 -> 0.3716) is REAL and
  INDEPENDENT of the think-kwarg harness bug. It is a genuine adapter-vs-generation tradeoff.

## Current Focus

- hypothesis: **TWO DISTINCT ROOT CAUSES FOUND**

  **RC-A (CONFIRMED): Judge parse regression = evaluation harness bug.**
  eval_judge/_run_eval_reasoning does NOT pass `chat_template_kwargs: {enable_thinking: False}`
  to vLLM. The merged Qwen3 model emits unclosed `<think>` blocks because the Tinker training
  renderer pre-filled the think block and the model never learned to close a real one. Responses
  output inside unclosed `<think>` fail the parse regex (`<think>.*?</think>` requires closing
  tag). Compounded by `_JSON_FIELD_TO_DIM` key-name mismatch that drops 4 high-weight dims
  even for parseable responses. Both defects are harness-only; the adapter and weight merge
  are mathematically correct (Tinker-runtime: 0 parse failures, Spearman 0.2626).

  **RC-B (CONFIRMED): WP-bench generation regression = genuine adapter-vs-generation tradeoff.**
  wp-bench output contains 0/344 `<think>` tags (E8), proving the usercustomize.py shim
  (enable_thinking=False) WAS active during generation. The shim was NOT the confound.
  The execution correlation drop (0.417 -> 0.292) is a real capability tradeoff caused by
  attention q/k/v/o and/or MoE per-expert deltas from the reasoning adapter shifting the
  model's code-generation distribution. This requires a separate fix (retrain with lower rank
  or targeted-module adapter).

- test: CONFIRMED for both root causes:
  - Tinker-runtime: 0/121 parse failures, Spearman 0.2626 (E3) — adapter IS correct
  - Merged-served: 13/13 unclosed `<think>`, 0 `</think>` present (E4)
  - eval_judge.py line 691: no extra_body / no chat_template_kwargs (E5)
  - wp-bench jsonl: 0/344 records have `<think>` (E8) — shim was active, regression is real
  - _JSON_FIELD_TO_DIM maps security_score not security (E6) — 4 dims dropped
- expecting: N/A — confirmed from static artifact inspection, no new GPU run needed.
- next_action: See Root Cause Report and Fix Direction below.

## Evidence

- E1 (2026-06-10, VERIFIED on disk by orchestrator): output/merge_v3/merge_report.json and
  output/merge_v4_nolmhead/merge_report.json have BYTE-IDENTICAL `fp32_control_detail` and IDENTICAL
  `per_expert_delta_differ_check` (w1=0.013921, w3=0.012874, w2=0.005419). Only differing field:
  `lm_head_applied` (v3=True, v4=False). => v3 and v4 staging models share identical MoE+attention
  weights; the only behavioral variable between them is the lm_head delta. v4 (no lm_head) parse
  0.248 > v3 (with lm_head) parse 0.190. Caveat: confirm v3-0.190 and v4-0.248 were measured on the
  SAME 121-row harness/config before treating the 0.058 gap as a real lm_head effect vs run noise.
- E2 (open): the regression already exists at v3 (parse 0.190, wp-bench 0.3716) with MoE+attention
  merge alone => suspect is in that merge or in the fusion process (H5), independent of lm_head.
- E3 (2026-06-10, CONFIRMED from artifacts): Tinker-runtime offline eval of wp-reasoning-v3
  (output/eval_reasoning/reasoning_v3_tinker/eval_judge_results.json):
  - excluded.parse_fail = 0 (out of 121)
  - Spearman HARD = 0.2626 (vs merged-v2 baseline 0.2678) — essentially identical
  - Spearman SOFT = 0.5361
  - Renderer: "qwen3_disable_thinking" (pre-fills `<think>\n\n</think>\n\n` before model output)
  This PROVES the adapter content is correct. The judge behavioral regression is NOT in the adapter
  weights or the weight-merge math.
- E4 (2026-06-10, CONFIRMED from captured artifacts):
  - output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/captured_responses.jsonl: 13/13
    captured responses have `<think>` open tag, 0/13 have `</think>` close tag. All are
    unclosed think blocks.
  - 8/13 responses complete naturally (end with `</judge_output>`) but ALL 8 stay inside the
    unclosed `<think>` — the model outputs its full judge analysis INSIDE the think block and
    never closes it.
  - 5/13 responses are truncated at max_tokens (capture used 2048, eval_judge uses 1024) before
    producing valid JSON — these are the hard parse failures.
  - By contrast: Tinker-runtime responses have 0/121 think tags (renderer pre-fills and hides
    the think block before the model output).
- E5 (2026-06-10, CONFIRMED from source code): eval/eval_judge.py line 691:
  `client.chat.completions.create(model=..., messages=..., max_tokens=1024, temperature=0.0)`
  — NO `extra_body` parameter, NO `chat_template_kwargs`. The vLLM endpoint receives no
  `enable_thinking` directive. The Qwen3 chat template (models/Qwen3-30B-A3B/tokenizer_config.json)
  shows: when `enable_thinking is false`, it prepends `<think>\n\n</think>\n\n` to force
  thinking-off mode. Without this kwarg, the merged model defaults to emitting `<think>`.
- E6 (2026-06-10, CONFIRMED from source code): eval/output_parsers.py lines 36-41:
  `_JSON_FIELD_TO_DIM` maps `security_score`, `performance_score`, `i18n_score`,
  `accessibility_score` — but the adapter (and Tinker-runtime training data) uses bare key names
  `security`, `performance`, `i18n`, `accessibility`. Only 3 of 9 dims are ever captured from
  the model's JSON output (wpcs_compliance, sql_safety, wp_api_usage). This is a SYMMETRIC
  bias (affects Tinker-runtime and merged-served equally) and is NOT the primary parse-failure
  cause — but it reduces Spearman signal for "successfully parsed" responses by dropping
  4 high-weight dims (D2_security weight=0.20 is the highest single dim).
- E7 (2026-06-10): The REVL-01A baseline of 0.2678 Spearman was measured on merged-v2 (no
  reasoning adapter) with 0 parse failures. The Tinker-runtime v3 achieves 0.2626 (E3) — within
  noise of the baseline. The merged-served v3 achieves only 0.056 Spearman (23 parse failures
  excluded). This 0.056 vs 0.2626 gap = harness bug, not model regression.
- E8 (2026-06-10, CONFIRMED by static grep): wp-bench output file
  output/eval_reasoning_v3/reasoning_merged/wp_bench_results_20260608_184438.jsonl
  has 344 result records, 0/344 contain `<think>` tags. The usercustomize.py shim
  (scripts/_wpbench_pth/usercustomize.py:39, injects chat_template_kwargs enable_thinking=False)
  WAS active during wp-bench generation. The wp-bench harness was NOT subject to the
  think-kwarg bug. Therefore the generation regression (execution corr 0.292 vs baseline 0.417,
  wp-bench score 0.3716 vs 0.4537) is a GENUINE adapter-vs-generation capability tradeoff,
  not a harness confound.

## Root Cause Report

### RC-A: Judge Parse Regression (CONFIRMED — explains parse_failure_rate and Spearman collapse)

**Root cause: eval_judge harness does not pass `enable_thinking=False` to the vLLM endpoint
when evaluating merged Qwen3 models, causing unclosed `<think>` blocks that make judge output
unparseable. This is an evaluation harness bug. The adapter and weight merge are correct.**

**Mechanism (step by step):**

1. wp-reasoning-v3 adapter was trained under Tinker using `qwen3_disable_thinking` renderer.
   This renderer calls vLLM with `chat_template_kwargs: {enable_thinking: False}`, which causes
   the Qwen3 chat template to prepend `<think>\n\n</think>\n\n` to every assistant turn.
   The model therefore learned to produce structured judge output DIRECTLY (with thinking
   already "done" by the empty pre-filled block). All Tinker-sampled responses have 0 think
   tags; the model was NEVER exposed to the pattern of opening and closing a real `<think>` block.

2. When the merged model is served by vLLM and eval_judge queries it WITHOUT
   `chat_template_kwargs: {enable_thinking: False}`, the Qwen3 chat template outputs only
   `<|im_start|>assistant\n` with no think pre-fill. The merged model (which now contains the
   reasoning adapter's weights) defaults to emitting `<think>` — but because it was trained on
   pre-filled empty think blocks, it never learned the `</think>` close pattern. It outputs all
   of its judge reasoning inside an unclosed `<think>` block.

3. eval_judge's parse_judge_response strips `<think>.*?</think>` with `re.DOTALL` — a
   non-greedy match that requires the closing tag. With no `</think>` present, the entire
   response remains with the `<think>` prefix, and:
   - For responses truncated at max_tokens (1024 tokens): JSON is incomplete, all 4 strategies
     fail -> `excluded["parse_fail"]` incremented. Rate = ~24-30% of examples.
   - For responses that complete naturally (end with `</judge_output>`): Strategy 4
     (`re.search(r'\{.*\}', text, re.DOTALL)`) DOES find the JSON -> parse succeeds.
     These are counted as "passed" but with sparse dim coverage (see RC-A secondary: E6).

4. The parse failure rate (0.190 v3, 0.248 v4) is proportional to the fraction of examples
   where the model's judge output exceeds max_tokens=1024. The Spearman collapse (0.2626 ->
   0.056 v3, 0.153 v4) is caused by: (a) excluding 23-30 pairs from the Spearman calculation
   (fewer pairs with worse coverage) and (b) sparse dimension mapping for remaining pairs (E6).

5. **The adapter IS working correctly. The weight merge IS mathematically correct.** The
   Tinker-runtime offline eval (E3) confirms 0 parse failures and Spearman 0.2626 — identical to
   the baseline — with the SAME adapter weights applied via LoRA-runtime. The judge degradation is
   100% explained by the missing `enable_thinking=False` kwarg in the eval harness.

**Note on merged-weight judge behavior:** E3 is the Tinker-runtime (adapter, not merged weights)
eval. A merged-weight judge eval with thinking-off has not yet been run. The assumption is that
the merged model with `enable_thinking=False` will reproduce E3's result (~0.2626 Spearman).
This is the confirming re-run direction.

**Files implicated:**
- `eval/eval_judge.py` line 691: missing `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- `eval/output_parsers.py` lines 36-41: `_JSON_FIELD_TO_DIM` secondary mismatch (security vs
  security_score etc.) — symmetric bias, see E6

### RC-B: WP-bench Generation Regression (CONFIRMED as real, cause partially open)

**Root cause: The reasoning adapter's attention and/or MoE weight deltas genuinely degrade
code-generation quality — this is a real adapter-vs-generation capability tradeoff, not a
harness artifact.**

**Evidence:** E8 confirms the usercustomize.py shim was active during wp-bench generation
(0/344 records have `<think>` tags). The wp-bench harness correctly disabled thinking mode.
The execution correlation drop (0.417 -> 0.292) and wp-bench score drop (0.4537 -> 0.3716)
happened under correct inference conditions. There is no harness confound for this regression.

**Most likely mechanism:** The reasoning adapter trained on judge tasks modified attention
q/k/v/o projections (and MoE expert weights) in ways that shift the model's code-generation
token distribution. WP-specific PHP generation patterns learned during base training are
partially overwritten by the reasoning fine-tune. This is the standard "catastrophic interference"
phenomenon in multi-task LoRA fine-tuning.

**What remains open:** The exact component(s) responsible (attention vs MoE vs both) and the
magnitude of each component's contribution. A component-ablation run (attention-only merge vs
MoE-only merge) would isolate the source, but requires separate human-gated GPU work.

### Fix Direction

**For RC-A (harness fix — no new merge required):**
- Add `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to all vLLM API calls
  in `eval/eval_judge.py` (`_run_eval_reasoning` line 691, and `run_eval` line 382).
- Add bare-key aliases to `eval/output_parsers.py` `_JSON_FIELD_TO_DIM`:
  `"security": "D2_security"`, `"performance": "D4_perf"`, `"i18n": "D6_i18n"`,
  `"accessibility": "D7_a11y"` — to capture the 4 high-weight dims previously dropped.
- Re-run REVL-01A on the existing v3 merged model with these harness fixes to confirm
  merged-weight Spearman recovers to ~0.2626 (matching Tinker-runtime E3).
- Cost: ~10-line code change + ~1h REVL-01A re-run on GB10. No new merge needed.

**For RC-B (adapter regression — human-gated decision required):**
- Option 1: Accept the tradeoff (judge capability added, code-gen degraded ~8 points).
- Option 2: Retrain adapter with lower LoRA rank (r=8 or r=16 instead of r=32) to reduce
  interference surface.
- Option 3: Retrain adapter targeting judge-specific layers only (exclude attention or limit
  to specific MoE experts).
- Option 4: Run component-ablation merges (attention-only, MoE-only) to isolate which
  component drives the generation regression, then exclude or constrain that component.
- Each option requires a new training or merge run (~GPU-heavy). Separate human gate required.

## Resolution

- root_cause: TWO independent root causes. RC-A: eval_judge harness missing
  enable_thinking=False kwarg causes unclosed think blocks -> parse failures and Spearman
  collapse. Harness bug only; adapter and merge are correct. RC-B: reasoning adapter
  attention/MoE deltas genuinely degrade wp-bench execution correlation (0.417->0.292).
  Confirmed real by E8 (wp-bench shim was active, 0/344 think tags, not a harness confound).
  Secondary harness issue: _JSON_FIELD_TO_DIM key-name mismatch drops 4 high-weight dims.
- fix: RC-A: add chat_template_kwargs enable_thinking=False to eval_judge.py lines 691+382,
  add bare-key aliases to _JSON_FIELD_TO_DIM, re-run REVL-01A on existing v3 staged model.
  RC-B: human-gated decision — retrain with lower rank / targeted modules, or accept tradeoff.
- verified: RC-A not yet applied (Tinker-runtime E3 = reference for expected outcome after fix).
  RC-B confirmed real via E8; root component (attn vs MoE) not yet isolated.

## RC-A APPLIED + CONFIRMED (2026-06-10)

- FIX (commit b88faa3): eval/eval_judge.py `_judge_create()` helper passes
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` at both judge call sites
  (run_eval:382 + _run_eval_reasoning:691), with a LOUD warn-on-fallback (strip_think cannot
  rescue an unclosed <think>, so a silent kwarg-drop must not masquerade as green). 61 eval tests
  pass. Deferred the _JSON_FIELD_TO_DIM dim-map change deliberately: it is SYMMETRIC (affected E3
  too), so leaving it keeps the re-run a clean single-variable test against E3.
- CONFIRM RUN (scripts/_04.4_revl01a_v3_confirm.py, output/eval_reasoning_v3/revl01a_v3_rcA_confirm.json):
  re-ran REVL-01A judge census on the EXISTING v3 staging model through the patched harness.
  - smoke 15: parse_fail 0/15, Spearman 0.2593
  - full 121: parse_fail **3/121 = 0.0248** (<=0.05 PASS; was 0.190), overall Spearman **0.2446**
    (delta -0.018 vs E3 0.2626, within ±0.03; baseline 0.2678), n_pairs 118
  - VERDICT: **RC-A CONFIRMED** — parse recovered AND Spearman ~= E3. The adapter + weight merge
    judge path are correct; the parse gate that disqualified plans 07/08 was harness-induced.
- RESIDUAL (minor): Spearman 0.2446 sits ~0.018 below E3 0.2626 — within tolerance, attributable
  to the 3 remaining parse-fails (118 vs 121 pairs) + small merged-weight bf16 numerical drift.
  Not pursued; immaterial to the gate.

## RC-B ATTRIBUTION (2026-06-10) — MoE carries the codegen damage

Two cheap probes, NO full ablation:

1. Per-expert delta-norm check (no GPU): deltas UNIFORM across all 128 experts (entropy 0.9990,
   top16-mass 0.1424, CV 0.096) -> no judge-expert subset -> MoE-SUBSET salvage DEAD.
2. Single-component wp-bench probe (WPBENCH_LIMIT=30, anchored): built attn-only + MoE-only merges
   (both lm_head-excluded), ran wp-bench subset vs baseline_v2 + v3_full anchors.
   Artifact: output/eval_reasoning_probe_dit02/dit02_attribution_result.json
   | model       | overall | correctness(exec) | knowledge |
   | baseline_v2 | 0.4857  | 0.375             | 0.633     |
   | v3_full     | 0.3810  | 0.292             | 0.500     |
   | attn-only   | 0.4429  | 0.375 (=baseline) | 0.533     |
   | MoE-only    | 0.4071  | 0.3125            | 0.533     |
   subset gap 0.105 ~= known full-set gap 0.082 (anchored). overall: MoE damage 0.079 (75% of gap)
   vs attn 0.043 (41%). CORRECTNESS (the RC-B execution signal) is unambiguous: attn-only execution
   = baseline EXACTLY (0.375), MoE-only = 0.3125 ~ v3_full. => **MoE per-expert deltas carry the
   codegen/execution damage; attention deltas are codegen-safe.** (Caveat: 30-task subset, noisy,
   but consistent across overall + execution + anchored.)

JUDGE-SKILL LOCATION (2026-06-10, SETTLED): judge census (parse+Spearman, RC-A-fixed harness) on
the two variants -> artifact dit02_judge_location_result.json:
  | variant   | parse_rate | Spearman |
  | attn-only | 1.000 (121/121 FAIL) | None (0 pairs) — NO judge ability |
  | MoE-only  | 0.066 (8/121)        | 0.3124 — BEST of all (> v3_full 0.2446, > baseline 0.2678) |
=> **Judge skill lives ENTIRELY in the MoE deltas.** Attention-only yields 100% unparseable judge
output (zero judge skill). MoE-only judges better than the full merge. So judge skill AND codegen
damage are BOTH in MoE — entangled in one component. Attention deltas are NET-HARMFUL: they add
codegen damage (0.043) and slightly LOWER judge Spearman (v3_full 0.2446 < MoE-only 0.3124) while
contributing no judge skill. (Auto-verdict said "INCOMPLETE" only because attn Spearman is None;
the None IS the signal — attention has no judge ability.)

CONSEQUENCE: "cut MoE" would kill the judge. The retrain cannot remove MoE; it must make the MoE
judge-training less codegen-destructive (lower MoE rank/LR, more wp_gen replay) AND can drop the
attention target entirely (net-harmful). MoE-only merge (Spearman 0.3124 / codegen 0.4071) beats
v3_full on both axes but still < baseline codegen 0.4537 -> retrain required.

## RESOLUTION

- RC-A: RESOLVED (harness fix shipped + empirically confirmed).
- RC-B: ROOT-ATTRIBUTED to MoE per-expert deltas (codegen-safe attention). Path chosen (human):
  attribution probe -> Phase-4.3 RETRAIN. Retrain direction: cut MoE interference — lower MoE LoRA
  rank and/or fewer expert layers (and/or lean attention-heavier since attention is codegen-safe),
  then re-merge + re-gate REVL-04. Open sub-question: confirm judge skill survives reduced-MoE
  (measure on the retrained candidate). The lm_head / attempt-2(q_proj) merge-variant track is moot.
