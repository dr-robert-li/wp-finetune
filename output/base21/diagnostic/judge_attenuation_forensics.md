# Phase 21 Diagnostic — JUDGE-03 Capture→Serve Attenuation Forensics

**Question:** why does the SAME judge checkpoint (`wp-judge-v4-s1-ep3`, seed 1) score
rho **0.8358** via Tinker `SamplingClient` capture but rho **0.7872** via
local-merge + vLLM serve, on the identical 121-item `openai_val.jsonl` wp_judge
set, both at `max_tokens=8192`, `temperature=0`?

Read-only analysis. No models touched, no GPU used. All numbers below are
directly reproducible from committed artifacts plus two throwaway scripts
(paths noted per section).

---

## 1. Per-item paired diff

Scored both captures with the unmodified `scripts/relabel/eval_relabel.py`
parser (`eval.output_parsers.parse_judge_scores` + `_derive_prose_overall`),
paired by `index` (both files enumerate the same 121 `<wp_judge>`-prefixed
rows in the same order — verified via `wj_rows` filter identity).

| Metric | Value |
|---|---|
| n common (both parsed) | 121 / 121 (0 parse failures either side) |
| n_identical_scores (exact) | 60 / 121 (49.6%) |
| n within ±1 pt | 62 / 121 |
| n within ±5 pt | 110 / 121 (90.9%) |
| mean \|Δscore\| | 1.90 pts (of 0–100 scale) |
| median \|Δscore\| | 1.0 pt |
| max \|Δscore\| | 26 pts (idx 84) |
| mean **signed** Δ (tinker − vllm) | 0.00 — no systematic score-level bias |
| mean overall score, tinker / vllm | 70.16 / 70.16 (identical means) |
| mean rank shift (Spearman rank position, n=121) | 7.73 positions |
| items with rank shift > 20 positions | 14 / 121 |

**Reading:** there is no uniform bias (means match exactly, most items agree
closely). The rho gap is driven by a **small, high-leverage subset** of items
whose reasoning/verdict genuinely flips — not by a constant offset across all
121 items.

**Paired bootstrap of the rho delta** (2000 resamples, same seed=7 percentile
bootstrap as `eval_relabel.py`, resampling row indices jointly so both rho
values are computed on the *same* resample each draw):

```
rho_tinker (on common 121) = 0.8358
rho_vllm   (on common 121) = 0.7872
paired Δrho median          = 0.0480   CI95 = [0.0131, 0.0934]
fraction of resamples where tinker > vllm = 99.9%
```

The gap is **directionally robust** (tinker beats vllm in 1994/2000 resamples)
but the CI is wide (n=121) — see §4 for how this compares to the historical
attenuation baseline.

**Leverage check** — how much of the gap is carried by the worst-divergence
items:

| Excluding top-K \|Δ\| items | n | rho_tinker | rho_vllm | gap |
|---|---|---|---|---|
| K=0 (full set) | 121 | 0.8358 | 0.7872 | 0.0486 |
| K=5 | 116 | 0.8280 | 0.8044 | **0.0236** |
| K=10 | 111 | 0.8256 | 0.7989 | 0.0267 |
| K=14 (rank-shift>20 set) | 107 | 0.8277 | 0.8029 | 0.0249 |
| K=30 | 91 | 0.8220 | 0.8054 | 0.0166 |

Dropping just **5 of 121 items** (4%) roughly **halves** the gap (0.049 →
0.024). This is the signature of a small number of borderline items tipping
over a decision boundary, not a broad-based quality regression from serving.

Script: `scripts/relabel/eval_relabel.py`-equivalent paired loader
(ad-hoc, `/tmp/.../judge21_paired_diff.py` — not committed, reproducible from
the snippet embedded in this report's git history / session transcript).

---

## 2. Output-text diff

Sampled the 5 worst-divergence items (idx 84, 78, 93, 5, 29) and compared
full text side-by-side.

**Pattern found: mid-response semantic divergence, not late-token numeric
jitter and not identical-text/different-score.** For most of the 121 items,
the two generations are close paraphrases of each other covering the same
observations (same code facts noticed, same overall assessment) — this
accounts for the 110/121 "within 5 pts" bucket. But for the worst items, the
two generations diverge into **materially different reasoning conclusions**
already within the first paragraph, and that early divergence propagates to
a different final verdict/score. Example (idx 84, tinker=46 vs vllm=72,
GT=53):

- **Tinker** opens: *"The REST controller registers multiple CREATABLE
  endpoints with permission_callback set to `__return_true`... all of which
  perform state-changing operations... without any authentication or
  aut[horization]"* → concludes `verdict: FAIL`, `security: 4`,
  `overall_score: 46`.
- **vLLM** opens on the same code but pivots to a mitigating detail not
  foregrounded by tinker: *"...Any unauthenticated request can mark a
  cours[e]... [but] logged in and has a legitimate enrollment relationship to
  the course"* → concludes `verdict: PASS`, `security: 6`,
  `overall_score: 72`.

Both completions are well-formed, both close the `<judge_output>{...}</judge_output>`
JSON envelope correctly, both parse without error. This is decisive against
a **parser bug** (both sides parse cleanly and consistently) and against
**pure late-token numeric jitter** (the divergence is not "same reasoning,
slightly different final digits" — it is a different reasoning path chosen
from early on). It is consistent with either (a) a real generation-fidelity
difference between the two serving backends acting on effectively the same
weights, surfacing at a genuine model decision-boundary item, or (b) prompt
non-determinism — ruled out in §3.

idx 5 and idx 29 show the same pattern at smaller magnitude: same code, same
general assessment, but a different specific WPCS/SQL-safety detail gets
foregrounded, shifting 1–2 sub-dimension scores by 1-2 points each, which
compounds into a 6-8 point overall delta via `_derive_prose_overall`'s
weighted sum.

---

## 3. Protocol audit

Read `scripts/capture_judge_responses_tinker.py` (Tinker path) and
`scripts/build_judge03_merge_serve.py` + `scripts/sieve_capture_judge_http.py`
(vLLM path) line by line. Findings:

| Dimension | Tinker capture | vLLM-served capture | Match? |
|---|---|---|---|
| `max_tokens` | 8192 | 8192 | ✅ |
| `temperature` | 0.0 | 0.0 | ✅ |
| dataset / row filter | `wp_judge_startswith`, same index enumeration | same enumeration (`sieve_capture_judge_http.capture`) | ✅ (by construction, both index-align to `wj_rows`) |
| prompt construction | `tinker_cookbook.renderers.get_renderer("qwen3_5_disable_thinking").build_generation_prompt(user_msgs)` | `openai.chat.completions.create(messages=user_msgs, extra_body={"chat_template_kwargs":{"enable_thinking": False}})` against vLLM's own `chat_template.jinja` from the merged model dir | **Verified byte-identical** — see below |
| stop sequences | explicit `stop=renderer.get_stop_sequences()` = `[248046]` (`<\|im_end\|>` token id) | none passed; relies on vLLM/model `generation_config.json` EOS | Cosmetic only — see below |
| system prompt / extra tokens | none injected (user-message-only) | none injected (user-message-only, per `sieve_capture_judge_http.py` docstring) | ✅ |
| `enable_thinking` guard | renderer is inherently no-think (empty `<think></think>` header injected, zero loss weight, confirmed in `gen01_format_decision.json`) | `_judge_create`'s RC-A `enable_thinking=False` `chat_template_kwargs` guard | ✅ same effective behavior |

**Rendered-prompt byte-diff (the decisive check for this section).** Built
the idx=84 prompt both ways under `.venv-tinker` (tinker_cookbook installed
there, merged model checkpoint available locally at
`models/Qwen3.6-35B-A3B-judge-v4-s1-merged`):

```
tinker_cookbook renderer text  vs.  merged-model chat_template.jinja
(tokenize=False, enable_thinking=False, add_generation_prompt=True)

TEXT IDENTICAL: True
```

Both render to the exact same string, ending
`...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` — the empty
think-block insertion noted in `gen01_format_decision.json` is present and
identical on both paths. **This rules out template/renderer mismatch as the
cause of the divergence.** The two capture paths feed the model
character-for-character the same input.

Stop-sequence handling is a real but cosmetic difference: Tinker's decoded
text *includes* the literal `<|im_end|>` suffix (its `tok.decode()` doesn't
strip special tokens), while vLLM's OpenAI-compatible response strips it —
but both terminate at the same generation boundary and both parse cleanly,
so this is not a contributor to the score divergence.

**bf16 merge precision (new finding, not previously called out in the code
review).** `scripts/merge_adapter.py` loads the base model via
`AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, ...)` and
merges the LoRA delta directly into bf16 weights (no fp32 accumulation
noted in the merge path). Tinker's `SamplingClient` samples from the
LoRA-adapted checkpoint via Tinker's own cloud inference stack, which
applies the adapter differently (whether at fp32-accumulated add or via a
separate forward-pass composition is not verifiable from this repo, but it
is a *different code path* than `merge_adapter.py`'s literal bf16 weight
merge). Merging ~240 target modules of LoRA deltas into bf16 base weights
introduces per-parameter rounding that a decompose-and-apply-at-inference
path does not necessarily incur identically. This, plus vLLM's distinct
attention/MoE kernel implementation vs Tinker's serving stack, is the most
plausible source of the small per-token logit perturbations needed to flip
a greedy-decoding (`temperature=0`) argmax at a low-margin token — which,
once flipped, cascades through the rest of the autoregressive generation
(exactly the pattern seen in §2).

---

## 4. Old-base precedent comparison

The project's own history (`.planning/V4-RERUN-ROADMAP.md`) records the old
base (Qwen3-30B-A3B, v1.3 judge) attenuating from a **0.8274** Tinker-capture
promotion figure to a **0.8017** vLLM-served single-seed shipping figure
(Δ=0.0257), i.e. this capture→serve gap is a **known, pre-existing
phenomenon on this project**, not new to the v4.0 base.

Recomputed directly, paired, on the same 121 items, using the actual old-base
raw captures available locally:

- Old-base Tinker capture: `output/relabel/eval_s1_ep3/judge_responses.jsonl`
  (seed-1, ep3, `max_tokens=2048`) → rho = **0.8274** (matches the recorded
  figure exactly).
- Old-base served (closest available matched artifact — llama.cpp bf16,
  `max_tokens=8192`, 0 parse failures):
  `output/packaging/ens8192/bf16_s1/judge_responses.jsonl` → rho = **0.7888**.
  (Note: this is *not* the exact historical 0.8017 vLLM figure, which traces
  back further to `output/sieve/optimal_k.json`, Phase 11, whose raw jsonl no
  longer exists locally; the `pkg03_ens8192_results.json` doc itself
  cross-validates that the 8192-token llama.cpp bf16 ensemble (0.8100) and
  the historical vLLM ensemble reference (0.8075) agree within noise, so this
  substitution is a reasonable same-methodology proxy, with the caveat that
  the *capture* side here is still capped at 2048 tokens — no old-base Tinker
  capture at 8192 exists in this repo to construct a perfectly matched pair.)

```
OLD BASE:  rho_capture=0.8274  rho_served=0.7888  raw_delta=0.0386
           paired-bootstrap Δ median=0.0375  CI95=[0.0067, 0.0749]
NEW BASE:  rho_capture=0.8358  rho_served=0.7872  raw_delta=0.0486
           paired-bootstrap Δ median=0.0480  CI95=[0.0131, 0.0934]
```

**The two bootstrap CIs overlap almost entirely** (old CI upper 0.0749 sits
well inside new CI, new CI lower 0.0131 sits inside old CI). **0.049 vs
0.026–0.039 is NOT statistically distinguishable at n=121.** The new base's
attenuation is not categorically worse than the old base's — it is the same
known phenomenon, sampled toward the high end of the same noise band. Given
n=121 and a seed-noise floor the project has independently measured at
~0.052 (`gate1_bf16_baseline.json`), a ~0.01-0.02 difference between two
attenuation estimates is well within expected sampling variance.

---

## 5. Verdict — ranked hypotheses

| Rank | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | **Merge/serving numerics (bf16 LoRA-merge rounding + Tinker-vs-vLLM kernel differences), amplified by greedy-decoding butterfly effect** | **Most likely** | Prompt is byte-identical (§3) — rules out template cause. Divergence is early/mid-response semantic (§2), not late-digit jitter or garbled parsing. Gap is carried by 5/121 high-leverage items, not a uniform bias (§1) — consistent with a few borderline items crossing a decision boundary because of small logit perturbations. `merge_adapter.py` merges LoRA deltas into bf16 weights (§3), a known source of small numeric drift vs. non-merged adapter application. |
| 2 | **Serving-template mismatch** | **Ruled out** | Direct byte-for-byte rendered-prompt comparison: `TEXT IDENTICAL: True` (§3). Both paths give the model the exact same input string and token stream. |
| 3 | **Parser bug** | **Ruled out** | 0/121 parse failures on both captures; worst-divergence items show clean, well-formed, differently-*reasoned* JSON on both sides, not malformed output being mis-extracted (§2). |
| 4 | **Noise / non-determinism at temp=0 alone** | **Partially explanatory but not sufficient alone** | Cannot fully explain a 0.049 gap that is 99.9%-directionally-consistent under paired bootstrap (§1) — true independent noise would show ~50/50 sign flips across resamples, not 99.9% one-directional. The directionality itself needs a systematic (if small) source, which points back to hypothesis #1 rather than pure sampling noise. |
| 5 | **Label/GT quality** | **Out of scope for this gap** | Both captures are scored against the identical GT label set; a GT problem would depress both rho figures equally, not create a directional capture-vs-serve split. |

**Top hypothesis:** the merged-and-vLLM-served model is numerically a very
close but not bit-identical realization of the Tinker-sampled checkpoint;
under greedy decoding, a handful of long chain-of-thought items sit close
enough to a token-level decision boundary that this numeric drift flips the
argmax, and the flip cascades into a different reasoning path and verdict
for ~5/121 items — enough to move rho by ~0.02-0.05, which is exactly the
same order of magnitude as the project's own historical old-base
capture→serve gap (§4).

**One cheap experiment to confirm:** serve the **unmerged** base model with
the LoRA adapter attached natively via vLLM's `--enable-lora` /
`--lora-modules` flag (pointing at the already-downloaded
`output/base21/judge03_s1_adapter/` directory — no new merge, no new
training, no new download), and re-run the *identical*
`sieve_capture_judge_http.capture()` + `eval_relabel.py` pipeline against
that endpoint. Three possible outcomes, each diagnostic:

1. **Unmerged-LoRA-via-vLLM rho ≈ 0.8358 (matches Tinker capture)** → the
   divergence is specifically in `merge_adapter.py`'s bf16 weight-merge step
   (hypothesis 1a: merge fidelity). Fix: merge in fp32 and downcast once at
   the end, or serve un-merged in production.
2. **Unmerged-LoRA-via-vLLM rho ≈ 0.7872 (matches merged-vLLM serve)** → the
   divergence is generic vLLM-vs-Tinker inference-kernel numerics
   (hypothesis 1b: engine numerics), independent of merging. No merge-side
   fix would help; this would be an inherent capture-methodology vs.
   shipping-methodology gap to simply accept and budget for (as the project
   already does for the old base, per §4).
3. **Unmerged-LoRA-via-vLLM rho lands in between** → both effects contribute
   partially; report the split.

This costs one vLLM boot + one 121-item capture (minutes, no training spend,
no re-merge, no re-download) and directly discriminates the two remaining
live sub-hypotheses under rank 1.
