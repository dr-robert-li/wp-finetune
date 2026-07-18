# Pitfalls Research — v4.0 Rerun on Qwen/Qwen3.6-35B-A3B

**Domain:** Fine-tuning/serving/quantizing pipeline rerun on a new MoE base (256-expert + shared expert,
hybrid Gated-DeltaNet/Gated-Attention, VL checkpoint) — Tinker LoRA SFT → merge → vLLM eval on DGX Spark
GB10 (aarch64/Blackwell sm_121) → optional Sieve/prune → llama.cpp Q8 GGUF packaging.
**Researched:** 2026-07-12
**Confidence:** MEDIUM-HIGH (primary sources: GitHub issues/discussions on QwenLM/Qwen3.6, vllm-project/vllm,
ggml-org/llama.cpp; cross-checked against `.planning/V4-RERUN-ROADMAP.md` locked findings)

**Note on scope:** this file supersedes the prior (2026-03/04) `PITFALLS.md`, which covered the v1.0/v1.2
pipeline on Qwen3-30B-A3B. This version is scoped to the v4.0 base-swap milestone per the research prompt.
General pipeline pitfalls not specific to this base swap are not repeated here — see git history for the
prior content if needed.

This file **re-verifies** the two locked items from the roadmap's "architecture-delta work items" section
and Stage-by-stage map, and adds newly-found community-reported issues as of 2026-07-12.

---

## Verification Status of Roadmap-Locked Items

| Locked claim (V4-RERUN-ROADMAP.md) | Status as of 2026-07-12 | Source |
|---|---|---|
| eos/pad token-ID mismatch, QwenLM/Qwen3.6 discussion #96 | **CONFIRMED still open, no code fix shipped — it's maintainer-classified "working as intended," not a bug.** Manual alignment is the only fix. | [discussion #96](https://github.com/QwenLM/Qwen3.6/discussions/96) |
| VL checkpoint `model.language_model.*` key prefix on merge/serve | **CONFIRMED**, and found to interact with a second prefix mode (`--language-model-only` remaps `model.layers.*` → `language_model.model.layers.*` at vLLM load time) — two different prefix conventions depending on merge-time vs. serve-time flag, not one. | vLLM recipes docs, HF model card |
| Q4-nf4 uniform quant router collapse "applies to any MoE base with a router" | **NOT YET independently confirmed on Qwen3.6-35B-A3B specifically** — no GitHub issue or community report found measuring router collapse on *this* model. The claim is a reasonable extrapolation from the Qwen3-30B-A3B measurement, reinforced by a *different* community quant strategy (APEX) that independently arrived at "protect shared expert, compress routed experts hardest" — but that is corroborating design intuition, not a repro of the failure mode on this base. **Re-verify empirically at Gate 2 of Stage 5, don't inherit blind.** | APEX quant card (mudler/Qwen3.6-35B-A3B-APEX-GGUF), no direct router-collapse report found |

---

## Critical Pitfalls

### Pitfall 1: eos/pad token mismatch is WAI, not a bug — silent SFT/generation corruption if untreated

**What goes wrong:**
`tokenizer.eos_token_id` (248046) ≠ `model.config.eos_token_id` (248044); `model.config.pad_token_id` is
`None`. If the SFT trainer falls back to `model.config.pad_token_id` for masking, or falls back to
`model.config.eos_token_id` for loss-stop / generation-stop, it uses the wrong ID — sequences may not stop
where the tokenizer thinks they should stop, or padding tokens get real gradient signal.

**Why it happens:**
Per the maintainer (KOKOSde, response dated 2026-03-24 on discussion #96): `model.config.eos_token_id` is
"mostly a default" and `generation_config.eos_token_id` is what generation actually follows — Qwen
intentionally ships both IDs (`model.generation_config.eos_token_id = [248046, 248044]`) for
thinking/non-thinking mode flexibility. This is deliberate multi-stop-token design, not an oversight, and
the maintainer has said it will not be "fixed" upstream. Any tooling (Tinker, HF `Trainer`,
`merge_adapter.py`) that reads only `model.config.*` instead of `tokenizer.*` or `generation_config.*` will
silently pick up the wrong ID.

**How to avoid:**
Explicit one-time alignment step (already scoped as Stage 1.5 in the roadmap): set
`model.config.eos_token_id = tokenizer.eos_token_id` and `model.config.pad_token_id =
tokenizer.pad_token_id` before Stage 2/3 SFT starts, and verify Tinker's LoRA training client actually
reads the tokenizer's IDs (not the raw config) for loss masking. Audit every place in the pipeline that
reads `model.config.eos_token_id`/`pad_token_id` directly instead of through the tokenizer.

**Warning signs:**
Generation that runs to max_tokens instead of stopping at a natural boundary; SFT loss that includes
padding tokens (loss doesn't converge as expected, or converges suspiciously low because pad content is
learned); eval harness truncation false-negatives that look like carry-forward-lesson-1 (8192-token cap
issue) but are actually a stop-token issue.

**Phase to address:** Phase 20 (Stage 1.5 gate, already scoped) — this pitfall is the reason that gate
exists; do not skip it as "trivial."

---

### Pitfall 2: vLLM has no first-party GB10/sm_121(a) support — DeltaNet CUDA-graph capture crashes are a distinct second failure mode on top of that

**What goes wrong:**
Two stacked problems on this exact hardware/architecture combo:
1. **Platform-level:** as of early 2026, vLLM does not officially support DGX Spark's `sm_121` Blackwell
   target. The stock NVIDIA vLLM container (`nvcr.io/nvidia/vllm:26.01-py3`) ships vLLM 0.13.0, which
   predates Qwen3.5/3.6 support entirely — a nightly build or source build with
   `TORCH_CUDA_ARCH_LIST` including `12.1`/`12.1a` is required just to get the model loading.
2. **Op-level, specific to Gated-DeltaNet layers:** vLLM issue
   [#35945](https://github.com/vllm-project/vllm/issues/35945) (reported 2026-03-04, vLLM
   `0.16.1rc1.dev197+g9a9d44246`) — `AssertionError: assert num_cache_lines >= batch` in
   `causal_conv1d_update` when CUDA-graph capture runs with a GDN (Gated DeltaNet) layer and
   `conv_state_indices` is supplied. Root cause: the assertion conflates the conv-state cache-pool size
   (4-6 for GDN layers) with the batch dimension; large-batch graph capture (e.g. batch 512) trips
   `4 >= 512` and fails. No confirmed upstream fix as of this search — treat as open. This is
   architecture-specific to the DeltaNet/linear-attention op family, so it will recur on Qwen3.6-35B-A3B's
   30 DeltaNet-MoE layers regardless of the sm_121 platform issue being solved separately.

**Why it happens:**
GB10/sm_121(a) is new silicon that landed after most of the CUDA-graph-capture code paths for linear
attention kernels were written and tested only against SM90/SM100 assumptions; the causal_conv1d kernel's
batch-vs-cache-lines assumption was never audited for the indices-provided code path.

**How to avoid:**
- Do not use the stock NGC vLLM container; build vLLM from source (or use a maintained community fork,
  e.g. `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash`'s 7-patch set) with `TORCH_CUDA_ARCH_LIST=12.1a` for
  native SASS, not PTX JIT.
- Run the DeltaNet-op smoke check (already scoped in Phase 20) with CUDA-graph capture **enabled**, not
  just eager mode — the #35945 failure only manifests during graph capture, so an eager-mode-only smoke
  test would give a false pass.
- If graph capture fails, fall back to `--enforce-eager` for the smoke/eval pass rather than blocking the
  whole rerun on an upstream vLLM fix; note the throughput cost.
- Reduce `--gpu-memory-utilization` from the default 0.90 to 0.80 — community guides (adadrag/qwen3.5-dgx-spark)
  report 0.90 is unstable for long-running sessions on GB10, 0.80 is the stable community-verified value.

**Warning signs:**
vLLM server crash or hang specifically at server startup during CUDA-graph capture (not at first request);
works fine with `--enforce-eager` but fails with graph capture on.

**Phase to address:** Phase 20 (DeltaNet-on-aarch64 op smoke check — already scoped). Extend that check's
acceptance criteria to explicitly run WITH CUDA-graph capture on, not just a bare load/generate smoke.

---

### Pitfall 3: VL checkpoint has two different key-prefix conventions depending on merge-time vs. serve-time path

**What goes wrong:**
`merge_adapter.py` and Tinker's adapter export need to target `model.language_model.*` keys because the
checkpoint is a VL (vision-language) model even though this project only trains/serves the text pathway.
Separately, at **serve** time, vLLM's `--language-model-only` flag does its own remap
(`model.layers.*` → `language_model.model.layers.*`) to load only the text-only weight subset. If the merge
step and the serve step assume the same prefix convention, weights silently fail to load into the right
module (either erroring loudly, which is the good case, or worse — partially loading and producing garbage
generations, which is the bad case if shapes happen to coincidentally match on some layers).

**Why it happens:**
Qwen3.6-35B-A3B ships as a VL checkpoint (vision tower + MTP head + shared LM backbone) even for the "text
model" use case (per the roadmap's Stage 5 sizing note: 67.0 GiB/checkpoint bf16 full vs. ~65.2 GiB with
the tower excluded via `--language-model-only`). Generic HF-style adapter-merge tooling written for a plain
`CausalLM` checkpoint assumes a flat `model.layers.*` key space and doesn't know about the
`language_model.` wrapper a VL config introduces.

**How to avoid:**
Explicitly test the merge → serve round-trip on Phase 20 (already scoped as "VL merge-path check"), not
just the merge step in isolation. Verify with a real generation smoke test after merge, served through the
SAME flag combination that Stage 2/3/4 will actually use (`--language-model-only` if text-only serving is
the plan). Do not assume "merge succeeded because the shapes loaded" is sufficient — shape-compatible
partial loads are the dangerous case, not the error case.

**Warning signs:**
Merge script runs to completion with no errors but the merged model's generations look untrained
(base-model-like output, adapter effect not visible) — classic silent partial-load symptom.

**Phase to address:** Phase 20 (already scoped: "VL merge-path check"). Explicitly extend the check to
include a served-generation smoke, not just a merge-completes-without-error check.

---

### Pitfall 4: LoRA on multimodal models is a narrower target than on the dense/MoE text-only case

**What goes wrong:**
vLLM currently only supports adding LoRA to the language-model portion for multimodal models — vision-LoRA
support is separate and immature (a community-reported vLLM bug,
[#28640](https://github.com/vllm-project/vllm/issues/28640), shows an `AssertionError` in `lora_shrink_op`
specifically for Qwen3-VL multimodal LoRA loading). Since this project's LoRA targets are text-only MoE
layers anyway (per the existing recipe: MoE-only LoRA rank 32, frozen router, frozen attn/unembed), this is
likely a non-issue in practice — but only if the training/merge/serve path never accidentally routes
through a vision-LoRA code path.

**Why it happens:**
The base checkpoint being VL-shaped means any tooling that auto-detects "is this a multimodal model" (some
vLLM/PEFT code paths do) may take a different, less-mature branch than the plain-text-LoRA path, even when
no vision LoRA weights exist.

**How to avoid:**
Confirm Tinker's adapter export and `merge_adapter.py` are operating purely in the text/LM code path
(no vision tower touched at all during SFT — text-only training data, no image tokens). At serve time,
confirm `--language-model-only` is set BEFORE LoRA is loaded, not after, so vLLM never initializes the
vision-LoRA code path.

**Warning signs:**
Any error mentioning `lora_shrink_op`, vision tower shapes, or `mm_processor` during what should be a pure
text serving/training path.

**Phase to address:** Phase 20 (VL merge-path check) and Phase 21/22 (SFT gen/judge) — confirm the training
run never touches vision-tower code.

---

### Pitfall 5: shared expert must be excluded from BOTH quantization aggressiveness AND Sieve/pruning — same principle, two different tools

**What goes wrong:**
The always-on shared expert (1 of 256+1) is architecturally different from the 8-of-256 routed experts:
it processes every token, so any quality loss to it hits 100% of traffic, not the ~3% traffic share a
single routed expert sees. The roadmap's Sieve/protected-mask tooling adaptation (work item 1, Phase 25)
already accounts for this on the pruning side. The SAME principle needs to be independently re-verified on
the **quantization** side (Stage 5 / Phase 28) — it is a distinct pipeline stage with distinct tooling
(GGUF conversion, not the Sieve profiler), so "we already handled shared-expert protection in Phase 25"
does NOT automatically cover Phase 28.

**Why it happens:**
Community MoE-GGUF quant strategies (e.g. the APEX quant card for this exact model family) independently
converged on "keep shared expert tensors at high precision, compress routed experts hardest, exploit the
~97% sparsity from only 8-of-256 routed experts being active" — this is corroborating evidence the
principle is architecture-general, but it is evidence from a THIRD PARTY's quant recipe, not a
verification that llama.cpp's default GGUF quant type assignment (or whatever tool this project's
packaging phase uses) does the same thing out of the box. Default/naive uniform quantization would NOT
know to special-case the shared expert.

**How to avoid:**
When selecting the GGUF quant recipe for Stage 5, explicitly check (not assume) that whatever
quantization tool/imatrix recipe is used treats the shared-expert tensor differently from routed-expert
tensors — either by using one of the community MoE-aware quant strategies (Unsloth Dynamic 2.0 UD- quants,
or the APEX-style role-aware precision gradient) rather than a naive uniform per-layer-type quant, or by
manually verifying the shared-expert tensor's assigned quant type in the resulting GGUF metadata.

**Warning signs:**
Quality regression concentrated across ALL outputs (not just edge cases) after quantization — a symptom of
shared-expert damage, distinct from the "collapses on rare inputs" symptom of routed-expert damage.

**Phase to address:** Phase 28 (Stage 5 packaging) — treat as a distinct checklist item from the Phase 25/26
Sieve shared-expert exclusion, even though the principle is the same.

---

### Pitfall 6: chat template emits empty `<think></think>` blocks in historical turns — prompt-drift/cache risk, and a latent SFT-data-formatting trap

**What goes wrong:**
QwenLM/Qwen3.6 GitHub issue [#131](https://github.com/QwenLM/Qwen3.6/issues/131) (opened 2026-04-09,
**appears still open**, no visible maintainer fix): the chat template renders an empty `<think>\n\n</think>`
wrapper for historical assistant turns even when `reasoning_content` is empty/absent, because the template
guard (`{%- if loop.index0 > ns.last_query_index %}`) doesn't check for actual reasoning content before
emitting the wrapper. Documented consequences are serving-side (prompt instability, prefix-cache
invalidation, wasted compute in multi-turn) — but the same template is very likely what any SFT data
assembly step uses to render training examples into the model's expected format. If this project's SFT
data pipeline renders judge/gen examples through the stock Qwen3.6 chat template (rather than hand-building
the prompt string), the SAME empty-think-block insertion could occur in **training** examples, not just
serving. Separately, a related issue in a different tool
([earendil-works/pi #3325](https://github.com/earendil-works/pi/issues/3325)) shows the template drops
prior-turn `<think>` content entirely unless `preserve_thinking=true` is explicitly passed in
`chat_template_kwargs` — a second, independent template footgun for any multi-turn formatting.

**Why it happens:** Template logic bug (missing `and reasoning_content` guard), not fixed upstream as of
this search.

**How to avoid:**
- Audit whatever renders the project's SFT training examples into Qwen3.6 chat-template format: if it goes
  through `tokenizer.apply_chat_template()`, check the rendered output for spurious empty `<think></think>`
  blocks in judge-mode or multi-turn examples, especially in the reasoning-mix SFT data (Phase 21) which by
  construction includes CoT/reasoning content — verify the loss mask does not accidentally include an
  EMPTY think block as if it were real reasoning content to imitate.
- If the project's existing single-turn, task-token-routed (`<wp_gen>`/`<wp_judge>`) format bypasses the
  chat template entirely (likely, given the existing multi-format export system predates this base), this
  pitfall may not apply — but that assumption should be explicitly verified for THIS base's tokenizer/
  template, not carried over unverified from the old base.
- If any multi-turn judge examples are used, pass `preserve_thinking=true` explicitly rather than relying
  on template defaults.
- Related, non-template-bug but same theme: the standard Qwen3-thinking distillation recipe masks all
  tokens between `<think>` and `</think>` out of the loss when the training target is not itself a
  reasoning trace — without that mask, ~70% of tokens in a reasoning-mix example can be thinking-trace
  tokens that pull the student toward imitating reasoning STYLE rather than the answer distribution. Verify
  the Phase 21 reasoning-mix SFT recipe's loss mask explicitly handles this for whatever think-tagging
  convention this base's tokenizer uses.

**Warning signs:** SFT loss curves with unusual token-count-vs-example-length ratios in judge-mode
examples; generated judge outputs containing visible empty `<think></think>` artifacts post-training.

**Phase to address:** Phase 21/22 (SFT data formatting, before training starts) — add a template-render
spot-check to the existing Stage 1.5-adjacent verification pass, not a new phase.

---

### Pitfall 7: llama.cpp GDN/hybrid-architecture support is version-sensitive — pin, don't assume "latest main" works

**What goes wrong:**
Multiple llama.cpp issues on Qwen3.5/3.6-35B-A3B GGUF loading and inference, spanning several months of the
model's early life:
- [#19903](https://github.com/ggml-org/llama.cpp/issues/19903) "unknown model" — early llama.cpp lacked
  the GDN/hybrid op registration for this architecture family at all.
- [#19857](https://github.com/ggml-org/llama.cpp/issues/19857) — fails to load Unsloth-converted GGUF on
  some llama.cpp builds (also discussed at
  [huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/62](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/62)).
- [#19915](https://github.com/ggml-org/llama.cpp/issues/19915) — `GGML_ASSERT(hparams.n_pos_per_embd() ==
  1 ...)` failure in `llama-kv-cache.cpp` specifically when `seq_add()`/multi-sequence (router/batch mode)
  is used with this model.
- [#22135](https://github.com/ggml-org/llama.cpp/issues/22135) — HIP/Windows crash loading a Q6_K GGUF
  around 120K context (memory exhaustion, not aarch64-specific, but a caution against assuming any
  quant-tier + long-context combo is safe without a smoke test).
- [#22425](https://github.com/ggml-org/llama.cpp/issues/22425) — Vulkan backend crash at ~45K tokens.
- [#23011](https://github.com/ggml-org/llama.cpp/issues/23011) — self-MTP speculative decoding much slower
  than baseline on Apple Metal (not this project's target backend, but signals the MTP head's speculative
  path is immature across backends generally — worth a quick check on whether MTP is enabled by default in
  whatever conversion tool is used, since an unintentionally-enabled slow MTP path could look like a
  packaging regression rather than a known upstream perf issue).

**Why it happens:** This is a young model family (Qwen3.5/3.6 hybrid DeltaNet+Attention+MoE+MTP); llama.cpp
support matured incrementally across several releases, and different backends (CUDA/HIP/Vulkan/Metal) hit
different edge cases at different times.

**How to avoid:** Pin a specific llama.cpp commit/release known-good for Qwen3.6-35B-A3B GDN + GGUF
conversion (check the bartowski/unsloth GGUF model cards for the llama.cpp version they built against, and
match it) rather than building against "main" unpinned. Run the GGUF conversion + a real-generation smoke
test (carry-forward lesson 2) on the CUDA backend specifically (this project's actual serving backend),
not assume CUDA is unaffected because the reported issues are HIP/Vulkan/Metal.

**Warning signs:** Conversion succeeds but load fails with an "unknown model" or GGML_ASSERT error; works
in single-sequence mode but fails specifically when multiple concurrent requests/sequences are served
(router mode) — matches the pair-serving use case this project's Stage 5 packaging targets.

**Phase to address:** Phase 28 (Stage 5 packaging). Add "confirm llama.cpp version pin + smoke-test the
CUDA backend under concurrent-sequence load" as an explicit precondition before the Q8 GGUF conversion pass,
since concurrent pair-serving (the whole reason Q8 is mandatory per the roadmap's memory finding) is
exactly the router/multi-sequence code path #19915 flags as fragile.

---

## Moderate Pitfalls

### Pitfall 8: Tinker's live model table training-context cap (64K) is already accounted for, but re-check if data assembly changes

**What goes wrong:** Tinker caps `Qwen/Qwen3.6-35B-A3B` training context at 64K tokens (verified fact,
already carried in the roadmap's Stage 2 line) versus the model's native 262K. Currently irrelevant because
wp-gen/judge training examples are function-level PHP + rubric text, far under 64K — but if any future data
assembly change (e.g. adding longer multi-file context, or long CoT judge traces) pushes example length
up, this cap becomes a silent truncation risk rather than a hard training-time error (unclear from search
whether Tinker truncates silently or errors on overflow).

**Prevention:** Add a one-line assertion in the data-loading step of Phase 21/22 that verifies max
tokenized example length is well under 64K before training starts, so this doesn't silently regress if the
dataset composition changes later in v4.0 or a future milestone.

### Pitfall 9: `trust_remote_code=True` required for both model AND tokenizer/processor; intermittent worker-init failures reported

**What goes wrong:** Loading requires `trust_remote_code=True` on both `AutoModelForCausalLM`/
`AutoModelForImageTextToText` and `AutoTokenizer`/`AutoProcessor` — a plain `AutoTokenizer.from_pretrained()`
without the flag will fail or silently mis-load custom tokenizer logic. Separately, vLLM issue
[#40249](https://github.com/vllm-project/vllm/issues/40249) reports intermittent `KeyError: 'qwen3_5_moe'`
from `transformers.CONFIG_MAPPING` during multi-worker init — a race/ordering issue in how the custom
architecture registers itself across worker processes, not a config content bug.

**Prevention:** Set `trust_remote_code=True` everywhere in the base bring-up scripts (Phase 20) and, if
using vLLM tensor-parallel with multiple workers, retry-on-failure or add an explicit warm registration
step before the actual serving smoke test — this project's single-GPU DGX Spark deployment likely uses a
single worker process, which reduces (does not eliminate — sub-process spawning still occurs) the surface
for this race.

### Pitfall 10: Tinker LoRA-target guidance says "all layers including MoE MLP" — no reported DeltaNet-specific LoRA-target unsupported error found, but unverified for this base

**What goes wrong:** No GitHub issue, discussion, or blog post was found specifically reporting that
Tinker's LoRA implementation rejects or mishandles DeltaNet/gated-linear-attention layers as LoRA targets.
Thinking Machines' own "LoRA Without Regret" guidance says results are best applying LoRA broadly,
including MoE MLP layers — consistent with this project's existing MoE-only LoRA recipe. This is an absence
of evidence, not evidence of absence: DeltaNet layers are architecturally distinct enough (linear-attention
gating params, not standard Q/K/V) that a target-module string mismatch (e.g. `target_modules` list
written for standard attention naming) could silently apply LoRA to zero DeltaNet parameters rather than
erroring.

**Prevention:** After the Phase 20 bring-up, before committing to the full Stage 2/3 SFT budget, print/log
the actual list of modules LoRA attached to and confirm it includes the expected MoE-MLP (and, if intended,
DeltaNet-gating) parameter names for this base's specific module naming — don't assume the target-module
config that worked on Qwen3-30B-A3B's naming scheme transfers unchanged.

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---|---|---|
| 1. eos/pad token mismatch (WAI, not upstream-fixed) | Phase 20 (Stage 1.5 gate) | Assert `model.config.{eos,pad}_token_id` == tokenizer's after alignment; smoke-generate confirms correct stop behavior |
| 2. vLLM sm_121 + DeltaNet CUDA-graph-capture assertion | Phase 20 (DeltaNet-on-aarch64 smoke) | Smoke test MUST run with CUDA-graph capture enabled, not eager-only; fallback to `--enforce-eager` documented if #35945 unresolved |
| 3. VL checkpoint dual key-prefix (merge-time vs. serve-time) | Phase 20 (VL merge-path check) | Merge → serve round-trip with real generation smoke, not merge-completes-without-error only |
| 4. Multimodal LoRA code-path narrower support | Phase 20/21/22 | Confirm no vision-tower/vision-LoRA code path is touched during text-only SFT/merge/serve |
| 5. Shared-expert protection must be re-verified at quantization, separately from Sieve | Phase 28 (Stage 5 packaging) | Inspect GGUF tensor-type metadata for the shared-expert tensor; confirm it is NOT uniformly quantized with routed experts |
| 6. Chat-template empty `<think>` blocks (serving AND possibly SFT data) | Phase 21/22 (SFT data formatting) | Spot-check rendered training examples for spurious empty think blocks and confirm think-span loss masking before training starts |
| 7. llama.cpp GDN-support version sensitivity | Phase 28 (Stage 5 packaging) | Pin llama.cpp commit matching bartowski/unsloth's build; smoke-test CUDA backend under concurrent-sequence (router) load |
| 8. Tinker 64K training-context cap | Phase 21/22 | Assert max tokenized example length << 64K before training |
| 9. `trust_remote_code` + intermittent worker-init KeyError | Phase 20 | Set flag everywhere in bring-up scripts; retry-on-failure for multi-worker init if TP>1 is ever used |
| 10. LoRA target-module naming unverified for DeltaNet layers | Phase 20/21 | Log actual LoRA-attached module list post-init; confirm MoE-MLP (and DeltaNet-gating, if intended) names match expectation |

---

## "Looks Done But Isn't" Checklist

- [ ] **eos/pad alignment step:** Often "looks done" because the SFT run doesn't error — verify by
  inspecting `model.config.eos_token_id`/`pad_token_id` directly post-alignment, not just "training ran to
  completion."
- [ ] **DeltaNet smoke check:** Often run in eager mode only because that's the default fast path — verify
  CUDA-graph capture was explicitly exercised, since that's the mode that actually fails (#35945).
- [ ] **VL merge:** Often "looks done" because the merge script exits 0 — verify with an actual served
  generation, since partial/shape-coincidental loads fail silently.
- [ ] **Q8 GGUF shared-expert handling:** Often assumed inherited from the Sieve/prune work (Phase 25) —
  verify independently in the packaging tool's own tensor-type assignment, since it's a different tool.
- [ ] **LoRA target-module list:** Often assumed unchanged from the old base's config — verify the actual
  attached-module log against this base's naming scheme.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---|---|---|
| eos/pad mismatch discovered post-SFT | LOW | Re-run generation-stop smoke test with corrected config; if training already happened on wrong pad masking, re-tokenize/re-run — cheap since Stage 2/3 SFT runs are ~$2-6 (Tinker) |
| DeltaNet CUDA-graph crash discovered mid-eval | LOW-MEDIUM | Fall back to `--enforce-eager`, document throughput hit, don't block the eval gate on an upstream vLLM fix |
| VL merge silent partial load discovered post-eval (garbage outputs) | MEDIUM | Re-run merge with corrected key-prefix handling; re-run affected eval arm only |
| Shared-expert over-quantized discovered post-packaging | LOW-MEDIUM | Re-run GGUF conversion with corrected quant recipe/imatrix; conversion is ~1h/model per roadmap cost anchor |
| LoRA silently applied to wrong/zero DeltaNet modules discovered post-SFT | MEDIUM | Re-run SFT with corrected target-module list — same ~$2-6/run cost as the original mistaken run, but adds calendar time |

## Sources

- [QwenLM/Qwen3.6 discussion #96](https://github.com/QwenLM/Qwen3.6/discussions/96) — eos/pad mismatch, maintainer response 2026-03-24
- [QwenLM/Qwen3.6 issue #131](https://github.com/QwenLM/Qwen3.6/issues/131) — empty think-block chat-template bug, opened 2026-04-09, open
- [QwenLM/Qwen3.6 discussion #55](https://github.com/QwenLM/Qwen3.6/discussions/55) — disabling think mode
- [vllm-project/vllm issue #35945](https://github.com/vllm-project/vllm/issues/35945) — GDN causal_conv1d_update CUDA-graph-capture AssertionError, reported 2026-03-04
- [vllm-project/vllm issue #28640](https://github.com/vllm-project/vllm/issues/28640) — multimodal LoRA `lora_shrink_op` AssertionError
- [vllm-project/vllm issue #40249](https://github.com/vllm-project/vllm/issues/40249) — intermittent `KeyError: 'qwen3_5_moe'` in multi-worker init
- [vllm-project/vllm issue #36275](https://github.com/vllm-project/vllm/issues/36275) — Qwen3.5 4B vLLM compatibility issues
- [ggml-org/llama.cpp issue #19903](https://github.com/ggml-org/llama.cpp/issues/19903) — unknown model (early GDN support gap)
- [ggml-org/llama.cpp issue #19857](https://github.com/ggml-org/llama.cpp/issues/19857) — fails to load Unsloth GGUF
- [ggml-org/llama.cpp issue #19915](https://github.com/ggml-org/llama.cpp/issues/19915) — `seq_add()`/multi-sequence GGML_ASSERT
- [ggml-org/llama.cpp issue #22135](https://github.com/ggml-org/llama.cpp/issues/22135) — HIP/Win11 Q6_K crash at ~120K context
- [ggml-org/llama.cpp issue #22425](https://github.com/ggml-org/llama.cpp/issues/22425) — Vulkan backend crash ~45K tokens
- [ggml-org/llama.cpp issue #23011](https://github.com/ggml-org/llama.cpp/issues/23011) — self-MTP slower than baseline on Metal
- [AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash](https://github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash) — 7-patch DGX Spark GB10/sm_121a vLLM build guide
- [adadrag/qwen3.5-dgx-spark](https://github.com/adadrag/qwen3.5-dgx-spark) — DGX Spark GB10 vLLM install/troubleshooting guide (gpu-memory-utilization 0.80 stability finding)
- [NVIDIA Developer Forums: Custom built vLLM + Qwen3.5-35B on DGX Spark](https://forums.developer.nvidia.com/t/custom-built-vllm-qwen3-5-35b-on-nvidia-dgx-spark-gb10-sustained-50-tok-s-1m-context/362590)
- [vLLM Recipes: Qwen/Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B) — `--language-model-only` flag behavior, weight-prefix remap
- [mudler/Qwen3.6-35B-A3B-APEX-GGUF](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF) — role-aware quant strategy, shared-expert high-precision rationale
- [bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF), [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — quant tier sizes, Unsloth Dynamic 2.0 MoE-aware quant notes
- [earendil-works/pi issue #3325](https://github.com/earendil-works/pi/issues/3325) — `preserve_thinking` chat-template kwarg trap
- [Thinking Machines: LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — LoRA target-module guidance (apply broadly, including MoE MLP)
- `.planning/V4-RERUN-ROADMAP.md` — locked findings this file re-verifies (dated 2026-07-11/12)

---
*Pitfalls research for: v4.0 rerun of wp-finetune pipeline on Qwen/Qwen3.6-35B-A3B*
*Researched: 2026-07-12*
