# Stack Research: v4.0 Rerun on Qwen/Qwen3.6-35B-A3B

**Domain:** Toolchain re-verification for an existing, locked fine-tuning pipeline swapping base models
**Researched:** 2026-07-12 (re-verifying claims locked 2026-07-11 in `19-NEXT-BASE-SELECTION.md` / `V4-RERUN-ROADMAP.md`)
**Confidence:** HIGH on existence/version claims (primary sources fetched live), MEDIUM on pricing-table column mapping and GB10 throughput numbers (fetch-tool summarization variance across repeated queries — see notes)

This file supersedes the prior (v1.2-era, 2026-04-04) `STACK.md` for this project. It does NOT re-litigate
the base-model *selection* (that lock stands — see item 6). It answers: what toolchain versions/flags/
gotchas does the v4.0 requirements pass need for Qwen/Qwen3.6-35B-A3B specifically.

---

## Re-verification of the 6 locked/flagged claims

### 1. Tinker: Qwen3.6-35B-A3B LoRA support, 64K context cap, pricing tier

**VERIFIED, with one correction and one new finding.**

- `Qwen/Qwen3.6-35B-A3B` is a live row in Tinker's model table, type "Hybrid + Vision", arch MoE, size
  Medium, **training context cap 64K** (confirmed independently by WebSearch snippet and two separate
  WebFetch passes of `tinker-docs.thinkingmachines.ai/tinker/models/` on 2026-07-12). Matches the locked
  doc exactly. Irrelevant blocker either way — WP function-level SFT examples are far under 64K.
- **Correction to the locked doc's pricing labels.** The 2026-07-11 doc wrote "LoRA pricing train/sample/eval
  $0.36 / $0.89 / $1.07." Direct re-fetch of the raw table today shows columns **Prefill $0.36 (cached
  $0.072) / Sample $0.54 (cached $0.108) / Train $1.07**, identical for both `Qwen/Qwen3.6-35B-A3B` and
  `Qwen/Qwen3.5-35B-A3B-Base`. The $0.89 figure does not appear anywhere in today's table for either row —
  it may have been a transient value, a different comparison row, or a column-order misread in the prior
  pass. The **conclusion still holds** ("same per-unit price tier, no cost-class jump between the two rows
  compared") but cite Prefill $0.36 / Sample $0.54 / Train $1.07 going forward, not $0.89.
- **New finding not in the locked doc: a Tinker-wide price increase lands 2026-07-17**, five days after
  this research pass. Per the page's own notice: "we are also increasing our prefill and sample prices by
  ~50% and our train prices by ~10% starting July 17." This applies uniformly across the table (not
  Qwen3.6-specific), so it does not change the "same tier" comparison, but it does mean: if v4.0 sign-off
  and Stage 2/3 Tinker spend happen after 2026-07-17, budget **~10% higher train cost** than the
  roadmap's $2/run and ~$6/3-seed anchors (train is the dominant cost driver for SFT runs; prefill/sample
  are minor by comparison for this workload).
- **New finding, more consequential: `Qwen3-30B-A3B` and `Qwen3-30B-A3B-Base` (the CURRENT project's base)
  were retired from Tinker on 2026-06-12** — "can no longer be used for training or inference." This
  doesn't block v4.0 (no plan to retrain the old base), but it retroactively validates the base-swap
  timing: staying on the old base was about to become untrainable on this vendor regardless. The
  fallback candidate `Qwen/Qwen3.5-35B-A3B-Base` remains live and unaffected (only the non-base
  `Qwen3.5-35B-A3B` instruct row was retired, matching the locked doc's existing note that the fallback
  has "Base" type only).

### 2. vLLM: serves on aarch64/GB10, `--language-model-only` drops the vision tower

**VERIFIED.**

- `vllm>=0.19.0` is the vendor-recommended minimum on the model's own HF README (re-confirmed 2026-07-12,
  matches the locked doc).
- The `--language-model-only` flag is documented in vLLM's own recipe page
  (`recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B`) with a working example: `vllm serve Qwen/Qwen3.6-35B-A3B-FP8
  --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.60 --language-model-only`.
- GB10 specifically: an NVIDIA Developer Forums thread ("Qwen/Qwen3.6-35B-A3B (and FP8) has landed - DGX
  Spark / GB10", posts dated 2026-04-16 through 04-20) shows the model already running on DGX Spark with
  `vllm 0.19.1rc1.dev337+g17d87168d.d20260416`, `--gpu-memory-utilization 0.7-0.8`, reporting 7,800+
  tok/s in some aggregate-throughput configuration. This is independent of, and predates, the project's own
  in-repo precedent (`CHANGELOG.md` D-03) cited in the locked doc — two independent confirmations the base
  already serves on this exact host class.
- **New, more important finding not in the locked doc:** HuggingFace's own `transformers` docs for this
  architecture (`qwen3_5_moe` model page) report that **GB10 (compute capability 12.1 / SM121) has no
  prebuilt `causal_conv1d` or `fla` kernel** — the Gated-DeltaNet linear-attention path silently falls back
  to a slower, more memory-hungry pure-PyTorch reference implementation unless `use_kernels=True` is passed
  to `from_pretrained()` (requires `pip install -U kernels`, and currently `trust_remote_code=True` because
  the community kernel repo `Atlas-Inference/gdn` isn't yet on the trusted-kernels allowlist). Measured
  numbers on `Qwen/Qwen3.6-35B-A3B` bf16, GB10/SM121, 1024-token prompt, greedy decode of 256 tokens:

  | `use_kernels` | TTFT (prefill) | Decode |
  |---|---|---|
  | `False` (PyTorch fallback, default) | 0.73 s | 16.3 tok/s |
  | `True` (`Atlas-Inference/gdn` Hub kernel) | 0.53 s (1.38x faster) | 16.7 tok/s |

  Decode throughput is roughly flat between the two paths (the single-token DeltaNet recurrence is
  memory-bandwidth-bound, not compute-bound) — the win is prefill-side and grows with prompt length. This
  applies to `transformers`-native inference paths (eval harness, HF-side smoke tests); vLLM's own kernel
  stack may differ, but this is the first hard aarch64/GB10-specific throughput evidence for this
  architecture and directly de-risks the roadmap's "DeltaNet-aarch64 op smoke check" item — upgrade its
  status from "inferred-OK" to **measured-OK with a known slow-path caveat**. Recommend the Phase 20
  bring-up smoke test explicitly try `use_kernels=True` and record whether the trusted-kernels flag is
  acceptable for this project's threat model, since eval-harness wall-clock (already budgeted at ~19
  min/wp-bench arm) is sensitive to per-token decode speed at this architecture's ~16 tok/s single-stream
  ceiling.

### 3. llama.cpp: supports the architecture for GGUF conversion (hybrid DeltaNet layers)

**VERIFIED, with a minimum-build-number correction and a caveat.**

- Mainline `llama.cpp` supports the architecture as `qwen35moe`. `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
  (the project's own preferred quantizer, per the v3.0 precedent) explicitly states **release b9222** was
  used to produce its quants, and that MTP-layer support landed in **b9180**. This is a harder, more
  specific floor than the locked doc's general "ecosystem check" language — record **llama.cpp >= b9180**
  as the practical minimum for this model's GGUF path (b9222+ if MTP-head quantization is wanted too).
- Community history matters here: search results also surfaced an earlier community fork
  (`tekintian/llama.cpp`) and in-flight PRs (#20700 for dense-variant MTP) that predate mainline support
  landing. This mirrors the pattern already flagged for DeltaNet-on-GB10 (item 2) — support existed as a
  community patch before merging upstream — but as of 2026-07-12 mainline support is confirmed live via
  the bartowski repo's own build references, so this is not a blocker, just a "don't use a stale llama.cpp
  checkout" reminder.
- **Caveat (MEDIUM confidence — surfaced via GitHub issue search, not independently reproduced):** at least
  one open llama.cpp issue reports a GGUF conversion/load block-count mismatch on a Qwen3.5-family hybrid
  MoE variant (`ggml-org/llama.cpp#24737`, "Qwen3.5-4B: GGUF conversion/load expects 33 blocks, model only
  has 32"). Not confirmed against the 35B-A3B checkpoint specifically, but same architecture family and same
  `convert_hf_to_gguf.py` code path — worth a smoke-check on block count immediately after conversion in
  Stage 5, before trusting the quantized artifact.
- Positive supporting detail: bartowski's repo notes Q4_0 "online repacking" improvements specifically for
  ARM performance — relevant since GB10 is aarch64.

### 4. Existing quantized checkpoint ecosystem: bartowski GGUF, unsloth GGUF, NVIDIA NVFP4, QuantTrio AWQ

**VERIFIED, all four confirmed live 2026-07-12.**

| Provider | Repo | Format |
|---|---|---|
| bartowski | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` | GGUF (Q8/Q6/Q5/Q4 ladder) |
| Unsloth | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | GGUF (Dynamic 2.0 quantization) |
| Unsloth | `unsloth/Qwen3.6-35B-A3B-NVFP4`, `-NVFP4-Fast` | NVFP4 |
| Unsloth | `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` | MLX (Apple Silicon, not relevant to GB10) |
| NVIDIA | `nvidia/Qwen3.6-35B-A3B-NVFP4` | NVFP4 (official, Blackwell-native) |
| QuantTrio | `QuantTrio/Qwen3.6-35B-A3B-AWQ` | AWQ 4-bit |

New since the lock: Unsloth published **NVFP4 quants on 2026-07-10** (two days before this research pass),
claiming ~1.7x speedup on 32GB VRAM. Also new: multiple independent third-party AWQ repos beyond QuantTrio
(`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, `mattbucci/Qwen3.6-35B-A3B-AWQ`) and a comparative GGUF quality
benchmark (`localbench.substack.com`, KL-divergence ranking across unsloth/bartowski/lmstudio-community/
ggml-org/mudler/AesSedai quants) — useful if Stage 5's quantization ladder wants a second opinion beyond
bartowski's own numbers. **Caveat found, not in the locked doc:** Qwen3.6 GGUFs reportedly do **not** work
in Ollama due to separate `mmproj` (vision projector) files that Ollama's loader doesn't yet split
correctly — irrelevant to this project (vLLM/llama.cpp direct, not Ollama-mediated) but worth knowing if
anyone reaches for Ollama as a quick local-serve shortcut during Stage 5 dev-looping.

### 5. transformers/PEFT version requirements for the VL checkpoint (`model.language_model.*` prefix)

**VERIFIED (architecture/class names + key prefix), UNVERIFIABLE (exact minimum transformers version floor).**

- Load classes confirmed directly from HF's own model doc page: `Qwen3_5MoeForConditionalGeneration` (VL,
  takes `vision_config` + `text_config`, this is the class for the shipped checkpoint) and
  `Qwen3_5MoeForCausalLM` (text-only variant, used in HF's own example against the non-VL
  `Qwen3-Next-80B-A3B-Instruct` sibling — confirms the codebase's text-only causal LM path exists for this
  model family, though the actual `Qwen/Qwen3.6-35B-A3B` checkpoint itself ships as the VL
  `ForConditionalGeneration` class per Axis 1 of the locked selection doc).
- `model.language_model.*` key prefix for the text backbone under the VL wrapper is corroborated by a
  community fine-tuning guide (Medium, "Fine-Tuning Qwen/Qwen3-VL-30B-A3B MoE Architecture with LoRA") — LoRA
  target modules for the sibling Qwen3-VL family use `model.visual.blocks`/`model.visual.merger` for the
  vision side and `model.language_model.layers.*.mlp.{gate,up,down}_proj` for the text side. This is
  **MEDIUM confidence** (a third-party blog post, not an official HF/Qwen doc) but internally consistent
  with the official HF `Qwen3_5MoeConfig` schema (separate `text_config`/`vision_config` sub-objects) — the
  prefix split is architecturally required, not just a convention, so the risk of this being wrong is low.
- **New, actionable finding for the SFT recipe, not in the locked doc:** HF's usage notes for this exact
  architecture state "When training or fine-tuning, set `output_router_logits=True` so the forward returns
  router logits and the load-balancing auxiliary loss is added to the total loss... Without it, experts can
  collapse to a few popular slots." This is a direct, named failure mode (router/expert collapse) relevant
  to Stage 2/3's MoE-only LoRA recipe — the requirements pass should make `output_router_logits=True` (and
  checking `router_aux_loss_coef`, default `0.001`) an explicit SFT config item, not an assumed default.
- **Exact minimum transformers version: UNVERIFIABLE via the tools used.** The HF docs page for this
  architecture is served at a `/v5.13.1/` doc path (i.e., `transformers` is on major version 5 by
  2026-07-12), and the model doc exists and is populated, meaning **some transformers >= 5.x release
  supports `Qwen3_5MoeForConditionalGeneration`** — but no changelog/release-notes page was fetched to pin
  the exact minimum. What was tried: fetching the model doc page (succeeded, confirms existence and class
  names) and a targeted search for "transformers PEFT Qwen3.6 model.language_model key prefix" (returned
  corroborating community content, not a version pin). **Recommendation for the requirements pass:** pin to
  latest `transformers` 5.x at execution time and verify `Qwen3_5MoeForConditionalGeneration` imports
  successfully as part of the Phase 20 bring-up smoke test, rather than trusting a version number
  transcribed here.

### 6. Newer Qwen release that would challenge the base lock

**Checked — nothing found that should reopen the lock.**

- **Qwen3.7 exists but is proprietary, not open-weight.** Qwen3.7-Max announced 2026-05-19
  ("Qwen3.7: The Agent Frontier," qwen.ai blog), Qwen3.7-Plus reached GA 2026-06-01, both served only via
  Alibaba Cloud Model Studio / API partners (Fireworks etc.) at $2.50 input / $7.50 output per M tokens.
  No open-weight release has been announced for either tier as of 2026-07-12. Since v4.0 requires local
  LoRA fine-tuning on downloaded weights (Tinker + on-prem GB10 serving), a closed API-only model cannot
  satisfy the pipeline's requirements regardless of capability — **does not challenge the lock**.
- **No newer Qwen3.6-35B-A3B revision found.** Search for an August-2026-or-later update/checkpoint
  revision returned only the original 2026-04-16 release artifacts (BF16, official FP8, NVIDIA NVFP4) —
  no evidence of a re-upload, patch, or "-2" revision as of this research date.
- **Conclusion: no action.** The locked base stands; nothing materially better shipped in the open-weight,
  same-size-class space between 2026-07-11 (lock date) and 2026-07-12 (this research pass).

---

## Toolchain summary for the requirements pass

| Component | Version / flag | Status | Source confidence |
|---|---|---|---|
| Tinker | `Qwen/Qwen3.6-35B-A3B`, 64K train context cap | VERIFIED | HIGH (live docs, 3 independent fetches) |
| Tinker pricing | Prefill $0.36 / Sample $0.54 / Train $1.07 (rising ~10-50% from 2026-07-17) | VERIFIED + price-hike alert | HIGH |
| vLLM | `>=0.19.0`, `--language-model-only` | VERIFIED | HIGH (vendor README + recipe page + 2 independent GB10 deployment reports) |
| GB10 DeltaNet kernel | no native SM121 build; `use_kernels=True` + `pip install -U kernels` for 1.38x prefill speedup, decode flat ~16 tok/s | VERIFIED (measured) | HIGH (official HF docs, includes numbers) |
| llama.cpp | mainline `qwen35moe` support, **minimum build b9180** (MTP b9180+, quant tooling tested b9222) | VERIFIED, more specific than lock | HIGH (bartowski's own repo) |
| llama.cpp GGUF block-count | possible off-by-one on Qwen3.5-family hybrid MoE (upstream issue #24737, unconfirmed on 35B-A3B) | FLAGGED, not confirmed | MEDIUM |
| Quant ecosystem | bartowski GGUF, unsloth GGUF/NVFP4/MLX, NVIDIA NVFP4, QuantTrio + 2 more AWQ repos | VERIFIED | HIGH |
| transformers | 5.x (`Qwen3_5MoeForConditionalGeneration`), pin latest at execution, verify via bring-up smoke test | PARTIALLY VERIFIED (class exists; exact min version unpinned) | MEDIUM |
| PEFT / LoRA key prefix | `model.language_model.*` for text-side LoRA targets under VL wrapper | VERIFIED (architecturally required + community-corroborated) | MEDIUM-HIGH |
| SFT config | `output_router_logits=True` required to avoid expert collapse (not in locked doc) | NEW FINDING | HIGH (official HF usage notes) |
| Qwen3.7 / newer base | proprietary, no open weights — does not challenge lock | VERIFIED, no reopen | HIGH |

---

## Integration notes for requirements/roadmap

- **Phase 20 (base bring-up) should gain two smoke-test line items** beyond what's in the roadmap: (a) load
  with `use_kernels=True` and confirm the `Atlas-Inference/gdn` kernel path is acceptable given
  `trust_remote_code=True`, or explicitly decide to accept the ~1.38x-slower prefill fallback; (b) after any
  GGUF conversion in Stage 5, verify block count / tensor count against the safetensors index before trusting
  the quantized artifact (block-count bug flagged above).
- **Stage 2/3 SFT config must set `output_router_logits=True`** (and sanity-check `router_aux_loss_coef`,
  default 0.001) — this is a hard requirement to avoid the exact "experts collapse to a few popular slots"
  failure mode HF's own docs name, and it was not previously called out anywhere in the locked planning docs.
- **Cost estimates should account for the 2026-07-17 Tinker price increase** if v4.0 sign-off happens after
  that date: train price rises ~10% (dominant driver), prefill/sample ~50% (minor line items for this
  workload). The roadmap's $2/run and ~$6/3-seed anchors likely still round to the same order of magnitude,
  but should be re-derived from the live table at actual spend time, not from either the 2026-07-11 or
  2026-07-12 snapshots in this doc.
- **llama.cpp version pin:** require `>=b9180` explicitly in any environment/setup doc for Stage 5 (the
  locked doc's "ecosystem check" language didn't carry a build-number floor; now it can).
- **transformers version:** do not hardcode a version number from this doc — pin "latest transformers 5.x"
  and let the Phase 20 bring-up smoke test (`from transformers import Qwen3_5MoeForConditionalGeneration`)
  be the actual gate.

## What NOT to add

- No need to add Ollama to the serving stack for this base — its GGUF loader currently mishandles the
  separate `mmproj` vision file for this architecture family; the project's existing vLLM/llama.cpp path
  is unaffected and should stay primary.
- No need to re-open the relabel campaign or the base-model lock based on anything found here — both
  remain sound per the discretion items already resolved in `V4-RERUN-ROADMAP.md`.
- No need to adopt NVFP4 or MLX quant formats for this project — GB10/aarch64 + llama.cpp/vLLM serving
  makes GGUF (bartowski, llama.cpp-native) and AWQ (vLLM-native) the relevant formats; NVFP4 is
  Blackwell-datacenter-oriented and MLX is Apple-Silicon-only, neither displaces the already-locked Q8 GGUF
  ship-tier decision.
- Do not add `causal-conv1d`/`fla` as hard pip dependencies — they have no GB10/SM121 build; the
  `kernels`-package Hub-kernel path (`use_kernels=True`) is the only currently-working acceleration route
  on this hardware, and even that is optional (fallback path works, just slower on prefill).

## Sources

All fetched/searched live on 2026-07-12 unless noted:

- https://tinker-docs.thinkingmachines.ai/tinker/models/ — live pricing/support/retirement table (3 separate fetch passes for cross-checking)
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B — model card, README, architecture
- https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe — architecture classes, GB10/SM121 kernel notes, `output_router_logits` guidance, doc-version path `v5.13.1`
- https://huggingface.co/docs/transformers/model_doc/qwen3_5 — dense-variant sibling doc (cross-check)
- https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B — vLLM `--language-model-only` example command
- https://forums.developer.nvidia.com/t/qwen-qwen3-6-35b-a3b-and-fp8-has-landed/366822 — DGX Spark/GB10 live deployment thread, posts 2026-04-16 to 04-20
- https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF — llama.cpp build-number requirements (b9180/b9222)
- https://github.com/ggml-org/llama.cpp/issues/24737 — flagged block-count issue on Qwen3.5-family hybrid MoE (unconfirmed on 35B-A3B)
- https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ, https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4, https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF, https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4 — quant ecosystem
- https://medium.com/@ishaafsalman/fine-tuning-qwen-qwen3-vl-30b-a3b-moe-architecture-with-lora-2365359e870f — `model.language_model.*` LoRA target-module corroboration (MEDIUM confidence, third-party)
- https://qwen.ai/blog?id=qwen3.7, https://www.marktechpost.com/2026/05/21/qwen-introduces-qwen3-7-max-a-reasoning-agent-model-with-a-1m-token-context-window/, https://www.marktechpost.com/2026/06/02/alibabas-qwen-team-launches-qwen3-7-plus-adding-vision-deep-reasoning-tool-invocation-and-autonomous-iteration-on-the-bailian-platform/ — Qwen3.7 proprietary-tier confirmation
- `.planning/V4-RERUN-ROADMAP.md`, `.planning/phases/19-next-base-rerun-roadmap/19-NEXT-BASE-SELECTION.md` (recovered via `git show HEAD~2:...`) — baseline claims being re-verified
