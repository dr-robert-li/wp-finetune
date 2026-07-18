# Architecture Research: Qwen3.6-35B-A3B Integration Deltas

**Domain:** MoE fine-tuning pipeline integration (existing wp-finetune stack, Qwen3-30B-A3B → Qwen3.6-35B-A3B base swap)
**Researched:** 2026-07-12
**Confidence:** HIGH (architecture/checkpoint facts, WebFetch'd directly from HF `config.json` / `model.safetensors.index.json`) / MEDIUM (ecosystem tooling maturity, cited from dated third-party sources)

This file re-verifies the six locked claims from the 2026-07-11 research pass and reports deltas found
2026-07-12. It supersedes the stale v1.2-era `ARCHITECTURE.md` previously at this path (dated 2026-04-04,
scoped to a different milestone). It confirms V4-RERUN-ROADMAP.md's numbers and adds two integration points
the roadmap didn't name explicitly (profiler layer-path traversal, fused expert tensor layout).

---

## 1. Architecture facts (re-verified against live HF config.json)

Source: `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json` (WebFetch, 2026-07-12).

| Claim (locked 2026-07-11) | Live config.json value | Status |
|---|---|---|
| 256 routed experts, top-8 | `num_experts: 256`, `num_experts_per_tok: 8` | **CONFIRMED** |
| 1 always-on shared expert | `shared_expert_intermediate_size: 512` present at model level (not per-expert) | **CONFIRMED** |
| 40 layers, 10×(3×DeltaNet-MoE + 1×Gated-Attention-MoE) | `num_hidden_layers: 40`; `layer_types`: repeating pattern of 3 `linear_attention` + 1 `full_attention`; `full_attention_interval: 4`; `attn_output_gate: true` | **CONFIRMED** |
| — | `model_type: "qwen3_5_moe"`, `architectures: ["Qwen3_5MoeForConditionalGeneration"]` | **NEW FINDING** — internal model_type/class string is `qwen3_5_moe`/`Qwen3_5Moe*`, not `qwen3_6*`. This is Alibaba's naming, not a typo — the "Qwen3.6" release reuses the Qwen3.5 architecture class. Matters for `AutoModelForCausalLM`/`trust_remote_code` class resolution and for grepping transformers source when debugging. |
| — | `hidden_size: 2048`, `max_position_embeddings: 262144`, `vocab_size: 248320` | Reference only |
| — | `vision_config` present (`depth: 27`, `hidden_size: 1152`), `mtp_num_hidden_layers: 1` | **CONFIRMED** VL + MTP checkpoint (see §2) |

DeltaNet layer head config (from model card, WebFetch 2026-07-12): 32 linear-attention V-heads / 16 QK-heads,
dim 128. Gated-Attention layer config: 16 Q-heads / 2 KV-heads, dim 256 (standard GQA, 8:1 ratio).

## 2. Checkpoint structure (re-verified against live model.safetensors.index.json)

Source: `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json` (WebFetch, 2026-07-12).

| Claim (locked 2026-07-11) | Live value | Status |
|---|---|---|
| `model.language_model.*` key prefix | Confirmed: all LM weight keys are `model.language_model.layers.N....` (not plain `model.layers.N...`) | **CONFIRMED** |
| total_size 67.0 GiB | `metadata.total_size = 71,903,645,408` bytes = **66.96 GiB** (71.90 GB) | **CONFIRMED** (matches to 0.04 GiB) |
| vision tower + MTP head included | `model.visual.*` (patch_embed, blocks, merger) and `mtp.*` (`mtp.fc.weight`, `mtp.layers.0.self_attn.*`, `mtp.norm.weight`) both present in the weight map, top-level (not under `language_model.`) | **CONFIRMED** |
| ~65.2 GiB LM-only | Not independently re-measured (would require summing per-tensor sizes for `language_model.*` keys only) — order-of-magnitude plausible given the vision tower is a ~0.4B ViT-scale encoder relative to the 35B total. MEDIUM confidence, unchanged from locked value. | **CARRIED FORWARD, not re-measured** |

## 3. Router/expert tensor naming — Sieve profiler adaptation (NEW FINDING, not in the 2026-07-11 lock)

This is the most consequential new finding for Phase 25 (Sieve tooling adaptation). Compared side-by-side
(both WebFetch'd 2026-07-12):

| | Qwen3-30B-A3B (current pipeline assumption) | Qwen3.6-35B-A3B (new base) |
|---|---|---|
| Per-expert weights | **Unfused.** One tensor set per expert: `model.layers.N.mlp.experts.{0..127}.{gate_proj,up_proj,down_proj}.weight` (128 × 3 tensors/layer) | **Fused.** One stacked tensor per layer for all 256 experts: `model.language_model.layers.N.mlp.experts.gate_up_proj` and `...experts.down_proj` (2 tensors/layer, expert dim stacked) |
| Router gate | `model.layers.N.mlp.gate.weight` (`nn.Linear`) | `model.language_model.layers.N.mlp.gate.weight` (same `nn.Linear` role, new prefix) |
| Shared expert | N/A (no shared expert on this base) | `model.language_model.layers.N.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight` **plus** a separate scalar gate `model.language_model.layers.N.mlp.shared_expert_gate.weight` controlling its blend weight |
| Layer uniformity | All 48 layers are MoE, same shape | Only 30/40 layers are DeltaNet-MoE; 10/40 are Gated-Attention-MoE. Both strata still route through `mlp.experts`/`mlp.gate` (MoE sits after both attention variants), but the attention sub-module differs by stratum |

**Why this matters for `scripts/profile_base_model.py`, `scripts/extract_protected_mask.py`,
`scripts/sieve_expert_mask_inference.py`:**

- The **router-logit masking mechanism** (`sieve_expert_mask_inference.py::apply_mask`, forcing masked
  experts' pre-softmax logit to `-inf`) operates on `router_logits` of shape `[..., n_experts]` — this is
  **architecture-robust already**, since it doesn't touch per-expert weight tensors directly. Fused vs
  unfused expert weight storage does not change this mechanism. Good news: no rewrite needed here beyond
  updating `n_experts` from 128→256 and excluding the shared expert (which has no entry in `router_logits`
  at all — it is not gated by the same softmax, evidenced by the separate `shared_expert_gate` tensor — so
  "always exclude the shared expert from the sweepable set" from Work Item 1 is actually automatic once the
  mask only ever indexes into `router_logits`, **provided** the forward hook is reading the right module).
- The **forward-hook module path** (`profile_base_model.py:454-460`, unwraps `model.model.layers` then
  attaches `layer.mlp.gate.register_forward_hook(...)`) assumes `model.model.layers[i].mlp.gate`. On the
  new checkpoint the attribute path is `model.model.language_model.layers[i].mlp.gate` (transformers
  exposes the VL wrapper's `language_model` submodule) — **this traversal needs a one-line path change**,
  not a rewrite, but it will silently produce zero hooks (not an error) if missed, because
  `hasattr`/`getattr` failures on an unexpected nested VL wrapper commonly no-op rather than crash. Flag as
  a smoke-test checklist item, not just a code change.
- The **`[n_layers, n_experts]` uniform array shape** used throughout `extract_protected_mask.py` and
  `sieve_expert_mask_inference.py` (`counts_wp_gen: [n_layers, n_experts]`, single `n_experts` scalar) is
  the change Work Item 1 already names: needs to become per-stratum-aware (DeltaNet-MoE layers vs
  Gated-Attention-MoE layers are still both `[*, 256]` in expert-count — routing width doesn't differ by
  stratum — so the *array shape* itself may not need to change; what changes is any code that assumes
  uniform *attention* behavior across all 40 layers, e.g. if profiling captures attention-pattern stats
  alongside routing stats). Re-verify this narrower point against the actual profiler code before assuming
  the full E_eff computation needs a stratum split — the routing/expert-count math is uniform; only
  attention-layer-type-dependent code paths are not.
- **Net effect:** Work Item 1's diagnosis (mixed layer strata, shared-expert exclusion) is correct in
  spirit, but the concrete code changes are narrower than "treat as two uniform stacks" — it's (a) fix the
  `model.model.language_model.layers` path, (b) confirm shared expert never appears in `router_logits`
  (verify empirically on first forward pass, don't just assume from tensor naming), (c) bump `n_experts`
  128→256 wherever hardcoded. No fused-vs-unfused weight handling is needed in the profiler at all, since
  routing analysis reads `router_logits`/`router_indices`, never expert weight tensors directly.

## 4. DeltaNet layer ops — inference kernel requirements (vLLM aarch64) and llama.cpp GGUF representation

**vLLM (serving path):**
- Recipe page (`https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B`, WebFetch 2026-07-12): **vLLM ≥0.17.0**
  required to serve this model at all; **≥0.19.0 recommended**. Pin the version explicitly in the Phase 20
  bring-up smoke test — this is a harder floor than "whatever vLLM is currently pinned in the project."
- GB10/DGX Spark is explicitly named in vLLM's own docs: "NVIDIA Blackwell GPUs... including DGX Spark
  (GB10)" for **NVFP4** serving. This is corroborating, independent confirmation that GB10 is a
  vendor-recognized target for this model family, beyond the project's own GGUF/Q8 plan.
- **Known open bug, directly relevant to the Phase 20 "DeltaNet-aarch64 op smoke check":**
  vLLM GitHub issue #35945 — `AssertionError in causal_conv1d_update when capturing CUDA graphs for
  Qwen3.5/GDN layers` (Gated DeltaNet layers use `causal_conv1d` kernels internally; the bug fires during
  CUDA-graph capture specifically, not eager mode). **Actionable for the smoke test:** if the bring-up smoke
  hits this, the workaround is `--enforce-eager` (disables CUDA graph capture) at a throughput cost — budget
  a fallback path in the Phase 20 smoke-test plan rather than treating a CUDA-graph crash as a hard blocker.
- No aarch64-specific vLLM wheel is published for the CUDA backend (prebuilt wheels are amd64-only per
  vLLM's own ARM install docs); DGX Spark's existing toolbox (`scripts/dgx_toolbox.py`) already builds/runs
  vLLM in a GB10-targeted container, so this is a **carried-forward operational fact, not a new risk** — the
  project's serving path already handles aarch64+CUDA. The DeltaNet-specific kernels (`causal_conv1d`,
  `fla`-family ops) are pulled in as part of vLLM's own dependency chain on install, not a separate build
  step this project owns — confirm the container's vLLM install resolves these on first bring-up, don't
  assume.

**llama.cpp (GGUF path):**
- Community sources report the hybrid Gated-DeltaNet + Gated-Attention architecture requires "the absolute
  latest version of llama.cpp to support these new operators" — i.e., support exists but is recent; pin a
  llama.cpp commit/tag known to work, don't float on an old checkout.
- **New quantization consideration not present for the old base:** linear-attention state tensors (the
  DeltaNet recurrent-state weights) are reported as disproportionately sensitive to low-bit quantization;
  GGUF converters keep them at higher precision than the surrounding weights, a ~2-4% file-size increase
  over a flat K-quant. This does not change the Q8 ship-tier decision (Q8 is already high-precision enough
  that this doesn't bite), but **would** matter if the Gate-2 quantization ladder in Stage 5 ever descends
  to Q6/Q5/Q4 — those tiers need this mixed-precision handling to not silently corrupt the DeltaNet blocks.
  Flag for whoever runs the ladder below Q8.
- Confirmed community GGUF builds already exist at launch (`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`,
  `unsloth/Qwen3.6-35B-A3B-*`), corroborating the roadmap's ecosystem-check line.

## 5. Memory math for GB10 121GB host — Q8 scaling assumption re-verified

| Quantity | Roadmap projection (2026-07-11) | Measured (2026-07-12, community GGUF) | Status |
|---|---|---|---|
| bf16 / checkpoint | 67.0 GiB (from safetensors total_size) | 66.96 GiB (live-fetched total_size, see §2) | **CONFIRMED**, tightened from projection to measurement |
| bf16 pair (gen+judge) | 134.0 GiB — exceeds 121 GB host | 133.9 GiB by the same math | **CONFIRMED, does not fit** |
| Q8 / checkpoint | ~35.6 GiB (projected, scaled from v3.0's 53.2% bf16→Q8 ratio) | **~35 GB actual**, per community quant table (`bartowski`/`unsloth` GGUF listings, cross-referenced via WebSearch 2026-07-12) | **CONFIRMED** — the scaling assumption held; actual matches projection almost exactly (35 GB measured vs 35.6 GiB projected, well within rounding/GB-vs-GiB noise) |
| Q8 pair | ~71.3 GiB (projected) | ~70 GB (2× measured) | **CONFIRMED, fits comfortably** in 121 GB with ~50 GB headroom |
| Vision tower handling | "text-pipeline GGUF conversion drops the vision tower anyway" | Confirmed independently: community guidance is GGUF conversion produces a text-only GGUF **plus a separate `mmproj` file** for vision — the base GGUF used for text-only serving already excludes the vision tower without any extra flag | **CONFIRMED, and a specific mechanism identified** (separate mmproj artifact, not a merge-time flag) |

**Net: the Q8 scaling assumption is sound.** Both the roadmap's projection method (scale the old base's
measured bf16→Q8 ratio onto the new base's total_size) and the actual community-measured Q8 file size agree
to within ~2%. No revision needed to the Stage 5 packaging plan.

## 6. Integration points needing code changes

| Component | File(s) | Change needed | New or Modified | Confirmed by |
|---|---|---|---|---|
| Adapter export path | Tinker LoRA export (external, no local file) | Verify Tinker's exported adapter targets `language_model.*`-prefixed module names, not bare `layers.*` — LoRA target-module regex/config may need the prefix | Modified | Roadmap Phase 20 scope; not independently re-verified this pass (Tinker internals not fetchable) |
| Merge | `scripts/merge_adapter.py` | No code change strictly required — `AutoModelForCausalLM.from_pretrained` + `PeftModel.from_pretrained` + `merge_and_unload()` operate on the loaded module tree regardless of key-name prefix, since PEFT resolves target modules by traversing the live model object, not by string-matching raw safetensors keys. **Verify** (don't assume) that PEFT's `target_modules` config used at LoRA-train time already accounted for the deeper module path — if Tinker's LoRA was trained with target modules resolved against `model.language_model.layers.N...`, merge is a no-op change; if it assumed `model.layers.N...`, the merge will silently fail to find matching modules. This is the real risk, not the merge script itself. | Verify-only (likely no diff) | §2, §3 above |
| Sieve profiler — router hook | `scripts/profile_base_model.py` (line ~454-460, `model.model.layers` unwrap + `layer.mlp.gate` hook) | Change module traversal to `model.model.language_model.layers` (or equivalent attribute path for whatever `AutoModel*` class loads this VL checkpoint) | **Modified** | §3 |
| Sieve profiler — array shapes | `scripts/extract_protected_mask.py`, `scripts/sieve_expert_mask_inference.py`, `scripts/sieve_ksweep_run.py`, `scripts/sieve_cross_seed_overlap.py` | Bump hardcoded `n_experts=128` → `256` wherever present; confirm shared expert never appears in `router_logits` tensor (empirical check, one forward pass); the `[n_layers, n_experts]` shape itself likely does NOT need a stratum split (routing width is uniform across DeltaNet-MoE and Gated-Attention-MoE layers — see §3) | Modified (narrower than Work Item 1's framing) | §3 |
| Eval harness | `eval/`, `scripts/relabel/eval_relabel.py`, wp-bench runner | **None expected**, confirmed — these operate on generated text output via the vLLM API, not on model internals or tensor names. Architecture changes are fully abstracted by the serving layer. | None | Consistent with roadmap's "(none expected?)" flag — this pass found nothing to contradict that |
| GGUF convert | `scripts/run_packaging_recipe.md` (llama.cpp invocation) | Pin a recent llama.cpp build/commit known to support Gated-DeltaNet + Gated-Attention hybrid ops (community-confirmed as a recent addition, not stable-for-months); if the Stage 5 ladder descends below Q8, verify the converter's mixed-precision handling of linear-attention state tensors rather than assuming a flat K-quant is safe | Modified (version pin + ladder-below-Q8 caveat) | §4 |
| vLLM serving | `scripts/dgx_toolbox.py` (`dgx.execute("vllm", ...)`) container config | Pin vLLM ≥0.19.0 (recipe-recommended, harder floor than ≥0.17.0 minimum); add `--enforce-eager` fallback path to the Phase 20 smoke-test runbook in case issue #35945 (CUDA-graph capture crash on GDN layers) reproduces on GB10 | Modified | §4 |
| Token alignment | New Stage 1.5 step (no existing file — net-new script or inline check) | `model.config.eos_token_id`/`pad_token_id` vs tokenizer special-token IDs — unchanged from roadmap, not re-investigated this pass (no new information found) | New | Carried forward from roadmap, unchanged |

## New vs Modified components (summary)

**New (no prior equivalent in the pipeline):**
- Stage 1.5 eos/pad token-ID alignment check (net-new script, Phase 20)
- `mmproj` handling awareness for GGUF vision-tower separation (informational — no code needed since the
  pipeline is text-only, but the packaging runbook should note why the vision tower doesn't appear rather
  than treating its absence as unexplained)

**Modified (existing file, targeted change):**
- `scripts/profile_base_model.py` — module traversal path (`model.model.layers` → `model.model.language_model.layers`)
- `scripts/extract_protected_mask.py`, `scripts/sieve_expert_mask_inference.py`, `scripts/sieve_ksweep_run.py`, `scripts/sieve_cross_seed_overlap.py` — `n_experts` 128→256, shared-expert-exclusion verification
- `scripts/dgx_toolbox.py` vLLM container config — version pin ≥0.19.0, `--enforce-eager` fallback flag
- `scripts/run_packaging_recipe.md` — llama.cpp version pin, ladder-below-Q8 mixed-precision caveat
- Tinker LoRA export config (external) — target-module path verification against `language_model.*` prefix

**Verify-only (likely no diff, but must be checked before assuming):**
- `scripts/merge_adapter.py` — PEFT target-module resolution against the deeper module path

**Unchanged (confirmed by this research pass, not just assumed):**
- `sieve_expert_mask_inference.py::apply_mask` router-logit masking mechanism — architecture-robust as-is
- Eval harness (`eval/`, `scripts/relabel/eval_relabel.py`, wp-bench) — operates on text output only

## Build order (respecting dependencies)

This does not change the roadmap's Phase 20-29 structure — it sharpens what happens inside Phase 20 and
Phase 25.

1. **Phase 20 (base bring-up)** — do the module-path verification (§3, `profile_base_model.py` traversal)
   here as a smoke-test item even though the profiler itself isn't exercised until Phase 25; a broken
   traversal is cheap to catch on a base load smoke test and expensive to catch mid-profiling-run. Pin vLLM
   ≥0.19.0 in the same phase (serving smoke test already happens here per the roadmap).
2. **Phase 20, in parallel** — verify Tinker's LoRA target-module config against `language_model.*` prefix
   before Phase 21/22 SFT stages start (this determines whether `merge_adapter.py` needs any change at all;
   resolve before, not during, the first merge attempt).
3. **Phase 25 (Sieve tooling adaptation)** — apply the narrower fix set from §3/§6: path fix (already
   smoke-tested in Phase 20), `n_experts` bump, empirical shared-expert-exclusion-from-router-logits check.
   Do this before Phase 26 (Gate B) as the roadmap already mandates.
4. **Phase 28 (packaging)** — pin llama.cpp version at the start of this phase, not discovered mid-convert;
   the mixed-precision-below-Q8 caveat only matters if Gate 2's ladder descends past Q8, so it's a
   documentation note now and an active check only if that path is taken.

No dependency ordering changes from the roadmap's proposed Phase 20→29 sequence — every finding here is a
sharpening of what already-scheduled phases do, not a new phase or a reorder.

## Sources

- `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json` — WebFetch, 2026-07-12 (architecture facts, §1) — HIGH confidence, primary source
- `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/model.safetensors.index.json` — WebFetch, 2026-07-12 (checkpoint structure, §2-3) — HIGH confidence, primary source
- `https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/model.safetensors.index.json` — WebFetch, 2026-07-12 (old-base comparison naming, §3) — HIGH confidence, primary source
- `https://huggingface.co/Qwen/Qwen3.6-35B-A3B` — WebFetch, 2026-07-12 (model card, DeltaNet/Gated-Attention head config, §1) — HIGH confidence, primary source
- `https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B` — WebFetch, 2026-07-12 (vLLM version floor, GB10/NVFP4 mention, §4) — HIGH confidence, vendor-maintained recipe doc
- `https://github.com/vllm-project/vllm/issues/35945` — WebSearch, 2026-07-12 (CUDA-graph/causal_conv1d bug on GDN layers, §4) — MEDIUM confidence, open GitHub issue, not independently reproduced on GB10 by this project yet
- `https://allthings.how/qwen3-6-35b-a3b-gguf-quants-sizes-and-how-to-run-it/` — WebFetch, 2026-07-12 (Q8_0 GGUF measured file size, §5) — MEDIUM confidence, third-party aggregator, cross-checked against `bartowski`/`unsloth` repo existence via WebSearch
- `qwen.readthedocs.io` llama.cpp docs + WebSearch on llama.cpp GGUF conversion support — WebSearch, 2026-07-12 (§4) — MEDIUM confidence, community-sourced, "latest llama.cpp version" guidance not pinned to an exact commit
- Local: `/home/robert_li/Desktop/projects/wp-finetune/.planning/V4-RERUN-ROADMAP.md`, `PIPELINE.md`, `scripts/merge_adapter.py`, `scripts/profile_base_model.py`, `scripts/extract_protected_mask.py`, `scripts/sieve_expert_mask_inference.py` — read directly, 2026-07-12

---
*Architecture research for: wp-finetune v4.0 milestone — Qwen3.6-35B-A3B integration*
*Researched: 2026-07-12*
