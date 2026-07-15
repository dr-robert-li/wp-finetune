# Phase 23-03 Extension: v4 Judge Served UNMERGED (Runtime LoRA) — Pre-Registration

**Written before any measurement.** Locks the primary metric, the win rule, and the fallback
before any adapter conversion, vLLM/llama.cpp boot, or capture begins.

## Why this run exists

`ext_q8_results.json` (23-02 extension, part 1) found the v4 judge's shipped-stack (llama.cpp
Q8_0) ensemble (0.8067) does **not** unequivocally beat v3's shipped ensemble (0.8056) — paired
bootstrap CI-lower is negative. `output/base21/diagnostic/DIAGNOSTIC_SYNTHESIS.md` attributes the
gap between Tinker-capture rho (0.8358, s1) and any served figure (~0.78-0.79 across vLLM-bf16 and
llama.cpp-Q8, both merged) to an **engine-numerics ceiling that both engines share when the LoRA
is merged into the base weights**, not a training or label defect. This extension tests the one
remaining lever: does serving the adapter **unmerged** (native runtime LoRA — `W·x + B·A·x`
computed as two separate ops, no bf16-rebake of the base weights) recover the capture-path rho
that the merge step destroys?

## Hypothesis (H1)

Precision swamping at merge time: LoRA deltas are small relative to bf16 ULP of the 67 GiB base;
baking them in loses them (confirmed: fp32-accumulation merge reproduces the same ~0.78 ceiling,
`DIAGNOSTIC_SYNTHESIS.md` exp3), while runtime LoRA preserves the delta exactly because it is
never added into the base tensor. H1 is consistent with all existing evidence: Tinker capture
(unmerged, reference implementation) 0.8358; vLLM-served-merged 0.7872; llama.cpp-Q8-merged
0.7877; engine-independent served ceiling ≈0.79.

**H1 confirmed if:** vLLM (or llama.cpp) serving the adapter unmerged scores s1 rho within noise
of 0.8358 (bootstrap CI overlap with the capture-path CI, `[0.7740, ...]` per VERDICT-EVAL4.md §3).

## Serving-mechanics finding (established by source inspection, before any GPU time spent)

`output/base21/diagnostic/exp2_unmerged_lora_rho.json` recorded vLLM `--enable-lora` rejecting the
adapter's routed-expert keys (`mlp.experts.w1/w2/w3`, Tinker's `target_parameters` 3D-batched
export) — status `blocked`, logged as a genuine finding, not routed around. This extension
re-examines that blocker at the source level rather than accepting it as final, because the
installed vLLM build (`0.20.2rc1.dev196+g84f7a5534`, `ghcr.io/spark-arena/dgx-vllm-eugr-nightly`)
does contain a MoE-LoRA implementation (`vllm/lora/layers/fused_moe.py`,
`FusedMoE3DWithLoRA` — the model's `Qwen3_5MoeForConditionalGeneration` class sets
`is_3d_moe_weight = True`, confirmed from `models/Qwen3.6-35B-A3B/config.json`
`architectures: ["Qwen3_5MoeForConditionalGeneration"]`).

Reading `vllm/lora/lora_model.py` (`check_unexpected_modules`, the exact function that produced
the exp2 error) and `vllm/lora/model_manager.py` (`_stack_moe_lora_weights`) shows vLLM expects a
specific **PEFT convention** for this architecture: two keys per MoE layer —
`mlp.experts.base_layer.{lora_A,lora_B}` (fused gate+up, `w13`) and `mlp.experts.{lora_A,lora_B}`
(down, `w2`) — not Tinker's three separate `w1`/`w2`/`w3` keys. Inspecting the Tinker export's raw
tensors (`output/base21/judge03_s1_adapter/adapter_model.safetensors`) shows this conversion is
**lossless, not approximate**: `w1.lora_A` and `w3.lora_A` are bit-identical per layer (verified,
3 layers sampled) — Tinker already trains gate_proj and up_proj sharing one LoRA-A, which is
exactly the structure `FusedMoE3DWithLoRA` requires (one shared A, a doubled-width B). No rank
inflation or block-diagonal padding is needed; this is a pure rename + reshape/concat, verified
shape-exact against `_stack_moe_lora_weights`'s internal reshape/permute chain (traced by hand
against the installed source, not guessed).

## Primary metric

Same as `ext_q8_preregistration.md`: 121-item `data/reasoning_dataset/openai_val.jsonl` val set,
`output/relabel/val_labels_v1.json` labels, `scripts/relabel/eval_relabel.py` /
`eval_relabel_ensemble.py` scorers, temp 0, 8192-token completion cap, unchanged from all prior
v4 judge measurements.

## Decision rule (extends, does not supersede, `ext_q8_preregistration.md`)

**Step 1 (single-seed H1 check):** if unmerged-served s1 rho falls within the capture-path
bootstrap CI (`[0.7740, ...]`, `judge03_rho.json`), H1 is CONFIRMED and the run proceeds to s0/s2.
If it does not (i.e. it lands back near the ~0.78-0.79 served ceiling), H1 is REJECTED and the
run stops at s1 — no s0/s2 capture, no ensemble, the part-1 verdict stands as final, recorded with
this negative result as an additional confirmation that the ceiling is not a mergeable-vs-unmerged
artifact.

**Step 2 (ensemble win, only if H1 confirmed):** 3-seed (s0/s1/s2) median ensemble scored via the
identical paired-bootstrap-vs-v3 rule as `ext_q8_preregistration.md` §"UNEQUIVOCAL WIN rule"
(10,000 resamples, `numpy.random.default_rng(1337)`, paired per-item delta, CI-lower > 0 against
v3's shipped ensemble point 0.8056). If this fires, the unequivocal-win verdict for the v4.0
milestone **flips to TRUE**, with serving-config = runtime-LoRA (base `models/Qwen3.6-35B-A3B` +
published adapter `output/base21/judge03_s1_adapter` — a shippable configuration, no merge step
required). Also record vs the pre-registered v4 target (0.87 ensemble / 0.85 single-seed).

## Engine fallback order

1. **vLLM `--enable-lora`** with the converted (renamed/reshaped) adapter — primary path, given
   the source-level confirmation above that the installed build has a matching MoE-LoRA kernel.
2. **llama.cpp `--lora`** (`convert_lora_to_gguf.py` over `models/_gguf/` base, or a fresh GGUF
   conversion of `models/Qwen3.6-35B-A3B` if no matching base GGUF exists) if vLLM rejects the
   converted adapter at a level deeper than naming (a genuine kernel/shape incompatibility, not
   fixable by the rename derived above).
3. **BLOCKED** (evidence recorded, part-1 verdict stands) if neither engine can apply the
   routed-expert LoRA at runtime without approximation.

## Failure disposition

Any outcome — H1 rejected at step 1, ensemble win not achieved at step 2, or BLOCKED at either
engine — is a valid recorded result. No re-running with different construction guesses to chase a
pass; the conversion is derived once, from source, and validated by shape assertions before any
serve attempt. If the mechanically-derived conversion produces a runtime error, that error is
recorded as evidence for the corresponding fallback tier, not silently patched by guessing a
different tensor layout.

## Scope note

This run measures the judge role only, via the same 121-item val set as every other v4 judge
measurement in this milestone. It does not reopen the gen-role verdict or relabel-campaign
condition (VERDICT-EVAL4.md §4, unchanged).

---
*Pre-registered 2026-07-15, before adapter conversion, before any vLLM/llama.cpp boot for this
extension.*
