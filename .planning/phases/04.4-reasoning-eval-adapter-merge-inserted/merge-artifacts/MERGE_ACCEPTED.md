# Phase 4.4 Merge Acceptance Record

**Date:** 2026-05-29
**Decision authority:** Multi-model council (GPT-5.5 / Claude Opus 4.8 / Gemini 3.1) + human

## Artifacts produced

| Artifact | Path | Status |
|----------|------|--------|
| v1 30_70 baseline | `models/qwen3-30b-wp-30_70-merged-v2/` | ACCEPTED (13 shards, 56.9 GiB) |
| ckpt-72 reasoning merge | `models/qwen3-30b-wp-30_70-reasoning-merged/` | PROMOTED (5-gate certified) |

> Model weight shards are gitignored. This record + the copied `*_merge_report.json`
> are the tracked audit trail.

## v1 baseline (merged-v2)

- **Method:** CPU-only raw-HF + PEFT `merge_and_unload()` (`scripts/_p0_unsloth_merge_v3.py`,
  launcher `scripts/_run_p0_remerge_v3.py`).
- **Why v3 (CPU):** v2 (Unsloth GPU, `scripts/_p0_unsloth_merge_v2.py`) OOMed on GB10 at
  adapter-load — `max_memory` is an accelerate placement hint, not a hard cap; GPU+CPU
  pinned pages exceeded the 121 GiB unified ceiling. CPU path removes NVRM entirely.
- **Structural finding:** v1 adapter (`adapters/qwen3-30b-wp-30_70/`) contains ZERO
  `gate_up_proj` LoRA tensors — 12674 keys = 12288 per-expert down_proj + 384 attention
  + 2 modules_to_save. `target_parameters: [gate_up_proj, down_proj]` was a no-op at
  training time. v3 correctly fuses everything that exists. Verified by
  `scripts/_p0_v3_diff_check.py`: down_proj LoRA-magnitude deltas present, gate/up
  byte-identical to base (no deltas to apply).
- **Pitfall-5 revision:** v1 partial state is a TRAINING gap, not a PEFT merge-time drop.

## ckpt-72 reasoning merge (reasoning-merged)

- **Method:** unsloth-static fused-MoE per-expert + PEFT attention
  (`scripts/_p0_merge_unsloth_static_moe.py`).
- **Adapter format:** Unsloth fused-experts shared-rank LoRA — `mlp.experts.base_layer.*`
  (fused gate_up_proj) + `mlp.experts.*` (fused down_proj), rank 4096 = 32 × 128 experts,
  stored flat 2D. transformers 5.3.0 stores base experts as fused stacked 3D params
  (`gate_up_proj (128,1536,2048)`, `down_proj (128,2048,768)`); gate_up stays fused (no split).
- **Math:** per-expert contiguous block `delta_e = B[:, e*R:(e+1)*R] @ A[e*R:(e+1)*R, :] * scale`.
  Council's initial "broadcast `(B@A)*scale` to all experts" interpretation was FALSIFIED
  pre-launch by `scripts/_p0_extraction_probe.py` (broadcast cos_sim 0.08 vs per-expert,
  12× over-magnitude). Per-expert math is byte-exact (max_diff < 1e-6) to Unsloth
  `_extract_lora_from_wrapper` (`unsloth_zoo/temporary_patches/moe_utils.py:421-426`).

## 5-gate certification (all PASS)

1. **static-classified** — Hypothesis A/C confirmed via Unsloth source probe; adapter has
   no router LoRA, so merge is pure static weight-level delta (no routing dependence).
2. **tensor-anchor** — `scripts/_p0_anchor_tensor_full.py`: per-expert extraction byte-exact
   to Unsloth reshape+permute, 384 checks (48 layers × 2 proj × 4 experts), max_diff 0.0.
3. **forward-anchor (bf16-calibrated)** — `scripts/_p0_anchor_forward_rankpath.py`: rank-path
   (activation-space LoRA on base) vs weight-path (merged) full MoE block, layers {0,23,47}
   × seeds {42,137,999}, 9/9 PASS, router-invariant. Thresholds cos≥0.99990, rel_l2≤1e-2,
   mean≤2e-3, max<0.1. Observed cos 0.99996.
4. **fp32-control** — `scripts/_p0_anchor_fp32_control.py`: candidate stored weights ==
   bf16(true fp32 merge), rms diff 2-3e-5 < bf16_floor 9e-5. PRIMARY certifier.
5. **merge_report counts** — gate_up 6144, down 6144 experts touched (48×128 each);
   per-expert-differ L0 e0 vs e1 = 0.000129 (confirms NOT broadcast).

## Threshold recalibration (council A+B)

Initial forward-anchor thresholds (cos≥0.99999, rel_l2≤1e-3) were fp32-grade and FAILED
AS EXPECTED — bf16-stored 30B weights physically cannot meet fp32 equivalence. fp32
weight-control isolates the bake-faithfulness question without downstream matmul
amplification → adopted as PRIMARY certifier; forward anchor recalibrated to bf16-aware
thresholds as corroboration with hard router-invariance requirement.

## Comparison framing (W5-01 banner)

Baseline (merged-v2) and reasoning-merged share the same v1 training-gap floor (neither
has gate/up MoE LoRA from v1 training; reasoning adds ckpt-72 deltas on top). Compare the
two to isolate the reasoning fine-tune's marginal effect. NOT a clean full-LoRA ablation —
no true-full-LoRA artifact exists.
