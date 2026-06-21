---
phase: 10
task: "W0-Task0"
created: 2026-06-21
---

# Merge Format Compatibility: Phase 9 Tinker RL Export vs merge_tinker_v3.py

## Expected Input Format (merge_tinker_v3.py)

`merge_tinker_v3.py` expects a **`checkpoint.tar`** file — the third distinct Tinker MoE LoRA
convention (different from PEFT-strided and Unsloth contiguous-block).

Tensor layout inside the tar:

| Gate matrix | lora_A shape | lora_B shape | Shared? |
|-------------|-------------|-------------|---------|
| w1, w3 | `[1, 32, 2048]` SHARED | `[128, 768, 32]` PER-EXPERT | A shared, B per-expert |
| w2 | `[128, 32, 768]` PER-EXPERT | `[1, 2048, 32]` SHARED | A per-expert, B shared |
| unembed (lm_head) | `[32, 2048]` | `[151936, 32]` | — |

Keys: `moe.layers.<N>.mlp.experts.w{1,2,3}.lora_{A,B}`, `unembed.lora_{A,B}`.

Per-expert distinctness verified: `w1.lora_B[0] vs [1] max_diff = 0.049` (confirming real MoE-only
LoRA, not a broadcast stub).

MoE-only path is the correct branch for RL checkpoints (full-MoE LoRA on frozen router).

---

## Verdict: COMPATIBLE

The Phase 9 Tinker RL training run uses the same Tinker-native MoE LoRA export format
(`checkpoint.tar`). Tinker's export convention has not changed between the SFT (Phase 4.3) and
RL (Phase 9) runs — both produce the identical key layout and tensor shape convention that
`merge_tinker_v3.py` was written and tested against.

No format adapter is required for Wave 1 Task 4. The Wave 1 command is:

```bash
python scripts/merge_tinker_v3.py \
    --checkpoint output/rl_checkpoints/<step>.tar \
    --base <qwen3-30b-base-path> \
    --out merged_rl_checkpoint/
```

---

## Critical: Merge Is Mandatory Before vLLM Serving

vLLM **cannot serve raw LoRA adapters** from a `checkpoint.tar`. The merge step fuses
`base_weight + lora_B @ lora_A` into full-rank tensors and writes a standard HuggingFace
checkpoint that vLLM can load directly.

Serving without merge = vLLM startup failure (no model weights, only delta tensors).

Pipeline order (Wave 1):

```
checkpoint.tar
  → merge_tinker_v3.py   (fuse LoRA into base model)
  → merged_rl_checkpoint/ (HF-format full weights)
  → vLLM :8020           (serve for eval_gen)
  → eval_gen / eval_judge (generate + judge)
  → bootstrap_gate.py    (CI-aware pass/fail gates)
  → wp-bench             (task-benchmark)
  → rlev02_report.py     (five-part conjunctive gate, final report)
```

---

## Checkpoint Selection for Wave 1

`output/rl_checkpoints/checkpoint_manifest.json` enumerates all saved steps.
Wave 1 Task 4 selects the **best-reward step** (highest `reward_mean` in the manifest before
any KL halt trigger). `bootstrap_gate.py check_no_routing_collapse` verifies KL/efrac bounds
are satisfied at that step before merge.
