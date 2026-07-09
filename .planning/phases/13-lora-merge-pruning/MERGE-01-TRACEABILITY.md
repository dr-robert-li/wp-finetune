# MERGE-01 — Traceability Record

**Status:** CLOSED (N/A-style — no new merge work performed)
**Requirement:** MERGE-01 — "Merge MoE-Sieve + RL LoRA adapters into base model weights before pruning"
**Closure rationale (13-CONTEXT.md, inherited verdict 5):** there are no unmerged adapters left to
merge. RL was rejected as a training path (no RL LoRA exists — see Phase 12 RL-CLOSED decision,
commit `8860e89`). The Sieve method (Phase 11) was training-free (routing-count k-sweep masking at
inference time, no gradient step, no LoRA produced). Consequently every checkpoint Phase 13 prunes
is **already a full merged model** — this record exists to make that assertion auditable, not to
perform a merge.

## Checkpoints covered

| Role | Path | num_local_experts | num_hidden_layers | index.json present |
|------|------|--------------------|---------------------|----------------------|
| gen (v1.2, ship) | `models/qwen3-30b-wp-30_70-reasoning-merged-v4` | 128 | 48 | yes |
| judge seed s0 | `models/_staging/qwen3-30b-wp-v1.3-s0-merged` | 128 | 48 | yes |
| judge seed s1 | `models/_staging/qwen3-30b-wp-v1.3-merged` | 128 | 48 | yes |
| judge seed s2 | `models/_staging/qwen3-30b-wp-v1.3-s2-merged` | 128 | 48 | yes |

Verified live via `AutoConfig.from_pretrained(path)` (no weight tensors loaded — config-only, per
plan instruction to avoid the ~60GB full-model load) plus an `os.path.exists` check for each
checkpoint's `model.safetensors.index.json`. All 4 pass: `num_local_experts == 128`,
`num_hidden_layers == 48`, index present. See `## Verification` below for the exact command and
output.

## Producing tool and provenance

**gen** was produced by `scripts/merge_adapter.py` (Tinker per-expert MoE-only merge; attention and
`lm_head` untouched). Provenance recorded in
`models/qwen3-30b-wp-30_70-reasoning-merged-v4/merge_report.json`:

- `merge_type`: `tinker_per_expert_moe_only_NO_attn_NO_lm_head`
- `base_path`: `models/Qwen3-30B-A3B`
- `adapter_tar`: `models/tinker_export/wp-reasoning-v4-winner/checkpoint.tar`
- `scale`: 1.0, `r`: 32, `lora_alpha`: 32
- `is_moe_only_adapter`: true; `attention_skipped`: true; `lm_head_applied`: false
- `tensor_anchor`: pass, `fp32_control_anchor`: pass, `forward_anchor`: pass,
  `anchors_all_pass`: true
- `shard_count`: 13, `wall_clock_sec`: 126.0

Note: the report's own `out_dir` field records the original staging path
(`models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4`); the checkpoint was subsequently promoted
to `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (its current, live path used throughout this
phase) after anchor certification passed. No new merge was run for this promotion — the report is
the same anchor-certified artifact, moved.

**Judge seeds** (s0/s1/s2) were produced by the same Tinker-export + `scripts/merge_adapter.py`
per-expert MoE-only merge pipeline during Phase 12 (RL two-model track, closed per commit `8860e89`
"RL CLOSED — ideal-conditions smoke killed 6/6, v1.3 final, two-model to Phase 11"). No
`merge_report.json` is present alongside the 3 judge seed directories on disk; their provenance is
the Phase 12 RL-track history, and their structural validity for this phase is established
independently by the live `AutoConfig` + index.json check below (same check used for gen), not by
re-reading a report file that does not exist for them.

## No adapters remain to merge

- **RL LoRA:** none. RL was rejected as a training path (Phase 12 closure, commit `8860e89`) —
  no RL adapter was ever produced, so there is nothing for MERGE-01 to merge from that source.
- **Sieve LoRA:** none. The Phase 11 Sieve method is training-free — `scripts/sieve_expert_mask_inference.py`
  masks router logits at inference time (a forward-hook, `-inf` on non-kept experts' logits); it
  never runs a gradient step and produces no adapter weights.

**MERGE-01 therefore closes with zero new merge code.** All 4 checkpoints Phase 13's AIMER/REAP
pruning will read (`output/prune/aimer_scores_gen.npy`, `output/prune/aimer_scores_judge.npy`, and
the corresponding REAP scores if run) are already fully-merged, ready-to-prune weight files.

## Verification

```bash
.venv-tinker/bin/python -c "from transformers import AutoConfig; import os; [print(p, AutoConfig.from_pretrained(p).num_local_experts, os.path.exists(os.path.join(p,'model.safetensors.index.json'))) for p in ['models/qwen3-30b-wp-30_70-reasoning-merged-v4','models/_staging/qwen3-30b-wp-v1.3-merged','models/_staging/qwen3-30b-wp-v1.3-s0-merged','models/_staging/qwen3-30b-wp-v1.3-s2-merged']]"
```

Output (2026-07-10):

```
models/qwen3-30b-wp-30_70-reasoning-merged-v4 128 True
models/_staging/qwen3-30b-wp-v1.3-merged 128 True
models/_staging/qwen3-30b-wp-v1.3-s0-merged 128 True
models/_staging/qwen3-30b-wp-v1.3-s2-merged 128 True
```

(`num_hidden_layers` checked identically = 48 for all 4; omitted from the plan's one-line print but
verified separately during this record's authoring — see table above.)
