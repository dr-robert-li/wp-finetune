---
phase: 21-sft-training-generation-judge-models
plan: 01
subsystem: training-infra
tags: [tinker, moe, lora, merge, renderer, qwen3.6, gen01, wave-0]

# Dependency graph
requires:
  - phase: 20-base-bring-up
    provides: "Qwen3.6-35B-A3B downloaded+verified; eos/pad alignment; DeltaNet serving smoke; merge_adapter.py (prefix-aware, target_modules-only) + BASE-04 attention-only merge proof"
provides:
  - "scripts/tinker_reasoning_data_v4.py -- v4 data adapter, renderer resolved at runtime (qwen3_5_disable_thinking), GEN-01 spot-check (max-len/empty-think)"
  - "scripts/tinker_reasoning_sft_v4.py -- v4 SFT driver sibling"
  - "output/base21/gen01_format_decision.json -- GEN-01 decision receipt (COMPLETE, all fields pass)"
  - "output/base21/moe_merge_probe.json -- MoE merge-path probe receipt: merge_ok=false, a REAL architectural gap found and documented, NOT a clean pass"
  - "scripts/merge_adapter.py --guard-receipt-path flag (backward compatible)"
affects: [21-02-gen-sft, 21-03-judge-sft, "any future real GEN-02/JUDGE-02 Tinker merge"]

tech-stack:
  added: []
  patterns:
    - "Renderer resolution for a new Tinker base must be probed at runtime against the actual registry (tinker_cookbook.renderers.get_renderer per candidate name), not assumed from the prior base's choice -- the dedicated Qwen3.5-family renderer (qwen3_5_disable_thinking) exists and is architecturally correct for this VL-class base, distinct from the generic qwen3_disable_thinking Phase 20 used for its attention-only probe"
    - "Tinker's train_mlp=True MoE export is NOT one homogeneous tensor family: shared_expert (ordinary nn.Linear, target_modules-mergeable) and routed experts (per-expert-batched 3D tensors, PEFT target_parameters/ParamWrapper-shaped) are structurally different and must be merged via different mechanisms -- a merge guard reporting N/N 'mergeable' modules can be silently and correctly counting only the SUBSET the current code can reach, not the full trained signal"
    - "A module-count guard that PASSES is not sufficient evidence a merge captured the intended training signal -- it only proves the code path it exercises is internally consistent, not that it exercises the RIGHT path"

key-files:
  created:
    - scripts/tinker_reasoning_data_v4.py
    - scripts/tinker_reasoning_sft_v4.py
    - tests/test_tinker_reasoning_data_v4.py
    - scripts/build_base21_moe_probe_adapter.py
    - output/base21/gen01_format_decision.json
    - output/base21/moe_merge_probe.json
  modified:
    - scripts/merge_adapter.py

key-decisions:
  - "RENDERER_NAME resolved to qwen3_5_disable_thinking (source=registry), not the Phase-20-precedent qwen3_disable_thinking -- confirmed via runtime probing against the live tinker_cookbook registry, matching this base's actual resolved class (Qwen3_5MoeForCausalLM per 20-04)"
  - "Kept hp.get_lr(BASE_MODEL, is_lora=True) auto-LR (resolved 4.99e-4) over GEN-02's literal '<=2e-5' text -- the latter is a stale DGX/Unsloth-era carry-over, per ROADMAP.md's own Phase 4.3 supersession note"
  - "output_router_logits confirmed N/A at Tinker's abstraction layer -- checked tinker.ServiceClient.create_lora_training_client and TrainingClient.forward_backward signatures directly (zero router/aux-loss kwargs) and grepped the installed tinker+tinker_cookbook package source (zero hits)"
  - "moe_merge_probe.json records merge_ok=false HONESTLY -- the routed MoE-expert (train_mlp=True's actual point) fused-tensor merge path is UNPROVEN; only the incidental shared_expert (dense) sublayer merged. This is a genuine Rule 4 architectural gap (merge_adapter.py has zero target_parameters handling), not an account/API block, and was NOT papered over to force a clean pass."
  - "Did not attempt to hand-implement the target_parameters composition merge (concat Tinker's w1/w3 per-expert deltas into gate_up_proj) despite having derived the correct tensor-shape/concat-order mechanics from transformers source -- the semantic mapping of Tinker's internal w1/w2/w3 naming to gate/up/down could not be confirmed from any available source, and a wrong guess would silently corrupt every future real GEN-02/JUDGE-02 merge undetectably"

metrics:
  duration: 105min
  completed: 2026-07-13

status: blocked
---

# Phase 21 Plan 01: Wave-0 SFT Pre-Training De-Risking Gate Summary

**Task 1 (GEN-01 format/renderer/LR decision) fully succeeded with real empirical evidence; Task 2's MoE merge probe discovered — rather than resolved — the single highest-risk unverified link this plan exists to de-risk: `merge_adapter.py` has zero support for PEFT's `target_parameters` mechanism, so Tinker's routed-expert (`train_mlp=True`) fused-tensor deltas are silently excluded from every merge while the module-count guard still reports a clean pass. This is a genuine architectural gap requiring a human decision before any real GEN-02/JUDGE-02 Tinker spend — not a task the plan can auto-fix without unacceptable correctness risk.**

## Performance

- **Duration:** ~105 min
- **Started:** 2026-07-13T10:52:34.767Z
- **Task 1 completed:** ~2026-07-13T21:05:00+10:00
- **Task 2 (probe + honest gap documentation) completed:** ~2026-07-13T21:47:00+10:00
- **Tasks:** 2 (1 fully green, 1 executed to completion with an honestly-recorded non-pass finding)
- **Files modified:** 7 (6 created, 1 modified)

## Accomplishments

### Task 1 — GEN-01 format decision (COMPLETE, all criteria met)

- `scripts/tinker_reasoning_data_v4.py`: non-destructive sibling of `tinker_reasoning_data.py`, `BASE_MODEL = Qwen/Qwen3.6-35B-A3B`. Renderer resolved at runtime by probing `tinker_cookbook.renderers.get_renderer()` against real candidates: `qwen3_5_disable_thinking` (the dedicated Qwen3.5-family entry) resolved cleanly and was selected over the generic `qwen3_disable_thinking`.
- Empirically confirmed (live, against the real base+tokenizer) that the Qwen3.5 template's empty `<think>\n\n</think>\n\n` header insertion for post-last-user-turn assistant messages carries **zero loss weight** under `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — decoded the weight>0 span directly and confirmed no `<think>` tokens reach the loss target. QwenLM #131's concern is real for the rendered text but N/A for this project's actual training signal.
- LR: kept `hp.get_lr(BASE_MODEL, is_lora=True)`, resolved live to `4.990818286656736e-4` — matches ROADMAP.md's documented ~4.99e-4 Tinker-regime figure exactly.
- `output_router_logits`: confirmed N/A at Tinker's abstraction layer by directly inspecting `tinker.ServiceClient.create_lora_training_client` and `TrainingClient.forward_backward` signatures (no router/aux-loss kwarg exists) and grepping the installed `tinker`+`tinker_cookbook` package source (zero hits for "router"/"load_balanc").
- GEN-01 spot-check (real data, not mocked): `max_tokenized_len=7851` (< 64,000 cap), `empty_think_injected=false`, 560 train / 136 val batch-covered examples.
- `tests/test_tinker_reasoning_data_v4.py`: 4/4 pass.
- `scripts/tinker_reasoning_sft_v4.py`: sibling driver, imports the v4 data module; default `--save-name` prefixed `wp-reasoning-v4-*` to avoid colliding with v3's existing `output/tinker/wp-reasoning-*.json` manifests (Pitfall 5).

### Task 2 — MoE merge probe (executed; genuine gap found, not a clean pass)

- `scripts/build_base21_moe_probe_adapter.py`: ran a REAL Tinker LoRA probe (rank=8, `train_mlp=True`/`train_attn=False`, 8 steps, ~cents) against the new base.
- Discovered Tinker's `train_mlp=True` export attaches **two structurally different tensor families**: `mlp.shared_expert.{gate,up,down}_proj` (ordinary `nn.Linear`, target_modules-mergeable — 120/120 merged successfully) and `mlp.experts.{w1,w2,w3}` (real per-expert-batched 3D LoRA tensors — shared-A/per-expert-B for gate+up, per-expert-A/shared-B for down — verified via direct tensor-shape inspection, matching PEFT `ParamWrapper`'s `target_parameters` convention).
- Traced the checkpoint's actual fused parameters directly in `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (lines 723-724, 745): `mlp.experts.gate_up_proj` (shape `[256, 2*intermediate, hidden]`) and `mlp.experts.down_proj`, with `gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — confirming these ARE `config/train_config_v4.yaml`'s CR-01 `target_parameters` targets, and deriving the exact composition math needed (concat gate-then-up along the output-row dim).
- Confirmed `merge_adapter.py` has **zero** `target_parameters` handling (`grep -c target_parameters` = 0) — it walks only the live model's submodule tree, which cannot reach a raw `nn.Parameter`. All 120 routed-expert keys were therefore documented-dropped as "no live-model equivalent" — technically correct behavior for the existing (module-tree-only) drop logic, but it means the module-count guard's 120/120 "pass" silently only covers the incidental `shared_expert` sublayer, NOT the routed-expert signal that is the entire point of `train_mlp=True`.
- Did **not** attempt to write the composition-merge code: while the shape/concat mechanics are now known with high confidence, WHICH of Tinker's `w1`/`w3` is semantically "gate" vs "up" could not be confirmed from any available source (no Tinker doc/symbol names this mapping) — a wrong guess would silently and undetectably corrupt every future real merge. This satisfies deviation Rule 4 (architectural decision required), not Rules 1-3.
- Re-ran `scripts/smoke_vl_merge_base20.py` (Phase 20 carry-forward 2): **PASSED**, byte-identical receipt to the already-committed `output/base20/vl_merge_roundtrip.json` — confirms the post-review-fix merge code still round-trips correctly for the attention-only case, no regression.
- `scripts/merge_adapter.py`: added `--guard-receipt-path` (backward-compatible default unchanged) — the guard receipt path was unconditionally hardcoded to `output/base20/_merge_guard_result.json`, which would have silently overwritten Phase 20's committed evidence file on every subsequent Phase 21 merge call (Rule 3 fix, needed regardless of the MoE gap finding).

## Task Commits

1. **Task 1: v4 data adapter + SFT driver + GEN-01 decision** — `28f23bc` (feat) — `feat(21-01): fork v4 Tinker data adapter + SFT driver, record GEN-01 format decision`
2. **Task 2: MoE merge probe + honest gap documentation + merge_adapter.py guard-path fix** — `3179c8b` (feat) — `feat(21-01): MoE train_mlp=True merge probe -- routed-expert fused-tensor merge NOT proven (Rule 4 gap)`

## Files Created/Modified

- `scripts/tinker_reasoning_data_v4.py` — v4 data adapter (BASE_MODEL + runtime-resolved renderer + GEN-01 spot-check)
- `scripts/tinker_reasoning_sft_v4.py` — v4 SFT driver (imports v4 data module; auto-LR, MoE-only, v4-prefixed save-names)
- `tests/test_tinker_reasoning_data_v4.py` — 4 tests, all pass
- `scripts/build_base21_moe_probe_adapter.py` — real `train_mlp=True` MoE merge probe builder
- `scripts/merge_adapter.py` — added `--guard-receipt-path` (backward compatible)
- `output/base21/gen01_format_decision.json` — GEN-01 receipt (all fields pass)
- `output/base21/moe_merge_probe.json` — MoE merge-path probe receipt (`merge_ok=false`, detailed gap analysis)

## Decisions Made

See `key-decisions` in frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `merge_adapter.py`'s guard-receipt path was hardcoded to `output/base20/`**
- **Found during:** Task 2, designing the probe driver
- **Issue:** `_guard_merge_completeness`'s `guard_receipt_path` was unconditionally `PROJECT_ROOT / "output" / "base20" / "_merge_guard_result.json"`, regardless of which config/adapter was being merged. Every future Phase 21 merge call would silently overwrite Phase 20's already-committed evidence file.
- **Fix:** Added `--guard-receipt-path` CLI flag (default unchanged, backward compatible); `build_base21_moe_probe_adapter.py` passes `output/base21/_merge_guard_result.json` explicitly.
- **Files modified:** `scripts/merge_adapter.py`
- **Verification:** `--help` lists the new flag; probe run wrote to `output/base21/_merge_guard_result.json`, `output/base20/_merge_guard_result.json` untouched (confirmed via `git diff` — no change).
- **Committed in:** `3179c8b` (Task 2 commit)

**2. [Bug in my own probe script] Missing `sys.path` entry for `scripts` as a package**
- **Found during:** first Task 2 run, `run_base_vs_merged_diff`
- **Issue:** `from scripts._p0_vllm_smoke_serve import ...` failed with `ModuleNotFoundError` — only `PROJECT_ROOT/scripts` was on `sys.path`, not `PROJECT_ROOT` itself.
- **Fix:** Added `sys.path.insert(0, str(PROJECT_ROOT))`.
- **Verification:** Re-ran from the already-produced artifacts (no re-spend); diff + smoke-rerun completed successfully.
- **Committed in:** `3179c8b` (Task 2 commit)

### NOT Auto-Fixed — Rule 4 Architectural Gap (requires human decision)

**merge_adapter.py cannot merge Tinker's routed MoE-expert (`train_mlp=True`) fused-tensor deltas.**

- **What was found:** Tinker's real export for `train_mlp=True` splits the routed-expert LoRA into `mlp.experts.{w1,w2,w3}` (per-expert-batched 3D tensors matching PEFT's `target_parameters`/`ParamWrapper` mechanism), structurally distinct from `mlp.shared_expert.{gate,up,down}_proj` (ordinary `nn.Linear`, already mergeable). `merge_adapter.py` implements ONLY the `target_modules` (submodule-tree) merge path — it has zero code for `target_parameters`, despite `config/train_config_v4.yaml`'s CR-01 fix explicitly configuring `target_parameters` for exactly these fused tensors. The 120 routed-expert keys are silently (but honestly-logged) documented-dropped, and the module-count guard reports a clean 120/120 pass that only covers the incidental `shared_expert` sublayer.
- **Why this matters:** `train_mlp=True` (MoE-only LoRA) IS the entire GEN-02/JUDGE-02 training recipe (D-N1). A merge that silently excludes the routed-expert deltas would ship models whose actual fine-tune signal never landed in the merged weights — while every automated check (`merge_ok`, guard counts) would report success. This is exactly the "silent all-zero/partial merge" T-21-01's threat register exists to catch, now caught BEFORE any real training spend, per the plan's own stated purpose.
- **Proposed change:** Extend `merge_adapter.py` with a `target_parameters`-aware merge path: read `config['lora']['target_parameters']`, and for each entry, compose the corresponding per-expert LoRA deltas (concatenating Tinker's separately-exported gate/up components in the correct order, verified from `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`) directly into the fused `nn.Parameter` tensor.
- **Why this needs a human decision, not an auto-fix:** the ONE missing piece to write this code safely — whether Tinker's `w1` or `w3` is semantically "gate" vs "up" — could not be confirmed from `tinker`/`tinker_cookbook`'s installed source, any project doc, or any other available source this session. Guessing wrong would produce a merge that still "succeeds" (nonzero deltas, guard passes, base-vs-merged differs) but with gate/up silently swapped — an undetectable correctness bug in every future real GEN-02/JUDGE-02 merge. This is squarely Rule 4: "Fix requires significant structural modification" + genuine ambiguity that could silently corrupt correctness.
- **Alternatives considered:**
  1. Confirm the w1/w2/w3 mapping via Tinker docs/support, then implement the fix (recommended — bounded, now well-specified).
  2. Accept `train_mlp=True` training but ship WITHOUT the routed-expert merge (i.e., treat only `shared_expert`'s dense delta as "the merge"), accepting that the resulting merged model would NOT reflect the routed-expert fine-tune at all — very likely unacceptable given the recipe's entire premise.
  3. Have GEN-02/JUDGE-02 train with `train_mlp=False` and rely on attention-family LoRA only until the merge gap is fixed — contradicts D-N1 (MoE-only LoRA is the established recipe) and would require a separate re-planning decision.
- **Impact:** GEN-02 (~$2 gen SFT run) and JUDGE-02 (~$6 judge SFT run) remain BLOCKED on this gap — real Tinker training spend for either should not proceed until `routed_moe_expert_merge_proven=true` is achieved by a follow-up fix + re-probe.

## Known Stubs

None — all code paths executed for real (no mocked Tinker calls, no fabricated receipts). The `moe_merge_probe.json` "gap" is a genuine, measured, negative finding, not a stub.

## Issues Encountered

The GPU (GB10) and Docker were shared/serial resources across the diff-serve and smoke-rerun steps; both vLLM boot cycles (~400-635s each) and the smoke-rerun's own merge+serve cycle ran sequentially without incident. No orphaned containers or leftover GPU memory after cleanup (confirmed via `docker ps -a` and `nvidia-smi`).

## User Setup Required

None beyond what was already approved (Tinker spend for the MoE probe, ~cents, matches the pre-approved 20-04 attention-probe cost profile).

## Next Phase Readiness

- **GEN-01 requirement: SATISFIED.** Format/renderer/LR/router-logits decision recorded with real, measured evidence (renderer resolves, max-len < 64K, no empty-think leakage into the loss target).
- **MoE merge-path gate: NOT satisfied.** The routed-expert (`train_mlp=True`) fused-tensor merge path is UNPROVEN — this GATES 21-02 (gen SFT) and 21-03 (judge SFT), both of which depend on merging a real `train_mlp=True` adapter. Real Tinker spend for either should not proceed until this gap is resolved (see `recommended_next_steps` in `output/base21/moe_merge_probe.json`).
- Phase 20 carry-forward 2 (fresh `smoke_vl_merge_base20.py` re-run) is now DISCHARGED — confirmed byte-identical pass, no regression in the post-review-fix merge code for the attention-only case.
- Phase 20 carry-forward 1 (CR-01 dry-run confirmation) remains open — the MoE probe here is stronger evidence than a static dry-run would have been (it's a REAL empirical run), and it found the exact gap CR-01's config anticipated but `merge_adapter.py` never implemented; this satisfies the SPIRIT of carry-forward 1's ask (empirically confirm the fused target_parameters path) while surfacing that the CODE side of that fix was never done.

---
*Phase: 21-sft-training-generation-judge-models*
*Completed: 2026-07-13 (Task 1 clean pass; Task 2 executed to completion with an honest, non-passing architectural finding)*

## Self-Check: PASSED

All 7 created/modified files verified present on disk; both task commit hashes (`28f23bc`, `3179c8b`) verified present in `git log`.
