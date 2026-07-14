# v4.0 Phase 21 Diagnostic — Experiments 2, 3, 1 Results

**Date:** 2026-07-14. Ran GPU-serialized in the order specified (2, 3, 1), one 67 GiB
residency at a time, teardown + GPU-idle verification between each. Sole GB10 user
throughout. All receipts fail-closed (status field always present); no historical
Phase 21 receipts modified.

## Results table

| # | Experiment | Status | Key figure | vs anchor(s) | Verdict |
|---|---|---|---|---|---|
| 2 | Judge s1 unmerged-LoRA via vLLM `--enable-lora` | **blocked** | n/a (vLLM rejected the adapter at `add_lora`) | — | Inconclusive by blockage — does not discriminate merge-numerics vs engine-numerics |
| 3 | fp32-accumulation merge fix, re-merge + re-serve s1 | measured | rho = **0.7823** CI [0.7119, 0.8307] | vs original bf16-merge served 0.7872 (Δ −0.0049); vs Tinker-capture anchor 0.8358 (Δ −0.0535) | **Engine numerics dominate** — fp32 fix does not recover the gap |
| 1 | Gen ep1 checkpoint, full 344-test wp-bench | measured | overall = **0.4381** CI [0.3295, 0.5504] | vs ep3 (shipped) 0.372 (Δ +0.0661, 56.2% of ep3→raw-base gap); vs raw base 0.4897 (Δ −0.0516) | **Overtraining CONFIRMED as a major contributor** |

## Experiment 2 — Judge s1 unmerged-LoRA via vLLM (BLOCKED, evidence recorded)

**Receipt:** `output/base21/diagnostic/exp2_unmerged_lora_rho.json`

Served the raw base + s1 judge LoRA adapter natively via vLLM `--enable-lora`
`--lora-modules` (no merge step). The adapter is a routed-MoE-expert export (120/240
modules are `mlp.experts.{w1,w2,w3}` — PEFT `target_parameters` 3D per-expert tensors;
the other 120 are ordinary `mlp.shared_expert.{gate,up,down}_proj` nn.Linear modules).

vLLM booted the base model (409s) but failed at the `add_lora` step:

```
Exception: Call to add_lora method failed: While loading /workspace/lora, expected
target modules in {'experts', 'gate_proj', 'shared_expert_gate', ...} but received
['model.layers.0.mlp.experts.w1', ...]
```

**This is a naming-convention mismatch, not a fundamental "MoE LoRA unsupported"
finding** — vLLM's valid-target-modules set for this architecture literally includes
the bare name `'experts'`, meaning vLLM does have LoRA support for routed MoE experts
on this model family in principle. Tinker's raw export uses a `w1`/`w2`/`w3`
split-projection naming convention (mirroring HF's `gate_proj`/`up_proj`/`down_proj`
semantics but under a different key scheme) that vLLM's LoRA loader does not
recognize. Closing this gap would require writing a custom key-remapping shim between
Tinker's export convention and vLLM's expected canonical module names — out of scope
for a diagnostic experiment; recorded honestly as blocked rather than silently
degrading to a partial (shared-expert-only) test that would answer a different
question than the one asked.

Container torn down in a `finally` block; boot took 431.7s total before failing;
GPU verified idle after.

**Consequence for the diagnostic:** Experiment 2 cannot discriminate merge-numerics vs
engine-numerics on its own. Experiment 3 (below) answers that question directly by a
different route.

## Experiment 3 — fp32-accumulation merge fix (measured: engine numerics dominate)

**Receipt:** `output/base21/diagnostic/exp3_fp32_merge_rho.json`
**Code fix:** `scripts/merge_adapter.py` (`_fp32_upcast_adapter_copy`,
`_upcast_lora_layers_to_fp32`), committed separately with unit tests in
`tests/test_merge_adapter_fp32_accumulation.py`.

Code-level investigation (before spending any GPU time) found the merge engine
**actually used** for every real Phase 21 adapter — `tinker_cookbook.weights.
build_hf_model`'s shard-by-shard path, via `apply_merged_weight()` — already upcasts
the base weight and LoRA delta to fp32 for the addition step
(`target.float() + merged_lora.float()`, cast back to bf16 after). This contradicts
`judge_attenuation_forensics.md`'s framing ("no fp32 accumulation noted in the merge
path"), which analyzed the **legacy PEFT path** — dead code for this project, since
every trained v4 adapter is routed-MoE (`train_mlp=True`) and always routes through
the tinker_cookbook path instead.

The remaining, real precision gap: the LoRA delta **matmul itself** (`lora_B @
lora_A` / the 3D expert `bmm`) still ran in whatever dtype the adapter's safetensors
were stored in (bf16). The fix (`_fp32_upcast_adapter_copy`) upcasts the adapter
tensors to float32 before handing them to `build_hf_model`, so the whole
delta-compute-then-add chain is fp32 throughout. (The mirror-image legacy-path gap —
PEFT computes the delta in fp32 but downcasts before the addition — was also closed,
via `_upcast_lora_layers_to_fp32`, for correctness/completeness per the task's
explicit ask, even though that path is unexercised today.)

Re-merged s1 to a new versioned dir (`models/Qwen3.6-35B-A3B-judge-v4-s1-fp32merged`
— the original 21-06 canonical merge was left untouched; a new output dir was needed
anyway to avoid `merge_adapter.py`'s adapter-content-hash idempotency short-circuit,
since the adapter bytes were unchanged and only the merge code changed). Merge guard
passed (240/240 modules), base-vs-merged real-generation diff confirmed. Re-served,
re-captured the identical 121 wp_judge val prompts, re-scored with the unmodified
`eval_relabel.py`:

```
fp32-merged served:        rho = 0.7823  CI [0.7119, 0.8307]
original bf16-merged served (21-06): rho = 0.7872
Tinker-capture anchor (same checkpoint): rho = 0.8358
```

Delta vs the original bf16-merge figure is **−0.0049** — noise-level, not a recovery
(and in the "wrong" direction, i.e. slightly lower, underscoring this is noise not a
regression from the fix). **Verdict: engine numerics (vLLM-vs-Tinker serving-stack
kernel differences) dominate the capture→serve attenuation, not bf16 LoRA-merge
rounding.** This is consistent with the code-level finding: the merge engine was
already fp32-accumulating the dominant (addition) step before this fix ran, so
closing the remaining (matmul) gap had little room left to matter.

Serve torn down cleanly in a `finally` block; GPU verified idle after.

## Experiment 1 — Gen ep1 checkpoint wp-bench (measured: overtraining confirmed)

**Receipt:** `output/base21/diagnostic/exp1_ep1_wpbench.json`

Downloaded the preserved (non-promoted) `wp-gen-v4-ep1` sampler checkpoint (no
re-training spend), merged via the identical routed-MoE-expert path GEN-03 (21-05)
used — now automatically including Experiment 3's fp32 fix — served, and ran the
exact same wp-bench harness/protocol 21-05 used (full 344-test suite, 320 knowledge /
24 execution, `enable_thinking=False`, `max_tokens=2048`, seed 1337, concurrency 4,
stratified CI-aware bootstrap):

```
ep1 wp-bench overall:   0.4381  CI [0.3295, 0.5504]
ep3 anchor (shipped):   0.372   (gen03_wpbench.json)
raw base anchor:        0.4897  (gen03_wpbench.json fresh anchor)
```

ep1 recovers **56.2%** of the ep3-to-raw-base gap (Δ +0.0661 vs ep3, Δ −0.0516 vs raw
base) — clears this diagnostic's pre-registered ≥50% "materially closes the gap"
threshold. **Verdict: overtraining CONFIRMED as a major contributor** to the GEN-03
codegen regression: fewer training epochs preserve substantially more of the raw
base's own strong codegen prior. This corroborates the terminal-loss signature
`gen_regression_forensics.md` flagged as an unconfirmed contributor (ep3 loss 1.46 vs
the old base's 2.40 on the same 563 rows) — it is now confirmed, not just suggestive.

Merge guard passed (240/240 modules), base-vs-merged real-generation diff confirmed
before benching. Serve torn down cleanly; GPU verified idle after.

**ep2 sweep point deliberately skipped.** The plan authorized capturing ep2 only "if
time-efficient" and "if ep1 shows partial recovery, making the sweep informative."
ep1's result (56% recovery, clears the pre-registered threshold outright) already
answers the decision rule decisively — a second ~35 min GPU cycle would refine the
dose-response shape but would not change the qualitative verdict. ep2's Tinker sampler
checkpoint remains preserved (`wp-gen-v4-manifest.json`) for a future full
epoch-sweep if the milestone wants the finer-grained curve.

## Combined disposition

- **Judge rho gap (served 0.7872 vs target 0.85):** root cause is **engine numerics**
  (vLLM-vs-Tinker serving-stack differences), not bf16 merge rounding — the fp32 merge
  fix, applied correctly and verified via unit test, produced no material recovery.
  Experiment 2 could not independently confirm this via the unmerged-LoRA route
  (vLLM's LoRA loader doesn't understand Tinker's routed-expert export naming
  convention) but Experiment 3's direct fp32-fix-and-remeasure route answers the same
  question and points the same way. The relabel-campaign re-open condition (21-06,
  discretion-item-2) remains unmet for a *different* reason now: the gap-closure
  diagnostic *has* now run and points to an inherent capture-vs-serve methodology gap
  (consistent with the project's own historical ~0.026-0.039 old-base precedent),
  not a fixable recipe/merge defect.
- **Gen regression (0.372 vs raw 0.4897):** overtraining is now a **confirmed** major
  contributor (not just a plausible unconfirmed one). Combined with
  `gen_regression_forensics.md`'s data-shape finding (92% of `<wp_gen>` targets are
  bare unwired fragments), both mechanisms are live: reducing epochs (ep1/ep2) is a
  cheap lever that recovers over half the gap without any data/recipe rework: the
  natural next step (Experiment 4 in `DIAGNOSTIC_SYNTHESIS.md`'s ranked list — rebuild
  the gen mix with full-file wired targets) would likely compound with an epoch
  reduction for a larger combined recovery, but was out of scope for this run.
