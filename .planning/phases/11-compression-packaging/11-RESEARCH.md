# Phase 11: Compression & Packaging (MoE-Sieve, two-model pair) - Research

**Researched:** 2026-07-08
**Domain:** MoE expert routing analysis, selective-training decision infrastructure, multi-model LoRA serving for Qwen3-30B-A3B
**Confidence:** MEDIUM (codebase-verified HIGH on serving/format constraints; MEDIUM/LOW on MoE-Sieve method specifics since it has no published reference implementation and is a project-internal method defined in `wp-moe.md`)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (ship artifact — do not relitigate)

| Role | Model | Metric | Source |
|---|---|---|---|
| wp_judge | **v1.3 3-seed median ENSEMBLE** | rho **0.842** (val n=121) | seeds s0/s1/s2, `eval_seed_curve.json` |
| wp_gen | **v1.2 SFT merged** | codegen **0.4616** wp-bench | `models/qwen3-30b-wp-30_70-reasoning-merged-v4` |

- Judge ensemble = 3 LoRA adapters (rank-32, MoE-only) over the SAME base, median-aggregated `overall` per item. Seed checkpoints (ep3 samplers, manifests in `output/tinker/`): s0 = `wp-reasoning-relabel-v1-full-ep3` (default seed) · s1 = `wp-reasoning-relabel-s1-ep3` (canonical, exported at `models/tinker_export/v1.3`, merged at `models/_staging/qwen3-30b-wp-v1.3-merged`) · s2 = `wp-reasoning-relabel-s2-ep3`. s0/s2 have Tinker sampler checkpoints only — Phase 11 must EXPORT them (same path as s1) to serve locally.
- **Fallback (pre-authorized):** single-seed s1 (rho 0.827, `PROMOTED_v1.3.json`) IF packaging measurement shows the ensemble cannot fit the GB10 memory wall or 3x judge latency breaks serving. Fallback exercise requires only a JOURNAL note, not a re-decision.
- Why ensemble is safe to compress around: seeds share one base -> ONE pruned base + 3 small LoRA adapters (multi-LoRA serving), NOT 3 pruned models. Judge inference = 3 adapter passes + median.
- No further training on this base. RL closed (2026-07-05, 6/6 kills); SFT gap-closure closed (2026-07-08, all levers negative — `output/relabel/gap_closure_summary.json`).

### HARD CONSTRAINTS

1. **Protected expert mask is inviolable.** `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` ([48,128] bool, 1,480 experts, immutable since Phase 7 sign-off 2026-06-19). MoE-Sieve selection AND AIMER/REAP pruning MUST exclude protected experts from removal.
2. **Layer stability notes now in the mask JSON** (`layer_stability_notes` key, added 2026-07-08): low-Jaccard band {9,13,14,31,35,36} + late-layer {45,46,47}. Phase 13 must pre-commit median-threshold (2,477-expert) headroom on these layers before pruning them.
3. **Routing profiles: profile the v1.2 SFT policy** (per 2026-07-03 amendment) for the gen model. Phase 7 profiles remain the protected-expert REFERENCE (do not regenerate the mask).
4. **Post-compression gates (regression bars):** judge ensemble rho >= 0.842 minus noise floor (seed sd 0.020; use `gate_noise_floors.json` conventions); judge single-seed >= 0.827 minus floor; gen wp-bench >= 0.4616 minus pre-registered tolerance (set at plan time, CI-aware per D-V4-10 hardening).
5. **GB10 memory wall** (`MEMORY-INVESTIGATION-bf16.md`): two 30B-A3B model instances do not co-reside in bf16. Packaging must sequence or quantize; measure before promising co-serving.

### Claude's Discretion / Open Questions (NOT pre-decided)

- Serving topology: one base + {3 judge LoRAs, v1.2 gen deltas} multi-LoRA? Or two merged instances swapped/quantized? (v1.2 gen is already merged; judge seeds are unmerged LoRA.)
- Do the 3 judge seeds route similarly enough that one Sieve profile covers all 3, or does Sieve need the union of 3 routing profiles? (Cheap check: E_eff overlap across seeds on the val stimulus.)
- Quantization decision cascade (Phase 15 gates) — bf16 baseline first, then int8/int4 A/B per ROADMAP.
- Whether Phase 12's A/B keeps all 5 k-sweep Sieve points or prunes the sweep given two models.

### Deferred Ideas (OUT OF SCOPE)

None listed in 11-CONTEXT.md beyond the items above (this phase is scope-narrow by design — it starts the Phases 11-15 chain).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIEVE-01 | Fresh routing profiling; LoRA r=32/alpha=64/dropout=0.05 on hot routed experts + attn (Q/K/V/O) + router gates + 4 shared experts; cold experts frozen; protected experts retained | Reuse `scripts/profile_merged_model.py` + `compute_concentration.py` + `extract_protected_mask.py` (Phase 7 infra, verified present and test-certified, 362 tests passing). **Tension**: literal "LoRA...applied to hot experts" = a NEW training run, but CONTEXT.md HARD-STATES "no further training on this base." See Open Questions §1 for reconciliation recommendation. |
| SIEVE-02 | Gen-hot experts trained on golden-only data; judge-hot on full spectrum | Same tension as SIEVE-01 — data-routing plan only meaningful if training happens. If training is dropped (recommended), this becomes a documented-but-not-executed spec, OR is re-scoped to Phase 13 MERGE-01/PRUNE calibration-data routing. |
| SIEVE-03 | Best gen/judge ratio from Phase 4 eval | Already resolved: ratio is 30/70 (canonical, `models/qwen3-30b-wp-30_70-reasoning-merged-v4`), confirmed by Phase 7's stimulus (`data/final_dataset/ratio_30_70/openai_train.jsonl`). No new decision needed. |
| SIEVE-04 | K-sweep at ~13/32/64 active experts | If training is frozen, k-sweep must be executed via **expert masking at inference** (zero out cold experts per k-budget, no gradient) rather than 3 separate LoRA training runs — see Open Questions §1. wp-bench harness (`wp-bench/` + `config/wp-bench.yaml`) already exists and is reusable. |
| SIEVE-05 | Optimal k = smallest budget within +/-1pp of full model on wp-bench (TOST, epsilon=2pp, 3+ seeds) | **No TOST implementation exists in this codebase** (`grep -rn "tost\|TOST\|equivalence"` returned zero hits in `eval/` and `scripts/`). Must be built new this phase — reuse `scripts/bootstrap_gate.py`'s existing bootstrap-CI patterns (`bootstrap_spearman_improvement`, `check_wpbench_gate`) as the statistical scaffold; TOST is two one-sided t-tests against +/-epsilon, a small addition, not a new dependency. |
</phase_requirements>

## Summary

Phase 11 sits at a genuine crossroads created by two amendments landing back-to-back: RL was rejected (2026-07-03) and the SFT gap-closure investigation closed with "all levers negative" (2026-07-08), locking the ship artifact as a frozen two-model pair (v1.3 3-seed judge ensemble + v1.2 gen). The ROADMAP's literal Phase 11 spec — "MoE-Sieve selective training: LoRA the hot experts, freeze the cold ones, k-sweep by retraining at each budget" — is a **training** procedure. CONTEXT.md's hard constraint "No further training on this base" directly contradicts that literal spec. This is the single most important thing the planner must resolve before writing tasks, and this research recommends a specific resolution (§ Open Questions).

Two codebase findings materially change the "Serving topology" open question from open-ended to mostly-answered. First, `scripts/merge_tinker_v3.py` proves that Tinker's MoE-expert LoRA delta is **not** a standard PEFT adapter — it is applied via custom per-expert weight arithmetic (`build_gate_up_delta`, `build_down_delta`) requiring a full CPU-side merge into base weights; only the *attention* LoRA (Q/K/V/O) is a real PEFT adapter that gets merged too in the same script (not served unmerged). Second, `scripts/serve_30_70_vllm.sh`'s own comment documents that vLLM rejects PEFT adapters with non-null `modules_to_save`, and separately `MEMORY-INVESTIGATION-bf16.md` notes "vLLM-LoRA is avoided for the fused-MoE adapter" as an established project convention. **Conclusion: vLLM `--enable-lora` / multi-LoRA runtime serving of the 3 judge seeds' MoE-expert deltas is not available with the tooling this project has built.** The practical topology is therefore: merge each of s0/s1/s2 into its own full 30B checkpoint (reusing/adapting `merge_tinker_v3.py`, which already has this exact convention hard-coded for v3), then serve them **sequentially** (swap containers) rather than concurrently — which sidesteps true multi-LoRA concurrency but does not sidestep the GB10 memory wall, since gen and judge both need to be resident during an actual generate-then-judge workflow.

The GB10 memory wall is independently confirmed by primary-source empirical data (`MEMORY-INVESTIGATION-bf16.md`): any bf16 in-process load of this 30B model peaks ~100-103 GiB on a 124.6 GiB machine due to an intrinsic ~2x CPU-side staging transient — vLLM's own streaming loader avoids this (loads at ~63 GiB via `--gpu-memory-utilization 0.55`), but running two vLLM instances (gen + judge) concurrently is still ~126 GiB, over budget. Packaging must sequence (load gen, generate, unload, load judge, judge, unload) or quantize one/both sides.

**Primary recommendation:** Re-scope SIEVE-01..05 to a training-free routing-analysis pipeline: (1) export s0/s2 from Tinker (s1 already exported+merged), (2) profile routing for the 3 judge-seed merged models by re-running the existing Phase 7 profiling scripts against each, reusing the v1.2 gen-side profile from Phase 7 directly (gen weights are provably unchanged — `PROMOTED_v1.3.json`: "wp_gen data unchanged so no codegen regression path"), (3) compute cross-seed E_eff/Jaccard overlap to decide 1-vs-3 Sieve profiles, (4) execute the k-sweep via **expert masking at inference** (zero cold-expert contribution, no LoRA training) against wp-bench with a newly-built TOST gate, satisfying SIEVE-04/05's letter while honoring the training freeze, and (5) hand the resulting hot/cold classification + optimal-k decision to Phase 13 as prep, since MERGE-01/PRUNE-01 need exactly this kind of expert-importance ranking anyway. Flag this re-scope explicitly to the user before planning locks it in — it changes what SIEVE-01/02 "training" language means in the plan's task list.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tinker checkpoint export (s0/s2) | Backend / Training infra | — | `tinker_export_checkpoint.py` calls Tinker's REST archive endpoint (cloud), downloads a tar to local disk — a data-pipeline/infra step, not model-serving. |
| Routing re-profiling (3 judge seeds) | Backend / GPU forward-pass | — | `profile_merged_model.py` runs inside `ngc-pytorch` DGX container; gradient-free hook-based forward pass, no external LLM. |
| Cross-seed routing overlap (E_eff/Jaccard) | Backend / Analysis | — | Pure numpy computation over `routing_report.jsonl` outputs; no GPU required once profiling is done (matches `compute_concentration.py`'s HOST-only design). |
| Expert masking + k-sweep wp-bench eval | Backend / Inference + Eval | Database/Storage (results JSON) | Requires a served model (vLLM) plus the deterministic wp-bench harness; no LLM judge in the loop for gen; judge-axis needs the judge model served. |
| TOST equivalence gate | Backend / Statistics | — | Pure statistical test over eval outputs, analogous to existing `bootstrap_gate.py` functions; no serving dependency. |
| Model merge (s0/s2 -> full checkpoints) | Backend / Training infra | — | CPU-side weight arithmetic (`merge_tinker_v3.py` pattern); one-time, offline, no serving tier involvement. |
| Multi-model serving topology decision | Backend / Serving (vLLM, Docker) | Database/Storage (model artifacts on disk) | Constrained by GB10 unified-memory ceiling and vLLM's LoRA format restrictions — a deployment-tier decision, not a training-tier one. |

## Standard Stack

### Core (already installed / already used by this project — reuse, do not add new)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | pinned in DGX container images (not in host venv — runs inside `ngc-pytorch`/vLLM containers) | Model loading for profiling and merge scripts | Already the project's model-loading library throughout Phases 3-10 `[VERIFIED: codebase]` |
| `peft` | same container-pinned | Attention-LoRA merge (`PeftModel.from_pretrained(...).merge_and_unload()`) in `merge_tinker_v3.py` | Already used for the ONE part of the adapter that is standard-PEFT-shaped `[VERIFIED: codebase — scripts/merge_tinker_v3.py:377-383]` |
| `numpy` | host + container | Routing count arrays, Jaccard/E_eff computation, bootstrap CI | Already the numeric backbone of `compute_concentration.py` / `extract_protected_mask.py` `[VERIFIED: codebase]` |
| `tinker` (Python SDK) | pinned per `.venv-tinker` | Checkpoint archive download for s0/s2 export | Already the project's cloud-training client since the 2026-06-07 Tinker pivot `[VERIFIED: codebase — scripts/tinker_export_checkpoint.py]` |
| `vLLM` (via `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` image) | pinned image tag, not a pip version this project controls directly | Serving both gen and judge merged checkpoints | Established serving image used by every `serve_*_vllm.sh` script `[VERIFIED: codebase]` |
| `scipy` (likely already a transitive dep via numpy/stats work) | — | TOST equivalence test (two one-sided t-tests, or `scipy.stats.ttest_ind` composition) | `scipy.stats` has no built-in TOST helper as of typical versions — implement the two-one-sided-test composition manually (a few lines), same pattern as `bootstrap_gate.py`'s hand-rolled bootstrap CI. `[ASSUMED — scipy availability in this project's eval-toolbox container not directly verified this session; grep did not confirm import]` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `safetensors` | container-pinned | Reading/writing per-expert weight tensors during merge | Already used in `merge_tinker_v3.py` (`save_file`) |
| `docker` (CLI, not Python) | 29.2.1 confirmed on host `[VERIFIED: docker --version]` | Container lifecycle for vLLM/ngc-pytorch | Existing pattern for every serve/profile script |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom expert-masking k-sweep (recommended) | Actual per-k LoRA sieve training runs (literal ROADMAP spec) | Violates the "no further training" hard constraint from CONTEXT.md; would require re-litigating a closed decision. Masking-only is strictly cheaper and training-free, at the cost of not testing whether *retraining* hot experts recovers quality lost by dropping cold ones (a real difference, flagged in Open Questions). |
| Hand-rolled TOST | A stats package with built-in TOST (e.g. `statsmodels.stats.weightstats`, which does have `ttost_ind`) | `statsmodels` may already be present for other eval work — worth checking before hand-rolling; if present, prefer `statsmodels.stats.weightstats.ttost_ind` over a hand-rolled version (ladder rung 5: already-installed dependency). `[ASSUMED — not verified this session whether statsmodels is installed; planner should check before choosing]` |
| Sequential merged-model serving (recommended for judge ensemble) | True concurrent multi-LoRA serving | Blocked by tooling (Tinker MoE-LoRA is not PEFT-adapter-shaped); would require building a custom in-process weight-swap inference path outside vLLM — large new engineering surface for a phase that is supposed to be compression/packaging, not a new serving engine. |

**Installation:** No new packages required for Phase 11's core work (profiling, export, merge, masking, TOST). All required libraries are already present in the DGX container images or the project's `.venv-tinker`. Verify `statsmodels` availability before choosing between hand-rolled TOST and `ttost_ind`:
```bash
python3 -c "import statsmodels.stats.weightstats as w; print(w.ttost_ind)" 2>&1 | head -3
```

**Version verification:** Not applicable in the traditional npm/pip-registry sense — this phase adds no new external dependencies. The relevant "versions" are pinned container image tags (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`) already in use project-wide; no action needed.

## Package Legitimacy Audit

**Not applicable — Phase 11 introduces no new external packages.** All work reuses the DGX container images, the `tinker` SDK, and Python libraries (`transformers`, `peft`, `numpy`, `safetensors`) already installed and used by Phases 3-10. If the planner decides to add `statsmodels` for TOST, run the standard gate at that time:
```bash
gsd_run query package-legitimacy check --ecosystem pypi statsmodels
pip index versions statsmodels
```
No packages removed or flagged this phase.

## Architecture Patterns

### System Architecture Diagram

```
Tinker cloud (training already done, frozen)
  s0 sampler ckpt ─┐
  s1 sampler ckpt ─┼─(tinker_export_checkpoint.py: signed archive URL → .tar)─┐
  s2 sampler ckpt ─┘                                                          │
                                                                               ▼
                                                          models/tinker_export/{s0,s1,s2}/
                                                                               │
                                             (merge_tinker_v3.py pattern:      │
                                              per-expert MoE delta apply       │
                                              + PEFT attn merge_and_unload     │
                                              + manual lm_head delta)          ▼
                                                          models/_staging/qwen3-30b-wp-v1.3-{s0,s1,s2}-merged/
                                                                               │
                          ┌────────────────────────────────────────────────────┴──────────────────────┐
                          ▼                                                                             ▼
        profile_merged_model.py (per seed, GPU, DGX ngc-pytorch container)          [gen side: REUSE Phase 7 profile
                          │  → routing_report.jsonl, jaccard_stability.json           directly — v1.2 gen weights
                          ▼                                                           are unchanged since Phase 7]
        compute_concentration.py (HOST, no GPU)
                          │  → concentration_report.json (E_eff, CV per seed)
                          ▼
        NEW: cross-seed overlap check (Jaccard/E_eff across s0 vs s1 vs s2 routing_report.jsonl)
                          │
                          ▼
        extract_protected_mask.py (VERIFY existing Phase-7 protected_expert_mask.npy
                                    is still a subset of retained experts — do NOT regenerate)
                          │
                          ▼
        Hot/cold expert classification per k-budget {13, 32, 64}
                          │
                          ▼
        Expert-masking k-sweep (serve via vLLM, zero cold-expert routing weight at inference,
                                 no gradient / no LoRA train) → wp-bench per k, per seed
                          │
                          ▼
        NEW: TOST equivalence gate (epsilon=2pp, vs full-model wp-bench, 3+ seeds)
                          │
                          ▼
        Optimal k decision + protected-expert-retention verification
                          │
                          ▼
        Phase 12 (A/B eval) → Phase 13 (MERGE-01 + AIMER/REAP prune, consumes this
                                          expert-importance ranking directly)
```

### Recommended Project Structure

```
scripts/
├── tinker_export_checkpoint.py     # EXISTING — reuse verbatim for s0/s2 export
├── merge_tinker_v3.py               # EXISTING — adapt/generalize for s0/s1/s2 (currently v3-specific)
├── profile_merged_model.py          # EXISTING — reuse per-seed (loop over 3 judge checkpoints)
├── compute_concentration.py         # EXISTING — reuse per-seed
├── extract_protected_mask.py        # EXISTING — do NOT regenerate the mask; use for verification only
├── sieve_cross_seed_overlap.py      # NEW — Jaccard/E_eff overlap across s0/s1/s2 routing_report.jsonl
├── sieve_expert_mask_inference.py   # NEW — zero-out cold experts at inference for k-sweep (no training)
└── tost_gate.py                     # NEW — TOST equivalence test, pattern-matched to bootstrap_gate.py
output/
└── sieve/                           # NEW output dir, mirrors output/profiling/ conventions
    ├── judge-s0/ judge-s1/ judge-s2/   # per-seed routing_report.jsonl, concentration_report.json
    ├── cross_seed_overlap.json
    └── k_sweep_results.json
```

### Pattern 1: Reuse Phase 7 profiling infra per-model instead of writing new profiling code

**What:** `profile_merged_model.py` already takes `--model-path` and `--output-dir` as CLI args — it is not hardcoded to `reasoning-merged-v4`.
**When to use:** For each of the 3 judge-seed merged checkpoints (and NOT for the gen model, which Phase 7 already profiled).
**Example:**
```bash
# Source: .claude/skills/wp-finetune:run-profiling/SKILL.md Step 1, generalized to a new model path
bash deps/dgx-toolbox/containers/ngc-pytorch.sh python3 -m scripts.profile_merged_model \
  --model-path models/_staging/qwen3-30b-wp-v1.3-s1-merged \
  --ratio ratio_30_70 \
  --output-dir output/sieve/judge-s1
```

### Pattern 2: Merge script generalization (not a new merge algorithm)

**What:** `merge_tinker_v3.py` hardcodes `DEFAULT_OUTPUT_DIR = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3"` and v3-specific adapter paths, but its core functions (`build_gate_up_delta`, `build_down_delta`, `build_lm_head_delta`, the attention-PEFT-merge step) are format-generic — they operate on any Tinker-exported adapter tar with the same `adapter_config.json` shape.
**When to use:** Export s0/s2 with `tinker_export_checkpoint.py`, then invoke a parameterized version of the merge script per seed rather than hand-writing a new merge path.
**Example:**
```python
# Source: scripts/merge_tinker_v3.py:280-330 (existing, verified pattern)
# Adapt: parameterize --adapter-tar and --output-dir instead of the hardcoded v3 defaults
```

### Pattern 3: Sequential (not concurrent) multi-model serving

**What:** Since vLLM cannot load the MoE-expert LoRA deltas as runtime adapters (`modules_to_save`/fused-MoE restriction, verified in `serve_30_70_vllm.sh` comments and `MEMORY-INVESTIGATION-bf16.md`), and GB10 cannot hold two 30B bf16 instances concurrently, judge-ensemble scoring for a given eval set should run as 3 sequential full passes (load s0 → score all items → unload → load s1 → score → unload → load s2 → score → unload), aggregating medians offline — not as one live multi-adapter server.
**When to use:** Any Phase 11/12 eval that needs the 3-seed ensemble judge score.
**Anti-pattern to avoid:** Do not attempt to keep gen + judge (any seed) loaded simultaneously without first measuring actual GB10 headroom at `--gpu-memory-utilization` settings used by this project (0.55, i.e. ~63GB per instance) — 2x63GB = 126GB, likely over the ~124.6GB total minus OS/desktop overhead confirmed in `MEMORY-INVESTIGATION-bf16.md`.

### Anti-Patterns to Avoid

- **Assuming vLLM `--enable-lora` works for the MoE-expert deltas:** It does not — the deltas are not PEFT-shaped for the MLP experts (they're Tinker-internal per-expert weight arithmetic). Only the attention LoRA sub-component is genuinely PEFT-compatible, and even that gets merged rather than served unmerged in the existing project convention.
- **Regenerating the protected-expert mask:** HARD CONSTRAINT — the mask is immutable since Phase 7 sign-off. Phase 11 consumes it, never recomputes it.
- **Training new LoRA sieve adapters without first getting explicit user sign-off on the training-freeze tension:** CONTEXT.md's "no further training on this base" is a locked decision; silently ignoring it to satisfy the ROADMAP's literal SIEVE-01 wording risks re-litigating a closed gap-closure investigation.
- **Re-profiling the gen model:** Gen weights are unchanged since Phase 7 (`PROMOTED_v1.3.json`: "wp_gen data unchanged so no codegen regression path") — re-running a 6.5-hour GPU profiling pass on gen would be pure waste.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TOST equivalence test | A from-scratch two-one-sided-t-test implementation with untested edge cases | `statsmodels.stats.weightstats.ttost_ind` if available in the eval-toolbox container; else a small, carefully-tested hand-rolled version modeled directly on `bootstrap_gate.py`'s existing bootstrap-CI code (same file, same review bar) | TOST has known edge cases (unequal variances, small n) that a mature stats library already handles; verify availability before hand-rolling |
| Tinker checkpoint export | A new download/verification script | `scripts/tinker_export_checkpoint.py` as-is, just pointed at s0/s2's sampler paths from the relabel manifests | Already handles the archive-format quirk (uncompressed tar despite `.gz` naming), signed-URL expiry, and size-sanity checks |
| Per-expert MoE weight merge | A new from-scratch tensor-surgery script | Generalize `scripts/merge_tinker_v3.py`'s `build_gate_up_delta`/`build_down_delta`/`build_lm_head_delta` functions, which are already unit-tested (`tests/phase4_4/test_tinker_merge_convention.py`) | This merge format has known subtle bugs (a broadcast bug that "would PASS a broadcast bug and FAIL a correct merge" per the script's own docstring) — the existing tests exist specifically to catch that class of error |
| Cross-model routing comparison | A new Jaccard/entropy library | `compute_jaccard_stability` and E_eff logic already in `profile_merged_model.py`/`compute_concentration.py`, applied across seeds instead of across subsample-vs-full | Same math (top-k set overlap per layer), different input pairing — no new algorithm needed |

**Key insight:** Almost everything Phase 11 needs already exists in this codebase in a test-certified form from Phase 7 and the Phase 4.3 Tinker pivot. The actual new work is thin: (1) parameterizing existing scripts to loop over 3 seeds instead of 1 model, (2) a cross-seed overlap computation (new but small, reusing existing primitives), (3) an expert-masking inference path (new), and (4) a TOST gate (new but small). The temptation to hand-roll a full "MoE-Sieve" training pipeline from the wp-moe.md pseudocode should be resisted until the training-freeze tension is explicitly resolved with the user.

## Common Pitfalls

### Pitfall 1: Treating "MoE-Sieve selective training" literally without checking the training-freeze constraint

**What goes wrong:** The planner writes tasks that spin up new Unsloth/Tinker LoRA training runs on hot experts, contradicting the locked "no further training on this base" decision from the 2026-07-08 gap-closure closure.
**Why it happens:** The ROADMAP's Phase 11 section (lines 591-619) predates the 2026-07-08 amendment and still describes a training procedure in full detail (`dgx.execute("unsloth_studio", ...)`, k-sweep loop that trains a sieve adapter per k).
**How to avoid:** Surface this explicitly in planning; get user confirmation on the re-scoped (training-free) interpretation before locking task lists. See Open Questions §1 below for the specific recommendation.
**Warning signs:** Any task that says "train sieve adapter at k={13,32,64}" without a checkpoint gate confirming this is intentional despite the freeze.

### Pitfall 2: Assuming the "fresh profiling pass" needs to touch the gen model

**What goes wrong:** Re-running the 6.5-hour GPU profiling pass on the already-profiled v1.2 gen model, wasting a full DGX session.
**Why it happens:** The amendment text says "Phase 11's fresh profiling pass profiles the v1.2 SFT policy" without clarifying that Phase 7 ALREADY profiled that exact same model (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`).
**How to avoid:** Confirm via file hash / model config comparison that the gen-side model in the ship decision is byte-identical to what Phase 7 profiled, then reuse `output/profiling/reasoning-merged-v4/` outputs directly for gen-side hot/cold classification.
**Warning signs:** A plan task that re-runs `profile_merged_model.py` against `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (the exact same path Phase 7 already profiled).

### Pitfall 3: Assuming vLLM multi-LoRA solves the serving topology question

**What goes wrong:** Planning a "one base + `--enable-lora` with 3 judge adapters" serving path that vLLM cannot actually execute for this project's adapter format.
**Why it happens:** vLLM genuinely supports multi-LoRA for standard PEFT adapters in general — but this project's MoE-expert deltas are not standard PEFT adapters (verified: `merge_tinker_v3.py`'s manual per-expert tensor arithmetic, `MEMORY-INVESTIGATION-bf16.md`'s explicit note that "vLLM-LoRA is avoided for the fused-MoE adapter").
**How to avoid:** Plan for sequential merged-model serving (3 separate 30B checkpoints for s0/s1/s2, swapped one at a time) as the default, and only invest in a custom concurrent-serving mechanism if profiling/latency numbers show sequential is unacceptable.
**Warning signs:** A task that says "configure vLLM `--enable-lora --lora-modules s0=... s1=... s2=...`" for the MoE-expert deltas.

### Pitfall 4: Underestimating disk/serving footprint of 3 separately-merged judge checkpoints

**What goes wrong:** 3 full 30B bf16 merges at ~57GB each = ~171GB of disk, plus the existing gen merge (~57GB) and base model copies — could exhaust disk before exhausting memory.
**Why it happens:** The "one pruned base + 3 small LoRA adapters" framing in CONTEXT.md sounds disk-cheap, but the *actual achievable* topology (given tooling constraints above) requires 3 full merges, not 3 small adapter files.
**How to avoid:** Check available disk (`df -h`) before committing to the merge-all-3 plan; consider whether Phase 13's pruning (which happens AFTER Sieve) could apply to each merged judge seed to shrink footprint before all 3 are kept long-term, or whether only s1 (already merged) needs to be kept post-fallback-evaluation.
**Warning signs:** Disk usage climbing past available headroom mid-phase; no disk check task in the plan.

### Pitfall 5: Missing that `models/tinker_export/v1.3/` (the directory named in CONTEXT.md) is currently EMPTY on disk

**What goes wrong:** A task assumes s1's LoRA adapter is present at `models/tinker_export/v1.3/` (as CONTEXT.md's prose states) and fails when the directory is found empty.
**Why it happens:** `PROMOTED_v1.3.json`'s actual `local_export` field says `models/tinker_export/wp-reasoning-v1.3` (a different path), and the directory that IS populated and correct is the already-merged full checkpoint at `models/_staging/qwen3-30b-wp-v1.3-merged/` (57GB, verified present, 13 shards). The bare LoRA archive for s1 was apparently not retained (or was cleaned up) after the merge — only the merged output survives.
**How to avoid:** Before writing export tasks for s0/s2, verify on disk exactly what artifact currently exists for s1 (`ls models/_staging/qwen3-30b-wp-v1.3-merged/`, confirmed 57GB/13 shards) and treat that as the ground truth for what "export + merge" should produce for s0/s2, not the CONTEXT.md prose path.
**Warning signs:** A task with a precondition check on `models/tinker_export/v1.3/adapter_config.json` that will fail immediately.

## Code Examples

### Reusable profiling invocation (Pattern 1)
```bash
# Source: .claude/skills/wp-finetune:run-profiling/SKILL.md, generalized
bash deps/dgx-toolbox/containers/ngc-pytorch.sh python3 -m scripts.profile_merged_model \
  --model-path <merged-judge-seed-path> \
  --ratio ratio_30_70 \
  --output-dir output/sieve/judge-<seed>
```

### Existing CI-aware gate pattern to model TOST after
```python
# Source: scripts/bootstrap_gate.py (existing, verified in codebase)
# def bootstrap_spearman_improvement(...) and def check_wpbench_gate(...)
# both already implement "compute a CI, require the CI bound (not the point
# estimate) to clear a pre-registered bar" — TOST should follow this same
# CI-aware disposition convention (D-09), just with a two-one-sided-test
# formulation instead of a single-sided CI-lower check.
```

### Existing merge-format verification test to extend, not bypass
```python
# Source: tests/phase4_4/test_tinker_merge_convention.py (existing)
# "would PASS a broadcast bug and FAIL a correct merge -- do NOT use it"
# (merge_tinker_v3.py:26) — any new per-seed merge invocation MUST run
# against this same test suite before being trusted for s0/s2.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Unsloth-based local LoRA training on DGX Spark | Tinker cloud-based LoRA training with local export/merge | 2026-06-07 ("Tinker cloud-LoRA pivot") | Phase 11 planning must NOT assume `dgx.execute("unsloth_studio", ...)` per the stale ROADMAP text — training (if it happens at all) would go through the Tinker driver pattern (`scripts/tinker_reasoning_sft.py`), and serving/merging goes through the `merge_tinker_v3.py`-style pattern, not Unsloth's adapter merge. |
| Single-model ship target | Two-model pair (frozen gen + judge ensemble) | 2026-07-08 amendment | Every downstream phase (11-15) must handle two independent compression/packaging tracks, not one. |
| RL-trained model as MoE-Sieve input | v1.2 SFT model as MoE-Sieve input | 2026-07-03 amendment | Any RL-specific profiling/reward-aware routing logic referenced in the original ROADMAP text is moot for this phase. |

**Deprecated/outdated:**
- ROADMAP Phase 11 prose describing `wp-finetune:run-sieve-training` as a to-be-created skill extending `run-training`/Unsloth: this predates both amendments and should be read as historical intent, not current instruction. The skill does not exist yet (`ls .claude/skills/` confirmed no `run-sieve-training` directory) — the planner is genuinely creating this from scratch, informed by the two amendments, not by the original stale spec.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `scipy`/`statsmodels` availability in the eval-toolbox or ngc-pytorch container is not directly verified this session | Standard Stack | If neither is present, TOST must be hand-rolled from scratch (small effort, but adds a verification task the plan should include) |
| A2 | The recommended re-scope (training-free k-sweep via expert masking) is presented as a recommendation, not a locked fact — it reinterprets SIEVE-01/02/04 language | Summary, Open Questions | If the user actually wants literal retraining despite the freeze (e.g., decides the freeze was gen/judge-model-specific, not Sieve-specific), the plan built on this research would need rework |
| A3 | Gen-side routing has not shifted since Phase 7 profiling, because "wp_gen data unchanged" per `PROMOTED_v1.3.json` | Architecture Patterns, Pitfall 2 | If some other change (e.g. a merge/quantization step) altered gen weights since Phase 7, reusing the old profile would silently misclassify hot/cold experts for the gen side |
| A4 | `merge_tinker_v3.py`'s functions generalize cleanly to s0/s2 without v3-specific assumptions beyond the hardcoded default paths | Architecture Patterns, Don't Hand-Roll | The script was written and tested for exactly one checkpoint (v3); undiscovered v3-specific assumptions (e.g. adapter shape, epoch count, target module list) could require more than path-parameterization to reuse for s0/s2 |

## Open Questions

1. **Does "no further training on this base" preclude MoE-Sieve's literal LoRA-retraining-of-hot-experts step, and if so, how should SIEVE-01/02/04 be satisfied?**
   - What we know: CONTEXT.md states the constraint as an unconditional hard fact tied to the closed gap-closure investigation ("all levers negative"). The ROADMAP's Phase 11 section (pre-dating both amendments) describes MoE-Sieve as fundamentally a training procedure (LoRA on hot experts, frozen cold experts, per-k retraining loop). These two are in direct tension.
   - What's unclear: Whether the freeze applies to "further training of the SHIP model's core capability" (gen/judge quality) specifically, or to ALL further gradient updates on this base including compression-prep training. The gap-closure investigation's 3 tested levers (capacity, loss-reshape, data-cleaning) were all attempts to improve JUDGE RHO — none of them were MoE-Sieve-style selective-expert training for compression. It is plausible the freeze was scoped to "don't try to raise quality further," not to "don't do any training-shaped step ever again."
   - Recommendation: Ask the user directly during `/gsd-discuss-phase` or plan review: "Does the training freeze block MoE-Sieve's hot-expert LoRA retraining step, or only further quality-improvement training?" If blocked (the conservative reading, and this research's default assumption), re-scope SIEVE-01/02/04/05 to a training-free routing-analysis + expert-masking pipeline as detailed in this document's Summary and Architecture sections. If NOT blocked, the original ROADMAP Phase 11 spec (train hot experts, k-sweep by retraining) can proceed largely as originally written, informed by the Tinker-pivot pattern instead of the stale Unsloth pattern.

2. **Do the 3 judge seeds route similarly enough for one Sieve profile to cover all 3?**
   - What we know: All 3 seeds share the same base + same rank-32 MoE-only LoRA recipe, differing only by training seed and (per `eval_seed_curve.json`) produce noticeably different rho (0.7957 / 0.8274 / 0.7901) — meaning the seeds are NOT numerically identical in behavior, which is suggestive (but not proof) that routing could differ too.
   - What's unclear: No cross-seed routing comparison currently exists — this is new analysis Phase 11 must produce (Architecture Pattern: cross-seed E_eff/Jaccard overlap, reusing existing Jaccard primitives).
   - Recommendation: Budget a cheap first step — profile all 3 seeds, compute pairwise Jaccard overlap per layer (same math as `compute_jaccard_stability`, applied seed-vs-seed instead of subsample-vs-full) — and let that number (not assumption) decide 1-vs-3-profile Sieve design. This is explicitly called out as "cheap" in CONTEXT.md and should be an early Wave-0 task, not deferred.

3. **What disk footprint is actually required to export + merge s0/s2 alongside s1's existing 57GB merge, and does the machine have headroom?**
   - What we know: s1's merge alone is 57GB; the base model is separately ~63GB; the gen merge is separately present.
   - What's unclear: Total available disk was not checked this session (out of scope of the file-reading task, but the planner should check `df -h` before committing to a plan that produces 2 more 57GB merges).
   - Recommendation: Add a Wave-0 environment check task for disk headroom, similar in spirit to the existing memory pre-check pattern in `wp-moe.md` (`/proc/meminfo` check before training).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| CUDA / GPU (GB10) | Profiling, merge validation, k-sweep serving | ✓ | GB10 (nvidia-smi confirmed) | — |
| Docker | Containerized profiling/serving | ✓ | 29.2.1 | — |
| `tinker` Python SDK (`.venv-tinker`) | s0/s2 checkpoint export | Not directly re-verified this session; prior phases (04.3/04.4, relabel SFT) used it successfully as recently as 2026-07-04 (`PROMOTED_v1.3.json` promoted_utc) | — | None needed — same environment as the still-active relabel SFT work |
| `peft`/`transformers` in DGX containers | Merge, profiling | Not present on host (by design — containers own these); confirmed used successfully in Phase 7 (2026-06-14/15) and Phase 4.3/4.4 merges | container-pinned | — |
| `statsmodels` (optional, for TOST) | SIEVE-05 gate | Not verified this session | — | Hand-roll TOST (two one-sided t-tests), same effort class as existing `bootstrap_gate.py` functions |
| Disk headroom for 2 additional 57GB merges | s0/s2 export+merge | Not checked this session | — | Check with `df -h` in Wave 0; if insufficient, prioritize cross-seed overlap check first (Open Question 2) to determine whether all 3 merges are even necessary, or serve/keep only a subset |

**Missing dependencies with no fallback:** None identified — `tinker` SDK and DGX containers are established, working infrastructure as of the most recent (2026-07-04 to 2026-07-08) work in this project.

**Missing dependencies with fallback:** `statsmodels` (fallback: hand-rolled TOST); disk headroom (fallback: prioritize the cross-seed overlap check to avoid unnecessary merges).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (no `pytest.ini`/`pyproject.toml` config found at project root — defaults apply; `tests/conftest.py` exists) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/test_concentration.py tests/test_bootstrap_ci.py tests/test_jaccard_stability.py -q` (existing Phase 7 tests directly relevant to reused code) |
| Full suite command | `pytest tests/ -q` (362 tests passing as of Phase 7 close; new Phase 11 tests should be added alongside, not replace) |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIEVE-01 | Cross-seed routing profile produced, protected experts subset-verified | unit + integration | `pytest tests/test_sieve_protected_retention.py -x` | ❌ Wave 0 (new) |
| SIEVE-02 | Data-routing spec documented per hot-expert group (if training path retained) OR N/A rationale documented (if masking-only path chosen) | unit | `pytest tests/test_sieve_data_routing.py -x` | ❌ Wave 0 (new, conditional on Open Question 1 resolution) |
| SIEVE-03 | Ratio confirmation (30/70) traces to Phase 4/7 artifacts, no new decision | manual-only (documentation check) | — | Justification: this is a traceability check against existing closed decisions, not new executable behavior |
| SIEVE-04 | K-sweep at 13/32/64 executes and produces per-k wp-bench scores | integration | `pytest tests/test_sieve_kswee_mask.py -x` (mocked masking logic) + manual DGX run for real wp-bench numbers | ❌ Wave 0 (new) |
| SIEVE-05 | TOST gate correctly identifies optimal k | unit | `pytest tests/test_tost_gate.py -x` | ❌ Wave 0 (new) |

### Sampling Rate
- **Per task commit:** `pytest tests/test_<new-file>.py -x` (fast, mocked)
- **Per wave merge:** `pytest tests/ -q` (full suite, catches regressions in reused Phase 7 code)
- **Phase gate:** Full suite green before `/gsd-verify-work`, plus at least one real (non-mocked) DGX profiling run and one real wp-bench k-sweep pass before declaring SIEVE-04/05 complete — mocked tests alone cannot validate actual routing/quality behavior on real hardware.

### Wave 0 Gaps
- [ ] `tests/test_sieve_protected_retention.py` — covers SIEVE-01 (protected-expert subset check across new hot/cold classification)
- [ ] `tests/test_tost_gate.py` — covers SIEVE-05 (TOST logic, modeled on existing `test_bootstrap_ci.py` conventions)
- [ ] `tests/test_sieve_kswee_mask.py` — covers SIEVE-04 (expert-masking-at-inference logic, mockable without GPU)
- [ ] Cross-seed overlap script + its own test — supports Open Question 2, not a formal requirement but load-bearing for the phase's central design decision
- [ ] Disk-headroom / environment pre-check script (pattern-matched to `wp-moe.md`'s `/proc/meminfo` training pre-check) — supports Pitfall 4 / Open Question 3

## Security Domain

`security_enforcement` is absent from `.planning/config.json` (default: enabled).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Phase 11 has no user-facing auth surface — internal DGX/Tinker pipeline only |
| V3 Session Management | No | Not applicable |
| V4 Access Control | Marginal | Tinker checkpoint archive URLs are signed and time-expiring (`expires_at` field, verified in `tinker_export_checkpoint.py`) — no additional control needed, already handled by Tinker's own signed-URL mechanism |
| V5 Input Validation | Marginal | `tinker_export_checkpoint.py` already validates archive integrity (size floor, tarfile format check) before use — extend the same validation discipline to any new merge/export scripts for s0/s2 |
| V6 Cryptography | No | No new cryptographic code — reuse Tinker SDK's signed-URL handling, never hand-roll |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silently regenerating/overwriting the immutable protected-expert mask | Tampering | Hard constraint already documented; add an explicit file-existence + checksum guard before any Phase 11 script writes to `output/profiling/reasoning-merged-v4/` (mirrors the existing path-collision guard in `profile_merged_model.py` for the base-model path) |
| Merge script silently applying a broadcast bug (all experts get an identical, wrong delta) instead of a correct per-expert merge | Tampering (data integrity, not security in the classic sense, but a correctness-critical failure mode this project explicitly guards against) | Reuse `sentinel_agreement`/`spearman_agree` verification functions from `merge_tinker_v3.py` and the existing `tests/phase4_4/test_tinker_merge_convention.py` suite for any new s0/s2 merge invocation |
| Downloading a Tinker archive whose signed URL has expired mid-download, or a corrupted/truncated tar being silently accepted | Tampering / Denial of Service | Already mitigated in `tinker_export_checkpoint.py` (size floor check, `tarfile.open` format verification) — reuse verbatim, don't bypass |

## Sources

### Primary (HIGH confidence — verified this session via direct file reads / grep / bash checks)
- `.planning/phases/11-compression-packaging/11-CONTEXT.md` — governing constraints for this phase
- `.planning/REQUIREMENTS.md` — SIEVE-01..05 exact wording
- `.planning/STATE.md`, `.planning/ROADMAP.md` (incl. both 2026-07-03 and 2026-07-08 amendments, Phases 11-15 success criteria lines 591-684)
- `wp-moe.md` — MoE-Sieve/AIMER/REAP method reference (project-internal spec, no external paper)
- `output/relabel/gap_closure_summary.json`, `eval_seed_curve.json`, `gate_noise_floors.json`
- `output/tinker/PROMOTED_v1.3.json`
- `output/profiling/reasoning-merged-v4/protected_expert_mask.json` (incl. `layer_stability_notes`)
- `.planning/phases/07-router-profiling-protected-expert-set/07-01-SUMMARY.md`, `07-02-SUMMARY.md`
- `.claude/skills/wp-finetune:run-profiling/SKILL.md`
- `output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md` — GB10 memory wall, primary empirical data
- `scripts/serve_30_70_vllm.sh`, `scripts/serve_reasoning_v3_vllm.sh` — vLLM modules_to_save / LoRA-serving constraints
- `scripts/merge_tinker_v3.py`, `scripts/tinker_export_checkpoint.py`, `scripts/tinker_reasoning_sft.py` — Tinker export/merge/train pattern
- `.planning/phases/04.3-reasoning-fine-tune-inserted/04.3-P4-TINKER-RESULTS.md`, `04.3-CONTEXT.md` — Unsloth-to-Tinker pivot rationale
- `.planning/config.json` — nyquist_validation enabled, security_enforcement absent (default enabled)
- Direct disk checks: `models/_staging/qwen3-30b-wp-v1.3-merged/` (57GB, 13 shards, confirmed present); `models/tinker_export/v1.3/` (confirmed EMPTY, contradicting CONTEXT.md prose path)
- Direct environment checks: `nvidia-smi` (GB10 confirmed), `docker --version` (29.2.1), `torch.cuda.is_available()` (True)

### Secondary (MEDIUM confidence)
- None beyond the primary sources above — this phase's research was entirely codebase-grounded; no external web search was needed or performed (MoE-Sieve/AIMER/REAP are project-internal methods with no external reference implementation to verify against).

### Tertiary (LOW confidence)
- `statsmodels`/`scipy` availability in eval-toolbox container — not verified this session, flagged as assumption A1.

## Metadata

**Confidence breakdown:**
- Standard stack / serving constraints: HIGH — directly verified via code (`merge_tinker_v3.py`, `serve_30_70_vllm.sh`) and primary empirical data (`MEMORY-INVESTIGATION-bf16.md`)
- Architecture (reuse of Phase 7 infra): HIGH — profiling/concentration/mask scripts are test-certified (362 passing tests) and directly reusable
- MoE-Sieve method specifics (SIEVE-01/02/04 training-vs-masking question): MEDIUM/LOW — this is a project-internal method with an unresolved tension against a locked decision; the recommendation is reasoned, not verified, and requires explicit user sign-off
- Pitfalls: HIGH — each pitfall traces to a specific, cited codebase artifact or primary-source empirical finding

**Research date:** 2026-07-08
**Valid until:** ~14 days (fast-moving phase; the training-freeze tension in particular should be resolved before this research goes stale, since a user decision on Open Question 1 will materially change the plan's task list)
