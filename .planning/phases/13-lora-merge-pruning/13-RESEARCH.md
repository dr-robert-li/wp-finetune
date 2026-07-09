# Phase 13: LoRA Merge & Pruning (AIMER primary, REAP optional) - Research

**Researched:** 2026-07-10
**Domain:** MoE weight-level expert pruning (post-merge checkpoint surgery), calibration-free vs calibration-based saliency ranking, vLLM serving mechanics for Qwen3-30B-A3B
**Confidence:** HIGH (both pruning methods are verified published papers with formulas already pinned in-repo; gate-before-remove machinery is existing, tested code; the vLLM-checkpoint-shape constraint is verified against the installed transformers source)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Inherited verdicts (LOCKED — do not relitigate)

1. **optimal_k = full (Phase 11, human sign-off 2026-07-10).** NO expert-count compression headroom
   at k<=64: wp-bench -22pp at k=64, judge collapses to 0/121 parseable at k<=32. Cause: E_eff ~88-99
   active experts/layer of 128. `output/sieve/prune_set_for_phase13.json` is the binding handoff.
2. **Ship pair:** v1.2 gen (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`, wp-bench 0.4484 vLLM) +
   v1.3 3-seed judge ensemble (merged s0/s1/s2 under `models/_staging/`, ens rho 0.8075 vLLM).
   Single-seed s1 fallback pre-authorized.
3. **vLLM serving gap:** Tinker-native numbers are sampler-specific; ALL Phase 13+ gates reference the
   vLLM-measured baselines (0.4484 / 0.8075) per `output/sieve/sanity_gate_recalibration.json`.
4. **Protected mask inviolable:** 1,480 experts ([48,128] bool, sha-pinned). NO pruning method may
   remove a protected expert. `layer_stability_notes`: low-Jaccard band {9,13,14,31,35,36} + late
   layers {45,46,47} carry a pre-committed **median-threshold (2,477-expert) headroom** obligation —
   pruning on those layers must be more conservative.
5. **MERGE-01 is largely moot:** there are no unmerged adapters left — RL was rejected (no RL LoRA)
   and the Sieve was training-free (no Sieve LoRA). Gen and all 3 judge seeds are ALREADY merged
   full checkpoints. MERGE-01 closes with an N/A-style traceability record, not new work.

### HARD CONSTRAINTS

1. Protected mask excluded from every candidate prune set (both methods, every ratio). Verify subset
   property programmatically per ratio; mask files byte-unchanged (sha check).
2. **PRUNE-03 gate-before-remove:** evaluate every method×ratio via GATING MASK first (the Phase 11
   `scripts/sieve_expert_mask_inference.py` machinery is exactly this — reuse it). Physical weight
   removal (PRUNE-06) happens ONLY for the winning variant after PRUNE-05 selection.
3. Regression bars (vLLM, like-for-like): gen wp-bench >= 0.4484 - 2pp; judge ensemble rho >= 0.8075 -
   0.052 (two-SE floor per `optimal_k.json` convention). Judge parse-rate must stay >= 95% (121-item
   val) — the k-sweep showed parse collapse is the judge's failure mode.
4. GB10 memory wall: one ~60GB model resident at a time; sequential serve/swap. In-process bf16 load
   peaks ~100GiB (2x staging transient).
5. No training of any kind. Pruning + router renormalization only.
6. Judge ensemble = 3 seeds sharing one routing profile (cross-seed Jaccard 0.933, `sieve_profile_mode
   = shared`). A single shared prune-set must be validated on ALL 3 seeds (ensemble rho gate), not
   just s1.

### Claude's Discretion

- Whether gen and judge get DIFFERENT prune sets or one shared set for operational simplicity
  (research recommendation below; final call is the planner's / human's).
- Whether REAP runs at all, and at how many ratios, given the ceiling established by AIMER@25%.
- Exact script/module layout for the new AIMER/REAP scorers (no existing convention to violate;
  follow the `scripts/sieve_*` naming and self-check pattern already established in Phase 11).

### Deferred Ideas (OUT OF SCOPE)

- None recorded in 13-CONTEXT.md beyond the inherited verdicts above.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MERGE-01 | Merge MoE-Sieve + RL LoRA adapters into base model weights before pruning | **N/A — traceability only.** No unmerged adapters exist (RL rejected, Sieve training-free). Gen = `models/qwen3-30b-wp-30_70-reasoning-merged-v4` (already a full merged checkpoint, produced by `scripts/merge_adapter.py`). Judge = `models/_staging/qwen3-30b-wp-v1.3-{merged,s0-merged,s2-merged}` (3 already-merged seed checkpoints). Plan should emit a short model-card-style traceability note, not run new merge code. |
| PRUNE-01 | AIMER pruning at 25/50/75% compression, weight-based, no calibration, ~1 sec | AIMER is a real, verified published method (arxiv 2603.18492). Formula and computation path below (`## Standard Stack`, `## Code Examples`). Computable directly from merged-checkpoint safetensors, no GPU/forward-pass required. |
| PRUNE-02 | Optional REAP pruning with WordPress calibration data, same 3 ratios | REAP is a real, verified published method (arxiv 2510.13999, ICLR 2026, CerebrasResearch/reap). Requires forward-pass calibration; cost estimate and hook-reuse pattern below. Recommend gating REAP's execution on AIMER@25% passing first (CONTEXT's own recommendation) — see `## Open Questions`. |
| PRUNE-03 | Gate both methods via gating mask before weight removal, 6-variant comparison table | `scripts/sieve_expert_mask_inference.py`'s `build_ksweep_mask`/`apply_mask` is directly reusable with ZERO code changes — see `## Architecture Patterns` Pattern 1. |
| PRUNE-04 | Domain specificity analysis: AIMER vs REAP expert-overlap per layer | Straightforward set-overlap (Jaccard) over the two methods' per-layer keep-masks at matched ratios; same pattern as `scripts/sieve_cross_seed_overlap.py` already in the repo. |
| PRUNE-05 | Select winning method+ratio, prefer higher compression at equivalent quality, reduce ratio incrementally if any dimension regresses | Regression bars and D2_security floor rule already defined in CONTEXT hard constraints; sequencing recommendation in `## Common Pitfalls` / `## Open Questions`. |
| PRUNE-06 | Physical expert removal + router re-normalization, HF-compatible checkpoint | **Critical finding, HIGH confidence:** `transformers`' Qwen3MoE implementation stores `config.num_experts` as a single scalar applied uniformly to every layer's expert-weight tensor shape. Physical pruning MUST keep the SAME NUMBER of experts in every layer (identity may differ per layer); see `## Architecture Patterns` Pattern 3 and `## Common Pitfalls`. |
</phase_requirements>

## Summary

Both pruning methods this phase needs are real, recently published, verified techniques — not
repo inventions. **AIMER** ("Calibration-Free Task-Agnostic MoE Pruning," arxiv 2603.18492) is a
pure weight-norm ranking: `score(expert) = P / sqrt(N*Q)` where P is the L1 norm, N the parameter
count, and Q the squared Frobenius norm, computed over each expert's gate/up/down projection
tensors. It needs no forward passes and no calibration data — it is a closed-form computation over
the merged checkpoint's safetensors, taking ~1-2 seconds per model regardless of ratio. **REAP**
("Router-weighted Expert Activation Pruning," arxiv 2510.13999, accepted ICLR 2026, reference repo
`github.com/CerebrasResearch/reap`) scores each expert by `S_j = mean_{x in active(j)}(g_j(x) *
||f_j(x)||_2)` — the product of router gate weight and expert output norm, averaged over calibration
tokens where that expert is active. It requires a forward-pass calibration sweep and is the
domain-aware comparison method. Both formulas are already pinned verbatim in `wp-moe.md` and match
the published papers exactly.

The single most load-bearing finding for this phase is that **the exact same masking function
already built in Phase 11 (`scripts/sieve_expert_mask_inference.py::build_ksweep_mask`) works for
AIMER/REAP ranking with literally zero code changes** — it is generic over "the array being
ranked" (routing counts today, AIMER/REAP scores tomorrow) and its `k` parameter is already the
per-layer keep-count semantic that 25/50/75% compression needs (k=96/64/32 of 128). `apply_mask`
needs no changes at all. This means PRUNE-03's gate-before-remove step is close to zero new
inference-time code — the vLLM router-logit patch, the wp-bench driver, and the judge-capture HTTP
client from Phase 11 all reuse directly.

The second load-bearing finding is a hard constraint on PRUNE-06: `transformers`' Qwen3MoE
`config.num_experts` is a single integer applied identically to every decoder layer's expert-weight
tensor (verified directly against the installed library source, not assumed). Physical pruning
therefore CANNOT vary the number of kept experts per layer — every layer must keep the same count
(e.g., 96/128 at 25% compression) even though WHICH experts are removed differs per layer. The
`layer_stability_notes` conservative-treatment obligation for the low-Jaccard/late-layer band must
be implemented by biasing WHICH experts are protected within that layer's fixed budget (favoring the
Phase 7 median-threshold 2,477-expert set on those 9 layers specifically), not by giving those layers
a larger keep-count.

Reconciling the CONTEXT's flagged E_eff ambiguity: there are NOT two different stimuli in play. Both
"mean ~60.7/72.7" and "max ~88-99" numbers come from the SAME Phase 7/11 concentration report
(`eeff_wp_gen`/`eeff_wp_judge` columns, `.mean` vs `.max` fields respectively) — mean is the typical
per-layer usage, max is the worst-case (bottleneck) layer. Since a uniform per-layer keep-count must
survive the worst layer, the MAX column is the correct ceiling reference. At 25% compression
(96 kept), gen's worst layer (max ~88) has headroom, but judge's worst layer (max ~99) does not —
this is a genuine, specific risk flagged in `## Common Pitfalls` below, not resolved by this
research (the gate-before-remove eval will catch it empirically if it manifests).

**Primary recommendation:** Run AIMER-only at 25% first (both models, gen and judge separately),
gate via the reused Phase 11 masking + eval machinery, and only proceed to 50%/75% or REAP if 25%
passes cleanly. This matches the CONTEXT's own stated expectation ("≤25% or nothing") and avoids
burning ~20+ hours of sequential GB10 wall-clock on ratios/methods that the E_eff evidence already
predicts will fail.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| AIMER scoring (weight norms) | Offline script (CPU, safetensors read) | — | Pure tensor-math over checkpoint weights; no serving, no GPU strictly required |
| REAP scoring (calibration forward pass) | Offline script (GPU forward hooks) | vLLM/HF model loaded once | Needs router gate values + expert output norms per calibration token; reuses the `RoutingCollector` hook pattern from `scripts/profile_base_model.py`, extended to capture activations not just counts |
| Gate-before-remove masking | vLLM serving (inference-time router-logit patch) | — | `scripts/_sieve_vllm_patch/sitecustomize.py` — already built, reused unmodified |
| Eval (wp-bench, judge capture) | vLLM serving + HTTP eval clients | Docker (wp-env-runtime containers) | Existing `eval.eval_gen`/`eval.eval_judge`/`scripts/sieve_capture_judge_http.py` |
| Physical expert removal (PRUNE-06) | Offline script (CPU, safetensors write) | — | Checkpoint surgery: slice expert tensors + router weight rows, rewrite `config.json`; no serving involved |
| Domain-specificity overlap (PRUNE-04) | Offline script (CPU, set ops on boolean masks) | — | Same pattern as `scripts/sieve_cross_seed_overlap.py` |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | 2.x (bundled w/ `.venv-tinker` transformers 5.5.3 install) | Tensor ops for AIMER norms + REAP forward hooks | Already the project's tensor runtime; `merge_adapter.py` and all `scripts/sieve_*` modules already depend on it |
| `numpy` | already installed | Mask boolean arrays, per-layer score arrays | Same convention as `sieve_expert_mask_inference.py` |
| `safetensors` | bundled with `transformers` | Direct weight-tensor reads for AIMER (no full model instantiation needed) | AIMER's own selling point is sub-second scoring; loading via `safetensors.safe_open` avoids the ~60GB full-model CPU load merge_adapter.py otherwise requires |
| `transformers` | 5.5.3 (verified installed, `.venv-tinker`) [VERIFIED: local install] | Model/config loading, `Qwen3MoeForCausalLM` reference for checkpoint surgery shapes | Already the project's model library |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scipy.stats` | already installed (used by `scripts/tost_gate.py`) | Not needed for pruning itself, but reuse the hand-rolled TOST pattern if PRUNE-05 wants an equivalence test between ratios | Only if PRUNE-05's "reduce compression until clean" rule is formalized as a statistical test rather than a fixed regression-bar check |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolling AIMER/REAP scoring in `scripts/` | Clone `github.com/CerebrasResearch/reap` and use its CLI | REAP repo requires `uv`/Docker build, no pip package, and is oriented around its own experiment harness (`experiments/pruning-cli.sh`) with lm-eval/EvalPlus/WildBench dependencies this project doesn't use. Both formulas are ~5-10 lines of tensor math each (already spelled out in `wp-moe.md`); hand-rolling matches the project's existing precedent (`scripts/tost_gate.py` hand-rolled TOST because `statsmodels` was absent) and avoids a heavyweight, off-convention dependency for a well-understood formula. |
| Full-model CPU load for AIMER | `safetensors.safe_open` direct tensor read | AIMER's entire value proposition is sub-second scoring; loading the full ~60GB model via `AutoModelForCausalLM` (as `merge_adapter.py` does) to then only read norms is wasteful. Direct safetensors access reads only the `mlp.experts.{gate_up,down}_proj` tensors needed. |

**Installation:** None. No new packages required — everything needed is already in `.venv-tinker`.

**Version verification:** `transformers==5.5.3` confirmed installed via `.venv-tinker/lib/python3.13/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py` (read directly, not assumed). `vllm` is NOT installed in `.venv-tinker` — it runs only inside the `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` Docker image used by `scripts/serve_30_70_vllm.sh` and the DGX Toolbox `start-vllm.sh`. Any PRUNE-06 checkpoint-shape work should be verified against the `transformers` Qwen3MoE source (confirmed above) since that is what both HF and vLLM's Qwen3MoE loader derive their tensor shapes from.

## Package Legitimacy Audit

**No new external packages required for this phase.** AIMER and REAP are both implemented as
hand-rolled tensor-math scripts reusing `torch`/`numpy`/`transformers`/`safetensors`, all already
installed and used by existing repo scripts (`merge_adapter.py`, `sieve_expert_mask_inference.py`).
No `pip install` / `npm install` / new registry dependency is introduced by this phase's plan.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none — no new packages) | — | — | — | — | — | N/A |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

*If the planner later decides to clone `github.com/CerebrasResearch/reap` directly instead of
hand-rolling (see Alternatives Considered above), that decision must go through this gate: it is a
GitHub source clone, not a registry package, so `npm view`/`pip index` checks do not apply — verify
instead via repo star count / commit recency / Cerebras org ownership (already corroborated: it is
the official reference implementation cited by the arxiv paper and mirrored to HuggingFace as
`cerebras/*-REAP-*` production model releases) before trusting its code.*

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────┐
                     │  Merged checkpoints (exist)  │
                     │  gen: reasoning-merged-v4    │
                     │  judge: v1.3 s0/s1/s2 (3x)   │
                     └──────────────┬──────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
     ┌────────▼─────────┐                       ┌─────────▼──────────┐
     │  AIMER scorer     │                       │  REAP scorer        │
     │  (safetensors read,│                      │  (forward-hook      │
     │   weight norms,    │                      │   calibration pass, │
     │   ~1-2 sec)         │                      │   WordPress data,   │
     │                     │                      │   ~hours)            │
     └────────┬────────────┘                      └─────────┬───────────┘
              │  scores[layer, expert]                        │  scores[layer, expert]
              └───────────────────┬────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ build_ksweep_mask(scores,   │  <- REUSED UNCHANGED from
                    │   protected_mask, k=96/64/32)│     scripts/sieve_expert_
                    │  per (method, ratio, model) │     mask_inference.py
                    └─────────────┬───────────────┘
                                  │  keep_mask [48,128] bool
                    ┌─────────────▼───────────────┐
                    │ vLLM serve + SIEVE_MASK_NPY   │  <- REUSED UNCHANGED:
                    │ (_sieve_vllm_patch router-    │     scripts/serve_30_70_
                    │  logit -inf patch, softmax    │     vllm.sh + sitecustomize.py
                    │  renormalizes automatically)  │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┴────────────────────┐
              │                                        │
    ┌─────────▼──────────┐                  ┌──────────▼───────────┐
    │ wp-bench (gen axis) │                  │ judge capture x3 seeds│
    │ eval.eval_gen        │                  │ sieve_capture_judge_  │
    │                       │                  │ http.py + eval_judge  │
    └─────────┬────────────┘                  └──────────┬───────────┘
              │                                          │
              └───────────────────┬──────────────────────┘
                                  │  6 variants x {gen, judge} results
                    ┌─────────────▼───────────────┐
                    │ PRUNE-03 comparison table     │
                    │ PRUNE-04 AIMER vs REAP overlap│
                    │ PRUNE-05 select method+ratio  │
                    └─────────────┬───────────────┘
                                  │  winning (method, ratio)
                    ┌─────────────▼───────────────┐
                    │ PRUNE-06 physical surgery:    │
                    │  slice expert tensors +       │
                    │  router.weight rows per layer │
                    │  (UNIFORM count/layer),        │
                    │  rewrite config.num_experts    │
                    └──────────────────────────────┘
```

### Recommended Project Structure

```
scripts/
├── aimer_prune.py            # NEW: safetensors weight-norm scoring, self-check like sieve_*
├── reap_prune.py              # NEW (only if PRUNE-02 runs): calibration forward-hook scoring
├── prune_overlap.py           # NEW: PRUNE-04 domain-specificity analysis (Jaccard per layer,
│                               #      same shape as sieve_cross_seed_overlap.py)
├── prune_apply_physical.py    # NEW: PRUNE-06 checkpoint surgery (slice + renormalize + rewrite config)
├── sieve_expert_mask_inference.py  # REUSED unchanged: build_ksweep_mask / apply_mask
├── sieve_capture_judge_http.py     # REUSED unchanged: judge HTTP capture
├── serve_30_70_vllm.sh              # REUSED unchanged (already has SIEVE_MASK_NPY hook)
└── _sieve_vllm_patch/sitecustomize.py  # REUSED unchanged: router-logit masking patch
```

### Pattern 1: Ranking-source-agnostic masking (zero-diff reuse)

**What:** `build_ksweep_mask(counts, protected, k)` takes ANY per-layer score array (higher = keep)
and a per-layer keep budget `k`; it is not routing-specific despite the name.
**When to use:** Every gate-before-remove step in PRUNE-03, for both AIMER and REAP scores, at all
three ratios.
**Example:**
```python
# Source: scripts/sieve_expert_mask_inference.py (existing, Phase 11)
from scripts.sieve_expert_mask_inference import build_ksweep_mask, apply_mask

# AIMER scores replace routing "counts" 1:1 — same function, same contract:
aimer_scores = compute_aimer_scores(merged_checkpoint_path)  # [48, 128] float, higher = keep
keep_mask_25pct = build_ksweep_mask(aimer_scores, protected_mask, k=96)  # 128*0.75=96
keep_mask_50pct = build_ksweep_mask(aimer_scores, protected_mask, k=64)
keep_mask_75pct = build_ksweep_mask(aimer_scores, protected_mask, k=32)
# apply_mask() at inference time is IDENTICAL to the k-sweep's usage — no changes needed.
```
The only new code needed is `compute_aimer_scores()`/`compute_reap_scores()` — the masking/gating
machinery itself needs no modification. Recommend renaming `build_ksweep_mask` to a more general
`build_prune_mask` at the call site (or via an alias import) for plan-readability, but this is
cosmetic, not functional.

### Pattern 2: AIMER scoring (closed-form, from wp-moe.md's own formula)

**What:** Weight-only importance score per expert, no calibration.
**When to use:** PRUNE-01, primary method, all 3 ratios, both models.
**Example:**
```python
# Formula source: wp-moe.md section "Planned: Expert Pruning (v3.0)" (matches arxiv 2603.18492)
# AIMER(expert) = P / sqrt(N * Q)
#   P = sum(|w_i|)     L1 norm across gate_proj + up_proj + down_proj for this expert
#   N = param count    same three tensors
#   Q = sum(w_i^2)      squared Frobenius norm, same three tensors
import torch
from safetensors import safe_open

def aimer_score_expert(gate_w, up_w, down_w) -> float:
    w = torch.cat([gate_w.flatten(), up_w.flatten(), down_w.flatten()])
    P = w.abs().sum()
    N = w.numel()
    Q = (w ** 2).sum()
    return (P / torch.sqrt(N * Q)).item()  # bounded [1/sqrt(N), 1], scale-invariant
```
Iterate this per (layer, expert) pair by reading the stacked `experts.gate_up_proj`/`down_proj`
tensors directly via `safetensors.safe_open(checkpoint_path, framework="pt")` — do NOT instantiate
the full `AutoModelForCausalLM` (60GB CPU load) just to read weight norms.

### Pattern 3: PRUNE-06 physical removal — uniform per-layer expert count is MANDATORY

**What:** `transformers.models.qwen3_moe.modeling_qwen3_moe` stores `self.num_experts =
config.num_experts` (a single int) and allocates `gate_up_proj`/`down_proj` as
`nn.Parameter(torch.empty(self.num_experts, ...))` — identical shape for every decoder layer.
[VERIFIED: read directly from `.venv-tinker/lib/python3.13/site-packages/transformers/models/
qwen3_moe/modeling_qwen3_moe.py` lines 220-261, this session]
**When to use:** Any physical checkpoint surgery in PRUNE-06.
**Example (the surgery, conceptually):**
```python
# For a target compression ratio (e.g. 25% -> keep 96/128 per layer):
for layer_idx in range(48):
    keep_idx = torch.tensor(sorted(np.where(keep_mask[layer_idx])[0]))  # len == 96 EVERY layer
    assert len(keep_idx) == KEEP_N, "PRUNE-06 requires a uniform per-layer keep count"
    new_gate_up_proj[layer_idx] = old_gate_up_proj[layer_idx][keep_idx]   # slice expert dim
    new_down_proj[layer_idx]    = old_down_proj[layer_idx][keep_idx]
    new_router_weight[layer_idx] = old_router.weight[keep_idx]           # gate.weight rows
# Softmax renormalization is AUTOMATIC once router.weight has fewer rows — no extra code needed
# (this is the same renormalization apply_mask()'s -inf trick achieves at inference time, but here
# the columns are gone entirely rather than logit-masked).
config.num_experts = KEEP_N   # single scalar update, applies to every layer uniformly
```
**Consequence for layer_stability_notes:** the low-Jaccard band {9,13,14,31,35,36} and late layers
{45,46,47} CANNOT get a larger keep-count than other layers (the checkpoint format forbids it).
Conservatism for those 9 layers must instead be expressed as: prefer protecting the Phase 7
median-threshold set (`output/profiling/reasoning-merged-v4/sensitivity_table.json`'s
`median_threshold` variant, `total_protected=2477`, `mask_size_per_layer` ~51-56/layer) on those
specific layers when the AIMER/REAP ranking would otherwise drop one of those experts, at the cost
of dropping a *lower*-ranked non-protected expert elsewhere in that same layer to preserve the
uniform per-layer count. **Note:** `sensitivity_table.json` stores only per-layer COUNTS for the
median-threshold variant, not the actual boolean mask — regenerate the actual per-expert
median-threshold mask via `scripts/extract_protected_mask.py` (same script that produced
`sensitivity_table.json`) if this headroom rule needs the literal expert-index set, not just counts.

### Anti-Patterns to Avoid

- **Re-implementing the masking/union-with-protected logic for AIMER/REAP:** `build_ksweep_mask` is
  already exactly this. A new implementation risks silently diverging from the tested, sha-verified
  protected-expert-union contract (`tests/test_sieve_ksweep_mask.py`).
- **Loading the full model to compute AIMER scores:** defeats AIMER's entire "~1 second" value
  proposition and burns unnecessary GB10 memory headroom; read tensors directly via safetensors.
- **Giving low-Jaccard/late layers a larger per-layer keep-count:** not representable in the
  HF/vLLM Qwen3MoE checkpoint format (`config.num_experts` is one scalar for the whole model).
- **Running REAP on Tinker-native (unmerged) checkpoints:** activation magnitudes must come from
  the SAME served form (vLLM-merged) that PRUNE-06's output will eventually be evaluated against,
  per the CONTEXT's vLLM-vs-Tinker serving-gap lesson from Phase 11.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Top-k-union-protected mask construction | A new AIMER/REAP-specific mask builder | `scripts/sieve_expert_mask_inference.py::build_ksweep_mask` (pass scores instead of counts) | Already tested (3/3 green), sha-verified against the protected mask, handles the "protected expert outside top-k expands the union" edge case correctly |
| Inference-time expert masking at eval | A new vLLM patch | `scripts/_sieve_vllm_patch/sitecustomize.py` + `SIEVE_MASK_NPY` env var | Already smoke-verified on real GB10 hardware; solves two subtle bugs (CUDA-graph capture constraint, device placement) that a naive reimplementation would rediscover |
| Judge response capture + scoring | A new HTTP judge client | `scripts/sieve_capture_judge_http.py` (4x concurrency) + `eval.eval_judge._judge_create` | Already reproduces the canonical Tinker-native rho bit-for-bit against historical captures — proven correct |
| TOST-style equivalence gating (if PRUNE-05 wants a formal statistical test) | A new stats routine | `scripts/tost_gate.py` (hand-rolled Welch-t TOST, `statsmodels` confirmed absent) | Same precedent already established for Phase 11's optimal-k decision |
| REAP reference implementation | Clone+adapt `CerebrasResearch/reap`'s full experiment harness | Hand-roll the ~10-line formula from `wp-moe.md` | The upstream repo's harness targets lm-eval/EvalPlus/WildBench and `uv`/Docker builds this project doesn't use; the formula itself is trivial tensor math already spelled out and needs no external code |

**Key insight:** almost everything PRUNE-01 through PRUNE-04 need already exists from Phase 11 —
this phase is primarily "write two scoring functions" plus "write one checkpoint-surgery script,"
not new infrastructure.

## Common Pitfalls

### Pitfall 1: E_eff mean vs max column confusion (already partially reconciled by this research)

**What goes wrong:** Treating the CONTEXT's "~88-99" ceiling and Phase 7's "60.7/72.7" figure as two
different measurements needing reconciliation, when they are the `.max` and `.mean` fields of the
SAME per-layer E_eff distribution in the SAME report
(`output/profiling/reasoning-merged-v4/concentration_report.json` /
`output/sieve/judge-s0/concentration_report.json`, both verified this session:
`eeff_wp_gen: {mean: 60.7-61.1, max: 87.8-88.0}`, `eeff_wp_judge: {mean: 72.6-72.7, max: 98.9-99.1}`).
**Why it happens:** Different documents quote different fields without naming which one.
**How to avoid:** Always cite mean AND max together when discussing pruning headroom; use max for
"can a uniform per-layer budget survive the worst layer" questions, mean for "typical headroom."
**Warning signs:** A plan or report citing a single E_eff number without specifying mean/max.

### Pitfall 2: 25% compression may still under-provision the judge model's worst layer

**What goes wrong:** 25% compression keeps 96/128 experts per layer. Gen's worst-case layer
(max E_eff ~88) has headroom under 96. **Judge's worst-case layer (max E_eff ~98.9-99.1) does NOT**
— 96 < 99. If that specific layer (layer_idx 0, per the per-layer data read this session, which also
carries 37 protected experts) loses even a few live-but-unprotected experts, judge-axis parse
collapse (the exact failure mode from the k-sweep) is a real risk at 25%, not just at 50/75%.
**Why it happens:** AIMER/REAP rank by WEIGHT NORM or CALIBRATION SALIENCE, not by ROUTING FREQUENCY
— a highly-routed-but-low-weight-norm expert can be pruned even though the E_eff evidence says that
layer needs ~99 active experts to avoid collapse. This is exactly the open question this phase exists
to answer (does weight-level ranking find headroom that routing-coldness could not), so it is not a
reason to change the plan — it IS the risk PRUNE-03's gate-before-remove step exists to catch.
**How to avoid:** Treat judge-axis 25% AIMER results with extra scrutiny on the parse-rate gate
(>=95%, per the HARD CONSTRAINTS) specifically for early/protected-heavy layers; do not assume 25%
"passes because it's conservative" without running the actual gate.
**Warning signs:** Judge parse-rate degradation concentrated in responses that would route heavily
through layer 0 or the low-Jaccard band, even at 25% compression.

### Pitfall 3: Uniform per-layer expert count is a hard checkpoint-format constraint, not a design choice

**What goes wrong:** Assuming layer_stability_notes' "more conservative on these 9 layers" means
"keep more experts in those layers." This is not representable in the HF/vLLM Qwen3MoE format.
**Why it happens:** `layer_stability_notes` reads naturally as a per-layer keep-count adjustment.
**How to avoid:** Implement conservatism via WHICH experts are protected on those 9 layers (bias
toward the Phase 7 median-threshold 2,477-expert set), never via a per-layer count that differs from
the model-wide ratio.
**Warning signs:** Any PRUNE-06 code that reads `config.num_experts` as a per-layer list, or a mask
whose per-layer `sum()` is not identical across all 48 layers for a given ratio.

### Pitfall 4: vLLM-vs-Tinker serving gap re-appearing in gated eval numbers

**What goes wrong:** Comparing a gated (masked) eval result against the CANONICAL Tinker-native
regression bars (0.842/0.827) instead of the vLLM-measured ones (0.8075/0.8017) already established
in `sanity_gate_recalibration.json`.
**Why it happens:** The canonical numbers are more prominently documented across the project history.
**How to avoid:** HARD CONSTRAINT 3 already locks this — every Phase 13 comparison must use the
vLLM-measured baselines. Repeat this explicitly in the plan's acceptance criteria.
**Warning signs:** A regression-bar check referencing 0.842 or 0.4616 rather than 0.8075/0.4484.

### Pitfall 5: Sequential GB10 swap cost compounds fast across 6-12 variants

**What goes wrong:** Underestimating total wall-clock. Phase 11's real-hardware run showed a single
full-arm attempt (vLLM boot + wp-bench + 3-seed judge capture) takes ~90-120 minutes even AFTER
concurrency fixes (12-15 min/seed judge capture with 4x concurrency, ~1hr wp-bench, ~10min vLLM
boot). With 6 variants (2 methods x 3 ratios) x 2 models (if gen/judge get separate prune sets) =
up to 12 full gated-eval cycles, naive sequencing is ~18-24 hours of wall-clock.
**Why it happens:** GB10's single ~60GB-model-at-a-time constraint forces strict serialization —
no parallel gating across variants.
**How to avoid:** Sequence AIMER@25% first (2 model variants, ~4 hours), gate on results before
expanding to 50%/75% or REAP, per this research's Primary Recommendation.
**Warning signs:** A plan that schedules all 6-12 variants unconditionally before any human/gate
checkpoint.

## Code Examples

### AIMER scoring skeleton (direct safetensors read, no full model load)

```python
# Source: formula from wp-moe.md (matches arxiv 2603.18492); tensor-read pattern is new for this
# phase but follows the same safetensors-direct-access style already implicit in merge_adapter.py's
# checkpoint handling.
from safetensors import safe_open
import torch

def compute_aimer_scores(checkpoint_dir: str, n_layers: int = 48, n_experts: int = 128):
    scores = torch.zeros(n_layers, n_experts)
    with safe_open(f"{checkpoint_dir}/model.safetensors", framework="pt") as f:
        for layer in range(n_layers):
            gate_up = f.get_tensor(f"model.layers.{layer}.mlp.experts.gate_up_proj")  # [E, ...]
            down = f.get_tensor(f"model.layers.{layer}.mlp.experts.down_proj")        # [E, ...]
            for e in range(n_experts):
                w = torch.cat([gate_up[e].flatten(), down[e].flatten()])
                P, N, Q = w.abs().sum(), w.numel(), (w ** 2).sum()
                scores[layer, e] = P / torch.sqrt(N * Q)
    return scores.numpy()
```
Note: actual Qwen3MoE tensor key names must be confirmed against the real checkpoint's
`model.safetensors.index.json` before implementation — `gate_up_proj`/`down_proj` naming above
follows the `modeling_qwen3_moe.py` attribute names verified this session, but the on-disk
safetensors key prefix should be checked directly (`python -c "import json; print(list(json.load(
open('.../model.safetensors.index.json'))['weight_map'].keys())[:20])"`) since HF checkpoint key
naming can differ slightly from in-memory attribute naming.

### REAP scoring skeleton (extends the existing RoutingCollector hook pattern)

```python
# Source: pattern extends scripts/profile_base_model.py's RoutingCollector (hooks
# Qwen3MoeTopKRouter gates on all 48 layers) to also capture gate softmax weight + expert
# output norm, not just routing counts. Formula: S_j = mean_{x in active(j)}(g_j(x) * ||f_j(x)||_2)
# (wp-moe.md, matches arxiv 2510.13999).
class REAPCollector:
    """Forward hook on each expert MLP + router: accumulate g_j(x)*||f_j(x)||_2 per active expert."""
    def __init__(self, n_layers, n_experts):
        self.sum_score = torch.zeros(n_layers, n_experts)
        self.count = torch.zeros(n_layers, n_experts)

    def hook(self, layer_idx, expert_idx, gate_weight, expert_output):
        norm = expert_output.norm(dim=-1)              # ||f_j(x)||_2 per token
        self.sum_score[layer_idx, expert_idx] += (gate_weight * norm).sum().item()
        self.count[layer_idx, expert_idx] += gate_weight.numel()

    def scores(self):
        return (self.sum_score / self.count.clamp(min=1)).numpy()
```
Calibration data: reuse `data/reasoning_dataset/openai_val.jsonl` (141 items, `<wp_judge>`-relevant)
for the judge axis, and a sample from `data/final_dataset/ratio_30_70/openai_train.jsonl` (34,855
items — take a representative subsample, e.g. 500-2,000, not the full set) for the gen axis, mixed
per wp-moe.md's "WordPress calibration data (gen + judge examples)" spec. Forward passes must run
on the SAME merged checkpoint (CPU or GPU) that will be masked/pruned — do not calibrate against the
base model or an unmerged adapter stack.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Calibration-based MoE pruning as the default (e.g. earlier REAP-predecessor router-calibration work) | Calibration-free weight-norm ranking (AIMER) shown competitive/stronger at 25-50% ratios across 7B-47B MoE models | AIMER published ~April 2026 (arxiv 2603.18492) | Removes the multi-hour calibration cost as a hard requirement for a first-pass pruning signal; this project's D-09 already made AIMER primary for exactly this reason |
| One-shot expert merging as a compression alternative to pruning | REAP's own paper argues pruning outperforms merging for generative tasks, since merging loses fine-grained routing control | REAP accepted ICLR 2026 (arxiv 2510.13999) | Directly validates this project's PRUNE-06 physical-removal approach over any expert-merging alternative |

**Deprecated/outdated:** none directly relevant — both methods used by this phase are current
(2026) state of the art, and Qwen3-30B-A3B is explicitly one of the models evaluated in REAP's own
published results (per the paper's own listed model families), which is a strong signal this
architecture is a realistic target for the method, not a mismatch.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Recommended calibration-data composition and sample count for REAP (500-2,000 gen examples + the 141-item judge val set) | `## Code Examples` | If too small, REAP scores may be noisy/unrepresentative of true WordPress routing; if too large, the "~hours" cost estimate from `wp-moe.md` could balloon. The reference repo's own composite mix used ~24,576 samples for a much larger model family (480B) — this project's 30B-A3B scale and single-domain (WordPress/PHP) calibration likely needs far fewer, but the exact number is not pinned by any project artifact and should be confirmed at planning time or via a quick pilot run. |
| A2 | Exact safetensors tensor key names (`mlp.experts.gate_up_proj` / `down_proj`) match the on-disk checkpoint format for `models/qwen3-30b-wp-30_70-reasoning-merged-v4` | `## Code Examples`, Pattern 2 | These names are read from the in-memory `modeling_qwen3_moe.py` module attributes (verified), but the ON-DISK safetensors key prefix (e.g. whether it's `model.layers.N.mlp.experts.gate_up_proj` or a differently-prefixed name) was not independently verified against the actual checkpoint's `model.safetensors.index.json` in this research session. Must be confirmed before implementation — low effort (one `json.load` call), high consequence if wrong (silent no-op or crash). |
| A3 | Whether gen and judge should get separate prune sets (recommended: yes, operationally natural given they already ship as separate checkpoints) is presented as a recommendation, not a locked decision | `## Open Questions` | CONTEXT.md explicitly leaves this as an open question for research/planning; the recommendation here is reasoned from the existing two-model-pair shipping architecture, not from new measurement. |

**If this table is empty:** N/A — see entries above.

## Open Questions (RESOLVED — dispositions in the committed plans)

> Q1 RESOLVED: separate prune sets per model — 13-04 gates gen and judge independently (own scores,
> own bars). Q2 RESOLVED: REAP gated on AIMER@25% passing — 13-05 Task 2 conditional branch.
> Q3 RESOLVED: tensor keys verified against the real model.safetensors.index.json — per-expert
> unstacked layout wired into 13-01/13-03/13-07 (research assumption A2 corrected).

1. **Separate vs shared prune set for gen vs judge?**
   - What we know: gen's E_eff (mean 60.7-61.1, max 87.8-88.0) is measurably lower than judge's
     (mean 72.6-72.7, max 98.9-99.1) on the SAME merged checkpoint's routing profile — gen genuinely
     has more headroom. The two models already ship as separate checkpoints (gen: single model;
     judge: 3-seed ensemble sharing one routing profile per `sieve_profile_mode=shared`).
   - What's unclear: whether the operational cost of maintaining two independently-pruned checkpoints
     (2x the PRUNE-01/02/03 variant count, 2x PRUNE-06 physical surgery, 2x model-card documentation)
     is worth the extra compression headroom gen might tolerate, versus a single shared prune set
     applied to both (simpler, but leaves gen's extra headroom unused and NEVER over-prunes judge).
   - Recommendation: separate prune sets, since the two models already require independent gating
     (judge needs all 3 seeds to pass the ensemble rho gate; gen needs only its own wp-bench gate) —
     the eval-variant cost is already doubled by the existing two-model architecture regardless of
     whether pruning is shared. This is Claude's-discretion territory per CONTEXT.md; final call
     belongs to planning/human sign-off.

2. **Does REAP's calibration cost justify running it at all before AIMER@25% results are known?**
   - What we know: CONTEXT.md's own stated recommendation is "run REAP only if AIMER@25% passes
     gates." wp-moe.md's ~3hr calibration estimate (per ratio, potentially per model) makes REAP the
     single most expensive step in this phase if run unconditionally at all 3 ratios x 2 models.
   - What's unclear: the exact calibration-sample count needed for a WordPress-scale domain (A1
     above) — this directly determines whether "~3 hours" holds or is optimistic/pessimistic at this
     project's actual calibration-set size.
   - Recommendation: gate REAP's execution on AIMER@25% passing (as CONTEXT already suggests), and
     pilot REAP's calibration pass on a small sample (e.g. 100-200 examples) first to confirm the
     per-sample forward-pass cost before committing to the full calibration run.

3. **Exact on-disk safetensors key naming for expert tensors (A2 above)** — a five-minute
   verification task at the start of plan execution, not resolvable from documentation alone without
   inspecting the actual checkpoint file.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Docker | vLLM serving for gated eval | Yes | 29.2.1 [VERIFIED: `docker info`] | — |
| `wp-bench` | Gen-axis gating eval | Yes, cloned at `wp-bench/` [VERIFIED: `test -d`] | — | — |
| `.venv-tinker` (torch/transformers/numpy/safetensors) | AIMER/REAP scoring scripts | Yes | transformers 5.5.3 [VERIFIED] | — |
| `vllm` | Gated-mask serving | Only inside `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` Docker image, NOT in `.venv-tinker` | image is "nightly"-tagged, exact vLLM package version not independently pinned this session | Existing `scripts/serve_30_70_vllm.sh` already handles this correctly — no new serving code needed |
| GPU (GB10) | All forward-pass/serving steps | Yes | NVIDIA GB10, unified memory (nvidia-smi reports memory fields as N/A — expected for this Grace-Blackwell architecture, not an error) [VERIFIED: `nvidia-smi --query-gpu`] | — |
| Disk space | New pruned checkpoints (PRUNE-06 only; gating steps write no new model copies) | 1.5TB free [VERIFIED: `df -h /`] | — | Ample headroom; a single pruned ~45GB checkpoint per model is a small fraction of free space |
| Merged checkpoints (gen + 3 judge seeds) | MERGE-01 traceability, all pruning/eval steps | Yes, all present under `models/` and `models/_staging/` [VERIFIED: `ls`] | — | — |
| Protected + median-threshold masks | PRUNE-01/02/06 protected-set enforcement | Protected mask (`.npy`/`.json`, sha-pinned) present; median-threshold mask exists only as PER-LAYER COUNTS in `sensitivity_table.json`, not as a boolean expert-index mask [VERIFIED: `ls` + `json.load`] | — | Regenerate the actual median-threshold boolean mask via `scripts/extract_protected_mask.py` if the layer_stability_notes headroom rule needs literal expert indices, not just counts |

**Missing dependencies with no fallback:** none identified.

**Missing dependencies with fallback:** median-threshold boolean mask (regenerable via existing script, see above).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1, run via `.venv-tinker/bin/python -m pytest` [VERIFIED: `.venv-tinker/bin/python -m pytest --version`] — system pytest is broken for this project per prior sessions; ALWAYS invoke via `.venv-tinker` |
| Config file | none found (no `pytest.ini`/`[tool.pytest.ini_options]`) — plain `tests/` directory auto-discovery |
| Quick run command | `.venv-tinker/bin/python -m pytest tests/test_aimer_prune.py tests/test_reap_prune.py tests/test_prune_overlap.py -x -q` (new Wave-0 test files, once created) |
| Full suite command | `.venv-tinker/bin/python -m pytest tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MERGE-01 | Traceability record only; no new merge logic | manual-only (documentation check) | N/A — assert merged checkpoint paths exist and load | N/A |
| PRUNE-01 | `compute_aimer_scores()` returns `[48,128]` float array, scale-invariant, deterministic | unit | `.venv-tinker/bin/python -m pytest tests/test_aimer_prune.py -x -q` | ❌ Wave 0 |
| PRUNE-01 | AIMER scores + `build_ksweep_mask` at k=96/64/32 never drop a protected expert (reuse existing masking test contract) | unit | `.venv-tinker/bin/python -m pytest tests/test_sieve_ksweep_mask.py -x -q` (existing, reused with AIMER-shaped input) | ✅ (existing) |
| PRUNE-02 | `compute_reap_scores()` returns `[48,128]` float array from a synthetic forward-hook fixture (no GPU) | unit | `.venv-tinker/bin/python -m pytest tests/test_reap_prune.py -x -q` | ❌ Wave 0 |
| PRUNE-03 | Gated eval regression bars enforced (wp-bench >= 0.4484-2pp; judge rho >= 0.8075-0.052; parse-rate >= 95%) | integration (real-hardware, per Phase 11 precedent) | reuse `scripts/sieve_ksweep_run.py`-style driver against AIMER/REAP masks | ❌ Wave 0 (new driver, same pattern) |
| PRUNE-04 | Per-layer Jaccard overlap between AIMER and REAP keep-masks at matched ratios | unit | `.venv-tinker/bin/python -m pytest tests/test_prune_overlap.py -x -q` | ❌ Wave 0 |
| PRUNE-05 | Selection rule applies regression bars + D2_security floor correctly on synthetic gate results | unit | `.venv-tinker/bin/python -m pytest tests/test_prune_selection.py -x -q` | ❌ Wave 0 |
| PRUNE-06 | Physical surgery preserves uniform per-layer expert count; router renormalizes; pruned model loads and produces coherent output | unit (shape/count assertions) + manual (coherent-output smoke check) | `.venv-tinker/bin/python -m pytest tests/test_prune_physical.py -x -q` + manual generate() smoke test | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** quick unit-test run (synthetic fixtures, no GPU, no model load) — matches the
  existing `tests/test_sieve_ksweep_mask.py` / `tests/test_sieve_cross_seed_overlap.py` convention
  (module-level `pytest.importorskip` so tests skip cleanly before the script exists, per Wave-0
  pattern).
- **Per wave merge:** full suite + at least one real-hardware gated-eval smoke run (reuse Phase 11's
  mandatory-real-hardware-run precedent — a scoring bug or masking bug will not surface in synthetic
  unit tests alone).
- **Phase gate:** full suite green + at least the AIMER@25% gated eval (both models) completed with
  a real comparison table, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/test_aimer_prune.py` — covers PRUNE-01 (synthetic weight tensors, no model load)
- [ ] `tests/test_reap_prune.py` — covers PRUNE-02 (synthetic forward-hook fixture)
- [ ] `tests/test_prune_overlap.py` — covers PRUNE-04 (synthetic boolean masks, Jaccard)
- [ ] `tests/test_prune_selection.py` — covers PRUNE-05 (synthetic gate-result table, regression-bar logic)
- [ ] `tests/test_prune_physical.py` — covers PRUNE-06 (synthetic small tensor, shape/count assertions)
- Framework install: none — `.venv-tinker`'s pytest 9.1.1 already covers this

## Security Domain

`security_enforcement` not found set to `false` in `.planning/config.json` — treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | This phase has no auth surface — offline model/checkpoint transformation, no new service endpoints |
| V3 Session Management | No | Same as above |
| V4 Access Control | No | Same as above |
| V5 Input Validation | Marginal | Calibration data (REAP) and checkpoint tensors are read from trusted, already-validated repo paths; no external/untrusted input parsed in this phase |
| V6 Cryptography | No | Protected-mask sha256 checks are integrity verification (already implemented), not cryptographic security controls in the ASVS sense |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silent protected-expert removal (a pruning bug drops a protected expert without erroring) | Tampering | Programmatic subset-property assertion (already HARD CONSTRAINT 1) + sha-pinned mask file checks before every prune-mask construction, matching the existing `sieve_expert_mask_inference.py` precedent (`_self_check()` asserts `np.all((~protected) | kept)`) |
| Quality regression shipped as "pruned model" without the D2_security dimension gate catching it | Tampering / Repudiation | PRUNE-05's explicit preference for D2_security retention + the existing per-dimension `eval_gate.py` gating logic |
| Monkeypatched vLLM internals (`_sieve_vllm_patch`) silently no-op'ing if the mask file is malformed | Tampering | Already addressed in Phase 11 (`sitecustomize.py` "raises loudly, does not silently no-op" per its own threat-flag note) — PRUNE-03 reuses this unchanged |

## Sources

### Primary (HIGH confidence)
- `.venv-tinker/lib/python3.13/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py` — read directly this session; confirms `config.num_experts` single-scalar constraint
- `output/profiling/reasoning-merged-v4/concentration_report.json`, `output/sieve/judge-s0/concentration_report.json` — read directly; source of the mean/max E_eff reconciliation
- `output/profiling/reasoning-merged-v4/sensitivity_table.json` — read directly; confirms median-threshold total_protected=2477, per-layer counts only (no boolean mask)
- `output/sieve/prune_set_for_phase13.json`, `output/sieve/optimal_k.json` — the binding Phase 11 handoff artifacts
- `scripts/sieve_expert_mask_inference.py`, `tests/test_sieve_ksweep_mask.py` — read directly; confirms zero-diff reuse for AIMER/REAP ranking
- `.planning/phases/11-compression-packaging/11-04-SUMMARY.md`, `11-05-SUMMARY.md` — runtime cost figures, vLLM-vs-Tinker serving gap, judge parse-collapse precedent
- `.planning/phases/07-router-profiling-protected-expert-set/07-HUMAN-REVIEW.md` — confirms E_eff table origin (mean/max columns, same report cited in prune_set_for_phase13.json)
- `wp-moe.md` — pins both AIMER and REAP formulas verbatim

### Secondary (MEDIUM confidence)
- [AIMER: Calibration-Free Task-Agnostic MoE Pruning (arxiv 2603.18492)](https://arxiv.org/abs/2603.18492) — WebSearch + WebFetch, confirms real published method, formula matches wp-moe.md, explicitly evaluates Qwen3-family MoE models
- [REAP the Experts: Why Pruning Prevails for One-Shot MoE compression (arxiv 2510.13999)](https://arxiv.org/abs/2510.13999) — WebSearch + WebFetch, accepted ICLR 2026, explicitly lists Qwen3-30B-A3B among evaluated models
- [github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) — WebFetch, confirms calibration-data scale conventions and CLI structure (used to inform the "don't clone the harness, hand-roll the formula" recommendation), no pip package

### Tertiary (LOW confidence)
- Calibration-sample-count recommendation for this project's scale (A1 in Assumptions Log) — extrapolated from the REAP reference repo's much-larger-model calibration mix, not independently benchmarked at this project's scale

## Metadata

**Confidence breakdown:**
- Standard stack (AIMER/REAP are real, formulas verified): HIGH — cross-checked against both wp-moe.md and the actual arxiv papers
- Architecture (mask-reuse, checkpoint-format constraint): HIGH — verified directly against installed `transformers` source and existing tested repo code, not assumed
- Pitfalls (E_eff reconciliation, judge worst-layer risk): HIGH — computed directly from concentration_report.json numbers read this session
- REAP calibration cost/sample-count specifics: MEDIUM-LOW — extrapolated from the reference repo's much-larger-scale convention; flagged in Assumptions Log for planner/human confirmation

**Research date:** 2026-07-10
**Valid until:** 30 days (stable domain — no fast-moving external dependency; the main volatility risk is the exact on-disk safetensors key naming, which is a one-time verification, not a time-decay concern)
