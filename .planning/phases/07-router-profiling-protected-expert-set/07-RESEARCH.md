# Phase 7: Router Profiling & Protected Expert Set - Research

**Researched:** 2026-06-14
**Domain:** Qwen3-MoE routing profiling, per-task expert affinity, protected expert mask extraction
**Confidence:** HIGH (all claims verified against codebase; no external package research needed)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Phase 7 profiles ONLY the promoted `reasoning-merged-v4` model and extracts the
protected-expert mask. The multi-ratio decision matrix (ROADMAP §7 SC5) is DROPPED as moot —
Phase-4 triage gave NO_SURVIVORS except 30/70, which v1.2 already merged/promoted.

**D-02:** The E_eff / concentration report (per-layer CV, cumulative coverage, layer-depth skew,
E_eff mean/max/variance) is kept as informational input to protected-expert identification and
as a pruning baseline (Phases 11/13) — NOT for ratio selection. SC1-SC4 + SC6 of ROADMAP §7
stand; SC5 is removed.

**D-03:** Conservative co-activation. An expert is flagged dual-purpose (protected, must-not-prune)
when it shows meaningful activation above its per-layer mean for BOTH `<wp_gen>` AND `<wp_judge>`.
Errs toward over-protecting — the judge skill is the fragile axis and wrongly pruning a
dual-purpose expert breaks the dual-mode model; over-protection only costs pruning headroom.

**D-04:** Report mask-size sensitivity across thresholds (e.g., mean / median / top-K intersection)
alongside the chosen conservative mask, so Phase 13 (AIMER/REAP pruning) can revisit the
protection/headroom trade-off with data rather than re-deciding blind.

**D-05 (AMENDED 2026-06-14):** Drive forward-pass routing capture with the **training data**
(`data/final_dataset/ratio_30_70/openai_train.jsonl`, the same stimulus that produced the D-08
baseline). The original "existing 4.4 captures" stimulus is SUPERSEDED — its "balanced gen/judge"
premise was false (17 wp_gen : 155 wp_judge, 9:1). Training data gives a clean matched E_eff delta
and ~600x more wp_gen signal. (Resolved via AskUserQuestion — see Open Question 1 below.)

**D-06 (literal, RATIFIED 2026-06-14):** Use the 10% subsample with Jaccard >= 0.94 **vs full-set
ranking** per ratio (ROADMAP §7 SC3); re-profile with a larger subsample if Jaccard fails. The
cross-subsample A-vs-B proxy floated in Open Question 2 was REJECTED — implement subsample-vs-full
literally (full 30/70 train set as reference ranking). (Resolved via AskUserQuestion.)

**D-07:** Profile the merged `reasoning-merged-v4` model. The MoE router was frozen during v1.2 LoRA,
so any routing shift comes from the weights feeding the gate — profiling the merged model captures
the net effect. Hook `Qwen3MoeSparseMoeBlock` gating outputs.

**D-08:** Compare E_eff against the existing Phase-4 `base_model_eeff.jsonl` to quantify the
fine-tuning routing shift (ROADMAP §7 SC4).

**D-09:** Any Phase-7 selection/quality gate (Jaccard threshold, E_eff comparison, protected-expert
cutoff) reports bootstrap CIs and uses CI-aware dispositions, not bare point-bars — the codified
lesson from Phase 04.4.

### Claude's Discretion

- Exact E_eff formula + concentration-metric implementation details (reuse `profile_base_model.py`).
- Subsample construction + Jaccard computation mechanics.
- Protected-mask export format (per-layer boolean mask file) for downstream consumption.
- Telemetry agent embedding (`observe-evaluation`) during profiling runs.

### Deferred Ideas (OUT OF SCOPE)

- Phase 8 judge-recalibration inheritance (`judge_recalibration.json`, score_offset=+3.58):
  matched Phase 7 at score 0.6 but belongs to Phase 8 (the reward pipeline consumes it).
  Left for Phase 8 — not folded here.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support | Single-Ratio Disposition |
|----|-------------|-----------------|--------------------------|
| PROF-01 | Gradient-free forward pass hooking Qwen3MoeSparseMoeBlock gating output; count-based ranking per layer | `RoutingCollector.make_hook()` in profile_base_model.py hooks `layer.mlp.gate`, applies `torch.topk(outputs, k=top_k)` on raw logits, accumulates per-layer counts. Fully implemented. | Active — hook merged model directly via AutoModelForCausalLM, no --adapter |
| PROF-02 | Per-task routing count by task token affinity (wp_gen vs wp_judge), not just aggregate | `set_token_types()` scans input_ids for WP_GEN_ID=151669 / WP_JUDGE_ID=151670 and tags each position. `_counts_wp_gen/_counts_wp_judge` accumulate separately. Fully implemented. | Active — see CRITICAL STIMULUS ISSUE below |
| PROF-03 | 10% subsample with Jaccard >= 0.94 stability verification against full-set ranking | NOT in profile_base_model.py. Must build: subsample N examples, profile both, compute Jaccard on top-K expert sets per layer vs full profile. | Active — new code required |
| PROF-04 | Concentration report: per-layer CV, cumulative coverage curve, layer-depth skew, E_eff mean/max/variance | Only E_eff (total/wp_gen/wp_judge) and subsample_n exist in write_profiling_jsonl(). CV, cumulative coverage curve, layer-depth skew are NOT implemented. | Active — new code required |
| PROF-05 | "Profile ALL surviving ratios from Phase 4 triage" | Written for multi-ratio world. Phase-4 triage returned NO_SURVIVORS except 30/70 (PROJECT.md line 93). The survivor set is {30/70} — not an arbitrary cut; the other 3 ratios produced unparseable judge output (100% skip rate). | TRIVIALLY SATISFIED: "all surviving ratios" = {30/70}. Profile exactly one merged model. No degenerate loop needed. |
| GATE-01 | Decision matrix combining Phase 4 eval score and Phase 7 routing concentration per surviving ratio | Written for multi-ratio world. With one survivor, "decision matrix" trivially selects it. | N/A — DEGENERATE: single survivor is the promoted canonical model. No selection decision exists. Document this rationale; do NOT build a one-candidate decision matrix. |
</phase_requirements>

---

## Summary

Phase 7 profiles the single promoted v1.2 model (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`) to
extract per-task (wp_gen vs wp_judge) expert routing affinity maps across all 48 MoE layers, and
produces the protected-expert mask (D-10) consumed by Phases 11 and 13 for MoE-Sieve / AIMER/REAP
pruning. The phase is entirely gradient-free — pure forward passes with routing hooks, no training.

The existing `scripts/profile_base_model.py` provides the hook infrastructure (`RoutingCollector`),
E_eff formula (`compute_eeff`), per-task token attribution (`set_token_types`), and output schema
(`write_profiling_jsonl`). Three capabilities are NOT yet implemented and must be built: Jaccard
subsample stability (PROF-03), concentration metrics beyond E_eff (PROF-04: CV, cumulative coverage
curve, layer-depth skew), and protected-mask extraction with sensitivity table (D-03/D-04).

A critical stimulus imbalance exists in the D-05 eval captures: `output/eval_reasoning_v4_winner/`
contains 17 wp_gen prompts (`eval_gen_results.jsonl`) vs 155 wp_judge prompts
(`reasoning_samples.jsonl` + `gen_samples.jsonl`). This 9:1 imbalance will produce unreliable
per-task routing statistics for wp_gen. Additionally, the existing `base_model_eeff.jsonl` baseline
was generated from ratio training data (`data/final_dataset/`), not eval captures — creating a
stimulus mismatch for D-08 delta reporting. These two findings together constitute the primary open
planning decision: the planner must choose a profiling stimulus strategy before tasks can be written.

**LOCKED-DECISION CONFLICT (requires discuss-phase re-decision before planning):**
D-05 locks the stimulus to the existing 4.4 eval captures. D-08 requires computing an E_eff delta
against `base_model_eeff.jsonl`, which was generated from training data (`data/final_dataset/`), not
eval captures. If Phase 7 profiles the merged model on eval captures only, the D-08 delta conflates
fine-tuning routing shift with stimulus change — defeating D-08's stated purpose. The two locked
decisions are mutually unsatisfiable for a clean delta. This must be resolved in discuss-phase before
planning can write tasks. See Open Question 1.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Forward-pass routing capture (PROF-01) | Local Python / DGX GPU | — | CUDA required; 13-shard model needs ~60GB VRAM; must run in eval_toolbox container |
| Per-task token attribution (PROF-02) | Local Python | — | Pure tensor scan on input_ids; no GPU needed but runs in same process as hook |
| Jaccard stability verification (PROF-03) | Local Python / DGX GPU | — | Requires two profile passes; compute is GPU-bound |
| Concentration metrics + E_eff (PROF-04) | Local Python post-processing | — | CPU-only; consumes JSONL output from profile pass |
| Protected-mask extraction + export (D-03/D-04) | Local Python post-processing | — | CPU-only; numpy boolean array [48, 128] |
| D-08 E_eff delta reporting | Local Python post-processing | — | Load base_model_eeff.jsonl + new JSONL; compute delta |
| Bootstrap CI computation (D-09) | Local Python post-processing | — | scipy.stats or numpy resample; CPU-only |
| DGX execution orchestration | Skill `wp-finetune:run-profiling` | DGX Toolbox | Extends run-evaluation pattern; GPU-bound passes inside eval_toolbox container |

---

## Standard Stack

### Core (no new packages — all already in environment)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| torch | already installed | Forward passes, topk routing, no_grad context | [VERIFIED: codebase] |
| transformers | already installed | AutoModelForCausalLM, AutoTokenizer | [VERIFIED: codebase] |
| numpy | already installed | Expert count arrays, mask computation, bootstrap resampling | [VERIFIED: codebase] |
| peft | already installed | PeftModel.get_base_model() (already handled by graceful hasattr check) | [VERIFIED: codebase] |

### Supporting (already in environment)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | already installed | Bootstrap CI computation (D-09) | For Jaccard CI, E_eff CI, mask cutoff CI |
| json / pathlib | stdlib | JSONL output, path handling | Throughout |

**No new packages needed.** Phase 7 is pure reuse of existing environment + stdlib. Package
Legitimacy Audit is N/A.

**Installation:** None required.

---

## Package Legitimacy Audit

No external packages to install. All dependencies are already present in the project environment.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

---

## Architecture Patterns

### System Architecture Diagram

```
[Stimulus: training JSONL or eval captures]
          |
          v
[Profile Script: profile_base_model.py (adapted)]
  |-- load merged model (AutoModelForCausalLM, no adapter)
  |-- register hooks: layer.mlp.gate for each of 48 MoE layers
  |-- for each example:
  |     |-- tokenize via chat template
  |     |-- set_token_types(input_ids)  <- tag wp_gen/wp_judge/pad
  |     |-- model(input_ids) [no_grad]  <- hook fires, accumulates counts
  |
  v
[RoutingCollector state: counts[48][128] x {total, wp_gen, wp_judge}]
  |
  |-- [10% Jaccard Stability Check]  <-- NEW
  |     |-- subsample examples
  |     |-- profile subsample
  |     |-- compute top-K Jaccard vs full profile per layer
  |     |-- gate: Jaccard >= 0.94 per layer (bootstrap CI on Jaccard)
  |     |-- if fail -> re-profile with larger subsample
  |
  v
[write_profiling_jsonl: routing_report.jsonl]  <- per-layer JSONL
  |
  v
[Post-processing: concentration.py (NEW)]
  |-- E_eff (reuse compute_eeff) -- total/wp_gen/wp_judge per layer
  |-- per-layer CV of expert counts
  |-- cumulative coverage curve (what % of routing do top-K experts cover)
  |-- layer-depth skew (early vs late layer concentration)
  |-- E_eff delta vs base_model_eeff.jsonl (D-08)
  |-- bootstrap CIs on all metrics (D-09)
  |
  v
[Protected mask extractor: extract_protected_mask.py (NEW)]
  |-- per layer: flag expert if count_wp_gen[e] > mean_wp_gen AND count_wp_judge[e] > mean_wp_judge
  |-- compute mean = n_tokens * top_k / n_experts per split per layer
  |-- sensitivity table: mask size at mean / median / top-K intersection thresholds
  |-- export: protected_expert_mask.npy [48, 128] boolean + sidecar JSON {layer_idx: [expert_ids]}
  |
  v
[Human review checkpoint -> GATE-01: trivially satisfied, document rationale]
  |
  v
[output/profiling/reasoning-merged-v4/]
  |-- routing_report.jsonl (per-layer per-expert counts)
  |-- concentration_report.json (E_eff, CV, coverage, skew, deltas, CIs)
  |-- protected_expert_mask.npy
  |-- protected_expert_mask.json (sidecar)
  |-- sensitivity_table.json (D-04)
```

### Recommended Project Structure

```
scripts/
├── profile_base_model.py       # PRIMARY REUSE: hook + RoutingCollector + write_profiling_jsonl
├── profile_merged_model.py     # NEW ADAPTER: adapted CLI for merged model + D-05 stimulus
├── compute_concentration.py    # NEW: CV + cumulative coverage + layer-depth skew + E_eff delta
├── extract_protected_mask.py   # NEW: mask extraction + sensitivity table + export

output/profiling/
├── base_model_eeff.jsonl       # EXISTING baseline (training data stimulus)
└── reasoning-merged-v4/
    ├── routing_report.jsonl    # per-layer expert counts (total/wp_gen/wp_judge)
    ├── concentration_report.json
    ├── protected_expert_mask.npy   # [48, 128] bool
    ├── protected_expert_mask.json  # {layer_idx: [expert_ids]}
    └── sensitivity_table.json

.claude/skills/
└── wp-finetune:run-profiling/
    └── SKILL.md                # NEW SKILL: DGX execution + observe-evaluation embedding
```

### Pattern 1: Loading Merged Model (no adapter unwrap needed)

The v4-winner is a fully merged HF checkpoint — NOT a LoRA adapter. Load directly:

```python
# Source: profile_base_model.py (adapted)
# For merged model: no --adapter flag, no PeftModel wrapper
# The existing hasattr(model, 'get_base_model') check handles this gracefully
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,  # or bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
# No adapter loading. No get_base_model() call needed.
```

### Pattern 2: Hook Registration (existing, hook at `layer.mlp.gate`)

```python
# Source: profile_base_model.py RoutingCollector.make_hook() and profile_base_model()
# Model config confirmed: num_hidden_layers=48, num_local_experts=128, num_experts_per_tok=8
# Hook fires on raw gate logits (before softmax/topk)
for i, layer in enumerate(model.model.layers):
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
        layer.mlp.gate.register_forward_hook(collector.make_hook(i))
# make_hook applies torch.topk(outputs, k=top_k) on raw logits
# outputs may be tensor or tuple; hook handles both via:
#   logits = outputs[0] if isinstance(outputs, tuple) else outputs
```

### Pattern 3: Per-Task Token Attribution (existing, verbatim reuse)

```python
# Source: profile_base_model.py set_token_types()
WP_GEN_ID = 151669    # <wp_gen> token
WP_JUDGE_ID = 151670  # <wp_judge> token
PAD_TOKEN_ID = 151643

# set_token_types scans input_ids for task tokens and tags each position:
# positions after <wp_gen> -> "wp_gen"
# positions after <wp_judge> -> "wp_judge"
# <pad> positions -> "pad"
# before any task token -> "other"
collector.set_token_types(input_ids)  # call BEFORE model forward pass
```

### Pattern 4: Jaccard Stability (must build)

```python
# Source: NEW CODE (not in profile_base_model.py)
# PROF-03: subsample validation
import numpy as np

def compute_jaccard_stability(full_counts, subsample_counts, top_k=8):
    """
    Per-layer Jaccard of top-K expert sets between full and subsample profiles.
    full_counts: [n_layers, n_experts]
    subsample_counts: [n_layers, n_experts]
    Returns: jaccard_per_layer [n_layers]
    """
    n_layers = full_counts.shape[0]
    jaccards = []
    for layer_idx in range(n_layers):
        full_topk = set(np.argsort(-full_counts[layer_idx])[:top_k])
        sub_topk = set(np.argsort(-subsample_counts[layer_idx])[:top_k])
        intersection = len(full_topk & sub_topk)
        union = len(full_topk | sub_topk)
        jaccards.append(intersection / union if union > 0 else 1.0)
    return np.array(jaccards)

# Gate: np.all(jaccards >= 0.94)
# If fail: re-profile with subsample_frac * 2 (up to full set)
```

### Pattern 5: Protected Mask Extraction (must build)

```python
# Source: NEW CODE — implements D-03 conservative co-activation rule
import numpy as np

def extract_protected_mask(
    counts_wp_gen: np.ndarray,   # [n_layers, n_experts]
    counts_wp_judge: np.ndarray, # [n_layers, n_experts]
) -> np.ndarray:
    """
    Returns bool mask [n_layers, n_experts].
    An expert is protected iff it activates above per-layer mean for BOTH gen AND judge.
    per-layer mean = n_tokens_in_split * top_k / n_experts (expected uniform activation)
    Equivalently: mean(counts_wp_gen[layer]) across experts.
    """
    # Conservative: use actual observed mean (not theoretical uniform)
    mean_gen = counts_wp_gen.mean(axis=1, keepdims=True)    # [n_layers, 1]
    mean_judge = counts_wp_judge.mean(axis=1, keepdims=True) # [n_layers, 1]
    mask = (counts_wp_gen > mean_gen) & (counts_wp_judge > mean_judge)
    return mask  # [n_layers, n_experts] bool

def export_mask(mask: np.ndarray, out_dir: str):
    """Export .npy boolean array + JSON sidecar for Phases 11/13."""
    import json, pathlib
    out = pathlib.Path(out_dir)
    np.save(out / "protected_expert_mask.npy", mask)
    sidecar = {
        str(layer_idx): [int(e) for e in np.where(mask[layer_idx])[0]]
        for layer_idx in range(mask.shape[0])
    }
    with open(out / "protected_expert_mask.json", "w") as f:
        json.dump(sidecar, f, indent=2)
```

### Pattern 6: Bootstrap CI (D-09)

```python
# Source: NEW CODE — implements D-09 CI-aware gate hygiene
import numpy as np

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Bootstrap CI on mean of values array. Returns (lower, upper)."""
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    return np.percentile(boot_means, [alpha/2*100, (1-alpha/2)*100])

# Apply to: Jaccard per-layer, E_eff delta, mask cutoff stability
```

### Anti-Patterns to Avoid

- **Loading merged model with --adapter flag:** v4-winner is a merged checkpoint, not a LoRA adapter. Passing an adapter path will either fail or double-apply weights.
- **Using only eval captures for wp_gen routing:** The D-05 stimulus has 17 wp_gen vs 155 wp_judge examples — a 9:1 imbalance that will produce unstable wp_gen routing statistics. Supplement with training data.
- **Building a decision matrix for GATE-01:** With one survivor (30/70), no selection decision exists. Document rationale and skip matrix construction.
- **Reporting bare point estimates for gates:** D-09 requires bootstrap CIs. A bare Jaccard=0.95 without CI is non-compliant.
- **Hooking after softmax:** The existing hook captures raw logits and applies `torch.topk` internally. Do not intercept post-softmax probabilities — they are not produced by the gate nn.Linear.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Expert routing capture | Custom hook system | `RoutingCollector` in `profile_base_model.py` | Already handles raw logits, topk, token attribution, pad skipping |
| E_eff computation | Custom entropy formula | `compute_eeff()` in `profile_base_model.py` | Already handles zero-count NaN, correct Shannon entropy |
| Output schema | Custom JSONL format | `write_profiling_jsonl()` in `profile_base_model.py` | D-08 baseline already uses this schema; changing it breaks delta computation |
| Bootstrap CI | Custom resampling | `numpy.random.choice` + `np.percentile` (or `scipy.stats.bootstrap`) | Sufficient; no statsmodels or specialized package needed |
| Token attribution | Regex on text | `set_token_types()` in `profile_base_model.py` | Already works on tokenized input_ids using known token IDs |

**Key insight:** The routing capture infrastructure is complete. New code is only needed for three post-processing steps (Jaccard, concentration metrics, mask extraction) and one upstream decision (stimulus).

---

## Critical Pitfalls

### Pitfall 1: D-05 Stimulus Imbalance (HIGHEST PRIORITY)

**What goes wrong:** The 4.4 eval captures intended as D-05 stimulus contain 17 wp_gen prompts
(`eval_gen_results.jsonl`) vs 155 wp_judge prompts (`reasoning_samples.jsonl` + `gen_samples.jsonl`
which are actually judge-format CTF/CoT tasks despite their names). A 9:1 imbalance produces
unreliable wp_gen routing statistics — the protected mask will be computed on noisy wp_gen counts.

**Naming trap:** `gen_samples.jsonl` is NOT wp_gen generation — it contains `<wp_judge>` prefixed
CTF-format judge prompts. Only `eval_gen_results.jsonl` (17 records) contains `<wp_gen>` prompts.

**Root cause:** The eval set was designed to measure judge quality, not to provide balanced stimulus
for routing profiling.

**How to avoid:** Supplement with training data. `data/final_dataset/ratio_30_70/openai_train.jsonl`
(34,855 records) has a 30/70 record-level split (29.3% wp_gen / 70.7% wp_judge by record count,
~15% / 85% by tokens since judge responses are longer) — providing ~10K wp_gen records at 10%
subsample vs only 17 from eval captures. It also matches the stimulus used for `base_model_eeff.jsonl`,
which would enable clean D-08 delta reporting. However, using it requires a discuss-phase re-decision
on D-05 (which locks stimulus to eval captures) — see Open Question 1.

**Warning signs:** wp_gen E_eff values with high variance across layers; protected mask flagging
fewer experts than judge mask; Jaccard instability on wp_gen counts with 10% subsample.

### Pitfall 2: D-08 Stimulus Mismatch (Baseline vs Candidate Use Different Stimuli)

**What goes wrong:** `base_model_eeff.jsonl` was generated from ratio training data
(`data/final_dataset/` via `discover_dataset_dirs()`). If Phase 7 profiles the merged model on
eval captures only, the E_eff delta in D-08 conflates fine-tuning effect with stimulus change —
the two most important confounders for "did fine-tuning change routing?".

**Root cause:** D-08 was specified assuming matching stimuli; D-05 was specified assuming eval
captures; neither specification noted the conflict.

**How to avoid:** This conflict cannot be resolved by the planner — it is a locked-decision
conflict between D-05 (eval captures) and D-08 (matching stimulus assumption). The planner must flag
this to discuss-phase for re-decision. Pending that decision, document the stimulus choice and its
implications in the routing report header. If the user re-decides to use training data as primary
stimulus, the existing `base_model_eeff.jsonl` can be used directly as the D-08 baseline with
matching stimulus — no re-profiling of the base model needed.

**Warning signs:** E_eff deltas that are inexplicably large or inconsistent across layers when the
merged model's router was frozen (D-07 states router was frozen during v1.2 LoRA — so deltas should
be modest if any).

### Pitfall 3: Hook Attachment on Wrong Module

**What goes wrong:** Hook registered on `layer.mlp` (the MoE block) instead of `layer.mlp.gate`
(the router nn.Linear) will capture post-expert activations, not routing decisions.

**How to avoid:** Existing code is correct: `layer.mlp.gate.register_forward_hook(...)`. Verify
with `hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate')` guard (already in profile_base_model.py).

**Warning signs:** Hook fires with output shape `[batch, seq_len, hidden_dim]` instead of
`[batch*seq_len, n_experts]` (gate logits shape).

### Pitfall 4: No CUDA Guard

**What goes wrong:** Profile script runs on CPU, takes hours instead of minutes, or silently
computes incorrect dtype.

**How to avoid:** Existing `--allow-cpu` guard (raises SystemExit if no CUDA unless flag set) should
be kept in adapted script. Run only inside DGX eval_toolbox container.

**Warning signs:** `torch.cuda.is_available()` returns False at script start.

### Pitfall 5: GATE-01 Decision Matrix Temptation

**What goes wrong:** Planner builds a one-row decision matrix for GATE-01, wasting a task and
producing a trivial artifact.

**How to avoid:** GATE-01 is N/A with documented rationale. The output artifact for GATE-01 is a
plain text rationale paragraph in the routing report, not a matrix. Document: "Single survivor
(30/70) from Phase-4 triage; GATE-01 degenerate; no selection decision required."

---

## State of the Art (Codebase-Specific)

| Component | Current State | Phase 7 Change |
|-----------|---------------|----------------|
| profile_base_model.py | Profiles base model on 5 ratio training datasets | Adapt to profile 1 merged model; stimulus TBD (open question) |
| RoutingCollector | Fully implemented (hook, token attribution, count accumulation) | Reuse as-is |
| write_profiling_jsonl | Outputs per-layer JSONL with expert_counts + E_eff per split | Reuse as-is; `model` field changes from "base" to "reasoning-merged-v4" |
| Jaccard stability | Not implemented | New code (PROF-03) |
| Concentration metrics | Only E_eff exists | New code: CV, cumulative coverage curve, layer-depth skew (PROF-04) |
| Protected mask | Not implemented | New code: D-03 co-activation rule + D-04 sensitivity table |
| Bootstrap CIs | Not implemented | New code (D-09) |
| Run-profiling skill | Doesn't exist | Create at planning time (extends run-evaluation pattern) |

---

## Open Questions (RESOLVED 2026-06-14)

> All three resolved at planning time via orchestrator + AskUserQuestion. Summary:
> Q1 → training data (D-05 amended). Q2 → subsample-vs-full literal (cross-subsample proxy REJECTED).
> Q3 → `.npy [48,128]` + `.json` sidecar adopted.

1. **(RESOLVED) LOCKED-DECISION CONFLICT: D-05 vs D-08 stimulus mismatch**
   - **RESOLUTION:** User amended D-05 → training data (`data/final_dataset/ratio_30_70/openai_train.jsonl`),
     matching the D-08 baseline stimulus. Clean delta, no baseline re-profile. (Original D-05's "balanced"
     premise was false.)
   - D-05 (locked): Profile stimulus = existing 4.4 eval captures (`output/eval_reasoning_v4_winner/`)
   - D-08 (locked): E_eff delta against `base_model_eeff.jsonl` (which was generated from training data via `discover_dataset_dirs()`)
   - Conflict: profiling on eval captures and computing delta vs training-data baseline conflates fine-tuning routing shift with stimulus change. Neither decision is in Claude's Discretion — both are locked.
   - Evidence: eval captures = 17 wp_gen + 155 wp_judge (9:1 imbalance); training data = 29.3% wp_gen / 70.7% wp_judge by record (34,855 total); baseline stimulus = training data confirmed via `discover_dataset_dirs()` in profile_base_model.py.
   - Resolution: This must go to discuss-phase. The planner cannot choose between locked decisions.
   - If user re-decides to training data: D-08 delta is clean; no baseline re-profile needed; update D-05 scope note.
   - If user confirms eval captures only: document D-08 delta as indicative only (different stimuli); planner adds a caveat task.

2. **(RESOLVED) Jaccard Stability Subsample Definition**
   - **RESOLUTION:** User ratified D-06 **literally** — "full set" means profile ALL examples as the
     reference ranking, then Jaccard(10% subsample, full) >= 0.94. The cross-subsample A-vs-B proxy
     recommended below was **REJECTED** (two noisy subsamples agreeing is weaker evidence than
     subsample-vs-full). Accept the ~10x compute (one-time, single model).
   - ~~Recommendation: cross-subsample Jaccard as a practical proxy~~ — REJECTED, see resolution.

3. **(RESOLVED) Protected Mask Export Format for Phases 11/13**
   - **RESOLUTION:** Adopted in plans — export both `.npy` `[48, 128]` boolean array and `.json`
     sidecar `{layer_idx: [expert_ids]}`.

---

## Environment Availability

| Dependency | Required By | Available | Notes | Fallback |
|------------|------------|-----------|-------|----------|
| CUDA GPU (DGX) | Forward-pass profiling | Required — must run in eval_toolbox container | 13 model shards, ~60GB VRAM total | None — profile script exits without CUDA |
| `models/qwen3-30b-wp-30_70-reasoning-merged-v4/` | PROF-01/07 (model to profile) | CONFIRMED EXISTS | 13 shards, config.json confirmed 48 layers / 128 experts / top-8 | — |
| `output/profiling/base_model_eeff.jsonl` | D-08 (E_eff baseline) | CONFIRMED EXISTS | 240 records (48 layers × 5 ratios), model="base", stimulus=training data | — |
| `output/eval_reasoning_v4_winner/` | D-05 (profiling stimulus — eval captures) | CONFIRMED EXISTS | 17 wp_gen + 155 wp_judge prompts (imbalanced — see Pitfall 1) | — |
| `data/final_dataset/ratio_30_70/openai_train.jsonl` | Alternative stimulus (if D-05 re-decided) | CONFIRMED EXISTS | 34,855 records, 30/70 by record count (~15/85 by tokens); matches D-08 baseline stimulus | — |
| `scripts/profile_base_model.py` | PROF-01/02 (hook infrastructure) | CONFIRMED EXISTS | 25.3K, fully functional | — |
| numpy, torch, transformers, peft | All PROF-* | Already installed in environment | — | — |
| `wp-finetune:run-profiling` skill | DGX execution orchestration | DOES NOT EXIST | Must create at planning time (Wave 0 gap) | — |

**Missing dependencies with no fallback:**
- CUDA GPU — hard requirement; all profiling must execute on DGX inside eval_toolbox container
- `wp-finetune:run-profiling` skill — must be created in Wave 0 of planning

**Missing dependencies with fallback:** None.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already present in project) |
| Config file | pytest.ini or pyproject.toml (check at planning time) |
| Quick run command | `pytest tests/test_profile_*.py -x -q` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PROF-01 | Hook fires on Qwen3MoeTopKRouter gate; counts accumulate correctly per layer | unit | `pytest tests/test_routing_collector.py::test_hook_accumulates -x` | Wave 0 gap |
| PROF-02 | set_token_types() correctly tags wp_gen/wp_judge/pad positions; counts split correctly | unit | `pytest tests/test_routing_collector.py::test_token_attribution -x` | Wave 0 gap |
| PROF-03 | Jaccard >= 0.94 between two 10% subsamples; fails correctly if < 0.94 | unit | `pytest tests/test_jaccard_stability.py::test_jaccard_gate -x` | Wave 0 gap |
| PROF-04 | CV computation correct; cumulative coverage sums to 1.0 at 128 experts; E_eff = known value on uniform counts | unit | `pytest tests/test_concentration.py -x` | Wave 0 gap |
| PROF-05 | N/A (trivially satisfied — single ratio = profile once) | — | — | — |
| GATE-01 | N/A (degenerate — single survivor; document rationale, skip matrix) | — | — | — |
| D-03 | Protected mask: expert above mean in BOTH splits is flagged; expert above mean in only one is NOT flagged | unit | `pytest tests/test_protected_mask.py::test_co_activation_rule -x` | Wave 0 gap |
| D-09 | Bootstrap CI computed correctly on known distribution; CI-aware gate fires correctly | unit | `pytest tests/test_bootstrap_ci.py -x` | Wave 0 gap |

### Sampling Rate

- **Per task commit:** `pytest tests/test_routing_collector.py tests/test_protected_mask.py -x -q`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_routing_collector.py` — covers PROF-01, PROF-02 (mock model with synthetic gate logits)
- [ ] `tests/test_jaccard_stability.py` — covers PROF-03
- [ ] `tests/test_concentration.py` — covers PROF-04 (E_eff, CV, cumulative coverage, layer-depth skew)
- [ ] `tests/test_protected_mask.py` — covers D-03 co-activation rule
- [ ] `tests/test_bootstrap_ci.py` — covers D-09 CI mechanics
- [ ] `.claude/skills/wp-finetune:run-profiling/SKILL.md` — execution skill (must be created)

---

## Security Domain

Phase 7 is pure local computation — no network access, no user input, no authentication surfaces,
no data exfiltration vectors. All data is local project files.

**Applicable ASVS Categories:**

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No auth surfaces — local script execution only |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | Local filesystem; no multi-user access |
| V5 Input Validation | Minimal | JSONL parsing; use standard `json.loads()` with exception handling on malformed records |
| V6 Cryptography | No | No cryptographic operations |

**Known Threat Patterns:** None applicable. The primary file integrity concern is accidental
overwrite of `output/profiling/base_model_eeff.jsonl` (the baseline). Recommend: profiling
script writes to `output/profiling/reasoning-merged-v4/` (distinct subdirectory), never to the
base JSONL path.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `layer.mlp.gate` hook still attaches correctly on merged model (not checked by loading model) | Architecture Patterns: Pattern 2 | Hook silently fails if gate module path changed; profiling produces zero counts. MITIGATION: script logs hook count at startup — verify 48 hooks registered. |
| A2 | MoE router was frozen during v1.2 LoRA (D-07 assertion) | Summary, D-08 pitfall | If router weights changed, E_eff delta interpretation changes (expected vs unexpected shift). MITIGATION: document assumption in routing report header; check CONTEXT.md D-07 confirms this. |
| A3 | `data/final_dataset/ratio_30_70/openai_train.jsonl` record split is 30/70 wp_gen/wp_judge | Pitfall 1 note | Verified: 29.3% wp_gen / 70.7% wp_judge by record in 1K sample. Token split is ~15/85 (judge responses longer). Not "balanced" — but 10K wp_gen records at 10% subsample vs 17 in eval captures is 600x more gen signal. [VERIFIED: codebase] |

**If this table is empty:** Not empty — 3 assumptions logged.

---

## Sources

### Primary (HIGH confidence — verified against codebase)

- `scripts/profile_base_model.py` — RoutingCollector, compute_eeff, write_profiling_jsonl, hook mechanics, CLI flags; read in full
- `output/profiling/base_model_eeff.jsonl` — schema verified (240 records, 5 ratios × 48 layers, stimulus=training data confirmed via discover_dataset_dirs in script)
- `models/qwen3-30b-wp-30_70-reasoning-merged-v4/config.json` — confirmed: num_hidden_layers=48, num_local_experts=128, num_experts_per_tok=8, decoder_sparse_step=1
- `output/eval_reasoning_v4_winner/` — all files inspected; stimulus imbalance verified by counting wp_gen/wp_judge tokens per file
- `data/final_dataset/ratio_30_70/openai_train.jsonl` — confirmed 34,855 records, wp_judge prefix in sample records
- `.planning/phases/07-router-profiling-protected-expert-set/07-CONTEXT.md` — all locked decisions D-01 through D-09 read verbatim
- `.planning/REQUIREMENTS.md` — PROF-01..05, GATE-01 read verbatim
- `.planning/STATE.md` — v4-winner promotion confirmed, Phase 7 unblocked
- `capture_manifest.json` — schema: `{judge_val: 121, sentinel: 24, reasoning_samples: 120, gen_samples: 35}` with `task_type` field

### Secondary (MEDIUM confidence)

- Advisor analysis confirming stimulus mismatch as load-bearing planning issue
- Project memory observations 3099-3102 (Phase 7 scope decisions, 2026-06-14)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all verified in codebase
- Architecture: HIGH — hook mechanics verified against actual code; model config confirmed
- Pitfalls: HIGH — stimulus imbalance verified by counting tokens in actual files
- Stimulus decision: LOW — requires user confirmation before task planning

**Research date:** 2026-06-14
**Valid until:** 2026-07-14 (stable domain — MoE hook mechanics, numpy, torch; model files are static)
