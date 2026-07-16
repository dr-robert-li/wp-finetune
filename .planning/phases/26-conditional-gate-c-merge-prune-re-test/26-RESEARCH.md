# Phase 26: Conditional Gate C — Merge + Prune Re-Test - Research

**Researched:** 2026-07-17
**Domain:** MoE expert pruning (AIMER weight-norm scorer), gate-before-remove eval on a served (not in-process) 35B/256-expert model, GB10 memory discipline
**Confidence:** HIGH (all claims below verified by reading the actual v3 scripts + inspecting the actual v4 checkpoint's tensor shapes this session; the only LOW-confidence items are the Q8-size projection in §4, explicitly flagged)

## Summary

Gate C reuses the Phase-13 (v3) prune stack almost entirely as designed — the eligibility-gate
logic, the mask-building primitive, and the served-vLLM eval pattern are shape/config-driven and
need no rewrite. **The one load-bearing adaptation is physical**: v4's merged checkpoint stores
expert weights as **stacked tensors** (`mlp.experts.gate_up_proj` `[256,1024,2048]`,
`mlp.experts.down_proj` `[256,2048,512]`, one key per layer) under a `language_model.` prefix,
whereas every v3 prune script (`aimer_prune.py`, `prune_apply_physical.py`) assumes v3's
**per-expert unstacked** key convention (`model.layers.{L}.mlp.experts.{E}.{proj}.weight`,
3 separate projections). None of the v3 key-parsing regexes or per-expert key builders match a
single tensor in the v4 checkpoint — this must be rewritten as index-select-along-dim-0, not
patched. `sieve_expert_mask_inference.build_ksweep_mask` (the k-sweep's own mask primitive) is
shape-driven and unaffected — it already works on v4 (proven live in Gate B).

Gate C also drops the entire gen/wp-bench axis: v4 has no `wp_gen` axis, so `prune_gated_eval.py`'s
dual-axis structure, Docker wp-bench reset, and the 3 hardcoded v3 floors
(`GEN_WPBENCH_FLOOR`, `JUDGE_ENS_RHO_FLOOR=0.7555`, `JUDGE_PARSE_FLOOR=0.95`) must **not** be
reused verbatim — Gate C's equivalence bar is the same-stack CI-aware TOST vs the Gate B full arm
(0.7935), exactly as `sieve_v4_tost_verdict.py` already implements, not a v3-derived fixed floor
(reusing v3's numeric floors here would be a goalpost swap, not a carry-forward). The scope is also
narrower than v3's 3-ratio sweep: Gate C tests **one point, k=224** (12.5% expert drop), not
25/50/75%.

**Primary recommendation:** Score AIMER (mean across the 3 merged judge checkpoints s0/s1/s2, per
the v3 shared-profile convention) directly against the stacked tensors via a rewritten
`compute_aimer_scores` that reduces along `dim=0` of `experts.gate_up_proj`/`experts.down_proj`
instead of iterating per-expert keys; build the k=224 keep-mask with the unchanged
`build_ksweep_mask` (scores replace routing counts 1:1 — already proven arch-agnostic in Gate B);
gate it through the unchanged patched-vLLM serve pattern + `sieve_v4_tost_verdict.py`'s TOST scorer
BEFORE calling a rewritten `prune_apply_physical.py` that slices the stacked tensors and rewrites
`config["text_config"]["num_experts"]` (not `num_local_experts`). Given the checkpoint's own weight
distribution (experts = ~90% of file size) and Gate B's own quality ceiling for what compression
survives, k=224's own math cannot close the 37.8→30.2 GiB gap (§4) — expect and record `no_winner`,
confirming rather than assuming it, per the routing decision.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Merge verification (SC1) | Filesystem / checkpoint | — | Confirm `models/Qwen3.6-35B-A3B-judge-v4-s1-merged` is already fully merged (no adapter files); no compute needed |
| AIMER scoring | Offline scoring script (CPU, disk-streaming) | — | Reads safetensors shards directly; never loads the model into a framework — same GB10-safe design as v3 |
| REAP scoring (optional) | Served-model calibration pass | Offline scoring script | v3's REAP needs live forward hooks on `mlp.gate` + `mlp.experts`; a GB10-safe version must run via the served-vLLM profiling hook family (`_sieve_profile_vllm_patch`), NOT an in-process `from_pretrained` (OOMs, see §5) |
| Keep-mask construction | Offline scoring script | — | `build_ksweep_mask` — arch-agnostic, unchanged from Gate B |
| Gate-before-remove eval | Served vLLM (patched) | Eval harness (CPU-side scoring) | Same pattern as Gate B's k-sweep: mask env var mounts into the container; judge captures scored offline against `val_labels_v1.json` |
| 3-seed ensemble confirmation | Served vLLM (patched) | — | Only runs if the s1 gate passes — mirrors Gate B's `maybe_run_ensemble` |
| Physical expert removal | Offline surgery script (CPU, disk-to-disk) | — | Only runs after the gate passes; never touches a running server |
| Disposition + selection | Offline scoring script | — | `prune_selection.py`'s eligibility logic is data-shape-driven, reusable once inputs are adapted to judge-only |

## Standard Stack

No new external packages. Gate C is 100% adaptation of already-installed, already-used tooling:

| Library | Version | Purpose | Provenance |
|---------|---------|---------|--------------|
| `safetensors` | already installed (used by `aimer_prune.py`, `prune_apply_physical.py`) | Stream/slice checkpoint tensors without a framework load | `[VERIFIED: codebase — scripts/aimer_prune.py:44]` |
| `numpy` | already installed | Score arrays, masks | `[VERIFIED: codebase]` |
| `torch` | already installed (`.venv-tinker`) | Tensor slicing for physical surgery | `[VERIFIED: codebase]` |
| `scipy.stats.spearmanr` | already installed | TOST scoring, reused verbatim from `sieve_v4_tost_verdict.py` | `[VERIFIED: codebase — scripts/sieve_v4_tost_verdict.py:89]` |

**Installation:** none required.

## Package Legitimacy Audit

Not applicable — this phase installs zero new external packages. All tooling is in-repo Python
reusing already-vetted, already-installed libraries (`safetensors`, `numpy`, `torch`, `scipy`,
`transformers`, `vllm`) that were already legitimacy-audited in prior phases (13, 20, 22, 25).

## Architecture Patterns

### System Architecture Diagram

```
                 ┌───────────────────────────────────────────────┐
                 │ models/Qwen3.6-35B-A3B-judge-v4-{s0,s1,s2}-merged │
                 │ (already merged; confirm merge-of-record, SC1)│
                 └───────────────────┬───────────────────────────┘
                                     │ safetensors.safe_open (streamed, CPU, no GPU)
                                     ▼
      ┌──────────────────────────────────────────────────────────┐
      │ compute_aimer_scores() [REWRITTEN for stacked tensors]    │
      │  per layer: read experts.gate_up_proj [256,1024,2048] +  │
      │  experts.down_proj [256,2048,512]; reduce P/N/Q per       │
      │  expert along dims (1,2); score = P/sqrt(N*Q)             │
      │  mean across s0/s1/s2 -> aimer_scores_judge_v4.npy [40,256]│
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌──────────────────────────────────────────────────────────┐
      │ build_gated_mask(scores, protected_expert_mask.npy, k=224)│
      │  = build_ksweep_mask (UNCHANGED, arch-agnostic)           │
      │  -> aimer_k224.npy [40,256] bool, 224 kept/layer          │
      └───────────────────────────┬────────────────────────────────┘
                                  │ SIEVE_MASK_NPY env
                                  ▼
      ┌──────────────────────────────────────────────────────────┐
      │ GATE-BEFORE-REMOVE (no weight touched yet)                │
      │ serve_30_70_vllm.sh (LANGUAGE_MODEL_ONLY=1) + patched vLLM│
      │  s1 seed only -> capture 121 judge items @ max_tokens=8192│
      │  score vs full-arm (0.7935) via sieve_v4_tost_verdict.py  │
      │  + D2_security retention (per-dim, prune_gated_eval._d2…) │
      └───────────────────────┬─────────────────────┬──────────────┘
                    TOST FAIL │                     │ TOST PASS + D2 OK + protected_retained
                              ▼                     ▼
                    ┌─────────────────┐   ┌───────────────────────────────┐
                    │ no_winner        │   │ 3-seed ensemble confirm (s0,s2)│
                    │ ship merged-     │   │ same serve pattern, median rho │
                    │ unpruned (SC3)   │   └───────────────┬───────────────┘
                    └─────────────────┘                    │ still passes
                                                            ▼
                                              ┌───────────────────────────────┐
                                              │ prune_apply_physical()         │
                                              │ [REWRITTEN: slice stacked axis │
                                              │  0, keep shared_expert/router  │
                                              │  untouched, rewrite            │
                                              │  text_config.num_experts=224]  │
                                              └───────────────────────────────┘
```

### Recommended Project Structure

```
scripts/
├── aimer_prune_v4.py          # NEW (or --arch flag on existing aimer_prune.py): stacked-tensor AIMER
├── prune_gate_v4.py            # NEW: judge-only gate-before-remove driver (replaces prune_gated_eval.py's dual-axis contract)
├── prune_apply_physical_v4.py  # NEW (or --arch flag): stacked-tensor physical surgery
├── prune_selection.py          # REUSE unchanged (data-shape-driven eligibility gate)
├── sieve_v4_tost_verdict.py    # REUSE unchanged (score_capture/tost_from_scores = the gate scorer)
├── sieve_expert_mask_inference.py  # REUSE unchanged (build_ksweep_mask)
└── sieve_arch.py                # REUSE unchanged (arch_dims, gb10_load_kwargs if any load is needed)
output/prune-v4/
├── aimer_scores_judge_v4.npy   # [40,256] float32
├── masks/aimer_k224.npy        # [40,256] bool
├── gated/aimer_224_judge.json  # gate result (TOST + D2 + protected_retained)
├── gated/aimer_224_d2.json     # D2_security retention/baseline
├── selection_v4.json           # winner | no_winner verdict
└── prune_methodology_v4.md     # model-card-ready lineage doc (mirrors output/prune/prune_methodology.md)
```

**Naming:** use a `_v4` suffix or new `output/prune-v4/` directory rather than overwriting
`output/prune/` — the v3 receipts (`output/prune/aimer_scores_judge.npy`, `selection.json`, etc.)
are the historical record for the v3 model family and must not be clobbered.

### Pattern 1: Stacked-tensor per-expert reduction (the core adaptation)

**What:** v3's `aimer_prune.py` builds one dict key per `(layer, expert, proj)` and streams 3
separate tensors per expert
(`scripts/aimer_prune.py:51-73`, confirmed against `model.safetensors.index.json` this session for
v3's family). v4's checkpoint has no such keys — confirmed via direct inspection of
`models/Qwen3.6-35B-A3B-judge-v4-s1-merged/model.safetensors.index.json`:

```
model.language_model.layers.0.mlp.experts.gate_up_proj  -> shape [256, 1024, 2048]  (fused gate+up)
model.language_model.layers.0.mlp.experts.down_proj      -> shape [256, 2048, 512]
model.language_model.layers.0.mlp.gate.weight            -> shape [256, 2048]        (router — unchanged v3-style layout, just prefixed)
model.language_model.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight  -> unstacked, per-layer, NOT part of the 256-expert axis — must be left untouched
model.language_model.layers.0.mlp.shared_expert_gate.weight                -> also untouched
```
`[VERIFIED: models/Qwen3.6-35B-A3B-judge-v4-s1-merged/model.safetensors.index.json + safe_open shape inspection, this session]`

**When to use:** Any time a Qwen3.5/3.6-MoE-family checkpoint (v4 base and beyond) needs per-expert
scoring or surgery — the stacked-tensor convention is the standing pattern going forward, not a
one-off.

**Example (rewritten `compute_aimer_scores`, replacing per-key iteration with axis reduction):**
```python
# Source: derived this session from scripts/aimer_prune.py's P/N/Q formula (unchanged),
# re-targeted at the stacked-tensor layout confirmed via safe_open above.
def compute_aimer_scores_v4(checkpoint_dir, n_layers=40, n_experts=256, prefix="model.language_model"):
    weight_map = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    P = np.zeros((n_layers, n_experts)); Q = np.zeros((n_layers, n_experts)); N = np.zeros((n_layers, n_experts), dtype=np.int64)
    for layer in range(n_layers):
        for key in (f"{prefix}.layers.{layer}.mlp.experts.gate_up_proj",
                    f"{prefix}.layers.{layer}.mlp.experts.down_proj"):
            with safe_open(checkpoint_dir / weight_map[key], framework="pt") as f:
                w = f.get_tensor(key).float()          # [256, dim_a, dim_b]
            P[layer] += w.abs().sum(dim=(1, 2)).numpy()
            Q[layer] += (w ** 2).sum(dim=(1, 2)).numpy()
            N[layer] += w[0].numel()                    # same per-expert element count for every expert
    return (P / np.sqrt(N * Q)).astype(np.float32)
```
The AIMER formula itself (`score = P / sqrt(N*Q)`, scale-invariant, bounded `[1/sqrt(N), 1]`) is
**unchanged** — only the tensor-access pattern changes. Keep `aimer_prune.py`'s existing
`--self-check` fixture-based test pattern; add a second stacked-tensor fixture (mirroring
`_write_fixture_checkpoint` in `prune_apply_physical.py:190-228`, which already builds
synthetic stacked-tensor checkpoints for its own self-check — reuse that fixture builder rather
than inventing a third one).

### Pattern 2: Judge-only gate-before-remove (drop the gen axis entirely)

**What:** `prune_gated_eval.py` is built around a v3 dual-axis (`gen`, `judge`) contract:
`run_gen_gate` (`scripts/prune_gated_eval.py:159-170`) calls wp-bench + Docker grader reset
(`_reset_wpbench_grader`, `prune_gated_eval.py:134-148`); `GEN_WPBENCH_FLOOR=0.4284` is a v3-measured
constant (`prune_gated_eval.py:62`). None of this applies to v4 — no `wp_gen` task token exists on
the v4 judge (confirmed in `scripts/sieve_arch.py:149-162`'s `resolve_task_token_ids`, and in
25-01's SUMMARY: "v4 judge has no wp_gen/wp_judge tokens" — single-task by construction).

**When to use:** Gate C's driver should be a stripped-down judge-only script — essentially
`sieve_ksweep_v4_run.py`'s serve/capture/score loop (`capture_seed`, `score_s1`,
`maybe_run_ensemble` — `scripts/sieve_ksweep_v4_run.py:138-230`), pointed at ONE mask (`k=224`, AIMER
scores) instead of the 5-arm routing-count grid, plus the D2_security retention check lifted from
`prune_gated_eval._d2_security_mean` (`scripts/prune_gated_eval.py:256-287`, itself
arch-agnostic — it only parses judge response text, never touches expert-weight layout).

**Example (reuse map — no new eval logic needed, just re-wiring):**
```python
# Source: scripts/sieve_ksweep_v4_run.py capture_seed/score_s1 (reuse verbatim)
#        + scripts/prune_gated_eval.py _d2_security_mean (reuse verbatim)
#        + scripts/sieve_v4_tost_verdict.py tost_from_scores (reuse verbatim, same-stack ref)
mask_path = build_gated_mask_v4(aimer_scores, protected, k=224)   # build_ksweep_mask, unchanged
cap_s1 = capture_seed_style(mask_path, seed="s1")                  # same boot_vllm/wait_healthy/stop_vllm calls
tost = tost_from_scores(score_capture(cap_s1)[0], full_arm_scores, labels)  # vs Gate B's 0.7935 reference
d2 = {"retention": _d2_security_mean({"s1": cap_s1}), "baseline": _d2_security_mean(baseline_full_capture)}
```

### Pattern 3: Config key + physical surgery adaptation

**What:** `prune_apply_physical.py`'s regexes
(`EXPERT_KEY_RE`/`ROUTER_KEY_RE`, `prune_apply_physical.py:53-57`) match v3's per-expert unstacked
keys and its config rewrite (`config["num_local_experts"] = k`, `prune_apply_physical.py:164`) targets
a flat config. v4's config is a **composite VL config** with the MoE dims nested under
`text_config` and the key is **`num_experts`**, not `num_local_experts`
(`[VERIFIED: models/Qwen3.6-35B-A3B-judge-v4-s1-merged/config.json]` — `text_config.num_experts: 256`,
also confirmed independently by `scripts/sieve_arch.py:57-65`'s `arch_dims()`, which already reads
`_text_config(config)` then `num_experts` for v4, `num_hidden_layers`/`num_experts` at the top level
for v3's flat config).

**When to use:** any physical surgery on a v4-family checkpoint.

**Example:**
```python
# Source: derived this session — slice stacked axis 0 instead of drop/rename per-expert keys
def apply_physical_v4(checkpoint_dir, keep_mask, out_dir, prefix="model.language_model"):
    ...
    for layer in range(n_layers):
        kept_idx = sorted(np.where(keep_mask[layer])[0].tolist())   # length k=224, ascending
        for suffix in ("experts.gate_up_proj", "experts.down_proj"):
            key = f"{prefix}.layers.{layer}.mlp.{suffix}"
            t = f.get_tensor(key)                 # [256, ...]
            tensors_out[key] = t[kept_idx]         # -> [224, ...], index_select along dim 0
        router_key = f"{prefix}.layers.{layer}.mlp.gate.weight"     # [256, 2048]
        tensors_out[router_key] = f.get_tensor(router_key)[kept_idx]  # -> [224, 2048]
        # shared_expert.* and shared_expert_gate.weight: copy unchanged, never sliced
    config["text_config"]["num_experts"] = k    # NOT config["num_local_experts"]
```
No renumbering/rename step is needed (unlike v3) — the stacked tensor's row order IS the expert
index, so `t[kept_idx]` both drops and renumbers in one op. This is *simpler* than v3's surgery, not
harder, once the axis-slicing insight is in place.

### Anti-Patterns to Avoid

- **Reusing v3's fixed regression floors (`GEN_WPBENCH_FLOOR`, `JUDGE_ENS_RHO_FLOOR=0.7555`,
  `JUDGE_PARSE_FLOOR=0.95`) as Gate C's bars.** These are v3-measured, v3-stack-specific numbers.
  Gate C's bar is the CI-aware TOST vs the Gate B same-stack full arm (ε=2pp, 0.7935) — reusing v3's
  numeric floor would silently change the equivalence standard (a goalpost move, exactly what
  `T-25-06`/the "no goalpost move" carry-forward threat prohibits).
- **In-process `from_pretrained` for AIMER scoring or REAP calibration.** AIMER never needs this
  (streams via `safe_open`); REAP's v3 design uses live forward hooks, which requires the model
  resident in a framework — on GB10 this is the exact failure mode from
  `.planning/debug/resolved/v4-judge-load-oom-recurrence.md` (host ~50 GiB + device ~67 GiB > 121 GiB
  pool). Any REAP attempt must go through the served-vLLM profiling-hook pattern
  (`scripts/_sieve_profile_vllm_patch`), not a bare `AutoModelForImageTextToText.from_pretrained`.
- **Assuming `num_local_experts` is still the config key.** It is `text_config.num_experts` for v4;
  writing the wrong key silently no-ops (transformers ignores an unknown config field) and ships a
  checkpoint whose `config.json` still claims 256 experts while the tensors have 224 rows —a
  shape-mismatch crash at next load, not caught until serve time.
  `scripts/sieve_arch.py:57-65` (`arch_dims`) already has the correct resolution logic to copy from.
- **Sweeping a ratio grid (25/50/75%) like v3.** Gate C's locked scope is a single point, k=224 —
  the routing decision from Gate B (25-02-SUMMARY.md) authorizes exactly one candidate. There is no
  pre-registered sanction for testing other ratios in this phase; if k=224 fails, the disposition is
  `no_winner`, not "try a milder ratio" (v3's monotonicity argument doesn't even apply here since
  k=224 is already the mildest ratio anyone measured).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Keep-mask from a score array | A new top-k-per-layer selector | `sieve_expert_mask_inference.build_ksweep_mask` (unchanged) | Already arch-agnostic (shape-driven), already proven live on v4 in Gate B, already has the protected-expert union guarantee tested |
| TOST-equivalence scoring | A new bootstrap/CI routine | `sieve_v4_tost_verdict.paired_bootstrap_delta` + `tost_from_scores` (unchanged) | Already implements the exact pre-registered CI-aware, paired, same-stack-reference TOST Gate B used and Gate C must match for "no goalpost move" |
| D2_security retention | A new per-dimension parser | `prune_gated_eval._d2_security_mean` (unchanged — pure judge-response-text parsing, arch-agnostic) | Already handles median-across-seeds ensembling and missing-parse degradation gracefully |
| vLLM masked serving | A new mask-application hook | `scripts/_sieve_vllm_patch` + `serve_30_70_vllm.sh`'s `SIEVE_MASK_NPY`/`LANGUAGE_MODEL_ONLY` env toggles (unchanged) | Already confirmed live against the installed vLLM's `qwen3_next.Qwen3NextSparseMoeBlock` resolver in Gate B (T-25-03) |
| Eligibility gate + winner selection | A new pass/fail rule table | `prune_selection.evaluate_variant`/`select_winner` (unchanged, data-shape-driven) | Already encodes "fail closed on missing fields," "physically feasible (k >= max_protected_per_layer)," and "smaller k wins ties" — exactly Gate C's stated success criteria |

**Key insight:** every piece of *evaluation* logic (masking, TOST, D2 parsing, eligibility) is
already architecture-agnostic and proven on v4. The only genuinely new code is the *tensor-layout*
adaptation (AIMER scoring + physical surgery) — because those two scripts are the only ones that
read raw checkpoint tensor keys instead of consuming already-shaped numpy arrays.

## Common Pitfalls

### Pitfall 1: Silently mismatched key prefix (`model.layers.` vs `model.language_model.layers.`)
**What goes wrong:** A rewritten AIMER/surgery script that keeps v3's `model.layers.{L}...` key
template will raise a clean `KeyError` in `aimer_prune.py`'s existing missing-key check
(`aimer_prune.py:75-80`) — which is good (fails loud) — but a hand-rolled version without that
check could silently score 0 experts and write a garbage/zero array.
**Why it happens:** v4's on-disk convention nests everything under `model.language_model.` (VL
composite model), confirmed in the index.json inspected this session; `sieve_arch.py`'s own
docstring (`sieve_arch.py:26-34`) already flags this exact trap for the *live* module tree
(`model.model.layers` vs `model.language_model.layers`) — the on-disk key prefix has the analogous
mismatch for the *checkpoint* tensor keys.
**How to avoid:** Derive the prefix from the checkpoint itself (grep the index.json's weight_map
for the first `.mlp.experts.` match and take everything before it), not a hardcoded string — one
prefix constant used everywhere, asserted against actual weight_map keys before scoring starts.
**Warning signs:** `compute_aimer_scores` returns an all-zero or all-NaN array; the missing-key
assertion should catch this before it gets that far — keep that assertion in the rewrite.

### Pitfall 2: AIMER catastrophic failure precedent (v3's own measured result)
**What goes wrong:** v3's AIMER@25 (the *mildest* v3 ratio, K=96/128=75% kept — LESS aggressive
than dropping only 12.5%... wait, actually MORE aggressive than v4's k=224/256=87.5% kept) measured
a **catastrophic collapse**: judge ensemble rho **0.1651 vs floor 0.7555 (−59.0pp)**, parse rate
**0.4463 vs floor 0.95** (parse collapse — under half the judge responses even parsed)
`[VERIFIED: output/prune/comparison_table.md + output/prune/selection.json, this session]`. This is
a far worse failure mode than Gate B's routing-count-based masking ever produced on v4 (which
stayed parse_fail 0/121 at every k down to 112/256).
**Why it happens:** AIMER is a **calibration-free weight-norm** score — it has no information about
which experts the judge domain actually routes to. It can select experts that are structurally
"important" by weight-norm but rarely activated for judge-style prompts, or vice versa, producing a
keep-set that differs sharply from the routing-count-based keep-set Gate B validated. Routing-count
masking and weight-norm masking are genuinely different selection criteria; a good result on one
does not predict a good result on the other.
**How to avoid:** Treat Gate C's gate-before-remove as a real, uncertain test — do not assume AIMER
at k=224 will behave like Gate B's k=224 arm just because the k is the same. Run the gate for real;
be prepared for AIMER to fail well before the size-math argument in §4 even becomes relevant. If
AIMER@224 also shows parse collapse, that is itself sufficient grounds for `no_winner` — do not average retained-quality
away or attribute a low score to noise.
**Warning signs:** parse_fail rising sharply above 0/121, or judge_s1_rho dropping far below both the
0.7935 full-arm reference and the CI-aware TOST epsilon band.

### Pitfall 3: Gate-before-remove violated (surgery before the eval passes)
**What goes wrong:** Physically removing experts before the gated eval has recorded a pass creates
an irreversible false "prune succeeded" state — the checkpoint on disk no longer matches what was
measured, and a security-relevant regression (D2_security) could ship undetected.
**Why it happens:** It's tempting to build the pruned checkpoint "in parallel" with the eval to save
wall-clock time (physical surgery + gate-eval could both start once the mask exists). This is
exactly the sequencing the locked scope forbids: "gate-before-remove: NO physical weight removal
until the gated eval passes" (CONTEXT.md).
**How to avoid:** Structure the driver so `prune_apply_physical_v4.main()` literally cannot run
without first reading a `gated/aimer_224_judge.json` (or equivalent) result file with `pass: true`
and `pass_d2_security: true` — an explicit precondition check at the top of the surgery script, not
just a procedural convention in a runbook. Mirror `prune_gated_eval.py`'s own pattern of writing a
result JSON as its side effect, then have the surgery script's `main()` `assert` on that file's
content before touching any tensor.
**Warning signs:** A pruned checkpoint directory existing on disk with no corresponding
`gated/*_judge.json` "pass: true" record, or a record whose timestamp is *after* the pruned
checkpoint's mtime (surgery ran first).

### Pitfall 4: Same-stack TOST reference drift
**What goes wrong:** Using the llama.cpp Q8 rho (0.8067) or the Tinker-native rho (0.8358) as the
"what pruning must not regress below" reference instead of the same-stack vLLM full-arm rho
(0.7935) — this is the exact `sanity_gate_recalibration` mistake Gate B's own reference explicitly
guards against (`optimal_k_v4.json`'s `tost_reference.note`, `sieve_v4_tost_verdict.py:13-14`).
**Why it happens:** The Q8/Tinker numbers are more familiar (they're the headline model-card
numbers) and are higher, making a pruned candidate look better by comparison than it really is
against the actual serving stack.
**How to avoid:** Reuse `sieve_v4_tost_verdict.tost_from_scores`'s existing `full` argument
unchanged — it is *already wired* to read the Gate B full-arm capture
(`output/sieve-v4/ksweep/kfull/s1/judge_responses.jsonl`, rho 0.7935). Do not substitute a
different reference number anywhere in the Gate C driver.
**Warning signs:** Any TOST computation whose `full_vals`/reference isn't sourced from
`output/sieve-v4/k_sweep_results_v4.json`'s `full` arm.

### Pitfall 5: GB10 OOM on any in-process full-model load
**What goes wrong:** Any code path that calls `AutoModelForImageTextToText.from_pretrained` (or
`AutoModelForCausalLM` — the WRONG class per T-25-01, see `scripts/profile_v4_judge.py:4-12`)
without `sieve_arch.gb10_load_kwargs()`'s single-device + streaming kwargs risks the exact OOM
documented in `.planning/debug/resolved/v4-judge-load-oom-recurrence.md` and the two
GB10-load-amplifier traps `sieve_arch.py:165-256` neutralizes (`caching_allocator_warmup`,
threaded shard materializer).
**Why it happens:** `device_map="auto"` on a GB10 unified-memory system double-counts the same
physical RAM as two independent pools (reports the whole 121 GiB as "GPU-free" AND lets the rest
land on "cpu"), so a 67 GiB model can OOM-kill even though it should fit comfortably in 121 GiB.
**How to avoid:** AIMER scoring never needs a model load at all (pure `safe_open` streaming — no
risk here). If REAP is attempted, it must go through vLLM (which already manages its own memory
budget correctly, proven in Gate B) or, if any in-process load is truly unavoidable, it MUST route
through `sieve_arch.gb10_load_kwargs()` (splatted, never combined with a separate `device_map=`)
exactly as the fix-commit `cc4e660` ("route all full-model loads through single-device
gb10_load_kwargs") already establishes as the single choke-point.
**Warning signs:** `nvidia-smi`/`free` showing a fast climb toward the 121 GiB ceiling before any
"Loading weights: N/M" progress line prints.

## Code Examples

### Verify the merge-of-record (SC1) before any scoring
```bash
# Source: derived this session — no adapter files should remain in a fully-merged checkpoint
ls models/Qwen3.6-35B-A3B-judge-v4-s1-merged/ | grep -i adapter   # expect: no output
python3 -c "
import json
c = json.load(open('models/Qwen3.6-35B-A3B-judge-v4-s1-merged/config.json'))
print(c['architectures'], c['text_config']['num_experts'], c['text_config']['num_hidden_layers'])
"
# expect: ['Qwen3_5MoeForConditionalGeneration'] 256 40
```

### Confirm k=224 clears the physical-feasibility floor before scheduling any GPU time
```bash
# Source: output/sieve-v4/eeff_report.json (max protected/layer already measured = 98)
python3 -c "
import numpy as np
p = np.load('output/sieve-v4/protected_expert_mask.npy')
print('max_protected_per_layer =', int(p.sum(axis=1).max()), '<= 224 ?', int(p.sum(axis=1).max()) <= 224)
"
# max_protected_per_layer = 98 <= 224 -> True (k=224 is comfortably physically feasible)
```

### Reuse the exact TOST reference from Gate B (do not recompute a new full arm)
```python
# Source: scripts/sieve_v4_tost_verdict.py:147-172 (compute_verdict) — same call pattern for Gate C
from scripts.sieve_v4_tost_verdict import score_capture, tost_from_scores, load_labels
full_scores, _ = score_capture("output/sieve-v4/ksweep/kfull/s1/judge_responses.jsonl")  # rho 0.7935
labels = load_labels()
masked_scores, parse_fail = score_capture("output/prune-v4/gated/aimer_224/s1/judge_responses.jsonl")
tost = tost_from_scores(masked_scores, full_scores, labels, eps=0.02)
```

## State of the Art

| Old Approach (v3, Phase 13) | Current Approach (v4, Phase 26) | When Changed | Impact |
|--------------------|------------------|--------------|--------|
| Per-expert unstacked safetensors keys (`experts.{E}.{proj}.weight`) | Stacked per-layer tensors (`experts.gate_up_proj`/`experts.down_proj`, `[256,...]`) | v4 base model format (Qwen3.5/3.6-MoE family) | AIMER scoring and physical surgery scripts must slice tensors, not iterate per-expert keys |
| Flat `config.num_local_experts` | Nested `config.text_config.num_experts` | v4's composite VL config | Physical surgery's config rewrite target changes |
| Dual-axis (gen wp-bench + judge) gate, v3-measured fixed floors | Judge-only, single task, same-stack CI-aware TOST vs Gate B's full arm | v4 judge has no wp_gen/wp_judge task split (25-01) | `prune_gated_eval.py`'s gen machinery is entirely inapplicable; the eligibility bar is TOST-based, not a fixed number |
| 3-ratio sweep (25/50/75%) | Single targeted point (k=224), routed from Gate B's non-inferiority read | Gate B's pre-registered TOST verdict (`no_winner`, optimal_k=full) with routing option B | Narrower scope — no ratio-expansion decision needed this phase |

**Deprecated/outdated:** v3's `output/sieve/prune_set_for_phase13.json` regression-bar block (gen
wp_bench/judge floors) does not apply to v4 and must not be imported.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | AIMER score should be computed as the mean across s0/s1/s2 merged checkpoints (v3 shared-profile convention, `aimer_prune.py`'s own docstring example at line 28-33), not s1-only | Pattern 1 / Recommended Structure | If s1-only is used instead, the resulting keep-mask may over-fit s1's specific weight values; using the mean is directionally safer but this is an interpretive choice, not something CONTEXT.md pins explicitly `[ASSUMED]` |
| A2 | The Q8-quantized size scales ~linearly with parameter count when only expert weights shrink (§4's projected 37.8→~33.6 GiB) | §4 feasibility read below | The real GGUF Q8_0 quantization has block-size/padding overhead that could make the real number diverge from a linear projection by a few percent either direction; treat the projected number as directional, not a promise `[ASSUMED — flagged explicitly]` |
| A3 | REAP, if attempted, should route through an extended `_sieve_profile_vllm_patch`-style served hook rather than a new in-process calibration pass | Don't Hand-Roll / Pitfall 5 | If a plan instead schedules an in-process REAP forward-hook pass (as v3's `REAPCollector` design implies), it risks the same GB10 OOM Gate B/25-01 already had to route around |

**If this table is empty:** N/A — see entries above; all three are clearly interpretive/projective
claims, not verified facts, and are called out inline in their sections as well.

## Open Questions

1. **Is REAP worth attempting at all, given the effort/payoff?**
   - What we know: REAP is optional per the locked scope; v3 never ran it (AIMER's catastrophic
     failure made it moot per PRUNE-02's conditional rule); a GB10-safe REAP needs new served-model
     hook engineering (extending the counting-hook pattern to capture gate-weighted output norms,
     not just top-k selection counts).
   - What's unclear: whether AIMER@224 will even pass its own gate — if it fails (plausible, given
     Pitfall 2's precedent), REAP becomes moot immediately, mirroring v3's own conditional-skip.
   - Recommendation: gate REAP on AIMER@224 passing first, exactly as v3's `PRUNE-02` rule did; do
     not build the REAP served-hook extension speculatively.

2. **Exact reduction the AIMER-scored k=224 mask achieves on real quantized size, vs the
   projected ~33.6 GiB estimate in §4.**
   - What we know: the linear estimate from measured bf16 tensor shapes (§4).
   - What's unclear: the real llama.cpp/GGUF Q8_0 conversion's block-size behavior on a
     224-expert-per-layer model — not something worth spending GPU/CPU time measuring unless the
     gate-before-remove eval actually passes (no point quantizing a candidate that failed quality).
   - Recommendation: only run an actual GGUF conversion + size measurement if AIMER@224 clears
     the gate; otherwise the §4 projection is sufficient to support the expected `no_winner`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Patched vLLM + `_sieve_vllm_patch` | Gate-before-remove serving | ✓ (confirmed live in Gate B, T-25-03) | same install as Phase 25 | — |
| `serve_30_70_vllm.sh` + `SIEVE_MASK_NPY`/`LANGUAGE_MODEL_ONLY` toggles | Serving the masked judge | ✓ (used verbatim in Gate B) | — | — |
| `safetensors`/`torch`/`numpy`/`scipy` | AIMER scoring, TOST | ✓ (already used throughout the prune/sieve stack) | — | — |
| Docker (wp-bench grader) | v3's gen axis only | N/A — not needed | — | Gate C is judge-only; no Docker dependency at all |
| GB10 single 35B residency | Serial arm execution (one container at a time) | ✓ (same discipline as Gate B) | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — this phase is a pure re-wiring of already-working Phase 25 infrastructure.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | assert-based `--self-check` CLI convention (established repo-wide; no pytest framework used for these scripts) |
| Config file | none — each script's `if __name__ == "__main__": if "--self-check" in sys.argv` |
| Quick run command | `python -m scripts.<script> --self-check` (per-script, no GPU) |
| Full suite command | N/A — no aggregate test runner; each prune script is self-check'd independently, matching v3's precedent |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GATE4-04 (SC1) | Merge-of-record confirmed, no adapter files, `num_experts`/`num_hidden_layers` correct | integration (CPU, no GPU) | `ls models/Qwen3.6-35B-A3B-judge-v4-s1-merged/*adapter* ; python3 -c "import json; assert json.load(open('.../config.json'))['text_config']['num_experts']==256"` | ✅ (both are shell/python one-liners, no new file needed) |
| GATE4-04 (SC1) | Stacked-tensor AIMER scoring is scale-invariant + deterministic on a synthetic fixture | unit | `python -m scripts.aimer_prune_v4 --self-check` | ❌ Wave 0 — rewrite `_self_check` to build a stacked-tensor fixture (reuse `prune_apply_physical.py:190-228`'s `_write_fixture_checkpoint` builder as the template) |
| GATE4-04 (SC2) | Gated mask never drops a protected expert; k=224 budget honored | unit | `python -m scripts.sieve_expert_mask_inference` self-check (unchanged, already passes) | ✅ existing |
| GATE4-04 (SC2) | D2_security retention computed correctly from a synthetic judge capture | unit | reuse `prune_gated_eval.py`'s `_d2_security_mean` — add a `--self-check` fixture if none exists (check before Wave 0) | ❌ Wave 0 — confirm `prune_gated_eval.py`'s current `_self_check` (lines 425-479) does NOT already cover `_d2_security_mean`; if not, add a small fixture |
| GATE4-04 (SC2) | Same-stack TOST vs Gate B's full arm, CI-aware | unit | `.venv-tinker/bin/python scripts/sieve_v4_tost_verdict.py --self-check` (unchanged, already passes) | ✅ existing |
| GATE4-04 (SC3) | Eligibility gate fails closed on missing fields; smaller-k-wins tie-break; `no_winner` is a valid, correctly-serialized verdict | unit | `python -m scripts.prune_selection --self-check` (unchanged, already passes) | ✅ existing |
| GATE4-04 (SC3) | Physical surgery (if ever invoked) slices stacked tensors correctly, preserves shared_expert untouched, rewrites `text_config.num_experts` | unit | new self-check on a synthetic stacked-tensor fixture (extend `prune_apply_physical.py`'s existing fixture builder to the v4-style stacked+prefixed layout) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** run the touched script's own `--self-check` (all CPU, no GPU, seconds each).
- **Per wave merge:** run every prune-adjacent script's `--self-check` in sequence (`aimer_prune_v4`,
  `sieve_expert_mask_inference`, `sieve_v4_tost_verdict`, `prune_selection`,
  `prune_apply_physical_v4`) before scheduling any GPU serve time.
- **Phase gate:** the actual gate-before-remove GPU run (serve + capture + TOST + D2) is the
  phase's real acceptance test — self-checks only validate the CPU-side logic feeding it.

### Wave 0 Gaps
- [ ] Stacked-tensor AIMER `--self-check` fixture (new, or extend `aimer_prune.py`'s existing one
      with an `--arch stacked` fork) — covers GATE4-04 SC1.
- [ ] Confirm/extend `prune_gated_eval._d2_security_mean`'s test coverage — covers GATE4-04 SC2.
- [ ] Stacked-tensor `prune_apply_physical_v4` `--self-check` fixture — covers GATE4-04 SC3.
- [ ] No new test framework install needed — the assert-based `--self-check` convention already
      covers this whole stack; just extend fixtures, don't introduce pytest.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | not in scope — offline scoring + local GPU serving |
| V3 Session Management | no | not in scope |
| V4 Access Control | no | not in scope |
| V5 Input Validation | yes | `verify_protected_sha`-style sha256 re-check of the protected mask before every gate run (`prune_gated_eval.py:104-115`) — reuse the pattern, re-pin the sha256 in a v4-specific manifest rather than reusing v3's `prune_set_for_phase13.json` |
| V6 Cryptography | n/a | no crypto in this phase beyond the existing sha256 integrity check (not a cryptographic security boundary, an integrity check) |

### Known Threat Patterns for this phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Physical weight removal before the gated eval passes, silently shipping a security-capability regression | Tampering (of the eval-then-ship guarantee) | Hard precondition check in the surgery script's `main()`: refuse to run without a `gated/*_judge.json` with `pass: true` AND a D2_security-within-tolerance record already on disk (Pitfall 3) |
| Tampered/regenerated protected-expert mask silently flowing into a gate run | Tampering | sha256 re-verification before every serve, mirroring `prune_gated_eval.verify_protected_sha` (already self-checked at `prune_gated_eval.py:444-470`) — re-pin against a v4-specific manifest, not v3's `prune_set_for_phase13.json` |
| D2_security dimension regression shipped undetected because the pruned model still "parses" (non-zero rho) while security-specific reasoning quietly degrades | Tampering / Repudiation of the security guarantee | `_d2_security_mean` retention-vs-baseline check, gated at the same tolerance discipline as v3's `D2_SECURITY_TOLERANCE_PP=0.02` (`prune_selection.py:69`) — reuse this constant, it is domain-generic (a percentage-point tolerance on a 0-10 rubric score), not v3-stack-specific like the rho/wp_bench floors are |

## Sources

### Primary (HIGH confidence — read directly this session)
- `scripts/aimer_prune.py` (full read) — AIMER formula, per-expert key convention, self-check pattern
- `scripts/reap_prune.py` (full read) — REAP formula, forward-hook design, deferred-execution precedent
- `scripts/prune_gated_eval.py` (full read) — gate-before-remove contract, v3 floors, D2_security parsing
- `scripts/prune_selection.py` (full read) — eligibility gate, winner-selection rule
- `scripts/prune_overlap.py` (full read) — AIMER-vs-REAP Jaccard overlap analysis
- `scripts/prune_apply_physical.py` (full read) — physical surgery mechanics, per-expert key regexes
- `scripts/sieve_arch.py` (full read) — arch_dims/layer_strata/gb10_load_kwargs, v4 vs v3 config resolution
- `scripts/sieve_expert_mask_inference.py` (full read) — build_ksweep_mask/apply_mask, arch-agnostic
- `scripts/sieve_ksweep_v4_run.py` (full read) — v4 judge-only serve/capture/score driver, reuse template
- `scripts/sieve_v4_tost_verdict.py` (full read) — CI-aware TOST scorer, same-stack reference discipline
- `models/Qwen3.6-35B-A3B-judge-v4-s1-merged/config.json` + `model.safetensors.index.json` (direct inspection this session, incl. `safe_open` shape confirmation) — stacked-tensor layout, `text_config.num_experts`, `language_model.` prefix, shared_expert tensors
- `output/sieve-v4/optimal_k_v4.json`, `k_sweep_results_v4.json`, `eeff_report.json`, `protected_expert_mask.npy` — Gate B verdict, TOST reference (0.7935), protected-mask feasibility (max 98/layer)
- `output/prune/comparison_table.md`, `output/prune/selection.json`, `output/prune/prune_methodology.md` — v3's measured AIMER@25 catastrophic-failure precedent
- `.planning/phases/25-conditional-gate-b-moe-sieve-re-test/25-01-SUMMARY.md`, `25-02-SUMMARY.md` — Gate B routing decision, no-wp_gen/wp_judge-tokens finding
- `.planning/phases/26-conditional-gate-c-merge-prune-re-test/CONTEXT.md` — locked scope
- `.planning/ROADMAP.md` Phase 26 section, `.planning/STATE.md` — success criteria, size figures (37.8/30.2 GiB), routing history

### Secondary (MEDIUM confidence)
- `.planning/phases/23-final-evaluation/23-02-SUMMARY.md`, `23-03-SUMMARY.md` — provenance of the 37.8 GiB (v4 Q8_0 GGUF) / 30.2 GiB (v3 Q8) figures used in §4

### Tertiary (LOW confidence, flagged inline)
- The §4 Q8-size linear-scaling projection (37.8 → ~33.6 GiB at k=224) — a back-of-envelope estimate
  from measured bf16 tensor shapes, not a real GGUF conversion measurement (see Assumption A2).

## §4: Feasibility read — can k=224 close the 37.8→30.2 GiB gap?

**Measured this session** (from the actual checkpoint tensor shapes, `[VERIFIED]`):
- Per-expert params: `gate_up_proj` (1024×2048) + `down_proj` (2048×512) = 3,145,728 params/expert.
- Per-layer expert total (256 experts): 805,306,368 params. Across 40 layers: **~32.21B params** are
  expert weights.
- The merged bf16 checkpoint is **67 GiB on disk** (`du -sh`, this session). At bf16 (2 bytes/param),
  32.21B expert params = **~60.0 GiB — ~90% of the entire checkpoint's bytes** are expert weights
  (shared_expert + router + everything else is the remaining ~10%).

**The math against k=224:**
- k=224 drops 32/256 = **12.5%** of experts, uniformly per layer.
- Expected bf16/Q8 size reduction ≈ 12.5% × 90% (expert fraction of total) ≈ **~11.2% of total
  checkpoint size**.
- Applied to the 37.8 GiB Q8_0 artifact (v4 judge's actual shipped-size reference,
  `[CITED: .planning/phases/23-final-evaluation/23-02-SUMMARY.md]`): projected pruned size ≈
  **37.8 × (1 − 0.112) ≈ ~33.6 GiB** `[ASSUMED — linear-scaling estimate, A2]`.
- The gap to close is 37.8 − 30.2 = **7.6 GiB (20.1% reduction needed)**. k=224's own arithmetic
  ceiling delivers roughly **half** of that (~4.2 GiB), leaving the pruned v4 judge an estimated
  **~3.4 GiB (≈11%) above v3's Q8** even in the best case where the gate passes cleanly.

**What would need to be true to close the full gap:** a ~22.4% expert drop (≈k≈199, i.e., below
k=224 and close to k=192) would be needed on pure size arithmetic. But Gate B already measured k=192
as *not* CI-aware-TOST-equivalent (CI `[-0.026, +0.047]`, spilling past +2pp — `optimal_k_v4.json`),
and k=144 (a k that would close the gap with room to spare) measured a **real ~3.7pp degradation**
(`k_sweep_results_v4.json`, mean_diff −0.0373). So the k values that could plausibly close the size
gap are the same k values Gate B already found do not hold quality — this is not new evidence, it is
the same finding restated at the size-vs-quality tradeoff level.

**Expected disposition: `no_winner`.** Even setting aside whether AIMER@224 passes its own
gate-before-remove eval at all (Pitfall 2's precedent suggests real doubt), the arithmetic ceiling of
the one candidate this phase is authorized to test cannot reach parity with v3's shipped artifact.

**What evidence would overturn this:**
1. AIMER@224 measuring a Q8-conversion size reduction meaningfully larger than the linear 11.2%
   estimate (e.g., disproportionate savings from better quantization efficiency on a
   narrower-per-layer expert set) — would need an actual GGUF conversion + `du -sh` measurement to
   confirm, not assumed.
2. A human sign-off explicitly accepting a still-larger (~33-34 GiB) pruned v4 judge as "close
   enough" to ship despite not reaching exact parity with v3's 30.2 GiB — this is a policy call, not
   something this research can resolve; the plan should route the decision to a
   `checkpoint:human-verify` if the gate passes but sizes still don't match.
3. The gate-before-remove eval itself passing with quality clearly *better* than expected — the
   phase's stated success criteria treat "wins on quality but not size" as a question outside this
   research's scope; if it happens, the plan's disposition logic must still weigh it honestly against
   the SC3 either-disposition framing (a winning method+ratio ships pruned only if it's actually a
   net win, not merely non-regressive).

## Metadata

**Confidence breakdown:**
- Standard stack / tooling reuse: HIGH — every script was read in full this session; every
  contract (input/output, key format) is cited with file:line.
- Stacked-tensor adaptation: HIGH — confirmed via direct `safe_open` shape inspection of the real
  merged checkpoint, not inferred from documentation.
- Feasibility read (§4): MEDIUM overall (HIGH on the measured tensor-shape math, LOW/ASSUMED on the
  linear-scaling projection to Q8 size — explicitly flagged, not presented as verified).
- Pitfalls: HIGH — Pitfall 2 (AIMER catastrophic-failure precedent) is a measured v3 result, not a
  hypothesis; Pitfalls 3-5 are carried forward from explicitly-recorded threats in CONTEXT.md and
  prior debug sessions.

**Research date:** 2026-07-17
**Valid until:** this research is tied to the current on-disk checkpoint layout and Gate B's
specific measured numbers — treat as valid for the duration of Phase 26 only (re-verify if the
merged checkpoint is regenerated or if Gate B's receipts are ever revised).
