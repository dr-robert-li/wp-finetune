# Phase 22 — Sieve/Protected-Mask Tooling Adaptation — VALIDATION

**Requirement:** GATE4-02 (single requirement, Phase 22).
**Goal (ROADMAP):** the MoE-Sieve profiler + protected-mask pipeline are adapted for the v4 judge's
256-expert / shared-expert / mixed-DeltaNet-Attention-strata architecture, verified before Conditional
Gate B (Phase 25), closing independently of the RL gate (Phase 24).

**Target model:** the v4 JUDGE — `models/Qwen3.6-35B-A3B-judge-v4-{s0,s1,s2}-merged` (present on disk).
NOT a gen model. The protected-mask REFERENCE for the v4 judge is profiled FRESH by Phase 25; Phase 22 only
makes the tooling SUPPORT a [40,256] mask + strata (v3's [48,128] 1480-expert mask does not carry over).

---

## The four ROADMAP success criteria → task map

| SC | ROADMAP text | Covered by | Verify |
|----|--------------|-----------|--------|
| SC1 | Profiler traversal corrected + n_experts 128→256 across the 4 affected scripts | 22-01 T1 (arch_dims, resolve_moe_layers), 22-01 T2 (wire 6 consumers), **22-02 T1** (40 hooks resolve on the real model) | pytest dims fixtures + `tooling_smoke.json` hooks_registered==40, router_logits_last_dim==256 |
| SC2 | DeltaNet-MoE vs Gated-Attention-MoE as separate strata in per-layer E_eff + k-sweep | 22-01 T1 (layer_strata), 22-01 T2 (per-stratum E_eff + stratum-aware mask meta), 22-02 T1 (strata on real model) | pytest strata pattern + `tooling_smoke.json` strata_counts {deltanet:30, attention:10} at [3,7,…,39] |
| SC3 | Empirical check shared expert never in router_logits, excluded from sweepable set | **22-02 T1** (real forward pass) | `tooling_smoke.json` router_logits_last_dim==256==config_num_experts, shared_expert_in_router_logits=false, shared_expert_module_present=true |
| SC4 | Tooling verified ready before Gate B; closes independent of RL gate | 22-02 T1 (readiness receipt) | `tooling_smoke.json` status=pass; phase depends only on Phase 20 (not 24) |

---

## Per-task verify map (every task has a runnable automated gate — Nyquist)

| Plan-Task | Automated verify |
|-----------|------------------|
| 22-01 T1 (sieve_arch + tests) | `pytest tests/test_sieve_arch.py -q` + `python scripts/sieve_arch.py --self-check` |
| 22-01 T2 (wire consumers + update tests) | `pytest tests/test_protected_mask.py tests/test_sieve_cross_seed_overlap.py tests/test_sieve_ksweep_mask.py tests/test_sieve_protected_retention.py tests/test_sieve_arch.py -q` + 3 `--self-check` entry points |
| 22-01 T3 (vLLM patch class resolution) | `pytest tests/test_sieve_vllm_patch.py -q` |
| 22-02 T1 (GB10 smoke) | offline re-assert of every `output/sieve-v4/tooling_smoke.json` field (status/hooks/dim/strata/root) |

---

## Concrete architecture facts pinned into the plans (from config.json + Phase 20/21 forensics)

- **Dims:** `num_hidden_layers=40`, `num_experts=256`, `num_experts_per_tok=8`, `shared_expert_intermediate_size=512`, `model_type="qwen3_5_moe"` (verified live in `models/Qwen3.6-35B-A3B/config.json`).
- **Strata:** `text_config.layer_types` — 30 `linear_attention` (DeltaNet-MoE) + 10 `full_attention` (Gated-Attention-MoE); `full_attention` at exactly indices **[3,7,11,15,19,23,27,31,35,39]** (`full_attention_interval=4`, so `(i%4)==3`). Verified by reading the 40-entry array.
- **Fused experts (informational):** per-layer `mlp.experts.gate_up_proj`/`down_proj` (256 stacked); the profiler reads `router_logits` via `mlp.gate`, never expert weight tensors, so fused-vs-unfused storage needs NO handling (ARCHITECTURE.md §3).

## Reconciled conflict — traversal path (binding for the executor)

ROADMAP SC1 says the literal path is `model.model.language_model.layers`. **Phase 20-04 empirically found the
LIVE in-memory module tree is FLAT `model.model.layers.*`** (AutoModelForCausalLM → Qwen3_5MoeForCausalLM;
`language_model.*` is the on-disk save/load convention only, auto-restored by transformers). ARCHITECTURE.md
§3 itself warns a wrong path "will silently produce zero hooks (not an error)". Therefore the plans do NOT
hardcode the ROADMAP's literal path — `sieve_arch.resolve_moe_layers` tries candidate roots in order
(`model.model.layers` → `model.model.language_model.layers` → `model.language_model.layers`), picks the first
that yields 40 layers, and **asserts hook-count == 40 (raises on mismatch)**. The 22-02 receipt records the
root that actually resolved, settling the question empirically. This satisfies SC1's INTENT (correct
traversal for the new base) without the silent-zero-hook trap the literal text would cause.

## Two additional catches folded into the plans

1. **No task tokens on v4:** the v4 judge tokenizer (vocab 248320) has no `<wp_gen>`/`<wp_judge>` extension
   (20-04). RoutingCollector's hardcoded IDs (151669/151670) would silently never match. Fix (22-01 T2):
   task-token IDs become constructor params resolved from the tokenizer via `resolve_task_token_ids`; on the
   v4 judge they resolve to (None, None) and tagging degrades to total-only — correct for the routing-shape
   smoke. The v4 judge's gen/judge split strategy (single-role model) is Phase 25's stimulus-design call.
2. **vLLM class name is container-only:** vLLM is NOT importable on the host (tinker venv); the qwen3_5_moe
   MoE-block class name is confirmable only inside the serving container. Fix (22-01 T3): ordered
   candidate-list resolver (qwen3_5_moe/qwen3_next first, qwen3_moe fallback), fail-loud if none resolve,
   selection logic unit-tested with stubs; live confirmation deferred to the container (22-02 / Phase 25).

---

## Multi-source coverage audit (all four source types — every item COVERED)

| Source | Item | Status | Plan |
|--------|------|--------|------|
| GOAL (ROADMAP Phase 22) | Profiler/mask pipeline adapted for 256-expert/shared/mixed-strata, verified before Gate B | COVERED | 22-01, 22-02 |
| REQ | GATE4-02 (traversal + n_experts 128→256 ×4 scripts + strata split + empirical shared-expert-exclusion) | COVERED | 22-01 (code), 22-02 (empirical) |
| RESEARCH (ARCHITECTURE.md §3/§6) | traversal-path fix; n_experts bump across extract_protected_mask / sieve_expert_mask_inference / sieve_ksweep_run / sieve_cross_seed_overlap; shared-expert-not-in-router_logits (verify empirically); NO fused-weight handling needed | COVERED | 22-01 T2 (4 named scripts + 2 profilers; sieve_ksweep_run audited shape-generic), 22-02 T1 (empirical) |
| RESEARCH (design guidance) | strata split is narrower than "two uniform stacks" (routing width uniform → mask stays [40,256]; only attention-type paths differ); vLLM patch may need qwen3_5_moe class; v3 compat additive | COVERED | 22-01 T1/T2 (per-stratum reporting on uniform [40,256]), 22-01 T3 (patch class), all tasks additive |
| CONTEXT (STATE.md) | target = v4 JUDGE not gen; protected-mask reference comes from Phase 25's own profile; bounded GB10 smoke not full 6h30m pass; changes additive to shared v3 scripts | COVERED | 22-02 (judge s1, bounded, receipt); 22-01 (config/data-derived, v3 fixtures retained) |

**No MISSING items.** Exclusions (not gaps): the actual v4 k-sweep + protected-mask profiling (Phase 25 /
GATE4-03); the vLLM-serving-time confirmation that masking APPLIES on the qwen3_5_moe block (Phase 25);
sieve_ksweep_run.py's v4 model paths + 256-scaled k-budgets (Phase 25 config).

## v3 back-compat guarantee

Every change is config-/data-derived and self-adapting (48/128 for a v3 config, 40/256 for v4) or gated
behind tokenizer/config resolution. Existing v3 tests keep their [48,128] fixtures AND gain [40,256] cases.
No v3 artifact (`output/profiling/reasoning-merged-v4/protected_expert_mask.npy`) is mutated. The
`mask.sum()==1480` and `shape==(48,128)` hard asserts are the only v3-specific lines removed (parameterized
to the loaded mask), because the v4 mask is a fresh Phase-25 profile of unknown count.
