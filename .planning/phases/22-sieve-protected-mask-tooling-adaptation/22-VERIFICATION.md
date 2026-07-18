---
phase: 22-sieve-protected-mask-tooling-adaptation
verified: 2026-07-15T16:35:00+10:00
status: passed
score: 4/4 roadmap success criteria verified (9/9 plan must_haves truths verified)
behavior_unverified: 0
overrides_applied: 0
---

# Phase 22: Sieve/Protected-Mask Tooling Adaptation Verification Report

**Phase Goal:** The MoE-Sieve profiler and protected-mask pipeline are adapted for the v4 judge's
256-expert, shared-expert, mixed-DeltaNet/Attention-strata architecture, so Conditional Gate B (Phase 25)
can run against audited tooling instead of tooling built for the old 128-expert uniform-attention base.
**Requirement:** GATE4-02 (single requirement, Phase 22).
**Verified:** 2026-07-15
**Status:** passed
**Re-verification:** No — initial verification.

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth (ROADMAP SC) | Status | Evidence |
|---|------|--------|----------|
| SC1 | Profiler module-traversal path corrected + n_experts bumped 128→256 across the 4 affected scripts | ✓ VERIFIED (intent satisfied, literal path text superseded — see note) | `sieve_arch.resolve_moe_layers` tries an ordered candidate list and asserts hook-count==n_layers (raises, not silent); `output/sieve-v4/tooling_smoke.json`: `hooks_registered: 40 == expected_n_layers: 40`, `resolved_traversal_root: "model.language_model.layers"` — real forward pass on the actual 67 GiB checkpoint, not a mock. `extract_protected_mask.py`/`sieve_cross_seed_overlap.py`/`sieve_expert_mask_inference.py`/`sieve_ksweep_run.py` (the 4 named scripts) all derive dims from config/JSONL, no `128` load-bearing literal remains (grep sweep, confirmed manually — see Anti-Patterns). **Note:** ROADMAP's literal path text `model.model.language_model.layers` does not match the empirically resolved root `model.language_model.layers`; this is explicitly reconciled in 22-VALIDATION.md (Phase 20-04's empirical flat-tree finding was known to override the ROADMAP's literal guess before 22-01 was planned) and closed empirically by the 22-02 receipt recording the actual root. SC1's *intent* (traversal correctness + no silent zero-hook) is met; the literal path string in ROADMAP.md is stale documentation, not a functional gap. |
| SC2 | DeltaNet-MoE and Gated-Attention-MoE treated as separate strata in per-layer E_eff + k-sweep masking, not one uniform stack | ✓ VERIFIED | `sieve_arch.layer_strata` derives strata from `config.layer_types`; `profile_merged_model.py` reports `strata_eeff` dict keyed `deltanet`/`attention` (mean/max/var/n_layers each) alongside collapsed stats, verified in diff read (not just claimed). Empirically on the real model: `output/sieve-v4/tooling_smoke.json`: `strata_counts: {deltanet: 30, attention: 10}`, `attention_layer_indices: [3,7,11,15,19,23,27,31,35,39]` — exact match to config.json's real `layer_types` array. |
| SC3 | Empirical check confirms shared expert never appears in `router_logits` and is excluded from the sweepable/prunable set | ✓ VERIFIED | `output/sieve-v4/tooling_smoke.json`: `router_logits_last_dim: 256 == config_num_experts: 256`, `shared_expert_in_router_logits: false`, `shared_expert_module_present: true` — captured from a real forward-hook fire on the `mlp.gate` output of the actual loaded model (`scripts/sieve_v4_tooling_smoke.py:133-143`), not inferred from tensor naming. |
| SC4 | Adapted tooling verified ready before Conditional Gate B (Phase 25); phase closes independently of the RL gate's outcome | ✓ VERIFIED | `output/sieve-v4/tooling_smoke.json`: `status: "pass"` (computed from hard asserts on every field, `scripts/sieve_v4_tooling_smoke.py:178-188`, not hand-set). Phase 22's `depends_on` is Phase 20 only (both plan frontmatters: `depends_on: []` for 22-01, `depends_on: ["22-01"]` for 22-02) — no dependency on Phase 24 (RL gate). ROADMAP.md: "Plans: 2/2 plans complete". REQUIREMENTS.md line 405: `GATE4-02 | Phase 22 | Complete`. |

**Score:** 4/4 ROADMAP success criteria verified.

### Plan-Level Must-Haves (22-01 + 22-02 frontmatter truths)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | No code path hardcodes 48/128 in a load-bearing position | ✓ VERIFIED | Grep sweep of all 9 touched files + manual read of every call site (see 22-REVIEW.md) — only import-back-compat module constants, docstring defaults never read, and comment/docstring text remain. |
| 2 | Each MoE layer carries a stratum label from config.layer_types; attention at [3,7,...,39] | ✓ VERIFIED | `tests/test_sieve_arch.py` + `output/sieve-v4/tooling_smoke.json` empirical match. |
| 3 | Per-layer E_eff reportable per stratum, not collapsed | ✓ VERIFIED | `profile_merged_model.py` `strata_eeff` dict, diff-read confirmed. |
| 4 | vLLM patch resolves qwen3_5_moe class first, fails LOUD if none resolve | ✓ VERIFIED | `_resolve_moe_block_class` ordered candidates + `RuntimeError` on exhaustion, re-raised (not swallowed) by `_install`; 5/5 `tests/test_sieve_vllm_patch.py` tests pass. |
| 5 | v3 [48,128]/1480 paths still load and pass | ✓ VERIFIED | v3 fixture cases retained in all 5 updated test files; 47/47 tests green including v3 cases. |
| 6 (22-02) | GB10 load registers EXACTLY 40 hooks on the real model | ✓ VERIFIED | Receipt `hooks_registered: 40`, committed at `8e7fe00`. |
| 7 (22-02) | Router forward output last dim == 256; shared expert has no router_logits entry | ✓ VERIFIED | Receipt `router_logits_last_dim: 256`, `shared_expert_in_router_logits: false`. |
| 8 (22-02) | Every profiled layer labelled deltanet/attention matching config.layer_types | ✓ VERIFIED | Receipt `strata_counts`/`attention_layer_indices` exact match. |
| 9 (22-02) | Receipt records resolved traversal root, declares tooling ready | ✓ VERIFIED | Receipt `resolved_traversal_root: "model.language_model.layers"`, `status: "pass"`. |

**Score:** 9/9 plan must_haves truths verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `output/sieve-v4/tooling_smoke.json` | Committed receipt, status=pass, all SC fields asserted | ✓ VERIFIED | Exists, committed at `8e7fe00 feat(22-02): bounded GB10 tooling smoke on real v4 judge — GATE4-02 SC1-4 pass`; `git ls-files` confirms tracked (force-added past `output/` gitignore, matching repo convention). All fields present and correct: `hooks_registered=40`, `expected_n_layers=40`, `router_logits_last_dim=256`, `config_num_experts=256`, `shared_expert_in_router_logits=false`, `shared_expert_module_present=true`, `strata_counts={deltanet:30,attention:10}`, `attention_layer_indices=[3,7,11,15,19,23,27,31,35,39]`, `resolved_traversal_root="model.language_model.layers"`, `status="pass"`. |
| `scripts/sieve_arch.py` | New arch-awareness helper | ✓ VERIFIED | Exists, all 5 functions implemented per spec, self-check prints OK, 13/13 unit tests pass. |
| `scripts/sieve_v4_tooling_smoke.py` | GB10 smoke harness | ✓ VERIFIED | Exists, produces the receipt above; `AutoModelForImageTextToText` loader fix reviewed (see 22-REVIEW.md). |
| 6 adapted profiler/mask/k-sweep scripts + vLLM patch | Config/data-derived dims, no v3 hardcode | ✓ VERIFIED | All 7 files diffed and read (`profile_base_model.py`, `profile_merged_model.py`, `extract_protected_mask.py`, `sieve_cross_seed_overlap.py`, `sieve_expert_mask_inference.py`, `sieve_protected_retention.py`, `_sieve_vllm_patch/sitecustomize.py`). |

### Test Suite

| Suite | Command | Result | Status |
|-------|---------|--------|--------|
| Full sieve test set (47 tests, 22-01-SUMMARY claim) | `pytest tests/test_sieve_arch.py tests/test_protected_mask.py tests/test_sieve_cross_seed_overlap.py tests/test_sieve_ksweep_mask.py tests/test_sieve_protected_retention.py tests/test_sieve_vllm_patch.py -q` | `47 passed in 0.14s` | ✓ PASS (re-run independently in this verification, not taken from SUMMARY) |
| `sieve_arch.py --self-check` | `.venv-tinker/bin/python scripts/sieve_arch.py --self-check` | `OK` | ✓ PASS |
| `sieve_expert_mask_inference.py --self-check` | same | `self-check OK` | ✓ PASS |
| `sieve_cross_seed_overlap.py --self-check` | same | `self-check OK` | ✓ PASS |
| `sieve_protected_retention.py --self-check` | same | `self-check OK` | ✓ PASS |

### Grep Sweep (load-bearing 48/128/1480)

`grep -n -E "\b(48|128|1480)\b"` across the 9 touched scripts + `sieve_arch.py`/`sieve_v4_tooling_smoke.py`
found only: (a) import-back-compat module constants (`N_LAYERS=48`/`N_EXPERTS=128` in
`sieve_cross_seed_overlap.py`, `sieve_expert_mask_inference.py`) confirmed unread by any function body except
as an empty-file fallback; (b) `RoutingCollector.__init__` keyword defaults, confirmed overridden at every
real call site; (c) `compute_eeff`'s unused `n_experts` parameter (pre-existing, not introduced this phase);
(d) docstrings/comments describing v3 for contrast; (e) `sieve_arch.demo()`'s intentional v3 fixture
literals. No load-bearing occurrence in an actual v4 code path. See 22-REVIEW.md for the full breakdown.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GATE4-02 | 22-01, 22-02 | Sieve/protected-mask tooling adapted before Gate B | ✓ SATISFIED | REQUIREMENTS.md line 405: `Complete`. All 4 ROADMAP SCs verified above. |

No orphaned requirements — GATE4-02 is the phase's single requirement and it's claimed by both plans.

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers, no empty-return stubs, no
hardcoded-empty-data patterns in any of the 9 reviewed files (see 22-REVIEW.md for the full sweep).

### Carry-Forward for Phase 25 (recorded per task instructions)

1. **Profiling target for the full k-sweep pass = the v4 judge s1 merged checkpoint**
   (`models/Qwen3.6-35B-A3B-judge-v4-s1-merged`), loaded via `AutoModelForImageTextToText`
   (`Qwen3_5MoeForConditionalGeneration`) — **not** `AutoModelForCausalLM`. This is the empirically-confirmed
   loader for this checkpoint's VL-composite, nested `model.language_model.layers.*` save convention. A
   *different* v4-family checkpoint shape (e.g. a flattened text-only merge matching Phase 20's LoRA-merge
   convention) would need the same meta-device key-diff re-verification before committing a full profiling
   run — the two known v4-family checkpoint shapes in this repo require different loader classes (22-02-SUMMARY
   "Open note for Phase 25").
2. **The fresh protected mask for the v4 judge comes from Phase 25's own profiling run**, not from this
   phase or from any v3 artifact. Phase 22 explicitly does not produce a v4 protected mask — it only proves
   the tooling that will produce one is correct. `sieve_protected_retention.py`'s v3-specific
   `shape==(48,128)`/`sum==1480` asserts were deliberately dropped (not replaced with v4-specific literals)
   because that count is unknown until Phase 25 profiles it.
3. **The vLLM router-mask patch's class-resolution needs live confirmation at Phase 25 serving time.** The
   ordered candidate list (`qwen3_5_moe`/`qwen3_next` first, `qwen3_moe` fallback) is best-known, not yet
   confirmed against an installed vLLM (vLLM is not importable on this host — only inside the GPU serving
   container). The resolver's scan-fallback (single `*SparseMoeBlock` class in a resolved module) gives it a
   second chance even if the literal candidate class name misses, and it fails loud (raises) rather than
   silently serving unmasked if no candidate resolves at all — but the exact class name is still unverified
   until Phase 25 actually serves the v4 judge through vLLM with `SIEVE_KEEP_MASK_NPY` set.

### Human Verification Required

None. All must-haves and ROADMAP success criteria are verifiable from committed artifacts, test runs, and
code inspection — no visual/UX/real-time behavior in scope for this phase.

### Gaps Summary

No gaps. All 4 ROADMAP success criteria, all 9 plan-level must-have truths, the committed receipt, the 47-test
suite, the 4 self-checks, and the grep sweep for load-bearing v3 literals all verify against the actual
codebase — not just SUMMARY.md claims. The one documentation nit (SC1's literal traversal-path text vs. the
empirically-resolved root) is pre-reconciled in 22-VALIDATION.md and does not affect functional correctness;
recorded as an Info-level note in 22-REVIEW.md, not a gap.

---

*Verified: 2026-07-15*
*Verifier: Claude (gsd-verifier)*
