# Phase 25 Planning Validation — Conditional Gate B: MoE-Sieve Re-Test on the v4 JUDGE

**Planned:** 2026-07-15
**Requirement:** GATE4-03 (single requirement)
**Plans:** 2 (25-01 profile+mask+pre-reg [W1], 25-02 driver+sweep+verdict [W2])
**Mode:** standard, coarse granularity, MVP_MODE false, tdd_mode false

---

## Multi-Source Coverage Audit

Every source item is COVERED by a plan. No item is MISSING; no scope reduced.

### GOAL (ROADMAP Phase 25 goal + success criteria)

| Item | Source | Covered by | Status |
|------|--------|------------|--------|
| Determine whether 256-expert routing has redundancy for expert-drop at equal quality | ROADMAP Phase 25 Goal | 25-01 (profile+E_eff) + 25-02 (sweep+TOST) | COVERED |
| SC1: re-profile on "Phase 24's final policy" using Phase-22 adapted tooling | ROADMAP SC1 | 25-01 T1 — **superseded**: Phase 24 (RL) SKIPPED → final policy == v4 judge SFT s1; documented in 25-01 objective + pre-reg receipt, NOT silently ignored | COVERED (supersession documented) |
| SC2: k-sweep at multiple budgets, TOST ε=2pp, CI-aware (bootstrap lower bound) | ROADMAP SC2 | 25-01 T3 (grid+TOST spec) + 25-02 T1/T2/T3 | COVERED |
| SC3: close with optimal_k OR no_winner; either unblocks Phase 26 | ROADMAP SC3 | 25-02 T3 (verdict + Phase-26 routing) | COVERED |

### REQ (REQUIREMENTS.md phase_req_ids)

| ID | Description | Covered by | Status |
|----|-------------|------------|--------|
| GATE4-03 | MoE-Sieve k-sweep re-test (TOST ε=2pp, CI-aware) on adapted tooling | 25-01 + 25-02 (both `requirements: [GATE4-03]`) | COVERED |

### RESEARCH (V4-RERUN-ROADMAP Gate B + carry-forward lessons)

| Item | Source | Covered by | Status |
|------|--------|------------|--------|
| 2x expert count → larger redundancy headroom hypothesis | Gate B (a) | 25-01 eeff_report E_eff/256 ratio | COVERED |
| 6h30m profiling floor warning (treat as floor, not ceiling) | Gate B (d) | 25-01 T1 bounded ~3-4K stimulus, justified vs floor | COVERED |
| Tooling adaptation must land BEFORE sweep (do not run against unaudited tooling) | Gate B (c) | Phase 22 (done); 25-01/02 consume it | COVERED |
| Carry-forward #1: 8192-token cap (truncation-aware) | lessons | 25-02 max_tokens=8192 (T-25-05) | COVERED |
| Carry-forward #3: --parallel / detached long runs | lessons | 25-01 detached profile, 25-02 detached sweep | COVERED |
| Carry-forward #4: CI-aware (bootstrap lower bound clears the bar) | lessons | 25-01 TOST spec + 25-02 verdict | COVERED |
| Carry-forward #5: pre-registration discipline (lock before results) | lessons | 25-01 T3 stage-1 pre-reg | COVERED |
| REAP 20%-drop "competitive" community signal (informal, not TOST-grade) | Gate B (a) | Informative only — the grid probes sub-full budgets so the equivalence boundary is measured, not assumed | COVERED (as prior, not gate) |

### CONTEXT (STATE.md reopened-scope + Phase 22 carry-forwards)

| Item | Source | Covered by | Status |
|------|--------|------------|--------|
| Target = v4 JUDGE only (s1 primary; ensemble seeds exist) | reopened-scope | 25-01/02 target `...-judge-v4-s1-merged`; gen/wp-bench dropped | COVERED |
| Success = expert-drop k within TOST ε=2pp of full → Phase 26 below 30.2 GiB Q8 | reopened-scope | 25-02 verdict | COVERED |
| no_winner (optimal_k=full) is a valid recorded outcome | reopened-scope | 25-02 T3, both dispositions route Phase 26 | COVERED |
| Loader = AutoModelForImageTextToText (NOT AutoModelForCausalLM) | 22-02 carry-forward #1 | 25-01 T1 loader guard (T-25-01) | COVERED |
| Fresh protected mask from THIS profile, not v3 | 22-02 carry-forward #2 | 25-01 T2 | COVERED |
| vLLM patch class-resolution needs LIVE confirmation | 22-02 carry-forward #3 | 25-02 T2 full-arm doubles as the check (T-25-03) | COVERED |
| Shared expert excluded by construction (256-wide router, shared absent) | tooling_smoke.json | 25-01 T2 mask build | COVERED |
| Per-stratum E_eff (deltanet vs attention), not collapsed | GATE4-02 SC2 | 25-01 T1/T2 strata_eeff | COVERED |

**Exclusions (not gaps):** Phase 24 (RL) is skipped by milestone decision — not a Phase 25 item. Gen/wp-bench
axis is out of scope (judge-only artifact per reopened-scope). Physical weight removal is Phase 26 (GATE4-04).

---

## Key planner decisions (with rationale)

1. **Judge-only sweep, single-seed s1 as the gating metric; ensemble reserved for a candidate winner.**
   The v4 judge has no `<wp_gen>`/`<wp_judge>` tokens (vocab 248320 — sieve_arch degrades to total-only), so
   routing is total-only and the whole model is the judge. Single-seed s1 is the cheapest same-stack gate;
   v3 showed masked judge output collapses to unparseable (0/121) well before the equivalence boundary, so a
   3-seed ensemble cannot rescue a collapsed arm. Ensemble (s0/s1/s2 merged) confirms a candidate winner
   only — bounding wall-clock (boot 385s + 121 @8192 per seed per arm).

2. **TOST reference = the same-stack vLLM full arm measured IN the sweep, not the llama.cpp Q8 0.8067.**
   The masked arms are served bf16 via patched vLLM; the sanity_gate_recalibration lesson (v3) requires both
   TOST sides on the same stack. bf16-vLLM-served s1 (~0.7872, ext_q8_results.json secondary_reads) is the
   expected full-arm ballpark and the sanity anchor.

3. **Bounded judge stimulus (~3-4K from openai_train_relabel_v1.jsonl), disjoint from the 121-item eval set.**
   Judge-only sweep → judge-distribution profile; ~10% of v3's full set where v3's Jaccard stability already
   held; keeps wall-clock under the 6h30m floor. Profile ≠ eval set (leakage hygiene).

4. **Protected mask = single-task mean-threshold on total counts.** D-03's gen∧judge co-activation rule
   degenerates to one task when there is only one task (no task tokens on the v4 judge).

5. **Two-stage pre-registration via the wave boundary.** Stage-1 grid+TOST spec is committed in 25-01 (W1)
   from measured E_eff BEFORE any rho exists; 25-02 (W2) reads it verbatim and cannot mutate it (T-25-06).

6. **Human sign-off on the verdict** (25-02 T3, blocking checkpoint) mirrors v3 Phase 11's optimal_k
   sign-off — the disposition routes Phase 26.

---

## Goal-backward must-haves (summary)

- **Goal:** a recorded, CI-aware TOST verdict (optimal_k or no_winner) on the v4 judge, on audited tooling,
  against a same-stack reference, with a locked grid.
- **Truths:** correct loader (not random weights); per-stratum E_eff; fresh mask from THIS profile; grid
  locked before rho; full arm = same-stack TOST reference; 8192-cap captures; both dispositions route Ph26.
- **Artifacts:** routing_report.jsonl, jaccard_stability.json, protected_expert_mask.npy, eeff_report.json,
  ksweep_preregistration.md (W1); sieve_ksweep_v4_run.py, sieve_v4_tost_verdict.py, k_sweep_results_v4.json,
  optimal_k_v4.json (W2).
- **Key links (where it breaks):** AutoModel class (spoofing a garbage profile); TOST reference stack;
  capture token cap; grid mutation after seeing rho; vLLM patch silent-unmask.

## Reachability check

Every must-have artifact has a concrete creation path: profile (25-01 T1 detached run) → mask (T2
extract_protected_mask) → E_eff report (T2) → pre-reg (T3) → driver (25-02 T1) → sweep receipts (T2) →
verdict (T3). No unreachable artifact. All scripts exist or are minimal adaptations of existing, tested
Phase-22 primitives.
