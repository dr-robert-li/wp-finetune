---
phase: 25-conditional-gate-b-moe-sieve-re-test
plan: 01
subsystem: infra
tags: [moe, routing-profile, vllm, sieve, e_eff, gb10]

requires:
  - phase: 22-sieve-protected-mask-tooling-adaptation
    provides: "arch-derived dims (40 layers, 256 experts), RoutingCollector/compute_eeff/compute_jaccard_stability wired for v4, extract_protected_mask single-task rule, vLLM MoE-block resolver"
provides:
  - "v4 judge routing profile over 34,855 judge prompts (routing_report.jsonl + jaccard_stability.json, per-stratum E_eff)"
  - "protected_expert_mask.npy [40,256] bool (3147 protected, mean 78.7/layer) from THIS profile's total counts"
  - "eeff_report.json (per-stratum + overall E_eff, E_eff/256 ratio 0.564, protected-per-layer stats)"
  - "ksweep_preregistration.md — stage-1 k grid {full,224,192,144,112} locked before any Wave-2 quality number"
  - "served-model profiling tooling (sitecustomize hook + launcher + driver) that sidesteps the in-process OOM"
affects: [25-02, phase-26, moe-sieve, prune]

tech-stack:
  added: []
  patterns:
    - "Served-model routing profile: profile via vLLM (memory manager holds weights) + a read-side gate hook, instead of an in-process from_pretrained that OOMs on the GB10 unified pool"
    - "Additive count-subtraction for a two-pass (full + subsample) profile against one running server (no restart)"

key-files:
  created:
    - scripts/_sieve_profile_vllm_patch/sitecustomize.py
    - scripts/serve_v4_profile_vllm.sh
    - scripts/drive_v4_routing_profile.py
    - output/sieve-v4/routing_report.jsonl
    - output/sieve-v4/jaccard_stability.json
    - output/sieve-v4/protected_expert_mask.npy
    - output/sieve-v4/eeff_report.json
    - output/sieve-v4/ksweep_preregistration.md
  modified:
    - scripts/sieve_arch.py
    - scripts/extract_protected_mask.py

key-decisions:
  - "APPROVED DEVIATION: profile via the SERVED model, not the plan's in-process profile_v4_judge.py — the in-process bf16 load OOMs on the GB10 (host staging ~50 GiB + device ~67 GiB > 121 GiB pool; no loader knob fixes it). Debug session .planning/debug/resolved/v4-judge-load-oom-recurrence.md. Served-hook counting is byte-identical to the in-process RoutingCollector (proven offline)."
  - "Stimulus = the full 34,855-example ratio_30_70/openai_train.jsonl (the same canonical stimulus v3 used), not the ~3-4K relabel sample the plan sketched — the served path made the full comparable stimulus cheap enough (~90 min, 0 failures), and 34,855 is directly v3-comparable + data-saturates the Jaccard test."
  - "SC1 supersession: Phase 24 (RL) skipped, so 'final policy' == the v4 judge SFT s1 merged checkpoint."
  - "k grid anchored to measured E_eff (max 224.5→224, mean 144.3→144, midpoint 192) with the aggressive floor at 112 (above the max 98 protected-per-layer, so protected-retention stays satisfiable)."

patterns-established:
  - "Served-model MoE profiling: reuse the mask patch's class resolver + gate hook, flip write→read; serve with --enforce-eager (hook must run per forward) and prefix-caching OFF (a cached prefix skips re-routing → undercount)."

requirements-completed: [GATE4-03]
---

## Accomplishments

**Profile (Task 1).** Produced the v4 judge s1 routing profile over **34,855 judge prompts** (17.4M routed
tokens/layer), all 40 layers, per-stratum E_eff. `routing_report.jsonl` + `jaccard_stability.json` written.
Loader guard (T-25-01, AutoModelForImageTextToText, 0 missing keys) ran in the served container. **E_eff mean
144.3/256**, per-stratum DeltaNet 144.1 / attention 145.0. **Jaccard mean 0.9722** (min 0.7778); the
min-based gate reads FALSE because the 5 flattest-routing layers (0,4,15,25,26) have an intrinsic top-8 tie
— confirmed by a boundary-margin diagnostic (rank8-vs-rank9 gap ~0.02%), NOT a data shortfall (a 563→34,855
expansion stabilized 10 of the 15 originally-unstable layers).

**Mask + E_eff report (Task 2).** `protected_expert_mask.npy` [40,256] bool, **3147 protected (mean 78.7,
min 65, max 98 / layer)**, single-task mean-threshold rule (v4 judge has no wp_gen/wp_judge tokens), shared
expert absent by construction (256-wide router). `eeff_report.json`: per-stratum + overall E_eff, **E_eff/256
ratio 0.564** — below v3's 0.69–0.77, i.e. relatively MORE redundancy headroom than v3 → sub-full k is worth
probing rather than assumed dead.

**Pre-registration (Task 3).** `ksweep_preregistration.md` — stage-1 grid **{full, 224, 192, 144, 112}** with
its E_eff-derivation rule, s1-primary gating (ensemble reserved for the winner), TOST ε=2pp CI-aware against
the same-stack vLLM full arm + a full-arm sanity floor (~0.72), both dispositions (optimal_k / no_winner)
wired to Phase 26, and the SC1 Phase-24-skip supersession. Locked before any Wave-2 rho exists.

## Deviations

- **Loader:** plan's in-process `profile_v4_judge.py` OOMs on the GB10 → replaced with the served-model path
  (new tooling: `_sieve_profile_vllm_patch/sitecustomize.py`, `serve_v4_profile_vllm.sh`,
  `drive_v4_routing_profile.py`). Approved, root-caused, and verified; debug session resolved.
- **Stimulus:** used the full 34,855-example ratio_30_70 set (v3-comparable) instead of a ~3-4K sample.
- Three bugs surfaced only by the real container run were fixed (flush np.save name-munging, enforce-eager
  warmup contamination via a subtracted baseline, prompt-length 400s via truncate-to-2047 + tolerant sender).

## Verification

- routing_report.jsonl: 40 layer rows, 34,855 prompts, strata_eeff present (deltanet+attention). ✅
- protected_expert_mask.npy: [40,256] bool, 3147>0 protected, shared expert excluded. ✅
- eeff_report.json: per-stratum + overall + E_eff/256 ratio (0.564) + protected-per-layer. ✅
- ksweep_preregistration.md: grid + derivation, s1 gate, TOST ε=2pp same-stack ref, both dispositions, SC1. ✅

## Next

Plan 25-02 (Wave 2): run the k-sweep at {full,224,192,144,112} through the patched vLLM, gate each arm's s1
rho vs the same-stack full arm by CI-aware TOST (ε=2pp), and record `optimal_k` or `no_winner` →
`output/sieve-v4/k_sweep_results_v4.json`.
