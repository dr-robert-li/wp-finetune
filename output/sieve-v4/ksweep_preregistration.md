# k-sweep pre-registration (STAGE 1) — v4 judge MoE-Sieve re-test (GATE4-03)

**Locked:** 2026-07-16, BEFORE any Wave-2 quality number exists. Committed by Plan 25-01.
Wave 2 (Plan 25-02) reads this file and MUST NOT alter the grid, metric, or TOST spec after seeing any rho.

**Profiled policy (SC1 supersession):** ROADMAP Phase 25 SC1 says routing is re-profiled on "Phase 24's
final policy." **Phase 24 (Conditional Gate A — RL) was SKIPPED** this milestone (no new reward family;
STATE.md 2026-07-15 reopened-scope note). With no RL step, the final policy **IS the v4 judge SFT s1
checkpoint** — `models/Qwen3.6-35B-A3B-judge-v4-s1-merged`. That is the checkpoint profiled and swept.

**Profile provenance:** produced via the SERVED-model path (`scripts/serve_v4_profile_vllm.sh` +
`scripts/drive_v4_routing_profile.py`), an approved deviation from the plan's in-process
`profile_v4_judge.py` — the in-process bf16 load OOMs on the GB10 (see
`.planning/debug/resolved/v4-judge-load-oom-recurrence.md`). The served hook's counting is byte-identical to
the in-process `RoutingCollector` (proven offline). Stimulus: 34,855-example `ratio_30_70/openai_train.jsonl`
(the same canonical stimulus v3 used; 17.4M routed tokens/layer). 0 request failures.

---

## 1. Measured E_eff (from `eeff_report.json`) — the grid input

| | value |
|---|---|
| overall E_eff mean | 144.3 / 256 |
| overall E_eff max | 224.5 / 256 |
| **E_eff / n_experts ratio** | **0.564** |
| per-stratum | DeltaNet 144.1 (30 layers), attention 145.0 (10 layers) |
| protected experts / layer (mask) | mean 78.7, min 65, **max 98** |

**Headroom read vs v3:** v3's E_eff/128 was ~0.69–0.77 → no headroom, optimal_k=full. v4's ratio **0.564 is
lower** — relatively MORE routing spread per expert than v3, so sub-full budgets are genuinely worth probing
rather than assumed dead. Countervailing signal: absolute E_eff is high (144/256) and the 5 flattest layers
(0,4,15,25,26) have intrinsically unstable top-8 rankings (Jaccard 0.778; the min-gate reads FALSE). Net:
the sweep must actually measure quality at sub-full k, not infer from the profile shape.

## 2. k grid (experts kept per layer) + derivation rule

**Grid: { full (256), 224, 192, 144, 112 }**

Derivation (reproducible from `eeff_report.json`, round-to-nearest-16):
- `full` = 256 — control arm.
- `224` = round16(max E_eff 224.5) — "keep everything the router actually uses."
- `144` = round16(mean E_eff 144.3) — the central effective count.
- `192` = round16(midpoint of mean and max) — brackets the mean→max band.
- `112` = round16(max-protected 98 + margin) — aggressive drop, still **≥ the 98 protected-per-layer
  floor** so protected-retention is satisfiable on every layer (a k below 98 could not retain all protected
  experts on the max-protected layer, an invalid arm).

Lowest admissible k is bounded below by max protected-per-layer (98); 112 is the aggressive-but-valid floor.

## 3. Gating metric

Single-seed **s1 judge Spearman rho** (primary checkpoint `...-judge-v4-s1-merged`) vs the same-stack vLLM
**full arm**, measured through the patched vLLM (`_sieve_vllm_patch` mask applied per k). Rationale: single-seed
s1 is the cheapest same-stack gate; if s1 parse-collapses at a k, the 3-seed ensemble (same central tendency)
will not rescue it (v3 showed total parse collapse at k=13/32). The **3-seed ensemble (s0/s1/s2)** is reserved
for CONFIRMATION only at a candidate winning k — not run on every arm.

## 4. TOST equivalence spec

- **epsilon = 0.02 (2 pp)**, CI-aware: the bootstrap **lower/upper bound** (not the point estimate) must
  clear the equivalence margin (carry-forward #4).
- **Reference = the same-stack vLLM `full` arm measured in Wave 2's first sweep arm** — NOT the llama.cpp Q8
  0.8067 nor the Tinker-capture 0.8358 (sanity_gate_recalibration lesson: both TOST sides must be the same
  serving stack).
- **Full-arm sanity floor:** register `full` rho ≥ ~0.72 (bf16-vLLM-served s1 ≈ 0.7872 anchor,
  `output/eval4/ext_q8_results.json` secondary_reads). If the full arm falls below the floor, the harness is
  misconfigured → **HALT the sweep** rather than read the gap as a masking result.

## 5. Dispositions (both valid recorded outcomes; both unblock Phase 26)

- **`optimal_k`** = the smallest k that (a) passes CI-aware TOST vs the full arm AND (b) retains every
  protected expert. → v4 judge compresses; Phase 26 merges + prunes at that k and re-checks vs v3's 30.2 GiB.
- **`no_winner`** (`optimal_k = full`) = no sub-full k is equivalent. → echoes v3; Phase 26 prune may still be
  attempted per ROADMAP, or the reopened compression question closes with v3 staying canonical.

The routing is recorded either way. No goalpost-moving after rho is read.
