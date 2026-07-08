# Phase 11 — Compression & Packaging (two-model pair) — CONTEXT

**Scaffolded:** 2026-07-08 · **Status:** ready for `/gsd-plan-phase`
**Scope note:** ROADMAP Phases 11-15 (Sieve → sieve-eval → merge+prune → final-eval → package).
This phase starts the chain; the 2026-07-03 and 2026-07-08 ROADMAP amendments govern.

## Scope decision (LOCKED 2026-07-08, user-selected): TRAINING-FREE SIEVE

Phase 11 is **training-free**: routing profile (v1.2 SFT policy + 3 judge seeds) → inference-time
expert-masking k-sweep (mask cold experts, measure wp-bench + judge rho at k≈13/32/64) → declare
optimal k + emit prune-set for Phase 13 AIMER. **No LoRA retraining, no recovery SFT** — the ROADMAP's
literal hot-expert retraining spec is superseded (it predates RL rejection + the frozen-weights ship
decision). SIEVE success criteria reinterpret "adapter checkpoint per k" as "expert-mask + eval record
per k"; optimal-k rule unchanged (smallest k within tolerance of full model, TOST epsilon=2pp, 3+ seeds
where seeds = judge ensemble members / eval bootstrap, not training seeds).

## Ship artifact (LOCKED — do not relitigate)

| Role | Model | Metric | Source |
|---|---|---|---|
| wp_judge | **v1.3 3-seed median ENSEMBLE** | rho **0.842** (val n=121) | seeds s0/s1/s2, `eval_seed_curve.json` |
| wp_gen | **v1.2 SFT merged** | codegen **0.4616** wp-bench | `models/qwen3-30b-wp-30_70-reasoning-merged-v4` |

- Judge ensemble = 3 LoRA adapters (rank-32, MoE-only) over the SAME base, median-aggregated
  `overall` per item. Seed checkpoints (ep3 samplers, manifests in `output/tinker/`):
  s0 = `wp-reasoning-relabel-v1-full-ep3` (default seed) · s1 = `wp-reasoning-relabel-s1-ep3`
  (canonical; LoRA archive `models/tinker_export/v1.3_pkg/checkpoint.tar.gz/`, MERGED 13-shard model at
  `models/_staging/qwen3-30b-wp-v1.3-merged/` — NOTE `models/tinker_export/v1.3/` is EMPTY, do not
  reference it) · s2 = `wp-reasoning-relabel-s2-ep3`.
  s0/s2 have Tinker sampler ckpts only — Phase 11 must EXPORT + merge them (via `scripts/merge_tinker_v3.py`
  path, Tinker MoE LoRA is NOT standard PEFT; vLLM cannot load it as a runtime adapter → sequential serving
  of merged checkpoints, not concurrent multi-LoRA). Disk headroom for 2 more ~57GB merges confirmed (1.6T free).
- **Fallback (pre-authorized):** single-seed s1 (rho 0.827, `PROMOTED_v1.3.json`) IF packaging
  measurement shows the ensemble cannot fit the GB10 memory wall or 3× judge latency breaks serving.
  Fallback exercise requires only a JOURNAL note, not a re-decision.
- Ensemble mechanics (REVISED per research): Tinker MoE-expert LoRA is NOT standard PEFT — vLLM cannot
  load it as a runtime adapter (`merge_tinker_v3.py` does manual per-expert tensor arithmetic). So the
  ensemble serves as **3 merged checkpoints, sequentially** (batch all val items through seed_i, swap,
  median at the end), NOT concurrent multi-LoRA. Judge latency = 3 sequential passes; fine for batch
  review workloads, the pre-authorized s1 fallback covers latency-sensitive serving.
- No further training on this base. RL closed (2026-07-05, 6/6 kills); SFT gap-closure closed
  (2026-07-08, all levers negative — `output/relabel/gap_closure_summary.json`).

## HARD CONSTRAINTS

1. **Protected expert mask is inviolable.** `output/profiling/reasoning-merged-v4/protected_expert_mask.npy`
   ([48,128] bool, 1,480 experts, immutable since Phase 7 sign-off 2026-06-19). MoE-Sieve selection AND
   AIMER/REAP pruning MUST exclude protected experts from removal. This is the mechanism that carries
   judge reasoning through the prune — reasoning is only LOST from here; the mask is the defense.
2. **Layer stability notes now in the mask JSON** (`layer_stability_notes` key, added 2026-07-08 per
   Phase 7 forward obligation): low-Jaccard band {9,13,14,31,35,36} + late-layer {45,46,47}. Phase 13
   must pre-commit median-threshold (2,477-expert) headroom on these layers before pruning them.
3. **Routing profiles: profile the v1.2 SFT policy** (per 2026-07-03 amendment) for the gen model.
   Phase 7 profiles remain the protected-expert REFERENCE (do not regenerate the mask).
4. **Post-compression gates (regression bars):** judge ensemble rho ≥ 0.842 − noise floor
   (seed sd 0.020; use `gate_noise_floors.json` conventions); judge single-seed ≥ 0.827 − floor;
   gen wp-bench ≥ 0.4616 − pre-registered tolerance (set at plan time, CI-aware per D-V4-10 hardening).
5. **GB10 memory wall** (`MEMORY-INVESTIGATION-bf16.md`): two 30B-A3B model instances do not co-reside
   in bf16. Packaging must sequence or quantize; measure before promising co-serving.

## Open questions for planning (NOT pre-decided)

- Serving topology: one base + {3 judge LoRAs, v1.2 gen deltas} multi-LoRA? Or two merged instances
  swapped/quantized? (v1.2 gen is already merged; judge seeds are unmerged LoRA.)
- Do the 3 judge seeds route similarly enough that one Sieve profile covers all 3, or does Sieve need
  the union of 3 routing profiles? (Cheap check: E_eff overlap across seeds on the val stimulus.)
- Quantization decision cascade (Phase 15 gates) — bf16 baseline first, then int8/int4 A/B per ROADMAP.
- Whether Phase 12's A/B keeps all 5 k-sweep Sieve points or prunes the sweep given two models.

## Key inputs

- `output/relabel/gap_closure_summary.json` — why no more training (3 levers negative).
- `output/relabel/eval_seed_curve.json` — ensemble N-curve (N=3 = knee; 0.842).
- `output/tinker/PROMOTED_v1.3.json` — canonical s1 record + export path.
- `output/profiling/reasoning-merged-v4/protected_expert_mask.{npy,json}` — the mask + stability notes.
- `wp-moe.md` — MoE-Sieve method reference.
- ROADMAP Phases 11-15 success criteria (lines ~570-670) with the two amendments applied.
