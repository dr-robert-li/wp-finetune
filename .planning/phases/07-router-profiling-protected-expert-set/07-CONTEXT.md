# Phase 7: Router Profiling & Protected Expert Set - Context

**Gathered:** 2026-06-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Gradient-free routing profiling of the **single promoted v1.2 model**
(`models/qwen3-30b-wp-30_70-reasoning-merged-v4`) to produce per-task (`<wp_gen>` vs `<wp_judge>`)
expert-affinity maps with E_eff/concentration metrics, and to extract the **protected expert set**
(D-10) — the per-layer mask of dual-purpose experts that MUST be retained through all later
MoE-Sieve / pruning phases. Output: routing report + E_eff comparison vs the Phase-4 base-model
baseline + an exported protected-expert per-layer mask for downstream consumption.

**Out of scope (reduced from ROADMAP §7):** multi-ratio profiling and the ratio-selection decision
matrix (original SC5). Phase-4 triage already returned **NO_SURVIVORS except 30/70** (PROJECT.md), so
there is exactly one ratio — already merged and promoted in v1.2. No ratio is being selected here.
</domain>

<decisions>
## Implementation Decisions

### Phase scope (ratio-selection moot)
- **D-01:** Phase 7 profiles ONLY the promoted `reasoning-merged-v4` model and extracts the
  protected-expert mask. The multi-ratio decision matrix (ROADMAP §7 SC5) is **DROPPED as moot** —
  Phase-4 triage gave NO_SURVIVORS except 30/70, which v1.2 already merged/promoted.
- **D-02:** The E_eff / concentration report (per-layer CV, cumulative coverage, layer-depth skew,
  E_eff mean/max/variance) is **kept as informational** input to protected-expert identification and
  as a pruning baseline (Phases 11/13) — NOT for ratio selection. SC1-SC4 + SC6 of ROADMAP §7 stand;
  SC5 is removed.

### Protected-expert set definition (D-10)
- **D-03:** **Conservative co-activation.** An expert is flagged dual-purpose (protected,
  must-not-prune) when it shows meaningful activation **above its per-layer mean for BOTH `<wp_gen>`
  AND `<wp_judge>`**. Errs toward over-protecting — the judge skill is the fragile axis and wrongly
  pruning a dual-purpose expert breaks the dual-mode model; over-protection only costs pruning
  headroom, which is recoverable later.
- **D-04:** Report **mask-size sensitivity across thresholds** (e.g., mean / median / top-K
  intersection) alongside the chosen conservative mask, so Phase 13 (AIMER/REAP pruning) can revisit
  the protection/headroom trade-off with data rather than re-deciding blind.

### Profiling stimulus + methodology
- **D-05:** Drive forward-pass routing capture with the **existing 4.4 captures** — the `<wp_gen>`
  generation tasks + `<wp_judge>` val prompts already used in the eval set — balanced gen/judge so
  per-task expert affinity is clean and consistent with how the model is evaluated.
- **D-06:** Use the **10% subsample with Jaccard ≥ 0.94** vs full-set ranking per ratio
  (ROADMAP §7 SC3); re-profile with a larger subsample if Jaccard fails.

### Profile target + baseline
- **D-07:** Profile the **merged `reasoning-merged-v4`** model. The MoE router was frozen during v1.2
  LoRA, so any routing shift comes from the weights feeding the gate — profiling the merged model
  captures the net effect. Hook `Qwen3MoeSparseMoeBlock` gating outputs.
- **D-08:** Compare E_eff against the existing **Phase-4 `base_model_eeff.jsonl`** to quantify the
  fine-tuning routing shift (ROADMAP §7 SC4).

### Gate hygiene (folded from D-V4-10)
- **D-09:** Any Phase-7 selection/quality gate (Jaccard threshold, E_eff comparison, protected-expert
  cutoff) reports **bootstrap CIs and uses CI-aware dispositions**, not bare point-bars — the codified
  lesson from Phase 04.4 (a point-bar inside its own noise band cries wolf). Measure identically on
  baseline and candidate.

### Claude's Discretion (→ researcher/planner)
- Exact E_eff formula + concentration-metric implementation details (reuse `profile_base_model.py`).
- Subsample construction + Jaccard computation mechanics.
- Protected-mask export format (per-layer boolean mask file) for downstream consumption.
- Telemetry agent embedding (`observe-evaluation`) during profiling runs.

### Folded Todos
- **CI-aware noise-band gates (D-V4-10, `phase7-8-ci-aware-noiseband-gates.md`):** folded into D-09 —
  Phase-7 gates use CI-aware dispositions. Originates from the Phase 04.4 finding that point-bars at
  small-n cross their own noise band.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope + requirements
- `.planning/ROADMAP.md` §"Phase 7: Router Profiling & Protected Expert Set" — goal, success criteria
  (note SC5 dropped per D-01), requirements PROF-01..05 + GATE-01.
- `.planning/REQUIREMENTS.md` — PROF-01..05, GATE-01 traceability.
- `.planning/PROJECT.md` — line 93 (Phase-4 triage NO_SURVIVORS except 30/70, the basis for D-01);
  line 117 (router-profiling task-token affinity); D-10 protected-expert decision.

### Inputs (the model + baseline being profiled)
- `models/qwen3-30b-wp-30_70-reasoning-merged-v4/` — the promoted v1.2 model under profiling (D-07).
- `output/profiling/base_model_eeff.jsonl` — Phase-4 base-model E_eff baseline for the shift
  comparison (D-08).
- Phase 4.4 eval captures (`output/eval_reasoning_v4_winner/` — wp_gen + wp_judge prompt sets) — the
  profiling stimulus (D-05).

### Downstream consumers (why the mask matters)
- `.planning/ROADMAP.md` §"Phase 11/13" — MoE-Sieve + AIMER/REAP pruning consume the protected mask.
- `.planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-D-V4-10-WAIVER.md` — origin of the
  CI-aware gate hygiene folded as D-09.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/profile_base_model.py` (25.3K): the Phase-4 base-model profiler that hooks
  `Qwen3MoeSparseMoeBlock` gating outputs and computes E_eff. **Primary reuse target** — adapt to
  profile the merged adapter (D-07) and emit per-task (gen/judge) routing tables.
- `output/profiling/base_model_eeff.jsonl`: existing base-model E_eff per layer — the comparison
  baseline (D-08).
- `scripts/triage_ratios.py`, `scripts/run_eval_triage.py`: Phase-4 triage (complete) — reference for
  per-ratio handling, but no longer driving selection (single ratio).

### Established Patterns
- Skill pattern `wp-finetune:run-evaluation` / `observe-evaluation` — Phase 7 extends this for
  GPU-bound profiling (`run-profiling` skill, created at planning time).
- DGX/local forward-pass execution; gradient-free (no training, no LLM API).

### Integration Points
- Profiling consumes the promoted canonical model dir (Phase 4.4 output) and the 4.4 eval prompt sets.
- Exports the protected-expert per-layer mask consumed by Phases 11 (MoE-Sieve) and 13 (pruning).
</code_context>

<specifics>
## Specific Ideas

- The protected mask should ship with a **sensitivity table** (mask size vs threshold), not just the
  single conservative mask, so the protection/prunability trade-off is revisitable in Phase 13 (D-04).
- E_eff is presented as a **routing-shift delta vs base** (D-08), not an absolute, to make the
  fine-tuning effect legible.
</specifics>

<deferred>
## Deferred Ideas

### Reviewed Todos (not folded)
- **Phase 8 judge-recalibration inheritance (`phase8-inherit-judge-recalibration.md`,
  resolves_phase 8):** matched Phase 7 at score 0.6 but belongs to Phase 8 (the reward pipeline
  consumes `judge_recalibration.json`). Left for Phase 8 — not folded here.

</deferred>

---

*Phase: 7-Router Profiling & Protected Expert Set*
*Context gathered: 2026-06-14*
