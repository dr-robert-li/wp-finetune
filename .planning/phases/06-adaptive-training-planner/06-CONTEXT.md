# Phase 6: Adaptive Training Planner - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning
**Source:** PRD Express Path (~/Downloads/wp_finetune_adaptive_plan.md)

<domain>
## Phase Boundary

Replace the temperature-zone-based Step 8.5 in run-training with a power-primary decision engine (v4.0) that correctly exploits thermal headroom on the DGX Spark GB10. The engine uses GPU watts as the primary routing signal and treats temperature only as a safety brake. Also adds batch/grad_accum coupling, Unsloth banner parsing, extended warmup probes, and failure classification.

Output: New adaptive-planner skill, new config/adaptive_planning.yaml, updated train_model.py, updated run-training SKILL.md, updated observe-training SKILL.md.

</domain>

<decisions>
## Implementation Decisions

### Power-Primary Routing (ADPT-01)
- GPU watts is the primary routing signal, NOT temperature
- Power zones: THROTTLED (thermal brake >=82C), CAPPED (>=80W), TARGET (50-80W), MODERATE (target with headroom), UNDERUTILIZED (<50W)
- Temperature only overrides at >=82C (safety brake), never primary routing
- Fallback to GPU utilization as proxy when power_watts unavailable

### Thermal Exploitation Ladder (ADPT-02)
- v4.0 rung order: batch (Rung 1) > prefetch (Rung 2) > workers (Rung 3) > save_steps (Rung 4) > eval_steps (Rung 5)
- Batch increase is FIRST (highest thermal impact), not last resort
- I/O Bottleneck Guard: if GPU util <30% AND watts <30W, pause Rung 1 and prioritize Rungs 2-3

### Config-Driven Thresholds (ADPT-03)
- All thresholds in config/adaptive_planning.yaml (power_zones, thermal_brake, memory, probe, ladder, worker_budget)
- No hardcoded values in skill logic

### Batch/Grad_Accum Coupling (BTCH-01)
- Formula: new_accum = max(1, effective_batch // new_batch) on every batch change
- effective_batch must remain constant across batch changes

### Unsloth Banner Parsing (BTCH-02, BTCH-03)
- Parse Unsloth startup banner for actual batch/grad_accum values
- Write actuals to telemetry/training/_unsloth_actuals.json
- Planner uses actuals (not config) as basis when override detected
- Actuals file consumed and deleted after planning cycle

### Extended Watchdog (TELE-01, TELE-02)
- MemoryWatchdogCallback extended to call GPUSampler every 50 steps
- Canonical JSONL schema adds power_watts and mem_available_mb fields
- GPUSampler imported from dgx-toolbox telemetry.sampler

### Failure Classification (TELE-03)
- classify_failure() from dgx-toolbox telemetry.failure_classifier
- Returns: NORMAL, OOM, HANG, THERMAL
- OOM: GPU idle + RAM >95% in final readings
- HANG: GPU idle + CPU busy + MemAvailable OK (driver issue, NOT OOM)
- THERMAL: thermal_pause file exists

### Threshold Update (TELE-04)
- observe-training: 80C warning -> 82C, 83C critical -> 85C
- Aligns with GB10 empirical data (runs 80-82C with no throttling)

### Warmup Probe (PROB-01)
- 3-5 real training steps via dgx-toolbox probe.py (not 1-step MemAvailable)
- Triggered when batch increased without prior anchor
- Probe samples power + memory + temperature during steps

### Anchor Store (PROB-02)
- AnchorStore from dgx-toolbox telemetry.anchor_store
- Config-hashed persistence: same config -> same anchor
- Cooldown tracking: N runs before re-probing same batch size
- Hard caps: failed batch sizes recorded, never auto-retried

### Skill Invocation (PROB-03)
- run-training Step 8.5 replaced with adaptive-planner skill call
- Skill receives $THERMAL_LOG, $RATIO, $TELEMETRY as context variables

### Model Scale Awareness
- compute_effective_scale() from dgx-toolbox telemetry.effective_scale
- Qwen3-30B-A3B BF16 LoRA -> effective ~27.4B -> tier "13-30B" -> ceiling 8
- This corrects the current UGC-derived ceiling of 4

### Claude's Discretion
- Internal code structure of the adaptive-planner skill (function decomposition)
- Error message wording
- Log format details beyond what's specified in adjustment_log template
- Test file structure for any unit tests

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Training Infrastructure
- `scripts/train_model.py` — Current training script with MemoryWatchdogCallback (to be extended)
- `.claude/skills/wp-finetune:run-training/SKILL.md` — Current run-training skill (Step 8.5 to be replaced)
- `.claude/skills/wp-finetune:observe-training/SKILL.md` — Current observe-training skill (thresholds to update)
- `config/train_config.yaml` — Current training config (batch=4, grad_accum=4, workers=3, prefetch=3)
- `config/dgx_toolbox.yaml` — DGX Toolbox config (EXTRA_MOUNTS, PYTHONPATH to add)

### Detailed Execution Plan
- `~/Downloads/wp_finetune_adaptive_plan.md` — Full v4.0 plan (2,280 lines, 6 tasks with acceptance criteria)

### Telemetry Data (Existing)
- `telemetry/training/thermal_history.json` — Existing thermal history (2 runs)
- `telemetry/training/adaptive_adjustments.md` — Existing adjustment log

### dgx-toolbox Dependency (Phase 13)
- `~/dgx-toolbox/telemetry/sampler.py` — GPUSampler class
- `~/dgx-toolbox/telemetry/effective_scale.py` — compute_effective_scale function
- `~/dgx-toolbox/telemetry/anchor_store.py` — AnchorStore class
- `~/dgx-toolbox/telemetry/failure_classifier.py` — classify_failure function
- `~/dgx-toolbox/telemetry/probe.py` — Warmup probe (used via dgx.execute, not imported)

</canonical_refs>

<specifics>
## Specific Ideas

- The detailed plan at ~/Downloads/wp_finetune_adaptive_plan.md contains 6 concrete tasks with full code, acceptance criteria, and verification commands
- Task ordering: 1) adaptive-planner skill, 2) adaptive_planning.yaml, 3) train_model.py, 4) run-training skill, 5) observe-training skill, 6) dgx_toolbox.yaml
- Tasks 1-2 can run in parallel (new files, no conflicts)
- Tasks 3-6 modify existing files and should be sequential
- The plan includes a "must_haves" section with truths, artifacts, and key_links for verification

</specifics>

<deferred>
## Deferred Ideas

- Triton/TensorRT-LLM optimized inference (v2)
- DPO/RLHF refinement using adaptive planner signals (v2)
- Multi-GPU scaling of adaptive planner (single Spark only for now)

</deferred>

---

*Phase: 06-adaptive-training-planner*
*Context gathered: 2026-03-31 via PRD Express Path*
